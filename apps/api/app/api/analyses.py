import hashlib
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased

from app.db.models import (
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_QUEUED,
    Analysis,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import get_session
from app.detection import (
    C2PA_PROVIDER,
    NVIDIA_PROVIDER,
    PROVENANCE_SIGNAL,
    SYNTHETIC_VIDEO_SIGNAL,
)
from app.media import MediaMetadata, MediaProbeError, MediaProbeUnavailable, probe_media
from app.normalization import needs_normalization
from app.storage import store_original

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])

ALLOWED_CONTENT_TYPES = frozenset({"video/mp4", "video/quicktime"})
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
TEMP_FILE_PREFIX = "deepguard-upload-"

# How many analyses the listing returns. The dashboard shows recent activity, so a fixed
# ceiling is all it needs; paging belongs to a phase that has a reason for it.
RECENT_ANALYSES_LIMIT = 50

# How many clips the listing hands to the dashboard per analysis. The list view shows
# the strongest few as evidence that the detector looked inside the video; a full timeline
# needs a detail page, which no phase has built yet.
DASHBOARD_SEGMENTS = 5


def discard_temp_file(path: str | Path) -> None:
    """Remove a staged temp file best-effort, never masking the error being handled."""
    try:
        os.unlink(path)
    except OSError:
        pass


def report_possible_orphan(*storage_keys: str) -> None:
    """Record objects a failed request may have left behind, without deleting them.

    Storage keys are content-addressed, so an object this request uploaded can be the
    very same object an earlier, already persisted analysis of identical bytes refers
    to. Nothing available here proves the current request created an object exclusively,
    and proving it would take the reference counting or ownership tracking that P1
    deliberately does not have. Destroying a forensic original another analysis still
    references is far worse than leaving an object behind, so the object is preserved
    and the condition is logged for operational follow-up.
    """
    # An original that needed no derivative is reported under one key, not twice.
    logger.warning(
        "Failed request may have left unreferenced MinIO objects behind: %s.",
        ", ".join(dict.fromkeys(storage_keys)),
    )


class CreatedAnalysis(BaseModel):
    """The staged analysis, as the upload left it.

    Everything here is established before any detector runs, so nothing in it is a
    finding: it is the media, its identity and where it was put. `status` is `queued`.
    """

    id: uuid.UUID
    status: str
    filename: str | None
    content_type: str
    size_bytes: int
    sha256: str
    storage_key: str
    metadata: MediaMetadata
    # Whether a derivative is needed at all — decided here, from the probe, but produced
    # by the worker. True on this response means one is owed, not that one exists.
    was_normalized: bool
    # The object downstream inference should read, when that is already settled. Null
    # whenever a derivative is owed: the transcode has not run, so there is no such object
    # yet and naming one would be a guess. When the original is already canonical this is
    # the original's key — no second artifact exists, and copying it would only duplicate
    # storage.
    derivative_storage_key: str | None = None
    # Present only when a real derivative exists, since it is that artifact's identity.
    derivative_sha256: str | None = None


class SegmentEvidence(BaseModel):
    """One clip the provider scored, as the provider scored it.

    `clip_index` is NVIDIA's frame index for the clip's middle frame, and `logit` its raw
    model output. There is no time range and no probability here, because NVIDIA reports
    neither for a clip; both would have to be invented to appear.
    """

    clip_index: int
    logit: float


class SyntheticVideoSignal(BaseModel):
    """The persisted NVIDIA synthetic-video signal, as the dashboard may see it.

    The provider's own identity, state and figure, nothing derived: no risk level, no
    verdict, and no rescaling of `score`, which stays NVIDIA's number on NVIDIA's scale.
    `score` is null for every non-SUCCESS status, because a detector that did not answer
    has no number and 0.0 would be a fabricated one.

    The stored `metadata` document is not passed through as it stands. On a failure it
    holds the provider exception's class name, which is internal diagnostic detail, so
    only the two aggregate figures a successful detection produces are named here.
    """

    provider: str
    signal_type: str
    status: str
    score: float | None
    provider_version: str | None
    # NVIDIA's aggregate logit and the number of clips it scored. Both absent unless the
    # detection succeeded and the provider reported them.
    logit: float | None
    total_clips: int | None
    # The strongest few clips behind the aggregate, highest logit first. Empty for a
    # detection that produced none — a failure, or a provider that reported no clips.
    # Compare against `total_clips` to see how much of the evidence this is.
    segments: list[SegmentEvidence]


class ProvenanceSignal(BaseModel):
    """The persisted C2PA provenance signal, as the dashboard may see it.

    Facts about what the file itself claims, and nothing derived from them. There is no
    score: provenance is the state of a signature, not a figure on a scale, and one here
    would sit beside NVIDIA's probability as though the two were comparable.

    The states a reader has to tell apart are carried, never merged:
    `status` is `SUCCESS` when the file was read — including when it holds no credentials,
    which `manifest_exists` reports — and `FAILED` when reading it did not work, which is
    not the same as media that carries nothing. `validation_state` is the C2PA SDK's own
    word for what it made of the signature, passed through untranslated and null whenever
    no manifest exists.

    Absent or invalid credentials are not evidence of manipulation, and nothing here says
    they are. Most media carries none at all.
    """

    provider: str
    signal_type: str
    status: str
    provider_version: str | None
    # Null on a failed reading: whether credentials exist is exactly what is unknown then.
    manifest_exists: bool | None
    validation_state: str | None
    # Who the manifest says made the media, and who signed for it. Secondary context, and
    # absent whenever the manifest does not name them.
    claim_generator: str | None
    signature_issuer: str | None
    # Where the file says its manifest lives, when it keeps it somewhere other than inside
    # itself. Present alongside `manifest_exists = false`, and the pair is a fourth fact in
    # its own right: provenance was claimed, it is simply not in these bytes. The URL was
    # recorded and deliberately never visited — fetching it would let an uploaded file
    # steer a request out of the worker — so nothing here says whether it resolves.
    remote_manifest_url: str | None


class MediaFacts(BaseModel):
    """What ffprobe established about the forensic original, as it was persisted.

    Kept as a nested object rather than flattened beside `declared_content_type`, because
    that is exactly the distinction worth preserving: everything here was read out of the
    bytes, while the declared MIME is only what the client claimed about them.

    Deliberately not `MediaMetadata`, the dataclass the upload response carries. That one
    is what ffprobe returned; this one is what the database kept, and the two differ —
    `major_brand` has no column, so the container evidence available here is
    `format_name`, ffprobe's name for the demuxer family. It is reported as ffprobe words
    it: `mov,mp4,m4a,3gp,3g2,mj2` covers MOV and MP4 alike, and narrowing that to "MP4"
    would be this layer claiming something the stored evidence does not establish.
    """

    format_name: str
    codec_name: str
    width: int
    height: int
    duration: float
    frame_rate: float
    # Null whenever ffprobe reported no pixel format for the stream.
    pix_fmt: str | None
    constant_frame_rate: bool


class AnalysisSummary(BaseModel):
    """One row of the dashboard listing.

    Deliberately narrower than `CreatedAnalysis`: only what the list view renders. The
    storage keys and derivative identity are not shown there, and risk and report fields
    do not exist yet.
    """

    id: uuid.UUID
    status: str
    created_at: datetime
    original_filename: str | None
    # Named for what it is: the MIME the client declared. ffprobe proves the bytes are
    # video, but it never confirms this string, so the listing must not imply it did.
    declared_content_type: str
    size_bytes: int
    original_sha256: str
    was_normalized: bool
    # Never null: an analysis and its media row are written in one transaction, and the
    # listing joins them inner, so a listed analysis always has these facts.
    media: MediaFacts
    # Null for an analysis that carries no such signal — anything stored before the
    # detector was wired in, and any later analysis whose signal row is absent.
    synthetic_video: SyntheticVideoSignal | None
    # Likewise null for an analysis processed before provenance was read at all. Not the
    # same as an analysis whose media carries no credentials: that is a signal that ran.
    provenance: ProvenanceSignal | None


@dataclass(frozen=True)
class StoredUpload:
    """Internal result of the single read pass.

    The temp path is deliberately kept out of the API response: it is server-internal
    filesystem layout, not a client-facing contract.
    """

    path: Path
    size_bytes: int
    sha256: str


async def store_upload(file: UploadFile) -> StoredUpload:
    """Consume the upload once in bounded chunks, sizing, hashing and spilling to disk.

    The upload stream is read exactly once: every chunk is counted, checked against the
    limit, fed to SHA-256 and written to the temp file in the same pass. On any failure
    the partial temp file is removed without masking the original error.
    """
    hasher = hashlib.sha256()
    size = 0

    temp_file = tempfile.NamedTemporaryFile(prefix=TEMP_FILE_PREFIX, delete=False)
    try:
        with temp_file:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit.",
                    )

                hasher.update(chunk)
                temp_file.write(chunk)
    except BaseException:
        discard_temp_file(temp_file.name)
        raise

    return StoredUpload(
        path=Path(temp_file.name),
        size_bytes=size,
        sha256=hasher.hexdigest(),
    )


def persist_analysis(
    session: Session,
    *,
    filename: str | None,
    content_type: str,
    stored: StoredUpload,
    storage_key: str,
    metadata: MediaMetadata,
    was_normalized: bool,
    derivative_storage_key: str | None,
) -> Analysis:
    """Write the queued analysis, its media and the job it is owed in one transaction.

    Called once the original is stored and probed, which is everything the worker needs to
    start: it can fetch those bytes, read their provenance, transcode them if they need it
    and detect against the result. The job is written here rather than appended afterwards
    because either every row exists or none does: an analysis committed without its job
    would be an upload accepted and then silently forgotten, with no queue entry to notice
    it and a client holding a `202` for work nobody will ever do.

    `derivative_sha256` is never written here. A derivative has its own content identity
    and this request has not produced one; the worker writes both derivative columns
    together, once the artifact they describe actually exists.

    Neither ffmpeg nor a detector runs before this, so the transaction lasts as long as
    three inserts and never spans the minutes either of them can take.

    On failure the session is rolled back and the stored objects are reported rather than
    deleted, since they are content-addressed and may be shared.
    """
    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED)
    session.add(analysis)
    # Assigns the analysis id the media row needs, still inside the same transaction.
    session.flush()

    session.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename=filename,
            content_type=content_type,
            size_bytes=stored.size_bytes,
            original_sha256=stored.sha256,
            original_storage_key=storage_key,
            format_name=metadata.format_name,
            codec_name=metadata.codec_name,
            width=metadata.width,
            height=metadata.height,
            duration=metadata.duration,
            frame_rate=metadata.frame_rate,
            pix_fmt=metadata.pix_fmt,
            constant_frame_rate=metadata.constant_frame_rate,
            was_normalized=was_normalized,
            derivative_storage_key=derivative_storage_key,
        )
    )

    session.add(AnalysisJob(analysis_id=analysis.id, status=JOB_STATUS_QUEUED))

    session.commit()

    return analysis


@router.post(
    "/analyses", response_model=CreatedAnalysis, status_code=status.HTTP_202_ACCEPTED
)
async def create_analysis(
    file: UploadFile, session: Session = Depends(get_session)
) -> CreatedAnalysis:
    """Admit an upload, prove it is real media, stage it, and queue it for detection.

    The declared content type is not proof that the bytes are a real MP4/MOV container,
    so admission alone never produces a `202`.

    The admitted upload is stored in MinIO as the forensic original, then probed with
    ffprobe to confirm the bytes really are video and to extract the metadata later steps
    need. The original is never rewritten (D013).

    Neither ffmpeg nor a detector is called here. Both are unbounded in a way the work
    above is not: reading, hashing and probing scale with the upload, which is already
    capped, while a transcode of a 4K source and an NVIDIA call both take as long as they
    take. Normalization used to run here and was bounded by a deadline that really asked
    "how long may a client wait" — a 4K HEVC upload passed validation, ran out of that
    deadline mid-transcode and was rejected as unprocessable media, having never been
    analysed at all. So the request now ends where the bounded work does: it commits the
    analysis and a `queued` job and answers `202 Accepted`, and the worker transcodes and
    detects on its own schedule (D020).

    `was_normalized` is decided here, because the decision needs `major_brand` and no
    column holds it. It says a derivative is *owed*, not that one exists — the response
    carries a null `derivative_storage_key` in that case, and the worker fills it in.

    The staged file is deleted before responding. Nothing local survives the request, and
    the worker reads the original back out of MinIO rather than depending on a temp file
    this process happened to leave behind.

    On failure the temp file is dropped; the stored object is kept, because its
    content-addressed key may be shared with an earlier analysis.
    """
    content_type = (file.content_type or "").strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type or 'unknown'}.",
        )

    try:
        stored = await store_upload(file)
    finally:
        await file.close()

    try:
        storage_key = store_original(stored.path, stored.sha256, content_type)
    except Exception:
        discard_temp_file(stored.path)
        # Endpoints, credentials and SDK errors stay in the server log, not the response.
        logger.exception("Storing the original upload in MinIO failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media storage unavailable",
        ) from None

    try:
        metadata = await probe_media(stored.path)
    except MediaProbeUnavailable:
        discard_temp_file(stored.path)
        report_possible_orphan(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffprobe is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except MediaProbeError:
        discard_temp_file(stored.path)
        report_possible_orphan(storage_key)
        logger.info("Rejected admitted upload %s as unusable media.", storage_key, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="invalid or unsupported video media",
        ) from None

    # Decided from the probed container evidence alone; `content_type` is only what the
    # client claimed, and a MOV declared as `video/mp4` must still be normalized.
    was_normalized = needs_normalization(metadata)
    # Null while a derivative is owed. Already-canonical media needs no second artifact,
    # so the original's own key is the answer and is known now.
    canonical_key = None if was_normalized else storage_key

    # Probing is the last step that reads local media here. The original is in MinIO, and
    # the worker fetches it from there.
    discard_temp_file(stored.path)

    try:
        analysis = persist_analysis(
            session,
            filename=file.filename,
            content_type=content_type,
            stored=stored,
            storage_key=storage_key,
            metadata=metadata,
            was_normalized=was_normalized,
            derivative_storage_key=canonical_key,
        )
    except SQLAlchemyError:
        try:
            session.rollback()
        except SQLAlchemyError:
            # A rollback that fails is worth knowing about, but the persistence failure
            # is the error being handled and must not be replaced by it.
            logger.exception("Rolling the analysis transaction back failed.")
        # Statements, connection strings and driver errors stay in the server log.
        logger.exception("Persisting the analysis failed.")
        report_possible_orphan(storage_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="analysis could not be persisted",
        ) from None

    return CreatedAnalysis(
        id=analysis.id,
        status=analysis.status,
        filename=file.filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=storage_key,
        metadata=metadata,
        was_normalized=was_normalized,
        derivative_storage_key=canonical_key,
    )


def signal_figure(metadata: object, key: str, expected: type | tuple[type, ...]) -> Any | None:
    """Read one aggregate figure out of a stored signal's metadata document.

    The document is provider-derived JSON that was written by an earlier release and read
    back here, so nothing guarantees its shape. A missing or wrongly typed value becomes
    null rather than an error: the figure is supporting detail, and one odd row must not
    take the whole listing down. Booleans are rejected for an int, since `True` is one in
    Python and "1 clip" would be a lie.
    """
    if not isinstance(metadata, dict):
        return None

    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, expected):
        return None

    return value


def signal_flag(metadata: object, key: str) -> bool | None:
    """Read one boolean fact out of a stored signal's metadata document.

    Separate from `signal_figure` because that one rejects booleans on purpose. Anything
    other than a real boolean reads as unknown rather than being coerced: `manifest_exists`
    is the field a reader leans on to tell "no credentials" from "could not tell", and a
    truthy string quietly becoming `true` would break exactly that distinction.
    """
    if not isinstance(metadata, dict):
        return None

    value = metadata.get(key)

    return value if isinstance(value, bool) else None


def provenance_signal(row: Any) -> ProvenanceSignal | None:
    """Turn the joined provenance columns into the response's signal, or nothing.

    As with the detection signal, the outer join leaves every column null when an analysis
    carries no provenance row, and `status` is the one that cannot be null on a real one.

    The stored metadata document is read defensively and never passed through: on a failed
    reading it holds the exception class name, which is internal diagnostic detail.
    """
    if row.provenance_status is None:
        return None

    return ProvenanceSignal(
        provider=row.provenance_provider,
        signal_type=row.provenance_signal_type,
        status=row.provenance_status,
        provider_version=row.provenance_provider_version,
        manifest_exists=signal_flag(row.provenance_metadata, "manifest_exists"),
        validation_state=signal_figure(row.provenance_metadata, "validation_state", str),
        claim_generator=signal_figure(row.provenance_metadata, "claim_generator", str),
        signature_issuer=signal_figure(row.provenance_metadata, "signature_issuer", str),
        remote_manifest_url=signal_figure(
            row.provenance_metadata, "remote_manifest_url", str
        ),
    )


def synthetic_video_signal(
    row: Any, segments: list[SegmentEvidence]
) -> SyntheticVideoSignal | None:
    """Turn the joined signal columns into the response's signal, or nothing.

    The outer join leaves every signal column null when an analysis carries no NVIDIA
    signal; `status` is the one that cannot be null on a real row, so it decides.
    """
    if row.signal_status is None:
        return None

    return SyntheticVideoSignal(
        provider=row.signal_provider,
        signal_type=row.signal_type,
        status=row.signal_status,
        score=row.signal_score,
        provider_version=row.signal_provider_version,
        # A logit that happens to be whole round-trips through JSON as an int.
        logit=signal_figure(row.signal_metadata, "logit", (int, float)),
        total_clips=signal_figure(row.signal_metadata, "total_clips", int),
        segments=segments,
    )


def strongest_segments(
    session: Session, signal_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[SegmentEvidence]]:
    """Fetch the top clips for every listed signal at once, grouped by signal.

    One statement covers the whole page, however many analyses it holds: the listing must
    not issue a query per analysis, and a correlated per-signal limit would do exactly
    that. Persisted evidence is already capped per signal (`MAX_PERSISTED_SEGMENTS`), so
    the bounded set is read back and trimmed to the display count here rather than in SQL.

    Signals with no stored evidence are simply absent from the result.
    """
    if not signal_ids:
        return {}

    rows = session.execute(
        select(
            AnalysisSegment.signal_id,
            AnalysisSegment.clip_index,
            AnalysisSegment.logit,
        )
        .where(AnalysisSegment.signal_id.in_(signal_ids))
        # The same ordering the evidence was selected by, so the strongest clip leads and
        # the sequence never shifts between calls.
        .order_by(
            AnalysisSegment.signal_id,
            AnalysisSegment.logit.desc(),
            AnalysisSegment.clip_index,
        )
    ).all()

    grouped: dict[uuid.UUID, list[SegmentEvidence]] = {}
    for row in rows:
        found = grouped.setdefault(row.signal_id, [])
        if len(found) < DASHBOARD_SEGMENTS:
            found.append(SegmentEvidence(clip_index=row.clip_index, logit=row.logit))

    return grouped


@router.get("/analyses", response_model=list[AnalysisSummary])
def list_analyses(session: Session = Depends(get_session)) -> list[AnalysisSummary]:
    """Return the most recent analyses, with both of their signals, for the dashboard.

    An inner join onto the media is correct here rather than restrictive: an analysis and
    its media row are written in one transaction, so an analysis without media cannot
    exist. The signals are outer joins, because either can genuinely be missing — analyses
    stored before a source was wired in have none — and each join names one provider and
    one signal type, so no row is multiplied by the evidence hanging off it.

    Two statements serve the whole page — the analyses with their signals, then the clip
    evidence for every detection signal on it. Both are fixed in number: neither grows
    with how many analyses are listed, so there is no query per analysis.

    Ordering falls back to the id because `created_at` defaults to the transaction
    timestamp, which two analyses committed together can share — without the tiebreak
    their relative order would be arbitrary between calls.
    """
    # The same table, joined a second time for the other signal. Each join is narrowed to
    # one provider and one signal type, so neither can multiply the listing's rows, and
    # both signals still arrive on the one statement.
    provenance = aliased(AnalysisSignal)

    try:
        rows = session.execute(
            select(
                Analysis.id,
                Analysis.status,
                Analysis.created_at,
                MediaFile.original_filename,
                MediaFile.content_type,
                MediaFile.size_bytes,
                MediaFile.original_sha256,
                MediaFile.was_normalized,
                # The probed facts about the original. Already on the joined row, so
                # naming them costs no extra statement.
                MediaFile.format_name,
                MediaFile.codec_name,
                MediaFile.width,
                MediaFile.height,
                MediaFile.duration,
                MediaFile.frame_rate,
                MediaFile.pix_fmt,
                MediaFile.constant_frame_rate,
                # Labelled, because `status` and `created_at` exist on both tables and the
                # unlabelled columns would collide in the result row.
                AnalysisSignal.id.label("signal_id"),
                AnalysisSignal.provider.label("signal_provider"),
                AnalysisSignal.signal_type.label("signal_type"),
                AnalysisSignal.status.label("signal_status"),
                AnalysisSignal.score.label("signal_score"),
                AnalysisSignal.provider_version.label("signal_provider_version"),
                AnalysisSignal.signal_metadata.label("signal_metadata"),
                provenance.provider.label("provenance_provider"),
                provenance.signal_type.label("provenance_signal_type"),
                provenance.status.label("provenance_status"),
                provenance.provider_version.label("provenance_provider_version"),
                provenance.signal_metadata.label("provenance_metadata"),
            )
            .join(MediaFile, MediaFile.analysis_id == Analysis.id)
            .outerjoin(
                AnalysisSignal,
                and_(
                    AnalysisSignal.analysis_id == Analysis.id,
                    AnalysisSignal.provider == NVIDIA_PROVIDER,
                    AnalysisSignal.signal_type == SYNTHETIC_VIDEO_SIGNAL,
                ),
            )
            .outerjoin(
                provenance,
                and_(
                    provenance.analysis_id == Analysis.id,
                    provenance.provider == C2PA_PROVIDER,
                    provenance.signal_type == PROVENANCE_SIGNAL,
                ),
            )
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(RECENT_ANALYSES_LIMIT)
        ).all()

        segments = strongest_segments(
            session, [row.signal_id for row in rows if row.signal_id is not None]
        )
    except SQLAlchemyError:
        # Statements, connection strings and driver errors stay in the server log.
        logger.exception("Reading the analysis listing failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analyses are temporarily unavailable",
        ) from None

    return [
        AnalysisSummary(
            id=row.id,
            status=row.status,
            created_at=row.created_at,
            original_filename=row.original_filename,
            declared_content_type=row.content_type,
            size_bytes=row.size_bytes,
            original_sha256=row.original_sha256,
            was_normalized=row.was_normalized,
            media=MediaFacts(
                format_name=row.format_name,
                codec_name=row.codec_name,
                width=row.width,
                height=row.height,
                duration=row.duration,
                frame_rate=row.frame_rate,
                pix_fmt=row.pix_fmt,
                constant_frame_rate=row.constant_frame_rate,
            ),
            synthetic_video=synthetic_video_signal(row, segments.get(row.signal_id, [])),
            provenance=provenance_signal(row),
        )
        for row in rows
    ]
