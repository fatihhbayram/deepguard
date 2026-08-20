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
from sqlalchemy.orm import Session

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
    Analysis,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import get_session
from app.media import MediaMetadata, MediaProbeError, MediaProbeUnavailable, probe_media
from app.normalization import (
    NormalizationError,
    NormalizationUnavailable,
    needs_normalization,
    normalize_to_mp4,
)
from app.nvidia_video import (
    NvidiaClipResult,
    NvidiaProviderError,
    NvidiaProviderTimeout,
    analyze_video,
)
from app.storage import derivative_key, store_derivative, store_original

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])

ALLOWED_CONTENT_TYPES = frozenset({"video/mp4", "video/quicktime"})
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
TEMP_FILE_PREFIX = "deepguard-upload-"

# How many analyses the listing returns. The dashboard shows recent activity, so a fixed
# ceiling is all it needs; paging belongs to a phase that has a reason for it.
RECENT_ANALYSES_LIMIT = 50

# Identity of the one detector wired in so far. NVIDIA's D009 answers exactly one
# question — whether the video is synthetic — so the pair is a constant, not a lookup.
NVIDIA_PROVIDER = "nvidia"
SYNTHETIC_VIDEO_SIGNAL = "synthetic_video"

# How many clip results one detection may leave behind. NVIDIA scores a clip every few
# frames, so a long video produces thousands and persisting all of them would let one
# provider's output grow the database without limit. The strongest evidence is kept and
# the rest is dropped; `total_clips` on the signal records how many there were, so a
# truncated set is never mistaken for the whole one (D019).
MAX_PERSISTED_SEGMENTS = 20

# How many of those the listing hands to the dashboard per analysis. The list view shows
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
    """The persisted analysis, as the pipeline established it."""

    id: uuid.UUID
    status: str
    filename: str | None
    content_type: str
    size_bytes: int
    sha256: str
    storage_key: str
    metadata: MediaMetadata
    was_normalized: bool
    # The object downstream inference should read. When the original is already
    # canonical this is the original's key: no second artifact exists, and inventing a
    # copy of it would only duplicate storage.
    derivative_storage_key: str
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


class AnalysisSummary(BaseModel):
    """One row of the dashboard listing.

    Deliberately narrower than `CreatedAnalysis`: only what the list view renders. The
    storage keys, ffprobe geometry and derivative identity are not shown there, and
    risk and report fields do not exist yet.
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
    # Null for an analysis that carries no such signal — anything stored before the
    # detector was wired in, and any later analysis whose signal row is absent.
    synthetic_video: SyntheticVideoSignal | None


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


async def create_derivative(
    original_path: Path, storage_key: str, metadata: MediaMetadata
) -> tuple[str, str, Path]:
    """Transcode the original into a stored canonical derivative and describe it.

    Returns the derivative's storage key, its own SHA-256, and the local file it still
    occupies. The derivative is deliberately left on disk: it is the artifact detector
    inference has to read, and downloading back from MinIO what is already here would be
    pointless. Deleting it is the caller's job, on every path.

    Any failure cleans up the local temp files and reports the request's stored objects
    as possible orphans; they are content-addressed and therefore never deleted here. The
    original file on disk is only read; the original MinIO object is preserved either
    way (D013).
    """
    try:
        derivative = await normalize_to_mp4(original_path, metadata)
    except NormalizationUnavailable:
        discard_temp_file(original_path)
        report_possible_orphan(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffmpeg is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except NormalizationError:
        discard_temp_file(original_path)
        report_possible_orphan(storage_key)
        logger.info("Could not normalize admitted upload %s.", storage_key, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="video could not be normalized for analysis",
        ) from None

    try:
        key = store_derivative(derivative.path, derivative.sha256)
    except Exception:
        discard_temp_file(derivative.path)
        discard_temp_file(original_path)
        # The upload may have created the derivative object before failing.
        report_possible_orphan(storage_key, derivative_key(derivative.sha256))
        # Endpoints, credentials and SDK errors stay in the server log, not the response.
        logger.exception("Storing the normalized derivative in MinIO failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media storage unavailable",
        ) from None

    return key, derivative.sha256, derivative.path


def strongest_clips(clips: tuple[NvidiaClipResult, ...]) -> list[AnalysisSegment]:
    """The provider's most strongly scored clips, capped at what may be persisted.

    Ranking is by NVIDIA's own logit, descending: a higher logit is the provider saying
    this clip looks more synthetic, so keeping the top of that ordering keeps the evidence
    a reader would actually want to see when the aggregate looks suspicious. The clip
    index breaks ties, so the same detection always yields the same rows.

    This is a selection, not a judgement. No threshold decides which clips are "suspicious"
    and nothing here is classified — the figures are handed on exactly as NVIDIA sent them,
    and what they mean is still nobody's call in this codebase.
    """
    ranked = sorted(clips, key=lambda clip: (-clip.logit, clip.index))

    return [
        AnalysisSegment(clip_index=clip.index, logit=clip.logit)
        for clip in ranked[:MAX_PERSISTED_SEGMENTS]
    ]


async def detect_synthetic_video(file_path: Path) -> tuple[AnalysisSignal, list[AnalysisSegment]]:
    """Ask NVIDIA about the local artifact and turn the outcome into forensic evidence.

    Every *provider* failure becomes a signal rather than an exception. A detector that
    fails is a fact about the detector, not about the upload: the analysis is still real,
    its media is still stored, and the failure is recorded in its own right so the gap is
    visible rather than silent.

    A failure on our own side is not that. `NvidiaLocalFileError` means the artifact this
    server prepared moments ago can no longer be read — a broken machine, not a broken
    provider — and recording it as a provider signal would blame NVIDIA for our fault and
    leave a real defect looking like routine evidence. It propagates.

    NVIDIA's probability is written down exactly as returned, on NVIDIA's own scale.
    Interpreting it — the `risk_level` column — is not this task's, or this layer's, job,
    so it stays null. `metadata` keeps only NVIDIA's two aggregate figures; the clip table
    it also returns is timeline evidence and travels separately, as `analysis_segments`
    rows, rather than being dumped into the JSON column as unbounded provider output.

    Returns the signal and the clip evidence behind it. A failure has no clip evidence:
    the list is empty, never a placeholder row.
    """
    signal = AnalysisSignal(
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
    )

    try:
        result = await analyze_video(file_path)
    except NvidiaProviderTimeout as error:
        # The one failure worth telling apart at a glance: NVIDIA may still have been
        # working, so this says nothing about the video either way.
        logger.warning("NVIDIA synthetic-video detection timed out.", exc_info=True)
        signal.status = SIGNAL_STATUS_TIMEOUT
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []
    except NvidiaProviderError as error:
        # Every other provider-side outcome — rejected credentials, an unreachable
        # service, a truncated stream — is one status. Telling them apart in the schema
        # would need product requirements that do not exist yet; the server log carries
        # the detail, and the message never reaches the client.
        logger.warning("NVIDIA synthetic-video detection failed.", exc_info=True)
        signal.status = SIGNAL_STATUS_FAILED
        # The failure kind, never NVIDIA's message: that text can quote request detail.
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []

    signal.status = SIGNAL_STATUS_SUCCESS
    # Untransformed and unrounded. This is NVIDIA's number, not DeepGuard's.
    signal.score = result.probability
    signal.provider_version = result.function_id
    signal.signal_metadata = {"logit": result.logit, "total_clips": result.total_clips}

    return signal, strongest_clips(result.clips)


def persist_analysis(
    session: Session,
    *,
    filename: str | None,
    content_type: str,
    stored: StoredUpload,
    storage_key: str,
    metadata: MediaMetadata,
    was_normalized: bool,
    derivative_storage_key: str,
    derivative_sha256: str | None,
    signal: AnalysisSignal,
    segments: list[AnalysisSegment],
) -> Analysis:
    """Write the completed analysis, its media and its detector evidence in one transaction.

    Called only once the whole pipeline has run, so the row is complete the moment it
    exists — including the evidence, which is why the signal is written here rather than
    appended afterwards. Either every row exists or none does; an analysis whose detector
    result silently went missing would be worse than no analysis at all, and a signal
    whose clip evidence went missing would misreport how much the detector actually said.

    Detection has already finished by the time this is called, so the transaction lasts
    as long as three inserts and never spans the minutes NVIDIA can take.

    On failure the session is rolled back and the stored objects are reported rather than
    deleted, since they are content-addressed and may be shared.
    """
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
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
            derivative_sha256=derivative_sha256,
        )
    )

    signal.analysis_id = analysis.id
    session.add(signal)

    if segments:
        # The segments hang off the signal, so its id has to exist before they can name
        # it — a second flush inside the same transaction, not a second transaction.
        session.flush()
        for segment in segments:
            segment.signal_id = signal.id
        session.add_all(segments)

    session.commit()

    return analysis


@router.post("/analyses", response_model=CreatedAnalysis)
async def create_analysis(
    file: UploadFile, session: Session = Depends(get_session)
) -> CreatedAnalysis:
    """Admit an upload by declared MIME type and size, then prove it is real media.

    The declared content type is not proof that the bytes are a real MP4/MOV container,
    so admission alone never produces a 200.

    The admitted upload is stored in MinIO as the forensic original, then probed with
    ffprobe to confirm the bytes really are video and to extract the metadata later
    tasks need. Media that is not already in the canonical provider shape gets a
    separate normalized derivative (D013) — the original is never rewritten.

    Whichever of the two is provider-compatible is then handed to NVIDIA's synthetic
    video detector while it is still on local disk. That call is made before any database
    work starts: it can take minutes, and no transaction is held open across it. NVIDIA
    failing does not fail the upload — the failure is persisted as a signal of its own.
    A failure to read our own staged artifact is a server fault, not a detector one, and
    is the exception: it surfaces as a 500 like any other broken-machine condition.

    The analysis, its media, the signal and the clip evidence behind it are persisted
    last, in one transaction, once nothing is left on local disk. On failure the temp
    files are dropped; the stored
    objects are kept, because their content-addressed keys may be shared with an earlier
    analysis.
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
    if was_normalized:
        canonical_key, derivative_sha256, inference_path = await create_derivative(
            stored.path, storage_key, metadata
        )
    else:
        # Already canonical: no transcode, and no duplicate object of the original. The
        # staged original is itself the artifact the detector should read.
        canonical_key, derivative_sha256 = storage_key, None
        inference_path = stored.path

    try:
        signal, segments = await detect_synthetic_video(inference_path)
    finally:
        # Detection is the last step that reads local media, so nothing may survive it —
        # not on the success path, and not when it raised something unforeseen.
        discard_temp_file(stored.path)
        if was_normalized:
            discard_temp_file(inference_path)

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
            derivative_sha256=derivative_sha256,
            signal=signal,
            segments=segments,
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
        report_possible_orphan(storage_key, canonical_key)
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
        derivative_sha256=derivative_sha256,
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
    """Return the most recent analyses, with their NVIDIA signal, for the dashboard.

    An inner join onto the media is correct here rather than restrictive: an analysis and
    its media row are written in one transaction, so an analysis without media cannot
    exist. The signal is a second, outer join, because it can genuinely be missing —
    analyses stored before the detector existed have none — and the join condition names
    the one detector wired in, so a later provider's signal cannot multiply these rows.

    Two statements serve the whole page — the analyses, then the clip evidence for every
    signal on it. Both are fixed in number: neither grows with how many analyses are
    listed, so there is no query per analysis.

    Ordering falls back to the id because `created_at` defaults to the transaction
    timestamp, which two analyses committed together can share — without the tiebreak
    their relative order would be arbitrary between calls.
    """
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
                # Labelled, because `status` and `created_at` exist on both tables and the
                # unlabelled columns would collide in the result row.
                AnalysisSignal.id.label("signal_id"),
                AnalysisSignal.provider.label("signal_provider"),
                AnalysisSignal.signal_type.label("signal_type"),
                AnalysisSignal.status.label("signal_status"),
                AnalysisSignal.score.label("signal_score"),
                AnalysisSignal.provider_version.label("signal_provider_version"),
                AnalysisSignal.signal_metadata.label("signal_metadata"),
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
            synthetic_video=synthetic_video_signal(row, segments.get(row.signal_id, [])),
        )
        for row in rows
    ]
