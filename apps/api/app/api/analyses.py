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
from sqlalchemy import and_, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased

from app.db.models import (
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_QUEUED,
    USER_ROLE_ADMIN,
    Analysis,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    ApiKey,
    MediaFile,
    User,
)
from app.db.session import get_session
from app.detection import (
    AASIST_PROVIDER,
    ACTIVE_SPEAKER_SIGNAL,
    AUDIO_AUTHENTICITY_SIGNAL,
    C2PA_PROVIDER,
    NVIDIA_PROVIDER,
    PROVENANCE_SIGNAL,
    SYNTHETIC_VIDEO_SIGNAL,
)
from app.media import (
    MAX_UPLOAD_BYTES,
    MediaMetadata,
    MediaProbeError,
    MediaProbeUnavailable,
    probe_media,
)
from app.normalization import needs_normalization
from app.observability import current_request_id
from app.storage import store_original
from app.web_auth import require_same_origin, require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])

ALLOWED_CONTENT_TYPES = frozenset({"video/mp4", "video/quicktime"})
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


class SpeakingSegment(BaseModel):
    """One stretch of video in which NVIDIA saw a tracked face speaking.

    Times are seconds from the start of the analysed video, as the aggregation over
    NVIDIA's per-frame evidence produced them. `face_id` is NVIDIA's own identifier for the
    face it tracked; `speaker_label` is pyannote's label for the voice NVIDIA matched that
    face to, and is null when it matched none — which is an observation about the segment,
    not missing data.

    There is no score here, because active speaker produces no figure per segment. A range
    is either in the timeline or it is not.
    """

    start_time: float
    end_time: float
    face_id: int
    speaker_label: str | None


class ActiveSpeakerSignal(BaseModel):
    """The persisted NVIDIA active-speaker signal, as the dashboard may see it.

    No `score`, unlike the synthetic-video signal: this detector reports a timeline, and a
    number here would sit in the same place as NVIDIA's synthetic probability as though the
    two could be compared (rule 11).

    A successful signal with an empty `segments` is a real result — the detector ran and
    saw no speaking face — and is not evidence of anything about the media. A `FAILED` or
    `TIMEOUT` signal carries no segments either, and the two states must not be read alike:
    one looked and found nothing, the other did not get to look.

    The stored metadata document is not passed through as it stands; on a failure it holds
    the provider exception's class name, which is internal diagnostic detail.
    """

    provider: str
    signal_type: str
    status: str
    provider_version: str | None
    # How many speaking runs the aggregation found, and whether the persisted timeline stops
    # short of them. Both absent unless the detection succeeded and recorded them — without
    # the pair, a truncated timeline would read as the whole one.
    total_speaking_segments: int | None
    segments_truncated: bool | None
    # The persisted timeline, chronological. Already capped where it was written
    # (`MAX_PERSISTED_SPEAKING_SEGMENTS`), and not trimmed again here: a timeline is only
    # readable as a contiguous run, so what was stored is what is handed on.
    segments: list[SpeakingSegment]


class AudioWindowEvidence(BaseModel):
    """One window of audio the local checkpoint was given, and the two figures it emitted.

    `clip_index` is the window's place in the chronological sequence DeepGuard cut, carried
    under the name of the column it is stored in. `start_time` and `end_time` are that
    window's sample bounds restated in seconds — `start_sample / 16000`, arithmetic on
    boundaries this codebase chose. They are **preprocessing bounds**: AASIST publishes no
    mapping from its fixed window to time and reports no segments, so nothing here says the
    model located anything between those two times.

    `logit` and `bona_fide_logit` are the graph's two outputs in graph order, untouched.
    Upstream reads the second as the bona fide column; that is the checkpoint's own fact
    about its model. Both are raw logits — not probabilities, not confidence, and not a
    threshold away from a verdict, because the model ships no threshold at all.
    """

    clip_index: int
    start_time: float
    end_time: float
    logit: float
    bona_fide_logit: float


class AudioAuthenticitySignal(BaseModel):
    """The persisted local audio-authenticity signal, as the dashboard may see it.

    No `score`, and there is no column of any kind here that could hold one: the checkpoint
    emits two raw logits per window with no softmax, threshold or calibration over them, so
    there is no file-level figure to report and none is derived. The windows are the whole
    of the evidence, and they are neither averaged nor ranked (rule 11).

    The same four states the other signals keep apart apply here and must not be merged: an
    analysis carrying no signal at all, a reading that did not happen (`FAILED`/`TIMEOUT`),
    a reading that ran and persisted no windows, and a reading with windows. Only the last
    says anything factual about the audio, and even then only what the model emitted.

    The stored metadata document is not passed through as it stands; on a failure it holds
    the exception's class name, which is internal diagnostic detail.
    """

    provider: str
    signal_type: str
    status: str
    # The repository and revision of the checkpoint that produced these figures. A different
    # revision is a different measurement, not a refinement of this one.
    provider_version: str | None
    # How many windows the sweep produced against how many were persisted, and whether the
    # stored set stops short. All three absent unless the reading succeeded and recorded
    # them — without them, a truncated prefix would read as the whole recording.
    total_audio_windows: int | None
    persisted_audio_windows: int | None
    windows_truncated: bool | None
    # The persisted windows, chronological. Already capped where they were written
    # (`MAX_PERSISTED_AUDIO_WINDOWS`) and not trimmed again here: this is a sweep from the
    # start of the audio, and cutting it a second time would move the boundary the count
    # above describes.
    windows: list[AudioWindowEvidence]


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
    """One analysis as its readers get it: the dashboard listing and the report route.

    Deliberately narrower than `CreatedAnalysis`: the storage keys and derivative identity
    are not shown to either reader. Both get the same fields because both are rendering the
    same stored record — the report is one analysis presented at length, not a different set
    of facts, and giving it its own model would be two places to add a field to and one
    place to forget.

    The risk fields are read straight off the `analyses` row. Nothing here evaluates a
    rule or looks at a detector score: the decision the worker committed is the decision,
    and recomputing it in a read path would let the listing disagree with the record it is
    supposed to be showing.
    """

    id: uuid.UUID
    status: str
    created_at: datetime
    # What the risk engine concluded, as it was persisted: `HIGH`, `MEDIUM` or `UNKNOWN`.
    # `LOW` is measured but not activated in ruleset v1, so no analysis carries it.
    #
    # Null and `UNKNOWN` are two different facts and must never be folded together. Null
    # is the absence of a decision — an analysis still queued or being worked on, or one
    # completed before the engine existed. `UNKNOWN` is a decision: the engine ran, a rule
    # fired, and the answer is that the evidence does not support a classification.
    risk_level: str | None = None
    # The trace behind the level: which immutable ruleset was in force, which single rule
    # fired, and which calibration measurement its thresholds came from. Null exactly when
    # `risk_level` is, since all four columns are written in one transaction.
    #
    # Without these a stored level is unreadable once the rules move on — `HIGH` under
    # `p7-v1.0.0` is not the same statement as `HIGH` under whatever replaces it.
    risk_rules_version: str | None = None
    risk_rule_id: str | None = None
    risk_calibration_id: str | None = None
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
    # Null for an analysis carrying no active-speaker signal at all, which is again not the
    # same as a detector that ran and found no speaking face.
    active_speaker: ActiveSpeakerSignal | None
    # Null for an analysis carrying no audio-authenticity signal — everything stored before
    # the local checkpoint was wired in. Not the same as a reading that ran and persisted no
    # windows, and not the same as one that could not run.
    audio_authenticity: AudioAuthenticitySignal | None


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


class ActiveAnalysisLimitReached(Exception):
    """One API key already has as much work outstanding as it is allowed.

    Not an `HTTPException`. This is raised from inside the persistence transaction, which
    has no business deciding a status code — the public route it belongs to maps it to a
    `429`, and the internal route can never see it because it passes no limit.
    """

    def __init__(self, active: int, limit: int) -> None:
        super().__init__(f"{active} active analyses, limit {limit}")
        self.active = active
        self.limit = limit


def active_analyses(session: Session, api_key_id: uuid.UUID) -> int:
    """How much work this key still has outstanding.

    Outstanding is `queued`, and that single status covers both waiting and running: the
    worker moves a job to `processing` but leaves its analysis `queued` until it finishes,
    so an analysis is `queued` from the moment it is accepted to the moment it is
    `completed` or `failed`. Counting that one status is therefore exactly the concurrent
    work in flight, and finished analyses drop out of it on their own — nothing has to
    decrement a counter, and a worker that dies mid-job cannot strand one.

    Counted in the database rather than by loading rows: this runs on the upload path, and
    what is wanted is the number, not the analyses.
    """
    return session.execute(
        select(func.count())
        .select_from(Analysis)
        .where(
            Analysis.api_key_id == api_key_id,
            Analysis.status == ANALYSIS_STATUS_QUEUED,
        )
    ).scalar_one()


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
    api_key_id: uuid.UUID | None = None,
    owner_id: uuid.UUID | None = None,
    max_active_analyses: int | None = None,
) -> Analysis:
    """Write the queued analysis, its media and the job it is owed in one transaction.

    `api_key_id` is the key that submitted the upload, and `owner_id` the signed-in person
    who did — never both, which is the invariant `ck_analyses_single_owner` holds in the
    database rather than trusting the two callers that pass them. The public routes pass a
    key and no user; the dashboard routes, since R1-T2, pass a user resolved from the
    session cookie and no key. Either is written in the same insert as the analysis rather
    than updated onto it afterwards: ownership is what both surfaces' isolation rests on,
    and a row that existed for even one commit without its owner would be a row no read
    could reach and no owner could be inferred for.

    `max_active_analyses` caps how much work one key may have outstanding, and only the
    public route passes it; the dashboard is not throttled. It is enforced *here* rather
    than at the top of the route because the check and the insert have to be one atomic
    step. A count taken before the upload and acted on after it is not a limit at all: two
    requests from the same key would both read four, both admit, and both insert. So the
    key's own row is locked `FOR UPDATE` first, the count is taken under that lock, and the
    lock is only released by the commit that writes the analysis — which means a second
    request for the same key waits at the lock and then counts a database that already
    contains the first one's row.

    The lock is taken *late*, once the bytes are stored and probed, and is held for three
    inserts. Locking at the start of the request instead would be simpler and much worse:
    it would hold the key's row across the read, the upload to MinIO and ffprobe, turning a
    concurrency limit into a mutex that lets one customer run exactly one upload at a time.

    Locking the `api_keys` row rather than the analyses is what makes this correct. There is
    no row to lock for an analysis that does not exist yet, so the thing every competing
    request for one key has in common — the key itself — is the serialization point.
    Different keys lock different rows and never wait on each other.

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
    if max_active_analyses is not None:
        # The serialization point. Everything from here to the commit below is one
        # transaction holding this key's row, so the count cannot be stale by the time the
        # insert lands.
        session.execute(
            select(ApiKey.id).where(ApiKey.id == api_key_id).with_for_update()
        ).scalar_one_or_none()

        active = active_analyses(session, api_key_id)
        if active >= max_active_analyses:
            raise ActiveAnalysisLimitReached(active, max_active_analyses)

    analysis = Analysis(
        status=ANALYSIS_STATUS_QUEUED, api_key_id=api_key_id, owner_id=owner_id
    )
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

    # The request that asked for this analysis, written with the job so the worker can bind
    # it to its own logs minutes later (R1-T4). Read from the request context rather than
    # taken as a parameter: all three submission routes would otherwise have to accept and
    # forward an argument none of them has any other use for, and the id is ambient context
    # by nature — see `app.observability`. Null outside a request, which is what a job
    # queued by anything other than one honestly is.
    session.add(
        AnalysisJob(
            analysis_id=analysis.id,
            status=JOB_STATUS_QUEUED,
            request_id=current_request_id(),
        )
    )

    session.commit()

    return analysis


@dataclass(frozen=True)
class AcceptedUpload:
    """A committed, queued analysis and everything the request established about it.

    Internal, and shaped by what the pipeline produced rather than by what any one response
    shows: the dashboard's `CreatedAnalysis` reports nearly all of it, the public API
    reports the id and the status and nothing else. Neither endpoint is the reason these
    fields exist, which is why the storage keys can sit here without being exposed anywhere
    they should not be.
    """

    analysis: Analysis
    filename: str | None
    content_type: str
    stored: StoredUpload
    storage_key: str
    metadata: MediaMetadata
    was_normalized: bool
    derivative_storage_key: str | None


async def accept_upload(
    file: UploadFile,
    session: Session,
    *,
    api_key_id: uuid.UUID | None = None,
    owner_id: uuid.UUID | None = None,
    max_active_analyses: int | None = None,
) -> AcceptedUpload:
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

    Extracted from the route at the second caller (P9-T2) rather than in advance: the
    public API accepts uploads under the same rules, and a second copy of this admission,
    validation and queueing would be a second place for the size limit, the content-type
    set, the orphan reporting and the failure mapping to drift apart. The route it came
    from now only shapes a response; every `HTTPException` a caller can raise is raised
    here, so both endpoints refuse the same upload for the same reason with the same status.

    `max_active_analyses` is the one thing the two callers genuinely differ on, and it is
    passed straight through to `persist_analysis`, which is where it has to be enforced.
    Set, it can end this call in `ActiveAnalysisLimitReached` — the only failure here that
    is not already an `HTTPException`, because the status it deserves is the public route's
    to choose.
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
            api_key_id=api_key_id,
            owner_id=owner_id,
            max_active_analyses=max_active_analyses,
        )
    except ActiveAnalysisLimitReached:
        # Rolled back explicitly rather than left to the session teardown: the transaction
        # is holding this key's row `FOR UPDATE`, and every other request from the same key
        # is queued behind it. Releasing it here is what keeps a refusal from delaying the
        # requests that are about to be refused too.
        session.rollback()
        # The bytes reached MinIO before the limit was consulted, and no analysis now
        # references them. They are content-addressed and may be shared with an earlier
        # analysis, so they are reported rather than deleted, as everywhere else here.
        report_possible_orphan(storage_key)
        raise
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

    return AcceptedUpload(
        analysis=analysis,
        filename=file.filename,
        content_type=content_type,
        stored=stored,
        storage_key=storage_key,
        metadata=metadata,
        was_normalized=was_normalized,
        derivative_storage_key=canonical_key,
    )


def created_analysis(accepted: AcceptedUpload) -> CreatedAnalysis:
    """The internal response for a staged analysis, however the media reached the pipeline.

    Extracted at the second caller (P10-T2), which is the URL route: an upload and a URL
    submission establish exactly the same facts, and two copies of this projection would be
    two places to add a field to and one place to forget.
    """
    return CreatedAnalysis(
        id=accepted.analysis.id,
        status=accepted.analysis.status,
        filename=accepted.filename,
        content_type=accepted.content_type,
        size_bytes=accepted.stored.size_bytes,
        sha256=accepted.stored.sha256,
        storage_key=accepted.storage_key,
        metadata=accepted.metadata,
        was_normalized=accepted.was_normalized,
        derivative_storage_key=accepted.derivative_storage_key,
    )


@router.post(
    "/analyses",
    response_model=CreatedAnalysis,
    status_code=status.HTTP_202_ACCEPTED,
    # The CSRF boundary, declared as a route dependency rather than as a parameter because
    # nothing in the handler needs its result: it either refuses the request or says nothing.
    # It is on the POST alone — the reads above change no state, and adding it to them would
    # make an origin header a condition of looking at the dashboard.
    dependencies=[Depends(require_same_origin)],
)
async def create_analysis(
    file: UploadFile,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
) -> CreatedAnalysis:
    """Accept a dashboard upload and report everything the request established about it.

    Two dependencies stand in front of this: the session that says who the caller is, and
    `require_same_origin`, which says the request came from this deployment's own web
    application rather than from a page somewhere else that happens to be open in the same
    browser. Neither covers the other.

    The owner is the account the session cookie resolves to, and nothing else. There is no
    `owner_id` in the request for a caller to choose: a field naming the owner would let a
    signed-in person file an analysis under somebody else's name, and every read below
    would then honour it. `require_user` establishes who this is; this route only passes
    that identity on.

    No `api_key_id` and no concurrency limit: this is the internal route, and the key
    ownership the public API's isolation rests on belongs to that surface. `owner_id` and
    `api_key_id` are never both set, which the database holds rather than this line.

    The work is `accept_upload`; what is left here is the response, which is wider than the
    public one on purpose — the dashboard is the same trust boundary as the server, so the
    storage keys and content identity it needs are not a leak to it.
    """
    return created_analysis(await accept_upload(file, session, owner_id=user.id))


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


def active_speaker_signal(
    row: Any, segments: list[SpeakingSegment]
) -> ActiveSpeakerSignal | None:
    """Turn the joined active-speaker columns into the response's signal, or nothing.

    Same rule as the other two joins: every column is null when an analysis carries no such
    signal, and `status` is the one a real row cannot have null.
    """
    if row.active_speaker_status is None:
        return None

    return ActiveSpeakerSignal(
        provider=row.active_speaker_provider,
        signal_type=row.active_speaker_signal_type,
        status=row.active_speaker_status,
        provider_version=row.active_speaker_provider_version,
        total_speaking_segments=signal_figure(
            row.active_speaker_metadata, "total_speaking_segments", int
        ),
        segments_truncated=signal_flag(row.active_speaker_metadata, "segments_truncated"),
        segments=segments,
    )


def audio_authenticity_signal(
    row: Any, windows: list[AudioWindowEvidence]
) -> AudioAuthenticitySignal | None:
    """Turn the joined audio columns into the response's signal, or nothing.

    Same rule as the other three joins: every column is null when an analysis carries no
    such signal, and `status` is the one a real row cannot have null.
    """
    if row.audio_status is None:
        return None

    return AudioAuthenticitySignal(
        provider=row.audio_provider,
        signal_type=row.audio_signal_type,
        status=row.audio_status,
        provider_version=row.audio_provider_version,
        total_audio_windows=signal_figure(row.audio_metadata, "total_audio_windows", int),
        persisted_audio_windows=signal_figure(
            row.audio_metadata, "persisted_audio_windows", int
        ),
        windows_truncated=signal_flag(row.audio_metadata, "windows_truncated"),
        windows=windows,
    )


def audio_windows(
    session: Session, signal_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[AudioWindowEvidence]]:
    """Fetch the persisted windows for every listed audio signal at once.

    One statement for the whole page, for the same reason as the other two evidence
    queries: the listing must not issue a query per analysis. Persisted evidence is already
    capped per signal (`MAX_PERSISTED_AUDIO_WINDOWS`), so the whole stored sweep is read
    back and handed on as it stands.

    Ordered by the window index, which is the order the audio was cut in — the only order
    this evidence has. It is total within a signal, so the same stored sweep reads back the
    same way on every call, and nothing here reorders the windows by logit: the checkpoint
    publishes no threshold, so there is no sense in which one window outranks another.

    Signals with no stored windows are simply absent from the result, which covers both a
    reading that failed and one that produced none; the signal's own status separates them.
    """
    if not signal_ids:
        return {}

    rows = session.execute(
        select(
            AnalysisSegment.signal_id,
            AnalysisSegment.clip_index,
            AnalysisSegment.start_time,
            AnalysisSegment.end_time,
            AnalysisSegment.logit,
            AnalysisSegment.bona_fide_logit,
        )
        .where(AnalysisSegment.signal_id.in_(signal_ids))
        .order_by(AnalysisSegment.signal_id, AnalysisSegment.clip_index)
    ).all()

    grouped: dict[uuid.UUID, list[AudioWindowEvidence]] = {}
    for row in rows:
        grouped.setdefault(row.signal_id, []).append(
            AudioWindowEvidence(
                clip_index=row.clip_index,
                start_time=row.start_time,
                end_time=row.end_time,
                logit=row.logit,
                bona_fide_logit=row.bona_fide_logit,
            )
        )

    return grouped


def speaking_timeline(
    session: Session, signal_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[SpeakingSegment]]:
    """Fetch the speaking segments for every listed active-speaker signal at once.

    One statement for the whole page, for the same reason as `strongest_segments`: the
    listing must not issue a query per analysis. Persisted evidence is already capped per
    signal (`MAX_PERSISTED_SPEAKING_SEGMENTS`), so the whole stored timeline is read back
    and handed on as it stands — trimming it here would leave gaps a reader could not tell
    from silence.

    Signals with no stored segments are simply absent from the result, which covers both a
    detection that failed and one that saw nobody speaking; the signal's own status is what
    separates those two.
    """
    if not signal_ids:
        return {}

    rows = session.execute(
        select(
            AnalysisSegment.signal_id,
            AnalysisSegment.start_time,
            AnalysisSegment.end_time,
            AnalysisSegment.face_id,
            AnalysisSegment.speaker_label,
        )
        .where(AnalysisSegment.signal_id.in_(signal_ids))
        # Chronological, with the same tie-breaks the evidence was written under: two faces
        # can begin speaking on the very same frame, and without a total order the timeline
        # could read back differently between calls.
        .order_by(
            AnalysisSegment.signal_id,
            AnalysisSegment.start_time,
            AnalysisSegment.end_time,
            AnalysisSegment.face_id,
        )
    ).all()

    grouped: dict[uuid.UUID, list[SpeakingSegment]] = {}
    for row in rows:
        grouped.setdefault(row.signal_id, []).append(
            SpeakingSegment(
                start_time=row.start_time,
                end_time=row.end_time,
                face_id=row.face_id,
                speaker_label=row.speaker_label,
            )
        )

    return grouped


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


def analysis_evidence_select():
    """The one statement that reads an analysis, its media and all four of its signals.

    Shared by the listing and the single-analysis endpoint, which differ only in how they
    narrow it: the listing orders and limits, the detail route filters by id. Extracted at
    the second real use rather than in advance — the alternative was duplicating ninety
    lines of columns and joins, and two copies would drift the moment a signal gained a
    field, leaving the report and the dashboard quietly disagreeing about the same row.

    An inner join onto the media is correct rather than restrictive: an analysis and its
    media row are written in one transaction, so an analysis without media cannot exist.
    The signals are outer joins, because any of them can genuinely be missing — analyses
    stored before a source was wired in have none — and each join names one provider and
    one signal type, so no row is multiplied by the evidence hanging off it.

    The risk decision rides along on the analysis row itself, so it costs neither a join
    nor a statement. It is read, never taken: nothing here calls the risk engine and
    nothing looks at a detector score to decide anything, because the decision the worker
    committed under a named ruleset is the decision, and a read path that recomputed it
    could quietly answer differently from the record.
    """
    # The same table, joined again for each further signal. Every join is narrowed to one
    # provider and one signal type, so none can multiply the rows, and all three signals
    # still arrive on the one statement.
    provenance = aliased(AnalysisSignal)
    active_speaker = aliased(AnalysisSignal)
    audio = aliased(AnalysisSignal)

    return (
        select(
                Analysis.id,
                Analysis.status,
                Analysis.created_at,
                # The persisted decision and its trace. Already on the analysis row, so
                # naming them costs no extra statement and nothing is recomputed.
                Analysis.risk_level,
                Analysis.risk_rules_version,
                Analysis.risk_rule_id,
                Analysis.risk_calibration_id,
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
                active_speaker.id.label("active_speaker_id"),
                active_speaker.provider.label("active_speaker_provider"),
                active_speaker.signal_type.label("active_speaker_signal_type"),
                active_speaker.status.label("active_speaker_status"),
                active_speaker.provider_version.label("active_speaker_provider_version"),
                active_speaker.signal_metadata.label("active_speaker_metadata"),
                audio.id.label("audio_id"),
                audio.provider.label("audio_provider"),
                audio.signal_type.label("audio_signal_type"),
                audio.status.label("audio_status"),
                audio.provider_version.label("audio_provider_version"),
                audio.signal_metadata.label("audio_metadata"),
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
            .outerjoin(
                active_speaker,
                and_(
                    active_speaker.analysis_id == Analysis.id,
                    active_speaker.provider == NVIDIA_PROVIDER,
                    active_speaker.signal_type == ACTIVE_SPEAKER_SIGNAL,
                ),
            )
        .outerjoin(
            audio,
            and_(
                audio.analysis_id == Analysis.id,
                audio.provider == AASIST_PROVIDER,
                audio.signal_type == AUDIO_AUTHENTICITY_SIGNAL,
            ),
        )
    )


def analysis_payloads(session: Session, rows: list[Any]) -> list[AnalysisSummary]:
    """Attach the stored evidence to already-read analysis rows and shape the response.

    Three further statements, whatever the number of rows: the clip evidence, the speaking
    timeline and the audio windows. They are kept apart because they read different columns
    in different orders — clips come back strongest first, the timeline chronologically, the
    windows in the order the audio was cut — and each takes every signal id at once, so
    neither caller pays a query per analysis.

    Everything is passed through exactly as stored, nulls included. Nothing here derives a
    figure, and in particular nothing turns a detector score into a classification.
    """
    segments = strongest_segments(
        session, [row.signal_id for row in rows if row.signal_id is not None]
    )
    timelines = speaking_timeline(
        session,
        [row.active_speaker_id for row in rows if row.active_speaker_id is not None],
    )
    windows = audio_windows(
        session, [row.audio_id for row in rows if row.audio_id is not None]
    )

    return [
        AnalysisSummary(
            id=row.id,
            status=row.status,
            created_at=row.created_at,
            # Passed through exactly as stored, null included.
            risk_level=row.risk_level,
            risk_rules_version=row.risk_rules_version,
            risk_rule_id=row.risk_rule_id,
            risk_calibration_id=row.risk_calibration_id,
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
            active_speaker=active_speaker_signal(
                row, timelines.get(row.active_speaker_id, [])
            ),
            audio_authenticity=audio_authenticity_signal(
                row, windows.get(row.audio_id, [])
            ),
        )
        for row in rows
    ]


def visible_to(statement, user: User):
    """Narrow a read of the analyses to the ones this account is allowed to see.

    One function, applied by both read routes, because the alternative is two ownership
    rules that can disagree — and the pair that disagrees is a listing that hides another
    person's analysis next to a detail route that serves it to anyone holding the id.

    An administrator's statement is returned untouched: the internal dashboard is where an
    operator looks at the whole system, and that is the role the distinction exists for.

    For everyone else the filter is `owner_id = <this account>`, and it is a `WHERE` clause
    rather than a check on the rows that come back. That single equality also settles the
    three cases the plan calls out separately, because none of them can satisfy it: another
    user's analysis names a different owner, an API-key submission names none, and an
    analysis stored before there were accounts names none either. All three are simply not
    in the result, and a caller cannot tell them apart from an id that never existed —
    which is the point of answering 404 rather than 403.
    """
    if user.role == USER_ROLE_ADMIN:
        return statement

    return statement.where(Analysis.owner_id == user.id)


@router.get("/analyses", response_model=list[AnalysisSummary])
def list_analyses(
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
) -> list[AnalysisSummary]:
    """Return the most recent analyses this account may see, with all four of their signals.

    Authenticated since R1-T2, and filtered in the statement: a signed-in USER is shown the
    analyses they own and an ADMIN is shown all of them. There is no parameter here for a
    caller to widen that with — the account comes from the session cookie, so changing which
    analyses come back means signing in as somebody else.

    Four statements serve the whole page — the analyses with their signals, then the three
    evidence reads in `analysis_payloads`. All four are fixed in number: none grows with how
    many analyses are listed, so there is no query per analysis.

    Ordering falls back to the id because `created_at` defaults to the transaction
    timestamp, which two analyses committed together can share — without the tiebreak their
    relative order would be arbitrary between calls.
    """
    try:
        rows = session.execute(
            visible_to(analysis_evidence_select(), user)
            .order_by(Analysis.created_at.desc(), Analysis.id.desc())
            .limit(RECENT_ANALYSES_LIMIT)
        ).all()

        return analysis_payloads(session, rows)
    except SQLAlchemyError:
        # Statements, connection strings and driver errors stay in the server log.
        logger.exception("Reading the analysis listing failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analyses are temporarily unavailable",
        ) from None


@router.get("/analyses/{analysis_id}", response_model=AnalysisSummary)
def get_analysis(
    analysis_id: uuid.UUID,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
) -> AnalysisSummary:
    """Return one analysis with all four of its signals and their stored evidence.

    The report route reads this. It exists so a single report is built from the row it is
    about: filtering the listing down to one analysis in the browser would make the report
    silently dependent on that analysis still being among the most recent, and would ship
    every other analysis to a page that shows one.

    Four statements, the same as the listing and for the same reason — the shape of the read
    does not change just because one row comes back.

    The response model is the listing's. That is not laziness: the report needs exactly the
    persisted facts the dashboard needs, and a second model repeating them would be two
    places to add a field to and one place to forget. The name says "summary" because what
    both readers get is a summary *of the stored evidence* — the full record lives in the
    database, and neither endpoint invents anything on top of it.

    A `uuid.UUID` path parameter means a malformed id is rejected by validation as a 422
    before any statement runs; only a well-formed id that names nothing reaches the 404.

    Since R1-T2 the ownership filter is part of that statement, and the 404 below therefore
    covers two things a caller cannot distinguish: an id that names no analysis, and one
    that names an analysis this account may not see. That is deliberate. A 403 for the
    second would confirm the id exists, which is exactly the fact a person guessing ids is
    trying to establish, so an analysis outside the caller's reach is simply not found.
    """
    try:
        rows = session.execute(
            visible_to(analysis_evidence_select(), user).where(
                Analysis.id == analysis_id
            )
        ).all()

        payloads = analysis_payloads(session, rows)
    except SQLAlchemyError:
        logger.exception("Reading analysis %s failed.", analysis_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="analyses are temporarily unavailable",
        ) from None

    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="analysis not found",
        )

    # The id is a primary key, so the narrowed select cannot return a second row.
    return payloads[0]
