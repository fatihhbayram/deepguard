"""The process that runs queued analyses.

Uploads stage the forensic original and record that the rest is owed; nothing in the API
ever transcodes or calls a detector. This is what does, in its own container, on its own
schedule.

Four transactions per job, never one:

1. claim — take a `queued` job, mark it `processing`, commit, release the lock;
2. nothing — the download, provenance, transcoding, diarization, both NVIDIA inferences
   and the two local checkpoints all happen with no transaction open at all;
3. persist — write the derivative's identity and every evidence row, and commit them;
4. conclude — classify the analysis from the evidence just committed, record the decision
   and close the job out.

Steps 3 and 4 are separate on purpose, and the order is the point: the risk engine reads
the evidence back out of the database rather than being handed the values in flight, so
what it classified is provably what a reader of that database will find behind the
classification (P7-T3). It also means a defect in the classification costs the decision and
not the evidence — the signals are already committed, and a job that cannot be concluded
fails with its forensic record intact.

The middle step is the reason for the first two. NVIDIA can take minutes on one video and
ffmpeg can take minutes on a 4K source; a transaction held open across either would pin a
connection and a row lock for the whole wait. Claiming is therefore deliberately separate
from finishing, and the row a worker holds between them is marked `processing` rather than
locked.

A claim is a lease, not a permanent title (P9-F1). Between step 1 and step 4 the job says
`processing` and nothing else can take it, so a worker that dies in the middle used to leave
it there forever — and since P9 that permanently consumed one of an API key's concurrency
slots. The claim therefore also writes a deadline, a background thread pushes that deadline
forward while the work runs, and every poll of this loop fails any job whose deadline has
passed. The two halves are what make it safe: the heartbeat is how a slow analysis is told
apart from a dead worker, and the conditional writes in `_set_status` are how a worker that
comes back from the dead is stopped from undoing its own recovery.

A job stopped by one of its own limits is failed but not emptied (R1-T3). Every external
operation carries a configurable deadline, and breaching one fails the job — the analysis did
not finish and must not be published as though it had. What it does not do is discard the
independent readings the job had already produced: a C2PA manifest, or a synthetic-video
verdict NVIDIA returned before an unrelated audio extraction timed out, are complete findings
in their own right (rule 11), and a deadline this worker set does not make them less true. So
`AnalysisTimedOut` carries what was obtained, `abandon_job` commits it, and the analysis is
left `failed` with no risk decision at all.

Nothing is claimed until the database is at this image's Alembic head (R1-T3). A worker
running against a half-migrated schema does not fail cleanly — it takes a job, trips over a
missing column several statements in, fails that analysis and takes the next one — so a
deployment that skipped a migration used to look like a burst of failed customer analyses.
`wait_for_schema` turns that into a worker that has not started yet and says why.

Every job is run under the id of the request that submitted it (R1-T4). The API binds one
to each request it serves and writes it onto the job row; `claim_job` reads it back and
`correlated` binds it for the length of the work, so the analysis this process does hours
later and the upload that asked for it are one grep apart. The queue is what carries it: a
correlation id held in memory would have ended with the response that queued the job.

Normalization moved here in P4-F2 (D020). It used to run on the upload request under a
deadline that was really about how long a client would wait, which rejected a perfectly
good 4K HEVC upload before anything had been analysed.

Run it with `python -m app.worker`.
"""

import asyncio
import contextvars
import logging
import signal
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PROCESSING,
    JOB_STATUS_QUEUED,
    Analysis,
    AnalysisJob,
    AnalysisSignal,
    MediaFile,
)
from app.db.schema import SchemaNotReady, check_schema_ready
from app.db.session import SessionLocal, engine
from app.limits import InvalidTimeout, validate as validate_limits
from app.detection import (
    ACTIVE_SPEAKER_SIGNAL,
    NVIDIA_PROVIDER,
    SYNTHETIC_VIDEO_SIGNAL,
    analyse_audio,
    detect_face_manipulation,
    detect_synthetic_video,
    extract_provenance,
    unanalysable_audio,
    undetectable_media,
)
from app.normalization import NormalizationError, NormalizationTimeout, normalize_to_mp4
from app.observability import bind_request_id, configure_logging, reset_request_id
from app.speaker_diarization import SpeakerDiarizationTimeout
from app.risk_engine import RiskDecision, SvdEvidence, evaluate
from app.storage import fetch_object, store_derivative

logger = logging.getLogger(__name__)

# How long to wait before asking for work again when there was none. Long enough that an
# idle worker is not a busy loop against PostgreSQL, short enough that an upload is picked
# up while the person who made it is still looking at the page.
IDLE_POLL_SECONDS = 2.0

# How long to wait after an unexpected failure in the loop itself — a database that has
# gone away, rather than a job that went wrong. Backing off further than the idle poll
# keeps a broken dependency from filling the log at full speed.
ERROR_BACKOFF_SECONDS = 5.0

TEMP_FILE_PREFIX = "deepguard-job-"

# What a failed job records. The exception's own text can quote credentials, storage
# endpoints or SQL, so the class name is what is written down and the traceback stays in
# the worker log.
MAX_ERROR_MESSAGE = 200

# How long a claim is believed for without being renewed (P9-F1). The window a worker has
# to prove it is still alive in, and therefore how long a crashed worker's job sits before
# anything reclaims the capacity it was holding.
#
# Three minutes is a deliberate overshoot on the heartbeat below. What it has to survive is
# not the analysis — that is renewed through, however long it takes — but a worker that is
# briefly unable to renew: a database failover, a paused container, a garbage-collection
# pause. Six missed heartbeats is a machine that is genuinely gone, not one having a bad
# second.
LEASE_SECONDS = 180

# How often a worker pushes the lease on the job it is running forward. Short enough that
# the lease is renewed many times over before it could expire, long enough that a job taking
# minutes costs a handful of tiny updates rather than a stream of them.
HEARTBEAT_SECONDS = 30

# What a recovered job records as its failure. A class name like every other failure here
# (`fail_job`), and for the same reason: it says what happened without quoting anything.
# Nobody caught an exception to produce it — the worker that would have is gone — so it is
# written literally.
STALE_LEASE_ERROR = "StaleWorkerLease"

# How long to wait between asking the database again whether it is at the schema this image
# expects (R1-T3). Longer than the idle poll: a migration is a deployment step measured in
# seconds at least, and a worker that cannot start yet should say so occasionally rather
# than filling the log while an operator runs `alembic upgrade`.
SCHEMA_POLL_SECONDS = 5.0


@dataclass(frozen=True)
class ClaimedJob:
    """A job this worker owns, as plain values rather than rows.

    Everything the work needs is read inside the claim transaction and carried out of it
    by value. After the commit there is no session and no lock, so an ORM row here would
    be an invitation to touch the database during transcoding and inference — the one
    thing the claim is structured to avoid.
    """

    job_id: uuid.UUID
    analysis_id: uuid.UUID
    # The forensic original, byte-for-byte as uploaded (D013). The only object that
    # exists for certain at claim time, and the only one provenance may be read from:
    # normalization re-encodes the video, which strips any C2PA manifest the upload
    # arrived with, so reading credentials off a derivative would report every normalized
    # upload as unsigned.
    original_storage_key: str
    # Whether a derivative has to be produced before a detector can read this media. The
    # upload decided it from the probe — the decision needs `major_brand`, which no column
    # holds — and this worker is what carries it out.
    normalization_required: bool
    # The rate the transcode has to hold constant. Read from `media_files` rather than
    # re-probed: it is the original's own rate, established once at upload.
    frame_rate: float
    # The request that asked for this analysis, as the API recorded it (R1-T4). Carried out
    # of the claim with everything else the work needs, and bound to this worker's logging
    # context for the length of the job, so a line written here and a line written by the
    # API minutes earlier say the same id.
    #
    # Null for a job queued before the column existed, or by anything that ran outside a
    # request. The job is run exactly the same way; its log lines simply carry no id, which
    # is the truth about them.
    request_id: str | None = None


@dataclass(frozen=True)
class AnalysisArtifact:
    """The local file a detector should be pointed at, and what it is.

    `storage_key` and `sha256` describe a derivative this worker created and uploaded, and
    are both null when no derivative was needed — the original was already canonical and
    is the artifact itself. They are what gets written back to `media_files`, so they name
    an object that provably exists by the time anything records them.
    """

    path: Path
    storage_key: str | None = None
    sha256: str | None = None


def lease_deadline():
    """How far ahead of *the database's* clock a fresh lease reaches.

    `now()` rather than a Python timestamp, and the arithmetic done in PostgreSQL, because
    the comparison that decides staleness happens there too. Two workers on machines whose
    clocks disagree would otherwise write deadlines on one timeline and have them judged on
    another — and a worker running a few minutes fast would have its live jobs recovered out
    from under it.
    """
    return func.now() + timedelta(seconds=LEASE_SECONDS)


def recover_stale_jobs(session: Session) -> int:
    """Fail every `processing` job whose worker stopped saying it was alive.

    A job is stale when its lease has run out: the worker that claimed it promised to push
    the deadline forward every `HEARTBEAT_SECONDS` and has not. That is the whole test, and
    it is deliberately not "this row has not changed in a while" — a real analysis writes
    nothing for the minutes it spends in ffmpeg and inference, so age would fail live work
    and spare a worker that crashed immediately after a write.

    Recovery is terminal. The job is failed rather than returned to `queued`, because
    nothing here knows why the worker died: a video that reliably kills the process would be
    handed straight back to the next worker and take that one down too, until the queue held
    nothing else. A failed analysis is visible, attributable and cheap for a customer to
    resubmit; a poison job that recirculates is none of those.

    The parent analysis fails in the same transaction, which is the point of the task. An
    analysis left `queued` behind a job nobody will ever run again looks like work still
    coming, and — since P9 — permanently consumes one of its API key's concurrency slots.

    Safe against several workers running it at once without `SKIP LOCKED`. Two of them
    matching the same row means the second blocks on the first's row lock, and PostgreSQL
    re-checks the `WHERE` clause against the committed row when it is released: the status
    is no longer `processing`, the row drops out, and only the worker that actually changed
    it gets it back from `RETURNING`. Each stale job is therefore recovered once and reported
    once, however many workers are looking.
    """
    recovered = session.execute(
        AnalysisJob.__table__.update()
        .where(
            AnalysisJob.status == JOB_STATUS_PROCESSING,
            # Never null for a claimed job, but stated rather than assumed: `NULL < now()`
            # is null, so a row that somehow had no lease would silently never be reached.
            AnalysisJob.lease_expires_at.is_not(None),
            AnalysisJob.lease_expires_at < func.now(),
        )
        .values(
            status=JOB_STATUS_FAILED,
            error_message=STALE_LEASE_ERROR,
            # A lease only means something while a job is running. Clearing it keeps a
            # terminal row from carrying a deadline nobody is keeping.
            lease_expires_at=None,
        )
        .returning(AnalysisJob.analysis_id)
    ).scalars().all()

    if not recovered:
        session.rollback()
        return 0

    session.execute(
        Analysis.__table__.update()
        .where(Analysis.id.in_(recovered))
        # No risk columns. Nothing classified these analyses, and null is the absence of a
        # conclusion rather than `UNKNOWN`, which is a conclusion a rule reached.
        .values(status=ANALYSIS_STATUS_FAILED)
    )
    session.commit()

    logger.warning(
        "Recovered %s stale job(s) whose worker stopped renewing its lease: %s.",
        len(recovered),
        ", ".join(str(analysis_id) for analysis_id in recovered),
    )

    return len(recovered)


def renew_lease(session: Session, job_id: uuid.UUID) -> bool:
    """Push one job's deadline forward. Returns whether the job was still this worker's.

    Conditional on the job still being `processing`, which is what stops a worker that was
    already recovered from quietly taking its job back. A recovered job is `failed`, the
    update matches nothing, and the caller learns to stop.
    """
    updated = session.execute(
        AnalysisJob.__table__.update()
        .where(
            AnalysisJob.id == job_id,
            AnalysisJob.status == JOB_STATUS_PROCESSING,
        )
        .values(lease_expires_at=lease_deadline())
    )
    session.commit()

    return updated.rowcount == 1


def _renew_until_stopped(job_id: uuid.UUID, stopped: threading.Event) -> None:
    """Renew one job's lease on its own connection until told to stop.

    Its own session, not the one running the job: a `Session` is not safe to use from two
    threads, and the thread that is working spends most of the job with no session open at
    all. Its own connection is also the only way this keeps working while the main thread is
    blocked in ffmpeg or waiting on NVIDIA — which is precisely when the lease needs
    renewing.

    A renewal that fails is logged and retried rather than escalated. The heartbeat cannot
    stop the analysis, and one missed update is survivable by design: the lease is six
    heartbeats long. Losing the *job* — the conditional update matching nothing — is
    different and does end the loop, because there is no longer anything to keep alive.
    """
    while not stopped.wait(HEARTBEAT_SECONDS):
        try:
            with SessionLocal() as session:
                if not renew_lease(session, job_id):
                    logger.warning(
                        "Job %s is no longer this worker's to renew; it was recovered as "
                        "stale. Its result will be discarded.",
                        job_id,
                    )
                    return
        except Exception:
            # The lease outlives several of these, so a database that comes back within a
            # couple of minutes costs nothing. One that does not will expire the lease, and
            # a worker that cannot reach PostgreSQL cannot finish the job either.
            logger.exception("Renewing the lease on job %s failed; will retry.", job_id)


@contextmanager
def leased(job_id: uuid.UUID) -> Iterator[None]:
    """Keep a claimed job's lease alive for the length of the block.

    A daemon thread, so a worker killed with the block still open cannot be held open by it
    — the whole mechanism is built for the case where this process dies without unwinding,
    and a non-daemon thread would make that shutdown hang.

    The thread is stopped and joined on the way out, including on the exception paths, so a
    finished job stops being renewed at once rather than holding capacity for another lease.

    It runs in a copy of this context rather than in a fresh one, which is what carries the
    job's request id into it (R1-T4). A new thread otherwise starts with every `ContextVar`
    at its default, so the one line this thread can write — the warning that the job was
    recovered out from under it — would be the one line about the job that could not be
    found by its correlation id.
    """
    stopped = threading.Event()
    heartbeat = threading.Thread(
        target=contextvars.copy_context().run,
        args=(_renew_until_stopped, job_id, stopped),
        name=f"lease-{job_id}",
        daemon=True,
    )
    heartbeat.start()

    try:
        yield
    finally:
        stopped.set()
        heartbeat.join(timeout=HEARTBEAT_SECONDS)


@contextmanager
def correlated(claimed: ClaimedJob) -> Iterator[None]:
    """Log everything inside the block under the request that asked for this job (R1-T4).

    The worker's half of the correlation the API starts. The id was written with the job at
    enqueue time and read back with the claim, so binding it here makes every line this
    process writes about this job — its own, SQLAlchemy's, a library's warning — carry the
    same id as the request that submitted the media, minutes or hours earlier.

    A job with no recorded id binds nothing rather than inventing one. An id the worker made
    up would correlate to no other line in any log, which is worse than an absent field: it
    looks like a trace and is not one.

    Unbound on the way out, including on the exception paths. This process runs job after
    job in one thread, so a binding left standing would attribute the next job's lines — and
    every idle poll between them — to the request that submitted the last one.
    """
    token = bind_request_id(claimed.request_id)
    try:
        yield
    finally:
        reset_request_id(token)


def claim_job(session: Session) -> ClaimedJob | None:
    """Take exclusive ownership of one queued job, or return nothing if there is none.

    `FOR UPDATE ... SKIP LOCKED` is what makes more than one worker safe: two of them
    racing for the same row do not queue behind each other and do not both get it — the
    loser skips that row and takes the next one. The lock lives only as long as this
    transaction, which is why the row is also moved to `processing` before committing:
    once the lock is gone, that status is the only thing stopping a second worker from
    picking the same job up.

    Only the job row is locked. The join to the media exists to read the storage key in
    the same statement, and locking media rows would make two workers contend over
    unrelated analyses of identical bytes.

    Oldest first, so a queue under load stays a queue rather than a stack.

    The claim also starts the lease (P9-F1), in this same transaction. A row that was
    `processing` for even one commit without a deadline would be a job no recovery could
    ever reach — `NULL < now()` is null — so the status and the promise to keep renewing it
    are made together or not at all.
    """
    row = session.execute(
        select(
            AnalysisJob,
            MediaFile.original_storage_key,
            MediaFile.was_normalized,
            MediaFile.frame_rate,
        )
        .join(MediaFile, MediaFile.analysis_id == AnalysisJob.analysis_id)
        .where(AnalysisJob.status == JOB_STATUS_QUEUED)
        .order_by(AnalysisJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True, of=AnalysisJob)
    ).first()

    if row is None:
        session.rollback()
        return None

    job, original_storage_key, normalization_required, frame_rate = row
    job.status = JOB_STATUS_PROCESSING
    job.lease_expires_at = lease_deadline()
    claimed = ClaimedJob(
        job_id=job.id,
        analysis_id=job.analysis_id,
        original_storage_key=original_storage_key,
        normalization_required=normalization_required,
        frame_rate=frame_rate,
        # Read here, inside the claim, like everything else this job needs: after the commit
        # there is no session, and fetching the correlation id in a second statement would be
        # a round trip for a value this one already has in hand.
        request_id=job.request_id,
    )
    session.commit()

    return claimed


@contextmanager
def fetched_artifact(storage_key: str, suffix: str = "") -> Iterator[Path]:
    """Download one stored object to a temp file for the block, and remove it after.

    The artifact is downloaded rather than passed along, because the process that stored
    it was a different one on a different machine's filesystem. The object store is the
    only thing the upload and this worker share.

    Removal happens on every path, failures included: a container that ran for a week
    would otherwise fill its disk with every video it had ever been asked about.
    """
    handle = tempfile.NamedTemporaryFile(prefix=TEMP_FILE_PREFIX, suffix=suffix, delete=False)
    handle.close()
    path = Path(handle.name)

    try:
        fetch_object(storage_key, path)
        yield path
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def prepared_artifact(claimed: ClaimedJob, original: Path) -> Iterator[AnalysisArtifact]:
    """Produce the artifact a detector should read, and clean up after the block.

    Media that is already canonical needs nothing: the original on disk is the artifact,
    it is not re-uploaded, and no derivative identity is invented for it.

    Anything else is transcoded here and the result stored in MinIO under its own
    content-addressed key, so the derivative is a real object before any row claims it
    exists. The local copy is removed on every path — a container that ran for a week
    would otherwise fill its disk with every transcode it had ever produced.

    Raises whatever the transcode or the upload raises. Deciding which of those is a
    broken machine and which is media that cannot be prepared is the caller's job.
    """
    if not claimed.normalization_required:
        yield AnalysisArtifact(path=original)
        return

    # The second asynchronous step, and the only reason this function is not a plain
    # call: ffmpeg is driven through asyncio, so it owns an event loop for its duration.
    derivative = asyncio.run(normalize_to_mp4(original, claimed.frame_rate))

    try:
        key = store_derivative(derivative.path, derivative.sha256)
        yield AnalysisArtifact(
            path=derivative.path, storage_key=key, sha256=derivative.sha256
        )
    finally:
        derivative.path.unlink(missing_ok=True)


def run_detection(path: Path):
    """Ask NVIDIA whether the prepared artifact looks synthetic.

    Asynchronous work driven from an otherwise synchronous process, so it owns an event
    loop for the length of that call and nothing else has to be written around one.
    """
    return asyncio.run(detect_synthetic_video(path))


def run_audio_evidence(path: Path, frame_rate: float):
    """Ask both audio questions about the prepared artifact, off one extracted WAV.

    A second event loop rather than one shared with the detection above, because the two
    are independent evidence and are deliberately not made to depend on each other's
    lifetime: one being slow, cancelled or broken must not reach into the other.

    Active speaker and AASIST share this loop because they share the WAV, and only that.
    The whole chain — audio extraction, diarization, NVIDIA, the local checkpoint —
    happens inside this one call, so the temporary WAV never outlives it, and the two
    answers it returns are still separate signals that never inform each other.
    """
    return asyncio.run(analyse_audio(path, frame_rate))


@dataclass(frozen=True)
class SignalEvidence:
    """One signal and the timeline rows that belong to it, before either is persisted.

    A pair rather than two loose variables because the two must not drift apart: segments
    hang off their signal's id, and a list that lost track of which signal produced it
    would be attached to the wrong evidence.
    """

    signal: AnalysisSignal
    segments: list


@dataclass(frozen=True)
class Evidence:
    """Everything asking about one prepared artifact produced, and what preparing it left.

    All three travel together because all three are asked about the same artifact and the
    transcode that produces it either serves them or none of them. They are still wholly
    separate findings — separate rows, separate statuses, separate evidence — and one
    failing says nothing about the others.

    The derivative's identity travels with them because it succeeds or fails with the
    preparation: there is no derivative to record when the transcode is what went wrong,
    and a key recorded without anything having read it would name an unused artifact.
    """

    detection: SignalEvidence
    # None when the face classifier was never invoked, which is exactly the case where the
    # transcode failed: it is the one artifact-dependent reading that is not attempted at
    # all when there is no artifact. The other three are recorded as `FAILED` there because
    # each is a source that was asked about this media and had no answer — NVIDIA takes MP4
    # and nothing else, and the audio is demuxed from the same missing derivative — whereas
    # nothing ever asked this classifier anything.
    #
    # The distinction is the same one `AnalysisTimedOut` draws and for the same reason: a
    # `FAILED` row is a finding ("this source was asked and could not answer"), and writing
    # one for a detector that was never reached would be recording a finding nobody made.
    # Absence is the honest record, and it is not the same fact as a failed reading.
    face_manipulation: SignalEvidence | None
    active_speaker: SignalEvidence
    audio_authenticity: SignalEvidence
    derivative_storage_key: str | None = None
    derivative_sha256: str | None = None

    def entries(self) -> tuple[SignalEvidence, ...]:
        """The artifact-dependent findings that exist, in the order they are written.

        Named rather than unpacked at the call site so the partial path below and the
        complete one persist through the same code: a signal added to this class and not to
        this tuple would be a visible omission, and one written on one path but not the
        other is exactly the drift the two paths must not have.

        The face-manipulation entry is dropped when it is absent rather than being written
        as an empty or failed row — see the field above. Every other entry is required, so
        an omission there is a `TypeError` at construction rather than a silently missing
        signal.
        """
        found = (
            self.detection,
            self.face_manipulation,
            self.active_speaker,
            self.audio_authenticity,
        )

        return tuple(entry for entry in found if entry is not None)


class AnalysisTimedOut(Exception):
    """A controllable external operation outlived its limit, partway through a job.

    Both halves of this class matter and they pull in opposite directions.

    It is an *exception*, because the job it belongs to did not finish and must not be
    published as though it had. Nothing here is classified: a risk level is a statement
    about media, and the operation that would have produced the evidence behind one never
    completed.

    It *carries evidence*, because independent findings the job had already produced are
    not made wrong by a later timeout. A C2PA manifest read off the forensic original and a
    synthetic-video verdict NVIDIA returned are complete, self-contained readings of this
    media (rule 11); the audio extraction that timed out afterwards says nothing about
    either. Discarding them would destroy real forensic findings over a scheduling event —
    the same thing `complete_job` refuses to do when a job is recovered as stale.

    So `produced` holds what was genuinely obtained, and it holds nothing else. The signals
    that were never reached are absent rather than written as `FAILED` rows: a failed signal
    is a finding — "this source was asked and had no answer" — and the sources behind the
    timeout were never asked. Recording them would turn the worker's own limit into evidence
    about the video, which is the one thing this class exists to avoid.
    """

    def __init__(
        self,
        error: BaseException,
        produced: tuple[SignalEvidence, ...] = (),
        derivative_storage_key: str | None = None,
        derivative_sha256: str | None = None,
    ) -> None:
        self.error = error
        self.produced = produced
        # Recorded only when something actually read the derivative. On a transcode that
        # timed out there is no derivative at all; on an audio extraction that did, the
        # detection above it read exactly this artifact, so the key names an object that
        # provably exists and was provably used.
        self.derivative_storage_key = derivative_storage_key
        self.derivative_sha256 = derivative_sha256
        super().__init__(type(error).__name__)


def analyse(claimed: ClaimedJob, original: Path) -> Evidence:
    """Prepare the artifact detection needs, ask every question about it, report the outcomes.

    One preparation serves them all: the transcode is the expensive step and the artifact
    it produces is exactly what each NIM wants and what the analysable audio is extracted
    from, so it is produced once and held for the length of every call rather than
    transcoded again per consumer.

    The audio pair is run second and cannot fail this job. `analyse_audio` records every
    failure of its own chains as a signal, so a missing Hugging Face token, an unreachable
    second NVIDIA function or a missing local checkpoint costs the affected signal and
    nothing else.

    Media that cannot be transcoded is a fact about the media, so it becomes a failed
    signal rather than a failed job — the same treatment a provider that refuses gets, and
    for the same reason: the provenance already read off this file must not be thrown away
    because ffmpeg could not produce an MP4. Three of the four artifact-dependent signals
    record it, because each of those sources was asked about this media and had no answer —
    NVIDIA takes MP4 and nothing else, and the audio the local checkpoint reads is extracted
    from the same derivative that was never produced.

    The face-manipulation reading is the exception, and deliberately: it is never invoked at
    all on this path, so it produces no row rather than a `FAILED` one. Recording a failure
    for a detector nothing ran would be a finding nobody made — the same distinction
    `AnalysisTimedOut` draws for the signals a timeout never reached.

    `NormalizationUnavailable` is deliberately not caught. ffmpeg missing from the image is
    a broken container, not broken media, and recording it as evidence about the video
    would leave a real defect looking like a routine gap. It propagates and fails the job,
    exactly as `NvidiaLocalFileError` does.

    Since R1-T3 a transcode that ran out of time, and audio extraction that did, are
    neither of those. `NormalizationTimeout` and `SpeakerDiarizationTimeout` are not
    subclasses of the error types caught here, so neither becomes a signal: a timeout is a
    statement about this worker — a machine under load, a limit set too tight — and writing
    it down as a finding about the media would attribute the machine's condition to the
    video. Both are re-raised as `AnalysisTimedOut`, which fails the job.

    What that re-raise carries is the point of it. Everything already obtained travels with
    the exception, so a timeout during audio extraction costs the audio signals and not the
    synthetic-video verdict NVIDIA had already returned. Failing the job and keeping its
    evidence are separate decisions here, and this is what lets `process_one` make them
    separately.
    """
    try:
        with prepared_artifact(claimed, original) as artifact:
            signal, segments = run_detection(artifact.path)
            detected = SignalEvidence(signal=signal, segments=segments)

            # Second, and before the audio chain, so that an audio extraction which runs out
            # of time costs neither this reading nor the one above it. It is local, blocking
            # CPU work with no socket and no deadline of its own — `app.limits` says why the
            # bounds there do not cover in-process inference — and it records its own
            # failures, so nothing it does can fail this job.
            face = SignalEvidence(
                signal=detect_face_manipulation(artifact.path), segments=[]
            )

            try:
                # The rate NVIDIA's frame indices have to be read against. It is the rate of
                # the artifact just handed over: a derivative was transcoded to hold exactly
                # this rate constant, and media that needed no derivative is the original,
                # whose probed rate this is.
                (speaker_signal, speaker_segments), (audio_signal, audio_segments) = (
                    run_audio_evidence(artifact.path, claimed.frame_rate)
                )
            except SpeakerDiarizationTimeout as error:
                # The detection above is finished and independent of anything the audio
                # chain would have produced, so it goes with the exception rather than
                # being lost to it. So does the derivative it read, which is why the
                # identity is carried too: the artifact provably exists and provably had a
                # reader.
                raise AnalysisTimedOut(
                    error,
                    produced=(detected, face),
                    derivative_storage_key=artifact.storage_key,
                    derivative_sha256=artifact.sha256,
                ) from error

            return Evidence(
                detection=detected,
                face_manipulation=face,
                active_speaker=SignalEvidence(
                    signal=speaker_signal, segments=speaker_segments
                ),
                audio_authenticity=SignalEvidence(
                    signal=audio_signal, segments=audio_segments
                ),
                derivative_storage_key=artifact.storage_key,
                derivative_sha256=artifact.sha256,
            )
    except NormalizationTimeout as error:
        # Nothing artifact-dependent was reached: the transcode is what timed out, and every
        # signal below it needed the artifact it never produced. `produced` is therefore
        # empty — which is not the same as "no evidence survives this job". The provenance
        # read off the forensic original happened before this call and is `process_one`'s to
        # keep, exactly as it would be on any other failure.
        logger.warning("Preparing the media for detection ran out of time.", exc_info=True)
        raise AnalysisTimedOut(error) from error
    except NormalizationError as error:
        logger.warning("Preparing the media for detection failed.", exc_info=True)
        return Evidence(
            detection=SignalEvidence(
                signal=undetectable_media(error, SYNTHETIC_VIDEO_SIGNAL), segments=[]
            ),
            # Explicitly nothing. The classifier is never invoked on this path — the call
            # that would have run it is inside the `with` block above, past the transcode
            # that raised — so there is no reading to record, failed or otherwise. Stated
            # here rather than left to a default so the omission is deliberate and visible.
            face_manipulation=None,
            active_speaker=SignalEvidence(
                signal=undetectable_media(error, ACTIVE_SPEAKER_SIGNAL), segments=[]
            ),
            audio_authenticity=SignalEvidence(
                signal=unanalysable_audio(error), segments=[]
            ),
        )


def complete_job(
    session: Session,
    claimed: ClaimedJob,
    evidence: Evidence,
    provenance_signal: AnalysisSignal,
) -> RiskDecision | None:
    """Persist everything this job produced, classify it, and close it out.

    Two transactions, in this order and never merged: the forensic evidence is committed
    first, and only then is it read back and classified. `persist_evidence` and
    `conclude_job` each say why.

    Returns the decision that was recorded, for the caller's log, or nothing if the job was
    recovered as stale while it ran. In that case the evidence stays: it is what this worker
    genuinely observed about the media, the signals are independent records in their own
    right (rule 11), and discarding real forensic findings to tidy up after a failed *worker*
    would destroy evidence over a scheduling event. What it may not do is publish a verdict,
    and that is exactly what `conclude_job` refuses.
    """
    persist_evidence(session, claimed, evidence, provenance_signal)

    return conclude_job(session, claimed)


def persist_evidence(
    session: Session,
    claimed: ClaimedJob,
    evidence: Evidence,
    provenance_signal: AnalysisSignal,
) -> None:
    """Write the derivative's identity and every evidence row, in one transaction.

    The derivative columns are written
    here rather than when the object was uploaded because that is the first moment the
    artifact provably exists *and* something has read it — a key committed earlier would
    name an object no analysis had used, and a key committed after would leave a completed
    analysis unable to say what was detected.

    Every signal that exists is written as its own row and none waits on another: they are
    independent evidence, and an analysis that got provenance and a speaker timeline but
    no synthetic-video verdict — or any other combination — records exactly that. An
    analysis carries at most five, and fewer when a source was never reached.
    Provenance and face manipulation are the two that never own segments — the first
    because a signature is not a timeline, the second because R3-T1's contract is one clip
    to one score — while the other three each own their own and never share a row.

    The job stays `processing` across this commit, and deliberately so. Nothing can pick it
    up in the gap — `claim_job` only ever takes `queued` rows — and leaving the status for
    the next transaction is what lets the classification be made from committed evidence
    rather than from values still in flight.

    A detector that failed still finishes the job, and so does media that could not be
    transcoded for one. Neither is a fact about this worker — both are already recorded as
    the signal's own status — and the work this job was queued to do was done.
    """
    write_evidence(
        session,
        claimed,
        (SignalEvidence(signal=provenance_signal, segments=[]), *evidence.entries()),
        derivative_storage_key=evidence.derivative_storage_key,
        derivative_sha256=evidence.derivative_sha256,
    )


def write_evidence(
    session: Session,
    claimed: ClaimedJob,
    persisted: tuple[SignalEvidence, ...],
    derivative_storage_key: str | None = None,
    derivative_sha256: str | None = None,
) -> None:
    """Commit exactly the signals it is given, and the derivative they were read from.

    Split out of `persist_evidence` in R1-T3 so a job that ran out of time writes its
    evidence through the same statements a job that finished does. A second, near-identical
    persistence path for the partial case would be the kind of duplicate that stops matching
    the moment either side gains a column, and the two differ only in *which* signals exist
    — never in how one is written.

    Which is why this takes a tuple rather than an `Evidence`. A partial job has no
    active-speaker or audio-authenticity finding to write, and their absence is the whole
    point: an empty row is not evidence, and a `FAILED` one would be a finding nobody made.
    """
    if derivative_storage_key is not None:
        _record_derivative(session, claimed, derivative_storage_key, derivative_sha256)

    for entry in persisted:
        entry.signal.analysis_id = claimed.analysis_id
        session.add(entry.signal)

    with_segments = [entry for entry in persisted if entry.segments]
    if with_segments:
        # Segments hang off their own signal, so every signal's id has to exist before any
        # of them can name one — a flush inside this same transaction, not a second one.
        session.flush()
        for entry in with_segments:
            for segment in entry.segments:
                segment.signal_id = entry.signal.id
            session.add_all(entry.segments)

    session.commit()


def persisted_svd_evidence(session: Session, analysis_id: uuid.UUID) -> SvdEvidence | None:
    """Read this analysis's synthetic-video signal back out of the database.

    Read back rather than carried over. The values that reach the risk engine are the ones
    a reader of `analysis_signals` will find behind the decision, which is the only way a
    stored classification can be re-derived from stored evidence — an in-memory score that
    differed from the committed row, for any reason, would leave the two disagreeing with
    nothing to say so.

    The lookup names the row; it does not decide eligibility. Provider and signal type are
    what identify *which* signal is the direct-risk one, and the risk engine checks them
    again along with the status and the exact function id, so the calibration binding is
    enforced by the rules rather than assumed from a query.

    `total_clips` comes out of the signal's JSON metadata, where the provider's aggregate
    figures live. It has no column of its own and is absent on every signal that failed.

    Returns nothing when there is no such signal at all. That absence is evidence in its own
    right, and the engine has a rule for it rather than this function having a default.
    """
    row = session.execute(
        select(
            AnalysisSignal.provider,
            AnalysisSignal.signal_type,
            AnalysisSignal.status,
            AnalysisSignal.provider_version,
            AnalysisSignal.score,
            AnalysisSignal.signal_metadata,
        ).where(
            AnalysisSignal.analysis_id == analysis_id,
            AnalysisSignal.provider == NVIDIA_PROVIDER,
            AnalysisSignal.signal_type == SYNTHETIC_VIDEO_SIGNAL,
        )
        # Two direct-risk signals on one analysis is a defect in how evidence was written,
        # not a case to pick a winner from. It raises rather than classifying half of it.
    ).one_or_none()

    if row is None:
        return None

    metadata = row.signal_metadata or {}

    return SvdEvidence(
        provider=row.provider,
        signal_type=row.signal_type,
        status=row.status,
        provider_version=row.provider_version,
        score=row.score,
        total_clips=metadata.get("total_clips"),
    )


def conclude_job(session: Session, claimed: ClaimedJob) -> RiskDecision | None:
    """Classify the analysis from its committed evidence and close the job out.

    Returns nothing when the job stopped being this worker's — see `_set_status`.

    The final analysis step, and it runs after persistence rather than before it for the
    reason `persisted_svd_evidence` gives. The decision and the statuses that publish it go
    in together: an analysis marked `completed` without a decision would show a finished
    analysis nobody classified, and a decision written without the job moving on would leave
    the classification invisible behind a job still in progress.

    Only the direct-risk signal is read. The provenance, face-manipulation, speaker-timeline
    and audio rows committed moments ago are not fetched, not passed in and cannot reach this
    decision — they remain forensic evidence in their own right and none of them is entitled
    to move a band (rule 11, and `app.risk_engine`). `persisted_svd_evidence` names one
    provider and one signal type, so the query cannot see them even by accident.

    Nothing here catches anything. The engine is total over the evidence it is given: a
    missing signal, a failed one, an uncalibrated deployment and unusable figures are each
    classified `UNKNOWN` by an explicit rule rather than raised, so there is no ambiguous
    failure left over to represent. What could still go wrong — a database that has gone
    away, a duplicate signal row, a defect in the rules — is exactly that, a defect, and it
    propagates to `process_one`, which logs it with its traceback and fails the job.
    Swallowing it as `UNKNOWN` would publish a classification nobody made and hide the
    defect behind it; the evidence is already committed and survives the failure either way.
    """
    decision = evaluate(persisted_svd_evidence(session, claimed.analysis_id))

    if not _set_status(
        session,
        claimed,
        JOB_STATUS_COMPLETED,
        ANALYSIS_STATUS_COMPLETED,
        decision=decision,
    ):
        # This job was recovered as stale while it ran, so the analysis is already `failed`
        # and this worker is no longer entitled to publish a verdict on it. Rolled back and
        # reported rather than forced: a recovery that a slow worker could undo would not be
        # a recovery, and the concurrency slot it released would silently be taken back.
        session.rollback()
        return None

    session.commit()

    return decision


def _record_derivative(
    session: Session,
    claimed: ClaimedJob,
    storage_key: str,
    sha256: str | None,
) -> None:
    """Name the artifact this analysis was detected against, now that it exists.

    Only ever called with a derivative this worker created and uploaded. Media that was
    already canonical had its own key written at upload and is not touched here: there is
    no second artifact, and overwriting the column would say there was.

    Takes the two values rather than the `Evidence` they used to come from, because since
    R1-T3 they also arrive on a job that timed out — where detection read the derivative and
    the signals that would have completed the `Evidence` do not exist.
    """
    session.execute(
        MediaFile.__table__.update()
        .where(MediaFile.analysis_id == claimed.analysis_id)
        .values(derivative_storage_key=storage_key, derivative_sha256=sha256)
    )


def abandon_job(
    session: Session,
    claimed: ClaimedJob,
    timed_out: AnalysisTimedOut,
    provenance_signal: AnalysisSignal | None,
) -> None:
    """Keep what a timed-out job proved, then close it out as failed (R1-T3).

    Two commits in a fixed order, and the order is the same one `complete_job` uses for the
    same reason: the evidence is committed first and on its own, so a defect in closing the
    job out costs the decision and not the forensic record.

    What survives is only what was actually obtained — the provenance read off the original,
    and any artifact-dependent signal the analysis got to before the limit was breached.
    Each is a complete, independent reading in its own right (rule 11), and a later timeout
    in an unrelated chain does not make an earlier one less true. Nothing is invented to
    stand in for what was not reached: the sources behind the timeout are absent from
    `analysis_signals`, not written as `FAILED`, because a failed signal means a source was
    asked and had no answer and these were never asked.

    Then the job fails, through the same `fail_job` every other worker-side failure uses.
    The analysis is `failed` and its risk columns stay null — the timeout is not reinterpreted
    as a finding, no rule was run over a partial evidence set, and null is the absence of a
    conclusion rather than `UNKNOWN`, which is a conclusion an explicit rule reached. So a
    reader of this analysis sees the signals that were genuinely produced under a status that
    says plainly the analysis did not finish.

    A job with nothing to keep — a transcode that timed out before any signal existed, and
    whose provenance read had not happened either — skips the first commit rather than
    opening an empty transaction.
    """
    persisted = tuple(
        entry
        for entry in (
            SignalEvidence(signal=provenance_signal, segments=[])
            if provenance_signal is not None
            else None,
            *timed_out.produced,
        )
        if entry is not None
    )

    if persisted or timed_out.derivative_storage_key is not None:
        write_evidence(
            session,
            claimed,
            persisted,
            derivative_storage_key=timed_out.derivative_storage_key,
            derivative_sha256=timed_out.derivative_sha256,
        )

    fail_job(session, claimed, timed_out.error)


def fail_job(session: Session, claimed: ClaimedJob, error: BaseException) -> None:
    """Record that this job could not be done, and why, without stopping the worker.

    Reached only for failures that are ours: the object store unreachable, the fetched
    artifact unreadable, a bug here. A provider that refused to answer is not one of
    them and never lands here.

    The parent analysis fails with the job. An analysis left `queued` behind a job that
    already gave up would look like work still coming, and nothing in this task will ever
    pick it up again.

    A job already recovered as stale is left exactly as recovery wrote it. Both outcomes are
    `failed`, so nothing is at stake but the reason, and overwriting `StaleWorkerLease` with
    whatever this worker tripped over on its way out would replace what actually happened —
    the worker was declared gone — with a symptom of it.
    """
    session.rollback()
    if not _set_status(
        session,
        claimed,
        JOB_STATUS_FAILED,
        ANALYSIS_STATUS_FAILED,
        # The class name, not the message: exception text can quote credentials, storage
        # endpoints or SQL. The traceback is already in the log above.
        error_message=type(error).__name__[:MAX_ERROR_MESSAGE],
    ):
        session.rollback()
        logger.warning(
            "Job %s failed after it had already been recovered as stale; the recorded "
            "failure is left as it was.",
            claimed.job_id,
        )
        return

    session.commit()


def _set_status(
    session: Session,
    claimed: ClaimedJob,
    job_status: str,
    analysis_status: str,
    error_message: str | None = None,
    decision: RiskDecision | None = None,
) -> bool:
    """Move a job and the analysis it belongs to into their end states together.

    Returns whether the job was still this worker's to finish. **Conditional on the row
    still being `processing`**, and that condition is what keeps stale recovery honest: a
    worker that lost its lease, was recovered, and then came back to life must not be able
    to overwrite `failed` with `completed`. Recovery is terminal, so a recovered job is no
    longer `processing`, this update matches nothing, and the caller is told it lost.

    Nothing is written when the condition fails — not the job row, not the analysis. The
    analysis update is inside the same guard rather than after it, or a resurrected worker
    would leave a `failed` job under a `completed` analysis.

    `decision` is the risk classification and its trace, written onto the analysis in the
    same statement that publishes its status. A job that failed passes none: the analysis
    was never classified, and its risk columns stay null — which is not `UNKNOWN`, a
    conclusion an explicit rule reached, but the absence of any conclusion at all.
    """
    updated = session.execute(
        AnalysisJob.__table__.update()
        .where(
            AnalysisJob.id == claimed.job_id,
            AnalysisJob.status == JOB_STATUS_PROCESSING,
        )
        # The lease ends with the job. Leaving a deadline on a terminal row would say a
        # worker was still running something that has already finished.
        .values(status=job_status, error_message=error_message, lease_expires_at=None)
    )

    if updated.rowcount == 0:
        return False

    analysis_values: dict[str, str] = {"status": analysis_status}
    if decision is not None:
        analysis_values |= {
            "risk_level": decision.risk_level,
            "risk_rules_version": decision.rules_version,
            "risk_calibration_id": decision.calibration_id,
            "risk_rule_id": decision.rule_id,
        }

    session.execute(
        Analysis.__table__.update()
        .where(Analysis.id == claimed.analysis_id)
        .values(**analysis_values)
    )

    return True


def process_one(session: Session) -> bool:
    """Claim and run a single job. Returns whether there was one to run.

    Recovery runs first, on the same poll that looks for work (P9-F1). It lives here rather
    than behind a scheduler because this loop is already the thing that runs on a timer, in
    the process that has a reason to care: capacity a crashed worker is holding is capacity
    this worker could be using. It costs one statement that normally matches no rows, on a
    loop that was going to query for a job anyway.

    Running it before the claim, not after, so a stale job's slot is released in time for the
    same pass to pick up whatever that release admits.

    Everything after the claim is guarded: a job this worker took and then failed to
    finish must not be left `processing` forever, because nothing in P3 goes back for it.

    Everything after the claim also runs under the claiming request's id (R1-T4). The
    binding starts at the claim rather than at the top of this function on purpose: recovery
    above belongs to no request — it fails other workers' jobs, several at a time — and
    attributing it to whichever job this pass happened to pick up afterwards would be a
    correlation that is simply untrue.
    """
    recover_stale_jobs(session)

    claimed = claim_job(session)
    if claimed is None:
        return False

    with correlated(claimed):
        logger.info(
            "Claimed job %s for analysis %s.", claimed.job_id, claimed.analysis_id
        )

        # From here to the end of the job, a background thread keeps saying this worker is
        # alive. Without it the lease would run out during any analysis longer than
        # `LEASE_SECONDS` and another worker would recover a job that was going perfectly
        # well.
        with leased(claimed.job_id):
            # Assigned before the block so the timeout handler can still reach it. A
            # transcode that runs out of time unwinds past the assignment below, and the
            # provenance read off the original — which happened before the transcode and is
            # complete — must not be lost to a variable that went out of scope.
            provenance_signal = None

            try:
                # One download serves both evidence sources. Provenance is read from these
                # bytes because they are the forensic original — normalization would strip
                # the manifest — and detection reads either these bytes or the derivative
                # transcoded from them, which is why the original is fetched first and kept
                # for the whole job.
                with fetched_artifact(claimed.original_storage_key) as original:
                    # Reading credentials never raises here: `extract_provenance` returns a
                    # failed signal instead. Nor does media that cannot be transcoded, nor
                    # anything the active-speaker chain can fail with — `analyse` turns all
                    # of those into failed signals. What can still reach the handler below is
                    # the object store, a broken image, a subprocess that outlived its
                    # configured limit (R1-T3), or a bug.
                    provenance_signal = extract_provenance(original)
                    evidence = analyse(claimed, original)
            except AnalysisTimedOut as timed_out:
                # A limit was breached partway through (R1-T3). The job is failed like any
                # other worker-side failure, and everything it had genuinely produced up to
                # that point is committed first: a timeout in one chain does not make an
                # independent reading taken earlier untrue, and throwing away real forensic
                # findings over this worker's own deadline would be destroying evidence to
                # tidy up.
                logger.warning(
                    "Job %s ran out of time in %s; failing it and keeping the evidence it "
                    "had already produced.",
                    claimed.job_id,
                    type(timed_out.error).__name__,
                )
                abandon_job(session, claimed, timed_out, provenance_signal)
                return True
            except Exception as error:
                # Not a detector saying no and not media that could not be prepared — a
                # failure on our own side. It is recorded against the job and the loop
                # carries on.
                logger.exception("Job %s failed.", claimed.job_id)
                fail_job(session, claimed, error)
                return True

            try:
                # Persisting the evidence and classifying it are two commits inside here. A
                # failure in the second leaves the first standing: the job fails with its
                # forensic record intact rather than losing evidence to a classification that
                # could not be made.
                decision = complete_job(session, claimed, evidence, provenance_signal)
            except Exception as error:
                logger.exception("Job %s could not be completed.", claimed.job_id)
                fail_job(session, claimed, error)
                return True

        if decision is None:
            # Recovered as stale while this ran. The analysis is already `failed` and its
            # slot already released; saying so is all that is left to do. The evidence this
            # job produced is committed and stays — see `complete_job`.
            logger.warning(
                "Job %s was recovered as stale while it ran; its verdict is discarded.",
                claimed.job_id,
            )
            return True

        logger.info(
            "Completed job %s with a %s detection signal, a %s face-manipulation signal, a "
            "%s active-speaker signal, a %s audio-authenticity signal and a %s provenance "
            "signal%s. Risk %s by %s under %s.",
            claimed.job_id,
            evidence.detection.signal.status,
            # `no` rather than a status, because absence here is not a status: the classifier
            # is the one reading that is never invoked when the transcode failed, and the
            # `Evidence` field says at length why that is recorded as no row rather than a
            # `FAILED` one. The line enumerates the signals this job wrote, so it has to be
            # able to say that this one was not written — printing `None` would read as a
            # status the database can hold.
            evidence.face_manipulation.signal.status
            if evidence.face_manipulation is not None
            else "no",
            evidence.active_speaker.signal.status,
            evidence.audio_authenticity.signal.status,
            provenance_signal.status,
            " against a normalized derivative" if evidence.derivative_storage_key else "",
            decision.risk_level,
            decision.rule_id,
            decision.rules_version,
        )

        return True


class Stopping:
    """Whether the process has been asked to shut down.

    `docker compose down` sends SIGTERM. Without this the worker would ignore it and be
    killed ten seconds later, mid-job; with it, the loop stops asking for new work and
    exits once the job in hand is finished.
    """

    def __init__(self):
        self.requested = False

    def request(self, *_) -> None:
        logger.info("Shutdown requested; finishing the current job.")
        self.requested = True

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.request)
        signal.signal(signal.SIGINT, self.request)


def wait_for_schema(stopping: Stopping, sleep=time.sleep) -> bool:
    """Block until the database is at this image's Alembic head. Returns whether it got there.

    The gate this loop needs and did not have (R1-T3). A worker started against a database
    that has not been migrated yet does not fail cleanly: it claims a job, gets several
    statements in, trips over a column that does not exist, fails that analysis, and takes
    the next one. A deployment that forgot `alembic upgrade head` therefore showed up as a
    run of failed customer analyses rather than as a deployment that forgot a migration. So
    the question is asked once, here, before anything is claimed.

    It waits rather than exits, which is the same choice `run` makes below and made for the
    same reason: during a rollout the migration and the new worker start within seconds of
    each other, and a worker that exited would be restarted by Compose into a condition that
    is about to resolve itself — a crash loop that reads, in the logs, exactly like a
    deployment that is broken. Waiting says the same thing once every `SCHEMA_POLL_SECONDS`
    and starts working the moment the migration lands, with no restart in between.

    Waiting is not the same as waiting silently. Every attempt logs what it found against
    what it expected, so a schema that is genuinely never going to be migrated is a
    repeating, specific line naming the revisions and the command to fix it, rather than a
    process that appears healthy and does nothing.

    A database that cannot be reached at all is treated the same way and deliberately not
    told apart: both are "not ready yet", both resolve without intervention when the
    dependency comes up, and both are logged with the exception behind them.

    Returns False when shutdown was requested while waiting, so a container being stopped
    mid-rollout exits promptly instead of holding SIGTERM until the migration lands.
    """
    while not stopping.requested:
        try:
            check_schema_ready(engine)
            return True
        except SchemaNotReady as error:
            logger.warning("%s Waiting for the migration.", error)
        except Exception:
            logger.exception(
                "The database schema could not be checked; waiting before trying again."
            )

        sleep(SCHEMA_POLL_SECONDS)

    return False


def run(stopping: Stopping, sleep=time.sleep) -> None:
    """Poll for work until asked to stop.

    A worker whose database has gone away backs off and tries again rather than exiting:
    the container would only be restarted into the same condition, and a crash loop is
    harder to read in the logs than a worker saying the same thing every five seconds.
    """
    while not stopping.requested:
        try:
            with SessionLocal() as session:
                worked = process_one(session)
        except Exception:
            logger.exception("The worker loop failed; retrying.")
            sleep(ERROR_BACKOFF_SECONDS)
            continue

        if not worked:
            sleep(IDLE_POLL_SECONDS)


def main() -> int:
    """Start the worker, after two checks that are cheaper to fail than a job is (R1-T3).

    The order is deliberate. Configuration is resolved first, from the environment alone,
    because a mistyped timeout needs no database to detect and a process that will refuse to
    run should refuse before it waits several minutes for a migration to explain that. A bad
    value exits non-zero: unlike a schema that is about to be migrated, nothing about a typo
    in a compose file resolves by waiting, so this is the one startup condition worth a
    restart loop that an operator will notice.

    The schema comes second and waits instead, for the reasons `wait_for_schema` gives.

    Returns the process's exit status rather than calling `sys.exit`, so the whole startup
    sequence is reachable from a test without a `SystemExit` to catch.
    """
    configure_logging()

    try:
        logger.info("Operation limits: %s.", validate_limits())
    except InvalidTimeout as error:
        logger.error("The worker is misconfigured and will not start: %s", error)
        return 1

    stopping = Stopping()
    stopping.install()

    if not wait_for_schema(stopping):
        logger.info("Shutdown was requested before the schema was ready.")
        return 0

    logger.info("DeepGuard worker started.")
    run(stopping)
    logger.info("DeepGuard worker stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
