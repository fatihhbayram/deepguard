"""The process that runs queued analyses.

Uploads stage the forensic original and record that the rest is owed; nothing in the API
ever transcodes or calls a detector. This is what does, in its own container, on its own
schedule.

Since R10-T2 a job is the **Fast Decision Path** and only that. It runs media acquisition and
the normalization the deciding detectors need, NVIDIA SVD, EfficientNet-B7 and the provenance
read, evaluates `evaluate_v5` over the evidence it committed, publishes the verdict, and is
finished. The evidence-only detectors — LipForensics, Effort, active speaker, AASIST — are no
longer in this path at all: they are Deep Evidence Enrichment, they run afterwards out of
`app.enrichment`, and nothing in this file waits for them.

The reason is latency, and the reason it is *safe* is `r9-v5.0.0`. Those four detectors are
`evidence_only` under the live ruleset: they are counted in no coverage denominator, they can
trip no rule, and their absence, failure or abstention removes no coverage. The job used to
spend minutes on them before an analysis was allowed to reach `completed`, and the verdict at
the end of those minutes was the verdict the first two detectors had already determined.
`docs/architecture/R10_EXECUTION_CONTRACT.md` is the contract this implements, and §3.2 is
specific about the consequence for `conclude_job`: the lip argument is still fetched and still
passed, and it is now normally absent, which under v5 is indistinguishable in outcome from
present.

Provenance stays here (§3.3). It is cheap, it is independent, it is read from the forensic
original rather than from a derivative, and it enters no coverage arithmetic under any ruleset
— so where it runs is a latency decision carrying no semantic weight, and moving it would have
bought nothing while changing the one reader that depends on seeing the original's bytes.

Four transactions per job, never one:

1. claim — take a `queued` job, mark it `processing`, commit, release the lock;
2. nothing — the download, provenance, transcoding, diarization, both NVIDIA inferences
   and the three local checkpoints all happen with no transaction open at all;
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

An idle poll runs the Deep Evidence work, and only an idle one (R10-T2). That ordering is
where the Fast Decision Path's strict resource priority actually lives: this loop asks for a
queued job first, every time, and only a poll that found none goes looking for enrichment. A
queued analysis is therefore never behind an enrichment task, which is what §4.2 requires when
it says Deep Evidence "must never contend for a resource in a way that can starve the Fast
Decision Path". It is the same ordering that has kept shadow mode behind production work since
R6-T1, applied to a second kind of secondary work — and enrichment goes first of the two,
because supplementary forensic evidence a customer will read outranks an experiment nobody
can.

An idle poll also runs one shadow experiment, and only an idle one (R6-T1). Uncalibrated
workloads are exercised on real traffic from this same loop, behind every queued job and
writing to a table of its own — see `app.shadow`, which says why none of what they observe
can reach the report, the public API or the risk engine. A job completed above queues one on
its way out; the analysis is already decided and closed by then and does not wait for it.

Since R6-T2 that experiment may run on a rented GPU, and the loop no longer waits for one to
finish either. `shadow.process_one` starts a remote workload on one poll and collects it on a
later one, so the step this loop takes is always short — an experiment already in flight
cannot delay the claiming of a job submitted while it runs, which a blocking wait was measured
doing before the split.

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
    EFFICIENTNET_B7_PROVIDER,
    FACE_MANIPULATION_SIGNAL,
    LIP_FORENSICS_SIGNAL,
    LIPFORENSICS_PROVIDER,
    NVIDIA_PROVIDER,
    SYNTHETIC_VIDEO_SIGNAL,
    detect_face_manipulation,
    detect_synthetic_video,
    extract_provenance,
    undetectable_media,
)
from app.normalization import NormalizationError, NormalizationTimeout, normalize_to_mp4
from app import enrichment, shadow
from app.observability import bind_request_id, configure_logging, reset_request_id
from app.risk_engine import (
    FaceEvidence,
    LipEvidence,
    RiskDecision,
    SvdEvidence,
    evaluate_v5,
)
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
    # Which product mode this submission asked for, or null for one that asked for none
    # (R10-T3). Carried out of the claim with everything else, and read exactly once — at
    # the very end of `process_one`, after the verdict has been published and the job
    # closed, to decide whether this analysis's Deep Evidence is queued or deferred.
    #
    # Nothing between the claim and the verdict looks at it, and that is the guarantee
    # rather than a description: a mode that reached a detector, a threshold or the engine
    # would be a product choice with forensic consequences, which §4.2 does not allow it to
    # be.
    enrichment_mode: str | None = None


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
    # The picture size of this artifact, measured off it when it was transcoded. Null on the
    # bypass path, where the artifact is the original and its analysed dimensions were
    # already written at upload from the same probe that admitted it — there is no second
    # artifact to measure and no second measurement to record.
    width: int | None = None
    height: int | None = None


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
        enrichment_mode=job.enrichment_mode,
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
            path=derivative.path,
            storage_key=key,
            sha256=derivative.sha256,
            # Measured by the transcode, off the file it just wrote. This is the geometry the
            # detectors below are about to see, and it is carried rather than re-derived
            # because the original's columns cannot reproduce it: ffmpeg has applied the
            # display matrix and the even-dimension pad by this point.
            width=derivative.width,
            height=derivative.height,
        )
    finally:
        derivative.path.unlink(missing_ok=True)


def run_detection(path: Path):
    """Ask NVIDIA whether the prepared artifact looks synthetic.

    Asynchronous work driven from an otherwise synchronous process, so it owns an event
    loop for the length of that call and nothing else has to be written around one.
    """
    return asyncio.run(detect_synthetic_video(path))


@dataclass(frozen=True)
class SignalEvidence:
    """One signal and the timeline rows that belong to it, before either is persisted.

    A pair rather than two loose variables because the two must not drift apart: segments
    hang off their signal's id, and a list that lost track of which signal produced it
    would be attached to the wrong evidence.
    """

    signal: AnalysisSignal
    segments: list


def local_readings(path: Path) -> tuple[SignalEvidence, ...]:
    """Ask the local checkpoint that decides about the prepared artifact.

    One reading since R10-T2, where this stopped being "every local checkpoint" and became
    "the local checkpoint the verdict needs". EfficientNet-B7 is `decision_eligible` under
    `r9-v5.0.0`; Effort and LipForensics are `evidence_only`, so they moved to Deep Evidence
    Enrichment and `app.enrichment` is what runs them now. What they cost was the point: the
    B7 costs seconds, Effort costs about forty of them and LipForensics costs minutes, and the
    verdict used to wait for all three to learn something only the first could tell it.

    Still a tuple, and still spread into `Evidence.entries`, which is the one thing here worth
    defending rather than collapsing. The shape is not an abstraction over interchangeable
    detectors — R5-T2 made that point and it still holds, and nothing dispatches or registers
    anything here — it is what lets this function return *nothing* on the path where the
    transcode failed and the checkpoint was never invoked. That distinction is forensic: a
    `FAILED` row means a source was asked and had no answer, and a detector nothing ran has
    made no finding to record. A single field would have had to carry a `None` that every
    reader downstream then had to interpret.

    It cannot fail this job. `detect_face_manipulation` records its own failures as a `FAILED`
    signal — see `app.detection` — so a missing checkpoint or an unreadable clip costs this
    reading and nothing else, and the risk engine has an explicit rule for the absence.
    """
    return (SignalEvidence(signal=detect_face_manipulation(path), segments=[]),)


@dataclass(frozen=True)
class Evidence:
    """Everything the Fast Decision Path produced about one prepared artifact.

    Narrowed in R10-T2 to what the automated verdict needs: NVIDIA's synthetic-video reading
    and EfficientNet-B7's face-manipulation reading, plus the identity of the derivative both
    were read from. The active-speaker and audio-authenticity fields that used to sit here are
    gone from this path — they are Deep Evidence, `app.enrichment` owns them, and their rows
    are written against the same analysis after this job has closed.

    They travel together because they are asked about the same artifact and the transcode that
    produces it either serves both or neither. They are still wholly separate findings —
    separate rows, separate statuses, separate evidence — and one failing says nothing about
    the other.

    The derivative's identity travels with them because it succeeds or fails with the
    preparation: there is no derivative to record when the transcode is what went wrong, and a
    key recorded without anything having read it would name an unused artifact. It is also what
    Deep Evidence later reads, which is a second reason it is committed by this path rather
    than invented by that one — enrichment reads the artifact the verdict was taken from, never
    a fresh transcode of its own.
    """

    detection: SignalEvidence
    # What the deciding local checkpoint made of the artifact. A tuple rather than one field
    # since R5-T2, and kept as one through R10-T2 even though only EfficientNet-B7 remains in
    # it, because the reason for the shape was never plurality — see `local_readings`. It is
    # empty exactly when the transcode failed and the checkpoint was never invoked, and an
    # empty tuple contributes no row rather than a `FAILED` one.
    #
    # The distinction is the same one `AnalysisTimedOut` draws and for the same reason: a
    # `FAILED` row is a finding ("this source was asked and could not answer"), and writing one
    # for a detector that was never reached would be recording a finding nobody made. Absence
    # is the honest record, and it is not the same fact as a failed reading.
    local_readings: tuple[SignalEvidence, ...]
    derivative_storage_key: str | None = None
    derivative_sha256: str | None = None
    # The geometry of the derivative the findings above were read from, as the transcode
    # measured it. Travels with the key for the same reason the hash does: it is a fact about
    # that artifact, it only exists when that artifact does, and recording it apart from the
    # key it belongs to would let the two describe different files.
    analyzed_width: int | None = None
    analyzed_height: int | None = None

    def entries(self) -> tuple[SignalEvidence, ...]:
        """The artifact-dependent findings that exist, in the order they are written.

        Named rather than unpacked at the call site so the partial path below and the complete
        one persist through the same code: a signal added to this class and not to this tuple
        would be a visible omission, and one written on one path but not the other is exactly
        the drift the two paths must not have.

        The local readings are spread in place, so an empty tuple contributes nothing rather
        than an empty or failed row — see the field above.
        """
        return (self.detection, *self.local_readings)


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
        analyzed_width: int | None = None,
        analyzed_height: int | None = None,
    ) -> None:
        self.error = error
        self.produced = produced
        # Recorded only when something actually read the derivative. On a transcode that
        # timed out there is no derivative at all; on an audio extraction that did, the
        # detection above it read exactly this artifact, so the key names an object that
        # provably exists and was provably used.
        self.derivative_storage_key = derivative_storage_key
        self.derivative_sha256 = derivative_sha256
        # Measured off that same derivative, and kept for the same reason the evidence above
        # is: a timeout in the audio chain does not make the geometry the detector already
        # read any less true.
        self.analyzed_width = analyzed_width
        self.analyzed_height = analyzed_height
        super().__init__(type(error).__name__)


def analyse(claimed: ClaimedJob, original: Path) -> Evidence:
    """Prepare the artifact the decision needs, ask the deciding detectors, report the outcomes.

    One preparation serves both: the transcode is the expensive step and the artifact it
    produces is exactly what the NIM wants and what EfficientNet-B7 reads, so it is produced
    once and held for the length of both calls rather than transcoded again per consumer.

    Since R10-T2 this is the Fast Decision Path and nothing else. The audio chain that used to
    run last here — active speaker and AASIST, off one extracted WAV — is Deep Evidence and
    runs out of `app.enrichment` against the derivative this function uploaded. So is
    LipForensics, and so is Effort. Removing them from here is the whole of R10's latency
    claim, and it is safe because none of the four is read by `r9-v5.0.0`: §3.2 of the
    execution contract works through what that means for the lip argument `conclude_job` still
    passes.

    Media that cannot be transcoded is a fact about the media, so it becomes a failed signal
    rather than a failed job — the same treatment a provider that refuses gets, and for the
    same reason: the provenance already read off this file must not be thrown away because
    ffmpeg could not produce an MP4. The synthetic-video signal records it, because NVIDIA was
    asked about this media and had no answer — it takes MP4 and nothing else.

    The local visual reading is the exception, and deliberately: it is never invoked at all on
    that path, so it produces no row rather than a `FAILED` one. Recording a failure for a
    detector nothing ran would be a finding nobody made — the same distinction
    `AnalysisTimedOut` draws for the signals a timeout never reached.

    `NormalizationUnavailable` is deliberately not caught. ffmpeg missing from the image is a
    broken container, not broken media, and recording it as evidence about the video would
    leave a real defect looking like a routine gap. It propagates and fails the job, exactly as
    `NvidiaLocalFileError` does.

    Since R1-T3 a transcode that ran out of time is neither of those. `NormalizationTimeout` is
    not a subclass of the error type caught here, so it never becomes a signal: a timeout is a
    statement about this worker — a machine under load, a limit set too tight — and writing it
    down as a finding about the media would attribute the machine's condition to the video. It
    is re-raised as `AnalysisTimedOut`, which fails the job.

    The audio-extraction timeout that used to be re-raised here went with the audio chain. A
    Deep Evidence component that runs out of time now fails that component's task and reaches
    nothing on this path at all, which is §7.4 — Deep Evidence failures never destroy Fast
    Decisions — holding at the one place it used to be untrue.
    """
    try:
        with prepared_artifact(claimed, original) as artifact:
            signal, segments = run_detection(artifact.path)
            detected = SignalEvidence(signal=signal, segments=segments)

            # Second, and after NVIDIA, so the cheap local reading is taken against the same
            # artifact the remote one saw. It is blocking CPU inference with no socket and no
            # deadline of its own — `app.limits` says why the bounds there do not cover
            # in-process inference — and it records its own failures, so nothing it does can
            # fail this job.
            local = local_readings(artifact.path)

            return Evidence(
                detection=detected,
                local_readings=local,
                derivative_storage_key=artifact.storage_key,
                derivative_sha256=artifact.sha256,
                analyzed_width=artifact.width,
                analyzed_height=artifact.height,
            )
    except NormalizationTimeout as error:
        # Nothing artifact-dependent was reached: the transcode is what timed out, and every
        # signal below it needed the artifact it never produced. `produced` is therefore empty
        # — which is not the same as "no evidence survives this job". The provenance read off
        # the forensic original happened before this call and is `process_one`'s to keep,
        # exactly as it would be on any other failure.
        logger.warning("Preparing the media for detection ran out of time.", exc_info=True)
        raise AnalysisTimedOut(error) from error
    except NormalizationError as error:
        logger.warning("Preparing the media for detection failed.", exc_info=True)
        return Evidence(
            detection=SignalEvidence(
                signal=undetectable_media(error, SYNTHETIC_VIDEO_SIGNAL), segments=[]
            ),
            # Explicitly nothing. The local checkpoint is not invoked on this path — the call
            # that would have run it is inside the `with` block above, past the transcode that
            # raised — so there is no reading to record, failed or otherwise. Stated here
            # rather than left to a default so the omission is deliberate and visible.
            local_readings=(),
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
    independent evidence, and an analysis that got provenance but no synthetic-video verdict —
    or any other combination — records exactly that. This path writes at most three of them
    since R10-T2: provenance, the synthetic-video verdict and the face-manipulation reading.
    The other rows an analysis carries are Deep Evidence and arrive later, written against this
    same analysis by `app.enrichment` through a session that cannot reach these three.

    Provenance and the local visual reading are the two that never own segments — the first
    because a signature is not a timeline, the second because R3-T1's contract is one clip to
    one score — while the synthetic-video row owns its own and shares them with nothing.

    **The order across the two paths is the invariant, not the transaction.** §7.5 requires
    that a verdict never become visible without its supporting evidence already being durable,
    and that is satisfied here by this commit landing before `conclude_job` publishes — the
    order the worker has always used, confirmed rather than assumed when R10-T2 inspected it.
    What is forbidden is the reverse, and nothing on the enrichment path can produce it: that
    path publishes no verdict.

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
        analyzed_width=evidence.analyzed_width,
        analyzed_height=evidence.analyzed_height,
    )


def write_evidence(
    session: Session,
    claimed: ClaimedJob,
    persisted: tuple[SignalEvidence, ...],
    derivative_storage_key: str | None = None,
    derivative_sha256: str | None = None,
    analyzed_width: int | None = None,
    analyzed_height: int | None = None,
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
        _record_derivative(
            session,
            claimed,
            derivative_storage_key,
            derivative_sha256,
            analyzed_width,
            analyzed_height,
        )

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


def _persisted_signal(
    session: Session, analysis_id: uuid.UUID, provider: str, signal_type: str
):
    """Read one of this analysis's detector signals back out of the database.

    Read back rather than carried over. The values that reach the risk engine are the ones a
    reader of `analysis_signals` will find behind the decision, which is the only way a stored
    classification can be re-derived from stored evidence — an in-memory score that differed
    from the committed row, for any reason, would leave the two disagreeing with nothing to
    say so.

    The lookup names the row; it does not decide eligibility. Provider and signal type are
    what identify *which* signal this is, and the risk engine checks them again along with the
    status and the exact deployment identifier, so each calibration binding is enforced by the
    rules rather than assumed from a query.

    Returns nothing when there is no such signal at all. That absence is evidence in its own
    right, and the engine has a rule for it rather than this function having a default.
    """
    return session.execute(
        select(
            AnalysisSignal.provider,
            AnalysisSignal.signal_type,
            AnalysisSignal.status,
            AnalysisSignal.provider_version,
            AnalysisSignal.score,
            AnalysisSignal.signal_metadata,
        ).where(
            AnalysisSignal.analysis_id == analysis_id,
            AnalysisSignal.provider == provider,
            AnalysisSignal.signal_type == signal_type,
        )
        # Two rows for one provider and signal type on one analysis is a defect in how
        # evidence was written, not a case to pick a winner from. It raises rather than
        # classifying half of it.
    ).one_or_none()


def persisted_svd_evidence(session: Session, analysis_id: uuid.UUID) -> SvdEvidence | None:
    """This analysis's NVIDIA synthetic-video signal, shaped for the rules.

    `total_clips` comes out of the signal's JSON metadata, where the provider's aggregate
    figures live. It has no column of its own and is absent on every signal that failed.
    """
    row = _persisted_signal(
        session, analysis_id, NVIDIA_PROVIDER, SYNTHETIC_VIDEO_SIGNAL
    )

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


def persisted_face_evidence(session: Session, analysis_id: uuid.UUID) -> FaceEvidence | None:
    """This analysis's EfficientNet-B7 face-manipulation signal, shaped for the rules.

    `frames_scored` comes out of the signal's JSON metadata, where `app.detection` records how
    many face crops the stored mean was actually taken over. Absent on every signal that
    failed, including the abstention this detector reports for a clip with no face in it —
    which the engine treats as no reading rather than as a finding of no manipulation.
    """
    row = _persisted_signal(
        session, analysis_id, EFFICIENTNET_B7_PROVIDER, FACE_MANIPULATION_SIGNAL
    )

    if row is None:
        return None

    metadata = row.signal_metadata or {}

    return FaceEvidence(
        provider=row.provider,
        signal_type=row.signal_type,
        status=row.status,
        provider_version=row.provider_version,
        score=row.score,
        frames_scored=metadata.get("frames_scored"),
    )


def persisted_lip_evidence(session: Session, analysis_id: uuid.UUID) -> LipEvidence | None:
    """This analysis's LipForensics mouth-dynamics signal, shaped for the rules.

    `windows_scored` comes out of the signal's JSON metadata, where `app.detection` records how
    many 25-frame runs actually held a trackable face and therefore how many logits the stored
    mean was taken over. Absent on every signal that failed, including the abstention this
    detector reports for a clip in which no run held a face — which the engine treats as no
    reading rather than as a finding of no manipulation.
    """
    row = _persisted_signal(
        session, analysis_id, LIPFORENSICS_PROVIDER, LIP_FORENSICS_SIGNAL
    )

    if row is None:
        return None

    metadata = row.signal_metadata or {}

    return LipEvidence(
        provider=row.provider,
        signal_type=row.signal_type,
        status=row.status,
        provider_version=row.provider_version,
        score=row.score,
        windows_scored=metadata.get("windows_scored"),
    )


def conclude_job(session: Session, claimed: ClaimedJob) -> RiskDecision | None:
    """Classify the analysis from its committed evidence and close the job out.

    Returns nothing when the job stopped being this worker's — see `_set_status`.

    The final analysis step, and it runs after persistence rather than before it for the
    reason the evidence readers above give. The decision and the statuses that publish it go
    in together: an analysis marked `completed` without a decision would show a finished
    analysis nobody classified, and a decision written without the job moving on would leave
    the classification invisible behind a job still in progress.

    Three signals are read, and only three: NVIDIA's synthetic-video score, EfficientNet-B7's
    face-manipulation score and LipForensics' mouth-dynamics score, each of which has an
    operating point measured for it alone — the first two in R4-T1, the third in R5-T3. The
    provenance, speaker-timeline and audio rows committed moments ago are not fetched, not
    passed in and cannot reach this decision — they remain forensic evidence in their own right
    and none of them is entitled to move a band (rule 11, and `app.risk_engine`). Each reader
    above names one provider and one signal type, so the queries cannot see them even by
    accident.

    **The ruleset this path decides under is `r9-v5.0.0` as of R9-T8B**, and the cutover is
    exactly here: an analysis is decided once, when it reaches this function, under whichever
    ruleset this deployment calls. Every analysis already holding a verdict was decided under
    the ruleset that was live when *it* passed through, and nothing in this deployment goes
    back for it — there is no backfill, no re-evaluation and no mapping of a stored `HIGH`,
    `MEDIUM` or `UNKNOWN` onto the R9 vocabulary (R9-T1 invariant 4). A reader resolves a
    stored decision through its own `risk_rules_version`, which is why an old report still
    reads under the rules that produced it (`app.risk_trace`).

    The mouth-dynamics row is the one that has moved three times. Under r4-v2.0.0 it was
    fetched by nothing; r5-v3.0.0 read it against R5-T3's operating point and could take a
    HIGH from it alone; r7-v4.0.0 fetched it still, and read it still, but took no HIGH from
    it — R7-T5 measured a 7.17% false HIGH rate on independent genuine media, 21 of the 22
    from that rule, and it was withdrawn; r9-v5.0.0 makes it `evidence_only` outright and
    does not read it at all — outside the coverage arithmetic rather than a zero inside it
    (`app.risk_engine`). It is fetched and handed over here regardless, on the same terms as
    the other two and for the same reason: which of the three may conclude anything, and
    which may be read at all, is the engine's business and not this function's. A worker that
    started skipping it because the current ruleset ignores it would be holding a rule of its
    own.

    The three are fetched separately and handed over separately. Nothing here compares, combines
    or reconciles them: the engine holds the rules, and this function's whole responsibility is
    that the evidence it passes is the evidence the database holds.

    Nothing here catches anything. The engine is total over the evidence it is given: a
    missing signal, a failed one, an uncalibrated deployment and unusable figures each land on
    an explicit rule rather than being raised — `INCONCLUSIVE` under v5, `UNKNOWN` under the
    versions before it — so there is no ambiguous failure left over to represent. What could still go wrong — a database that has gone
    away, a duplicate signal row, a defect in the rules — is exactly that, a defect, and it
    propagates to `process_one`, which logs it with its traceback and fails the job.
    Swallowing it as `UNKNOWN` would publish a classification nobody made and hide the
    defect behind it; the evidence is already committed and survives the failure either way.
    """
    decision = evaluate_v5(
        persisted_svd_evidence(session, claimed.analysis_id),
        persisted_face_evidence(session, claimed.analysis_id),
        persisted_lip_evidence(session, claimed.analysis_id),
    )

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

    # The Deep Evidence execution record, in the transaction that publishes the verdict
    # (R10 follow-up). Not after it, which is where R10-T2 put it: the enrichment axis is
    # projected from these rows and their absence means `LEGACY_SINGLE_STAGE` — an analysis
    # that ran before the two-stage architecture existed. Written in a later transaction,
    # they left a gap in which a freshly decided analysis was `DECIDED` with no execution
    # record, which is precisely the shape of a 2025 row, and any reader landing in that gap
    # would have called a new analysis historical. Here there is no gap: an observer sees
    # the verdict and the rows together or sees neither.
    #
    # Below the guard, deliberately. A worker whose claim was recovered while it ran has
    # already returned above, so a job that is no longer entitled to publish stages nothing
    # and cannot collide with the rows the winning worker wrote.
    #
    # The mode is read here and nowhere earlier in this file. It chooses the status these
    # rows are written in — queued, or deferred as `not_requested` — and reaches no
    # detector, no threshold and no rule on the way (§4.2).
    staged = enrichment.stage(
        session,
        claimed.analysis_id,
        decision.rules_version,
        claimed.enrichment_mode,
    )

    if staged == 0:
        # The only path to an analysis that is decided under R10 and still projects as
        # legacy. Unreachable in this build — the ruleset the engine decides under declares
        # its membership, and only one component is switchable off — so it is logged rather
        # than guarded against: a deployment that reaches it has a configuration problem
        # this line names.
        logger.warning(
            "Analysis %s was decided with no Deep Evidence components staged; its "
            "enrichment axis will read as legacy.",
            claimed.analysis_id,
        )

    session.commit()

    return decision


def _record_derivative(
    session: Session,
    claimed: ClaimedJob,
    storage_key: str,
    sha256: str | None,
    analyzed_width: int | None = None,
    analyzed_height: int | None = None,
) -> None:
    """Name the artifact this analysis was detected against, and record its shape.

    Only ever called with a derivative this worker created and uploaded. Media that was
    already canonical had its own key written at upload and is not touched here: there is
    no second artifact, and overwriting the column would say there was.

    Takes the values rather than the `Evidence` they used to come from, because since
    R1-T3 they also arrive on a job that timed out — where detection read the derivative and
    the signals that would have completed the `Evidence` do not exist.

    The analysed dimensions are written here, in the same statement as the key, because they
    describe that exact object: they were measured off the derivative by the transcode that
    produced it, and a row naming a derivative whose geometry came from somewhere else would
    be the very confusion this task exists to remove. They are null when the probe failed,
    and null is written rather than skipped for the same reason — the alternative is leaving
    an earlier value standing beside a newer artifact.

    Media that was already canonical is not touched here at all. It has no derivative, its
    key was written at upload, and its analysed dimensions were written there too from the
    probe that admitted it.
    """
    session.execute(
        MediaFile.__table__.update()
        .where(MediaFile.analysis_id == claimed.analysis_id)
        .values(
            derivative_storage_key=storage_key,
            derivative_sha256=sha256,
            analyzed_width=analyzed_width,
            analyzed_height=analyzed_height,
        )
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
            analyzed_width=timed_out.analyzed_width,
            analyzed_height=timed_out.analyzed_height,
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

        # The verdict is published and the decision job is closed, so the analysis is now
        # decision-complete: the report can be opened and nothing below may change what it
        # says. Only at this point is anything else allowed to be scheduled against it.
        #
        # Deep Evidence is already scheduled. Its rows were written by `conclude_job`, in the
        # same transaction as the verdict, because the absence of those rows is what
        # `LEGACY_SINGLE_STAGE` means and an analysis must never be decided and momentarily
        # indistinguishable from a pre-R10 one. Membership still comes from the ruleset the
        # analysis was *decided* under, which is why it could not have been done at upload.
        #
        # The experiment is different and stays here, in a table of its own (R6-T1). Nothing
        # projects an analysis's state from a `ShadowRun`, so there is no reading of the
        # analysis that its absence can corrupt — and `shadow.enqueue` does not raise, because
        # a customer's analysis must not fail because an experiment could not be queued.
        shadow.enqueue(session, claimed.analysis_id)

        logger.info(
            "Completed job %s with a %s detection signal, %s and a %s provenance signal%s. "
            "Risk %s by %s under %s.",
            claimed.job_id,
            evidence.detection.signal.status,
            # The local reading named with the signal it wrote, and `no local readings` when
            # there is none — because absence here is not a status. It is the reading never
            # invoked when the transcode failed, and the `Evidence` field says at length why
            # that is recorded as no row rather than a `FAILED` one. This line enumerates what
            # the job wrote, so it has to be able to say it was not written; printing `None`
            # would read as a status the database can hold.
            ", ".join(
                f"a {entry.signal.status} {entry.signal.signal_type} signal"
                for entry in evidence.local_readings
            )
            or "no local readings",
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

    Since R6-T1 an idle poll also looks for shadow work, and only an idle one. Experiments
    run in this process because it is the process that already polls PostgreSQL on a timer,
    and they run strictly behind production work because nothing about an experiment is
    allowed to be in front of a customer's analysis.
    """
    while not stopping.requested:
        try:
            with SessionLocal() as session:
                worked = process_one(session)

                if not worked:
                    # Deep Evidence only on a poll that found no decision job (R10-T2). That
                    # ordering is the Fast Decision Path's strict resource priority, and it is
                    # structural rather than a scheduling hint: a queued job is claimed on
                    # every pass before this line is reached, so enrichment cannot be holding
                    # the worker when one arrives to be claimed — and an analysis submitted
                    # while a component is running is picked up by the next poll after it,
                    # never queued behind the rest of the enrichment backlog.
                    #
                    # `enrichment.process_one` swallows its own failures, so a broken component
                    # cannot reach the backoff below, cannot stop the next job being claimed,
                    # and cannot turn an idle poll into an error. §7.4 requires exactly that:
                    # Deep Evidence failures never destroy Fast Decisions, and a loop this one
                    # crashed out of would have destroyed the next one.
                    worked = enrichment.process_one(session)

                if not worked:
                    # Production work first, always, and shadow work only on a poll that found
                    # none (R6-T1). That ordering is what makes shadow mode structurally
                    # incapable of delaying an analysis: an experiment is never claimed while a
                    # queued job exists. It now sits behind enrichment as well as behind the
                    # decision, because supplementary evidence a customer will read outranks an
                    # experiment no customer can. `shadow.process_one` swallows its own
                    # failures, so a broken experiment cannot reach the backoff below, cannot
                    # stop the next job being claimed, and cannot turn an idle poll into an
                    # error.
                    worked = shadow.process_one(session)
        except Exception:
            logger.exception("The worker loop failed; retrying.")
            sleep(ERROR_BACKOFF_SECONDS)
            continue

        if not worked:
            sleep(IDLE_POLL_SECONDS)

    # On the way out, let go of any remote shadow call this worker started (R6-T2). The row
    # is left to its lease, exactly as it would be if this process had died — what this saves
    # is a GPU container still running for an answer nobody is going to collect.
    shadow.abandon_pending()


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
