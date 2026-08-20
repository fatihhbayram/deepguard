"""The process that runs queued analyses.

Uploads stage media and record that detection is owed; nothing in the API ever calls a
detector. This is what does, in its own container, on its own schedule.

Three transactions per job, never one:

1. claim — take a `queued` job, mark it `processing`, commit, release the lock;
2. nothing — download and inference happen with no transaction open at all;
3. finish — write the evidence and close the job out.

The middle step is the reason for the other two. NVIDIA can take minutes on one video,
and a transaction held open across that would pin a connection and a row lock for the
whole wait. Claiming is therefore deliberately separate from finishing, and the row a
worker holds between them is marked `processing` rather than locked.

Run it with `python -m app.worker`.
"""

import asyncio
import logging
import os
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
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
    MediaFile,
)
from app.db.session import SessionLocal
from app.detection import detect_synthetic_video
from app.storage import fetch_object

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


@dataclass(frozen=True)
class ClaimedJob:
    """A job this worker owns, as plain values rather than rows.

    Everything the work needs is read inside the claim transaction and carried out of it
    by value. After the commit there is no session and no lock, so an ORM row here would
    be an invitation to touch the database during inference — the one thing the claim is
    structured to avoid.
    """

    job_id: uuid.UUID
    analysis_id: uuid.UUID
    # The provider-compatible object to detect against. When the upload normalized the
    # media this is the derivative; when the original was already canonical, P1 stored
    # the original's own key in this column, so it is the right object either way and
    # needs no branch here.
    storage_key: str


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
    """
    row = session.execute(
        select(AnalysisJob, MediaFile.derivative_storage_key)
        .join(MediaFile, MediaFile.analysis_id == AnalysisJob.analysis_id)
        .where(AnalysisJob.status == JOB_STATUS_QUEUED)
        .order_by(AnalysisJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True, of=AnalysisJob)
    ).first()

    if row is None:
        session.rollback()
        return None

    job, storage_key = row
    job.status = JOB_STATUS_PROCESSING
    claimed = ClaimedJob(
        job_id=job.id, analysis_id=job.analysis_id, storage_key=storage_key
    )
    session.commit()

    return claimed


def run_detection(claimed: ClaimedJob):
    """Fetch the media, detect against it, and leave nothing on disk either way.

    The artifact is downloaded rather than passed along, because the process that stored
    it was a different one on a different machine's filesystem. The object store is the
    only thing the upload and this worker share.
    """
    handle = tempfile.NamedTemporaryFile(prefix=TEMP_FILE_PREFIX, suffix=".mp4", delete=False)
    handle.close()
    path = Path(handle.name)

    try:
        fetch_object(claimed.storage_key, path)

        # The one asynchronous step in an otherwise synchronous process. Detection is all
        # this worker waits on, so it owns an event loop for the length of that call and
        # nothing else has to be written around one.
        return asyncio.run(detect_synthetic_video(path))
    finally:
        path.unlink(missing_ok=True)


def complete_job(session: Session, claimed: ClaimedJob, signal, segments) -> None:
    """Write the evidence and close the job out, in one transaction.

    The evidence and the statuses that claim it exists go in together. A signal committed
    without its job moving on would be detected twice by the next worker; a job marked
    `completed` without its signal would report evidence that is not there.

    A detector that failed still finishes the job. The provider not answering is a fact
    about the provider, already recorded as the signal's own status — the work this job
    was queued to do was done, and calling that a job failure would confuse a broken
    detector with a broken worker.
    """
    signal.analysis_id = claimed.analysis_id
    session.add(signal)

    if segments:
        # The segments hang off the signal, so its id has to exist before they can name
        # it — a second flush inside the same transaction, not a second transaction.
        session.flush()
        for segment in segments:
            segment.signal_id = signal.id
        session.add_all(segments)

    _set_status(session, claimed, JOB_STATUS_COMPLETED, ANALYSIS_STATUS_COMPLETED)
    session.commit()


def fail_job(session: Session, claimed: ClaimedJob, error: BaseException) -> None:
    """Record that this job could not be done, and why, without stopping the worker.

    Reached only for failures that are ours: the object store unreachable, the fetched
    artifact unreadable, a bug here. A provider that refused to answer is not one of
    them and never lands here.

    The parent analysis fails with the job. An analysis left `queued` behind a job that
    already gave up would look like work still coming, and nothing in this task will ever
    pick it up again.
    """
    session.rollback()
    _set_status(
        session,
        claimed,
        JOB_STATUS_FAILED,
        ANALYSIS_STATUS_FAILED,
        # The class name, not the message: exception text can quote credentials, storage
        # endpoints or SQL. The traceback is already in the log above.
        error_message=type(error).__name__[:MAX_ERROR_MESSAGE],
    )
    session.commit()


def _set_status(
    session: Session,
    claimed: ClaimedJob,
    job_status: str,
    analysis_status: str,
    error_message: str | None = None,
) -> None:
    """Move a job and the analysis it belongs to into their end states together."""
    session.execute(
        AnalysisJob.__table__.update()
        .where(AnalysisJob.id == claimed.job_id)
        .values(status=job_status, error_message=error_message)
    )
    session.execute(
        Analysis.__table__.update()
        .where(Analysis.id == claimed.analysis_id)
        .values(status=analysis_status)
    )


def process_one(session: Session) -> bool:
    """Claim and run a single job. Returns whether there was one to run.

    Everything after the claim is guarded: a job this worker took and then failed to
    finish must not be left `processing` forever, because nothing in P3 goes back for it.
    """
    claimed = claim_job(session)
    if claimed is None:
        return False

    logger.info("Claimed job %s for analysis %s.", claimed.job_id, claimed.analysis_id)

    try:
        signal, segments = run_detection(claimed)
    except Exception as error:
        # Not a detector saying no — a failure on our own side. It is recorded against
        # the job and the loop carries on to the next one.
        logger.exception("Job %s failed.", claimed.job_id)
        fail_job(session, claimed, error)
        return True

    try:
        complete_job(session, claimed, signal, segments)
    except Exception as error:
        logger.exception("Job %s could not be completed.", claimed.job_id)
        fail_job(session, claimed, error)
        return True

    logger.info("Completed job %s with a %s signal.", claimed.job_id, signal.status)

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


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    stopping = Stopping()
    stopping.install()

    logger.info("DeepGuard worker started.")
    run(stopping)
    logger.info("DeepGuard worker stopped.")


if __name__ == "__main__":
    main()
