"""The worker, against real PostgreSQL.

Claiming is the whole point of this module, and claiming is a concurrency property: two
workers reaching for the same row, and exactly one of them getting it. No fake session
can show that, so the job tests here run against the database from docker-compose and
skip when none is reachable. Only MinIO and NVIDIA are stood in for.

The loop tests at the bottom need no database — what they check is when the worker asks
for work again, not what it finds.
"""

import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import detection, nvidia_video, storage, worker
from app.db.models import (
    Analysis,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import SessionLocal, engine

VIDEO_BYTES = b"canonical-mp4-bytes"
NVIDIA_PROBABILITY = 0.8734567165374756
NVIDIA_LOGIT = 1.9142135381698608
NVIDIA_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"

# Test jobs are backdated so that oldest-first claiming reaches them before anything a
# real upload left in the queue. It makes these tests deterministic against a shared
# development database without touching a single row they do not own.
BACKDATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def queue(database):
    """Creates queued uploads exactly as the route commits them, and removes them again."""
    created = []

    def enqueue(*, was_normalized=False, age=0):
        with SessionLocal() as session:
            analysis = Analysis(status="queued")
            session.add(analysis)
            session.flush()
            digest = uuid.uuid4().hex
            session.add(
                MediaFile(
                    analysis_id=analysis.id,
                    original_filename="clip.mp4",
                    content_type="video/mp4",
                    size_bytes=len(VIDEO_BYTES),
                    original_sha256=digest,
                    original_storage_key=f"originals/{digest}",
                    format_name="mov,mp4,m4a,3gp,3g2,mj2",
                    codec_name="h264",
                    width=1920,
                    height=1080,
                    duration=12.34,
                    frame_rate=30.0,
                    pix_fmt="yuv420p",
                    constant_frame_rate=True,
                    was_normalized=was_normalized,
                    derivative_storage_key=(
                        f"derivatives/{digest}.mp4" if was_normalized else f"originals/{digest}"
                    ),
                    derivative_sha256=digest if was_normalized else None,
                )
            )
            job = AnalysisJob(
                analysis_id=analysis.id,
                status="queued",
                created_at=BACKDATE + timedelta(seconds=age),
            )
            session.add(job)
            session.commit()
            created.append(analysis.id)

            return analysis.id, job.id

    yield enqueue

    with SessionLocal() as session:
        for analysis_id in created:
            # Media, job, signal and segments all go through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


@pytest.fixture
def fake_storage(monkeypatch):
    """Stand in for MinIO, writing plausible video bytes wherever the worker asks."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.fetched = []
            # Where the artifact was written, captured during the call: the worker
            # deletes it before returning, so looking afterwards would prove nothing.
            self.paths = []

        def fget_object(self, bucket, key, file_path):
            self.fetched.append(key)
            self.paths.append(Path(file_path))
            if self.error:
                raise self.error
            Path(file_path).write_bytes(VIDEO_BYTES)

    recorder = Recorder()
    monkeypatch.setattr(storage, "client", recorder)
    return recorder


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """Stand in for the detector, so no test in this module can reach NVIDIA."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.clips = (
                nvidia_video.NvidiaClipResult(index=0, logit=-2.25),
                nvidia_video.NvidiaClipResult(index=8, logit=3.5),
            )
            self.analysed_bytes = []

        async def analyze(self, file_path, **kwargs):
            self.analysed_bytes.append(Path(file_path).read_bytes())
            if self.error:
                raise self.error
            return nvidia_video.NvidiaVideoResult(
                logit=NVIDIA_LOGIT,
                probability=NVIDIA_PROBABILITY,
                total_clips=7,
                csv_data="0,-2.25\n8,3.5\n",
                function_id=NVIDIA_FUNCTION_ID,
                clips=self.clips,
            )

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_video", recorder.analyze)
    return recorder


def release(claimed) -> None:
    """Put a job this test did not create back the way it found it.

    These tests share a development database with real uploads, and `claim_job` takes
    whatever is queued. Test jobs are backdated so they are always claimed first, but a
    claim that reaches past them has taken someone else's work — and leaving it
    `processing` would strand it, because nothing in P3 recovers a stalled job.
    """
    if claimed is None:
        return

    with SessionLocal() as session:
        session.query(AnalysisJob).filter(AnalysisJob.id == claimed.job_id).update(
            {"status": "queued"}
        )
        session.commit()


def read_job(job_id) -> AnalysisJob:
    """The job as another connection sees it — committed state, not session state."""
    with SessionLocal() as reader:
        return reader.query(AnalysisJob).filter(AnalysisJob.id == job_id).one()


def read_analysis(analysis_id) -> Analysis:
    with SessionLocal() as reader:
        return reader.query(Analysis).filter(Analysis.id == analysis_id).one()


@pytest.mark.integration
def test_claiming_a_job_marks_it_processing_and_commits(queue):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    assert claimed.job_id == job_id
    assert claimed.analysis_id == analysis_id
    # Read on another connection: if the claim had not committed, this would still say
    # queued — and the transaction would still be holding the lock through inference.
    assert read_job(job_id).status == "processing"


@pytest.mark.integration
def test_claiming_carries_the_canonical_storage_key_out_of_the_transaction(queue):
    _, job_id = queue(was_normalized=True)

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    # The provider-compatible artifact, which for normalized media is the derivative.
    assert claimed.storage_key.startswith("derivatives/")


@pytest.mark.integration
def test_a_locked_job_is_skipped_rather_than_waited_for(queue):
    """The property `SKIP LOCKED` exists for, isolated.

    One session holds the row lock without committing. A second must come back with
    nothing at once — not block until the first is done, which is what a plain
    `FOR UPDATE` would do and what would serialize every worker in the fleet behind the
    slowest video.
    """
    queue()

    holder = SessionLocal()
    try:
        held = worker.claim_job(holder)
        assert held is not None

        with SessionLocal() as other:
            # If the statement ever blocks instead of skipping, this makes it fail
            # loudly rather than hang the suite forever.
            other.execute(text("SET lock_timeout = '2000ms'"))
            second = worker.claim_job(other)
            release(second)

        # Whatever else the queue happens to hold, the locked row is not what the second
        # worker came back with.
        assert second is None or second.job_id != held.job_id
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.integration
def test_racing_workers_never_claim_the_same_job_twice(queue):
    """Several workers, several jobs, started together on purpose.

    Every job must be claimed, and none of them twice. A duplicate claim would mean the
    same video analysed twice and two conflicting sets of evidence for one analysis.
    """
    expected = {queue(age=index)[1] for index in range(6)}
    start = threading.Barrier(3)
    claimed = []
    guard = threading.Lock()

    def work():
        start.wait()
        with SessionLocal() as session:
            while True:
                job = worker.claim_job(session)
                if job is None:
                    return
                if job.job_id not in expected:
                    # Reached past this test's jobs into the real queue. Put it back.
                    release(job)
                    return
                with guard:
                    claimed.append(job.job_id)

    threads = [threading.Thread(target=work) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(claimed) == sorted(set(claimed))
    assert set(claimed) == expected


@pytest.mark.integration
def test_a_processed_job_completes_with_its_evidence(queue, fake_storage):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    assert read_job(job_id).status == "completed"
    assert read_job(job_id).error_message is None
    # The analysis moves with its job: detection is the work it was waiting for.
    assert read_analysis(analysis_id).status == "completed"

    with SessionLocal() as reader:
        signal = reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).one()
        segments = (
            reader.query(AnalysisSegment)
            .filter_by(signal_id=signal.id)
            .order_by(AnalysisSegment.logit.desc())
            .all()
        )

    assert signal.provider == "nvidia"
    assert signal.status == "SUCCESS"
    # NVIDIA's own number, on NVIDIA's own scale.
    assert signal.score == NVIDIA_PROBABILITY
    assert signal.risk_level is None
    assert [(s.clip_index, s.logit) for s in segments] == [(8, 3.5), (0, -2.25)]


@pytest.mark.integration
def test_the_worker_detects_the_canonical_object_it_fetched(queue, fake_storage, fake_nvidia):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    # Nothing is handed between the upload and the worker but a storage key, so the
    # object store is what the detector's input has to come from — and for normalized
    # media the object to fetch is the derivative, not the original.
    assert len(fake_storage.fetched) == 1
    assert fake_storage.fetched[0].startswith("derivatives/")
    assert fake_nvidia.analysed_bytes == [VIDEO_BYTES]


@pytest.mark.integration
def test_the_downloaded_artifact_is_removed_afterwards(queue, fake_storage):
    queue()

    with SessionLocal() as session:
        worker.process_one(session)

    # The worker holds no media between jobs; a container that ran for a week would
    # otherwise fill its disk with every video it had ever been asked about.
    assert fake_storage.paths
    assert [path for path in fake_storage.paths if path.exists()] == []


@pytest.mark.integration
def test_a_provider_failure_still_completes_the_job(queue, fake_storage, fake_nvidia):
    analysis_id, job_id = queue()
    fake_nvidia.error = nvidia_video.NvidiaProviderTimeout("deadline exceeded")

    with SessionLocal() as session:
        worker.process_one(session)

    # NVIDIA not answering is a fact about NVIDIA, recorded as the signal's own status.
    # The job did the work it was queued to do, so it is not the job that failed.
    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"

    with SessionLocal() as reader:
        signal = reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).one()

    assert signal.status == "TIMEOUT"
    assert signal.score is None


@pytest.mark.integration
def test_an_unreadable_artifact_fails_the_job_and_the_analysis(queue, fake_storage):
    analysis_id, job_id = queue()
    fake_storage.error = OSError("connection reset")

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    # This one is ours, not the detector's. An analysis left queued behind a job that
    # already gave up would look like work still coming.
    assert read_job(job_id).status == "failed"
    assert read_analysis(analysis_id).status == "failed"

    with SessionLocal() as reader:
        assert reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).count() == 0


@pytest.mark.integration
def test_a_failed_job_records_the_failure_kind_and_nothing_more(queue, fake_storage):
    _, job_id = queue()
    fake_storage.error = OSError("connect to minio:9000 failed for user deepguard")

    with SessionLocal() as session:
        worker.process_one(session)

    # Exception text can quote credentials, storage endpoints or SQL. The class name is
    # what an operator needs from the table; the traceback stays in the worker log.
    assert read_job(job_id).error_message == "OSError"


@pytest.mark.integration
def test_a_failed_job_does_not_stop_the_next_one(queue, fake_storage):
    _, failing = queue(age=0)
    _, following = queue(age=1)
    fake_storage.error = OSError("connection reset")

    with SessionLocal() as session:
        worker.process_one(session)
        fake_storage.error = None
        worker.process_one(session)

    # One broken job is not a broken worker.
    assert read_job(failing).status == "failed"
    assert read_job(following).status == "completed"


# The loop. What it checks is when the worker asks for work again, not what it finds, so
# nothing below needs a database.


class FakeClock:
    """Records what the loop slept for instead of actually sleeping."""

    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


def run_loop(monkeypatch, outcomes):
    """Drive `run` through a fixed sequence of `process_one` results, then stop."""
    stopping = worker.Stopping()
    remaining = list(outcomes)

    def process_one(_session):
        if not remaining:
            # Stop without a further pause, so the sequence under test is the only thing
            # the clock records.
            stopping.requested = True
            return True
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    clock = FakeClock()
    monkeypatch.setattr(worker, "process_one", process_one)
    monkeypatch.setattr(worker, "SessionLocal", lambda: _NullSession())
    worker.run(stopping, sleep=clock)

    return clock


class _NullSession:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_the_loop_does_not_pause_while_there_is_work(monkeypatch):
    clock = run_loop(monkeypatch, [True, True, True])

    # A backlog is drained as fast as it can be, not one job per poll interval.
    assert clock.slept == []


def test_an_empty_queue_is_polled_rather_than_busy_looped(monkeypatch):
    clock = run_loop(monkeypatch, [False, False])

    assert clock.slept == [worker.IDLE_POLL_SECONDS, worker.IDLE_POLL_SECONDS]


def test_a_broken_database_backs_off_instead_of_exiting(monkeypatch):
    clock = run_loop(monkeypatch, [SQLAlchemyError("connection lost"), True])

    # Restarting the container would only meet the same condition, and a crash loop is
    # harder to read in the logs than a worker saying the same thing every few seconds.
    assert clock.slept == [worker.ERROR_BACKOFF_SECONDS]


def test_the_loop_stops_when_shutdown_is_requested(monkeypatch):
    stopping = worker.Stopping()
    stopping.request()

    called = []
    monkeypatch.setattr(worker, "process_one", lambda _s: called.append(1))

    worker.run(stopping, sleep=FakeClock())

    # SIGTERM from `docker compose down`: stop asking for new work rather than being
    # killed ten seconds later, mid-job.
    assert called == []
