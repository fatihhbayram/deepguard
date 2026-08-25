"""Proof that the suite and the running worker are on different databases.

Every other integration module assumes this. It used to be false: the tests and the
development `api-worker` shared one database, and the worker would occasionally claim a job
`test_racing_workers_never_claim_the_same_job_twice` had queued, turning a passing
concurrency test red for reasons that had nothing to do with concurrency.

Two names in a config file are not proof of anything, so nothing here checks names alone:
the separation is demonstrated by writing a job the suite owns and then failing to find it
from the development database, and by leaving it queued for longer than the worker's poll
interval to show that nothing came for it.

These tests are meaningful precisely when the development stack is up. With `api-worker`
stopped they still pass, and prove less — which is the honest position, and why the suite
is no longer run with the worker stopped.
"""

import time
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Analysis, AnalysisJob, MediaFile
from app.db.session import SessionLocal, database_url, engine
from app.worker import IDLE_POLL_SECONDS

pytestmark = pytest.mark.integration

# How long a job is left in the queue to show that nothing takes it. The worker asks for
# work every `IDLE_POLL_SECONDS`, so several intervals is several chances to have claimed
# it — this is the substance of the proof, not a delay hiding a race.
WORKER_POLL_CYCLES = 4


@pytest.fixture(scope="module")
def database():
    """The suite's engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def development_engine(database, development_database):
    """A connection to the database the API and the worker actually use.

    Everything but the database name comes from `database_url()`, the application's own
    function, rather than from a second copy of that logic here — so where this connects
    cannot drift from where the worker connects. Only the name is swapped back, which is
    the whole of the difference between the two.
    """
    development = create_engine(
        database_url().set(database=development_database), pool_pre_ping=True
    )

    try:
        yield development
    finally:
        development.dispose()


@pytest.fixture
def queued_job(database):
    """One queued analysis and its job, committed to the suite's database and removed after.

    Shaped exactly as the upload route commits them, because a job the worker would refuse
    to claim for some *other* reason would prove nothing about which database it is in.
    """
    with SessionLocal() as session:
        analysis = Analysis(status="queued")
        session.add(analysis)
        session.flush()
        digest = uuid.uuid4().hex
        session.add(
            MediaFile(
                analysis_id=analysis.id,
                original_filename="isolation.mp4",
                content_type="video/mp4",
                size_bytes=1024,
                original_sha256=digest,
                original_storage_key=f"originals/{digest}",
                format_name="mov,mp4,m4a,3gp,3g2,mj2",
                codec_name="h264",
                width=1280,
                height=720,
                duration=1.0,
                frame_rate=30.0,
                pix_fmt="yuv420p",
                constant_frame_rate=True,
                was_normalized=False,
                derivative_storage_key=f"originals/{digest}",
                derivative_sha256=None,
            )
        )
        job = AnalysisJob(analysis_id=analysis.id, status="queued")
        session.add(job)
        session.commit()

        identifiers = (analysis.id, job.id)

    yield identifiers

    with SessionLocal() as session:
        session.query(Analysis).filter(Analysis.id == identifiers[0]).delete()
        session.commit()


def test_the_suite_is_not_pointed_at_the_development_database(
    suite_database_url, development_database
):
    """The precondition everything else here rests on."""
    assert suite_database_url.database != development_database
    # And the application really did bind to it, rather than the override being set and
    # some earlier import having already frozen an engine against the development database.
    assert engine.url.database == suite_database_url.database


def test_the_two_databases_are_separate_on_the_same_server(development_engine):
    """Separate databases, not separate servers: the isolation is structural, not remote."""
    assert development_engine.url.host == engine.url.host
    assert development_engine.url.port == engine.url.port
    assert development_engine.url.database != engine.url.database


def test_a_job_the_suite_queued_does_not_exist_in_the_development_database(
    queued_job, development_engine
):
    analysis_id, job_id = queued_job

    with development_engine.connect() as connection:
        found = connection.execute(
            text("SELECT count(*) FROM analysis_jobs WHERE id = :id"), {"id": job_id}
        ).scalar()
        analyses = connection.execute(
            text("SELECT count(*) FROM analyses WHERE id = :id"), {"id": analysis_id}
        ).scalar()

    # The worker polls `analysis_jobs` in the database above and nothing else. A row that
    # is not in it is a row the worker has no way of reaching.
    assert found == 0
    assert analyses == 0


def test_a_running_development_worker_leaves_the_suite_s_job_alone(queued_job):
    """The behavioural half: with the stack up, nothing comes for this job.

    Every claim the worker makes moves a row to `processing` and touches `updated_at`, so
    both are checked — a worker that claimed and finished the job inside the wait would
    still leave `updated_at` moved even though the status had come to rest.
    """
    _, job_id = queued_job

    with SessionLocal() as session:
        before = session.get(AnalysisJob, job_id).updated_at

    time.sleep(IDLE_POLL_SECONDS * WORKER_POLL_CYCLES)

    with SessionLocal() as session:
        after = session.get(AnalysisJob, job_id)

        assert after.status == "queued"
        assert after.error_message is None
        assert after.updated_at == before


def test_the_development_database_carries_the_same_schema(development_engine):
    """Both databases are migrated by the same Alembic history, so neither is a special case.

    This is what keeps the isolation from becoming a reason for the suite to test a schema
    the application never runs against.
    """
    with development_engine.connect() as connection:
        development_revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()

    with engine.connect() as connection:
        test_revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()

    assert test_revision == development_revision
