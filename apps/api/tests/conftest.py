"""Binds the whole suite to a database of its own, before anything opens a connection.

The integration tests need real PostgreSQL — claiming a job is a locking property, and
`SELECT ... FOR UPDATE SKIP LOCKED` cannot be demonstrated against anything else. What they
did not need was the *development* database, which is what they used to get, and a live
`api-worker` polling that database every two seconds would occasionally claim a job the
suite had just queued and fail a concurrency test that was working perfectly. The race was
real, intermittent, and cost more than one green run's credibility (P4/P5).

So the suite gets `deepguard_test`, a separate database on the same server. The worker's
`POSTGRES_DB` still says `deepguard` and nothing here changes that: the two processes are
pointed at different databases, which is why the worker cannot see a test job rather than
merely being unlikely to.

The binding has to happen here, at import, and before `app.db.session` is imported by any
test module — that module builds its engine at import time, so a single earlier import would
pin the whole suite to the development database. Setting `DATABASE_URL` is how the
application already accepts an explicit database, so nothing in `app/` needed changing for
this. Everything below is test-owned.

`tests/test_database_isolation.py` proves the separation behaviourally rather than trusting
the two names to differ.
"""

import os
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

# The API directory: where `alembic.ini` lives and where `alembic upgrade` has to run.
API_ROOT = Path(__file__).resolve().parent.parent

# The database the suite owns. Named apart from the development one on purpose — the point
# of this file is that the two are different databases, not the same one under two names.
DEFAULT_TEST_DATABASE = "deepguard_test"

# PostgreSQL's own maintenance database, which is where `CREATE DATABASE` has to be issued
# from: a connection cannot create the database it is connected to.
MAINTENANCE_DATABASE = "postgres"


def development_database_name() -> str:
    """What the API and the worker are pointed at, read the way they read it."""
    return os.getenv("POSTGRES_DB", "deepguard")


def resolve_test_database_url() -> URL:
    """The suite's own database, from explicit test configuration.

    `TEST_DATABASE_URL` wins when it is set. Otherwise the database name comes from
    `TEST_POSTGRES_DB` and everything else — host, port, credentials — is the development
    server's, because it is the same server: what is being separated is the database, not
    the deployment, and inventing a second set of credentials to reach it would be more
    configuration for no more isolation.

    Refuses to hand back the development database. A misconfigured `TEST_DATABASE_URL`, or
    a `TEST_POSTGRES_DB` someone set to `deepguard`, would put the suite straight back into
    the race this file exists to remove — and it would do it silently, because everything
    would still pass until the worker happened to win one. That is worth an error at
    collection rather than a surprise in a later phase.
    """
    configured = os.getenv("TEST_DATABASE_URL")
    url = (
        make_url(configured)
        if configured
        else URL.create(
            drivername="postgresql+psycopg2",
            username=os.getenv("POSTGRES_USER", "deepguard"),
            password=os.getenv("POSTGRES_PASSWORD", "deepguard"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("TEST_POSTGRES_DB", DEFAULT_TEST_DATABASE),
        )
    )

    if url.database == development_database_name():
        raise RuntimeError(
            "The test database is configured as the development database "
            f"('{url.database}'). The suite would share it with the running worker; "
            "set TEST_POSTGRES_DB or TEST_DATABASE_URL to a separate database."
        )

    return url


# Bound at import, before any test module can import `app.db.session` and freeze an engine
# against the development database. Nothing has connected yet — this is a string.
TEST_DATABASE_URL = resolve_test_database_url()
os.environ["DATABASE_URL"] = TEST_DATABASE_URL.render_as_string(hide_password=False)


def create_test_database(url: URL) -> None:
    """Create the suite's database if the server does not have it yet.

    Issued from the maintenance database and outside a transaction, which is what
    PostgreSQL requires of `CREATE DATABASE`. Existence is checked first because there is
    no `IF NOT EXISTS` for it.
    """
    server = create_engine(
        url.set(database=MAINTENANCE_DATABASE), isolation_level="AUTOCOMMIT"
    )

    try:
        with server.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            ).scalar()

            if not exists:
                # The name is ours, not a request's, and comes from the environment this
                # process was started with — but it still cannot be a bound parameter in
                # DDL, so it is quoted as an identifier rather than interpolated raw.
                connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        server.dispose()


def migrate_test_database(url: URL) -> None:
    """Bring the suite's database to the current Alembic head.

    The real `alembic upgrade head`, against the same migration files production runs, so
    the schema under test cannot drift from the schema that ships. A handwritten test
    schema would be exactly the thing that silently stops matching.
    """
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        cwd=API_ROOT,
        env={**os.environ, "DATABASE_URL": url.render_as_string(hide_password=False)},
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Migrating the test database failed "
            f"(alembic exited with {result.returncode}):\n{result.stderr.strip()}"
        )


# The browser origin the suite's dashboard mutations claim to come from (R1-T2).
#
# A name no deployment uses, so a test that forgets to send the header cannot pass by
# accidentally matching something real, and nothing here has to agree with whatever
# `DEEPGUARD_WEB_ORIGIN` happens to be set to in the environment the suite is run in.
DASHBOARD_ORIGIN = "http://dashboard.test"


@pytest.fixture(autouse=True)
def configured_web_origin(monkeypatch):
    """Declare the accepted web origin for every test in the suite.

    The API refuses a session-authenticated mutation whose `Origin` is not on this list, and
    refuses everything when the list is empty — which is the fail-secure default and would
    otherwise turn every submission test in the suite into a 403 about configuration rather
    than about the thing under test.

    So the environment is stated here, once, and the tests that are *about* the check
    override it with their own `monkeypatch` call, which runs after this fixture and wins.

    Imported inside the function rather than at module scope: this file binds `DATABASE_URL`
    at import time and must finish doing so before anything pulls in `app.db.session`, which
    builds its engine on import.
    """
    from app.web_auth import WEB_ORIGIN_VARIABLE

    monkeypatch.setenv(WEB_ORIGIN_VARIABLE, DASHBOARD_ORIGIN)


@pytest.fixture(scope="session")
def suite_database_url() -> URL:
    """The database this suite owns, for the tests that assert the separation itself."""
    return TEST_DATABASE_URL


@pytest.fixture(scope="session")
def development_database() -> str:
    """The database the API and the worker use, which nothing in this suite may touch."""
    return development_database_name()


@pytest.fixture(scope="session", autouse=True)
def prepared_test_database():
    """Create and migrate the suite's database once, before any test opens a session.

    A server that is not there is not an error here. The unit tests need no database at
    all, and the integration tests already skip themselves when they cannot reach one — so
    an unreachable PostgreSQL leaves this quiet and lets those skips do their job.

    A server that *is* there and cannot be migrated is a different matter and raises: the
    integration tests would otherwise fail one by one against a half-built schema, which
    reads as broken code rather than a database that was never brought up to head.
    """
    try:
        create_test_database(TEST_DATABASE_URL)
    except SQLAlchemyError:
        return

    migrate_test_database(TEST_DATABASE_URL)
