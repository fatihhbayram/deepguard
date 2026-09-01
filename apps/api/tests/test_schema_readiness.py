"""The gate that keeps the worker off a half-migrated database (R1-T3).

Two halves are tested and they need different things. The check itself is about what
`alembic_version` says against what the migration directory heads are, so it needs a real
database — and the suite already has one, brought to head by `conftest.py`, which makes the
happy path free and the unhappy one a matter of writing a different revision into that
table and putting it back.

The worker's use of it needs no database at all: what matters there is that nothing is
claimed until the check passes, that shutdown is honoured while waiting, and that a
database which cannot be reached is waited on rather than crashed on. Those are properties
of the loop, and a stand-in for the check shows them without a server.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import limits, worker
from app.db import schema
from app.db.session import SessionLocal, engine


def database_is_reachable() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return False

    return True


@pytest.fixture
def stamped():
    """Write an arbitrary revision into `alembic_version`, and put the real one back.

    Restoring in a fixture rather than at the end of each test, so a failing assertion
    cannot leave the suite's database claiming to be at a revision that does not exist —
    every later integration test would then be testing a lie about its own schema.
    """
    original = schema.applied_revisions(engine)

    def stamp(revision: str) -> None:
        with SessionLocal() as session:
            session.execute(text("DELETE FROM alembic_version"))
            session.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
                {"revision": revision},
            )
            session.commit()

    yield stamp

    with SessionLocal() as session:
        session.execute(text("DELETE FROM alembic_version"))
        for revision in original:
            session.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:revision)"),
                {"revision": revision},
            )
        session.commit()


def test_the_migration_directory_has_exactly_one_head():
    # Not a property of this task's code, but the property every other assertion here rests
    # on: two heads would mean there is no single schema for a worker to be at, and the
    # check would start refusing to start any worker at all.
    assert len(schema.expected_revisions()) == 1


@pytest.mark.integration
def test_a_database_at_head_is_ready():
    if not database_is_reachable():
        pytest.skip("PostgreSQL is not reachable")

    # `conftest.py` migrates the suite's database to head before anything runs, so this is
    # the ordinary deployed state and it must pass without raising.
    schema.check_schema_ready(engine)


@pytest.mark.integration
def test_a_database_behind_head_is_not_ready(stamped):
    if not database_is_reachable():
        pytest.skip("PostgreSQL is not reachable")

    # What a deployment that started the worker before running `alembic upgrade head` looks
    # like: the version table names a revision that is not the current one.
    stamped("e58c4d97c70a")

    with pytest.raises(schema.SchemaNotReady) as raised:
        schema.check_schema_ready(engine)

    assert raised.value.found == frozenset({"e58c4d97c70a"})
    assert raised.value.expected == schema.expected_revisions()


@pytest.mark.integration
def test_a_database_ahead_of_head_is_not_ready(stamped):
    if not database_is_reachable():
        pytest.skip("PostgreSQL is not reachable")

    # The rollback case, refused for the same reason as the one above: an old worker against
    # a schema it does not know about is the same class of mismatch, and letting it through
    # would mean a failed deployment silently ran half a fleet against unknown tables.
    stamped("ffffffffffff")

    with pytest.raises(schema.SchemaNotReady):
        schema.check_schema_ready(engine)


def test_the_error_says_what_to_do_about_it():
    error = schema.SchemaNotReady(
        expected=frozenset({"a1b2c3d4e5f6"}), found=frozenset({"0f1e2d3c4b5a"})
    )

    message = str(error)
    assert "a1b2c3d4e5f6" in message
    assert "0f1e2d3c4b5a" in message
    assert "alembic upgrade head" in message


def test_a_database_that_was_never_migrated_reads_as_no_revisions():
    error = schema.SchemaNotReady(expected=frozenset({"a1b2c3d4e5f6"}), found=frozenset())

    # The empty set has to render as something, or the message reads "found ." and an
    # operator cannot tell an unmigrated database from a truncated log line.
    assert "found none" in str(error)


# --- the worker's use of the check ---------------------------------------------------------


class FakeSchema:
    """Answers the readiness check from a script, and counts how often it was asked."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def check(self, _engine):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome


def test_the_worker_waits_until_the_schema_is_ready(monkeypatch):
    fake = FakeSchema(
        schema.SchemaNotReady(expected=frozenset({"head"}), found=frozenset({"old"})),
        schema.SchemaNotReady(expected=frozenset({"head"}), found=frozenset({"old"})),
    )
    monkeypatch.setattr(worker, "check_schema_ready", fake.check)
    slept = []

    assert worker.wait_for_schema(worker.Stopping(), sleep=slept.append) is True
    # Two refusals, then the answer it was waiting for.
    assert fake.calls == 3
    assert slept == [worker.SCHEMA_POLL_SECONDS, worker.SCHEMA_POLL_SECONDS]


def test_a_ready_schema_is_not_waited_on(monkeypatch):
    fake = FakeSchema()
    monkeypatch.setattr(worker, "check_schema_ready", fake.check)
    slept = []

    assert worker.wait_for_schema(worker.Stopping(), sleep=slept.append) is True
    assert fake.calls == 1
    assert slept == []


def test_a_database_that_cannot_be_reached_is_waited_on_rather_than_crashed_on(monkeypatch):
    # Not told apart from a schema that is behind, deliberately: both are "not ready yet"
    # and both resolve without intervention when the dependency comes up.
    fake = FakeSchema(SQLAlchemyError("connection refused"))
    monkeypatch.setattr(worker, "check_schema_ready", fake.check)

    assert worker.wait_for_schema(worker.Stopping(), sleep=lambda _: None) is True
    assert fake.calls == 2


def test_shutdown_while_waiting_stops_the_wait(monkeypatch):
    stopping = worker.Stopping()
    fake = FakeSchema(
        schema.SchemaNotReady(expected=frozenset({"head"}), found=frozenset()),
        schema.SchemaNotReady(expected=frozenset({"head"}), found=frozenset()),
    )
    monkeypatch.setattr(worker, "check_schema_ready", fake.check)

    # SIGTERM arriving mid-rollout, modelled as the signal handler running during the sleep.
    def stop(_seconds):
        stopping.request()

    assert worker.wait_for_schema(stopping, sleep=stop) is False
    # Asked once, refused, asked to stop — and never asked again.
    assert fake.calls == 1


def test_a_worker_that_is_already_stopping_asks_nothing(monkeypatch):
    stopping = worker.Stopping()
    stopping.request()
    fake = FakeSchema()
    monkeypatch.setattr(worker, "check_schema_ready", fake.check)

    assert worker.wait_for_schema(stopping, sleep=lambda _: None) is False
    assert fake.calls == 0


def test_a_misconfigured_timeout_stops_the_worker_before_it_reaches_the_database(
    monkeypatch,
):
    fake = FakeSchema()
    monkeypatch.setattr(worker, "check_schema_ready", fake.check)
    monkeypatch.setenv(limits.NORMALIZATION_TIMEOUT_VARIABLE, "15m")

    # Non-zero, so Compose restarts it and an operator sees a container that will not come
    # up rather than one that came up ignoring the bound they set.
    assert worker.main() == 1
    # And it gave up before waiting on a migration that has nothing to do with the problem.
    assert fake.calls == 0
