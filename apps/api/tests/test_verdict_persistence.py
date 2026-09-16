"""That the database can hold an R9 verdict, and that going back refuses to shorten one.

`analyses.risk_level` was `varchar(16)` until R9-T8A, which fits every verdict ruleset v4
can produce and neither of the two long ones `r9-v5.0.0` produces. Nothing writes a v5
verdict yet — `evaluate_v5` still has no caller in the production path, and this file does
not give it one — so the question here is only whether the *storage* is ready: whether a v5
verdict survives a round trip through real PostgreSQL and comes back out of the API as the
exact string that went in.

Three things are proven, and the third is the one that matters most later:

  * the three v5 verdicts persist and serialize exactly, with no truncation;
  * the v4 verdicts and the null that means "not decided yet" are untouched by the widening;
  * the migration's `downgrade()` refuses to run once a verdict too long for the old width
    is stored, rather than truncating it into a value no ruleset can explain.

Everything here needs real PostgreSQL. A `varchar` width is a property of the server, not of
SQLAlchemy — a fake session would accept a 33-character verdict into a 16-character column
without complaint, which is exactly the failure this file exists to catch.
"""

import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from app.auth import generate_api_key
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_QUEUED,
    SIGNAL_STATUS_SUCCESS,
    USER_ROLE_ADMIN,
    Analysis,
    AnalysisSignal,
    ApiKey,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine, get_session
from app.detection import NVIDIA_PROVIDER, SYNTHETIC_VIDEO_SIGNAL
from app.main import app
from app.risk_engine import (
    CALIBRATION_ID,
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_UNKNOWN,
    RULES_VERSION,
    RULES_VERSION_V5,
    RULE_V5_HIGH_SVD,
    VERDICT_INCONCLUSIVE,
    VERDICT_MANIPULATION_DETECTED,
    VERDICT_NO_SIGNAL,
)
from app.web_auth import hash_password, require_user
from tests.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.integration

# Where `alembic.ini` lives, for the subprocess migrations at the bottom of this file.
API_ROOT = Path(__file__).resolve().parent.parent

# The width the column had before R9-T8A, and the width it has now. Named here rather than
# imported from the migration: these tests assert the contract, and reading the numbers out
# of the thing under test would make them agree with it by construction.
LEGACY_WIDTH = 16
WIDTH = 64

# The whole R9 vocabulary, and the two that could not be stored before this task.
V5_VERDICTS = (VERDICT_MANIPULATION_DETECTED, VERDICT_NO_SIGNAL, VERDICT_INCONCLUSIVE)
V5_VERDICTS_TOO_LONG_FOR_THE_OLD_COLUMN = tuple(
    verdict for verdict in V5_VERDICTS if len(verdict) > LEGACY_WIDTH
)

# What every ruleset through `r7-v4.0.0` can have written. `LOW` is measured and never
# activated, so no stored row carries it and it is not asserted here as though one might.
V4_VERDICTS = (RISK_HIGH, RISK_MEDIUM, RISK_UNKNOWN)


@pytest.fixture(scope="module")
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def session(database):
    """A real session whose analyses and keys are removed again, in that order.

    Analyses first: `analyses.api_key_id` is `ON DELETE RESTRICT`, so a key cannot be
    deleted while an analysis still names it.
    """
    analyses = []
    keys = []

    with SessionLocal() as db:
        yield db, analyses, keys

        db.rollback()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.flush()
        for key_id in keys:
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()


def store_decided_analysis(
    session,
    *,
    risk_level,
    rules_version,
    rule_id,
    owner=None,
    status=ANALYSIS_STATUS_COMPLETED,
) -> Analysis:
    """Persist one completed analysis carrying a decision, and the media row reads join on.

    The four risk columns are written together, which is how the worker writes them: a
    stored level with no trace beside it is not a shape this schema ever holds, and testing
    the level in isolation would be testing a row that cannot exist.
    """
    db, analyses, _ = session

    analysis = Analysis(
        status=status,
        api_key_id=owner.id if owner is not None else None,
        risk_level=risk_level,
        risk_rules_version=rules_version,
        risk_rule_id=rule_id,
        risk_calibration_id=CALIBRATION_ID,
    )
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    digest = hashlib.sha256(str(analysis.id).encode()).hexdigest()
    db.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename="clip.mp4",
            content_type="video/mp4",
            size_bytes=4096,
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
            was_normalized=False,
        )
    )
    db.commit()

    return analysis


@pytest.fixture
def administrator(session):
    """A real administrator account, for the dashboard reads below."""
    db, _, _ = session

    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password("test-account-password"),
        role=USER_ROLE_ADMIN,
    )
    db.add(user)
    db.commit()

    yield user

    db.rollback()
    db.query(Analysis).filter(Analysis.owner_id == user.id).delete()
    db.flush()
    db.query(User).filter(User.id == user.id).delete()
    db.commit()


@pytest.fixture
def reader(session, administrator):
    """A client over the live session, signed in as an administrator."""
    db, _, _ = session
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[require_user] = lambda: administrator

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def issue_key(session) -> tuple[str, ApiKey]:
    """Persist a real key and hand back the plaintext to authenticate with."""
    db, _, keys = session
    generated = generate_api_key()

    key = ApiKey(name="verdict-persistence", key_hash=generated.key_hash, is_active=True)
    db.add(key)
    db.commit()
    keys.append(key.id)

    return generated.plaintext, key


# --- the column itself -------------------------------------------------------------------


def test_the_migration_widened_both_risk_level_columns(database):
    """The width is read off the server, because the server is what enforces it.

    Asserted against the two tables that declare the risk-level type and nothing else: the
    point of R9-T8A is that these two moved, not that every short column in the schema did.
    """
    inspector = inspect(database)

    for table in ("analyses", "analysis_signals"):
        column = next(
            column
            for column in inspector.get_columns(table)
            if column["name"] == "risk_level"
        )
        assert column["type"].length == WIDTH, table


def test_every_v5_verdict_fits_the_column_with_room_left(database):
    """The width is not merely enough today, and the assertion says which of those it is.

    Without the second half, a vocabulary that grew a verdict of exactly 64 characters would
    pass this file while leaving no margin at all, which is the situation R9-T8A was called
    in to end rather than to repeat one size up.
    """
    longest = max(len(verdict) for verdict in V5_VERDICTS)

    assert longest <= WIDTH
    assert longest < WIDTH
    # And the premise of the whole task: the old width really could not hold them.
    assert longest > LEGACY_WIDTH
    assert V5_VERDICTS_TOO_LONG_FOR_THE_OLD_COLUMN == (
        VERDICT_MANIPULATION_DETECTED,
        VERDICT_NO_SIGNAL,
    )


# --- persistence and serialization -------------------------------------------------------


@pytest.mark.parametrize("verdict", V5_VERDICTS)
def test_a_v5_verdict_round_trips_through_postgresql(session, verdict):
    """Written, committed, and read back in a second session as the same string.

    Re-read through a fresh query rather than off the object still in the identity map,
    which would return the Python string it was handed and prove nothing about the column.
    """
    db, _, _ = session
    analysis = store_decided_analysis(
        session,
        risk_level=verdict,
        rules_version=RULES_VERSION_V5,
        rule_id=RULE_V5_HIGH_SVD,
    )
    db.expunge_all()

    stored = db.query(Analysis).filter(Analysis.id == analysis.id).one()

    assert stored.risk_level == verdict
    assert len(stored.risk_level) == len(verdict)
    assert stored.risk_rules_version == RULES_VERSION_V5


@pytest.mark.parametrize("verdict", V5_VERDICTS)
def test_a_v5_verdict_serializes_exactly_through_the_dashboard_route(
    session, reader, verdict
):
    """The internal read path returns the stored verdict unabbreviated and unmapped.

    `risk_level` is asserted for exact equality rather than membership or prefix: a route
    that shortened `NO_CALIBRATED_MANIPULATION_SIGNAL` for display, or folded it onto a v4
    word, would be reporting a decision nobody took.
    """
    analysis = store_decided_analysis(
        session,
        risk_level=verdict,
        rules_version=RULES_VERSION_V5,
        rule_id=RULE_V5_HIGH_SVD,
    )

    response = reader.get(f"/api/v1/analyses/{analysis.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == verdict
    assert body["risk_rules_version"] == RULES_VERSION_V5
    assert body["risk_rule_id"] == RULE_V5_HIGH_SVD


@pytest.mark.parametrize("verdict", V5_VERDICTS)
def test_a_v5_verdict_serializes_exactly_through_the_public_api(session, verdict):
    """The customer-facing contract carries the same string, at full length.

    A separate assertion from the dashboard's because it is a separate Pydantic model over
    the same column, and the public one is the contract someone integrates against.
    """
    plaintext, key = issue_key(session)
    analysis = store_decided_analysis(
        session,
        risk_level=verdict,
        rules_version=RULES_VERSION_V5,
        rule_id=RULE_V5_HIGH_SVD,
        owner=key,
    )

    with TestClient(app) as client:
        response = client.get(
            f"/api/public/v1/analyses/{analysis.id}",
            headers={"Authorization": f"Bearer {plaintext}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == verdict
    assert body["risk_rules_version"] == RULES_VERSION_V5


@pytest.mark.parametrize("verdict", V4_VERDICTS)
def test_a_legacy_verdict_is_unaffected_by_the_widening(session, reader, verdict):
    """`HIGH`, `MEDIUM` and `UNKNOWN` persist and serialize exactly as they did.

    The widening rewrote no rows, and this is the assertion that says so from the outside:
    a v4 decision is still stored under `p7-v1.0.0`, still reads back as the same word, and
    is not re-labelled into the R9 vocabulary on the way out. Nothing maps one onto the
    other — each row is read under the ruleset version it names.
    """
    db, _, _ = session
    analysis = store_decided_analysis(
        session,
        risk_level=verdict,
        rules_version=RULES_VERSION,
        rule_id="R200",
    )
    db.expunge_all()

    stored = db.query(Analysis).filter(Analysis.id == analysis.id).one()
    assert stored.risk_level == verdict
    assert stored.risk_rules_version == RULES_VERSION

    response = reader.get(f"/api/v1/analyses/{analysis.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == verdict
    assert body["risk_rules_version"] == RULES_VERSION
    assert body["risk_level"] not in V5_VERDICTS


def test_an_undecided_analysis_still_persists_a_null_verdict(session, reader):
    """Null is the absence of a decision and must not have become an empty string.

    A widening that had gone through a `USING`, a default, or a backfill is most likely to
    show up here — as `''` where there was nothing — and null is the one value in this
    column that a reader may never fold together with a verdict.
    """
    db, _, _ = session
    # The same helper, with all four risk columns left unwritten — which is exactly the row
    # the API creates at submission and the worker has not yet decided. Its media row is
    # there because the dashboard read joins onto one, not because this test needs it.
    analysis = store_decided_analysis(
        session,
        risk_level=None,
        rules_version=None,
        rule_id=None,
        status=ANALYSIS_STATUS_QUEUED,
    )
    db.expunge_all()

    stored = db.query(Analysis).filter(Analysis.id == analysis.id).one()
    assert stored.risk_level is None
    assert stored.risk_rules_version is None

    response = reader.get(f"/api/v1/analyses/{analysis.id}")

    assert response.status_code == 200
    assert response.json()["risk_level"] is None


def test_a_persisted_signal_still_carries_no_risk_level(session):
    """`analysis_signals.risk_level` was widened with the other one and stays null.

    Widening it was about keeping one type one width, not about starting to write it: risk
    is a decision about the analysis, and a level beside every provider's number is the
    per-signal verdict rule 11 exists to refuse. If that ever changes it will be a task that
    says so, and this assertion is what that task has to come back and delete.
    """
    db, _, _ = session
    analysis = store_decided_analysis(
        session,
        risk_level=VERDICT_MANIPULATION_DETECTED,
        rules_version=RULES_VERSION_V5,
        rule_id=RULE_V5_HIGH_SVD,
    )

    signal = AnalysisSignal(
        analysis_id=analysis.id,
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
        score=0.97,
        status=SIGNAL_STATUS_SUCCESS,
    )
    db.add(signal)
    db.commit()
    db.expunge_all()

    stored = db.query(AnalysisSignal).filter(AnalysisSignal.id == signal.id).one()
    assert stored.risk_level is None


# --- the migration, forwards and back ----------------------------------------------------
#
# These run against a disposable database of their own rather than the suite's. A downgrade
# is a schema change, and running one against the database every other integration test
# shares would leave the rest of the suite testing a schema that is not the one that ships —
# including when the downgrade under test is the one that is supposed to fail halfway.

THIS_REVISION = "f8c1e4a20d75"
PREVIOUS_REVISION = "a2d8f61c95b4"


def alembic(command: list[str], url) -> subprocess.CompletedProcess:
    """Run the real migrations against a given database, as a subprocess.

    Through `sys.executable -m` rather than the bare `alembic` name, so this does not depend
    on the interpreter running the suite also having its `bin/` on `PATH` — and the real CLI
    rather than an in-process `command.upgrade()`, because a failing downgrade has to be
    observed as the exit status an operator would see, not as an exception this test caught.
    """
    return subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        cwd=API_ROOT,
        env={**os.environ, "DATABASE_URL": url.render_as_string(hide_password=False)},
        capture_output=True,
        text=True,
    )


@pytest.fixture
def disposable_database(database):
    """A fresh database, migrated to head by the real migrations, dropped afterwards.

    Fresh rather than a copy: `alembic upgrade head` on an empty database is the other half
    of what this file checks — that the chain from nothing to head still runs clean once a
    new revision is on the end of it.
    """
    name = f"deepguard_migration_{uuid.uuid4().hex[:12]}"
    url = make_url(TEST_DATABASE_URL).set(database=name)
    server = create_engine(
        make_url(TEST_DATABASE_URL).set(database="postgres"),
        isolation_level="AUTOCOMMIT",
    )

    try:
        with server.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{name}"'))

        upgraded = alembic(["upgrade", "head"], url)
        assert upgraded.returncode == 0, upgraded.stderr

        yield url
    finally:
        with server.connect() as connection:
            connection.execute(
                text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            )
        server.dispose()


@pytest.fixture
def database_at_the_widening(disposable_database):
    """The disposable database wound back to the revision these downgrade tests are about.

    `disposable_database` migrates to head, which is what the three tests above want: they ask
    whether the chain from nothing still runs clean with a new revision on the end of it. The
    downgrade tests want something narrower and are explicit about it since R10-T2 put a
    revision after this one — they are about *this* migration's conditional downgrade, so they
    start at this migration rather than at whatever happens to be last. `downgrade -1` from
    head would otherwise reverse somebody else's revision and assert nothing about this one.
    """
    wound_back = alembic(["downgrade", THIS_REVISION], disposable_database)
    assert wound_back.returncode == 0, wound_back.stderr
    assert current_revision(disposable_database) == THIS_REVISION

    return disposable_database


def insert_analysis(url, risk_level: str | None) -> uuid.UUID:
    """Write one row straight into the disposable database, ORM uninvolved.

    Raw SQL on purpose: these tests are about what the column accepts, and going through
    the mapped class would put a second thing — the model's declared width — between the
    test and the answer.
    """
    analysis_id = uuid.uuid4()
    target = create_engine(url)

    try:
        with target.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO analyses (id, status, created_at, risk_level) "
                    "VALUES (:id, :status, now(), :risk_level)"
                ),
                {
                    "id": analysis_id,
                    "status": ANALYSIS_STATUS_COMPLETED,
                    "risk_level": risk_level,
                },
            )
    finally:
        target.dispose()

    return analysis_id


def stored_width(url, table: str) -> int | None:
    """The `risk_level` width PostgreSQL currently reports for a table."""
    target = create_engine(url)

    try:
        with target.connect() as connection:
            return connection.execute(
                text(
                    "SELECT character_maximum_length FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = 'risk_level'"
                ),
                {"table": table},
            ).scalar_one()
    finally:
        target.dispose()


def current_revision(url) -> str:
    target = create_engine(url)

    try:
        with target.connect() as connection:
            return connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
    finally:
        target.dispose()


def test_a_fresh_database_migrates_to_head_with_the_wide_columns(disposable_database):
    """`alembic upgrade head` from empty leaves both verdict columns wide.

    The widening is asserted rather than the revision. R9-T8A was head when this was written
    and R10-T2 put a revision after it, and the claim worth keeping was never "this migration
    is last" — it was "a database built from nothing can hold an R9 verdict".
    """
    assert stored_width(disposable_database, "analyses") == WIDTH
    assert stored_width(disposable_database, "analysis_signals") == WIDTH
    # The widening is in the applied history, wherever head has moved to since.
    history = alembic(["history"], disposable_database)
    assert THIS_REVISION in history.stdout


def test_the_migration_directory_has_exactly_one_head(disposable_database):
    """Two heads would mean `upgrade head` is ambiguous and this branch went in beside one."""
    heads = alembic(["heads"], disposable_database)

    assert heads.returncode == 0, heads.stderr
    assert len([line for line in heads.stdout.splitlines() if line.strip()]) == 1


def test_the_models_and_the_migrated_schema_agree(disposable_database):
    """`alembic check` against a database at head: no operation left un-migrated.

    This is what catches a model widened without a migration, or a migration that widened
    something the models did not — the two halves of R9-T8A disagreeing with each other.
    """
    checked = alembic(["check"], disposable_database)

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "No new upgrade operations detected" in checked.stdout


@pytest.mark.parametrize("verdict", (*V4_VERDICTS, None))
def test_downgrade_succeeds_when_every_stored_verdict_fits_the_old_width(
    database_at_the_widening, verdict
):
    """A database that never stored a v5 verdict can still go back, as it always could.

    The guard has to be conditional rather than a blanket refusal: a deployment that
    upgraded and decided against R9 before anything wrote a long verdict has lost nothing,
    and a migration that could not be reversed in that state would be a one-way door bolted
    on for a risk that had not materialised.
    """
    insert_analysis(database_at_the_widening, verdict)

    downgraded = alembic(["downgrade", "-1"], database_at_the_widening)

    assert downgraded.returncode == 0, downgraded.stderr
    assert current_revision(database_at_the_widening) == PREVIOUS_REVISION
    assert stored_width(database_at_the_widening, "analyses") == LEGACY_WIDTH
    assert stored_width(database_at_the_widening, "analysis_signals") == LEGACY_WIDTH


@pytest.mark.parametrize("verdict", V5_VERDICTS_TOO_LONG_FOR_THE_OLD_COLUMN)
def test_downgrade_refuses_when_a_stored_verdict_is_too_long(
    database_at_the_widening, verdict
):
    """The refusal, which is the whole point of the conditional downgrade.

    Three things are asserted and all three are load-bearing. It fails — a downgrade that
    "succeeded" here would have truncated a forensic verdict. The revision is unchanged —
    so the failure is not a half-applied downgrade the operator now has to reason about.
    And both columns are still wide — no DDL was emitted at all, because the check runs
    before any of it.
    """
    insert_analysis(database_at_the_widening, verdict)

    downgraded = alembic(["downgrade", "-1"], database_at_the_widening)

    assert downgraded.returncode != 0
    assert current_revision(database_at_the_widening) == THIS_REVISION
    assert stored_width(database_at_the_widening, "analyses") == WIDTH
    assert stored_width(database_at_the_widening, "analysis_signals") == WIDTH


def test_the_refusal_says_what_is_in_the_way(database_at_the_widening):
    """The operator is told which table, how many rows, and which verdict.

    Asserted because the reason this check exists at all is legibility: PostgreSQL already
    refuses the narrowing on its own, and an error that did not name the offending value
    would be no more use than the one the server gives.
    """
    insert_analysis(database_at_the_widening, VERDICT_NO_SIGNAL)

    downgraded = alembic(["downgrade", "-1"], database_at_the_widening)
    reported = downgraded.stdout + downgraded.stderr

    assert downgraded.returncode != 0
    assert "analyses.risk_level" in reported
    assert VERDICT_NO_SIGNAL in reported
    assert str(LEGACY_WIDTH) in reported


def test_the_guard_covers_the_signal_column_too(database_at_the_widening):
    """A long value in `analysis_signals.risk_level` blocks the downgrade as well.

    Nothing writes that column today, which is exactly why it is worth an assertion: a
    guard that only looked at `analyses` would pass every test above and truncate silently
    on the first database where the other column had been put to use.
    """
    analysis_id = insert_analysis(database_at_the_widening, None)
    target = create_engine(database_at_the_widening)

    try:
        with target.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO analysis_signals "
                    "(id, analysis_id, provider, signal_type, status, risk_level, created_at) "
                    "VALUES (:id, :analysis_id, :provider, :signal_type, :status, "
                    ":risk_level, now())"
                ),
                {
                    "id": uuid.uuid4(),
                    "analysis_id": analysis_id,
                    "provider": NVIDIA_PROVIDER,
                    "signal_type": SYNTHETIC_VIDEO_SIGNAL,
                    "status": SIGNAL_STATUS_SUCCESS,
                    "risk_level": VERDICT_MANIPULATION_DETECTED,
                },
            )
    finally:
        target.dispose()

    downgraded = alembic(["downgrade", "-1"], database_at_the_widening)
    reported = downgraded.stdout + downgraded.stderr

    assert downgraded.returncode != 0
    assert "analysis_signals.risk_level" in reported
    assert current_revision(database_at_the_widening) == THIS_REVISION
    assert stored_width(database_at_the_widening, "analyses") == WIDTH
