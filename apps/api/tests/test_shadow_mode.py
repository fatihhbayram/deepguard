"""Shadow mode, and the four things it is not allowed to touch (R6-T1).

Shadow mode runs an uncalibrated experimental workload on real traffic. Everything worth
testing about it is a negative: the observation it produces must not become evidence, must
not reach a customer through either API, must not reach the report, and must not be able to
move a risk band. Those are the four claims `app.shadow` makes, and they are made
structurally — a table nothing joins to, and a risk engine that refuses anything but its
three calibrated types.

Structural claims need structural tests, so this module works from both ends. The behavioural
tests run a real shadow workload against real PostgreSQL and then go looking for its
observation everywhere it is forbidden to be — including through the HTTP responses the
dashboard and a paying integration actually receive. The static tests read the source of every
customer-facing module and of the risk engine and assert that none of them so much as names
the shadow table, because that absence is the guarantee: a filter can be forgotten in one
query, and a table nobody selects cannot be.

The end-to-end proof that a completed analysis queues a shadow run *without waiting for it*
lives in `tests/test_worker.py`, where the fixtures that run a whole job already are.
"""

import tokenize
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import modal_client, shadow
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    SHADOW_RUN_STATUS_COMPLETED,
    SHADOW_RUN_STATUS_FAILED,
    SHADOW_RUN_STATUS_PROCESSING,
    SHADOW_RUN_STATUS_QUEUED,
    SIGNAL_STATUS_SUCCESS,
    Analysis,
    AnalysisSignal,
    ApiKey,
    MediaFile,
    ShadowRun,
    User,
    USER_ROLE_ADMIN,
)
from app.db.session import SessionLocal, engine, get_session
from app.main import app
from app.risk_engine import (
    SVD_PROVIDER,
    SVD_PROVIDER_VERSION,
    SVD_SIGNAL_TYPE,
    SvdEvidence,
    UncalibratedEvidence,
    evaluate,
)
from app.auth import generate_api_key
from app.web_auth import hash_password, require_user
from app.worker import (
    persisted_face_evidence,
    persisted_lip_evidence,
    persisted_svd_evidence,
)

# The API directory and the web application, for the tests that read source rather than run it.
API_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = API_ROOT.parent / "web"

# The names that would have to appear in a module for it to be able to read a shadow
# observation at all. Spelled as the table, the model and the module, because a reader could
# arrive at the rows through any of the three.
SHADOW_NAMES = ("shadow_runs", "ShadowRun", "app.shadow", "shadow_mode")

# A string the stub workload writes into its observation and nothing else in the system
# produces, so a response containing it can only have got it from `shadow_runs`.
OBSERVATION_MARKER = "observed"

# The score the production signal below carries. NVIDIA's calibrated deployment, comfortably
# under its operating point (0.9551), so the analysis lands on a band that a stray shadow
# reading could plausibly be imagined to move.
PRODUCTION_SCORE = 0.1648


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
    """A real session whose analyses and accounts are removed again afterwards."""
    analyses = []
    users = []
    keys = []

    with SessionLocal() as db:
        yield db, analyses, users, keys

        db.rollback()
        for analysis_id in analyses:
            # Media, signals and shadow runs all go with it through ON DELETE CASCADE —
            # which is itself worth noticing: the shadow table is a leaf hanging off the
            # analysis, so deleting an analysis takes its experiments with it.
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.flush()
        # Analyses first, then their owners: both ownership columns are ON DELETE RESTRICT.
        for user_id in users:
            db.query(User).filter(User.id == user_id).delete()
        for key_id in keys:
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    """Every test in this module means the *in-process* workload (R6-T2).

    Shadow execution acquired a second backend in R6-T2, chosen by `DEEPGUARD_SHADOW_MODAL`
    and written onto the row at enqueue time. This module's subject is unchanged by that —
    the four things a shadow observation may not touch — but several of its tests assert
    against `STUB_WORKLOAD` and its evidence, and a suite run on a machine that has Modal
    configured would otherwise be quietly testing the remote path instead. Autouse, because
    the alternative is remembering which of these tests is backend-sensitive.

    The remote backend has a module of its own: `tests/test_modal_shadow.py`.
    """
    monkeypatch.delenv(modal_client.MODAL_SHADOW_VARIABLE, raising=False)


@pytest.fixture
def shadow_mode(monkeypatch):
    """Turn shadow mode on for this test, the way a deployment would."""
    monkeypatch.setenv(shadow.SHADOW_MODE_VARIABLE, "true")


def store_completed_analysis(session, api_key=None) -> Analysis:
    """Persist a completed analysis with one calibrated production signal and its decision.

    Everything shadow mode must not disturb, in one row and one signal: a finished status, a
    recorded decision, and the forensic evidence that decision was taken from.
    """
    db, analyses, _, _ = session

    analysis = Analysis(
        status=ANALYSIS_STATUS_COMPLETED,
        api_key_id=api_key.id if api_key is not None else None,
    )
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    digest = uuid.uuid4().hex
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
            derivative_storage_key=f"originals/{digest}",
            derivative_sha256=None,
        )
    )
    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider=SVD_PROVIDER,
            signal_type=SVD_SIGNAL_TYPE,
            status=SIGNAL_STATUS_SUCCESS,
            provider_version=SVD_PROVIDER_VERSION,
            score=PRODUCTION_SCORE,
            signal_metadata={"total_clips": 7},
        )
    )

    # Committed before it is classified, exactly as the worker does it: `conclude_job` reads
    # the evidence back out of the database rather than classifying values still in flight,
    # and this session does not autoflush.
    db.commit()

    # The decision the worker would have recorded, written the way it writes it.
    decision = evaluate(persisted_svd_evidence(db, analysis.id))
    analysis.risk_level = decision.risk_level
    analysis.risk_rules_version = decision.rules_version
    analysis.risk_rule_id = decision.rule_id
    analysis.risk_calibration_id = decision.calibration_id
    db.commit()

    return analysis


@pytest.fixture
def analysed(session) -> Analysis:
    """A completed analysis submitted through the dashboard, owned by nobody."""
    return store_completed_analysis(session)


@pytest.fixture
def customer(session):
    """A B2B API key and the plaintext to authenticate the public API with."""
    db, _, _, keys = session
    generated = generate_api_key()

    key = ApiKey(name="shadow-isolation", key_hash=generated.key_hash, is_active=True)
    db.add(key)
    db.commit()
    keys.append(key.id)

    return generated.plaintext, key


@pytest.fixture
def customer_analysed(session, customer) -> Analysis:
    """A completed analysis a paying integration submitted and may read back."""
    _, key = customer
    return store_completed_analysis(session, api_key=key)


@pytest.fixture
def administrator(session):
    """A real administrator account, for the dashboard read."""
    db, _, users, _ = session

    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password("test-account-password"),
        role=USER_ROLE_ADMIN,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


@pytest.fixture
def reader(session, administrator):
    """A client over the product app bound to the live session, signed in."""
    db, _, _, _ = session
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[require_user] = lambda: administrator

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def read_run(run_id: uuid.UUID) -> ShadowRun:
    """The shadow run as the database holds it now, on a session of its own."""
    with SessionLocal() as db:
        return db.query(ShadowRun).filter(ShadowRun.id == run_id).one()


def runs_for(analysis_id: uuid.UUID) -> list[ShadowRun]:
    with SessionLocal() as db:
        return db.query(ShadowRun).filter(ShadowRun.analysis_id == analysis_id).all()


def production_state(analysis_id: uuid.UUID) -> dict:
    """Everything about an analysis that shadow mode is forbidden to change."""
    with SessionLocal() as db:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).one()
        signals = (
            db.query(AnalysisSignal)
            .filter(AnalysisSignal.analysis_id == analysis_id)
            .order_by(AnalysisSignal.provider, AnalysisSignal.signal_type)
            .all()
        )

        return {
            "status": analysis.status,
            "risk_level": analysis.risk_level,
            "risk_rules_version": analysis.risk_rules_version,
            "risk_rule_id": analysis.risk_rule_id,
            "risk_calibration_id": analysis.risk_calibration_id,
            "signals": [
                (s.provider, s.signal_type, s.status, s.score, s.provider_version,
                 s.signal_metadata)
                for s in signals
            ],
        }


def decision_for(analysis_id: uuid.UUID):
    """The verdict the worker's own readers and rules reach for this analysis, now.

    The real path, not a re-implementation: the three `persisted_*_evidence` readers followed
    by `evaluate`, which is exactly what `app.worker.conclude_job` does.
    """
    with SessionLocal() as db:
        return evaluate(
            persisted_svd_evidence(db, analysis_id),
            persisted_face_evidence(db, analysis_id),
            persisted_lip_evidence(db, analysis_id),
        )


# --- executing -------------------------------------------------------------------------


@pytest.mark.integration
def test_a_queued_workload_is_claimed_and_its_observation_recorded(analysed, shadow_mode):
    with SessionLocal() as db:
        assert shadow.enqueue(db, analysed.id) is True

    queued = runs_for(analysed.id)
    assert [run.status for run in queued] == [SHADOW_RUN_STATUS_QUEUED]
    # Queued is queued: enqueueing runs nothing, which is what lets a production job commit a
    # shadow run and walk away from it.
    assert queued[0].evidence is None

    with SessionLocal() as db:
        assert shadow.process_one(db) is True

    run = read_run(queued[0].id)
    assert run.status == SHADOW_RUN_STATUS_COMPLETED
    assert run.workload == shadow.STUB_WORKLOAD
    assert run.provider_version == shadow.STUB_WORKLOAD_VERSION
    assert run.evidence == {OBSERVATION_MARKER: True, "analysis_id": str(analysed.id)}
    # A finished run holds no lease. A deadline on a terminal row is a promise nobody is
    # keeping.
    assert run.lease_expires_at is None
    assert run.error_message is None


@pytest.mark.integration
def test_an_observation_is_written_to_the_shadow_table_and_to_nothing_else(
    analysed, shadow_mode
):
    before = production_state(analysed.id)

    with SessionLocal() as db:
        shadow.enqueue(db, analysed.id)
    with SessionLocal() as db:
        shadow.process_one(db)

    # The whole isolation claim, at the level of rows: the experiment ran, and the forensic
    # record and the decision taken from it are byte-for-byte what they were.
    assert runs_for(analysed.id)[0].status == SHADOW_RUN_STATUS_COMPLETED
    assert production_state(analysed.id) == before


@pytest.mark.integration
def test_a_deployment_that_did_not_ask_for_shadow_mode_queues_nothing(analysed, monkeypatch):
    monkeypatch.delenv(shadow.SHADOW_MODE_VARIABLE, raising=False)

    with SessionLocal() as db:
        assert shadow.enqueue(db, analysed.id) is False

    assert runs_for(analysed.id) == []


@pytest.mark.integration
@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_the_flag_means_what_it_says(analysed, monkeypatch, value):
    # A compose file that carries `DEEPGUARD_SHADOW_MODE=false` means false, which
    # "any non-empty value is on" would have got exactly backwards.
    monkeypatch.setenv(shadow.SHADOW_MODE_VARIABLE, value)

    with SessionLocal() as db:
        assert shadow.enqueue(db, analysed.id) is False

    assert runs_for(analysed.id) == []


@pytest.mark.integration
def test_the_same_workload_is_never_queued_twice_for_one_analysis(analysed, shadow_mode):
    with SessionLocal() as db:
        assert shadow.enqueue(db, analysed.id) is True
    with SessionLocal() as db:
        # Refused by the unique constraint, reported as "not queued", and raised at nobody:
        # a job concluded twice must not double the corpus a calibration would be measured on
        # and must not fail because it tried.
        assert shadow.enqueue(db, analysed.id) is False

    assert len(runs_for(analysed.id)) == 1


@pytest.mark.integration
def test_a_workload_that_fails_leaves_the_production_analysis_exactly_as_it_was(
    analysed, shadow_mode, monkeypatch
):
    before = production_state(analysed.id)
    before_decision = decision_for(analysed.id)

    def explode(_claimed):
        raise RuntimeError("the experimental model fell over")

    monkeypatch.setattr(shadow, "run_stub_workload", explode)

    with SessionLocal() as db:
        shadow.enqueue(db, analysed.id)
    with SessionLocal() as db:
        # True: there *was* a run to do. Whether it succeeded is a fact about the experiment.
        assert shadow.process_one(db) is True

    run = runs_for(analysed.id)[0]
    assert run.status == SHADOW_RUN_STATUS_FAILED
    # The class name and not the message, so an exception that quoted a credential could not
    # write it into a column.
    assert run.error_message == "RuntimeError"
    assert run.evidence is None
    assert run.lease_expires_at is None

    assert production_state(analysed.id) == before
    assert decision_for(analysed.id) == before_decision


@pytest.mark.integration
def test_nothing_propagates_out_of_shadow_execution(analysed, shadow_mode, monkeypatch):
    """A broken shadow subsystem is reported as "no work done", never as an exception.

    The worker loop calls this on an idle poll. If it could raise, a defect in an experiment
    would trip the loop's error backoff and slow down the claiming of real jobs — which is the
    one thing shadow mode is not allowed to cost.
    """
    def explode(*_args, **_kwargs):
        raise SQLAlchemyError("connection lost")

    monkeypatch.setattr(shadow, "recover_stale_runs", explode)

    with SessionLocal() as db:
        assert shadow.process_one(db) is False


@pytest.mark.integration
def test_a_run_whose_worker_died_is_recovered_rather_than_held_forever(analysed, shadow_mode):
    with SessionLocal() as db:
        run = ShadowRun(
            analysis_id=analysed.id,
            workload=shadow.STUB_WORKLOAD,
            status=SHADOW_RUN_STATUS_PROCESSING,
            lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        db.add(run)
        db.commit()
        run_id = run.id

    with SessionLocal() as db:
        assert shadow.recover_stale_runs(db) == 1

    recovered = read_run(run_id)
    assert recovered.status == SHADOW_RUN_STATUS_FAILED
    assert recovered.error_message == shadow.STALE_LEASE_ERROR
    assert recovered.lease_expires_at is None
    # Recovery is a fact about an experiment and about nothing else.
    assert production_state(analysed.id)["status"] == ANALYSIS_STATUS_COMPLETED


@pytest.mark.integration
def test_a_live_run_is_not_recovered(analysed, shadow_mode):
    with SessionLocal() as db:
        db.add(
            ShadowRun(
                analysis_id=analysed.id,
                workload=shadow.STUB_WORKLOAD,
                status=SHADOW_RUN_STATUS_PROCESSING,
                lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
        )
        db.commit()

    with SessionLocal() as db:
        assert shadow.recover_stale_runs(db) == 0

    assert runs_for(analysed.id)[0].status == SHADOW_RUN_STATUS_PROCESSING


# --- what a customer receives ----------------------------------------------------------


@pytest.mark.integration
def test_the_public_read_carries_no_shadow_data(customer, customer_analysed, shadow_mode):
    """What a paying integration receives, after an experiment ran against its analysis."""
    plaintext, _ = customer

    with SessionLocal() as db:
        shadow.enqueue(db, customer_analysed.id)
    with SessionLocal() as db:
        shadow.process_one(db)

    assert runs_for(customer_analysed.id)[0].status == SHADOW_RUN_STATUS_COMPLETED

    # No session override: the public route opens its own against the same database, which is
    # the arrangement a real customer's read runs under.
    with TestClient(app) as client:
        response = client.get(
            f"/api/public/v1/analyses/{customer_analysed.id}",
            headers={"Authorization": f"Bearer {plaintext}"},
        )

    assert response.status_code == 200
    body = response.json()
    # A real, complete, classified analysis came back — the isolation is being read off a
    # populated payload, not an empty one.
    assert body["id"] == str(customer_analysed.id)
    assert body["risk_level"] == "MEDIUM"
    assert [signal["provider"] for signal in body["signals"]] == [SVD_PROVIDER]
    assert_no_shadow_data(response.text)


@pytest.mark.integration
def test_the_dashboard_read_carries_no_shadow_data(analysed, shadow_mode, reader):
    with SessionLocal() as db:
        shadow.enqueue(db, analysed.id)
    with SessionLocal() as db:
        shadow.process_one(db)

    response = reader.get(f"/api/v1/analyses/{analysed.id}")

    assert response.status_code == 200
    body = response.json()
    # The analysis is genuinely there and genuinely complete — this is not a test that passed
    # because nothing came back.
    assert body["id"] == str(analysed.id)
    assert body["status"] == ANALYSIS_STATUS_COMPLETED
    assert_no_shadow_data(response.text)
    # The forensic evidence is served, unchanged, beside no experimental field at all.
    assert body["synthetic_video"]["score"] == PRODUCTION_SCORE
    assert body["risk_level"] == "MEDIUM"


@pytest.mark.integration
def test_the_dashboard_listing_carries_no_shadow_data(analysed, shadow_mode, reader):
    with SessionLocal() as db:
        shadow.enqueue(db, analysed.id)
    with SessionLocal() as db:
        shadow.process_one(db)

    response = reader.get("/api/v1/analyses")

    assert response.status_code == 200
    assert str(analysed.id) in response.text
    assert_no_shadow_data(response.text)


def assert_no_shadow_data(payload: str) -> None:
    """Nothing a shadow run produced, and nothing that names where it lives, is in here."""
    for name in (*SHADOW_NAMES, OBSERVATION_MARKER, shadow.STUB_WORKLOAD_VERSION):
        assert name not in payload, f"{name!r} reached a response"


# --- what the risk engine will accept ---------------------------------------------------


def test_the_risk_engine_refuses_a_shadow_observation():
    observation = shadow.ShadowObservation(provider_version="stub-1", evidence={"score": 1.0})

    # Every parameter, not just the first: a guard on one argument would leave the other two
    # as the way in.
    for keyword in ("svd", "face", "lip"):
        with pytest.raises(UncalibratedEvidence):
            evaluate(**{keyword: observation})


def test_the_risk_engine_refuses_a_type_that_merely_looks_like_evidence():
    """A subclass is refused too, which `isinstance` would have admitted.

    This is the shape the mistake would really take: an experimental reading wrapped in
    something that inherits the calibrated type to get through, and then scored against a
    threshold measured for a different detector entirely.
    """
    class ExperimentalEvidence(SvdEvidence):
        pass

    impostor = ExperimentalEvidence(
        provider=SVD_PROVIDER,
        signal_type=SVD_SIGNAL_TYPE,
        status=SIGNAL_STATUS_SUCCESS,
        provider_version=SVD_PROVIDER_VERSION,
        score=0.99,
        total_clips=7,
    )

    with pytest.raises(UncalibratedEvidence):
        evaluate(impostor)


def test_the_guard_names_the_type_and_never_the_value():
    observation = shadow.ShadowObservation(
        provider_version="stub-1", evidence={"secret": "do-not-log-me"}
    )

    with pytest.raises(UncalibratedEvidence) as raised:
        evaluate(observation)

    assert "ShadowObservation" in str(raised.value)
    assert "do-not-log-me" not in str(raised.value)


def test_absent_evidence_is_still_evidence():
    # None is what a missing signal looks like, and the rules decide on it. The guard must not
    # turn the most common input into an error.
    assert evaluate(None, None, None).rule_id == "R010"


@pytest.mark.integration
def test_the_verdict_is_identical_whether_a_shadow_run_succeeded_failed_or_never_existed(
    analysed, shadow_mode, monkeypatch
):
    """The requirement stated as one test, over all three states a shadow run can be in."""
    never_ran = decision_for(analysed.id)

    with SessionLocal() as db:
        shadow.enqueue(db, analysed.id)
    with SessionLocal() as db:
        shadow.process_one(db)

    assert runs_for(analysed.id)[0].status == SHADOW_RUN_STATUS_COMPLETED
    succeeded = decision_for(analysed.id)

    with SessionLocal() as db:
        db.query(ShadowRun).filter(ShadowRun.analysis_id == analysed.id).delete()
        db.commit()

    monkeypatch.setattr(
        shadow, "run_stub_workload", lambda _claimed: (_ for _ in ()).throw(RuntimeError())
    )
    with SessionLocal() as db:
        shadow.enqueue(db, analysed.id)
    with SessionLocal() as db:
        shadow.process_one(db)

    assert runs_for(analysed.id)[0].status == SHADOW_RUN_STATUS_FAILED
    failed = decision_for(analysed.id)

    assert never_ran == succeeded == failed
    # And it is a real verdict, not three identical UNKNOWNs from an analysis with no
    # evidence: this one was classified by a rule, from a calibrated signal.
    assert never_ran.risk_level == "MEDIUM"
    assert never_ran.rule_id == "R201"


# --- structural isolation ----------------------------------------------------------------


def executable_source(path: Path) -> str:
    """The module with its comments and string literals removed.

    These modules explain themselves at length, and several of them explain shadow mode
    precisely because they must never read it — `app.risk_engine`\'s guard says so in its own
    docstring. A raw substring search would therefore fail on the prose that documents the
    guarantee, which would teach the next reader to delete the explanation rather than to keep
    the isolation. Tokenizing and dropping comments and strings leaves what the module can
    actually execute, which is the thing under test.
    """
    dropped = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE}

    with path.open("rb") as handle:
        return " ".join(
            token.string
            for token in tokenize.tokenize(handle.readline)
            if token.type not in dropped
        )


def customer_facing_sources() -> list[Path]:
    """Every module that answers a customer, plus the risk engine.

    Named as a list rather than "everything under `app/`", because `app/worker.py` and
    `app/db/models.py` legitimately name the shadow table — one runs the experiments and the
    other declares where they are stored.
    """
    return [
        *sorted((API_ROOT / "app" / "api").rglob("*.py")),
        API_ROOT / "app" / "risk_engine.py",
        API_ROOT / "app" / "main.py",
    ]


@pytest.mark.parametrize(
    "source", customer_facing_sources(), ids=lambda path: path.name
)
def test_no_customer_facing_module_can_reach_the_shadow_table(source):
    """The isolation, asserted as the absence it is.

    A response cannot leak a table it never selects. This is deliberately a source-level
    assertion rather than a check of one payload: a payload test proves today's fields are
    clean, and this proves there is no query to add a field to.
    """
    code = executable_source(source)

    for name in SHADOW_NAMES:
        assert name not in code, f"{source.name} names {name!r}"


def test_the_risk_engine_reads_no_shadow_table():
    # Restated on its own rather than left inside the parametrization above, because this one
    # is the safety property the whole phase rests on and should fail by name.
    assert "ShadowRun" not in executable_source(API_ROOT / "app" / "risk_engine.py")


@pytest.mark.skipif(not WEB_ROOT.exists(), reason="the web application is not present")
def test_the_web_application_knows_nothing_about_shadow_mode():
    """The report is rendered from the dashboard payload, and neither knows this exists."""
    sources = [
        path
        for pattern in ("*.ts", "*.tsx")
        for path in (WEB_ROOT / "app").rglob(pattern)
    ]

    assert sources, "no web sources were found to check"

    for path in sources:
        text = path.read_text()
        for name in SHADOW_NAMES:
            assert name not in text, f"{path.name} names {name!r}"
