"""The command that asks for deferred Deep Evidence, and everything it may not do.

`POST /api/v1/analyses/{id}/enrichment` exists so a Quick Scan can become a full report
later. Its whole contract is that it creates *execution* work and touches nothing else: the
verdict, its trace, its coverage and the evidence it was taken from are already persisted
and stay exactly as they are.

Run against real PostgreSQL, through the production app, with a real session per request.
Not a fake session, for two reasons this file is largely about:

- the idempotency is a property of one conditional `UPDATE ... WHERE status =
  'not_requested'`, and a fake session that records statements proves nothing about what
  two of them do to one row;
- the concurrency test needs two connections genuinely racing, which is the only way to
  show that a second request cannot duplicate a component's task.

Ownership is real too. Nothing here overrides `require_user`: the accounts sign in through
the login route and present cookies the way a browser does, because "another user's
analysis is a 404" is a claim about the application's own resolution of identity.
"""

import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import enrichment, risk_engine
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_QUEUED,
    ENRICHMENT_TASK_STATUS_ABSTAINED,
    ENRICHMENT_TASK_STATUS_COMPLETED,
    ENRICHMENT_TASK_STATUS_FAILED,
    ENRICHMENT_TASK_STATUS_NOT_REQUESTED,
    ENRICHMENT_TASK_STATUS_PROCESSING,
    ENRICHMENT_TASK_STATUS_QUEUED,
    PRODUCT_MODE_DEEP_ANALYSIS,
    PRODUCT_MODE_QUICK_SCAN,
    SIGNAL_STATUS_SUCCESS,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    AnalysisEnrichmentTask,
    AnalysisJob,
    AnalysisSignal,
    AuthSession,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine, get_session
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

RULES_V5 = "r9-v5.0.0"

# A test fixture, not a credential.
PASSWORD = "correct-horse-battery-staple"

# Written out rather than read back from a response, so a change to either has to be made
# here deliberately.
NOT_FOUND_BODY = {"detail": "analysis not found"}
NO_VERDICT_DETAIL = "this analysis has no verdict, so there is no enrichment to request"
LEGACY_DETAIL = (
    "this analysis ran before Deep Evidence was a separate stage and carries no enrichment "
    "execution record"
)

# The five columns §6.1 freezes, plus the trace the report renders them through. Read as one
# tuple before and after every request in this file.
DECISION_FIELDS = (
    "status",
    "risk_level",
    "risk_rule_id",
    "risk_rules_version",
    "risk_calibration_id",
)


@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    """`TestClient` speaks plain HTTP and a browser's jar discards a `Secure` cookie sent
    over it, so without this every sign-in below would succeed and the next request would
    arrive with no cookie at all."""
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "development")


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
def records(database):
    """Everything this test created, removed in constraint order afterwards.

    Analyses are deleted before the accounts that own them: both ownership columns are
    `ON DELETE RESTRICT`, deliberately, so a forensic record outlives the credential that
    made it — and a missed analysis would block its owner and leak rows into the next test.
    Tasks, signals, media and jobs go with the analysis through `ON DELETE CASCADE`.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []

    yield users, analyses

    with SessionLocal() as db:
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.flush()
        for user_id in users:
            db.query(Analysis).filter(Analysis.owner_id == user_id).delete()
        db.flush()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


@pytest.fixture(autouse=True)
def live(database):
    """The production app, given a real session per request.

    A session per request rather than one shared session, which is what the application
    gets in production and what makes the concurrency test below mean anything: two
    requests racing through one `Session` would be serialized by that object rather than by
    the database, and the race the endpoint has to survive would never happen.
    """

    def session_per_request():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = session_per_request

    yield

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def immediate_enrichment(monkeypatch):
    """A deployment that queues enrichment with the decision, unless a mode says otherwise.

    Set explicitly so these tests read the same whatever the environment they run in has
    configured: what is under test is the mode and the command, not the policy.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)


def make_user(records, *, role: str = USER_ROLE_USER) -> User:
    users, _ = records

    with SessionLocal() as db:
        user = User(
            email=f"{uuid.uuid4().hex}@example.com",
            password_hash=hash_password(PASSWORD),
            role=role,
        )
        db.add(user)
        db.commit()
        users.append(user.id)

        return user


def make_analysis(
    records,
    owner: User,
    *,
    status: str = ANALYSIS_STATUS_COMPLETED,
    decided: bool = True,
    mode: str | None = PRODUCT_MODE_QUICK_SCAN,
    enqueue: bool = True,
) -> uuid.UUID:
    """A decided analysis with its media, its job, and the evidence the verdict rests on.

    The two deciding detectors are written as the worker writes them — `SUCCESS`, with the
    exact provider deployments the calibration was measured against — so the report route
    builds a real RiskTrace with real decision coverage over this row. That is what makes
    "the coverage did not move" an assertion about something rather than about two nulls.

    The enrichment task rows are created by `enrichment.stage`, never by hand: the
    membership, the frozen ruleset version and the initial status are R10-T2's to decide,
    and a fixture that wrote its own rows would be testing the fixture.

    They are staged in the *same transaction* as the analysis, which is how production
    writes them — `app.worker.conclude_job` stages them beside the verdict — so no fixture
    here can produce the decided-but-recordless state the endpoint used to be able to see.
    """
    _, analyses = records

    with SessionLocal() as db:
        analysis = Analysis(
            status=status,
            owner_id=owner.id,
            risk_level="NO_CALIBRATED_MANIPULATION_SIGNAL" if decided else None,
            risk_rules_version=RULES_V5 if decided else None,
            risk_rule_id=risk_engine.RULE_V5_NO_SIGNAL if decided else None,
            risk_calibration_id=risk_engine.CALIBRATION_ID if decided else None,
        )
        db.add(analysis)
        db.flush()
        analyses.append(analysis.id)

        digest = uuid.uuid4().hex + uuid.uuid4().hex
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
        db.add(
            AnalysisJob(
                analysis_id=analysis.id,
                status="completed",
                enrichment_mode=mode,
            )
        )

        if decided:
            db.add(
                AnalysisSignal(
                    analysis_id=analysis.id,
                    provider=risk_engine.SVD_PROVIDER,
                    signal_type=risk_engine.SVD_SIGNAL_TYPE,
                    status=SIGNAL_STATUS_SUCCESS,
                    score=0.1234,
                    provider_version=risk_engine.SVD_PROVIDER_VERSION,
                    signal_metadata={"total_clips": 7, "scored_clips": 7},
                )
            )
            db.add(
                AnalysisSignal(
                    analysis_id=analysis.id,
                    provider=risk_engine.FACE_PROVIDER,
                    signal_type=risk_engine.FACE_SIGNAL_TYPE,
                    status=SIGNAL_STATUS_SUCCESS,
                    score=0.0456,
                    provider_version=risk_engine.FACE_PROVIDER_VERSION,
                    signal_metadata={"frames_scored": 6},
                )
            )

        if enqueue and decided:
            enrichment.stage(db, analysis.id, RULES_V5, mode)

        db.commit()
        analysis_id = analysis.id

    return analysis_id


def set_component_states(analysis_id: uuid.UUID, *states: str) -> None:
    """Put this analysis's components into the terminal or in-flight states of a fixture.

    Written directly, because these are outcomes the *worker* produces and no API creates.
    They are assigned in a stable component order so a test naming three states names the
    same three components on every run.
    """
    with SessionLocal() as db:
        tasks = (
            db.query(AnalysisEnrichmentTask)
            .filter(AnalysisEnrichmentTask.analysis_id == analysis_id)
            .order_by(
                AnalysisEnrichmentTask.provider, AnalysisEnrichmentTask.signal_type
            )
            .all()
        )

        assert len(states) == len(tasks), (
            f"the fixture names {len(states)} states for {len(tasks)} components"
        )

        for task, state in zip(tasks, states):
            task.status = state

        db.commit()


def task_rows(analysis_id: uuid.UUID) -> list[tuple[str, str, str]]:
    with SessionLocal() as db:
        return [
            (task.provider, task.signal_type, task.status)
            for task in db.query(AnalysisEnrichmentTask)
            .filter(AnalysisEnrichmentTask.analysis_id == analysis_id)
            .order_by(
                AnalysisEnrichmentTask.provider, AnalysisEnrichmentTask.signal_type
            )
            .all()
        ]


def states_over(analysis_id: uuid.UUID, *leading: str, rest: str) -> tuple[str, ...]:
    """`leading` states for the first components, `rest` for every one after them.

    Sized from the rows rather than written out, because how many components an analysis
    has is the frozen ruleset's answer and a deployment may have a detector switched off.
    A test that hard-coded four would fail on a deployment running three, for a reason that
    has nothing to do with what it is testing.
    """
    components = len(task_rows(analysis_id))

    assert components > len(leading), (
        f"this analysis has {components} components; the fixture names {len(leading)}"
    )

    return tuple(leading) + (rest,) * (components - len(leading))


def signed_in(user: User) -> TestClient:
    """A client holding a real session for this account, opened through the login route."""
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


def ask(client: TestClient, analysis_id) -> object:
    """The command, made the way the dashboard would make it."""
    return client.post(
        f"/api/v1/analyses/{analysis_id}/enrichment",
        headers={"Origin": DASHBOARD_ORIGIN},
    )


def report(client: TestClient, analysis_id) -> dict:
    response = client.get(f"/api/v1/analyses/{analysis_id}")

    assert response.status_code == 200

    return response.json()


def decision_of(payload: dict) -> dict:
    """Everything about the automated decision that this command may not move."""
    return {
        **{field: payload[field] for field in DECISION_FIELDS},
        "risk_trace": payload["risk_trace"],
        "synthetic_video": payload["synthetic_video"],
        "face_manipulation": payload["face_manipulation"],
    }


# --------------------------------------------------------------------------------------
# The command does what it is for
# --------------------------------------------------------------------------------------


def test_a_deferred_analysis_can_be_asked_for_its_deep_evidence(records):
    """`ENRICHMENT_NOT_REQUESTED` becomes owed, which is the whole point of the endpoint."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    client = signed_in(owner)

    before = report(client, analysis_id)
    assert before["aggregate_enrichment_state"] == enrichment.ENRICHMENT_NOT_REQUESTED

    response = ask(client, analysis_id)

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_id"] == str(analysis_id)
    assert body["decision_state"] == enrichment.DECIDED
    assert body["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PENDING
    assert body["components_queued"] == len(task_rows(analysis_id))
    assert {state for _, _, state in task_rows(analysis_id)} == {
        ENRICHMENT_TASK_STATUS_QUEUED
    }


def test_the_projection_the_report_reads_moves_with_it(records):
    """The frontend is told by the API, so the API's own read has to agree with the command."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    client = signed_in(owner)

    commanded = ask(client, analysis_id).json()
    after = report(client, analysis_id)

    assert after["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PENDING
    assert after["aggregate_enrichment_state"] == commanded["aggregate_enrichment_state"]
    assert after["decision_state"] == commanded["decision_state"]
    assert after["per_component_state"] == commanded["per_component_state"]
    assert {component["state"] for component in after["per_component_state"]} == {
        ENRICHMENT_TASK_STATUS_QUEUED
    }


# --------------------------------------------------------------------------------------
# Idempotency, including under concurrency
# --------------------------------------------------------------------------------------


def test_asking_twice_queues_nothing_the_second_time(records):
    """The repeat is a success that created nothing, and says so with a zero."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    client = signed_in(owner)

    first = ask(client, analysis_id).json()
    rows_after_first = task_rows(analysis_id)

    second = ask(client, analysis_id)

    assert second.status_code == 200
    body = second.json()
    assert body["components_queued"] == 0
    assert first["components_queued"] > 0
    # No second set of tasks, and no component moved by the repeat.
    assert task_rows(analysis_id) == rows_after_first
    assert body["aggregate_enrichment_state"] == first["aggregate_enrichment_state"]
    assert body["per_component_state"] == first["per_component_state"]


def test_concurrent_requests_cannot_duplicate_a_components_task(records):
    """Two requests racing on one analysis, through two connections.

    The guarantee is not "the second one found the first one's work". It is that the
    transition is one conditional `UPDATE`, so the database decides the outcome: the
    components are queued exactly once between them, and no request inserts a task row at
    all. A read-then-insert endpoint would pass every other test in this file and fail this
    one.

    Two accounts rather than two clients for one account, and not for variety: a successful
    sign-in revokes whatever session that account already had (`app.web_auth.start_session`),
    so a second sign-in for the owner would quietly invalidate the first client's cookie and
    the "race" would be one request and one 401. An owner and an administrator both hold
    valid sessions over this analysis at the same time, which is also the real shape of this
    collision — the person and the operator pressing the same button.
    """
    owner = make_user(records)
    administrator = make_user(records, role=USER_ROLE_ADMIN)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    components = len(task_rows(analysis_id))
    assert components > 1, "the fixture must have several components to race over"

    clients = [signed_in(owner), signed_in(administrator)]
    start = threading.Barrier(len(clients))
    results: list[object] = [None] * len(clients)

    def run(index):
        start.wait(timeout=10)
        results[index] = ask(clients[index], analysis_id)

    threads = [
        threading.Thread(target=run, args=(index,)) for index in range(len(clients))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(result is not None for result in results), "a request never returned"
    assert [result.status_code for result in results] == [200, 200]

    rows = task_rows(analysis_id)
    # One row per component still, and every one of them queued exactly once.
    assert len(rows) == components
    assert len({(provider, signal) for provider, signal, _ in rows}) == components
    assert {state for _, _, state in rows} == {ENRICHMENT_TASK_STATUS_QUEUED}

    # And the work was claimed by one of the two, not by both.
    assert sum(result.json()["components_queued"] for result in results) == components


def test_six_concurrent_transitions_queue_each_component_exactly_once(records):
    """The same race at the orchestration level, wider and without HTTP in the way.

    Six connections call the transition together. Whatever interleaving PostgreSQL chooses,
    the component rows are the same rows — one per component, none inserted, none
    duplicated — and the sum of what the callers report queueing is exactly the number of
    components: every component was moved once and no caller was told it had moved a
    component that another one moved.

    Two requests through the endpoint prove the contract; this proves the mechanism under
    more pressure than two threads can apply, which is where a read-then-act implementation
    would come apart.
    """
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    components = len(task_rows(analysis_id))
    assert components > 1

    racers = 6
    start = threading.Barrier(racers)
    queued: list[int] = [-1] * racers

    def run(index):
        # A session per thread, because a session is a connection and the race is between
        # connections. Sharing one would serialize them in the client and prove nothing.
        with SessionLocal() as db:
            start.wait(timeout=10)
            queued[index] = enrichment.request(db, analysis_id)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(racers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(count >= 0 for count in queued), "a caller never returned"
    assert sum(queued) == components, queued

    rows = task_rows(analysis_id)
    assert len(rows) == components
    assert len({(provider, signal) for provider, signal, _ in rows}) == components
    assert {state for _, _, state in rows} == {ENRICHMENT_TASK_STATUS_QUEUED}


def test_asking_while_it_is_pending_is_a_no_op(records):
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_DEEP_ANALYSIS)
    client = signed_in(owner)

    assert report(client, analysis_id)["aggregate_enrichment_state"] == (
        enrichment.ENRICHMENT_PENDING
    )
    before = task_rows(analysis_id)

    response = ask(client, analysis_id)

    assert response.status_code == 200
    assert response.json()["components_queued"] == 0
    assert response.json()["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PENDING
    assert task_rows(analysis_id) == before


def test_asking_while_it_is_processing_is_a_no_op(records):
    """A component a worker is holding is not re-asked, and its lease is not disturbed."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_DEEP_ANALYSIS)
    set_component_states(
        analysis_id,
        *states_over(
            analysis_id,
            ENRICHMENT_TASK_STATUS_PROCESSING,
            rest=ENRICHMENT_TASK_STATUS_QUEUED,
        ),
    )
    client = signed_in(owner)
    before = task_rows(analysis_id)

    response = ask(client, analysis_id)

    assert response.status_code == 200
    assert response.json()["components_queued"] == 0
    assert response.json()["aggregate_enrichment_state"] == (
        enrichment.ENRICHMENT_PROCESSING
    )
    assert task_rows(analysis_id) == before


def test_a_completed_enrichment_is_not_run_again(records):
    """The evidence is already there; asking for it again must not re-run a detector."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_DEEP_ANALYSIS)
    # One abstention among them, because an abstention is a success on this axis and the
    # aggregate below has to still read as complete.
    set_component_states(
        analysis_id,
        *states_over(
            analysis_id,
            ENRICHMENT_TASK_STATUS_ABSTAINED,
            rest=ENRICHMENT_TASK_STATUS_COMPLETED,
        ),
    )
    client = signed_in(owner)
    before = task_rows(analysis_id)

    response = ask(client, analysis_id)

    assert response.status_code == 200
    assert response.json()["components_queued"] == 0
    assert response.json()["aggregate_enrichment_state"] == (
        enrichment.ENRICHMENT_COMPLETE
    )
    assert task_rows(analysis_id) == before


# --------------------------------------------------------------------------------------
# Asking is not retrying
# --------------------------------------------------------------------------------------


def test_a_partial_enrichment_is_not_silently_retried(records):
    """The failed component stays failed.

    Asking for enrichment and re-running enrichment that already ran are two different
    requests, and only the first one exists. If this endpoint quietly re-queued failures,
    the only control a user has would silently be a retry button, and a detector would be
    re-run by somebody who asked for nothing of the sort.
    """
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_DEEP_ANALYSIS)
    set_component_states(
        analysis_id,
        *states_over(
            analysis_id,
            ENRICHMENT_TASK_STATUS_FAILED,
            ENRICHMENT_TASK_STATUS_ABSTAINED,
            rest=ENRICHMENT_TASK_STATUS_COMPLETED,
        ),
    )
    client = signed_in(owner)
    before = task_rows(analysis_id)

    response = ask(client, analysis_id)

    assert response.status_code == 200
    assert response.json()["components_queued"] == 0
    assert response.json()["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PARTIAL
    assert task_rows(analysis_id) == before
    assert ENRICHMENT_TASK_STATUS_FAILED in {state for _, _, state in task_rows(analysis_id)}


def test_a_wholly_failed_enrichment_is_not_silently_retried(records):
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_DEEP_ANALYSIS)
    set_component_states(
        analysis_id,
        *states_over(
            analysis_id,
            ENRICHMENT_TASK_STATUS_FAILED,
            rest=ENRICHMENT_TASK_STATUS_FAILED,
        ),
    )
    client = signed_in(owner)
    before = task_rows(analysis_id)

    response = ask(client, analysis_id)

    assert response.status_code == 200
    assert response.json()["components_queued"] == 0
    assert response.json()["aggregate_enrichment_state"] == enrichment.ENRICHMENT_FAILED
    assert task_rows(analysis_id) == before


# --------------------------------------------------------------------------------------
# The states that are refused
# --------------------------------------------------------------------------------------


def test_a_legacy_analysis_is_refused_and_not_upgraded(records):
    """§8.2: a pre-R10 analysis is never re-run to bring it into the new architecture.

    It has no execution record because none existed when it ran, and creating one now would
    manufacture a fact about how it was executed. The refusal names that rather than
    pretending something was scheduled.
    """
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=None, enqueue=False)
    client = signed_in(owner)

    assert report(client, analysis_id)["aggregate_enrichment_state"] == (
        enrichment.LEGACY_SINGLE_STAGE
    )
    before = report(client, analysis_id)

    response = ask(client, analysis_id)

    assert response.status_code == 409
    assert response.json() == {"detail": LEGACY_DETAIL}
    # Nothing was created, so it is still legacy rather than newly "not requested".
    assert task_rows(analysis_id) == []
    assert report(client, analysis_id) == before


def test_an_undecided_analysis_is_refused(records):
    """No verdict, so there is nothing for supplementary evidence to supplement."""
    owner = make_user(records)
    analysis_id = make_analysis(
        records, owner, status=ANALYSIS_STATUS_QUEUED, decided=False, enqueue=False
    )
    client = signed_in(owner)

    response = ask(client, analysis_id)

    assert response.status_code == 409
    assert response.json() == {"detail": NO_VERDICT_DETAIL}
    assert task_rows(analysis_id) == []


def test_an_analysis_whose_decision_failed_is_refused(records):
    owner = make_user(records)
    analysis_id = make_analysis(
        records, owner, status=ANALYSIS_STATUS_FAILED, decided=False, enqueue=False
    )
    client = signed_in(owner)

    response = ask(client, analysis_id)

    assert response.status_code == 409
    assert response.json() == {"detail": NO_VERDICT_DETAIL}
    assert task_rows(analysis_id) == []


# --------------------------------------------------------------------------------------
# Authorization: the same rule the reads use, and no shortcut for a command
# --------------------------------------------------------------------------------------


def test_another_users_analysis_is_the_same_404_the_read_routes_give(records):
    """A command must not confirm an id the reads conceal.

    The 404 is the whole answer: a 403 would establish that the analysis exists, which is
    the fact somebody guessing ids is trying to learn, and the read routes refuse to give
    it. Nothing is queued either — a refusal that created work would be worse than a leak.
    """
    owner = make_user(records)
    stranger = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    before = task_rows(analysis_id)

    response = ask(signed_in(stranger), analysis_id)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND_BODY
    assert task_rows(analysis_id) == before
    assert {state for _, _, state in task_rows(analysis_id)} == {
        ENRICHMENT_TASK_STATUS_NOT_REQUESTED
    }


def test_an_administrator_may_ask_for_anybodys_analysis(records):
    """The same widening the listing gives an operator, and not a wider one."""
    owner = make_user(records)
    administrator = make_user(records, role=USER_ROLE_ADMIN)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)

    response = ask(signed_in(administrator), analysis_id)

    assert response.status_code == 200
    assert response.json()["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PENDING


def test_an_unauthenticated_caller_is_refused_before_anything_is_queued(records):
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    before = task_rows(analysis_id)

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/analyses/{analysis_id}/enrichment",
            headers={"Origin": DASHBOARD_ORIGIN},
        )

    assert response.status_code == 401
    assert task_rows(analysis_id) == before


def test_a_cross_origin_request_is_refused_before_anything_is_queued(records):
    """The same CSRF boundary every other dashboard mutation carries."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    before = task_rows(analysis_id)

    response = signed_in(owner).post(
        f"/api/v1/analyses/{analysis_id}/enrichment",
        headers={"Origin": "https://not-this-deployment.example"},
    )

    assert response.status_code == 403
    assert task_rows(analysis_id) == before


def test_an_unknown_id_is_a_404(records):
    owner = make_user(records)

    response = ask(signed_in(owner), uuid.uuid4())

    assert response.status_code == 404
    assert response.json() == NOT_FOUND_BODY


# --------------------------------------------------------------------------------------
# The decision is untouched — the reason this endpoint is allowed to exist
# --------------------------------------------------------------------------------------


def test_the_decision_and_its_evidence_are_identical_across_the_request(records):
    """Invariant I3 from the read side: nothing about the verdict moves.

    The five frozen columns, the RiskTrace the report renders — its rule, its ruleset, its
    calibration and its decision coverage — and both deciding detectors' signals, compared
    as whole documents before and after. Comparing the parsed payloads rather than named
    fields is deliberate: a field added later is covered without this test being edited.
    """
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    client = signed_in(owner)

    before = report(client, analysis_id)
    assert before["risk_trace"] is not None, "the fixture must produce a real trace"
    assert before["risk_trace"]["decision_coverage"] is not None
    assert before["synthetic_video"] is not None and before["face_manipulation"] is not None

    assert ask(client, analysis_id).status_code == 200

    after = report(client, analysis_id)

    assert decision_of(after) == decision_of(before)
    # Said again field by field, so a failure names which one moved.
    for field in DECISION_FIELDS:
        assert after[field] == before[field], field
    assert after["risk_trace"]["decision_coverage"] == (
        before["risk_trace"]["decision_coverage"]
    )
    assert after["risk_trace"]["contributions"] == before["risk_trace"]["contributions"]

    # And the enrichment axis is the only thing that moved.
    assert before["aggregate_enrichment_state"] != after["aggregate_enrichment_state"]


def test_the_persisted_columns_themselves_are_untouched(records):
    """Read from the row rather than through the API, so no serializer can hide a write."""
    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    client = signed_in(owner)

    def persisted():
        with SessionLocal() as db:
            analysis = db.get(Analysis, analysis_id)
            signals = {
                (signal.provider, signal.signal_type): (
                    signal.status,
                    signal.score,
                    signal.provider_version,
                    signal.signal_metadata,
                )
                for signal in db.query(AnalysisSignal)
                .filter(AnalysisSignal.analysis_id == analysis_id)
                .all()
            }

            return (
                {field: getattr(analysis, field) for field in DECISION_FIELDS},
                signals,
            )

    before = persisted()

    assert ask(client, analysis_id).status_code == 200

    assert persisted() == before


def test_the_risk_engine_is_never_invoked(records, monkeypatch):
    """§7.2: not to decide, not to confirm, not to compare.

    Every entry point the engine exposes is replaced with one that fails the test if it is
    called, over the module the API would reach it through. A verdict is taken once, and a
    request for supplementary evidence is not an occasion to take it again.
    """
    called: list[str] = []

    for name in ("evaluate", "evaluate_v5"):
        def spy(*args, _name=name, **kwargs):
            called.append(_name)
            raise AssertionError(f"the enrichment request invoked risk_engine.{_name}")

        monkeypatch.setattr(risk_engine, name, spy)

    owner = make_user(records)
    analysis_id = make_analysis(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    client = signed_in(owner)

    assert ask(client, analysis_id).status_code == 200
    # The repeat and the refusals reach different branches; none of them may evaluate.
    assert ask(client, analysis_id).status_code == 200
    assert report(client, analysis_id)["aggregate_enrichment_state"] == (
        enrichment.ENRICHMENT_PENDING
    )

    assert called == []


def test_the_route_reaches_no_risk_engine_reference_and_names_no_detector():
    """The two absences the endpoint's contract rests on, read off the source.

    The route may not interpret component membership — that is the frozen ruleset's, read
    from the task rows — and it may not reach the engine. Both are asserted over the
    function body because both are absences, and an absence is not demonstrated by one
    request that happened not to hit the line.
    """
    import inspect

    from app.api import analyses as analyses_module

    body = inspect.getsource(analyses_module.request_enrichment)

    for forbidden in ("risk_engine", "evaluate_v5", "evaluate("):
        assert forbidden not in body, forbidden

    for detector in ("lipforensics", "lip_forensics", "aasist", "effort",
                     "active_speaker", "audio_authenticity", "face_forgery"):
        assert detector not in body, detector

    # It creates no rows of its own and rewrites no forensic table: the only mutation it
    # performs is the one R10-T2 owns.
    assert "enrichment.request(" in body
    assert "session.add" not in body
    assert "AnalysisSignal" not in body
    assert "AnalysisEnrichmentTask" not in body


# --------------------------------------------------------------------------------------
# The false-legacy window is gone (R10 follow-up)
# --------------------------------------------------------------------------------------


def claim_for_publication(records, owner: User, *, mode: str | None) -> object:
    """An analysis one statement short of being decided, and the claim that will decide it.

    Queued, job `processing`, both deciding detectors already committed — the state
    `app.worker.conclude_job` is called in. Nothing here stages enrichment rows: staging is
    what the publication itself now does, and this fixture exists to watch it do it.
    """
    from app import worker

    _, analyses = records

    with SessionLocal() as db:
        analysis = Analysis(status=ANALYSIS_STATUS_QUEUED, owner_id=owner.id)
        db.add(analysis)
        db.flush()
        analyses.append(analysis.id)

        digest = uuid.uuid4().hex + uuid.uuid4().hex
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
        db.add(
            AnalysisSignal(
                analysis_id=analysis.id,
                provider=risk_engine.SVD_PROVIDER,
                signal_type=risk_engine.SVD_SIGNAL_TYPE,
                status=SIGNAL_STATUS_SUCCESS,
                score=0.1234,
                provider_version=risk_engine.SVD_PROVIDER_VERSION,
                signal_metadata={"total_clips": 7, "scored_clips": 7},
            )
        )
        db.add(
            AnalysisSignal(
                analysis_id=analysis.id,
                provider=risk_engine.FACE_PROVIDER,
                signal_type=risk_engine.FACE_SIGNAL_TYPE,
                status=SIGNAL_STATUS_SUCCESS,
                score=0.0456,
                provider_version=risk_engine.FACE_PROVIDER_VERSION,
                signal_metadata={"frames_scored": 6},
            )
        )
        job = AnalysisJob(
            analysis_id=analysis.id, status="processing", enrichment_mode=mode
        )
        db.add(job)
        db.commit()

        return worker.ClaimedJob(
            job_id=job.id,
            analysis_id=analysis.id,
            original_storage_key="originals/unused",
            normalization_required=False,
            frame_rate=30.0,
            enrichment_mode=mode,
        )


def test_the_endpoint_never_answers_a_false_legacy_409_during_publication(records):
    """The window this endpoint used to have, hunted for through the endpoint itself.

    Deep Evidence rows are now staged in the transaction that publishes the verdict, so an
    analysis is never `DECIDED` with no execution record and the legacy branch below cannot
    fire for an R10 analysis. This drives the endpoint in a tight loop across a real
    publication and keeps every answer: a 409 is permitted only while the analysis has no
    verdict, and the legacy refusal must never appear at all.

    The run is asserted to have spanned the publication — a 409 for no verdict before it
    and a 200 after — so a pass cannot come from missing the moment entirely.
    """
    from app import worker

    owner = make_user(records)
    claimed = claim_for_publication(records, owner, mode=PRODUCT_MODE_QUICK_SCAN)
    analysis_id = claimed.analysis_id
    client = signed_in(owner)

    answers: list[tuple[int, str]] = []
    # The asker takes at least one answer *before* the publication starts, and keeps going
    # until it gets one from after it. A barrier would only have started the two together,
    # and an HTTP round trip is slower than a publication — the first draft of this test
    # sampled once, after the fact, and proved nothing.
    sampled = threading.Event()

    def keep_asking():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            response = ask(client, analysis_id)
            detail = (
                response.json().get("detail", "")
                if response.status_code >= 400
                else response.json()["aggregate_enrichment_state"]
            )
            answers.append((response.status_code, detail))
            sampled.set()
            # A 200 can only come from after the verdict landed, so it is the end of the run.
            if response.status_code == 200:
                return

    asker = threading.Thread(target=keep_asking)
    asker.start()

    assert sampled.wait(timeout=20), "the asker never made its first request"
    with SessionLocal() as session:
        decision = worker.conclude_job(session, claimed)
    asker.join(timeout=30)

    assert decision is not None
    assert answers, "the asker made no requests"

    # The refusal that used to be possible here, and must not be any more.
    assert LEGACY_DETAIL not in [detail for _, detail in answers], answers

    # Every 409 was the honest one: this analysis had no verdict yet.
    for code, detail in answers:
        assert code in (200, 409), (code, detail)
        if code == 409:
            assert detail == NO_VERDICT_DETAIL, detail

    # And the run really did cross the publication.
    assert (409, NO_VERDICT_DETAIL) in answers
    assert any(code == 200 for code, _ in answers)

    # One row per component afterwards, and no duplicate.
    rows = task_rows(analysis_id)
    assert len({(provider, signal) for provider, signal, _ in rows}) == len(rows)
    assert rows, "the publication staged no components"
