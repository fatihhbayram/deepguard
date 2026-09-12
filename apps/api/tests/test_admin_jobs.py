"""Who may read the detection queue, what a job row tells them, and what this route will not do.

The sibling of `test_admin_users.py` and written to its conventions: real PostgreSQL, real
sessions opened through the real login route, one client per account, and every row this file
creates removed in constraint order afterwards. The reasons are the same ones stated there —
a role check demonstrated against a stubbed session proves only that the stub agreed.

Four things are asserted, in the order they matter:

*Authorization.* `/api/v1/admin/jobs` is behind `require_admin`, so an anonymous caller gets
401 and a signed-in ordinary user gets 403.

*Exposure.* The listing carries the job row's operational fields and nothing else. Asserted
against the whole payload rather than against a named field, so a column added to
`AnalysisJob` later cannot arrive on the wire by having been added.

*The derived stale flag.* `is_stale` is the server's answer, computed against the database
clock in the statement that reads the rows. The three cases that distinguish it from "the
lease is in the past" are covered: a claimed job whose lease has expired is stale, a claimed
job whose lease is still live is not, and a terminal job whose lease happens to sit in the
past is not — status gates it, and a failed job is not something a worker is still holding.

*The absence of a retry.* R8-T3 authorizes no requeue, because re-running a job that already
wrote evidence would insert a second `AnalysisSignal` per provider. That is asserted as a
property of the application's route table rather than as a promise in a comment: no route
under this prefix accepts anything but GET, today or after somebody adds one.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import (
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PROCESSING,
    JOB_STATUS_QUEUED,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    AnalysisJob,
    AuthSession,
    User,
)
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password

pytestmark = pytest.mark.integration

JOBS_URL = "/api/v1/admin/jobs"

# A test fixture, not a credential: nothing outside this file uses it and no deployment ships
# it.
PASSWORD = "correct-horse-battery-staple"

FORBIDDEN_BODY = {"detail": "Insufficient permissions"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}

# The fields an administrator is shown, and the entire set of them. Written out here rather
# than read back off the first response, so that widening the contract has to be a deliberate
# edit to this line rather than something a test quietly accepts.
VISIBLE_FIELDS = {
    "id",
    "analysis_id",
    "status",
    "request_id",
    "lease_expires_at",
    "error_message",
    "created_at",
    "updated_at",
    "is_stale",
}

@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    """`TestClient` speaks plain HTTP, and a browser's jar discards a `Secure` cookie sent over
    it. Without this the sign-ins below would succeed and the very next request would arrive
    with no cookie at all — a failure appearing nowhere near its cause."""
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
def session(database):
    """A real session that removes every row this file created, in constraint order.

    Sessions before users, because `auth_sessions.user_id` is a foreign key. Analyses last:
    `analysis_jobs.analysis_id` cascades, so deleting the parent takes the job with it and
    there is no separate job cleanup to forget.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, analyses

        db.rollback()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.commit()


def make_user(session, *, role: str = USER_ROLE_USER) -> User:
    """A persisted account, registered for cleanup."""
    db, users, _ = session

    user = User(
        # Unique per call, so two runs against the same database cannot collide on the unique
        # index over the email.
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def make_job(
    session,
    *,
    status: str = JOB_STATUS_QUEUED,
    lease_expires_at: datetime | None = None,
    error_message: str | None = None,
    request_id: str | None = None,
    created_at: datetime | None = None,
) -> AnalysisJob:
    """A job and the analysis it belongs to, in whatever state the caller needs.

    `created_at` is settable because the ordering assertion needs two rows the database can
    tell apart, and rows inserted in the same transaction share the `now()` default — the
    timestamps would be identical and the test would be asserting the tiebreaker.
    """
    db, _, analyses = session

    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED)
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    job = AnalysisJob(
        analysis_id=analysis.id,
        status=status,
        lease_expires_at=lease_expires_at,
        error_message=error_message,
        request_id=request_id,
    )
    if created_at is not None:
        job.created_at = created_at

    db.add(job)
    db.commit()

    return job


def signed_in(user: User) -> TestClient:
    """A client holding a real session for this account, opened through the login route."""
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


def listed(client: TestClient) -> dict[str, dict]:
    """The listing, keyed by job id, so a test can find its own rows among everyone's."""
    response = client.get(JOBS_URL)

    assert response.status_code == 200

    return {row["id"]: row for row in response.json()}


def administrator(session) -> TestClient:
    """A signed-in administrator, which is what every assertion below the authorization
    section is made through."""
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


# --- authorization --------------------------------------------------------------------


def test_an_anonymous_caller_cannot_list_jobs(database):
    with TestClient(app) as client:
        response = client.get(JOBS_URL)

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_ordinary_user_cannot_list_jobs(session):
    """403 and not 401. The session is perfectly good; the role is not, and the status says
    which — telling a signed-in person to sign in again would send them round a loop that
    cannot fix anything."""
    response = signed_in(make_user(session)).get(JOBS_URL)

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


def test_an_administrator_can_list_jobs(session):
    job = make_job(session)

    rows = listed(administrator(session))

    assert str(job.id) in rows


# --- exposure -------------------------------------------------------------------------


def test_a_listed_job_carries_the_operational_fields_and_no_others(session):
    job = make_job(session, request_id="req-abc123")

    row = listed(administrator(session))[str(job.id)]

    assert set(row) == VISIBLE_FIELDS
    assert row["analysis_id"] == str(job.analysis_id)
    assert row["status"] == JOB_STATUS_QUEUED
    assert row["request_id"] == "req-abc123"


def test_a_failed_job_carries_the_workers_own_error_message(session):
    """Carried across unchanged. The text is what the process that failed wrote down, and a
    route that rephrased it would be answering "why did this fail" with its own guess."""
    job = make_job(
        session, status=JOB_STATUS_FAILED, error_message="RuntimeError: detector unreachable"
    )

    row = listed(administrator(session))[str(job.id)]

    assert row["status"] == JOB_STATUS_FAILED
    assert row["error_message"] == "RuntimeError: detector unreachable"


def test_a_job_that_has_not_failed_reports_no_error(session):
    """Null rather than an empty string, which would read as a failure with nothing to say."""
    job = make_job(session, status=JOB_STATUS_COMPLETED)

    assert listed(administrator(session))[str(job.id)]["error_message"] is None


def test_the_listing_is_newest_first(session):
    now = datetime.now(timezone.utc)
    older = make_job(session, created_at=now - timedelta(hours=2))
    newer = make_job(session, created_at=now - timedelta(hours=1))

    order = [row["id"] for row in administrator(session).get(JOBS_URL).json()]

    assert order.index(str(newer.id)) < order.index(str(older.id))


# --- the derived stale flag -----------------------------------------------------------


def test_a_claimed_job_whose_lease_has_expired_is_stale(session):
    job = make_job(
        session,
        status=JOB_STATUS_PROCESSING,
        lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )

    assert listed(administrator(session))[str(job.id)]["is_stale"] is True


def test_a_claimed_job_whose_lease_is_live_is_not_stale(session):
    job = make_job(
        session,
        status=JOB_STATUS_PROCESSING,
        lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert listed(administrator(session))[str(job.id)]["is_stale"] is False


def test_a_terminal_job_with_an_old_lease_is_not_stale(session):
    """Status gates the flag. A failed job is not work anybody is still holding, however long
    ago its lease ran out — labelling it stale would send an operator looking for a dead worker
    that finished reporting hours ago."""
    job = make_job(
        session,
        status=JOB_STATUS_FAILED,
        lease_expires_at=datetime.now(timezone.utc) - timedelta(hours=3),
        error_message="stale lease",
    )

    assert listed(administrator(session))[str(job.id)]["is_stale"] is False


def test_a_queued_job_is_not_stale(session):
    """Nothing has claimed it, so there is no lease to expire — and `NULL < now()` is null,
    which is why the expression states the null check rather than relying on the comparison."""
    job = make_job(session)

    assert listed(administrator(session))[str(job.id)]["is_stale"] is False


# --- the absence of a retry -----------------------------------------------------------


def test_no_job_route_accepts_a_mutation():
    """The constraint R8-T3 is built around, asserted against the application's own route
    table rather than against this file's memory of what was written.

    A retry is a POST or a PATCH, and there is no safe one to offer: re-running a job that
    already persisted evidence would write a second `AnalysisSignal` per provider. So the
    assertion is not "there is no route called retry" — a name is easy to avoid — but that
    nothing under this prefix answers anything except GET.
    """
    mutations = {
        (route.path, method)
        for route in app.routes
        if getattr(route, "path", "").startswith(JOBS_URL)
        for method in getattr(route, "methods", set())
        if method not in {"GET", "HEAD", "OPTIONS"}
    }

    assert mutations == set()


def test_posting_to_the_jobs_route_is_refused(session):
    """The same fact over HTTP, from a caller who is allowed to read the listing: an
    administrator is not a caller the route table makes an exception for."""
    response = administrator(session).post(JOBS_URL, json={})

    assert response.status_code == 405
