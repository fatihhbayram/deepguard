"""Who may read the operational summary, and what its numbers are counting.

The sibling of `test_admin_jobs.py` and written to its conventions: real PostgreSQL, real
sessions opened through the real login route, one client per account, and every row this file
creates removed in constraint order afterwards.

One thing here that the job tests do not have to deal with. `/api/v1/admin/analytics` returns
aggregates over *every* row in the deployment, so a test cannot find its own rows in the
response the way `listed()` does next door — there are no ids in an aggregate, and any other
row committed in the last seven days is inside the same counts. Asserting absolute numbers
would make this file pass alone and fail in a suite, which is the worst failure a test can
have.

So every counting assertion here is a **difference**: read the payload, commit known rows,
read it again, and assert what moved. That is a stronger assertion than an absolute total
anyway — it says this row landed in this bucket and in no other, which is the actual claim —
and it holds whatever else is in the database.

What is asserted, in the order it matters:

*Authorization.* Behind `require_admin`, so an anonymous caller gets 401 and a signed-in
ordinary user gets 403.

*Shape.* The payload is fully formed with the schema's whole vocabulary present as zeros, on
any database, including one where the window is empty. That is the empty-state guarantee: a
quiet week must render as zeros rather than as missing keys or a 500.

*Bucketing.* A row lands where its stored value says and nowhere else — in particular that a
null `risk_level` is counted under `UNDECIDED` and not under `UNKNOWN`.

*Detector semantics.* The two claims the constraint turns on: a `FAILED` signal counts as one
failure for its own provider, and an analysis that never got a signal from a provider adds
nothing to that provider's counts — a missing signal is not a failure.

*No scores.* `AnalysisSignal.score` must not reach the wire in any form, asserted against a
signal seeded with a distinctive score rather than against a field name, so a future payload
that renamed or nested it would still be caught.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.admin_analytics import (
    ACQUISITION_METHODS,
    ANALYSIS_STATUSES,
    JOB_STATUSES,
    RISK_LEVELS,
    RISK_UNDECIDED,
    SIGNAL_STATUSES,
    WINDOW_DAYS,
    WINDOW_LABEL,
)
from app.db.models import (
    ACQUISITION_METHOD_UPLOAD,
    ACQUISITION_METHOD_URL,
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_FAILED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    AnalysisJob,
    AnalysisSignal,
    AuthSession,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password

pytestmark = pytest.mark.integration

ANALYTICS_URL = "/api/v1/admin/analytics"

# A test fixture, not a credential: nothing outside this file uses it and no deployment ships
# it.
PASSWORD = "correct-horse-battery-staple"

FORBIDDEN_BODY = {"detail": "Insufficient permissions"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}

# The whole of what an administrator is shown, written out here rather than read back off the
# first response, so widening the contract has to be a deliberate edit to this line.
VISIBLE_FIELDS = {
    "window",
    "analyses_total",
    "analyses_by_status",
    "jobs_by_status",
    "risk_distribution",
    "acquisition",
    "detectors",
}

# A provider name no detector in this system uses, so the rows a test seeds under it cannot
# collide with real signals another test or a development database left behind. Unique per
# run for the same reason the fixture emails are.
def unique_provider() -> str:
    return f"test-provider-{uuid.uuid4().hex[:12]}"


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
    `analysis_jobs.analysis_id`, `analysis_signals.analysis_id` and `media_files.analysis_id`
    all cascade, so deleting the parent takes the children with it and there is no separate
    cleanup for any of them to forget.
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
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def make_analysis(
    session,
    *,
    status: str = ANALYSIS_STATUS_COMPLETED,
    risk_level: str | None = None,
    created_at: datetime | None = None,
) -> Analysis:
    """One analysis, in whatever state the caller needs, registered for cleanup.

    `created_at` is settable so the window itself can be tested — a row dated outside it must
    not be counted, and the only way to have such a row is to write the timestamp rather than
    take the column's `now()` default.
    """
    db, _, analyses = session

    analysis = Analysis(status=status, risk_level=risk_level)
    if created_at is not None:
        analysis.created_at = created_at

    db.add(analysis)
    db.commit()
    analyses.append(analysis.id)

    return analysis


def make_media(session, analysis: Analysis, *, acquisition_method: str | None) -> None:
    """The media row belonging to an analysis, which is what the acquisition counts read.

    Every other column is filler: this file asserts nothing about the media's properties, only
    about how it arrived, and the rest are `NOT NULL` and have to hold something.
    """
    db, _, _ = session

    db.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename="clip.mov",
            content_type="video/quicktime",
            size_bytes=4096,
            original_sha256=uuid.uuid4().hex + uuid.uuid4().hex,
            original_storage_key="originals/" + uuid.uuid4().hex,
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
            codec_name="h264",
            width=1920,
            height=1080,
            duration=12.34,
            frame_rate=30.0,
            pix_fmt="yuv420p",
            constant_frame_rate=True,
            was_normalized=False,
            acquisition_method=acquisition_method,
        )
    )
    db.commit()


def make_signal(
    session,
    analysis: Analysis,
    *,
    provider: str,
    status: str = SIGNAL_STATUS_SUCCESS,
    score: float | None = None,
) -> None:
    """One detector's persisted answer about one analysis."""
    db, _, _ = session

    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider=provider,
            signal_type="deepfake",
            status=status,
            score=score,
        )
    )
    db.commit()


def make_job(session, analysis: Analysis, *, status: str) -> None:
    """One unit of detection work against an analysis."""
    db, _, _ = session

    db.add(AnalysisJob(analysis_id=analysis.id, status=status))
    db.commit()


def signed_in(user: User) -> TestClient:
    """A client holding a real session for this account, opened through the login route."""
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


def administrator(session) -> TestClient:
    """A signed-in administrator, which is what every assertion below the authorization
    section is made through."""
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def read(client: TestClient) -> dict:
    """The analytics payload."""
    response = client.get(ANALYTICS_URL)

    assert response.status_code == 200

    return response.json()


# --- authorization --------------------------------------------------------------------


def test_an_anonymous_caller_cannot_read_analytics(database):
    with TestClient(app) as client:
        response = client.get(ANALYTICS_URL)

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_ordinary_user_cannot_read_analytics(session):
    """403 and not 401. The session is perfectly good; the role is not, and the status says
    which — telling a signed-in person to sign in again would send them round a loop that
    cannot fix anything."""
    response = signed_in(make_user(session)).get(ANALYTICS_URL)

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


def test_an_administrator_can_read_analytics(session):
    payload = read(administrator(session))

    assert set(payload) == VISIBLE_FIELDS
    assert payload["window"] == WINDOW_LABEL


# --- shape and the empty window -------------------------------------------------------


def test_every_known_category_is_present_even_at_zero(session):
    """The empty-state guarantee, asserted as a property of the payload rather than by
    emptying the database — which a shared PostgreSQL will not allow and which would prove
    less anyway.

    What must hold on a quiet week is that the schema's whole vocabulary is present with zeros
    in it. A key that vanishes when its count reaches zero is indistinguishable, to a reader,
    from a metric that was never wired up; this asserts it cannot vanish. Counts are asserted
    as non-negative integers rather than as any particular number, because other rows in the
    database are inside these totals and their values are not this test's business.
    """
    payload = read(administrator(session))

    for field, categories in (
        ("analyses_by_status", ANALYSIS_STATUSES),
        ("jobs_by_status", JOB_STATUSES),
        ("risk_distribution", RISK_LEVELS),
        ("acquisition", ACQUISITION_METHODS),
    ):
        for category in categories:
            assert category in payload[field], f"{field} lost its {category!r} key"
            assert isinstance(payload[field][category], int)
            assert payload[field][category] >= 0

    assert isinstance(payload["analyses_total"], int)
    assert payload["analyses_total"] >= 0


def test_a_provider_with_no_signals_this_week_is_absent_rather_than_zeroed(session):
    """The distinction the detector table turns on.

    A provider appears because it has rows. A detector nobody invoked is not listed at zero,
    because a row of zeros in a health table reads as "asked and never answered" — which is
    the opposite of what it would mean.
    """
    payload = read(administrator(session))

    assert unique_provider() not in payload["detectors"]


# --- bucketing ------------------------------------------------------------------------


def counts_before_and_after(session, seed) -> tuple[dict, dict]:
    """The payload either side of committing some rows.

    Every counting assertion in this file is a difference across this helper rather than an
    absolute number, for the reason the module docstring gives: these are deployment-wide
    aggregates and every other row in the database is inside them.
    """
    client = administrator(session)
    before = read(client)
    seed()

    return before, read(client)


def test_an_analysis_is_counted_under_its_stored_status(session):
    before, after = counts_before_and_after(
        session, lambda: make_analysis(session, status=ANALYSIS_STATUS_QUEUED)
    )

    assert after["analyses_total"] == before["analyses_total"] + 1
    assert (
        after["analyses_by_status"][ANALYSIS_STATUS_QUEUED]
        == before["analyses_by_status"][ANALYSIS_STATUS_QUEUED] + 1
    )
    # And in no other bucket.
    assert (
        after["analyses_by_status"][ANALYSIS_STATUS_COMPLETED]
        == before["analyses_by_status"][ANALYSIS_STATUS_COMPLETED]
    )


def test_an_analysis_with_no_risk_level_is_counted_as_undecided(session):
    """The constraint this route exists to get right.

    A null `risk_level` is the absence of a persisted decision. It is reported under its own
    name and specifically *not* under `UNKNOWN`, which is a decision the risk engine reached
    and wrote down — collapsing the two would turn "nobody has assessed this" into "we assessed
    it and could not tell".
    """
    before, after = counts_before_and_after(
        session, lambda: make_analysis(session, risk_level=None)
    )

    assert after["risk_distribution"][RISK_UNDECIDED] == (
        before["risk_distribution"][RISK_UNDECIDED] + 1
    )
    assert after["risk_distribution"]["UNKNOWN"] == before["risk_distribution"]["UNKNOWN"]


def test_an_analysis_with_a_risk_level_is_counted_under_it(session):
    before, after = counts_before_and_after(
        session, lambda: make_analysis(session, risk_level="HIGH")
    )

    assert after["risk_distribution"]["HIGH"] == before["risk_distribution"]["HIGH"] + 1
    assert after["risk_distribution"][RISK_UNDECIDED] == (
        before["risk_distribution"][RISK_UNDECIDED]
    )


def test_media_is_counted_under_the_method_it_arrived_by(session):
    def seed():
        make_media(
            session,
            make_analysis(session),
            acquisition_method=ACQUISITION_METHOD_URL,
        )

    before, after = counts_before_and_after(session, seed)

    assert after["acquisition"][ACQUISITION_METHOD_URL] == (
        before["acquisition"][ACQUISITION_METHOD_URL] + 1
    )
    assert after["acquisition"][ACQUISITION_METHOD_UPLOAD] == (
        before["acquisition"][ACQUISITION_METHOD_UPLOAD]
    )


def test_a_job_is_counted_under_its_stored_status(session):
    def seed():
        make_job(session, make_analysis(session), status=JOB_STATUS_FAILED)

    before, after = counts_before_and_after(session, seed)

    assert after["jobs_by_status"][JOB_STATUS_FAILED] == (
        before["jobs_by_status"][JOB_STATUS_FAILED] + 1
    )


def test_rows_older_than_the_window_are_not_counted(session):
    """The window is the point of the route, so it is asserted rather than assumed.

    A day clear of the boundary, not a second, so the test cannot fail on the round trip
    between the two reads.
    """
    outside = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS + 1)

    before, after = counts_before_and_after(
        session,
        lambda: make_analysis(session, risk_level="HIGH", created_at=outside),
    )

    assert after["analyses_total"] == before["analyses_total"]
    assert after["risk_distribution"]["HIGH"] == before["risk_distribution"]["HIGH"]


# --- detector health ------------------------------------------------------------------


def test_a_provider_appears_with_every_status_seeded_at_zero(session):
    """One SUCCESS makes the provider visible, and it is visible with its whole vocabulary.

    `FAILED: 0` is a reading; a missing `FAILED` key is a blank the reader has to interpret.
    All three statuses are seeded because `TIMEOUT` is a distinct forensic fact from `FAILED`
    and the table must not quietly collapse the two.
    """
    provider = unique_provider()
    analysis = make_analysis(session)
    make_signal(session, analysis, provider=provider, status=SIGNAL_STATUS_SUCCESS)

    payload = read(administrator(session))

    assert payload["detectors"][provider] == {
        SIGNAL_STATUS_SUCCESS: 1,
        SIGNAL_STATUS_FAILED: 0,
        **{
            status: 0
            for status in SIGNAL_STATUSES
            if status not in (SIGNAL_STATUS_SUCCESS, SIGNAL_STATUS_FAILED)
        },
    }


def test_a_failed_signal_counts_as_a_failure_for_its_own_provider_only(session):
    """A `FAILED` signal is an operational fact about the provider that wrote it, and it stays
    with that provider. The second detector in this test answered successfully about the same
    analysis and its counts are untouched — failures do not propagate across a shared
    analysis.
    """
    failing = unique_provider()
    healthy = unique_provider()
    analysis = make_analysis(session)

    make_signal(session, analysis, provider=failing, status=SIGNAL_STATUS_FAILED)
    make_signal(session, analysis, provider=healthy, status=SIGNAL_STATUS_SUCCESS)

    detectors = read(administrator(session))["detectors"]

    assert detectors[failing][SIGNAL_STATUS_FAILED] == 1
    assert detectors[failing][SIGNAL_STATUS_SUCCESS] == 0
    assert detectors[healthy][SIGNAL_STATUS_FAILED] == 0
    assert detectors[healthy][SIGNAL_STATUS_SUCCESS] == 1


def test_an_analysis_a_provider_never_answered_about_is_not_a_failure(session):
    """The constraint stated as an executable claim: **a missing signal is not a failure.**

    One provider is asked about two analyses and answers once. Nothing anywhere increments for
    the analysis it was never invoked on — its `FAILED` count stays zero, and its `SUCCESS`
    count is one rather than one-of-two, because there is no denominator in this table for a
    never-invoked provider to fall short of.
    """
    provider = unique_provider()
    answered = make_analysis(session)
    make_analysis(session)  # Committed in the window, and this provider is never asked.

    make_signal(session, answered, provider=provider, status=SIGNAL_STATUS_SUCCESS)

    tally = read(administrator(session))["detectors"][provider]

    assert tally == {
        SIGNAL_STATUS_SUCCESS: 1,
        **{status: 0 for status in SIGNAL_STATUSES if status != SIGNAL_STATUS_SUCCESS},
    }


def test_no_detector_score_reaches_the_payload(session):
    """Asserted against the value rather than against a field name.

    A test for `"score" not in payload` would pass the day somebody nested the number under
    another key or averaged it into a "confidence". The seeded score is a distinctive number
    and the claim is that it appears nowhere in the serialized body at all.
    """
    provider = unique_provider()
    make_signal(
        session,
        make_analysis(session),
        provider=provider,
        status=SIGNAL_STATUS_SUCCESS,
        score=0.917531,
    )

    body = administrator(session).get(ANALYTICS_URL).text

    assert "0.917531" not in body
    assert "score" not in body
