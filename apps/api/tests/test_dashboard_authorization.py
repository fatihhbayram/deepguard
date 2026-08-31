"""Who sees which analyses, proven against real PostgreSQL and real sign-ins.

Ownership is not a property of a route. It is a `WHERE` clause, and only a database that
actually holds one account's analyses alongside another's can show that the clause keeps
them apart — a fake session asked for an analysis hands back whichever row it was given, and
a test written on top of that passes no matter what the filter says. So everything here runs
against the database, through the production app, with sessions issued by the real login
route and presented the way a browser presents them.

Four kinds of analysis exist in this system and a signed-in USER may see exactly one of
them: their own. Another user's, an API key's, and one owned by nobody — everything stored
before there were accounts — are all equally out of reach, and all three are refused with
the same 404 as an id that names nothing at all. That uniformity is the point of the file:
a 403 would confirm the id is real, which is the fact somebody guessing ids is trying to
establish.

An ADMIN sees all four, because the internal dashboard is where an operator looks at the
whole system. That is the only thing the role changes here.

The public API is checked from this side too. Its isolation is P9's and must survive: a
browser cookie presented to it authenticates nothing, and an upload made with a key is owned
by the key and by no account.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app import media, storage
from app.auth import generate_api_key
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    SINGLE_OWNER_CONSTRAINT,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    ApiKey,
    AuthSession,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine, get_session
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

LISTING_URL = "/api/v1/analyses"
PUBLIC_URL = "/api/public/v1/analyses"

# A test fixture, not a credential: nothing outside this file uses it and no deployment
# ships it.
PASSWORD = "correct-horse-battery-staple"

# The one answer every unreachable analysis gets, written out rather than read back from the
# first response so that a change to it has to be made here deliberately.
NOT_FOUND_BODY = {"detail": "analysis not found"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}


@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    """`TestClient` speaks plain HTTP, and a browser's jar discards a `Secure` cookie sent
    over it. Without this the sign-ins below would succeed and the very next request would
    arrive with no cookie at all — a failure appearing nowhere near its cause."""
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
    """A real session that removes everything these tests created, in constraint order.

    Analyses go first and are found by their owner rather than by a list of ids, because
    some of them are committed by the upload route rather than by this file and are never
    named here. Both ownership columns are `ON DELETE RESTRICT` — deliberately, so a
    forensic record outlives the credential that made it — which means a single missed
    analysis would block the account or key behind it and leak three rows into the next
    test.
    """
    users: list[uuid.UUID] = []
    keys: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, keys, analyses

        db.rollback()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(Analysis).filter(Analysis.owner_id == user_id).delete()
        for key_id in keys:
            db.query(Analysis).filter(Analysis.api_key_id == key_id).delete()
        db.flush()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        for key_id in keys:
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()


class FakeMinio:
    """Stand-in for the object store, so an upload needs no live MinIO."""

    def __init__(self) -> None:
        self.uploads = []

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        pass

    def fput_object(self, bucket, key, file_path, content_type=None):
        self.uploads.append((bucket, key))


@pytest.fixture(autouse=True)
def fake_minio(monkeypatch):
    monkeypatch.setattr(storage, "client", FakeMinio())


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch):
    """Replace only the subprocess call, so real parsing and validation still run."""
    probe = json.dumps(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "12.34",
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "12.34",
                "tags": {"major_brand": "mp42"},
            },
        }
    )

    async def run(path):
        return probe

    monkeypatch.setattr(media, "_run_ffprobe", run)


def make_user(session, *, role: str = USER_ROLE_USER) -> User:
    """A persisted account, registered for cleanup."""
    db, users, _, _ = session

    user = User(
        # Unique per call, so two runs against the same database cannot collide on the
        # unique index over the email.
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def make_api_key(session) -> tuple[ApiKey, str]:
    """A persisted key and its one plaintext copy, registered for cleanup."""
    db, _, keys, _ = session
    generated = generate_api_key()

    key = ApiKey(name="dashboard-isolation", key_hash=generated.key_hash)
    db.add(key)
    db.commit()
    keys.append(key.id)

    return key, generated.plaintext


def make_analysis(
    session, *, owner: User | None = None, api_key: ApiKey | None = None
) -> Analysis:
    """An analysis and the media row every read joins onto, committed as the pipeline does.

    `owner` and `api_key` are both optional and never both given: an analysis owned by
    nobody is a real state — everything stored before there were accounts — and it is one of
    the four kinds this file is about.
    """
    db, _, _, analyses = session

    analysis = Analysis(
        status=ANALYSIS_STATUS_COMPLETED,
        owner_id=owner.id if owner is not None else None,
        api_key_id=api_key.id if api_key is not None else None,
    )
    db.add(analysis)
    db.flush()
    db.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename="clip.mov",
            content_type="video/quicktime",
            size_bytes=4096,
            original_sha256="a" * 64,
            original_storage_key="originals/" + "a" * 64,
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
    analyses.append(analysis.id)

    return analysis


@pytest.fixture
def live(session):
    """The production app bound to the live session, with no dependency faked but that one.

    Nothing here overrides `require_user`. The whole question this file asks is which
    account the application resolves and what it then shows them, and an override would
    answer the first half by hand.
    """
    db, _, _, _ = session
    app.dependency_overrides[get_session] = lambda: db

    yield

    app.dependency_overrides.clear()


def signed_in(user: User) -> TestClient:
    """A client holding a real session for this account, opened through the login route.

    A client of its own per account rather than one jar re-pointed between sign-ins: two
    sessions in one jar is a state no browser is in, and a test that got it wrong would look
    like an authorization failure.
    """
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


def listed_ids(client: TestClient) -> set[str]:
    response = client.get(LISTING_URL)

    assert response.status_code == 200

    return {row["id"] for row in response.json()}


def report(client: TestClient, analysis_id) -> object:
    return client.get(f"{LISTING_URL}/{analysis_id}")


# --- no session at all ---------------------------------------------------------------


def test_an_anonymous_caller_cannot_list_analyses(live):
    with TestClient(app) as client:
        response = client.get(LISTING_URL)

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_anonymous_caller_cannot_read_a_report(live, session):
    user = make_user(session)
    analysis = make_analysis(session, owner=user)

    with TestClient(app) as client:
        response = report(client, analysis.id)

    # 401 and not 404: the caller is not being told the analysis is missing, they are being
    # told to sign in. Answering 404 here would be indistinguishable from a real absence and
    # would send a signed-out operator looking for a record that is sitting right there.
    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


# --- one user's dashboard is their own ------------------------------------------------


def test_a_user_lists_only_their_own_analyses(live, session):
    """All four kinds exist at once, and exactly one of them comes back.

    Written as one test over all four rather than four tests over one each, because what has
    to hold is the whole of the listing: a filter that let any of the other three through
    would be a leak, and only a listing that contains them all can show which ones it drops.
    """
    alice = make_user(session)
    bob = make_user(session)
    key, _ = make_api_key(session)

    mine = make_analysis(session, owner=alice)
    theirs = make_analysis(session, owner=bob)
    customers = make_analysis(session, api_key=key)
    legacy = make_analysis(session)

    with signed_in(alice) as client:
        listed = listed_ids(client)

    assert str(mine.id) in listed
    assert listed.isdisjoint({str(theirs.id), str(customers.id), str(legacy.id)})


def test_a_user_cannot_read_another_users_analysis(live, session):
    alice = make_user(session)
    bob = make_user(session)
    hers = make_analysis(session, owner=alice)

    with signed_in(bob) as client:
        response = report(client, hers.id)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND_BODY


def test_a_user_cannot_read_an_api_key_owned_analysis(live, session):
    """A customer's analysis is not a dashboard analysis, and does not become one."""
    user = make_user(session)
    key, _ = make_api_key(session)
    customers = make_analysis(session, api_key=key)

    with signed_in(user) as client:
        response = report(client, customers.id)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND_BODY


def test_a_user_cannot_read_an_analysis_owned_by_nobody(live, session):
    """Everything the dashboard committed before there were accounts stays out of reach.

    The tempting reading of a null owner is "unowned, so anyone may see it", and it is
    exactly wrong: those rows were submitted when the dashboard authenticated nobody, so
    nothing records who they belong to, and handing them to whoever signs in first would be
    a guess about their ownership rather than a fact about it.
    """
    user = make_user(session)
    legacy = make_analysis(session)

    with signed_in(user) as client:
        response = report(client, legacy.id)

    assert response.status_code == 404
    assert response.json() == NOT_FOUND_BODY


def test_every_refusal_is_the_same_as_an_id_that_names_nothing(live, session):
    """Byte-identical, so the response cannot be used to tell a real id from an absent one.

    This is the whole reason these are 404s. If a cross-user read answered differently from
    a read of a made-up id — a different status, a different body, a different header — then
    a caller with a list of ids could sort them into "exists" and "does not", which is the
    inventory an attacker wants and the one thing the status code is chosen to withhold.
    """
    alice = make_user(session)
    bob = make_user(session)
    key, _ = make_api_key(session)

    unreachable = [
        make_analysis(session, owner=alice).id,
        make_analysis(session, api_key=key).id,
        make_analysis(session).id,
        # An id that names no row at all — the baseline the other three must match.
        uuid.uuid4(),
    ]

    with signed_in(bob) as client:
        responses = [report(client, analysis_id) for analysis_id in unreachable]

    signatures = {
        (response.status_code, response.text, response.headers.get("content-type"))
        for response in responses
    }
    assert len(signatures) == 1


# --- the administrator's view ---------------------------------------------------------


def test_an_admin_lists_every_kind_of_analysis(live, session):
    admin = make_user(session, role=USER_ROLE_ADMIN)
    user = make_user(session)
    key, _ = make_api_key(session)

    theirs = make_analysis(session, owner=user)
    customers = make_analysis(session, api_key=key)
    legacy = make_analysis(session)

    with signed_in(admin) as client:
        listed = listed_ids(client)

    assert {str(theirs.id), str(customers.id), str(legacy.id)} <= listed


def test_an_admin_reads_a_users_analysis(live, session):
    admin = make_user(session, role=USER_ROLE_ADMIN)
    user = make_user(session)
    theirs = make_analysis(session, owner=user)

    with signed_in(admin) as client:
        response = report(client, theirs.id)

    assert response.status_code == 200
    assert response.json()["id"] == str(theirs.id)


# --- what a submission records --------------------------------------------------------


def upload(client: TestClient, url: str = LISTING_URL, headers: dict | None = None):
    """One upload. The dashboard's carries a browser `Origin`; a customer's does not.

    That asymmetry is real and not tidiness: the dashboard route refuses a submission
    without an accepted origin, and the public route has no such check because a browser
    never attaches an `Authorization` header on its own — so a customer's server posts with
    no origin at all and must keep working.
    """
    return client.post(
        url,
        files={"file": ("clip.mp4", b"pretend-this-is-video", "video/mp4")},
        headers=headers if headers is not None else {"Origin": DASHBOARD_ORIGIN},
    )


def test_a_web_upload_is_owned_by_the_signed_in_account(live, session):
    """The account comes from the cookie and lands in `owner_id`, with no key beside it."""
    db, _, _, _ = session
    user = make_user(session)

    with signed_in(user) as client:
        response = upload(client)

    assert response.status_code == 202

    stored = db.get(Analysis, uuid.UUID(response.json()["id"]))
    assert stored.owner_id == user.id
    assert stored.api_key_id is None


def test_a_web_upload_appears_on_its_owners_dashboard_and_nobody_elses(live, session):
    """The round trip: what a submission records is what the next read filters on.

    Ownership written one way and read another would be two rules that happen to agree
    today, and this is the test that would notice on the day they stopped.
    """
    alice = make_user(session)
    bob = make_user(session)

    with signed_in(alice) as client:
        submitted = upload(client).json()["id"]
        assert submitted in listed_ids(client)

    with signed_in(bob) as client:
        assert submitted not in listed_ids(client)
        assert report(client, submitted).status_code == 404


def test_a_public_upload_is_owned_by_the_key_and_ignores_a_web_session(live, session):
    """P9's isolation, from this side: the public surface does not look at a cookie.

    The client here holds a genuine, working web session *and* presents a key. The analysis
    it creates must be owned by the key and by no account — if the cookie leaked into this
    path, a customer's submission would land on somebody's dashboard, and the mutual
    exclusion the database holds would make the write fail outright.
    """
    db, _, _, _ = session
    user = make_user(session)
    key, plaintext = make_api_key(session)

    with signed_in(user) as client:
        # No `Origin`, because a customer's server does not send one — which also shows the
        # dashboard's CSRF check is not quietly standing in front of this route.
        response = upload(
            client, url=PUBLIC_URL, headers={"Authorization": f"Bearer {plaintext}"}
        )

    assert response.status_code == 202

    stored = db.get(Analysis, uuid.UUID(response.json()["id"]))
    assert stored.api_key_id == key.id
    assert stored.owner_id is None


def test_a_public_read_is_not_opened_by_a_web_session(live, session):
    """And the other direction on the read path: a cookie is not a key.

    The analysis is the customer's own, so the only thing standing between this client and a
    200 is the credential — and the cookie it holds is not one the public surface accepts.
    """
    user = make_user(session)
    key, _ = make_api_key(session)
    customers = make_analysis(session, api_key=key)

    with signed_in(user) as client:
        response = client.get(f"{PUBLIC_URL}/{customers.id}")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


# --- the invariant underneath it all ---------------------------------------------------


def test_an_analysis_still_cannot_be_owned_by_both(live, session):
    """The check constraint that makes the two families exclusive, still biting.

    Everything above rests on it: a row naming a user *and* a key would appear on that
    user's dashboard and in that customer's public reads at once, and no `WHERE` clause on
    either side would be wrong. It is asserted here as well as in `test_web_auth.py` because
    this is the task that started writing the column the constraint guards.
    """
    db, _, _, _ = session
    user = make_user(session)
    key, _ = make_api_key(session)

    db.add(
        Analysis(
            status=ANALYSIS_STATUS_COMPLETED, owner_id=user.id, api_key_id=key.id
        )
    )

    with pytest.raises(IntegrityError) as raised:
        db.commit()

    assert SINGLE_OWNER_CONSTRAINT in str(raised.value)
    db.rollback()


# --- signing out ----------------------------------------------------------------------


def test_signing_out_closes_the_dashboard(live, session):
    """The session that read the listing a moment ago stops working, cookie and all.

    Through the real logout route, and checked against the dashboard rather than against
    `/auth/me`: what a person means by signing out is that this browser can no longer read
    their analyses.
    """
    user = make_user(session)
    make_analysis(session, owner=user)

    with signed_in(user) as client:
        assert client.get(LISTING_URL).status_code == 200

        assert (
            client.post(
                "/api/v1/auth/logout", headers={"Origin": DASHBOARD_ORIGIN}
            ).status_code
            == 204
        )

        after = client.get(LISTING_URL)

    assert after.status_code == 401
    assert after.json() == UNAUTHENTICATED_BODY
