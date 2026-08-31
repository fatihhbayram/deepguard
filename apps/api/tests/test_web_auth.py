"""Web sign-in: password hashing, opaque sessions, role checks and the isolation around them.

Three properties are what this file exists for, and each of them is a security property
rather than a behaviour:

1. Every sign-in failure is one failure. An unknown address, a wrong password and a
   deactivated account are indistinguishable in status, body and headers.
2. The plaintext session token is never persisted. Asserted against the real table, read as
   raw text, so a column added later that happened to capture it would fail here.
3. A web cookie authenticates nothing on the public B2B surface, and an API key
   authenticates nothing on the web surface. The two credential families do not overlap.

The pure functions are tested without a database. Everything about sessions needs one —
expiry is compared against PostgreSQL's clock, rotation is a transaction, and the check
constraint is the database's — so those live below the integration marker.

The two guarded routes are mounted on a throwaway app defined here. What they exist to prove
is the behaviour of `require_user` and `require_admin` themselves — including the 403 for a
role, which no product route asks for yet — and mounting them here keeps those tests from
depending on whatever the dashboard happens to require this month. The product routes that
do now demand a session are covered where they live, in
`tests/test_dashboard_authorization.py`.
"""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.auth import router as auth_router
from app.auth import generate_api_key
from app.db.models import (
    ANALYSIS_STATUS_QUEUED,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    ApiKey,
    AuthSession,
    User,
)
from app.db.session import SessionLocal, engine, get_session
from app.main import app as production_app
from app.web_auth import (
    ENVIRONMENT_VARIABLE,
    SESSION_COOKIE_NAME,
    SESSION_ENTROPY_BYTES,
    SESSION_LIFETIME,
    cookie_is_secure,
    generate_session_token,
    hash_password,
    hash_session_token,
    normalize_email,
    require_admin,
    require_user,
    verify_password,
)
from tests.conftest import DASHBOARD_ORIGIN

# A password long enough to be realistic and constant so the tests are not comparing against
# a value that changes between runs. It is a test fixture, not a credential: nothing outside
# this file uses it and no deployment ships it.
PASSWORD = "correct-horse-battery-staple"

# The two refusals, written out rather than read back from the first response, so a change
# to either has to be made here deliberately.
INVALID_CREDENTIALS_BODY = {"detail": "Invalid email or password"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}
FORBIDDEN_BODY = {"detail": "Insufficient permissions"}


@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    """Declare the plain-HTTP environment for every test in this module.

    `TestClient` speaks `http://testserver`, and a cookie jar discards a `Secure` cookie
    that arrives over plain HTTP — exactly as a browser does. Since the application now
    fails secure, a suite that inherited whatever `DEEPGUARD_ENV` happened to be set to
    would sign in successfully and then find itself unauthenticated on the very next
    request, with the failure appearing in the session tests rather than anywhere near its
    cause.

    So the environment is stated here rather than inherited. That is also the honest
    description of what this suite is: an HTTP client, which is what `development` names.
    The handful of tests that are *about* the flag override this with their own
    `monkeypatch` call, which runs after this fixture and therefore wins.
    """
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "development")


# --- normalization ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["alice@example.com", "Alice@Example.com", "  alice@example.com  ", "ALICE@EXAMPLE.COM"],
    ids=["plain", "mixed-case", "padded", "upper"],
)
def test_email_normalization_collapses_case_and_whitespace(raw):
    assert normalize_email(raw) == "alice@example.com"


def test_email_normalization_keeps_plus_addressing_and_dots():
    """Only case and whitespace. Dots and `+` tags are part of the address's identity.

    Stripping them is one mail provider's delivery convention, not a property of addresses,
    and applying it here would merge accounts belonging to different people.
    """
    assert normalize_email("Alice.B+deepguard@Example.com") == (
        "alice.b+deepguard@example.com"
    )


# --- password hashing ---------------------------------------------------------------------


def test_password_hash_is_argon2id():
    """The stored hash must be Argon2id specifically, not Argon2i or Argon2d.

    Read off the encoded hash's own identifier rather than trusted from the library's
    defaults, because those defaults are what a future upgrade could move.
    """
    assert hash_password(PASSWORD).startswith("$argon2id$")


def test_password_hash_does_not_contain_the_password():
    assert PASSWORD not in hash_password(PASSWORD)


def test_the_same_password_hashes_differently_every_time():
    """A per-hash salt, which is what stops two accounts sharing a password from showing it."""
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_the_correct_password_verifies():
    assert verify_password(hash_password(PASSWORD), PASSWORD) is True


@pytest.mark.parametrize(
    "attempt",
    [PASSWORD + "!", PASSWORD.upper(), PASSWORD[:-1], "", " " + PASSWORD],
    ids=["appended", "case-changed", "truncated", "empty", "padded"],
)
def test_a_wrong_password_does_not_verify(attempt):
    assert verify_password(hash_password(PASSWORD), attempt) is False


def test_an_unparseable_stored_hash_is_a_mismatch_not_a_crash():
    """A corrupted or hand-edited row answers "no", the way any other mismatch does."""
    assert verify_password("not-an-argon2-hash", PASSWORD) is False


# --- session tokens -----------------------------------------------------------------------


def test_session_token_has_at_least_256_bits_of_entropy():
    """Base64url carries 6 bits a character, so 256 bits needs at least 43 of them.

    Asserting the length is how an edit that shortens the token — the cheap way to make a
    cookie tidier — fails here rather than quietly weakening every session.
    """
    assert len(generate_session_token()) * 6 >= SESSION_ENTROPY_BYTES * 8


def test_session_tokens_are_unique():
    assert len({generate_session_token() for _ in range(100)}) == 100


def test_session_token_hash_is_sha256_and_hides_the_token():
    token = generate_session_token()
    digest = hash_session_token(token)

    assert digest == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    assert token not in digest


def test_session_token_hashing_is_deterministic_and_token_specific():
    first = generate_session_token()
    second = generate_session_token()

    assert hash_session_token(first) == hash_session_token(first)
    assert hash_session_token(first) != hash_session_token(second)


# --- the cookie's Secure flag ---------------------------------------------------------------


def test_the_cookie_is_secure_in_production(monkeypatch):
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "production")
    assert cookie_is_secure() is True

    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "PRODUCTION")
    assert cookie_is_secure() is True


@pytest.mark.parametrize(
    "value",
    ["development", "test", "DEVELOPMENT", "  test  "],
    ids=["development", "test", "upper", "padded"],
)
def test_the_cookie_is_not_secure_in_the_named_plain_http_environments(monkeypatch, value):
    """Local HTTP must keep working: a browser discards a `Secure` cookie sent over it.

    These two names are the whole of the exception, which is why they are asserted
    individually rather than as "anything that is not production".
    """
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, value)

    assert cookie_is_secure() is False


def test_the_cookie_is_secure_when_the_environment_is_unset(monkeypatch):
    """The fail-secure default, and the reason this function is written the way it is.

    An operator who forgets `DEEPGUARD_ENV` on a production host must not get a cookie that
    works perfectly while sending the session token in clear. The opposite mistake — the
    flag on where there is no TLS — breaks sign-in immediately and is found in seconds, so
    the unset case belongs on this side of the line.
    """
    monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)

    assert cookie_is_secure() is True


@pytest.mark.parametrize(
    "value",
    ["", "   ", "staging", "prod", "developement", "local"],
    ids=["empty", "whitespace", "staging", "abbreviated", "misspelled", "local"],
)
def test_an_unrecognized_environment_still_gets_a_secure_cookie(monkeypatch, value):
    """Anything not on the list is treated as production, typos included.

    `developement` is in this list deliberately. A misspelling of an insecure environment
    name is exactly the accident that must not silently disable the flag — under a
    "is it production?" test it would have.
    """
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, value)

    assert cookie_is_secure() is True


# --- against real PostgreSQL -----------------------------------------------------------------

integration = pytest.mark.integration


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
    """A real session whose accounts, sessions and analyses are removed again.

    Sessions and analyses are deleted before the users they point at: `auth_sessions`
    cascades but `analyses.owner_id` is `RESTRICT`, so a leftover analysis would block the
    account's deletion and leak the row into the next test.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []
    keys: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, analyses, keys

        db.rollback()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        for key_id in keys:
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()


def make_user(
    session,
    *,
    email: str | None = None,
    password: str = PASSWORD,
    role: str = USER_ROLE_USER,
    is_active: bool = True,
) -> User:
    """A persisted account, registered for cleanup."""
    db, users, _, _ = session

    user = User(
        # Unique per call, so two tests running against the same database cannot collide on
        # the unique index.
        email=normalize_email(email or f"{uuid.uuid4().hex}@example.com"),
        password_hash=hash_password(password),
        role=role,
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def make_api_key(session) -> tuple[ApiKey, str]:
    """A persisted API key and its one plaintext copy, registered for cleanup."""
    db, _, _, keys = session
    generated = generate_api_key()

    key = ApiKey(name="web-auth-isolation", key_hash=generated.key_hash)
    db.add(key)
    db.commit()
    keys.append(key.id)

    return key, generated.plaintext


def insert_session(session, user: User, **overrides) -> str:
    """A session row placed directly, so its expiry and revocation can be chosen.

    Returns the plaintext token, which exists only here and in the row's digest — the same
    arrangement the login route produces, staged by hand so a test can create the states a
    login never would.
    """
    db, _, _, _ = session
    token = generate_session_token()

    values = {
        "user_id": user.id,
        "token_hash": hash_session_token(token),
        "expires_at": datetime.now(timezone.utc) + SESSION_LIFETIME,
        "revoked_at": None,
    }
    values.update(overrides)

    db.add(AuthSession(**values))
    db.commit()

    return token


@pytest.fixture
def app(session):
    """The real auth routes, plus two routes that exist only to be guarded.

    The product has no route requiring a user or an administrator yet — R1-T1 builds the
    foundation and stops there — so the dependencies are mounted here. The auth router is
    the real one, so a test signs in through exactly the code the application runs.
    """
    db, _, _, _ = session

    test_app = FastAPI()
    test_app.include_router(auth_router)

    @test_app.get("/user-only")
    def user_only(user: User = Depends(require_user)) -> dict:
        return {"id": str(user.id), "role": user.role}

    @test_app.get("/admin-only")
    def admin_only(user: User = Depends(require_admin)) -> dict:
        return {"id": str(user.id), "role": user.role}

    test_app.dependency_overrides[get_session] = lambda: db

    yield test_app

    test_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def login(client, email: str, password: str = PASSWORD):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


def logout(client):
    """A sign-out, as the web application makes it.

    Carrying the `Origin` a browser attaches to every POST: since R1-T2 this route is behind
    `require_same_origin`, because ending somebody's session is a state change and a page on
    another site has no business making it. What the check does with a foreign or missing
    origin is `tests/test_dashboard_csrf.py`; the tests here are about what a real sign-out
    does to the session.
    """
    return client.post("/api/v1/auth/logout", headers={"Origin": DASHBOARD_ORIGIN})


def assert_invalid_credentials(response) -> None:
    assert response.status_code == 401
    assert response.json() == INVALID_CREDENTIALS_BODY


def assert_unauthenticated(response) -> None:
    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


# --- the migration --------------------------------------------------------------------------


@integration
def test_migration_created_the_users_table(database):
    inspector = inspect(database)

    assert "users" in inspector.get_table_names()

    columns = {column["name"] for column in inspector.get_columns("users")}
    assert columns == {
        "id",
        "email",
        "password_hash",
        "role",
        "is_active",
        "created_at",
    }

    unique_email = [
        index
        for index in inspector.get_indexes("users")
        if index["column_names"] == ["email"] and index["unique"]
    ]
    assert unique_email, "email must carry a unique index"


@integration
def test_migration_created_the_sessions_table_with_a_unique_hash_index(database):
    inspector = inspect(database)

    assert "auth_sessions" in inspector.get_table_names()

    columns = {column["name"] for column in inspector.get_columns("auth_sessions")}
    assert columns == {
        "id",
        "user_id",
        "token_hash",
        "created_at",
        "expires_at",
        "revoked_at",
    }

    unique_hash = [
        index
        for index in inspector.get_indexes("auth_sessions")
        if index["column_names"] == ["token_hash"] and index["unique"]
    ]
    assert unique_hash, "token_hash must carry a unique index"


@integration
def test_the_sessions_table_holds_no_plaintext_column(database):
    """There is no column a token could be stored in, whatever a later edit tried.

    A stronger statement than "we do not write it": the only string columns on the table are
    the 64-character digest and nothing else.
    """
    columns = {
        column["name"]: column for column in inspect(database).get_columns("auth_sessions")
    }

    assert columns["token_hash"]["type"].length == 64


@integration
def test_analyses_gained_a_nullable_indexed_owner(database):
    inspector = inspect(database)

    owner = [
        column
        for column in inspector.get_columns("analyses")
        if column["name"] == "owner_id"
    ]
    assert owner, "analyses must carry owner_id"
    assert owner[0]["nullable"] is True

    indexed = [
        index
        for index in inspector.get_indexes("analyses")
        if index["column_names"] == ["owner_id"]
    ]
    assert indexed, "owner_id must be indexed"


@integration
def test_the_existing_api_key_owner_column_is_untouched(database):
    """P9's ownership column must survive R1-T1 exactly as it was."""
    inspector = inspect(database)

    api_key_column = [
        column
        for column in inspector.get_columns("analyses")
        if column["name"] == "api_key_id"
    ]
    assert api_key_column, "analyses must still carry api_key_id"
    assert api_key_column[0]["nullable"] is True

    indexed = [
        index
        for index in inspector.get_indexes("analyses")
        if index["column_names"] == ["api_key_id"]
    ]
    assert indexed, "api_key_id must still be indexed"


# --- the ownership check constraint -----------------------------------------------------------


@integration
def test_an_analysis_may_be_owned_by_a_user(session):
    db, _, analyses, _ = session
    user = make_user(session)

    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED, owner_id=user.id)
    db.add(analysis)
    db.commit()
    analyses.append(analysis.id)

    assert analysis.owner_id == user.id
    assert analysis.api_key_id is None


@integration
def test_an_analysis_may_be_owned_by_an_api_key(session):
    db, _, analyses, _ = session
    key, _ = make_api_key(session)

    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED, api_key_id=key.id)
    db.add(analysis)
    db.commit()
    analyses.append(analysis.id)

    assert analysis.owner_id is None


@integration
def test_an_analysis_may_be_owned_by_nobody(session):
    """The dashboard's uploads, and every row that predates accounts."""
    db, _, analyses, _ = session

    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED)
    db.add(analysis)
    db.commit()
    analyses.append(analysis.id)

    assert analysis.owner_id is None
    assert analysis.api_key_id is None


@integration
def test_an_analysis_cannot_be_owned_by_both(session):
    """The one case the constraint exists to refuse, refused by PostgreSQL.

    Not by a check in the two functions that write these columns — this is asserted at the
    database because that is where it is enforced, and a row owned twice is what would make
    one caller's analysis reachable through the other authentication path.
    """
    db, _, _, _ = session
    user = make_user(session)
    key, _ = make_api_key(session)

    db.add(
        Analysis(status=ANALYSIS_STATUS_QUEUED, owner_id=user.id, api_key_id=key.id)
    )

    with pytest.raises(IntegrityError) as refused:
        db.commit()
    db.rollback()

    assert "ck_analyses_single_owner" in str(refused.value)


# --- signing in ---------------------------------------------------------------------------


@integration
def test_login_returns_the_user_and_sets_a_session_cookie(session, client):
    user = make_user(session)

    response = login(client, user.email)

    assert response.status_code == 200
    assert response.json() == {
        "id": str(user.id),
        "email": user.email,
        "role": USER_ROLE_USER,
    }
    assert client.cookies.get(SESSION_COOKIE_NAME)


@integration
def test_login_normalizes_the_submitted_address(session, client):
    """The address is found however it is typed, because both sides normalize identically."""
    user = make_user(session, email="Casey@Example.com")

    response = login(client, "  CASEY@example.COM  ")

    assert response.status_code == 200
    assert response.json()["id"] == str(user.id)


@integration
def test_the_session_cookie_carries_every_protective_flag(session, client, monkeypatch):
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "development")
    user = make_user(session)

    header = login(client, user.email).headers["set-cookie"]

    assert "HttpOnly" in header
    # Compared case-insensitively: RFC 6265bis makes the attribute value case-insensitive
    # and Starlette writes it lowercase, so pinning the casing would be asserting Starlette's
    # spelling rather than the protection.
    assert "samesite=lax" in header.lower()
    assert f"Max-Age={int(SESSION_LIFETIME.total_seconds())}" in header
    assert "Path=/" in header
    # Local development is plain HTTP, where a `Secure` cookie would simply be discarded.
    # Matched as an attribute rather than as a bare substring, so the assertion cannot be
    # decided by the random token happening to contain those letters.
    assert "; secure" not in header.lower()


@integration
def test_the_session_cookie_is_secure_in_production(session, client, monkeypatch):
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "production")
    user = make_user(session)

    header = login(client, user.email).headers["set-cookie"]

    assert "; secure" in header.lower()
    assert "HttpOnly" in header


@integration
def test_the_session_cookie_is_secure_when_the_environment_is_unset(
    session, client, monkeypatch
):
    """The fail-secure default, asserted on a real `Set-Cookie` rather than on the helper.

    `cookie_is_secure` is unit-tested above; this is the same property one level up, where a
    wiring mistake between the helper and `set_cookie` would show.
    """
    monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)
    user = make_user(session)

    header = login(client, user.email).headers["set-cookie"]

    assert "; secure" in header.lower()


@integration
def test_the_login_response_body_never_carries_the_token(session, client):
    """The token is `HttpOnly` in the cookie; echoing it into the body would undo that."""
    user = make_user(session)

    response = login(client, user.email)
    token = client.cookies.get(SESSION_COOKIE_NAME)

    assert token not in response.text
    assert hash_session_token(token) not in response.text


# --- every sign-in failure is the same failure ------------------------------------------------


@integration
def test_an_unknown_address_is_refused(session, client):
    assert_invalid_credentials(login(client, "nobody@example.com"))


@integration
def test_a_wrong_password_is_refused(session, client):
    user = make_user(session)

    assert_invalid_credentials(login(client, user.email, "not-the-password"))


@integration
def test_a_deactivated_account_is_refused(session, client):
    user = make_user(session, is_active=False)

    assert_invalid_credentials(login(client, user.email))


@integration
def test_all_sign_in_failures_are_byte_identical(session, client):
    """Unknown, wrong and deactivated must be indistinguishable to the caller.

    Compared as whole responses rather than as three assertions on the same constant,
    because the property under test is that they cannot be told apart — a difference in any
    of status, body or headers would tell an unauthenticated caller which addresses have
    accounts here.
    """
    known = make_user(session)
    deactivated = make_user(session, is_active=False)

    responses = [
        login(client, "nobody@example.com"),
        login(client, known.email, "not-the-password"),
        login(client, deactivated.email),
    ]

    signatures = {
        (r.status_code, r.text, r.headers.get("content-type")) for r in responses
    }
    assert len(signatures) == 1


@integration
def test_a_failed_sign_in_sets_no_cookie(session, client):
    user = make_user(session)

    response = login(client, user.email, "not-the-password")

    assert "set-cookie" not in response.headers
    assert not client.cookies.get(SESSION_COOKIE_NAME)


@integration
def test_a_failed_sign_in_never_echoes_the_password(session, client):
    user = make_user(session)

    response = login(client, user.email, "not-the-password")

    assert "not-the-password" not in response.text


# --- the session authenticates -----------------------------------------------------------------


@integration
def test_me_returns_the_signed_in_user(session, client):
    user = make_user(session)
    login(client, user.email)

    response = client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "id": str(user.id),
        "email": user.email,
        "role": USER_ROLE_USER,
    }


@integration
def test_me_without_a_cookie_is_unauthenticated(client):
    assert_unauthenticated(client.get("/api/v1/auth/me"))


@integration
def test_a_random_token_is_unauthenticated(client):
    """A token that was never issued names no row, so it authenticates nobody."""
    client.cookies.set(SESSION_COOKIE_NAME, generate_session_token())

    assert_unauthenticated(client.get("/api/v1/auth/me"))


@integration
def test_an_expired_session_is_unauthenticated(session, client):
    """Expiry is the database's judgement, against a row whose deadline has passed."""
    user = make_user(session)
    token = insert_session(
        session, user, expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)
    )
    client.cookies.set(SESSION_COOKIE_NAME, token)

    assert_unauthenticated(client.get("/api/v1/auth/me"))


@integration
def test_a_revoked_session_is_unauthenticated(session, client):
    user = make_user(session)
    token = insert_session(session, user, revoked_at=datetime.now(timezone.utc))
    client.cookies.set(SESSION_COOKIE_NAME, token)

    assert_unauthenticated(client.get("/api/v1/auth/me"))


@integration
def test_a_session_stops_working_when_the_account_is_deactivated(session, client):
    """Deactivating an account must end its access now, not when its session expires."""
    db, _, _, _ = session
    user = make_user(session)
    login(client, user.email)

    assert client.get("/api/v1/auth/me").status_code == 200

    user.is_active = False
    db.commit()

    assert_unauthenticated(client.get("/api/v1/auth/me"))


@integration
def test_presenting_the_stored_digest_as_the_token_is_refused(session, client):
    """Knowing what is in the table must not be enough to authenticate with it."""
    user = make_user(session)
    token = insert_session(session, user)
    client.cookies.set(SESSION_COOKIE_NAME, hash_session_token(token))

    assert_unauthenticated(client.get("/api/v1/auth/me"))


# --- rotation and sign-out ---------------------------------------------------------------------


@integration
def test_signing_in_again_revokes_the_previous_session(session, client):
    """One live session per account: the second sign-in ends the first.

    Checked from both ends — the old cookie stops authenticating, and the old row carries a
    revocation time — because either alone could pass while the other was broken.
    """
    db, _, _, _ = session
    user = make_user(session)

    login(client, user.email)
    first_token = client.cookies.get(SESSION_COOKIE_NAME)

    login(client, user.email)
    second_token = client.cookies.get(SESSION_COOKIE_NAME)

    assert first_token != second_token

    revoked = db.execute(
        AuthSession.__table__.select().where(
            AuthSession.token_hash == hash_session_token(first_token)
        )
    ).one()
    assert revoked.revoked_at is not None

    client.cookies.set(SESSION_COOKIE_NAME, first_token)
    assert_unauthenticated(client.get("/api/v1/auth/me"))

    client.cookies.set(SESSION_COOKIE_NAME, second_token)
    assert client.get("/api/v1/auth/me").status_code == 200


@integration
def test_rotation_leaves_exactly_one_live_session(session, client):
    db, _, _, _ = session
    user = make_user(session)

    for _ in range(3):
        login(client, user.email)

    live = db.execute(
        AuthSession.__table__.select().where(
            AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None)
        )
    ).all()

    assert len(live) == 1


@integration
def test_one_users_sign_in_does_not_disturb_another(session, client, app):
    """Rotation is per account. A shared revocation would sign everybody out at once."""
    first = make_user(session)
    second = make_user(session)

    with TestClient(app) as first_client:
        login(first_client, first.email)

        login(client, second.email)

        assert first_client.get("/api/v1/auth/me").status_code == 200


@integration
def test_logout_revokes_the_session_and_clears_the_cookie(session, client):
    user = make_user(session)
    login(client, user.email)
    token = client.cookies.get(SESSION_COOKIE_NAME)

    response = logout(client)

    assert response.status_code == 204

    # The cookie is gone from the browser, and — the part that matters — the token no longer
    # works even for a caller who kept a copy of it.
    client.cookies.set(SESSION_COOKIE_NAME, token)
    assert_unauthenticated(client.get("/api/v1/auth/me"))


@integration
def test_logout_without_a_session_still_succeeds(client):
    """A browser holding a dead cookie must be able to get rid of it, not be told 401."""
    assert logout(client).status_code == 204

    client.cookies.set(SESSION_COOKIE_NAME, generate_session_token())
    assert logout(client).status_code == 204


@integration
def test_logout_is_idempotent(session, client):
    user = make_user(session)
    login(client, user.email)
    token = client.cookies.get(SESSION_COOKIE_NAME)

    client.cookies.set(SESSION_COOKIE_NAME, token)
    first = logout(client)
    client.cookies.set(SESSION_COOKIE_NAME, token)
    second = logout(client)

    assert first.status_code == second.status_code == 204


# --- authorization ------------------------------------------------------------------------------


@integration
def test_a_user_reaches_a_user_route(session, client):
    user = make_user(session)
    login(client, user.email)

    response = client.get("/user-only")

    assert response.status_code == 200
    assert response.json() == {"id": str(user.id), "role": USER_ROLE_USER}


@integration
def test_an_admin_reaches_a_user_route(session, client):
    """`require_admin` narrows `require_user`; it does not replace it."""
    admin = make_user(session, role=USER_ROLE_ADMIN)
    login(client, admin.email)

    assert client.get("/user-only").status_code == 200


@integration
def test_an_admin_reaches_an_admin_route(session, client):
    admin = make_user(session, role=USER_ROLE_ADMIN)
    login(client, admin.email)

    response = client.get("/admin-only")

    assert response.status_code == 200
    assert response.json() == {"id": str(admin.id), "role": USER_ROLE_ADMIN}


@integration
def test_a_user_is_forbidden_from_an_admin_route(session, client):
    """403, not 401. The caller is authenticated; it is the role that is insufficient.

    Answering 401 would tell a signed-in person their session had gone bad and send them
    back to a login they do not need.
    """
    user = make_user(session)
    login(client, user.email)

    response = client.get("/admin-only")

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


@integration
def test_an_unauthenticated_caller_gets_401_from_an_admin_route(client):
    """Not 403: there is nobody to have insufficient permissions yet."""
    assert_unauthenticated(client.get("/admin-only"))


@integration
def test_a_revoked_session_cannot_reach_a_user_route(session, client):
    user = make_user(session)
    login(client, user.email)
    logout(client)

    assert_unauthenticated(client.get("/user-only"))


# --- the plaintext token is never persisted ---------------------------------------------------


@integration
def test_a_session_row_stores_no_plaintext_anywhere(session, client):
    """The whole row, read back as raw text, must not contain the token.

    Read as `auth_sessions::text` rather than through the model, so a column added later
    that happened to capture the plaintext fails this test instead of slipping past a fixed
    list of fields.
    """
    db, _, _, _ = session
    user = make_user(session)
    login(client, user.email)
    token = client.cookies.get(SESSION_COOKIE_NAME)

    row = db.execute(
        text(
            "SELECT auth_sessions::text AS whole_row FROM auth_sessions "
            "WHERE token_hash = :digest"
        ),
        {"digest": hash_session_token(token)},
    ).scalar_one()

    assert hash_session_token(token) in row
    assert token not in row


@integration
def test_no_table_anywhere_holds_the_session_token(session, client, database):
    """Stronger than the row check: the token appears in no table in the database.

    A session token written into an audit table, a log table or an analyses column would
    pass the test above and still be a persisted credential. Every table is scanned as raw
    text, so the assertion does not depend on knowing where such a leak might land.
    """
    db, _, _, _ = session
    user = make_user(session)
    login(client, user.email)
    token = client.cookies.get(SESSION_COOKIE_NAME)

    tables = inspect(database).get_table_names()

    for table in tables:
        found = db.execute(
            text(f'SELECT count(*) FROM "{table}" t WHERE t::text LIKE :needle'),
            {"needle": f"%{token}%"},
        ).scalar_one()

        assert found == 0, f"the session token was persisted in {table}"


@integration
def test_the_lookup_never_sends_the_plaintext_token_to_the_database(session, client):
    """Authentication queries by digest, so no statement or parameter carries the token."""
    db, _, _, _ = session
    user = make_user(session)
    login(client, user.email)
    token = client.cookies.get(SESSION_COOKIE_NAME)

    statements: list[str] = []
    original_execute = db.execute

    def record(statement, *args, **kwargs):
        compiled = getattr(statement, "compile", None)
        if compiled is not None:
            rendered = compiled()
            statements.append(str(rendered) + repr(rendered.params))
        return original_execute(statement, *args, **kwargs)

    db.execute = record
    try:
        assert client.get("/api/v1/auth/me").status_code == 200
    finally:
        db.execute = original_execute

    assert statements, "the session lookup issued no statement"
    for rendered in statements:
        assert token not in rendered
    assert any(hash_session_token(token) in rendered for rendered in statements)


# --- the two credential families do not overlap -------------------------------------------------


@integration
def test_a_web_cookie_does_not_authenticate_the_public_api(session):
    """The P9 isolation R1-T1 must not weaken: a browser session is not an API key.

    Run against the production app, so the routes are the ones that ship. The cookie is a
    real one from a real sign-in, presented to the public surface with no `Authorization`
    header — and refused with the API key's own 401, which is the proof that the public
    dependency never looked at a cookie at all.
    """
    db, _, _, _ = session
    user = make_user(session)

    production_app.dependency_overrides[get_session] = lambda: db
    try:
        with TestClient(production_app) as client:
            assert login(client, user.email).status_code == 200
            assert client.cookies.get(SESSION_COOKIE_NAME)

            read = client.get(f"/api/public/v1/analyses/{uuid.uuid4()}")
            submitted = client.post(
                "/api/public/v1/analyses/url", json={"url": "https://example.com/v.mp4"}
            )
    finally:
        production_app.dependency_overrides.clear()

    for refusal in (read, submitted):
        assert refusal.status_code == 401
        assert refusal.json() == {"detail": "Invalid or missing API key"}
        assert refusal.headers["WWW-Authenticate"] == "Bearer"


@integration
def test_an_api_key_does_not_authenticate_the_web_session_routes(session):
    """The other direction: a valid bearer token is not a sign-in.

    `require_user` reads a cookie and nothing else, so a caller holding a working API key is
    simply unauthenticated here — which keeps a leaked B2B credential from reaching whatever
    the web surface later puts behind a user.
    """
    db, _, _, _ = session
    _, plaintext = make_api_key(session)

    production_app.dependency_overrides[get_session] = lambda: db
    try:
        with TestClient(production_app) as client:
            response = client.get(
                "/api/v1/auth/me", headers={"Authorization": f"Bearer {plaintext}"}
            )
    finally:
        production_app.dependency_overrides.clear()

    assert_unauthenticated(response)


@integration
def test_the_dashboard_routes_require_a_web_session(session):
    """R1-T2 puts the gate on, and puts it only where it belongs.

    The analyses listing is the dashboard's own view and now refuses a caller it cannot
    identify. `/health` is checked in the same breath because it is the route that must
    stay open: it is what tells an operator, and the container's own healthcheck, whether
    this process is answering — and a health probe that had to sign in would report the
    deployment as broken the moment authentication was misconfigured, which is precisely
    when a truthful answer matters most.
    """
    db, _, _, _ = session

    production_app.dependency_overrides[get_session] = lambda: db
    try:
        with TestClient(production_app) as client:
            listing = client.get("/api/v1/analyses")
            health = client.get("/health")
    finally:
        production_app.dependency_overrides.clear()

    assert_unauthenticated(listing)
    assert health.status_code == 200


@integration
def test_a_signed_in_user_reaches_the_dashboard_listing(session):
    """The gate opens for a real sign-in, through the routes that ship.

    The whole path, in one test: sign in over the real login route, receive the cookie, and
    have the cookie jar present it to the listing the way a browser would. Everything about
    who sees which analyses is proven in `tests/test_dashboard_authorization.py`; what is
    established here is that a session issued by this module is a session that module's
    dependency accepts.
    """
    db, _, _, _ = session
    user = make_user(session)

    production_app.dependency_overrides[get_session] = lambda: db
    try:
        with TestClient(production_app) as client:
            assert login(client, user.email).status_code == 200
            response = client.get("/api/v1/analyses")
    finally:
        production_app.dependency_overrides.clear()

    assert response.status_code == 200
