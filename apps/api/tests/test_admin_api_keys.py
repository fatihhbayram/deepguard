"""Issuing and retiring public-API credentials, and what must never escape while it happens.

The sibling of `test_admin_users.py` and `test_admin_audit.py`, written to their conventions:
real PostgreSQL, real sessions opened through the real login route, one client per account, and
every row this file creates removed afterwards. A role check demonstrated against a stubbed
session proves only that the stub agreed.

The claims divide into four groups, and they are not of equal weight.

*Plaintext isolation.* The one that justifies the file. The secret is returned by exactly one
request and exists nowhere else afterwards — not in the row, not on the audit event, not in the
listing. These tests assert it against **stored values**, not against field names: a test for
`"plaintext" not in body` would pass the day somebody spelled the key differently, so what is
actually checked is that the specific secret string that came back does not appear in the
things it could have leaked into.

*The lifecycle.* A key is created active and works; revoking it stops it working on the very
next request; revoking it again does nothing at all. The "works" and "stops working" halves are
asserted through the **public API** rather than by reading `is_active` out of the database,
because the property being claimed is about access and not about a column — a test that checked
the flag would keep passing if `require_api_key` stopped consulting it.

*The audit trail.* One event per thing that actually happened. Two for a create-then-revoke,
still two after revoking three more times, and zero fields of credential material on either.

*Authorization.* Every route refuses an anonymous caller with 401 and a signed-in
non-administrator with 403, and the two mutations refuse a request with no `Origin`.

Counting convention, inherited from `test_admin_audit.py`: this deployment's tables are shared
with the rest of the suite, so nothing here asserts an absolute row count. Events are counted
**for a specific key id** that this file created, and the key listing is filtered to the ids
this file created before anything is asserted about it.
"""

import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.auth import API_KEY_PREFIX, generate_api_key
from app.db.models import (
    AUDIT_ACTION_API_KEY_CREATED,
    AUDIT_ACTION_API_KEY_REVOKED,
    AUDIT_TARGET_API_KEY,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    ApiKey,
    AuthSession,
    User,
)
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

KEYS_URL = "/api/v1/admin/api-keys"
AUDIT_URL = "/api/v1/admin/audit"

# The public surface these credentials are for. Read with an id that names nothing, on purpose
# — see `public_read` for why that is the cleanest possible probe of authentication.
PUBLIC_ANALYSIS_URL = "/api/public/v1/analyses"

# A test fixture, not a credential: nothing outside this file uses it and no deployment ships
# it.
PASSWORD = "correct-horse-battery-staple"

FORBIDDEN_BODY = {"detail": "Insufficient permissions"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}
UNAUTHENTICATED_KEY_BODY = {"detail": "Invalid or missing API key"}

# What a listed key carries, and the entire set of it. Written out here rather than read back
# off the first response, so that widening the contract has to be a deliberate edit to this
# line rather than something a test quietly accepts. `key_hash` is absent from this set and
# that absence is the assertion.
VISIBLE_FIELDS = {"id", "name", "is_active", "created_at", "last_used_at"}

# The creation response, which is the one payload in this application that carries a secret.
CREATED_FIELDS = VISIBLE_FIELDS | {"plaintext"}


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
    """A real session that removes every row this file created.

    Three kinds of row, cleared in the one order that works. Audit events first — they hold no
    foreign key at all, so this is tidiness rather than necessity, but they are scoped to the
    keys and accounts below and become unreachable once those are gone. Then the keys, which is
    safe here only because this file never attaches an analysis to one: `analyses.api_key_id`
    is `ON DELETE RESTRICT`, so a used key could not be deleted. Then sessions before users,
    which is the genuinely load-bearing ordering — `auth_sessions.user_id` really is a foreign
    key.
    """
    users: list[uuid.UUID] = []
    keys: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, keys

        db.rollback()
        for key_id in keys:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.target_id == str(key_id)
            ).delete(synchronize_session=False)
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        for user_id in users:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.actor_id == user_id
            ).delete(synchronize_session=False)
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


def make_user(session, *, role: str = USER_ROLE_USER, is_active: bool = True) -> User:
    """A persisted account, registered for cleanup."""
    db, users, _ = session

    user = User(
        # Unique per call, so two runs against the same database cannot collide on the unique
        # index over the email.
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


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
    """A signed-in administrator, which is what every route below is exercised through."""
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def create(client: TestClient, session=None, *, name="Customer X", **body):
    """Issue a key, sent the way the dashboard sends the request.

    The origin header is not optional: the route carries `require_same_origin`, and a request
    without it is refused before any of the logic this file is about.

    Registers the new key for cleanup when a session is handed in and the request succeeded.
    Passing the session is how a test says "this row is mine"; the authorization tests below
    deliberately do not, because they expect no row to be created.
    """
    response = client.post(
        KEYS_URL, json={"name": name, **body}, headers={"Origin": DASHBOARD_ORIGIN}
    )

    if session is not None and response.status_code == 201:
        _, _, keys = session
        keys.append(uuid.UUID(response.json()["id"]))

    return response


def revoke(client: TestClient, key_id, **kwargs):
    """Retire a key, sent the way the dashboard sends the request."""
    headers = kwargs.pop("headers", {"Origin": DASHBOARD_ORIGIN})

    return client.post(f"{KEYS_URL}/{key_id}/revoke", headers=headers, **kwargs)


def authorization(plaintext: str) -> dict[str, str]:
    """The header a B2B caller presents a key in."""
    return {"Authorization": f"Bearer {plaintext}"}


def public_read(plaintext: str):
    """Probe the public API's authentication with a key, and nothing else.

    Reads an analysis id that was never issued. That makes the response a clean statement about
    the *credential* rather than about any record: a key the API accepts gets 404 because the
    analysis does not exist, and a key it does not accept gets 401 before the lookup happens at
    all. Nothing has to be stored to ask the question, which is what keeps this file from
    creating analyses it would then be unable to delete — `analyses.api_key_id` is
    `ON DELETE RESTRICT`.
    """
    with TestClient(app) as client:
        return client.get(
            f"{PUBLIC_ANALYSIS_URL}/{uuid.uuid4()}", headers=authorization(plaintext)
        )


def stored_key(session, key_id) -> ApiKey:
    """The row as the database actually holds it."""
    db, _, _ = session
    db.commit()  # Drop this session's snapshot so the API's committed writes are visible.

    key = db.get(ApiKey, uuid.UUID(str(key_id)))
    assert key is not None, "the key was not persisted"

    return key


def events_for(session, key_id) -> list[AdminAuditEvent]:
    """Every recorded event about one key, newest first, read straight from the database.

    Scoped to a key this file created — never a count of the whole table. Other tests write
    audit rows too, and an assertion about "how many events exist" would pass alone and fail in
    a suite.
    """
    db, _, _ = session
    db.commit()

    return list(
        db.execute(
            select(AdminAuditEvent)
            .where(AdminAuditEvent.target_id == str(key_id))
            .order_by(AdminAuditEvent.created_at.desc(), AdminAuditEvent.id.desc())
        )
        .scalars()
        .all()
    )


def count_for(session, key_id) -> int:
    """How many events stand against one key."""
    db, _, _ = session
    db.commit()

    return db.execute(
        select(func.count(AdminAuditEvent.id)).where(
            AdminAuditEvent.target_id == str(key_id)
        )
    ).scalar_one()


# --- plaintext isolation ---------------------------------------------------------------


def test_creation_returns_the_plaintext_and_the_contract_around_it(session):
    """201, the secret, and exactly the fields that belong on this one response."""
    client = administrator(session)

    response = create(client, session, name="Acme")
    body = response.json()

    assert response.status_code == 201
    assert set(body) == CREATED_FIELDS
    assert body["name"] == "Acme"
    assert body["is_active"] is True
    assert body["last_used_at"] is None
    assert body["plaintext"].startswith(API_KEY_PREFIX)


def test_the_database_stores_the_digest_and_not_the_secret(session):
    """What is on the row is SHA-256 of the full presented string, and nothing else.

    Asserted by recomputing the digest rather than by checking that `key_hash` merely differs
    from the plaintext — the claim is that the stored value is the *right* digest, which is
    also what makes the public API able to verify it.
    """
    client = administrator(session)
    body = create(client, session).json()
    plaintext = body["plaintext"]

    key = stored_key(session, body["id"])

    assert key.key_hash == hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
    assert key.key_hash != plaintext
    assert plaintext not in key.name


def test_the_secret_is_returned_once_and_never_again(session):
    """Every subsequent read of that key, through every route that reports it.

    The listing is the route a reader would go back to, and the revoke response is the other
    payload this key ever produces. Neither carries the secret, and neither carries the digest.
    Checked against the actual values rather than against field names: a test for
    `"plaintext" not in row` would pass the day somebody nested it under another key.
    """
    client = administrator(session)
    body = create(client, session).json()
    plaintext, key_id = body["plaintext"], body["id"]
    digest = stored_key(session, key_id).key_hash

    listing = client.get(KEYS_URL)
    row = next(row for row in listing.json() if row["id"] == key_id)
    revoked = revoke(client, key_id)

    assert set(row) == VISIBLE_FIELDS
    assert set(revoked.json()) == VISIBLE_FIELDS

    for payload in (listing.text, revoked.text):
        assert plaintext not in payload
        assert digest not in payload


def test_no_key_material_reaches_the_audit_log(session):
    """Neither the secret nor the digest, on either of the two events.

    Asserted against the whole serialized row rather than against `changes` alone, and against
    the stored values rather than against field names — the claim is that these specific
    strings appear nowhere on the event at all, however somebody might have spelled the field
    that carried them.
    """
    client = administrator(session)
    body = create(client, session).json()
    plaintext, key_id = body["plaintext"], body["id"]
    digest = stored_key(session, key_id).key_hash

    revoke(client, key_id)

    for event in events_for(session, key_id):
        row = (
            f"{event.changes}{event.action}{event.target_type}{event.target_id}"
            f"{event.actor_email_snapshot}{event.target_email_snapshot}{event.request_id}"
        )

        assert plaintext not in row
        assert digest not in row


def test_the_audit_screen_never_reports_key_material(session):
    """The other reader of these events, checked end to end.

    `test_no_key_material_reaches_the_audit_log` asserts the row; this asserts what the audit
    API actually serves, which is what a person would be looking at. The two are the same claim
    at different distances, and this is the one that would catch a future route that joined
    something back in.
    """
    client = administrator(session)
    body = create(client, session).json()
    plaintext, key_id = body["plaintext"], body["id"]
    digest = stored_key(session, key_id).key_hash

    revoke(client, key_id)
    audit = client.get(AUDIT_URL)

    assert audit.status_code == 200
    assert plaintext not in audit.text
    assert digest not in audit.text


def test_a_caller_cannot_supply_its_own_key_material(session):
    """There is no field a request could set even if the route wanted to read one.

    `ApiKeyCreate` forbids extra fields, so each of these is a 422 rather than a request that
    is accepted with the field quietly dropped. The difference matters: an administrator who
    believes they pinned a key's hash and was silently ignored is in a worse position than one
    who was told the field does not exist.
    """
    client = administrator(session)

    for field, value in (
        ("key_hash", "0" * 64),
        ("plaintext", "dg_live_chosen"),
        ("is_active", False),
        ("id", str(uuid.uuid4())),
    ):
        assert create(client, **{field: value}).status_code == 422, field


# --- the lifecycle, asserted through the public API -------------------------------------


def test_a_new_key_authenticates_against_the_public_api(session):
    """A key that exists is a key that works — there is no activation step.

    404 and not 401: the analysis id is one that was never issued, so the credential was
    accepted and the lookup found nothing. That is the distinction this whole group rests on.
    """
    client = administrator(session)
    plaintext = create(client, session).json()["plaintext"]

    response = public_read(plaintext)

    assert response.status_code == 404, "the new key was not accepted by the public API"


def test_revoking_a_key_ends_its_access_immediately(session):
    """The same key, before and after, through the same request.

    Asserted through the public surface rather than by reading `is_active` out of the row: the
    property is about access, and a test that checked the column would keep passing if
    `require_api_key` stopped consulting it.
    """
    client = administrator(session)
    body = create(client, session).json()
    plaintext = body["plaintext"]

    assert public_read(plaintext).status_code == 404, "the key should start out working"

    assert revoke(client, body["id"]).status_code == 200

    refused = public_read(plaintext)

    assert refused.status_code == 401
    assert refused.json() == UNAUTHENTICATED_KEY_BODY


def test_revoking_one_key_does_not_disturb_another(session):
    """The blast radius of a revocation is one credential.

    Worth stating because the revoke route reads its target `FOR UPDATE` and writes through the
    ORM session — the shape in which a mis-scoped `WHERE` deactivates a table rather than a row.
    """
    client = administrator(session)
    retired = create(client, session, name="retired").json()
    kept = create(client, session, name="kept").json()

    revoke(client, retired["id"])

    assert public_read(retired["plaintext"]).status_code == 401
    assert public_read(kept["plaintext"]).status_code == 404


def test_an_existing_key_issued_outside_this_api_still_works(session):
    """Keys minted before R8-T6 are ordinary rows and nothing here changed that.

    Written straight to the table the way `test_public_api.py` writes one, then listed and
    revoked through the new routes. The management lifecycle did not introduce a marker that
    divides the keys it created from the ones it did not.
    """
    db, _, keys = session
    generated = generate_api_key()
    legacy = ApiKey(name="legacy", key_hash=generated.key_hash)
    db.add(legacy)
    db.commit()
    keys.append(legacy.id)

    client = administrator(session)

    assert public_read(generated.plaintext).status_code == 404

    listed = {row["id"] for row in client.get(KEYS_URL).json()}
    assert str(legacy.id) in listed

    assert revoke(client, legacy.id).status_code == 200
    assert public_read(generated.plaintext).status_code == 401


# --- revocation as a no-op --------------------------------------------------------------


def test_revoking_an_already_revoked_key_changes_nothing(session):
    """200, the key as it stands, and no second write.

    Not an error: the caller asked for a state and that is the state. The assertion that
    matters is the event count — one revocation is one event, however many times it is
    requested.
    """
    client = administrator(session)
    key_id = create(client, session).json()["id"]

    assert revoke(client, key_id).status_code == 200
    first = events_for(session, key_id)

    for _ in range(3):
        again = revoke(client, key_id)

        assert again.status_code == 200
        assert again.json()["is_active"] is False

    assert count_for(session, key_id) == 2, "create and revoke, and nothing after"
    assert [event.id for event in events_for(session, key_id)] == [
        event.id for event in first
    ], "the repeated revocations wrote a row"


def test_revoking_a_key_that_does_not_exist_is_a_404(session):
    """And writes nothing. There is nothing to conceal from a caller entitled to list every
    key in the system, so this is a plain 404 rather than the uniform one the public API gives.
    """
    client = administrator(session)
    missing = uuid.uuid4()

    response = revoke(client, missing)

    assert response.status_code == 404
    assert count_for(session, missing) == 0


# --- the audit trail --------------------------------------------------------------------


def test_creating_a_key_writes_exactly_one_event(session):
    """One request, one row, naming the key and its label — and not its material."""
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)

    body = create(client, session, name="Acme").json()
    key_id = body["id"]

    assert count_for(session, key_id) == 1

    event = events_for(session, key_id)[0]

    assert event.actor_id == admin.id
    assert event.actor_email_snapshot == admin.email
    assert event.action == AUDIT_ACTION_API_KEY_CREATED
    assert event.target_type == AUDIT_TARGET_API_KEY
    assert event.target_id == key_id
    assert event.changes == {"name": {"old": None, "new": "Acme"}}


def test_revoking_a_key_writes_exactly_one_event(session):
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)
    key_id = create(client, session).json()["id"]

    revoke(client, key_id)

    assert count_for(session, key_id) == 2

    event = events_for(session, key_id)[0]

    assert event.actor_id == admin.id
    assert event.action == AUDIT_ACTION_API_KEY_REVOKED
    assert event.target_type == AUDIT_TARGET_API_KEY
    assert event.changes == {"is_active": {"old": True, "new": False}}


def test_an_event_about_a_key_snapshots_no_target_address(session):
    """Null, and deliberately so: a key is not a person and has no address.

    The actor's address is on the row because a person did do this; the target column stays
    empty rather than being filled with the key's name, which would put a label in a column
    every other row means an email by.
    """
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)
    key_id = create(client, session).json()["id"]
    revoke(client, key_id)

    for event in events_for(session, key_id):
        assert event.target_email_snapshot is None
        assert event.actor_email_snapshot == admin.email


def test_key_events_carry_the_changes_shape_the_audit_screen_renders(session):
    """Every value under `changes` is an `{old, new}` pair.

    The audit surface renders one component over every event and parses each entry as that
    pair; a flat payload here — `{"name": "Acme"}` — would not merely render oddly, it would
    fail the parse and take the whole listing down with it. So the shape is asserted
    structurally rather than by comparing against one expected literal, which is what keeps a
    third action added later from quietly breaking that page.
    """
    client = administrator(session)
    key_id = create(client, session).json()["id"]
    revoke(client, key_id)

    for event in events_for(session, key_id):
        assert event.changes, "an event with an empty diff says nothing happened"
        for field, change in event.changes.items():
            assert isinstance(change, dict), field
            assert set(change) == {"old", "new"}, field


def test_an_events_actor_is_the_authenticated_administrator(session):
    """Taken from the session `require_admin` resolved, never from the request body."""
    admin = make_user(session, role=USER_ROLE_ADMIN)
    other = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)

    key_id = create(client, session).json()["id"]

    assert events_for(session, key_id)[0].actor_id == admin.id
    assert events_for(session, key_id)[0].actor_id != other.id


# --- the listing -------------------------------------------------------------------------


def test_the_listing_reports_revoked_keys_too(session):
    """A table that hid them would hide exactly the rows somebody came here to check."""
    client = administrator(session)
    retired = create(client, session, name="retired").json()["id"]
    revoke(client, retired)

    rows = {row["id"]: row for row in client.get(KEYS_URL).json()}

    assert retired in rows
    assert rows[retired]["is_active"] is False


def test_the_listing_is_newest_first(session):
    client = administrator(session)
    first = create(client, session, name="first").json()["id"]
    second = create(client, session, name="second").json()["id"]

    mine = [row["id"] for row in client.get(KEYS_URL).json() if row["id"] in {first, second}]

    assert mine == [second, first]


# --- names --------------------------------------------------------------------------------


def test_a_name_is_stored_stripped(session):
    """One normalization, so the label on the row and the label on the event are the same."""
    client = administrator(session)
    body = create(client, session, name="  Acme  ").json()

    assert body["name"] == "Acme"
    assert stored_key(session, body["id"]).name == "Acme"
    assert events_for(session, body["id"])[0].changes["name"]["new"] == "Acme"


def test_an_unusable_name_is_refused(session):
    """Blank, whitespace-only, and longer than the column.

    The last one is validated rather than left to PostgreSQL: a value this API accepted and the
    database refused would surface as a 500 on a request that was merely too long.
    """
    client = administrator(session)

    assert create(client, name="").status_code == 422
    assert create(client, name="   ").status_code == 422
    assert create(client, name="x" * 256).status_code == 422
    assert create(client, name="x" * 255).status_code == 201, "the column's length must fit"

    # The one that succeeded is a real row; register it so the fixture clears it up.
    _, _, keys = session
    listing = administrator(session).get(KEYS_URL).json()
    keys.extend(
        uuid.UUID(row["id"]) for row in listing if row["name"] == "x" * 255
    )


# --- authorization --------------------------------------------------------------------------


def test_an_anonymous_caller_cannot_reach_any_route(database):
    with TestClient(app) as client:
        listing = client.get(KEYS_URL)
        creation = client.post(
            KEYS_URL, json={"name": "x"}, headers={"Origin": DASHBOARD_ORIGIN}
        )
        revocation = client.post(
            f"{KEYS_URL}/{uuid.uuid4()}/revoke", headers={"Origin": DASHBOARD_ORIGIN}
        )

    for response in (listing, creation, revocation):
        assert response.status_code == 401
        assert response.json() == UNAUTHENTICATED_BODY


def test_an_ordinary_user_cannot_reach_any_route(session):
    """403 and not 401. The session is perfectly good; the role is not, and the status says
    which — telling a signed-in person to sign in again would send them round a loop that
    cannot fix anything."""
    client = signed_in(make_user(session))

    listing = client.get(KEYS_URL)
    creation = create(client)
    revocation = revoke(client, uuid.uuid4())

    for response in (listing, creation, revocation):
        assert response.status_code == 403
        assert response.json() == FORBIDDEN_BODY


def test_an_ordinary_user_cannot_mint_a_working_key(session):
    """The refusal above, asserted as the thing it actually prevents.

    A 403 on the creation route is only interesting if no credential came out of it. Nothing
    was returned to present, so what is checked is that the key table did not grow.
    """
    db, _, _ = session
    client = signed_in(make_user(session))

    before = db.execute(select(func.count(ApiKey.id))).scalar_one()
    assert create(client).status_code == 403
    db.commit()

    assert db.execute(select(func.count(ApiKey.id))).scalar_one() == before


def test_a_mutation_without_an_origin_is_refused(session):
    """The CSRF boundary, on both writes. A missing `Origin` is refused rather than waved
    through — the shape a forged submission has is precisely the one that sends no header."""
    client = administrator(session)
    key_id = create(client, session).json()["id"]

    creation = client.post(KEYS_URL, json={"name": "forged"})
    revocation = client.post(f"{KEYS_URL}/{key_id}/revoke")

    assert creation.status_code == 403
    assert revocation.status_code == 403
    assert stored_key(session, key_id).is_active is True, "the forged revocation took effect"


def test_reading_the_listing_needs_no_origin(session):
    """Reads are not state-changing and do not carry the check — stated so that adding it to
    the router wholesale would fail here rather than break the page silently."""
    assert administrator(session).get(KEYS_URL).status_code == 200
