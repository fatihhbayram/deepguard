"""Who may administer accounts, what they may change, and the two states they may not create.

Everything here runs against real PostgreSQL through the production application, with
sessions opened by the real login route and presented the way a browser presents them. That
is not ceremony: the rules under test are a role comparison and a `COUNT` over a table, and
neither can be demonstrated against a stubbed session that hands back whatever account it was
given. A fake admin is admin because the fake says so, which is the one thing worth proving.

Four things are asserted, in the order they matter:

*Authorization.* `/api/v1/admin/users` is behind `require_admin`, so an anonymous caller gets
401 and a signed-in ordinary user gets 403 — the distinction the two statuses carry, and the
reason the dependency is layered rather than reimplemented.

*Exposure.* The listing returns five identity fields. The `User` row it is built from also
holds an Argon2id password hash, and the test that no response ever carries one is written
against the whole payload rather than against a named field, so a column added to `User`
later cannot pass by not being on anybody's list.

*The self-modification rule.* An administrator cannot change their own role or activation,
because that is the change nobody can undo for them.

*The minimum-administrator invariant.* At least one active administrator must remain. This
one cannot be reached over HTTP, and the reason is the rule above it: the caller is an active
administrator and cannot be the target, so one always survives. That is a property of today's
rule set and not of the invariant, so it is asserted where it is decidable — by calling the
route function with an administrator whose account was deactivated after their session was
resolved, which is a state a live deployment can actually be in for the width of one request.
See `test_the_last_active_administrator_cannot_be_demoted`.

R8-T8 completed the lifecycle and added three more things to assert, all of them below the
existing sections:

*Creation.* An administrator can put an account on file with a password they chose, the same
address cannot be used twice, and the password is stored only as a hash — proved by signing
the new account in with the plaintext rather than by inspecting the column's prefix.

*The address.* A PATCH may now change it. That is asserted alongside the two safety rules it
must not weaken: the self-modification refusal still fires on the target's id whatever field
the body names, and the minimum-administrator check is still about role and activation.

*The password reset, which is the one operation here with three writes in it.* The new hash,
the revocation of every open session and the audit event must commit together, so the tests
assert the whole of that outcome: the old password stops working, the new one works, sessions
opened before the reset are dead, and exactly one event was written carrying neither form of
the password. A refused reset is asserted to have written none of the three.

*What never appears.* `test_no_audit_event_ever_carries_a_password` walks every event these
tests produced and searches the serialized JSON for the plaintexts, rather than checking named
fields — a test against a field list stops catching anything the day somebody adds a field.

Two conventions inherited from `test_dashboard_authorization.py`, which this file is the
sibling of: accounts are cleaned up in constraint order by the `session` fixture, and each
account gets its own client because two sessions in one cookie jar is a state no browser is
ever in.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.api.admin_users import UserChange, update_user
from app.db.models import (
    AUDIT_ACTION_USER_CREATED,
    AUDIT_ACTION_USER_PASSWORD_RESET,
    AUDIT_ACTION_USER_UPDATED,
    AUDIT_TARGET_USER,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    AuthSession,
    User,
)
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import (
    ENVIRONMENT_VARIABLE,
    MINIMUM_PASSWORD_LENGTH,
    SESSION_COOKIE_NAME,
    hash_password,
    verify_password,
)
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

USERS_URL = "/api/v1/admin/users"

# A test fixture, not a credential: nothing outside this file uses it and no deployment ships
# it.
PASSWORD = "correct-horse-battery-staple"

# The one a reset moves an account to. A second distinct value rather than a mutation of the
# first, so a test can assert that the old password stopped working and the new one started
# — two facts that a single value could not tell apart.
NEW_PASSWORD = "tromboneS-are-not-a-vegetable"

assert len(PASSWORD) >= MINIMUM_PASSWORD_LENGTH
assert len(NEW_PASSWORD) >= MINIMUM_PASSWORD_LENGTH

FORBIDDEN_BODY = {"detail": "Insufficient permissions"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}

# The fields an administrator is shown, and the entire set of them. Written out here rather
# than read back off the first response, so that widening the contract has to be a deliberate
# edit to this line rather than something a test quietly accepts.
VISIBLE_FIELDS = {"id", "email", "role", "is_active", "created_at"}


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
    """A real session that removes every account this file created, in constraint order.

    Sessions before users: `auth_sessions.user_id` is a foreign key, and an account with a
    live session cannot be deleted. Nothing here creates an analysis, which is the other thing
    that pins a user row down.

    The audit events these tests now write are deliberately not swept up here. Nothing in
    `admin_audit_events` references an account — `actor_id` is a bare UUID, by design — so an
    account deletes cleanly with its events still standing, which is exactly the property that
    table exists to have. Leaving them is the assertion.
    """
    users: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users

        db.rollback()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


@pytest.fixture
def sole_administrators(session):
    """Deactivate every administrator this file did not create, and put them back afterwards.

    The minimum-administrator invariant is a count over the whole table, so a test about "the
    last active administrator" is only about the last one if the database does not also hold
    somebody else's. The suite owns `deepguard_test` and cleans up after itself, so normally
    there is nobody — but a run that was interrupted between a test's assertions and its
    teardown leaves an administrator behind, and the failure that causes appears in this file
    and is about a row no test in it created, which is the hardest kind to read.

    So the state is stated rather than assumed. The rows are restored in teardown whatever the
    test did, and nothing is deleted: these accounts are not this file's to remove.
    """
    db, _ = session

    quarantined = [
        user.id
        for user in db.execute(
            select(User).where(User.role == USER_ROLE_ADMIN, User.is_active.is_(True))
        )
        .scalars()
        .all()
    ]

    for user_id in quarantined:
        db.query(User).filter(User.id == user_id).update({"is_active": False})
    db.commit()

    yield

    db.rollback()
    for user_id in quarantined:
        db.query(User).filter(User.id == user_id).update({"is_active": True})
    db.commit()


def make_user(session, *, role: str = USER_ROLE_USER, is_active: bool = True) -> User:
    """A persisted account, registered for cleanup."""
    db, users = session

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


def patch(client: TestClient, user_id, body: dict, *, origin: str = DASHBOARD_ORIGIN):
    """A change submitted the way the web application submits one, origin header included."""
    return client.patch(f"{USERS_URL}/{user_id}", json=body, headers={"Origin": origin})


def listed(client: TestClient) -> dict[str, dict]:
    """The listing, keyed by account id, so a test can find its own rows among everyone's."""
    response = client.get(USERS_URL)

    assert response.status_code == 200

    return {row["id"]: row for row in response.json()}


def stored(session, user_id) -> User:
    """The account as the database now holds it, read fresh rather than from a stale object."""
    db, _ = session
    db.expire_all()

    return db.execute(select(User).where(User.id == user_id)).scalar_one()


def stored_by_email(session, email: str) -> User | None:
    """The account holding this address, or None. Used to assert a refused creation wrote none.

    Normalized on the way in, because that is the form the column holds and a test that looked
    for the raw string would report "no account" about an account that exists.
    """
    db, _ = session
    db.commit()  # Drop this session's snapshot so the API's committed writes are visible.

    return db.execute(
        select(User).where(User.email == email.strip().lower())
    ).scalar_one_or_none()


def create(client: TestClient, body: dict, *, origin: str = DASHBOARD_ORIGIN):
    """A creation submitted the way the web application submits one, origin header included."""
    return client.post(USERS_URL, json=body, headers={"Origin": origin})


def detail(client: TestClient, user_id):
    """One account, read through the route the detail page reads."""
    return client.get(f"{USERS_URL}/{user_id}")


def reset(client: TestClient, user_id, password: str, *, origin: str = DASHBOARD_ORIGIN):
    """A password reset, submitted the way the web application submits one."""
    return client.post(
        f"{USERS_URL}/{user_id}/password",
        json={"password": password},
        headers={"Origin": origin},
    )


def new_account_body(**overrides) -> dict:
    """A creation payload with a unique address, so two runs cannot collide on the index."""
    return {
        "email": f"{uuid.uuid4().hex}@example.com",
        "password": PASSWORD,
        **overrides,
    }


def created(session, response) -> User:
    """The account a successful creation made, registered for this file's cleanup.

    Registration is the point of it. `make_user` adds the ids it inserts to the fixture's list,
    and an account created over HTTP has to be added the same way or it outlives the test — in a
    table with a unique index, where the next run's assertions about "no account with this
    address" would then be about a row nobody meant to leave behind.
    """
    _, users = session
    user_id = uuid.UUID(response.json()["id"])
    users.append(user_id)

    return stored(session, user_id)


def open_session_count(session, user_id) -> int:
    """How many sessions this account still holds open, straight from the table.

    Counted on `revoked_at IS NULL` rather than by trying the cookie, because the two questions
    are different: a revoked session and an expired one are both refused at the door, and only
    the first is what a reset is supposed to have done. The tests that care about the door use a
    real request as well.
    """
    db, _ = session
    db.commit()  # Drop this session's snapshot so the API's committed writes are visible.

    return db.execute(
        select(func.count(AuthSession.id)).where(
            AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None)
        )
    ).scalar_one()


def events_for(session, user_id) -> list[AdminAuditEvent]:
    """Every recorded event about one account, newest first, read straight from the database.

    Scoped to an account this file created — never a count of the whole table. Other tests write
    audit rows too, and an assertion about "how many events exist" would pass alone and fail in
    a suite. The same helper `test_admin_audit.py` uses, restated here for the reason that file
    restates the fixtures: two test modules are not each other's library.
    """
    db, _ = session
    db.commit()

    return list(
        db.execute(
            select(AdminAuditEvent)
            .where(AdminAuditEvent.target_id == str(user_id))
            .order_by(AdminAuditEvent.created_at.desc(), AdminAuditEvent.id.desc())
        )
        .scalars()
        .all()
    )


# --- authorization --------------------------------------------------------------------


def test_an_anonymous_caller_cannot_list_accounts(database):
    with TestClient(app) as client:
        response = client.get(USERS_URL)

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_anonymous_caller_cannot_change_an_account(session):
    target = make_user(session)

    with TestClient(app) as client:
        response = patch(client, target.id, {"role": USER_ROLE_ADMIN})

    assert response.status_code == 401
    assert stored(session, target.id).role == USER_ROLE_USER


def test_an_ordinary_user_cannot_list_accounts(session):
    """403 and not 401. The session is perfectly good; the role is not, and the status says
    which — telling a signed-in person to sign in again would send them round a loop that
    cannot fix anything."""
    response = signed_in(make_user(session)).get(USERS_URL)

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


def test_an_ordinary_user_cannot_change_an_account(session):
    """The refusal that matters most in this file: without it, promotion to administrator is
    a request any signed-in account can make about itself."""
    caller = make_user(session)
    target = make_user(session)

    response = patch(signed_in(caller), target.id, {"role": USER_ROLE_ADMIN})

    assert response.status_code == 403
    assert stored(session, target.id).role == USER_ROLE_USER


def test_an_ordinary_user_cannot_promote_themselves(session):
    caller = make_user(session)

    response = patch(signed_in(caller), caller.id, {"role": USER_ROLE_ADMIN})

    assert response.status_code == 403
    assert stored(session, caller.id).role == USER_ROLE_USER


def test_a_change_from_an_unaccepted_origin_is_refused(session):
    """The CSRF check the other dashboard mutations carry, on the route that grants access."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(
        signed_in(administrator),
        target.id,
        {"role": USER_ROLE_ADMIN},
        origin="http://evil.test",
    )

    assert response.status_code == 403
    assert stored(session, target.id).role == USER_ROLE_USER


# --- the listing ----------------------------------------------------------------------


def test_an_administrator_sees_every_account_including_deactivated_ones(session):
    """Deactivated accounts are listed, not hidden. They are the ones somebody came here to
    reactivate, and a listing that filtered them would make that impossible from this page."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    active = make_user(session)
    deactivated = make_user(session, is_active=False)

    rows = listed(signed_in(administrator))

    assert str(active.id) in rows
    assert str(deactivated.id) in rows
    assert str(administrator.id) in rows
    assert rows[str(deactivated.id)]["is_active"] is False
    assert rows[str(active.id)]["role"] == USER_ROLE_USER
    assert rows[str(administrator.id)]["role"] == USER_ROLE_ADMIN


def test_the_listing_carries_no_credential(session):
    """Asserted over the whole payload rather than against a named field.

    `assert "password_hash" not in row` would keep passing on the day somebody adds a column
    called something else. Comparing the key set to the contract catches anything new,
    including the thing nobody thought to forbid."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    make_user(session)

    rows = listed(signed_in(administrator))

    for row in rows.values():
        assert set(row) == VISIBLE_FIELDS


def test_a_changed_account_is_returned_without_a_credential(session):
    """The PATCH response is built by the same narrowing as the listing, and says so here —
    a second serializer is a second chance to spread an ORM row into a body."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(signed_in(administrator), target.id, {"role": USER_ROLE_ADMIN})

    assert response.status_code == 200
    assert set(response.json()) == VISIBLE_FIELDS


# --- changing an account --------------------------------------------------------------


def test_an_administrator_can_promote_an_account(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(signed_in(administrator), target.id, {"role": USER_ROLE_ADMIN})

    assert response.status_code == 200
    assert response.json()["role"] == USER_ROLE_ADMIN
    assert stored(session, target.id).role == USER_ROLE_ADMIN


def test_an_administrator_can_deactivate_and_reactivate_an_account(session):
    """Deactivation is `is_active`, matching what the column already means. Nothing is deleted
    and nothing new is invented: the account stays on file so the analyses it submitted remain
    attributable, which is why `User` has the flag in the first place."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    client = signed_in(administrator)

    deactivated = patch(client, target.id, {"is_active": False})

    assert deactivated.status_code == 200
    assert deactivated.json()["is_active"] is False
    assert stored(session, target.id).is_active is False

    reactivated = patch(client, target.id, {"is_active": True})

    assert reactivated.status_code == 200
    assert stored(session, target.id).is_active is True


def test_a_deactivated_account_can_no_longer_sign_in(session):
    """What deactivation is actually for, asserted end to end rather than as a stored boolean.
    `authenticate` folds activity into the lookup, so the account matches nothing — and gets
    the same 401 as an address that never existed."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    assert patch(signed_in(administrator), target.id, {"is_active": False}).status_code == 200

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login", json={"email": target.email, "password": PASSWORD}
        )

    assert response.status_code == 401


def test_one_change_leaves_the_other_field_alone(session):
    """What makes this a PATCH. The page's role control and its activation control are two
    separate submissions, and neither may carry the other's value back as a side effect."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN, is_active=False)

    response = patch(signed_in(administrator), target.id, {"is_active": True})

    assert response.status_code == 200
    assert stored(session, target.id).role == USER_ROLE_ADMIN


def test_an_empty_change_alters_nothing(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(signed_in(administrator), target.id, {})

    assert response.status_code == 200
    assert stored(session, target.id).role == USER_ROLE_USER
    assert stored(session, target.id).is_active is True


def test_an_account_that_does_not_exist_is_a_404(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = patch(signed_in(administrator), uuid.uuid4(), {"role": USER_ROLE_ADMIN})

    assert response.status_code == 404


# --- validation -----------------------------------------------------------------------


@pytest.mark.parametrize("role", ["Admin", "admin", "superuser", "", "ADMIN "])
def test_a_role_this_system_does_not_have_is_refused(session, role):
    """`"Admin"` is the one that matters. It is the spelling a person types, it is not the
    spelling the column means, and stored unchecked it would strip an administrator of every
    privilege while looking exactly like the request that granted them."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(signed_in(administrator), target.id, {"role": role})

    assert response.status_code == 422
    assert stored(session, target.id).role == USER_ROLE_USER


def test_a_field_this_endpoint_does_not_have_is_refused(session):
    """A caller who thinks `{"active": false}` deactivates somebody is told the word is wrong,
    rather than handed a 200 and an account that still works."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(signed_in(administrator), target.id, {"active": False})

    assert response.status_code == 422
    assert stored(session, target.id).is_active is True


def test_a_password_hash_cannot_be_written_through_this_route(session):
    """The same refusal as above, aimed at the field that would matter most: this route is the
    only one an administrator can name another account on, and it must not be a way to set a
    credential on one."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    original = stored(session, target.id).password_hash

    response = patch(
        signed_in(administrator), target.id, {"password_hash": hash_password("chosen")}
    )

    assert response.status_code == 422
    assert stored(session, target.id).password_hash == original


# --- the two safety rules -------------------------------------------------------------


def test_an_administrator_cannot_demote_themselves(session):
    """The change nobody can undo for them: after it, the route that would put it back is one
    they can no longer reach."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = patch(signed_in(administrator), administrator.id, {"role": USER_ROLE_USER})

    assert response.status_code == 400
    assert stored(session, administrator.id).role == USER_ROLE_ADMIN


def test_an_administrator_cannot_deactivate_themselves(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = patch(signed_in(administrator), administrator.id, {"is_active": False})

    assert response.status_code == 400
    assert stored(session, administrator.id).is_active is True


def test_the_self_modification_rule_is_checked_before_the_body(session):
    """Refused on who the target is, not on what was asked. An administrator who sends their
    own id with an empty body is still refused, so the rule cannot be walked around by finding
    a change that happens to be a no-op."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    assert patch(signed_in(administrator), administrator.id, {}).status_code == 400


def test_an_administrator_can_be_demoted_while_another_one_remains(session):
    """The invariant's other side, and the one that is reachable over HTTP: demoting an
    administrator is an ordinary, permitted change whenever it leaves somebody in the role."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN)

    response = patch(signed_in(administrator), target.id, {"role": USER_ROLE_USER})

    assert response.status_code == 200
    assert stored(session, target.id).role == USER_ROLE_USER


def test_the_last_active_administrator_cannot_be_demoted(session, sole_administrators):
    """The minimum-administrator invariant, asserted at the level where it is decidable.

    It cannot be reached over HTTP today, and the reason is the self-modification rule rather
    than anything about the invariant: every caller of this route is an active administrator
    and cannot be its target, so one always survives. Asserting it through the client would
    mean asserting it cannot happen, which is a test of the wrong rule — and one that would go
    on passing after somebody removed this check.

    So the route function is called directly, with the one state the two rules leave open: an
    administrator whose own account was deactivated *after* their session was resolved. That
    is not a contrivance. `require_user` reads activity when the session is looked up, and a
    second administrator deactivating them a moment later leaves exactly this in flight for
    the width of one request.

    `sole_administrators` has put every administrator the suite did not create out of the
    count, so "the last one" is a fact about this test's two accounts rather than about
    whatever else the database happens to hold.
    """
    db, _ = session
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN)

    # The race, made explicit: the acting administrator is no longer active, so the target is
    # the only active administrator left in the table.
    db.query(User).filter(User.id == administrator.id).update({"is_active": False})
    db.commit()

    with pytest.raises(Exception) as raised:
        update_user(
            user_id=target.id,
            change=UserChange(role=USER_ROLE_USER),
            administrator=administrator,
            session=db,
        )

    assert getattr(raised.value, "status_code", None) == 400
    assert stored(session, target.id).role == USER_ROLE_ADMIN


def test_the_last_active_administrator_cannot_be_deactivated(session, sole_administrators):
    """The same invariant, reached by the other field. Deactivating the last administrator
    empties the role just as demoting them does, and a check written only against `role` would
    let this one straight through."""
    db, _ = session
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN)

    db.query(User).filter(User.id == administrator.id).update({"is_active": False})
    db.commit()

    with pytest.raises(Exception) as raised:
        update_user(
            user_id=target.id,
            change=UserChange(is_active=False),
            administrator=administrator,
            session=db,
        )

    assert getattr(raised.value, "status_code", None) == 400
    assert stored(session, target.id).is_active is True


def test_a_deactivated_administrator_does_not_hold_the_role_open(session, sole_administrators):
    """An administrator who cannot sign in is not an administrator the invariant may count.
    Counting them would let the last *usable* administrator be demoted while a deactivated row
    stood in for them, which is a system with nobody able to administer it and a check that
    says otherwise."""
    db, _ = session
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN)
    make_user(session, role=USER_ROLE_ADMIN, is_active=False)

    db.query(User).filter(User.id == administrator.id).update({"is_active": False})
    db.commit()

    with pytest.raises(Exception) as raised:
        update_user(
            user_id=target.id,
            change=UserChange(role=USER_ROLE_USER),
            administrator=administrator,
            session=db,
        )

    assert getattr(raised.value, "status_code", None) == 400


# --- creating an account ----------------------------------------------------------------


def test_an_anonymous_caller_cannot_create_an_account(database):
    with TestClient(app) as client:
        response = create(client, new_account_body())

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_ordinary_user_cannot_create_an_account(session):
    """403 and not 401, for the reason the listing gives it: the session is good, the role is
    not. This is also the refusal that keeps account creation from being a way around the role
    system — an ordinary user who could create an account could create an administrator."""
    response = create(signed_in(make_user(session)), new_account_body(role=USER_ROLE_ADMIN))

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


def test_a_creation_from_an_unaccepted_origin_is_refused(session):
    """The CSRF check the other dashboard mutations carry. What a forged form would be driving
    here is the creation of an account, and `role` is on the body."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body(role=USER_ROLE_ADMIN)

    response = create(signed_in(administrator), body, origin="http://evil.test")

    assert response.status_code == 403
    assert stored_by_email(session, body["email"]) is None


def test_an_administrator_can_create_an_account(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body()

    response = create(signed_in(administrator), body)

    assert response.status_code == 201
    account = created(session, response)
    assert account.email == body["email"]
    assert account.role == USER_ROLE_USER
    assert account.is_active is True


def test_a_created_account_defaults_to_an_ordinary_active_user(session):
    """The defaults lean the safe way. A creation that forgot to send a role creates a user,
    never an administrator — the same direction the column's server default leans."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = create(signed_in(administrator), new_account_body())

    assert response.status_code == 201
    assert response.json()["role"] == USER_ROLE_USER
    assert response.json()["is_active"] is True
    created(session, response)


def test_a_created_account_may_be_an_administrator(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = create(
        signed_in(administrator), new_account_body(role=USER_ROLE_ADMIN, is_active=False)
    )

    assert response.status_code == 201
    account = created(session, response)
    assert account.role == USER_ROLE_ADMIN
    assert account.is_active is False


def test_a_created_account_can_sign_in_with_the_password_that_was_set(session):
    """What proves the password was stored usefully, and it is proved end to end rather than by
    inspecting the column. A test that asserted the hash starts with `$argon2` would pass on a
    row whose hash was of the wrong string; signing in is the only assertion that cannot."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body()

    response = create(signed_in(administrator), body)
    created(session, response)

    with TestClient(app) as client:
        signin = client.post(
            "/api/v1/auth/login", json={"email": body["email"], "password": body["password"]}
        )

    assert signin.status_code == 200


def test_a_created_account_stores_only_a_hash_of_the_password(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body()

    account = created(session, create(signed_in(administrator), body))

    assert body["password"] not in account.password_hash
    assert account.password_hash != body["password"]
    assert verify_password(account.password_hash, body["password"])


def test_a_creation_response_carries_no_credential(session):
    """The creation response is built by the same narrowing the listing is, and says so here —
    a second serializer would be a second chance to spread an ORM row into a body, and this is
    the one route whose request object holds a plaintext password."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = create(signed_in(administrator), new_account_body())

    assert set(response.json()) == VISIBLE_FIELDS
    created(session, response)


def test_a_duplicate_address_is_refused(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    existing = make_user(session)

    response = create(signed_in(administrator), new_account_body(email=existing.email))

    assert response.status_code == 409


def test_a_duplicate_address_is_refused_whatever_its_case(session):
    """The unique index is over the normalized form, so the refusal has to be too. Without
    normalization in the validator the check would find nothing and the insert would then
    collide — a 500 where a 409 belongs."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    existing = make_user(session)

    response = create(
        signed_in(administrator), new_account_body(email=f"  {existing.email.upper()}  ")
    )

    assert response.status_code == 409


def test_a_created_address_is_stored_normalized(session):
    """Stored as login will look it up. An account stored `Alice@Example.com` and looked up
    `alice@example.com` is an account that exists and cannot be signed into."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    raw = f"  {uuid.uuid4().hex.upper()}@Example.COM  "

    response = create(signed_in(administrator), new_account_body(email=raw))

    assert response.status_code == 201
    account = created(session, response)
    assert account.email == raw.strip().lower()

    with TestClient(app) as client:
        signin = client.post(
            "/api/v1/auth/login", json={"email": raw, "password": PASSWORD}
        )

    assert signin.status_code == 200


@pytest.mark.parametrize("email", ["", "   "])
def test_a_blank_address_is_refused(session, email):
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    response = create(signed_in(administrator), new_account_body(email=email))

    assert response.status_code == 422


def test_an_over_long_address_is_refused(session):
    """Refused by the route rather than by the column, so it is a 422 naming the field instead
    of a database error nobody catches."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    local = "a" * 320

    response = create(signed_in(administrator), new_account_body(email=f"{local}@example.com"))

    assert response.status_code == 422


def test_a_password_below_the_floor_is_refused(session):
    """The floor is `MINIMUM_PASSWORD_LENGTH` from `app.web_auth`, shared with the bootstrap
    script so the two ways of setting a password cannot disagree about what one may be."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body(password="a" * (MINIMUM_PASSWORD_LENGTH - 1))

    response = create(signed_in(administrator), body)

    assert response.status_code == 422
    assert stored_by_email(session, body["email"]) is None


@pytest.mark.parametrize("role", ["Admin", "admin", "superuser", ""])
def test_a_created_account_cannot_hold_a_role_this_system_does_not_have(session, role):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body(role=role)

    response = create(signed_in(administrator), body)

    assert response.status_code == 422
    assert stored_by_email(session, body["email"]) is None


def test_a_creation_cannot_name_a_password_hash(session):
    """`extra="forbid"` aimed at the field that would matter most: a caller who could supply
    their own `password_hash` could create an account whose credential the server never saw."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body()
    body["password_hash"] = hash_password("chosen")

    response = create(signed_in(administrator), body)

    assert response.status_code == 422
    assert stored_by_email(session, body["email"]) is None


def test_a_created_account_is_recorded(session):
    """One event, carrying the state the account came into existence with — and in the `{old,
    new}` shape every other event uses, because the audit screen rejects the whole listing when
    one row is not that shape."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    body = new_account_body(role=USER_ROLE_ADMIN, is_active=False)

    account = created(session, create(signed_in(administrator), body))
    recorded = events_for(session, account.id)

    assert len(recorded) == 1
    event = recorded[0]
    assert event.action == AUDIT_ACTION_USER_CREATED
    assert event.target_type == AUDIT_TARGET_USER
    assert event.actor_id == administrator.id
    assert event.actor_email_snapshot == administrator.email
    assert event.target_email_snapshot == account.email
    assert event.changes == {
        "email": {"old": None, "new": account.email},
        "role": {"old": None, "new": USER_ROLE_ADMIN},
        "is_active": {"old": None, "new": False},
    }


def test_a_refused_creation_records_nothing(session):
    """The event is added to the same session as the row, so a refusal that raises before the
    commit takes both with it. Asserted against the account the duplicate collided with, which
    is the only id a refused creation leaves anything to look for."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    existing = make_user(session)

    assert create(signed_in(administrator), new_account_body(email=existing.email)).status_code == 409
    assert events_for(session, existing.id) == []


# --- reading one account ------------------------------------------------------------------


def test_an_anonymous_caller_cannot_read_an_account(session):
    target = make_user(session)

    with TestClient(app) as client:
        response = detail(client, target.id)

    assert response.status_code == 401


def test_an_ordinary_user_cannot_read_an_account(session):
    caller = make_user(session)
    target = make_user(session)

    response = detail(signed_in(caller), target.id)

    assert response.status_code == 403


def test_an_administrator_can_read_one_account(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN, is_active=False)

    response = detail(signed_in(administrator), target.id)

    assert response.status_code == 200
    assert response.json()["id"] == str(target.id)
    assert response.json()["email"] == target.email
    assert response.json()["role"] == USER_ROLE_ADMIN
    assert response.json()["is_active"] is False


def test_one_account_is_returned_without_a_credential(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = detail(signed_in(administrator), target.id)

    assert set(response.json()) == VISIBLE_FIELDS


def test_reading_an_account_that_does_not_exist_is_a_404(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    assert detail(signed_in(administrator), uuid.uuid4()).status_code == 404


# --- changing an address --------------------------------------------------------------


def test_an_administrator_can_change_an_address(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    moved = f"{uuid.uuid4().hex}@example.com"

    response = patch(signed_in(administrator), target.id, {"email": moved})

    assert response.status_code == 200
    assert response.json()["email"] == moved
    assert stored(session, target.id).email == moved


def test_a_changed_address_is_stored_normalized(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    raw = f"  {uuid.uuid4().hex.upper()}@Example.COM  "

    response = patch(signed_in(administrator), target.id, {"email": raw})

    assert response.status_code == 200
    assert stored(session, target.id).email == raw.strip().lower()


def test_the_old_address_stops_working_and_the_new_one_starts(session):
    """The account is renamed, not duplicated. `authenticate` looks up the stored form and there
    is only ever one, so the change is total the moment it commits."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    was = target.email
    moved = f"{uuid.uuid4().hex}@example.com"

    assert patch(signed_in(administrator), target.id, {"email": moved}).status_code == 200

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/login", json={"email": was, "password": PASSWORD}
            ).status_code
            == 401
        )

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/login", json={"email": moved, "password": PASSWORD}
            ).status_code
            == 200
        )


def test_an_address_already_in_use_is_refused(session):
    """A 409, and nothing written. Renaming one account onto another's address would otherwise
    be the one way this route could make two accounts answer to one sign-in."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    occupant = make_user(session)

    response = patch(signed_in(administrator), target.id, {"email": occupant.email})

    assert response.status_code == 409
    assert stored(session, target.id).email != occupant.email
    assert events_for(session, target.id) == []


def test_an_address_already_in_use_is_refused_whatever_its_case(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    occupant = make_user(session)

    response = patch(signed_in(administrator), target.id, {"email": occupant.email.upper()})

    assert response.status_code == 409


def test_resubmitting_an_account_s_own_address_is_a_no_op(session):
    """An account is never told its own address is taken. `email_taken` excludes the row being
    edited, and the diff is empty, so nothing is written and nothing is recorded."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = patch(signed_in(administrator), target.id, {"email": target.email.upper()})

    assert response.status_code == 200
    assert stored(session, target.id).email == target.email
    assert events_for(session, target.id) == []


def test_changing_an_address_leaves_the_other_fields_alone(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN, is_active=False)

    response = patch(
        signed_in(administrator), target.id, {"email": f"{uuid.uuid4().hex}@example.com"}
    )

    assert response.status_code == 200
    assert stored(session, target.id).role == USER_ROLE_ADMIN
    assert stored(session, target.id).is_active is False


def test_changing_an_address_does_not_end_the_account_s_sessions(session):
    """A session authenticates an account by id. Signing somebody out because an operator fixed
    a typo in their address would be a revocation with no security question behind it."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    theirs = signed_in(target)

    assert open_session_count(session, target.id) == 1
    assert patch(
        signed_in(administrator), target.id, {"email": f"{uuid.uuid4().hex}@example.com"}
    ).status_code == 200

    assert open_session_count(session, target.id) == 1
    assert theirs.get("/api/v1/auth/me").status_code == 200


@pytest.mark.parametrize("email", ["", "   "])
def test_a_blank_address_cannot_be_patched_in(session, email):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    was = target.email

    response = patch(signed_in(administrator), target.id, {"email": email})

    assert response.status_code == 422
    assert stored(session, target.id).email == was


def test_a_change_of_address_is_recorded_as_a_diff(session):
    """Old and new both on the row. A log that recorded only the new address could not say what
    the account used to be called, which is the question somebody reading it is asking."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    was = target.email
    moved = f"{uuid.uuid4().hex}@example.com"

    assert patch(signed_in(administrator), target.id, {"email": moved}).status_code == 200

    recorded = events_for(session, target.id)

    assert len(recorded) == 1
    assert recorded[0].action == AUDIT_ACTION_USER_UPDATED
    assert recorded[0].changes == {"email": {"old": was, "new": moved}}
    # The snapshot is the address at the moment of the event — the old one. The new one is in
    # the diff, and a row holding only the new address would lose the pairing.
    assert recorded[0].target_email_snapshot == was


def test_one_event_covers_an_address_a_role_and_an_activation_together(session):
    """Three fields moved by one request is one row's worth of change, however many of them
    moved — and the diff carries all three rather than the action name implying one."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    moved = f"{uuid.uuid4().hex}@example.com"

    response = patch(
        signed_in(administrator),
        target.id,
        {"email": moved, "role": USER_ROLE_ADMIN, "is_active": False},
    )

    assert response.status_code == 200
    recorded = events_for(session, target.id)
    assert len(recorded) == 1
    assert set(recorded[0].changes) == {"email", "role", "is_active"}


def test_the_self_modification_rule_still_fires_on_an_address(session):
    """The rule is about who the target is, not about what was asked, and adding a third field
    to the body must not have opened a way round it."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    was = administrator.email

    response = patch(
        signed_in(administrator),
        administrator.id,
        {"email": f"{uuid.uuid4().hex}@example.com"},
    )

    assert response.status_code == 400
    assert stored(session, administrator.id).email == was


def test_the_last_active_administrator_can_still_be_renamed(session, sole_administrators):
    """The minimum-administrator invariant is about the role and the activation. An address has
    no bearing on how many administrators there are, and a check widened to refuse this would be
    refusing a change that breaks nothing."""
    db, _ = session
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN)

    db.query(User).filter(User.id == administrator.id).update({"is_active": False})
    db.commit()

    moved = f"{uuid.uuid4().hex}@example.com"
    result = update_user(
        user_id=target.id,
        change=UserChange(email=moved),
        administrator=administrator,
        session=db,
    )

    assert result.email == moved
    assert stored(session, target.id).role == USER_ROLE_ADMIN


# --- resetting a password -----------------------------------------------------------------


def test_an_anonymous_caller_cannot_reset_a_password(session):
    target = make_user(session)
    was = stored(session, target.id).password_hash

    with TestClient(app) as client:
        response = reset(client, target.id, NEW_PASSWORD)

    assert response.status_code == 401
    assert stored(session, target.id).password_hash == was


def test_an_ordinary_user_cannot_reset_a_password(session):
    """The refusal that matters most on this route: without it, taking over any account in the
    deployment is a request any signed-in person can make."""
    caller = make_user(session)
    target = make_user(session)
    was = stored(session, target.id).password_hash

    response = reset(signed_in(caller), target.id, NEW_PASSWORD)

    assert response.status_code == 403
    assert stored(session, target.id).password_hash == was


def test_a_reset_from_an_unaccepted_origin_is_refused(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    was = stored(session, target.id).password_hash

    response = reset(signed_in(administrator), target.id, NEW_PASSWORD, origin="http://evil.test")

    assert response.status_code == 403
    assert stored(session, target.id).password_hash == was


def test_an_administrator_can_reset_a_password(session):
    """The whole outcome, asserted through the door rather than against the column: the old
    password stops working and the new one starts."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = reset(signed_in(administrator), target.id, NEW_PASSWORD)

    assert response.status_code == 204
    assert response.content == b"", "204 means no body, not a serialized null"

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/login", json={"email": target.email, "password": PASSWORD}
            ).status_code
            == 401
        )

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/login", json={"email": target.email, "password": NEW_PASSWORD}
            ).status_code
            == 200
        )


def test_a_reset_stores_only_a_hash(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    assert reset(signed_in(administrator), target.id, NEW_PASSWORD).status_code == 204

    account = stored(session, target.id)
    assert NEW_PASSWORD not in account.password_hash
    assert verify_password(account.password_hash, NEW_PASSWORD)
    assert not verify_password(account.password_hash, PASSWORD)


def test_a_reset_revokes_every_session_the_account_held(session):
    """The half of the operation that makes it a reset rather than a rename. A credential
    replaced because it may have been compromised is not replaced at all if the sessions opened
    with it keep working — so the revocation is asserted at the door as well as in the table."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    theirs = signed_in(target)

    assert theirs.get("/api/v1/auth/me").status_code == 200
    assert open_session_count(session, target.id) == 1

    assert reset(signed_in(administrator), target.id, NEW_PASSWORD).status_code == 204

    assert open_session_count(session, target.id) == 0
    assert theirs.get("/api/v1/auth/me").status_code == 401


def test_a_reset_does_not_touch_another_account_s_sessions(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    bystander = make_user(session)
    signed_in(bystander)

    assert reset(signed_in(administrator), target.id, NEW_PASSWORD).status_code == 204

    assert open_session_count(session, bystander.id) == 1


def test_a_reset_leaves_the_account_s_other_fields_alone(session):
    """A reset changes the credential and nothing else. `extra="forbid"` on `PasswordReset` is
    what makes that structural: there is no field on the body that could carry a role."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN, is_active=False)
    was = target.email

    assert reset(signed_in(administrator), target.id, NEW_PASSWORD).status_code == 204

    account = stored(session, target.id)
    assert account.email == was
    assert account.role == USER_ROLE_ADMIN
    assert account.is_active is False


def test_a_reset_cannot_carry_anything_but_a_password(session):
    """`extra="forbid"` on the one route that writes a credential. A caller who could smuggle a
    `role` into a reset could grant themselves administrative access through the endpoint whose
    audit event says only that a password moved."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    response = signed_in(administrator).post(
        f"{USERS_URL}/{target.id}/password",
        json={"password": NEW_PASSWORD, "role": USER_ROLE_ADMIN},
        headers={"Origin": DASHBOARD_ORIGIN},
    )

    assert response.status_code == 422
    assert stored(session, target.id).role == USER_ROLE_USER


def test_a_reset_password_below_the_floor_is_refused(session):
    """The same floor creation enforces. A floor that differed between "set at creation" and
    "reset afterwards" would mean the weaker of the two was the one that actually held."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    was = stored(session, target.id).password_hash

    response = reset(signed_in(administrator), target.id, "a" * (MINIMUM_PASSWORD_LENGTH - 1))

    assert response.status_code == 422
    assert stored(session, target.id).password_hash == was


def test_resetting_a_password_for_an_account_that_does_not_exist_is_a_404(session):
    administrator = make_user(session, role=USER_ROLE_ADMIN)

    assert reset(signed_in(administrator), uuid.uuid4(), NEW_PASSWORD).status_code == 404


def test_an_administrator_may_reset_their_own_password_and_is_signed_out(session):
    """Not covered by the self-modification rule, which exists for the changes nobody can undo
    for them. They chose the new password and can sign in with it immediately — and their own
    session goes with the rest, which is the point rather than a side effect."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    theirs = signed_in(administrator)

    assert reset(theirs, administrator.id, NEW_PASSWORD).status_code == 204

    assert open_session_count(session, administrator.id) == 0
    assert theirs.get("/api/v1/auth/me").status_code == 401

    with TestClient(app) as client:
        assert (
            client.post(
                "/api/v1/auth/login",
                json={"email": administrator.email, "password": NEW_PASSWORD},
            ).status_code
            == 200
        )


def test_a_reset_is_recorded_as_one_event_carrying_no_password(session):
    """One event, in the `{old, new}` shape the audit screen can parse, saying that the
    credential moved and what it cost the account. Neither the plaintext nor the stored hash is
    anywhere on the row."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    signed_in(target)

    assert reset(signed_in(administrator), target.id, NEW_PASSWORD).status_code == 204

    recorded = events_for(session, target.id)

    assert len(recorded) == 1
    event = recorded[0]
    assert event.action == AUDIT_ACTION_USER_PASSWORD_RESET
    assert event.target_type == AUDIT_TARGET_USER
    assert event.actor_id == administrator.id
    assert event.target_email_snapshot == target.email
    assert event.changes == {
        "password_reset": {"old": False, "new": True},
        "sessions_revoked": {"old": 1, "new": 0},
    }


def test_the_hash_the_revocation_and_the_event_arrive_together(session):
    """The atomicity requirement, asserted as the three outcomes it produces rather than by
    inspecting the transaction. If any of them had been committed separately, one of these three
    assertions would be able to disagree with the other two."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    signed_in(target)

    assert reset(signed_in(administrator), target.id, NEW_PASSWORD).status_code == 204

    assert verify_password(stored(session, target.id).password_hash, NEW_PASSWORD)
    assert open_session_count(session, target.id) == 0
    assert len(events_for(session, target.id)) == 1


def test_a_refused_reset_writes_none_of_the_three(session):
    """The other half of the same property. A 404 raises before anything is added to the
    session, so the account keeps its password, its sessions and its empty log."""
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    signed_in(target)
    was = stored(session, target.id).password_hash

    assert reset(signed_in(administrator), target.id, "short").status_code == 422

    assert stored(session, target.id).password_hash == was
    assert open_session_count(session, target.id) == 1
    assert events_for(session, target.id) == []


# --- what no event ever carries -----------------------------------------------------------


def test_no_audit_event_ever_carries_a_password(session):
    """Searched over the serialized payload rather than against named fields.

    `assert "password" not in event.changes` would keep passing on the day somebody records a
    credential under another key, and checking the plaintext alone would miss the Argon2id hash.
    This drives all three writing routes, then looks for either form of either password anywhere
    in the JSON the row actually holds — which is what a reader of the audit screen sees.
    """
    administrator = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(administrator)
    body = new_account_body()

    account = created(session, create(client, body))
    assert patch(client, account.id, {"role": USER_ROLE_ADMIN}).status_code == 200
    assert reset(client, account.id, NEW_PASSWORD).status_code == 204

    recorded = events_for(session, account.id)
    assert len(recorded) == 3, "all three routes must have written exactly one event each"

    stored_hash = stored(session, account.id).password_hash

    for event in recorded:
        payload = json.dumps(event.changes)
        assert PASSWORD not in payload
        assert NEW_PASSWORD not in payload
        assert stored_hash not in payload
        assert "password_hash" not in payload
