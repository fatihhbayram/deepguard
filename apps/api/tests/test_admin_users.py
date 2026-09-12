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

Two conventions inherited from `test_dashboard_authorization.py`, which this file is the
sibling of: accounts are cleaned up in constraint order by the `session` fixture, and each
account gets its own client because two sessions in one cookie jar is a state no browser is
ever in.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.api.admin_users import UserChange, update_user
from app.db.models import USER_ROLE_ADMIN, USER_ROLE_USER, AuthSession, User
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

USERS_URL = "/api/v1/admin/users"

# A test fixture, not a credential: nothing outside this file uses it and no deployment ships
# it.
PASSWORD = "correct-horse-battery-staple"

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
