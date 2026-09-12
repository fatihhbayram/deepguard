"""What gets recorded when an administrator changes an account, and what deliberately is not.

The sibling of `test_admin_users.py` and written to its conventions: real PostgreSQL, real
sessions opened through the real login route, one client per account, and every row this file
creates removed in constraint order afterwards. A role check demonstrated against a stubbed
session proves only that the stub agreed.

The claims here divide into two halves.

*What the write side does.* This is the half that matters, because an audit log is only worth
reading if the rows in it correspond exactly to the things that happened. Three cases:

- A change that sticks writes **exactly one** event, carrying only the fields that actually
  moved, with the actor taken from the session rather than from anything the caller sent.
- A request that changes nothing writes **zero**. The dashboard submits both controls on every
  save, so "save without touching anything" is the ordinary case, not a strange one, and a log
  full of those is a log whose real entries are buried.
- A request refused by an invariant writes **zero**, and specifically: the refusal and the
  event are in one transaction, so there is no arrangement in which the change is rejected and
  the record of it survives.

*What the read side does.* Authorization, and the one structural promise — the emails on the
payload are the snapshots frozen on the event row, not a live join to `users`. That is
asserted by renaming an account after the fact and checking that the event still says what it
said, which is the only way to tell the two implementations apart from outside.

Counting convention, inherited from `test_admin_analytics.py`: this table is deployment-wide
and other tests write to it, so nothing here asserts an absolute row count. Events are always
counted **for a specific target id** that this file created, which is unique per test.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import (
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
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

AUDIT_URL = "/api/v1/admin/audit"
USERS_URL = "/api/v1/admin/users"

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
    "actor_id",
    "actor_email_snapshot",
    "action",
    "target_type",
    "target_id",
    "target_email_snapshot",
    "changes",
    "request_id",
    "created_at",
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
    """A real session that removes every row this file created.

    The audit events go first for tidiness and not for correctness: nothing in
    `admin_audit_events` references an account, so the order here is free. `actor_id` is a
    bare UUID by design — see `AdminAuditEvent` — which is what lets an event outlive the
    person it names. This file writes a lot of events on purpose and clears up after itself
    so the shared table does not grow a run's worth of rows every time the suite is run.

    Sessions before users is the one ordering that is load-bearing:
    `auth_sessions.user_id` really is a foreign key.
    """
    users: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users

        db.rollback()
        for user_id in users:
            db.query(AdminAuditEvent).filter(
                (AdminAuditEvent.actor_id == user_id)
                | (AdminAuditEvent.target_id == str(user_id))
            ).delete(synchronize_session=False)
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
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


def administrator(session) -> TestClient:
    """A signed-in administrator, which is what every mutation below is made through."""
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def patch(client: TestClient, user: User, **body):
    """A change to an account, sent the way the dashboard sends one.

    The origin header is not optional: the route carries `require_same_origin`, and a request
    without it is refused before any of the logic this file is about.
    """
    return client.patch(
        f"{USERS_URL}/{user.id}", json=body, headers={"Origin": DASHBOARD_ORIGIN}
    )


def events_for(session, target: User) -> list[AdminAuditEvent]:
    """Every recorded event about one account, newest first, read straight from the database.

    Scoped to a target this file created — never a count of the whole table. Other tests write
    audit rows too, and an assertion about "how many events exist" would pass alone and fail
    in a suite.
    """
    db, _ = session
    db.commit()  # Drop this session's snapshot so the API's committed writes are visible.

    return list(
        db.execute(
            select(AdminAuditEvent)
            .where(AdminAuditEvent.target_id == str(target.id))
            .order_by(AdminAuditEvent.created_at.desc(), AdminAuditEvent.id.desc())
        )
        .scalars()
        .all()
    )


def count_for(session, target: User) -> int:
    """How many events stand against one account."""
    db, _ = session
    db.commit()

    return db.execute(
        select(func.count(AdminAuditEvent.id)).where(
            AdminAuditEvent.target_id == str(target.id)
        )
    ).scalar_one()


# --- the write side: a change that sticks ---------------------------------------------


def test_a_role_change_writes_exactly_one_event(session):
    """One request, one row. Not two, and not one per field."""
    client = administrator(session)
    target = make_user(session)

    assert patch(client, target, role=USER_ROLE_ADMIN).status_code == 200

    assert count_for(session, target) == 1


def test_an_event_records_only_the_fields_that_moved(session):
    """`changes` is a diff, not a restatement of the account.

    The request below sends `is_active` as well, at the value the account already holds. That
    field must not appear: a payload restating untouched fields would make every row read as a
    change to everything, and the no-op check next door is built on the same emptiness test.
    """
    client = administrator(session)
    target = make_user(session, role=USER_ROLE_USER, is_active=True)

    patch(client, target, role=USER_ROLE_ADMIN, is_active=True)

    changes = events_for(session, target)[0].changes

    assert changes == {"role": {"old": USER_ROLE_USER, "new": USER_ROLE_ADMIN}}


def test_an_event_records_both_fields_when_both_move(session):
    client = administrator(session)
    target = make_user(session, role=USER_ROLE_USER, is_active=True)

    patch(client, target, role=USER_ROLE_ADMIN, is_active=False)

    changes = events_for(session, target)[0].changes

    assert changes == {
        "role": {"old": USER_ROLE_USER, "new": USER_ROLE_ADMIN},
        "is_active": {"old": True, "new": False},
    }


def test_the_actor_is_the_authenticated_administrator(session):
    """Taken from the session `require_admin` resolved, never from the request.

    The body below names somebody else as an actor. The model forbids unknown fields, so the
    request is refused outright — which is the strongest form of the guarantee: there is no
    field a caller could set even if the route wanted to read one. The successful request that
    follows shows whose id actually lands on the row.
    """
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)
    target = make_user(session)

    impostor = patch(client, target, role=USER_ROLE_ADMIN, actor_id=str(target.id))
    assert impostor.status_code == 422, "the body must not be able to name an actor"

    patch(client, target, role=USER_ROLE_ADMIN)

    event = events_for(session, target)[0]

    assert event.actor_id == admin.id
    assert event.action == AUDIT_ACTION_USER_UPDATED
    assert event.target_type == AUDIT_TARGET_USER
    assert event.target_id == str(target.id)


def test_an_event_snapshots_both_addresses_as_they_read_at_the_time(session):
    admin = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    patch(signed_in(admin), target, role=USER_ROLE_ADMIN)

    event = events_for(session, target)[0]

    assert event.actor_email_snapshot == admin.email
    assert event.target_email_snapshot == target.email


def test_no_credential_material_reaches_the_event(session):
    """Asserted against the stored values rather than against field names.

    A test for `"password_hash" not in changes` would pass the day somebody nested it. The
    claim here is that the account's hash appears nowhere on the row at all.
    """
    admin = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)

    patch(signed_in(admin), target, is_active=False)

    event = events_for(session, target)[0]
    body = f"{event.changes}{event.actor_email_snapshot}{event.target_email_snapshot}"

    assert target.password_hash not in body
    assert admin.password_hash not in body


def test_an_event_survives_the_deletion_of_both_accounts_it_names(session):
    """The property the table is shaped for: an event outlives the people in it.

    Nothing in `admin_audit_events` is a foreign key — not the actor, not the target — so
    deleting both accounts is an ordinary DELETE that leaves the event exactly where it was.
    A reference would have made this impossible in one of three ways: `RESTRICT` refusing the
    deletion, `CASCADE` destroying the evidence, `SET NULL` erasing who did it. All three let
    account housekeeping rewrite history, which is the one thing an audit log may not permit.

    The row is still fully readable afterwards, because it never depended on those rows: the
    addresses were snapshotted onto the event itself.
    """
    db, users = session
    admin = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session)
    admin_email, target_email = admin.email, target.email

    patch(signed_in(admin), target, role=USER_ROLE_ADMIN)
    event_id = events_for(session, target)[0].id

    # Both accounts removed the way anything would remove them, with no audit cleanup first.
    for account in (admin, target):
        db.query(AuthSession).filter(AuthSession.user_id == account.id).delete()
        db.query(User).filter(User.id == account.id).delete()
        users.remove(account.id)
    db.commit()

    survivor = db.get(AdminAuditEvent, event_id)

    assert survivor is not None, "the event did not survive the accounts it names"
    assert survivor.actor_email_snapshot == admin_email
    assert survivor.target_email_snapshot == target_email
    assert survivor.actor_id == admin.id

    # This file's own rows, cleared up now that the fixture can no longer reach them by user.
    db.query(AdminAuditEvent).filter(AdminAuditEvent.id == event_id).delete()
    db.commit()


# --- the write side: what writes nothing ----------------------------------------------


def test_a_request_that_changes_nothing_writes_no_event(session):
    """The ordinary case, not a strange one: the dashboard submits both controls on every save.

    200, because the caller asked for a state and that is the state — nothing about "you asked
    for what was already true" is an error. But nothing is recorded, because nothing happened.
    """
    client = administrator(session)
    target = make_user(session, role=USER_ROLE_USER, is_active=True)

    response = patch(client, target, role=USER_ROLE_USER, is_active=True)

    assert response.status_code == 200
    assert response.json()["role"] == USER_ROLE_USER
    assert count_for(session, target) == 0


def test_an_empty_patch_writes_no_event(session):
    client = administrator(session)
    target = make_user(session)

    assert patch(client, target).status_code == 200
    assert count_for(session, target) == 0


def test_a_refused_self_modification_writes_no_event(session):
    """The refusal and the record are in one transaction, so a rejected change leaves nothing.

    Counted against the administrator's own id as the target, which is what the request asked
    to change.
    """
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)

    response = client.patch(
        f"{USERS_URL}/{admin.id}",
        json={"role": USER_ROLE_USER},
        headers={"Origin": DASHBOARD_ORIGIN},
    )

    assert response.status_code == 400
    assert count_for(session, admin) == 0


def test_a_change_refused_by_an_invariant_writes_no_event(session):
    """The last-active-administrator rule, reached where it is actually reachable.

    `update_user` refuses a self-change before this rule can bite, so the invariant is
    exercised the way `test_admin_users.py` exercises it: through the function directly, with
    an administrator acting on a *different* administrator who is the only other one left.
    Whatever the arrangement, the claim is the same — the request raises, and no row survives.
    """
    from fastapi import HTTPException

    from app.api.admin_users import UserChange, update_user

    admin = make_user(session, role=USER_ROLE_ADMIN)
    target = make_user(session, role=USER_ROLE_ADMIN)
    db, _ = session

    # Everybody else's administrators stood down, so `target` really is the last one besides
    # the actor — the suite shares a database and the invariant is a count over the whole
    # table.
    others = (
        db.execute(
            select(User).where(
                User.role == USER_ROLE_ADMIN,
                User.is_active.is_(True),
                User.id.notin_([admin.id, target.id]),
            )
        )
        .scalars()
        .all()
    )
    for other in others:
        other.is_active = False
    # The actor steps aside too, leaving `target` as the only active administrator: demoting
    # them is then the change the invariant exists to refuse.
    admin.is_active = False
    db.commit()

    try:
        with pytest.raises(HTTPException) as refusal:
            update_user(
                user_id=target.id,
                change=UserChange(role=USER_ROLE_USER),
                administrator=admin,
                session=db,
            )

        assert refusal.value.status_code == 400
        assert count_for(session, target) == 0
    finally:
        db.rollback()
        for other in others:
            other.is_active = True
        db.commit()


# --- the read side --------------------------------------------------------------------


def test_an_anonymous_caller_cannot_read_the_audit_log(database):
    with TestClient(app) as client:
        response = client.get(AUDIT_URL)

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_ordinary_user_cannot_read_the_audit_log(session):
    """403 and not 401. The session is perfectly good; the role is not, and the status says
    which — telling a signed-in person to sign in again would send them round a loop that
    cannot fix anything."""
    response = signed_in(make_user(session)).get(AUDIT_URL)

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


def test_a_listed_event_carries_the_recorded_fields_and_no_others(session):
    client = administrator(session)
    target = make_user(session)

    patch(client, target, role=USER_ROLE_ADMIN)

    rows = {row["target_id"]: row for row in client.get(AUDIT_URL).json()}
    row = rows[str(target.id)]

    assert set(row) == VISIBLE_FIELDS
    assert row["action"] == AUDIT_ACTION_USER_UPDATED
    assert row["target_type"] == AUDIT_TARGET_USER
    assert row["changes"] == {"role": {"old": USER_ROLE_USER, "new": USER_ROLE_ADMIN}}


def test_the_listing_is_newest_first(session):
    client = administrator(session)
    target = make_user(session, role=USER_ROLE_USER, is_active=True)

    patch(client, target, role=USER_ROLE_ADMIN)
    patch(client, target, is_active=False)

    mine = [row for row in client.get(AUDIT_URL).json() if row["target_id"] == str(target.id)]

    assert len(mine) == 2
    assert "is_active" in mine[0]["changes"], "the later change should come first"
    assert "role" in mine[1]["changes"]


def test_the_listing_reports_snapshots_rather_than_the_accounts_current_address(session):
    """The structural promise, asserted the only way it can be from outside.

    The target is renamed after the change was recorded. A read that joined `users` would now
    report the new address; the event is a statement about the past and has to keep saying what
    it said. This is what distinguishes the two implementations, and it is why the columns are
    duplicated at all.
    """
    client = administrator(session)
    target = make_user(session)
    original_email = target.email

    patch(client, target, role=USER_ROLE_ADMIN)

    db, _ = session
    target.email = f"renamed-{uuid.uuid4().hex}@example.com"
    db.commit()

    rows = {row["target_id"]: row for row in client.get(AUDIT_URL).json()}

    assert rows[str(target.id)]["target_email_snapshot"] == original_email
    assert rows[str(target.id)]["target_email_snapshot"] != target.email


# --- append-only ----------------------------------------------------------------------


def test_the_audit_surface_offers_nothing_but_reads(database):
    """Append-only asserted as a property of the route table, not as a promise in a comment.

    No route under `/api/v1/admin/audit` accepts anything but GET — today, or after somebody
    adds one to `admin_audit.py` without reading why it should not exist.

    Read off the OpenAPI schema rather than off `app.routes`. FastAPI defers an included
    router until the application starts, so the route list is a set of opaque `_IncludedRouter`
    objects with no paths on them until then — a scan of it finds nothing and the assertion
    would pass by looking at an empty list, which is the worst way for this particular test to
    be green. The schema is built from the resolved routes and is the honest source.
    """
    paths = app.openapi()["paths"]
    audit_paths = {
        path: spec for path, spec in paths.items() if path.startswith("/api/v1/admin/audit")
    }

    assert audit_paths, "the audit routes were not found"
    for path, spec in audit_paths.items():
        assert set(spec) == {"get"}, f"{path} accepts a mutation: {sorted(spec)}"
