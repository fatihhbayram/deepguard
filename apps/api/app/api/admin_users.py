"""The routes an administrator manages other people's accounts through.

Mounted under `/api/v1/admin`, on the internal surface beside the dashboard's own routes and
deliberately not on `/api/public/v1`. The public API authenticates by API key and has no
notion of a person at all; account administration is something a signed-in administrator
does from the dashboard, and putting it anywhere a bearer token could reach would make a key
able to grant itself an account.

Nothing here invents a role system. `require_admin` is the same dependency the rest of the
application uses to decide whether a session may reach a privileged action, and
`USER_ROLE_USER` / `USER_ROLE_ADMIN` are the same two constants the column is written with —
a second spelling of "admin" living in this file would be a second answer to the only
question that matters here, and the day the two disagreed the wrong one would be the one
granting access.

What an account is reduced to on the way out is `AdminUser` below: five identity fields and
nothing else. The `User` row also carries an Argon2id password hash, and a serializer that
simply handed the ORM object to pydantic would put it on the wire — so the response model is
an explicit field list rather than a `from_attributes` passthrough of the whole row, and a
column added to `User` later cannot leak by being added.

Two rules constrain what a PATCH may do, and both are enforced here rather than in the
browser. They are stated in full on `update_user`.

Since R8-T5 a successful change also writes an `AdminAuditEvent` — in the same transaction,
so the account and the record of who changed it commit or roll back together. `update_user`
says what that means for the request that changes nothing and for the one that is refused.

R8-T8 completed the lifecycle: an account can now be created here, read on its own, have its
address changed, and have its password reset. Three things about that are decisions rather
than details, and all three are the reason this docstring is longer than it was.

**There is no delete.** Not a delete behind a confirmation, not a soft delete under another
name — no route, at all, that removes a `users` row. `media_files.uploader_id` references it
`ON DELETE RESTRICT`, so an account that has ever uploaded anything cannot be removed without
taking the forensic record with it; and `admin_audit_events.actor_id` and
`analysis_reviews.reviewer_id` keep a user's id precisely so that who did what survives. A
delete route would therefore be a control that fails with a constraint error on every account
worth deleting and destroys history on the rest. Deactivation is the removal this system has:
`is_active` false ends every way in — `authenticate` folds it into the lookup and
`session_user` folds it into session resolution — while every record the account is named on
stays attributable. The dashboard says "Deactivate" for that reason and not as a euphemism.

**A password is written and never read back.** `hash_password` is the only thing that touches
one, the Argon2id digest is the only form that is stored, and no response, no log line and no
audit payload in this module carries either the plaintext or the hash. `AdminUser` is an
explicit field list rather than a `from_attributes` passthrough, which is what makes that a
property of the code rather than of everyone remembering.

**Resetting a password ends the sessions it opened.** A credential that has been replaced
because it may have been compromised is not replaced at all if the sessions opened with it
keep working — so the reset writes the new hash, revokes every open `AuthSession` for that
account and records the event in one transaction. `reset_password` says why it must be one.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import (
    AUDIT_ACTION_USER_CREATED,
    AUDIT_ACTION_USER_PASSWORD_RESET,
    AUDIT_ACTION_USER_UPDATED,
    AUDIT_TARGET_USER,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    User,
)
from app.db.session import get_session
from app.observability import current_request_id
from app.web_auth import (
    MAX_PASSWORD_LENGTH,
    MINIMUM_PASSWORD_LENGTH,
    hash_password,
    normalize_email,
    require_admin,
    require_same_origin,
    revoke_user_sessions,
)

logger = logging.getLogger(__name__)

# The roles an account may be moved between, as the model spells them. A frozen set rather
# than a tuple so membership is the O(1) test it reads as, and built from the constants so
# that a third role added to `models.py` is a one-line change there rather than a string
# somebody has to remember is also written down over here.
ASSIGNABLE_ROLES = frozenset({USER_ROLE_USER, USER_ROLE_ADMIN})

# The widest address the column can hold — `String(320)` on `User.email`, which is RFC 3696's
# 64-character local part plus its 255-character domain plus the `@`. Read from the model's
# own column rather than written as `320` here, so the two cannot drift into a request that
# validates and then fails on insert.
MAX_EMAIL_LENGTH = User.__table__.c.email.type.length

# Every route below is behind `require_admin`, stated once on the router rather than repeated
# per route. A dependency listed per route is a dependency that can be forgotten on the next
# one added to this file, and the route it is forgotten on is an administrative route open to
# every signed-in account. Declared here it applies to whatever is mounted, including routes
# nobody has written yet.
#
# `require_admin` is layered on `require_user`, so the two refusals arrive in the order they
# should: no session at all is 401 and an authenticated non-administrator is 403.
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class AdminUser(BaseModel):
    """An account as an administrator is shown it.

    The five fields an operator needs to answer "who has an account here, what may they do,
    and can they still sign in" — and deliberately not one more. No password hash, no session
    token, no session id, no last-seen timestamp. The first two are credentials and the rest
    is a wider contract than the screen consuming it has any use for.

    Built field by field in `visible_user` rather than by `from_attributes`, so that this
    narrowing is something the code does rather than something it would stop doing if a
    column were added to `User`.
    """

    model_config = ConfigDict(from_attributes=False)

    id: uuid.UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime


def assignable_role(value: str | None) -> str | None:
    """Refuse any role this system does not have.

    Checked against `ASSIGNABLE_ROLES`, which is built from the model's own constants, so the
    set of things these endpoints can write is the set of things the column means. Without it
    a route would happily store `"Admin"`, `"superuser"` or the empty string — none of which
    grants anything, all of which would leave an account whose role reads like something and
    matches nothing, and one of which (`"Admin"`, the spelling a human will type) would
    silently strip an administrator of their access while looking like it had granted it.

    A function reused by two validators rather than a `Literal` annotation, because the allowed
    values are the constants and not two strings retyped here; the cost is that the refusal is
    a 422 from pydantic rather than a hand-written 400, which is the right status for a body
    whose field is not a value the field can take.
    """
    if value is not None and value not in ASSIGNABLE_ROLES:
        raise ValueError("role must be one of: " + ", ".join(sorted(ASSIGNABLE_ROLES)))

    return value


def usable_email(value: str | None) -> str | None:
    """The address as it will be stored: normalized, non-blank, and short enough to fit.

    **Normalized in the validator, which is what makes normalization unskippable.** Every route
    that accepts an address gets it back through `normalize_email` before any of them has run a
    line of its own, so there is no path through this module on which a raw `Alice@Example.com `
    reaches the uniqueness check or the column. That matters more than it looks: the unique
    index is over the normalized form, so a route that checked for duplicates before
    normalizing would find none and then insert a row that collides with one already there —
    or, worse on the update path, store an address whose own login lookup can never match it.

    Blankness is tested after normalization for the same reason `usable_name` in
    `admin_api_keys.py` strips first: `"   "` is not an address, and an account whose email
    column is empty is one nobody can sign into and nobody can identify in the listing.

    No syntax check beyond that. Not a regular expression and not pydantic's `EmailStr`: an
    address's real validity is whether mail reaches it, this application never sends any, and a
    pattern strict enough to be worth running is a pattern that rejects addresses that work.
    What is enforced is what the system actually depends on — one stored form, non-empty, and
    inside the column.
    """
    if value is None:
        return None

    email = normalize_email(value)

    if not email:
        raise ValueError("email must not be blank")

    if len(email) > MAX_EMAIL_LENGTH:
        raise ValueError(f"email must be at most {MAX_EMAIL_LENGTH} characters")

    return email


class UserChange(BaseModel):
    """What a PATCH may ask to change about an account.

    All three fields are optional and all three default to "leave it alone", which is what
    makes this a PATCH rather than a PUT: the dashboard's address field, its role control and
    its activation control are separate submissions, and each has to be able to travel without
    restating the others' current values. Sending none is a no-op that returns the account
    unchanged.

    `extra="forbid"` so a body naming a field this model does not have is refused rather than
    silently ignored. A caller who thinks they are deactivating somebody by sending
    `{"active": false}` should be told the word is wrong, not handed back a 200 and an
    account that still works. It is also what keeps `password` and `password_hash` off this
    route: a credential is set through `reset_password` below, where revoking the account's
    sessions is part of the same transaction, and a PATCH that could quietly write one would be
    a way to replace somebody's password while leaving their old sessions open.

    **There is no `password` here and there will not be one.** The two operations are separate
    on purpose — see `reset_password`.
    """

    model_config = ConfigDict(extra="forbid")

    email: str | None = None
    role: str | None = None
    is_active: bool | None = None

    # Both validators delegate to the module-level functions above, which `UserCreate` uses
    # too. The rules about what a role and an address may be are the same whether an account is
    # being created or edited, and two copies of them would be two rules with one of them
    # eventually lagging.
    @field_validator("role")
    @classmethod
    def check_role(cls, value: str | None) -> str | None:
        return assignable_role(value)

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str | None) -> str | None:
        return usable_email(value)


class UserCreate(BaseModel):
    """What a creation request says: an address, a password, and the account's starting state.

    `role` and `is_active` carry defaults rather than being required, and the defaults are the
    safe ones — an ordinary user who can sign in. A creation form that forgot to send a role
    creates a user, never an administrator, which is the same direction `User.role`'s server
    default leans for the same reason.

    `password` is bounded at both ends. The floor is `MINIMUM_PASSWORD_LENGTH` from
    `app.web_auth`, shared with `create_admin.py` so the bootstrap script and this route cannot
    disagree about what a password may be. The ceiling is there because Argon2 will faithfully
    spend time on however many megabytes it is handed, and an unbounded field would let the
    caller choose the server's workload — the same bound `Credentials` puts on a sign-in.

    **The plaintext is on this object and nowhere else.** It is hashed in `create_user`, the
    digest is what reaches the row, and this model is never logged, never returned and never
    put in an audit payload. `Field(repr=False)` is part of that rather than cosmetic: pydantic
    models print their fields, and a `logger.exception` that formatted a request body — or a
    traceback renderer that showed a local — would otherwise put the password in the log.

    `extra="forbid"` for the ordinary reason and a sharper one: it is what makes it impossible
    for a caller to name `password_hash` or `id`. A request that tried is a 422, not a request
    accepted with the field dropped.
    """

    model_config = ConfigDict(extra="forbid")

    email: str
    password: str = Field(
        min_length=MINIMUM_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH, repr=False
    )
    role: str = USER_ROLE_USER
    is_active: bool = True

    @field_validator("role")
    @classmethod
    def check_role(cls, value: str) -> str:
        return assignable_role(value)

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str) -> str:
        return usable_email(value)


class PasswordReset(BaseModel):
    """The new password, and nothing whatever else.

    A model of one field rather than a bare string body, so the route takes JSON like every
    other mutation here and so `extra="forbid"` can do its work: a caller cannot smuggle a
    `role`, an `is_active` or a `password_hash` into the one request that is allowed to write a
    credential. What a reset changes is the credential; anything else about the account is a
    PATCH, and keeping the two apart is what makes each route's audit event tell the truth.

    Bounded by the same two numbers `UserCreate` uses, because it is the same operation on the
    same column — a floor that differed between "set at creation" and "reset afterwards" would
    mean the weaker of the two was the one that actually held.

    `repr=False` for the reason it is on `UserCreate.password`: this object must not print its
    contents into a log line or a traceback.
    """

    model_config = ConfigDict(extra="forbid")

    password: str = Field(
        min_length=MINIMUM_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH, repr=False
    )


def visible_user(user: User) -> AdminUser:
    """The stored account narrowed to what an administrator is told about it.

    The one place a `User` becomes a response in this module, so there is one place to read
    to know what leaves it — and the field list is written out rather than spread from the
    ORM object, which is what keeps `password_hash` off the wire by construction instead of
    by everyone remembering to exclude it.
    """
    return AdminUser(
        id=user.id,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
    )


def refused(detail: str) -> HTTPException:
    """The 400 a change that would break a system invariant gets.

    400 and not 403: the caller is an administrator and is allowed to administer accounts —
    what they are not allowed to do is leave the system in this particular state. A 403 would
    read as "you may not do this kind of thing", which is untrue and would send an operator
    looking at their own permissions instead of at the change they asked for. The detail says
    which rule it was, because unlike an authentication failure there is nothing here worth
    keeping from the caller: they already know every account in the system.
    """
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def audit_event(
    administrator: User,
    target: User,
    action: str,
    changes: dict[str, dict[str, object]],
) -> AdminAuditEvent:
    """The record of something done to an account, ready to be added to the caller's session.

    Built here rather than at the three call sites so that the fields that must be the same on
    all of them — the actor, the target type, the request id, and above all the absence of
    anything secret — are written once.

    The actor is `administrator.id`, the account `require_admin` resolved the session to, and
    never anything the request body could name. All three body models forbid extra fields, so
    there is no field a caller could set even if this function wanted to read one.

    Both email snapshots are copies taken as they read right now; see `AdminAuditEvent` for why
    they are snapshots rather than a join. `target_email_snapshot` is the address the account
    had at the moment of the event, which on a change of address is the *old* one — the new one
    is in `changes`, and a row that recorded only the new address would make the log unable to
    say what it used to be.

    `changes` is `{"field": {"old": ..., "new": ...}}` and must stay that shape on every event
    this module writes. It is not a house style: `parseChanges` in `apps/web/app/admin/audit.ts`
    rejects a payload whose entries are not that pair, and it rejects the whole *listing* when
    one row fails — so a single flat payload written here would blank the audit screen for every
    event in the log, not just its own. The callers below say what they put in it; what none of
    them puts in it is a password, in either form.
    """
    return AdminAuditEvent(
        actor_id=administrator.id,
        actor_email_snapshot=administrator.email,
        action=action,
        target_type=AUDIT_TARGET_USER,
        target_id=str(target.id),
        target_email_snapshot=target.email,
        changes=changes,
        request_id=current_request_id(),
    )


def email_taken(session: Session, email: str, *, excluding: uuid.UUID | None = None) -> bool:
    """Whether some other account already answers to this address.

    Asked against the normalized form, which is the only form that ever reaches here —
    `usable_email` normalizes in the validator, before any route body runs — and which is the
    form the unique index is over. Comparing a raw address against that index would find
    nothing and then collide on insert.

    `excluding` is the account being edited. Without it a PATCH that resubmits an account's own
    current address would be told the address is taken, by itself.

    This is a check, not the enforcement. The unique index is, and the two callers both handle
    the `IntegrityError` that arrives when another request inserts the same address between
    this query and the commit. What the check buys is the difference between a 409 that names
    the problem and a 500 from a constraint nobody caught.
    """
    query = select(func.count(User.id)).where(User.email == email)

    if excluding is not None:
        query = query.where(User.id != excluding)

    return session.execute(query).scalar_one() > 0


def conflict(detail: str) -> HTTPException:
    """The 409 a request that collides with an account already on file gets.

    409 rather than the 400 `refused` gives, because the two are different failures and an
    operator reading the message should be able to tell which one they hit. `refused` means
    "this change would leave the system in a state it may not be in" — a rule about the
    outcome. This means "something is already there" — a fact about the present, which the
    caller can resolve by looking at the listing they are entitled to read in full.
    """
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def account_changes(
    user: User, email: str, role: str, is_active: bool
) -> dict[str, dict[str, object]]:
    """What this request would actually change about the account, field by field.

    Only the fields that genuinely move. A payload restating an untouched field would make
    every audit row read as a change to everything, and — more importantly — it would make the
    "did anything change" question below unanswerable, because a dict that always has three
    keys in it is never empty.

    Computed against the stored row, so it must be called *before* the mutation. After the
    assignment the old values are gone and every diff is empty.

    Booleans and strings go into this dict as themselves; it is serialized to JSON as the
    event's `changes` payload. The addresses that go in on a change of email are the account's
    own, old and new — personal data, and already in `target_email_snapshot` on every row this
    table holds, so recording the pair here widens nothing. Nothing in this dict is a hash or a
    token, and no route adds a password to it.
    """
    changes: dict[str, dict[str, object]] = {}

    if user.email != email:
        changes["email"] = {"old": user.email, "new": email}

    if user.role != role:
        changes["role"] = {"old": user.role, "new": role}

    if user.is_active != is_active:
        changes["is_active"] = {"old": user.is_active, "new": is_active}

    return changes


def other_active_admins(session: Session, user: User) -> int:
    """How many active administrators there would still be if this account were not one.

    Counted in the database rather than by loading the accounts and counting them here: the
    only thing the answer is used for is a comparison against zero, and fetching every
    administrator row to establish that would read a table to learn a number PostgreSQL can
    return.

    Exclusive of `user` by id, which is what makes the caller's arithmetic simple — the count
    is the floor that survives whatever happens to this account, so the invariant check adds
    one back only when the change leaves them an active administrator.
    """
    return session.execute(
        select(func.count(User.id)).where(
            User.role == USER_ROLE_ADMIN,
            User.is_active.is_(True),
            User.id != user.id,
        )
    ).scalar_one()


@router.get("/users", response_model=list[AdminUser])
def list_users(session: Session = Depends(get_session)) -> list[AdminUser]:
    """Every account in the system, oldest first.

    Unfiltered, and that is the point of the route: an administrator's question is "who has an
    account here", and a listing that hid deactivated accounts would hide exactly the ones
    somebody came here to reactivate. The `is_active` field carries the distinction instead,
    which lets the page show it rather than the API decide it.

    Ordered by `created_at` so the listing is stable between renders. Without an `ORDER BY`
    PostgreSQL is free to return the rows in whatever order it finds them, which changes as
    rows are updated — an operator would see the table reshuffle itself every time they
    changed somebody's role, and the row under the pointer would not be the row they clicked.
    The id is the tiebreaker, because two accounts created in the same transaction share a
    timestamp and a sort with ties in it is not a stable sort.

    No pagination. This is the account table of an internal deployment, not a customer-facing
    directory, and a limit-and-cursor scheme for a table of tens of rows would be machinery
    built for a scale that does not exist — added when it does, against a real number.
    """
    users = (
        session.execute(select(User).order_by(User.created_at, User.id)).scalars().all()
    )

    return [visible_user(user) for user in users]


@router.post(
    "/users",
    response_model=AdminUser,
    status_code=status.HTTP_201_CREATED,
    # A state change, so it carries the same origin check every other dashboard mutation does.
    # The session cookie is `SameSite=Lax` and would not be attached to a cross-site request in
    # the first place; this is the independent check behind that, made by the server that owns
    # the data. What a forged request would be driving here is the creation of an account — and
    # `role` is on the body, so the account it created could be an administrator.
    dependencies=[Depends(require_same_origin)],
)
def create_user(
    payload: UserCreate,
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUser:
    """Put a new account on file, with a password the administrator chose.

    An address, a password, and the two fields that say what the account starts as. There is no
    invitation and no email: R8-T8 does not send mail, and a route that pretended to would be a
    flow whose second half does not exist. The administrator sets the password and tells the
    person what it is, which is the honest version of what this deployment can actually do.

    **The plaintext exists in this process for the length of one request.** `hash_password`
    turns it into an Argon2id digest, the digest is what the row gets, and `payload` is not
    logged, not echoed and not put in the audit event. The response is an `AdminUser`, which
    has no password field at all — so there is no path by which this route could return the
    credential even if somebody later spread the ORM row into a body.

    **The address is already normalized by the time this function runs.** `usable_email` does it
    in the validator, so `email_taken` asks about the same form the unique index is over and the
    row is written in the form login will look it up by. A route that checked before normalizing
    would find no duplicate and then insert one.

    Two defences against a duplicate, and both are needed. The check answers the common case
    with a 409 that names the problem. The `IntegrityError` handler answers the race — two
    administrators creating the same address at once, where both checks pass and one insert
    loses — with the same 409 rather than a 500 out of a constraint nobody caught. The unique
    index is what actually enforces it; the check is what makes the refusal readable.

    **The account and the record of who created it commit together.** The event is added to this
    same session, so one `commit()` writes both or neither — the arrangement `update_user`
    established and `admin_api_keys.py` follows, and for the reason all three give: an audit row
    written in a second transaction is a row that survives a rollback of the thing it describes.

    `session.flush()` before the event because `target_id` and `target_email_snapshot` are read
    off the row. The id is generated in Python so it is available earlier, but flushing keeps
    the event built from the row as the session actually holds it — which is what stays correct
    if that default ever moves to the server.

    No last-administrator invariant here, and none is needed: creating an account cannot reduce
    a count. Creating an active administrator raises it, and creating anything else leaves it.
    """
    if email_taken(session, payload.email):
        # The address, because this caller is entitled to list every account in the system and
        # can already see which addresses exist. There is nothing to conceal from them, and a
        # refusal that would not say which field collided would send an operator guessing.
        raise conflict("An account with that email already exists.")

    user = User(
        email=payload.email,
        # The only line in this module that turns a password into anything, and the only form
        # of it that is ever stored.
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=payload.is_active,
    )
    session.add(user)

    try:
        session.flush()
    except IntegrityError:
        # The race the check above cannot close: another request inserted this address between
        # that query and this flush. The session is poisoned once a constraint has fired, so it
        # is rolled back before the refusal — otherwise the next statement on it, including the
        # one `get_session` issues on the way out, raises over the top of this.
        session.rollback()
        logger.info("A duplicate account creation was refused.")

        raise conflict("An account with that email already exists.") from None

    session.add(
        audit_event(
            administrator,
            user,
            AUDIT_ACTION_USER_CREATED,
            # The state the account came into existence with, written in the same old/new shape
            # a modification uses, with null standing for "there was no account". The uniform
            # shape is not cosmetic — see `audit_event`.
            #
            # No password and no hash. `is_active` and `role` are here because an account
            # created as an active administrator is a materially different event from one
            # created as a deactivated user, and the log should not require a reader to go and
            # look at the account's present state to find out which it was.
            {
                "email": {"old": None, "new": user.email},
                "role": {"old": None, "new": user.role},
                "is_active": {"old": None, "new": user.is_active},
            },
        )
    )

    session.commit()
    session.refresh(user)

    # The id and the role. Not the address — personal data that does not need to be in a log
    # line to make it actionable, and the audit row holds it — and above all not `payload`,
    # which carries the plaintext and is the object it would be easiest to format in here.
    logger.info(
        "Administrator %s created user %s with role=%s active=%s.",
        administrator.id,
        user.id,
        user.role,
        user.is_active,
    )

    return visible_user(user)


@router.get("/users/{user_id}", response_model=AdminUser)
def read_user(
    user_id: uuid.UUID, session: Session = Depends(get_session)
) -> AdminUser:
    """One account, for the screen that shows a single account.

    The same five fields the listing gives, through the same `visible_user` narrowing — a second
    serializer would be a second chance to spread an ORM row into a body, and this is the row
    with the password hash on it.

    It exists rather than having the detail page filter the listing, because a page that found
    its account by scanning a list of every account in the deployment would be reading the whole
    table to render one row, and would have no way to tell "this id does not exist" from "this
    id is not on the page I happened to fetch".

    404 for an id that names nothing, and not the uniform 404 the analyses routes give an
    unreachable record. There is nothing to conceal from this caller: they are entitled to list
    every account in the system and can see for themselves which ids exist.
    """
    user = session.execute(
        select(User).where(User.id == user_id)
    ).scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="user not found"
        )

    return visible_user(user)


@router.patch(
    "/users/{user_id}",
    response_model=AdminUser,
    # A state change, so it carries the same origin check every other dashboard mutation does.
    # The session cookie is `SameSite=Lax` and would not be attached to a cross-site request
    # in the first place; this is the independent check behind that, made by the server that
    # owns the data. It matters more here than on a submission: the thing a forged request
    # would be driving is the granting of administrative access.
    dependencies=[Depends(require_same_origin)],
)
def update_user(
    user_id: uuid.UUID,
    change: UserChange,
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminUser:
    """Change an account's address, its role, its activation, or any of them — subject to two
    safety rules.

    **An administrator may not change their own role or activation.** Not because it would be
    unsafe in itself, but because it is the one change nobody can undo for them: an
    administrator who demotes themselves by a mis-click is no longer able to reach this route
    to put it back, and an operator who deactivates themselves has locked the door from the
    inside with the key still in it. The remedy in both cases is another administrator or a
    hand-written UPDATE in psql, and a control that can put a deployment in that position is
    a control that should refuse. The check is by id, because that is the account the session
    resolved to — comparing emails would compare two normalizations and one of them could be
    stale.

    **At least one active administrator must remain.** A system with no administrator has no
    way back either: nothing in the application can promote an account, so recovery is again
    psql. This is enforced here as a real check and not as an assumption, even though the rule
    above already makes it unreachable through this route today — the caller is an active
    administrator who cannot be the target, so there is always at least one left standing. The
    invariant is not a consequence of the self-modification rule and must not be left resting
    on it: the day somebody relaxes that rule, or adds a bulk route, or writes an account
    deletion, this is the check that is already in place rather than the one that had to be
    remembered. `tests/test_admin_users.py` asserts it at the level where it is reachable and
    says so.

    The target row is read `FOR UPDATE`. Two administrators editing the same account at the
    same moment would otherwise both read the old row and the second write would silently
    undo the first; the lock makes the second wait and read what the first actually committed.
    It does not serialize two administrators editing *different* accounts, which is the
    theoretical hole in the invariant above — and it stays theoretical only because the self-
    modification rule keeps the count from ever being able to reach zero. A change to that
    rule is a change that has to revisit this paragraph.

    404 for an id that names nothing. Not the uniform 404 the analyses routes give an
    unreachable record, and for a reason: there is nothing to conceal from this caller, who is
    entitled to list every account in the system and can see for themselves which ids exist.

    **A change that sticks is recorded; nothing else is.** Since R8-T5 a successful mutation
    also writes one `AdminAuditEvent`, added to this same session so that the account and the
    record of who changed it commit together. Three cases fall out of where that `session.add`
    sits, and all three are deliberate:

    A request that would change nothing returns before any of it — no UPDATE, no event. A
    request refused by either rule above raises before it, so the event is never added; and
    even if a later rule were added below the `add`, the raise would roll the session back and
    take the event with it. And a request that succeeds writes exactly one event, never two,
    because the fields it touched are one row's worth of change however many of them moved.

    The audit row is not a second answer to "what does this account look like now" — that is
    the account itself. It is the answer to "who made it look like this, and when", which is
    the one question the `users` table cannot answer at all.

    **The address is a third field this may change, added in R8-T8, and neither safety rule
    bends for it.** The self-modification check is on the target's id and runs before the body
    is looked at, so an administrator still cannot reach their own row through this route by any
    field — including this one. The minimum-administrator check is about `role` and `is_active`
    and an address has no bearing on it, so it is left exactly where it was rather than widened
    to a condition that would have to exclude the new field anyway.

    Changing an address is not a way to take over another account. It renames one account; the
    account it is being renamed to the address *of* is refused, by the uniqueness check below
    and by the unique index behind it. And the old address stops working immediately, because
    `authenticate` looks an account up by the stored form and there is only ever one.

    **A change of address does not end the account's sessions, and that is deliberate.** A
    session authenticates an account by id; the address is a label on it, and revoking somebody's
    session because an operator corrected a typo in their email would be a sign-out with no
    security question behind it. The operation that does end sessions is the one that replaces
    the credential they were opened against — `reset_password` below.
    """
    if user_id == administrator.id:
        logger.info("Administrator %s was refused a change to their own account.", administrator.id)
        raise refused("An administrator cannot change their own role or activation.")

    target = session.execute(
        select(User).where(User.id == user_id).with_for_update()
    ).scalar_one_or_none()

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="user not found"
        )

    # What the account would be afterwards. Resolved before anything is written, so the
    # invariant below is checked against the outcome rather than against a row that has
    # already been half-changed and would have to be put back.
    email = change.email if change.email is not None else target.email
    role = change.role if change.role is not None else target.role
    is_active = change.is_active if change.is_active is not None else target.is_active

    # What would actually move. Computed here, against the row as it still stands, because
    # after the assignment below there is nothing left to compare against.
    changes = account_changes(target, email, role, is_active)

    # A request that changes nothing writes nothing — no UPDATE and, above all, no audit row.
    # An audit log that records the times somebody opened the form and pressed save without
    # touching a control is a log whose real entries are buried in noise, and the dashboard
    # submits both controls together, so this is the common case rather than a strange one.
    #
    # It returns 200 with the account as it stands. The caller asked for a state and that is
    # the state; nothing about "you asked for what was already true" is an error.
    if not changes:
        return visible_user(target)

    # Only a change that stops this account being an active administrator can reduce the
    # count, so only that case is asked about. Promoting somebody, reactivating somebody, or
    # editing an ordinary account cannot break the invariant and does not pay for a query.
    leaves_admin_role = target.role == USER_ROLE_ADMIN and target.is_active
    stays_active_admin = role == USER_ROLE_ADMIN and is_active

    if leaves_admin_role and not stays_active_admin and other_active_admins(session, target) == 0:
        logger.info("A change to user %s was refused: it would leave no active administrator.", target.id)
        raise refused("At least one active administrator must remain.")

    # Asked only when the address actually moves, which `changes` already establishes — a PATCH
    # that resubmits the account's own current address is a no-op that returned above, and one
    # that changes only the role should not pay for a query about an address nobody touched.
    #
    # `excluding` is the target, so an account is never told its own address is taken. The real
    # enforcement is the unique index; this is what makes the refusal a 409 that names the
    # problem rather than an `IntegrityError` on commit.
    if "email" in changes and email_taken(session, email, excluding=target.id):
        logger.info("A change to user %s was refused: the address is already in use.", target.id)
        raise conflict("An account with that email already exists.")

    # The event is built *before* the assignment and added after it, which matters for exactly
    # one field. `target_email_snapshot` is the address the account had at the moment of the
    # event, and on a change of address that is the old one — the new one is already in
    # `changes`, and a row that recorded only the new address would leave the log unable to say
    # what the account used to be called. Built after the assignment it would snapshot the new
    # address and lose the pairing.
    event = audit_event(administrator, target, AUDIT_ACTION_USER_UPDATED, changes)

    target.email = email
    target.role = role
    target.is_active = is_active

    # The record of the change, added to the session that holds the change itself so that one
    # `commit()` writes both or neither. That is the whole reason this is here rather than in
    # a listener or a line after the commit: those write the audit row in a second transaction,
    # which is exactly the arrangement that produces a log entry for a change that got rolled
    # back, or loses the entry for one that stuck.
    #
    # It is the `add` that has to be here and not the construction: an object that is never
    # added is never written, so building it a few lines earlier changes nothing about which
    # transaction it lands in. See `audit_event` for whose id ends up on it.
    session.add(event)

    try:
        session.commit()
    except IntegrityError:
        # The race the check above cannot close: another request took this address between that
        # query and this commit. The session is poisoned once a constraint has fired, so it is
        # rolled back — which also takes the audit event with it, correctly, because the change
        # it describes did not happen.
        session.rollback()
        logger.info("A change to user %s lost a race for the address.", target.id)

        raise conflict("An account with that email already exists.") from None

    session.refresh(target)

    # Both accounts, because an administrative action is only useful in a log if it says who
    # did it as well as what was done. No email and no hash: the ids are what the database
    # can be queried by, and the addresses are personal data that does not need to be in a log
    # line to make it actionable.
    logger.info(
        "Administrator %s set user %s to role=%s active=%s.",
        administrator.id,
        target.id,
        target.role,
        target.is_active,
    )

    return visible_user(target)


@router.post(
    "/users/{user_id}/password",
    status_code=status.HTTP_204_NO_CONTENT,
    # A state change, and the same origin check as the two above. This one replaces a
    # credential, which makes it the request in this module a forged form would most want to
    # drive: a password an attacker chose is an account an attacker can sign into.
    dependencies=[Depends(require_same_origin)],
)
def reset_password(
    user_id: uuid.UUID,
    payload: PasswordReset,
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> Response:
    """Replace an account's password, end every session it still holds, and record that.

    **The three writes are one transaction, and the ordering of this function is the whole of
    why.** The new hash, the revocation of the account's open sessions and the audit event are
    all added to the same session and committed once. Split across two transactions, each pair
    has a failure that is worse than the operation not happening at all:

    A hash committed before the revocation leaves a window — and after a crash, a permanent
    state — in which the password has been changed and the sessions opened with the old one are
    still live. That is precisely the situation a reset exists to end, so a reset that can leave
    it is not a reset. A revocation committed before the hash signs everybody out and leaves the
    old password working, which is a lockout that fixes nothing. And an audit row written
    separately either records a reset that rolled back or loses one that stuck — the thing the
    log is read for is which of those happened.

    **`revoke_user_sessions` does not commit, and must not start.** It is shared with
    `start_session`, which has the same requirement for the same reason; see it for why the
    revocation is one `UPDATE ... WHERE revoked_at IS NULL` rather than a read followed by a
    loop.

    The row is read `FOR UPDATE`, as `update_user` reads it. Two administrators resetting the
    same account at once would otherwise both write a hash and the loser's password would be the
    one that silently did not take — and the operator would have told somebody a password that
    does not work.

    **An administrator may reset their own password, and it signs them out.** The self-
    modification rule above is not applied here, because it exists for changes nobody can undo
    for them: demoting themselves removes their access to the route that would put it back.
    Setting their own password does not — they chose the new one and can sign in with it
    immediately. The sessions revoked include their own, which is correct rather than a side
    effect: the point of the revocation is that no session survives a credential it was not
    opened against, and an exception for the caller would be an exception for exactly the
    session most likely to be the compromised one.

    **Nothing about the password reaches the response, the log or the event.** The response is
    204 with no body at all, which is not laziness about a return value — there is no account
    state to report that a subsequent GET does not already give, and a body here would be a
    place for a credential to end up. The log line names the two accounts and the number of
    sessions closed. The audit payload says that the password moved and how many sessions that
    ended, and neither the plaintext nor the Argon2id digest is in either.

    The event's `changes` is `{"password_reset": {"old": false, "new": true}}` and not the flat
    `{"password_reset": true}` the plan for this task named. The shape is not a preference: the
    audit screen's `parseChanges` rejects any entry that is not an `{old, new}` pair, and it
    rejects the entire listing when one row fails — a flat payload here would blank the audit
    log for every event in it. The boolean pair says the same thing in the shape every other
    row is in.

    404 for an id that names nothing, for the reason the other routes give it: there is nothing
    to conceal from a caller entitled to list every account in the system.
    """
    target = session.execute(
        select(User).where(User.id == user_id).with_for_update()
    ).scalar_one_or_none()

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="user not found"
        )

    # The hash first, the revocation second, the event third, one commit for all three. The
    # order within the transaction does not matter to the database — nothing here reads what the
    # previous line wrote — but it is written in the order the docstring argues about so that a
    # reader checking the claim can check it against the code directly above.
    target.password_hash = hash_password(payload.password)

    revoked = revoke_user_sessions(session, target.id)

    session.add(
        audit_event(
            administrator,
            target,
            AUDIT_ACTION_USER_PASSWORD_RESET,
            {
                # That it happened. There is no old value to record and no new one that could be
                # written down — the plaintext is gone at the end of this request and the digest
                # is a credential's stored form, which a log that reprints it has widened access
                # to. A boolean is the whole of what can honestly be said.
                "password_reset": {"old": False, "new": True},
                # And what it cost the account. "This ended three sessions" is the operationally
                # interesting half of a reset and cannot be recovered afterwards: once the
                # `UPDATE` has run, the rows it touched are indistinguishable from sessions that
                # were revoked at any other time.
                "sessions_revoked": {"old": revoked, "new": 0},
            },
        )
    )

    session.commit()

    # The two accounts by id and the number of sessions closed. Never `payload`, which carries
    # the plaintext, and never `target.password_hash`, which is the credential's stored form —
    # this is the line in this module where either mistake would be easiest to make.
    logger.info(
        "Administrator %s reset the password for user %s, ending %d session(s).",
        administrator.id,
        target.id,
        revoked,
    )

    # Built explicitly rather than returned as `None`. The route is declared 204 and FastAPI
    # would serialize a `None` return into a four-byte `null` body, which is a body on a status
    # that is defined as having none.
    return Response(status_code=status.HTTP_204_NO_CONTENT)
