"""The two routes an administrator manages other people's accounts through.

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
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import USER_ROLE_ADMIN, USER_ROLE_USER, User
from app.db.session import get_session
from app.web_auth import require_admin, require_same_origin

logger = logging.getLogger(__name__)

# The roles an account may be moved between, as the model spells them. A frozen set rather
# than a tuple so membership is the O(1) test it reads as, and built from the constants so
# that a third role added to `models.py` is a one-line change there rather than a string
# somebody has to remember is also written down over here.
ASSIGNABLE_ROLES = frozenset({USER_ROLE_USER, USER_ROLE_ADMIN})

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


class UserChange(BaseModel):
    """What a PATCH may ask to change about an account.

    Both fields are optional and both default to "leave it alone", which is what makes this a
    PATCH rather than a PUT: the dashboard's role control and its activation control are two
    separate submissions, and each has to be able to travel without restating the other's
    current value. Sending neither is a no-op that returns the account unchanged.

    `extra="forbid"` so a body naming a field this model does not have is refused rather than
    silently ignored. A caller who thinks they are deactivating somebody by sending
    `{"active": false}` should be told the word is wrong, not handed back a 200 and an
    account that still works.
    """

    model_config = ConfigDict(extra="forbid")

    role: str | None = None
    is_active: bool | None = None

    @field_validator("role")
    @classmethod
    def assignable_role(cls, value: str | None) -> str | None:
        """Refuse any role this system does not have.

        Checked against `ASSIGNABLE_ROLES`, which is built from the model's own constants, so
        the set of things this endpoint can write is the set of things the column means.
        Without it the route would happily store `"Admin"`, `"superuser"` or the empty string
        — none of which grants anything, all of which would leave an account whose role reads
        like something and matches nothing, and one of which (`"Admin"`, the spelling a human
        will type) would silently strip an administrator of their access while looking like it
        had granted it.

        A validator rather than a `Literal` annotation because the allowed values are the
        constants, not two strings retyped here; the cost is that the refusal is a 422 from
        pydantic rather than a hand-written 400, which is the right status for a body whose
        field is not a value the field can take.
        """
        if value is not None and value not in ASSIGNABLE_ROLES:
            raise ValueError("role must be one of: " + ", ".join(sorted(ASSIGNABLE_ROLES)))

        return value


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
    """Change an account's role, its activation, or both — subject to two safety rules.

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
    role = change.role if change.role is not None else target.role
    is_active = change.is_active if change.is_active is not None else target.is_active

    # Only a change that stops this account being an active administrator can reduce the
    # count, so only that case is asked about. Promoting somebody, reactivating somebody, or
    # editing an ordinary account cannot break the invariant and does not pay for a query.
    leaves_admin_role = target.role == USER_ROLE_ADMIN and target.is_active
    stays_active_admin = role == USER_ROLE_ADMIN and is_active

    if leaves_admin_role and not stays_active_admin and other_active_admins(session, target) == 0:
        logger.info("A change to user %s was refused: it would leave no active administrator.", target.id)
        raise refused("At least one active administrator must remain.")

    target.role = role
    target.is_active = is_active
    session.commit()
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
