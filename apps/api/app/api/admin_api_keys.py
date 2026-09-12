"""The routes an administrator issues and retires public-API credentials through (R8-T6).

Mounted under `/api/v1/admin`, the fifth module on that prefix and the first of them that
writes something other than an account. What it manages is the credential the *other* surface
authenticates with: `app/auth.py` verifies keys on `/api/public/v1`, and nothing here touches
that verification. This is the lifecycle around it — minting a key, listing what has been
minted, and ending one — and the split is deliberate. The code that decides whether a
presented credential is good should not also be the code that issues credentials.

**The plaintext key leaves this application exactly once, in the body of the creation
response.** That is the whole security proposition of this file and it is worth stating as a
list of the places it is not:

- Not in the database. The row stores `key_hash`, the SHA-256 digest, and there is no column
  for the plaintext to go in — `ApiKey` says why that is a digest and not a password hash.
- Not in the audit log. The event written beside the creation carries the key's name and its
  id, and neither the secret nor the digest. An audit screen that reprinted the digest would
  widen what a reader of that screen can see, for nothing.
- Not in a log line. `create_api_key` logs the id and the name, which is what an operator
  needs to correlate; the `GeneratedApiKey` object is never formatted into a message, and the
  one place it is unpacked is the `return` below.
- Not in the listing. `AdminApiKey` has no field for it and `visible_key` could not fill one
  — the value does not exist any more by then, anywhere, which is the point.

So DeepGuard cannot show a customer their key a second time. It can only issue another one
and revoke the first, and the page above this API says so at the moment it matters.

**Revocation is `is_active = False`, never a DELETE.** The column is already what
`require_api_key` filters on, so clearing it ends the key's access on the very next request
with no change to the verification path. Deleting the row would be worse in two ways: the
analyses that key submitted reference it (`analyses.api_key_id` is `ON DELETE RESTRICT`, so
the delete would simply fail for any key that was ever used), and a credential that authorized
real forensic work should stay on file so that work remains attributable after it stops
working.

**Revoking an already-revoked key does nothing at all** — no write, no audit row, and a 200
carrying the key as it stands. `revoke_api_key` says why that is not merely an optimization.

Nothing here edits a key. There is no rename and no reactivation: a reactivation route would
mean a revoked credential could come back, which is exactly the property an operator revoking
one is relying on not existing, and a rename would edit the label an audit row already froze.
A key is created, and it is revoked, and beyond that it is history.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import generate_api_key
from app.db.models import (
    AUDIT_ACTION_API_KEY_CREATED,
    AUDIT_ACTION_API_KEY_REVOKED,
    AUDIT_TARGET_API_KEY,
    AdminAuditEvent,
    ApiKey,
    User,
)
from app.db.session import get_session
from app.observability import current_request_id
from app.web_auth import require_admin, require_same_origin

logger = logging.getLogger(__name__)

# The longest label a key may carry, as the column stores it. Read off `ApiKey.name`'s
# `String(255)` rather than chosen here: a value this file accepted and the database refused
# would surface as a 500 on a request that was merely too long, which is a validation error
# wearing a server fault's clothes.
MAX_KEY_NAME_LENGTH = 255

# Every route below is behind `require_admin`, declared once on the router rather than per
# route, so a route added to this file later cannot be an administrative route somebody forgot
# to guard. The same shape the four admin modules beside it use, and the same dependency.
#
# It matters more here than on any of them. The routes below mint credentials for the external
# surface, so a gap in this guard would not merely expose data — it would let whoever found it
# issue themselves an API key.
#
# `require_admin` is layered on `require_user`, so an anonymous caller is refused with 401 and
# an authenticated non-administrator with 403.
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class AdminApiKey(BaseModel):
    """A key as an administrator is shown it.

    Five fields, and the absent sixth is the entire point: `key_hash` is not here. It is the
    stored form of a credential, and while a digest is not directly usable, publishing it to a
    screen puts it into browser history, screenshots and support tickets for no operational
    benefit — nothing an administrator does with a key needs its digest.

    Built field by field in `visible_key` rather than by `from_attributes`, which is what keeps
    the digest off the wire by construction rather than by everyone remembering to exclude it.
    A column added to `ApiKey` later reaches this payload when somebody decides it should, not
    by having been added.
    """

    model_config = ConfigDict(from_attributes=False)

    id: uuid.UUID
    name: str
    is_active: bool
    created_at: datetime
    # Null until the key has authenticated a request. Nothing writes this column yet — see
    # `ApiKey` — so it reads null on every row today; it is carried because the day it is
    # written, "has this credential ever been used" is the first question an operator
    # deciding whether to revoke one will ask.
    last_used_at: datetime | None


class CreatedApiKey(AdminApiKey):
    """The creation response, and the only object in this application that carries a secret.

    A separate model from `AdminApiKey` rather than an optional `plaintext` field on it. With
    one model the secret would be a field that is usually null, which is a field that a future
    listing route could populate by accident and that every reader of the type has to reason
    about. As a subclass used by exactly one route, the listing physically cannot return a
    plaintext: there is no field on its response model to put one in.

    This object exists for the duration of one response. Nothing persists it and nothing logs
    it; see the module docstring for the full list of places it does not go.
    """

    # The full key, `dg_live_` prefix included, exactly as the caller must present it. Shown
    # once and then gone — DeepGuard retains only the digest and cannot reproduce this value.
    plaintext: str


class ApiKeyCreate(BaseModel):
    """What a creation request may say: a label, and nothing else.

    `extra="forbid"` so a body naming a field this model does not have is refused rather than
    quietly ignored. That is the ordinary reason to forbid extras, and here there is a sharper
    one: it is what makes it impossible for a caller to name the key material. A request that
    tried to supply its own `key_hash`, `plaintext` or `is_active` is a 422, not a request that
    is accepted with the field dropped — the difference between a caller being told the field
    does not exist and a caller believing they set it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str

    @field_validator("name")
    @classmethod
    def usable_name(cls, value: str) -> str:
        """A label somebody can actually identify the key by.

        Stripped, then required to be non-empty and to fit the column. Stripping first is what
        makes `"   "` a refusal rather than a stored key labelled with three spaces — a row an
        operator would see as blank in the table and could not tell apart from a rendering
        fault.

        The stripped value is what gets returned and therefore what gets stored, so the label
        in the database matches the label in the audit event exactly. Two normalizations of the
        same name would be two answers to "which key was this".
        """
        name = value.strip()

        if not name:
            raise ValueError("name must not be blank")

        if len(name) > MAX_KEY_NAME_LENGTH:
            raise ValueError(f"name must be at most {MAX_KEY_NAME_LENGTH} characters")

        return name


def visible_key(key: ApiKey) -> AdminApiKey:
    """The stored key narrowed to what an administrator is told about it.

    The one place an `ApiKey` becomes a response in this module, so there is one place to read
    to know what leaves it — and the field list is written out rather than spread from the ORM
    object, which is what keeps `key_hash` off the wire by construction.
    """
    return AdminApiKey(
        id=key.id,
        name=key.name,
        is_active=key.is_active,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
    )


def audit_event(administrator: User, key: ApiKey, action: str, changes: dict) -> AdminAuditEvent:
    """The record of something done to a key, ready to be added to the caller's session.

    Built here rather than at the two call sites so that the fields that must be the same on
    both — the actor, the target type, the request id, and above all the absence of anything
    secret — are written once.

    The actor is the account `require_admin` resolved the session to, never anything the
    request body could name. `ApiKeyCreate` forbids extra fields, so there is no field a caller
    could set even if this function wanted to read one.

    **`target_email_snapshot` is null, and that is a statement rather than an omission.** The
    column holds the address of the *person* an administrative action was taken against, and a
    key is not a person — it has a name, which is already in `changes` for the creation event,
    and no address at all. Putting the administrator's own address there would make the log
    read as though they had acted on themselves; putting the key's name there would put a label
    in a column every other row means an email by.

    `changes` is `{"field": {"old": ..., "new": ...}}`, the shape `admin_users.py` writes and
    the shape the audit screen renders. The callers below say what they put in it; what neither
    of them puts in it is the plaintext or the digest.
    """
    return AdminAuditEvent(
        actor_id=administrator.id,
        actor_email_snapshot=administrator.email,
        action=action,
        target_type=AUDIT_TARGET_API_KEY,
        target_id=str(key.id),
        target_email_snapshot=None,
        changes=changes,
        request_id=current_request_id(),
    )


@router.get("/api-keys", response_model=list[AdminApiKey])
def list_api_keys(session: Session = Depends(get_session)) -> list[AdminApiKey]:
    """Every key ever issued, newest first, revoked ones included.

    Unfiltered, for the reason the account listing is: a table that hid revoked keys would hide
    exactly the rows somebody came here to check. "Did I already revoke that customer's key"
    is the question this screen exists to answer, and it cannot be answered by a listing that
    shows only the keys that still work. `is_active` carries the distinction instead, which
    lets the page show it rather than the API decide it.

    Newest first, unlike the accounts table, which is oldest first. The orders differ because
    the questions do: an account table is a roster and reads top-down, while the key an
    operator is looking for is almost always the one just issued. The id is the tiebreaker so
    the listing is stable between renders — two keys created in the same second would otherwise
    reshuffle themselves on every read.

    No pagination, and the same reasoning `list_users` gives: this is the credential table of
    an internal deployment, and a cursor scheme for a table of tens of rows is machinery for a
    scale that does not exist. Unlike the audit log next door, this table does not grow on its
    own — a row appears only when an administrator deliberately creates one.
    """
    keys = (
        session.execute(
            select(ApiKey).order_by(ApiKey.created_at.desc(), ApiKey.id.desc())
        )
        .scalars()
        .all()
    )

    return [visible_key(key) for key in keys]


@router.post(
    "/api-keys",
    response_model=CreatedApiKey,
    status_code=status.HTTP_201_CREATED,
    # A state change, so it carries the same origin check every other dashboard mutation does.
    # The session cookie is `SameSite=Lax` and would not be attached to a cross-site request in
    # the first place; this is the independent check behind that, made by the server that owns
    # the data. What a forged request would be driving here is the minting of a credential for
    # the external API, which is the strongest reason in this codebase to have it.
    dependencies=[Depends(require_same_origin)],
)
def create_api_key(
    payload: ApiKeyCreate,
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> CreatedApiKey:
    """Mint a key, store its digest, and return the secret — once.

    The plaintext in this response is the only copy that will ever exist. `generate_api_key`
    returns it alongside the digest, the digest is what goes in the row, and the moment this
    function returns there is nothing anywhere that can reproduce the secret. A caller who
    loses it has to be issued a new key; that is a property of the design and not a gap in it,
    and it is why the screen above this route presents the value the way it does.

    The key is written active. There is no draft state and no enable step: a key that exists
    is a key that works, because an "issued but not yet active" credential would be a second
    state for `require_api_key` to reason about in exchange for nothing an administrator wants.

    **The row and the record of who created it commit together.** The event is added to this
    same session, so one `commit()` writes both or neither — the arrangement `admin_users.py`
    established and for the same reason: an audit row written in a second transaction is a row
    that survives a rollback of the thing it describes, or is lost when that thing sticks.

    `session.flush()` before the event because `target_id` needs the key's id. The default is
    generated in Python rather than by the database, so the id is in fact available earlier;
    flushing anyway keeps the event built from the row as the session actually holds it, which
    is what stays correct if that default ever moves to the server.

    The `changes` payload is `{"name": {"old": null, "new": "..."}}` — a creation, written in
    the same old/new shape a modification uses, with null standing for "there was no key". The
    uniform shape is not cosmetic: the audit screen renders every event through one component
    that reads that structure, and a flat payload here would be an event it could not parse.
    """
    generated = generate_api_key()

    key = ApiKey(name=payload.name, key_hash=generated.key_hash)
    session.add(key)
    session.flush()

    session.add(
        audit_event(
            administrator,
            key,
            AUDIT_ACTION_API_KEY_CREATED,
            # The name, and deliberately nothing else. Not the plaintext, not the digest, and
            # not `is_active` — the latter because an event that records "created inactive:
            # false" is recording the absence of a state this route cannot produce.
            {"name": {"old": None, "new": key.name}},
        )
    )

    session.commit()
    session.refresh(key)

    # The id and the label, which is what an operator correlating this line with the audit row
    # or the key table needs. `generated` is never formatted into a log message — not here and
    # not anywhere — and this is the line that would be the easiest place to get that wrong.
    logger.info(
        "Administrator %s issued API key %s (%s).", administrator.id, key.id, key.name
    )

    return CreatedApiKey(
        **visible_key(key).model_dump(),
        plaintext=generated.plaintext,
    )


@router.post(
    "/api-keys/{key_id}/revoke",
    response_model=AdminApiKey,
    # A state change, and the same origin check as above. This one ends a customer's access
    # rather than granting one, which is no less worth protecting from a forged request.
    dependencies=[Depends(require_same_origin)],
)
def revoke_api_key(
    key_id: uuid.UUID,
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> AdminApiKey:
    """End a key's access, or confirm that it has already ended.

    A POST to a verb rather than a PATCH setting `is_active` to false, because false is the
    only value this route can write. A PATCH would advertise a field with two settings and then
    refuse one of them — there is no reactivation here, deliberately, since a revoked
    credential coming back is exactly what an operator revoking one is relying on being
    impossible.

    The row is read `FOR UPDATE`. Two administrators revoking the same key at the same moment
    would otherwise both read it active and both write the audit row, producing two events for
    one revocation; the lock makes the second wait, and what it then reads is a key that is
    already inactive, which falls into the no-op below.

    **An already-revoked key is not an error, and is not a write.** It returns 200 with the key
    as it stands, and — the part that matters — it adds no audit event. That is not an
    optimization. An audit log in which "this key was revoked" appears three times because an
    operator clicked twice and somebody reloaded is a log that cannot answer *when* the key
    stopped working, which is the only question anybody asks of it. One revocation is one
    event, however many times it is requested.

    It is also why the no-op returns before the mutation rather than assigning `False` over
    `False` and relying on the diff being empty: the emptiness test would work, and it would
    put the guarantee in a comparison instead of in the control flow, one edit away from
    somebody adding a field that always differs.

    404 for an id that names nothing. Not the uniform 404 the public API gives an unreachable
    analysis, and for the same reason `update_user` gives: there is nothing to conceal from
    this caller, who is entitled to list every key in the system and can see which ids exist.
    """
    key = session.execute(
        select(ApiKey).where(ApiKey.id == key_id).with_for_update()
    ).scalar_one_or_none()

    if key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="api key not found"
        )

    if not key.is_active:
        # Already revoked. No UPDATE, no event, and the key as it stands — see the docstring
        # for why this returns before the mutation rather than through it.
        return visible_key(key)

    key.is_active = False

    session.add(
        audit_event(
            administrator,
            key,
            AUDIT_ACTION_API_KEY_REVOKED,
            # The one field that moved. The name is not restated: it did not change, and an
            # event that repeats it would make this row read as a change to the label as well.
            {"is_active": {"old": True, "new": False}},
        )
    )

    session.commit()
    session.refresh(key)

    logger.info(
        "Administrator %s revoked API key %s (%s).", administrator.id, key.id, key.name
    )

    return visible_key(key)
