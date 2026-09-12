"""The record of what administrators have done to other people's accounts (R8-T5).

Mounted under `/api/v1/admin` beside the account, job and analytics routes, behind the same
`require_admin`. It is the fourth module on that prefix and the narrowest: one GET, no
parameters, no filters.

**There is no write route here, and there is no delete route, and that is what makes this an
audit log.** The rows are written by `update_user` in `admin_users.py`, inside the transaction
that makes the change they describe — that is the only code in this application that inserts
one. Nothing anywhere issues an UPDATE or a DELETE against the table. PostgreSQL would accept
either; what guarantees append-only here is that no endpoint exists to ask for it, which is a
guarantee somebody has to deliberately remove rather than one they can forget to apply.

A "clear the log" control is the obvious next thing to want on a screen like this and it is
exactly the control an audit log must not have. A record that the person being recorded can
erase is not evidence of anything. If these rows ever need to stop growing, that is a
retention policy — a decision about how long history is kept, applied by something outside the
request path — and not a button.

**Every name on this payload is a snapshot, and none of it is joined.** `actor_email_snapshot`
and `target_email_snapshot` are read straight off the event row, where they were frozen at the
moment of the change. This route never touches the `users` table. Joining would be easy and
would look more normalized, and it would mean that renaming an account rewrote the history of
what that account did, and that deleting one blanked it — the event would quietly stop saying
what happened. `AdminAuditEvent` carries the full reasoning.

So an address here answers "who was this, then", not "who is this now". The two diverge the
day somebody changes their email, and when they do this screen is supposed to keep showing the
old one.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AdminAuditEvent
from app.db.session import get_session
from app.web_auth import require_admin

logger = logging.getLogger(__name__)

# How many events one read returns. A ceiling rather than a page size, the same shape and the
# same reasoning as `RECENT_JOB_LIMIT` in `admin_jobs.py`: there is no cursor here and no
# `?limit=`, because the question this screen answers is "what has been done to accounts
# lately" and that is the top of the table.
#
# This table only grows — nothing prunes it, by design — so an unbounded `SELECT` here is a
# query that works for a year and then returns everything. Paging back through the history is
# a real requirement the day somebody has it, against a real number and with a real cursor.
RECENT_AUDIT_LIMIT = 100

# Every route below is behind `require_admin`, declared once on the router rather than per
# route, so a route added to this file later cannot be an administrative route somebody forgot
# to guard. Same shape as the three admin modules beside it, and the same dependency — not a
# second role check written here, which would be a second answer to the only question that
# matters.
#
# `require_admin` is layered on `require_user`, so an anonymous caller is refused with 401 and
# an authenticated non-administrator with 403.
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class AdminAuditEntry(BaseModel):
    """One recorded change, as an administrator is shown it.

    The whole event row. That is unusual in this codebase — `AdminUser` and `AdminJob` both
    narrow what they are built from — and it is right here for a reason the others do not
    have: this table was designed for exactly one reader, holds nothing but what that reader
    is meant to see, and has no credential, hash or token on it to withhold. There is no
    narrowing to do.

    Built field by field rather than by `from_attributes` anyway, keeping the convention its
    siblings set: a column added to `AdminAuditEvent` later reaches this payload when somebody
    decides it should, not by having been added.
    """

    model_config = ConfigDict(from_attributes=False)

    id: uuid.UUID
    actor_id: uuid.UUID
    # As it read when the change was made. Not the account's address now — see the module
    # docstring.
    actor_email_snapshot: str | None
    action: str
    target_type: str
    target_id: str
    target_email_snapshot: str | None
    # `{"field": {"old": ..., "new": ...}}`, holding only the fields that actually moved.
    # Passed through as stored: this route does not summarise the change into a sentence,
    # because a sentence is an interpretation and the page can render the structure.
    changes: dict[str, Any]
    request_id: str | None
    created_at: datetime


def visible_entry(event: AdminAuditEvent) -> AdminAuditEntry:
    """The stored event as a response.

    The one place an `AdminAuditEvent` becomes a payload in this module, so there is one place
    to read to know what leaves it.
    """
    return AdminAuditEntry(
        id=event.id,
        actor_id=event.actor_id,
        actor_email_snapshot=event.actor_email_snapshot,
        action=event.action,
        target_type=event.target_type,
        target_id=event.target_id,
        target_email_snapshot=event.target_email_snapshot,
        changes=event.changes,
        request_id=event.request_id,
        created_at=event.created_at,
    )


@router.get("/audit", response_model=list[AdminAuditEntry])
def list_audit_events(session: Session = Depends(get_session)) -> list[AdminAuditEntry]:
    """The most recent administrative changes, newest first.

    One statement against one table. No join to `users`, deliberately and structurally: the
    names on these rows are the snapshots the event carries, and the module docstring says why
    resolving them live would let a rename edit the past.

    Ordered by `created_at` descending with the id as the tiebreaker, so the listing is stable
    between renders — two events written in the same transaction share a timestamp, and a sort
    with ties in it reshuffles itself on every read. That is not hypothetical here: one request
    writes one event, but a future bulk action would write several at once, and the ordering
    should not start jittering the day it does.
    """
    events = (
        session.execute(
            select(AdminAuditEvent)
            .order_by(AdminAuditEvent.created_at.desc(), AdminAuditEvent.id.desc())
            .limit(RECENT_AUDIT_LIMIT)
        )
        .scalars()
        .all()
    )

    return [visible_entry(event) for event in events]
