"""The routes an administrator records Ground Truth through (R12-T2).

Mounted under `/api/v1/admin`, beside the review routes and deliberately not beside them in
one file. A review is an opinion about a report; Ground Truth is what the media actually is.
`app/ground_truth.py` states why the two — and the automated verdict, and provenance — must
never stand in for each other, and this module keeps that true at the storage boundary.

**Keyed by the bytes.** The path names a SHA-256, not an analysis and not a media row. The
same file can be uploaded under any number of analyses, and the truth about it is the same for
all of them, so it is recorded once against `MediaFile.original_sha256` and read back by that
hash wherever the bytes appear.

**The only table this module writes is `ground_truth`, plus the audit row for it.** No
statement here names `analyses`, `analysis_signals`, `analysis_reviews` or a provenance value.
`media_files` is read, and locked, to establish that the bytes exist; it is never written.

**`manipulation_family` is never accepted and never stored.** The body is
`GroundTruthContract`, which forbids undeclared fields, so a caller who sends a family gets a
422 rather than a silently dropped field. The response carries the family derived from the
stored label on the way out.

**Every change is audited with every field.** One `AdminAuditEvent` per mutation, in the same
transaction, carrying `old` and `new` for `source_class`, `label` and `notes` — all three,
whether they moved or not, and `old: null` for each on creation. That is a deliberate departure
from the review events, which withhold the note's text: Ground Truth is what evaluations are
scored against, and "what did it say when this evaluation ran" has to be answerable from the
log alone. A request that changes nothing writes nothing, as everywhere else on this prefix.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.admin_analyses import FORBIDDEN_NOTE_CHARACTERS
from app.db.models import (
    AUDIT_ACTION_GROUND_TRUTH_CREATED,
    AUDIT_ACTION_GROUND_TRUTH_UPDATED,
    AUDIT_TARGET_GROUND_TRUTH,
    AdminAuditEvent,
    GroundTruth,
    MediaFile,
    User,
)
from app.db.session import get_session
from app.ground_truth import (
    GroundTruthContract,
    GroundTruthLabel,
    ManipulationFamily,
    SourceClass,
    derive_manipulation_family,
)
from app.observability import current_request_id
from app.web_auth import require_admin, require_same_origin

logger = logging.getLogger(__name__)

# Exactly the spelling `MediaFile.original_sha256` is written in: 64 lowercase hex characters.
# Uppercase is refused rather than folded, so there is one spelling of a hash in the URL, in
# the table and in the audit log, and a caller who sent another is told rather than rewritten.
SHA256_PATTERN = r"^[0-9a-f]{64}$"

# Behind `require_admin` at the router, for the reason the sibling modules give: a route added
# here later cannot be one somebody forgot to guard. 401 for an anonymous caller, 403 for an
# authenticated non-administrator, on the GET as much as the PUT.
router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class GroundTruthState(BaseModel):
    """The Ground Truth recorded for one set of bytes.

    Built field by field in `visible_ground_truth`, never by `from_attributes`, so a column
    added to `GroundTruth` later reaches this payload only when somebody decides it should.
    """

    model_config = ConfigDict(from_attributes=False)

    media_sha256: str
    source_class: SourceClass
    label: GroundTruthLabel

    # Derived from `label` on every read and never stored — see `GroundTruth`.
    manipulation_family: ManipulationFamily

    notes: str | None

    # Who last recorded it, and the address they had at the time.
    actor_id: uuid.UUID
    actor_email_snapshot: str | None

    created_at: datetime
    updated_at: datetime


def visible_ground_truth(record: GroundTruth) -> GroundTruthState:
    """The stored record as a response — the one place a `GroundTruth` becomes a payload."""
    return GroundTruthState(
        media_sha256=record.media_sha256,
        source_class=record.source_class,
        label=record.label,
        manipulation_family=derive_manipulation_family(record.label),
        notes=record.notes,
        actor_id=record.actor_id,
        actor_email_snapshot=record.actor_email_snapshot,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def media_or_404(session: Session, sha256: str, *, lock: bool) -> None:
    """Establish that some analysed media carries these bytes, or a 404.

    Ground Truth for a hash no upload ever produced is a label for nothing this system has
    seen, so it is refused rather than stored in wait for a file that may never arrive.

    With `lock`, the first matching media row is taken `FOR UPDATE`, ordered so two requests
    lock the same row. That serializes two administrators writing the same hash at the same
    moment: without it both would read "no record", both would insert, and the second would be
    a duplicate-key 500. The lock writes nothing to `media_files`.
    """
    query = (
        select(MediaFile.id)
        .where(MediaFile.original_sha256 == sha256)
        .order_by(MediaFile.id)
        .limit(1)
    )

    if lock:
        query = query.with_for_update()

    if session.execute(query).scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="media not found"
        )


def stored_ground_truth(session: Session, sha256: str) -> GroundTruth | None:
    """The record for these bytes, or None where nobody has recorded one."""
    return session.execute(
        select(GroundTruth).where(GroundTruth.media_sha256 == sha256)
    ).scalar_one_or_none()


def ground_truth_changes(
    previous: GroundTruth | None, statement: GroundTruthContract
) -> dict:
    """The full before/after of all three recorded fields.

    Every field appears, moved or not, so each event is a complete statement of what the
    record said before and after it. `old` is null for all three on creation: before the first
    record there was no statement, which is not the same as a statement of `UNKNOWN`.
    """
    return {
        "source_class": {
            "old": None if previous is None else previous.source_class,
            "new": statement.source_class,
        },
        "label": {
            "old": None if previous is None else previous.label,
            "new": statement.label,
        },
        "notes": {
            "old": None if previous is None else previous.notes,
            "new": statement.notes,
        },
    }


@router.get("/ground-truth/{sha256}", response_model=GroundTruthState)
def get_ground_truth(
    sha256: str = Path(pattern=SHA256_PATTERN),
    session: Session = Depends(get_session),
) -> GroundTruthState:
    """The Ground Truth recorded for these bytes.

    Two different 404s, told apart by their detail: `media not found` for a hash no upload
    carries, and `ground truth not recorded` for known bytes nobody has labelled. The second is
    deliberately not a 200 reporting `UNKNOWN` — `UNKNOWN` is a statement somebody makes, and
    answering it for a file nobody has spoken about would put a statement in their mouth.
    """
    media_or_404(session, sha256, lock=False)

    record = stored_ground_truth(session, sha256)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="ground truth not recorded"
        )

    return visible_ground_truth(record)


@router.put(
    "/ground-truth/{sha256}",
    response_model=GroundTruthState,
    dependencies=[Depends(require_same_origin)],
)
def set_ground_truth(
    statement: GroundTruthContract,
    sha256: str = Path(pattern=SHA256_PATTERN),
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> GroundTruthState:
    """Record or revise the Ground Truth for these bytes.

    A PUT because the body is the whole statement: an absent `notes` means "no notes", not
    "leave the notes as they are".

    **The actor is the session, never the body.** `GroundTruthContract` has no field an
    identity could be put in; the record and the audit event both take `administrator`.

    **A request that changes nothing writes nothing** — neither the record nor an audit row —
    and returns the record as it stands.
    """
    # PostgreSQL cannot store a NUL in a text column, and the rest of the control characters
    # are not something anybody typed on purpose; refused here, as the review note refuses
    # them, rather than surfacing as a 500 or being silently rewritten.
    if statement.notes is not None and any(
        character in FORBIDDEN_NOTE_CHARACTERS for character in statement.notes
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="notes must be plain text",
        )

    media_or_404(session, sha256, lock=True)

    record = stored_ground_truth(session, sha256)

    if (
        record is not None
        and record.source_class == statement.source_class
        and record.label == statement.label
        and record.notes == statement.notes
    ):
        return visible_ground_truth(record)

    # Captured before anything is assigned: after the mutation there is nothing to compare.
    changes = ground_truth_changes(record, statement)
    action = (
        AUDIT_ACTION_GROUND_TRUTH_UPDATED
        if record is not None
        else AUDIT_ACTION_GROUND_TRUTH_CREATED
    )

    if record is None:
        record = GroundTruth(media_sha256=sha256)
        session.add(record)

    record.source_class = statement.source_class
    record.label = statement.label
    record.notes = statement.notes
    record.actor_id = administrator.id
    record.actor_email_snapshot = administrator.email

    # Added to the session holding the record, so one `commit()` writes both or neither.
    # `target_email_snapshot` is null: the target is a set of bytes, not a person.
    session.add(
        AdminAuditEvent(
            actor_id=administrator.id,
            actor_email_snapshot=administrator.email,
            action=action,
            target_type=AUDIT_TARGET_GROUND_TRUTH,
            target_id=sha256,
            target_email_snapshot=None,
            changes=changes,
            request_id=current_request_id(),
        )
    )

    session.commit()
    session.refresh(record)

    # The label and source, never the notes: free prose stays out of the application log.
    logger.info(
        "Administrator %s recorded ground truth for %s as %s (source: %s).",
        administrator.id,
        sha256,
        record.label,
        record.source_class,
    )

    return visible_ground_truth(record)
