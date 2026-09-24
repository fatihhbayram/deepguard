"""The routes an administrator records dataset governance through (R12-T5).

Mounted under `/api/v1/admin`, beside the Ground Truth routes and in a file of its own: what a
file is (Ground Truth) and where it came from and may be used (governance) are two statements,
revised independently. The rules — one split per lineage, derived files in their parent's
lineage, one lineage per recording, no cycles, no split moves — are in
`app/dataset_governance.py`; this module owns the transaction around them.

**Keyed by the bytes**, as Ground Truth is: the path names `MediaFile.original_sha256`, in its
one canonical spelling.

**The tables this module writes are `lineage_splits`, `media_governance`, and the audit row.**
No statement here names a verdict, a signal, a review, provenance or Ground Truth.

**One writer at a time.** Every PUT takes one transaction-scoped advisory lock before it reads
anything. The guards read across records — a parent's chain, every record of a recording, a
lineage another request may be creating — so a per-row lock would let two requests each pass
their checks against a state the other is about to change (two halves of an A → B → A loop,
or one new lineage created with two splits). Governance writes are rare administrative
actions; serializing all of them costs nothing that matters.

**Every change is audited with every field.** One `AdminAuditEvent` per mutation, carrying
`old` and `new` for all six semantic fields — `dataset_split` included, though it is stored on
the lineage — moved or not, with `old: null` for each on creation. A request that changes
nothing writes nothing.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dataset_governance import (
    SHA256_PATTERN,
    DatasetSplit,
    GovernanceRejected,
    MediaGovernanceContract,
    check_statement,
    media_exists,
    stored_governance,
    stored_lineage,
)
from app.db.models import (
    AUDIT_ACTION_DATASET_GOVERNANCE_CREATED,
    AUDIT_ACTION_DATASET_GOVERNANCE_UPDATED,
    AUDIT_TARGET_DATASET_GOVERNANCE,
    AdminAuditEvent,
    LineageSplit,
    MediaGovernance,
    User,
)
from app.db.session import get_session
from app.observability import current_request_id
from app.web_auth import require_admin, require_same_origin

logger = logging.getLogger(__name__)

# An arbitrary constant naming "dataset governance writes" to `pg_advisory_xact_lock`. Released
# by the commit or rollback that ends the transaction.
GOVERNANCE_WRITE_LOCK = 0x52313254_35  # "R12T5"

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class DatasetGovernanceState(BaseModel):
    """The governance recorded for one set of bytes, with the split of its lineage.

    Built field by field in `visible_governance`, never by `from_attributes`.
    """

    model_config = ConfigDict(from_attributes=False)

    media_sha256: str
    source_lineage_id: str
    dataset_split: DatasetSplit
    recording_identity: str | None
    transformations: list[str] | None
    derived_from_sha256: str | None
    generation_pipeline: str | None

    actor_id: uuid.UUID
    actor_email_snapshot: str | None

    created_at: datetime
    updated_at: datetime


def visible_governance(
    record: MediaGovernance, lineage: LineageSplit
) -> DatasetGovernanceState:
    return DatasetGovernanceState(
        media_sha256=record.media_sha256,
        source_lineage_id=record.source_lineage_id,
        dataset_split=lineage.dataset_split,
        recording_identity=record.recording_identity,
        transformations=record.transformations,
        derived_from_sha256=record.derived_from_sha256,
        generation_pipeline=record.generation_pipeline,
        actor_id=record.actor_id,
        actor_email_snapshot=record.actor_email_snapshot,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def governance_snapshot(
    record: MediaGovernance | None, lineage: LineageSplit | None
) -> dict | None:
    """The six semantic fields as stored, or None before the first record."""
    if record is None:
        return None

    return {
        "source_lineage_id": record.source_lineage_id,
        "dataset_split": None if lineage is None else lineage.dataset_split,
        "recording_identity": record.recording_identity,
        "transformations": record.transformations,
        "derived_from_sha256": record.derived_from_sha256,
        "generation_pipeline": record.generation_pipeline,
    }


def governance_changes(previous: dict | None, statement: MediaGovernanceContract) -> dict:
    """Old and new for every field, moved or not; `old` is null for each on creation."""
    new = statement.model_dump()

    return {
        name: {"old": None if previous is None else previous[name], "new": new[name]}
        for name in (
            "source_lineage_id",
            "dataset_split",
            "recording_identity",
            "transformations",
            "derived_from_sha256",
            "generation_pipeline",
        )
    }


@router.get("/dataset-governance/{sha256}", response_model=DatasetGovernanceState)
def get_dataset_governance(
    sha256: str = Path(pattern=SHA256_PATTERN),
    session: Session = Depends(get_session),
) -> DatasetGovernanceState:
    """The governance recorded for these bytes.

    `media not found` for a hash no upload carries; `dataset governance not recorded` for known
    bytes nobody has governed — never a 200 inventing a lineage or a split.
    """
    if not media_exists(session, sha256):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="media not found")

    record = stored_governance(session, sha256)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="dataset governance not recorded"
        )

    return visible_governance(record, stored_lineage(session, record.source_lineage_id))


@router.put(
    "/dataset-governance/{sha256}",
    response_model=DatasetGovernanceState,
    dependencies=[Depends(require_same_origin)],
)
def set_dataset_governance(
    statement: MediaGovernanceContract,
    sha256: str = Path(pattern=SHA256_PATTERN),
    administrator: User = Depends(require_admin),
    session: Session = Depends(get_session),
) -> DatasetGovernanceState:
    """Record or revise the governance for these bytes.

    A PUT because the body is the whole statement: an absent optional field means "none", not
    "leave it as it is". A new lineage is created with the stated split; an existing lineage
    must be stated with the split it already has, or the request is a 409.
    """
    session.execute(select(func.pg_advisory_xact_lock(GOVERNANCE_WRITE_LOCK)))

    record = stored_governance(session, sha256)

    try:
        lineage = check_statement(session, sha256, statement, record)
    except GovernanceRejected as rejection:
        session.rollback()
        raise HTTPException(
            status_code=rejection.status_code, detail=rejection.detail
        ) from None

    previous = governance_snapshot(record, lineage if record is not None else None)

    if previous is not None and previous == statement.model_dump():
        session.rollback()
        return visible_governance(record, lineage)

    # Captured before anything is assigned: after the mutation there is nothing to compare.
    changes = governance_changes(previous, statement)
    action = (
        AUDIT_ACTION_DATASET_GOVERNANCE_UPDATED
        if record is not None
        else AUDIT_ACTION_DATASET_GOVERNANCE_CREATED
    )

    if lineage is None:
        lineage = LineageSplit(
            source_lineage_id=statement.source_lineage_id,
            dataset_split=statement.dataset_split,
        )
        session.add(lineage)
        # The governance row references it; the lineage has to exist first.
        session.flush()

    if record is None:
        record = MediaGovernance(media_sha256=sha256)
        session.add(record)

    record.source_lineage_id = statement.source_lineage_id
    record.recording_identity = statement.recording_identity
    record.transformations = statement.transformations
    record.derived_from_sha256 = statement.derived_from_sha256
    record.generation_pipeline = statement.generation_pipeline
    record.actor_id = administrator.id
    record.actor_email_snapshot = administrator.email

    session.add(
        AdminAuditEvent(
            actor_id=administrator.id,
            actor_email_snapshot=administrator.email,
            action=action,
            target_type=AUDIT_TARGET_DATASET_GOVERNANCE,
            target_id=sha256,
            target_email_snapshot=None,
            changes=changes,
            request_id=current_request_id(),
        )
    )

    session.commit()
    session.refresh(record)
    session.refresh(lineage)

    logger.info(
        "Administrator %s recorded dataset governance for %s (lineage %s, split %s).",
        administrator.id,
        sha256,
        record.source_lineage_id,
        lineage.dataset_split,
    )

    return visible_governance(record, lineage)
