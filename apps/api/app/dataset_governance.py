"""Where a set of bytes came from, and which evaluation split it belongs to (R12-T5).

Ground Truth (`app/ground_truth.py`) says what a file *is*. Dataset governance says where it
*came from* and what it may be *used for*: the source lineage it descends from, the recording it
was cut from, the transformations that produced it, the file it was derived from, the generation
pipeline where somebody knows it, and the split it is assigned to.

The split exists to keep evaluation honest. A detector calibrated on one clip and then scored
on a re-encode of the same clip is being scored on its own training data, and the number that
comes out is a measurement of memory rather than of detection. So the split is not a property
of a file at all:

- **The split belongs to the lineage.** `dataset_split` lives on `LineageSplit`, keyed by
  `source_lineage_id`, and a governance record carries only the lineage id. Two files of one
  lineage cannot sit in two splits because there is only one row that says what the split is.
- **A derived file inherits its parent's lineage.** A record naming `derived_from_sha256` must
  name the parent's `source_lineage_id` exactly, and the parent must already be governed. A
  re-encode, a crop or a screen capture of a calibration clip is therefore a calibration clip.
- **A recording is in one lineage.** Every record carrying one `recording_identity` must carry
  the same `source_lineage_id`, so two takes of one recording cannot be filed under two
  lineages and reach two splits that way.
- **A split is never moved.** An existing lineage keeps the split it was created with, and an
  existing record keeps its lineage. Correcting a split assignment is a different task with its
  own consequences for every evaluation already run against it; here it is refused.
- **The derivation graph has no cycles.** A file cannot be its own ancestor, which is checked by
  walking the parent chain on every write rather than assumed from the order records arrived in.

Nothing here reads or writes a verdict, a signal, a review, provenance, or Ground Truth.
`manipulation_family` is not accepted: it is derived from the Ground Truth label and nowhere
else, and a second copy of it here could disagree with the first.

This module holds the vocabulary, the request contract and the guards. The guards read the
database through the session they are given; the route (`app/api/admin_dataset_governance.py`)
owns the transaction, the lock, the write and the audit event.
"""

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LineageSplit, MediaFile, MediaGovernance

DatasetSplit = Literal[
    "CALIBRATION",
    "VALIDATION",
    "TEST",
    "HOLDOUT",
]

# Exactly the spelling `MediaFile.original_sha256` is written in, and the one the Ground Truth
# route accepts. Uppercase is refused rather than folded, for the reason given there.
SHA256_PATTERN = r"^[0-9a-f]{64}$"

# Identifiers an administrator names: a lineage and a recording. Deliberately narrow — no
# whitespace, no case folding — so `lineage-1` and `lineage-1 ` cannot become two lineages that
# look like one, which is exactly the leak this module exists to prevent.
IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"

# A transformation step, as recorded: `reencode`, `resize`, `screen_capture`.
TRANSFORMATION_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"

# Free text, but one line of it: printable, no control characters.
GENERATION_PIPELINE_PATTERN = r"^[^\x00-\x1f\x7f]{1,128}$"

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN)]
Transformation = Annotated[str, Field(pattern=TRANSFORMATION_PATTERN)]


class MediaGovernanceContract(BaseModel):
    """A governance statement about one set of bytes: its lineage, split and derivation.

    `extra="forbid"` so a caller naming `manipulation_family`, a verdict, an actor or a label is
    told so with a 422 instead of having the field silently dropped.

    `dataset_split` is part of the statement although it is stored on the lineage: an
    administrator states the split they believe the file is in, and a statement that disagrees
    with the lineage's existing split is refused rather than quietly ignored.
    """

    model_config = ConfigDict(extra="forbid")

    source_lineage_id: Identifier
    dataset_split: DatasetSplit
    recording_identity: Identifier | None = None
    transformations: list[Transformation] | None = None
    derived_from_sha256: Sha256 | None = None
    # Recorded by a person who knows it. Never inferred from a detector, a filename or metadata.
    generation_pipeline: Annotated[
        str, Field(pattern=GENERATION_PIPELINE_PATTERN)
    ] | None = None


class GovernanceRejected(Exception):
    """A statement the stored state does not allow.

    Carries the HTTP status the route answers with, so the rules and their answers are stated
    in one place: 404 for bytes nobody uploaded, 422 for a statement that is wrong on its own
    (a file derived from itself), 409 for one that conflicts with what is already recorded.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


NOT_FOUND = 404
CONFLICT = 409
UNPROCESSABLE = 422


def media_exists(session: Session, sha256: str) -> bool:
    """Whether any upload carries these bytes."""
    return (
        session.execute(
            select(MediaFile.id).where(MediaFile.original_sha256 == sha256).limit(1)
        ).scalar_one_or_none()
        is not None
    )


def stored_governance(session: Session, sha256: str) -> MediaGovernance | None:
    return session.get(MediaGovernance, sha256)


def stored_lineage(session: Session, source_lineage_id: str) -> LineageSplit | None:
    return session.get(LineageSplit, source_lineage_id)


def chain_reaches(
    start: str, target: str, parent_of: Callable[[str], str | None]
) -> bool:
    """Whether walking parents from `start` ever arrives at `target`.

    `start` itself counts, so `chain_reaches(x, x, ...)` is true. The walk is explicit and
    complete — every ancestor is visited — rather than relying on records having been created
    parent-first: an update can re-point an old record at a newer one, and only a walk sees the
    loop that makes.

    A `seen` set stops the walk on a loop that is already stored (which the guards below make
    impossible to write, but which a walk must not spin on forever if one ever appears).
    """
    seen: set[str] = set()
    current: str | None = start

    while current is not None and current not in seen:
        if current == target:
            return True
        seen.add(current)
        current = parent_of(current)

    return False


def check_statement(
    session: Session,
    sha256: str,
    statement: MediaGovernanceContract,
    existing: MediaGovernance | None,
) -> LineageSplit | None:
    """Every guard, in order; raises `GovernanceRejected` on the first that fails.

    Returns the stored `LineageSplit` for the statement's lineage, or None when the lineage is
    new and the caller has to create it. Reads only; the caller writes.
    """
    # Byte identity: nothing is governed that this system has never seen.
    if not media_exists(session, sha256):
        raise GovernanceRejected(NOT_FOUND, "media not found")

    parent_sha256 = statement.derived_from_sha256

    if parent_sha256 == sha256:
        raise GovernanceRejected(UNPROCESSABLE, "media cannot be derived from itself")

    # A record keeps its lineage. Moving it would move the file to whatever split the new
    # lineage is in — the split reassignment this task does not implement.
    if existing is not None and existing.source_lineage_id != statement.source_lineage_id:
        raise GovernanceRejected(CONFLICT, "source lineage cannot be changed")

    # One lineage, one split, and the split is never moved.
    lineage = stored_lineage(session, statement.source_lineage_id)

    if lineage is not None and lineage.dataset_split != statement.dataset_split:
        raise GovernanceRejected(
            CONFLICT,
            f"lineage is assigned to {lineage.dataset_split}; split reassignment is not supported",
        )

    # One recording, one lineage.
    if statement.recording_identity is not None:
        other_lineage = session.execute(
            select(MediaGovernance.source_lineage_id)
            .where(
                MediaGovernance.recording_identity == statement.recording_identity,
                MediaGovernance.source_lineage_id != statement.source_lineage_id,
                MediaGovernance.media_sha256 != sha256,
            )
            .limit(1)
        ).scalar_one_or_none()

        if other_lineage is not None:
            raise GovernanceRejected(
                CONFLICT, "recording identity already belongs to another source lineage"
            )

    if parent_sha256 is not None:
        # The parent must be governed, not merely uploaded: an ungoverned parent has no lineage
        # to inherit, so the child's lineage could not be checked against anything.
        if not media_exists(session, parent_sha256):
            raise GovernanceRejected(CONFLICT, "parent media not found")

        parent = stored_governance(session, parent_sha256)

        if parent is None:
            raise GovernanceRejected(CONFLICT, "parent governance not recorded")

        if parent.source_lineage_id != statement.source_lineage_id:
            raise GovernanceRejected(
                CONFLICT, "derived media must share its parent's source lineage"
            )

        # Walk the new parent's ancestry. If it arrives back here, this write would make the
        # file its own ancestor (A → B → A, or any longer loop).
        def parent_of(node: str) -> str | None:
            record = stored_governance(session, node)
            return None if record is None else record.derived_from_sha256

        if chain_reaches(parent_sha256, sha256, parent_of):
            raise GovernanceRejected(CONFLICT, "derivation would create a cycle")

    return lineage
