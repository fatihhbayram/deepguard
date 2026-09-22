"""What a piece of media actually is, stated by someone in a position to know it.

Ground Truth is the answer the rest of the system is measured against, and it is only worth
anything while it stays independent of everything it is used to measure. Three things in
this codebase look like they could stand in for it, and none of them may:

- an **automated verdict** (`app.risk_engine`) is what the detectors concluded. Ground
  Truth that echoed a verdict would score every detector as correct by construction;
- a **Human Review** is what an analyst concluded from the evidence in front of them. It is
  a judgement about the report, not knowledge of how the file was made;
- **provenance** (`app.provenance_status`) is who signed the bytes and whether the
  signature holds. A valid manifest says the signer stands behind the bytes as signed, not
  that what was signed is genuine — so it cannot establish a label here either.

That separation is enforced as a shape rather than promised in prose. Nothing in this
module imports those modules, `derive_manipulation_family` takes a label and nothing else,
and `GroundTruthContract` refuses any field it does not declare — there is no argument or
field through which a verdict, a review or a manifest could reach a label.
`tests/test_ground_truth.py` pins each of those.

Two axes, and one derived value:

- `source_class` — how the label is known. `OWNER_KNOWN` when whoever made or holds the
  original states it, `CONTROLLED_TEST` when the media was produced under our own control,
  `EXTERNAL_VERIFIED` when an independent party established it, and `UNKNOWN` when none of
  those applies;
- `label` — what the media is;
- `manipulation_family` — the family a label belongs to. It is derived from `label` by the
  one mapping below and never accepted from a caller, so a family cannot be supplied that
  contradicts its label.

`UNKNOWN` is a real state on every axis, not a missing value. "We do not know what this
is" is a statement Ground Truth has to be able to make, and it must never be filled in with
a guess — least of all a guess taken from a verdict.

This is the domain contract only. Persistence, actors, timestamps and the audit trail
around a Ground Truth record belong to the storage layer, not to this vocabulary.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

SourceClass = Literal[
    "OWNER_KNOWN",
    "CONTROLLED_TEST",
    "EXTERNAL_VERIFIED",
    "UNKNOWN",
]

GroundTruthLabel = Literal[
    "GENUINE",
    "AI_GENERATED",
    "FACE_SWAP",
    "AUDIO_MANIPULATION",
    "OTHER_MANIPULATION",
    "UNKNOWN",
]

ManipulationFamily = Literal[
    "NONE",
    "GENERATED_VIDEO",
    "FACE_SWAP",
    "AUDIO_MANIPULATION",
    "OTHER",
    "UNKNOWN",
]

# One row per label and no fallback: a label missing from this table is a defect, and a
# lookup that quietly answered `UNKNOWN` for it would hide the defect behind a valid state.
_FAMILY_BY_LABEL: dict[GroundTruthLabel, ManipulationFamily] = {
    "GENUINE": "NONE",
    "AI_GENERATED": "GENERATED_VIDEO",
    "FACE_SWAP": "FACE_SWAP",
    "AUDIO_MANIPULATION": "AUDIO_MANIPULATION",
    "OTHER_MANIPULATION": "OTHER",
    "UNKNOWN": "UNKNOWN",
}


def derive_manipulation_family(label: GroundTruthLabel) -> ManipulationFamily:
    """The family a Ground Truth label belongs to — the only source of that family.

    Raises `KeyError` for anything that is not a `GroundTruthLabel`, rather than mapping it
    onto `UNKNOWN`.
    """
    return _FAMILY_BY_LABEL[label]


class GroundTruthContract(BaseModel):
    """A Ground Truth statement: how it is known, and what the media is.

    `extra="forbid"` is what keeps `manipulation_family` derived. Pydantic's default would
    silently drop an undeclared field, so a caller passing a family — or a verdict, a review
    or a provenance state — would have it discarded without being told. Here it is refused.
    """

    model_config = ConfigDict(extra="forbid")

    source_class: SourceClass
    label: GroundTruthLabel
    notes: str | None = None

    @property
    def manipulation_family(self) -> ManipulationFamily:
        return derive_manipulation_family(self.label)
