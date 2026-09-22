"""The Ground Truth vocabulary, its one derived value, and its independence.

Three things are pinned here.

**The vocabulary.** Each axis is written out literally rather than read back from
`typing.get_args`, so adding, removing or renaming a state fails a test instead of quietly
passing through one that recomputes the same list.

**The derivation.** `FAMILY_BY_LABEL` is the mapping written out one row at a time, so a
test cannot agree with a defect by computing the family the same wrong way the module did.
`manipulation_family` is only ever derived: a caller cannot supply one, so a caller cannot
supply one that contradicts its label.

**Independence** from automated verdicts, Human Review and provenance. It is asserted as a
shape, not replayed as behaviour, and without importing any of those modules: the module
imports nothing from the application, the derivation takes a label and nothing else, the
contract declares three fields, and anything else handed to it is refused.
"""

import ast
import inspect
from typing import get_args

import pytest
from pydantic import ValidationError

from app import ground_truth
from app.ground_truth import (
    GroundTruthContract,
    GroundTruthLabel,
    ManipulationFamily,
    SourceClass,
    derive_manipulation_family,
)

SOURCE_CLASSES = ["OWNER_KNOWN", "CONTROLLED_TEST", "EXTERNAL_VERIFIED", "UNKNOWN"]

LABELS = [
    "GENUINE",
    "AI_GENERATED",
    "FACE_SWAP",
    "AUDIO_MANIPULATION",
    "OTHER_MANIPULATION",
    "UNKNOWN",
]

FAMILIES = [
    "NONE",
    "GENERATED_VIDEO",
    "FACE_SWAP",
    "AUDIO_MANIPULATION",
    "OTHER",
    "UNKNOWN",
]

FAMILY_BY_LABEL = {
    "GENUINE": "NONE",
    "AI_GENERATED": "GENERATED_VIDEO",
    "FACE_SWAP": "FACE_SWAP",
    "AUDIO_MANIPULATION": "AUDIO_MANIPULATION",
    "OTHER_MANIPULATION": "OTHER",
    "UNKNOWN": "UNKNOWN",
}


# --- the vocabulary ---------------------------------------------------------------------


def test_the_vocabulary_is_exactly_what_the_contract_names():
    assert list(get_args(SourceClass)) == SOURCE_CLASSES
    assert list(get_args(GroundTruthLabel)) == LABELS
    assert list(get_args(ManipulationFamily)) == FAMILIES


@pytest.mark.parametrize("source_class", SOURCE_CLASSES)
def test_every_source_class_is_accepted(source_class):
    contract = GroundTruthContract(source_class=source_class, label="GENUINE")
    assert contract.source_class == source_class


@pytest.mark.parametrize("label", LABELS)
def test_every_label_is_accepted(label):
    contract = GroundTruthContract(source_class="CONTROLLED_TEST", label=label)
    assert contract.label == label


@pytest.mark.parametrize(
    "source_class",
    ["", "owner_known", "OWNER", "VERIFIED", "VERIFIED_AUTHENTIC", "HUMAN_REVIEW", None, 1],
)
def test_a_source_class_outside_the_vocabulary_is_refused(source_class):
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class=source_class, label="GENUINE")


@pytest.mark.parametrize(
    "label",
    [
        "",
        "genuine",
        "AUTHENTIC",
        "FAKE",
        "MANIPULATED",
        "LIKELY_MANIPULATED",
        "GENERATED_VIDEO",
        "OTHER",
        "NONE",
        None,
        0,
    ],
)
def test_a_label_outside_the_vocabulary_is_refused(label):
    """Includes verdict-shaped words and family words: neither is a Ground Truth label."""
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class="CONTROLLED_TEST", label=label)


def test_both_axes_are_required():
    with pytest.raises(ValidationError):
        GroundTruthContract(label="GENUINE")
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class="OWNER_KNOWN")


def test_notes_are_optional_free_text():
    assert GroundTruthContract(source_class="OWNER_KNOWN", label="GENUINE").notes is None
    contract = GroundTruthContract(
        source_class="OWNER_KNOWN", label="GENUINE", notes="recorded on the owner's phone"
    )
    assert contract.notes == "recorded on the owner's phone"


# --- UNKNOWN is a state, not a gap -------------------------------------------------------


def test_unknown_is_an_explicit_valid_state_on_every_axis():
    contract = GroundTruthContract(source_class="UNKNOWN", label="UNKNOWN")
    assert contract.source_class == "UNKNOWN"
    assert contract.label == "UNKNOWN"
    assert contract.manipulation_family == "UNKNOWN"


def test_unknown_is_never_filled_in():
    """An absent label is refused, not defaulted — and in particular never to GENUINE."""
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class="UNKNOWN")
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class="UNKNOWN", label=None)


# --- the derivation ----------------------------------------------------------------------


@pytest.mark.parametrize(("label", "family"), FAMILY_BY_LABEL.items())
def test_every_label_maps_to_its_one_family(label, family):
    assert derive_manipulation_family(label) == family
    contract = GroundTruthContract(source_class="CONTROLLED_TEST", label=label)
    assert contract.manipulation_family == family


def test_the_mapping_covers_every_label_and_lands_in_the_family_vocabulary():
    assert set(FAMILY_BY_LABEL) == set(LABELS)
    assert set(FAMILY_BY_LABEL.values()) <= set(FAMILIES)


def test_the_mapping_is_deterministic():
    for label, family in FAMILY_BY_LABEL.items():
        assert {derive_manipulation_family(label) for _ in range(5)} == {family}


def test_the_family_does_not_depend_on_how_the_label_is_known():
    for label, family in FAMILY_BY_LABEL.items():
        assert {
            GroundTruthContract(source_class=source_class, label=label).manipulation_family
            for source_class in SOURCE_CLASSES
        } == {family}


def test_a_label_outside_the_vocabulary_has_no_family():
    """No fallback: an unmapped label is an error, not a quiet `UNKNOWN`."""
    for label in ["GENERATED_VIDEO", "FAKE", "", None]:
        with pytest.raises(KeyError):
            derive_manipulation_family(label)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_family_cannot_be_supplied(family):
    """Not even the matching one — the family is derived, never accepted."""
    with pytest.raises(ValidationError):
        GroundTruthContract(
            source_class="CONTROLLED_TEST", label="GENUINE", manipulation_family=family
        )


def test_a_family_cannot_contradict_its_label():
    with pytest.raises(ValidationError):
        GroundTruthContract(
            source_class="CONTROLLED_TEST", label="GENUINE", manipulation_family="FACE_SWAP"
        )
    with pytest.raises(ValidationError):
        GroundTruthContract.model_validate(
            {
                "source_class": "CONTROLLED_TEST",
                "label": "FACE_SWAP",
                "manipulation_family": "NONE",
            }
        )


def test_the_family_cannot_be_assigned():
    contract = GroundTruthContract(source_class="CONTROLLED_TEST", label="GENUINE")
    with pytest.raises(AttributeError):
        contract.manipulation_family = "FACE_SWAP"
    assert contract.manipulation_family == "NONE"


def test_the_family_follows_the_label():
    """There is no stored family that could drift from the label it came from."""
    contract = GroundTruthContract(source_class="CONTROLLED_TEST", label="GENUINE")
    contract.label = "FACE_SWAP"
    assert contract.manipulation_family == "FACE_SWAP"
    assert "manipulation_family" not in contract.model_dump()


# --- independence from verdicts, Human Review and provenance -----------------------------


def test_the_contract_declares_only_its_own_three_fields():
    """A fourth field would be somewhere for a verdict, a review or a manifest to attach."""
    assert list(GroundTruthContract.model_fields) == ["source_class", "label", "notes"]


def test_the_derivation_takes_a_label_and_nothing_else():
    assert list(inspect.signature(derive_manipulation_family).parameters) == ["label"]


def test_the_module_depends_on_nothing_in_the_application():
    """No import from `app` at all — so no verdict, review or provenance module can reach it."""
    tree = ast.parse(inspect.getsource(ground_truth))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            imported.add(node.module.split(".")[0])
    assert imported == {"typing", "pydantic"}


@pytest.mark.parametrize(
    "field",
    [
        # automated verdict
        {"verdict": "LIKELY_MANIPULATED"},
        {"risk_score": 0.91},
        {"decision": "MANIPULATION_DETECTED"},
        {"detector_scores": {"svd": 0.8}},
        # Human Review
        {"review": "confirmed fake"},
        {"human_review": {"decision": "AI_GENERATED"}},
        {"reviewer_label": "FACE_SWAP"},
        # provenance
        {"provenance": "PROVENANCE_PRESENT"},
        {"c2pa": {"manifest": "present"}},
        {"provenance_status": "UNVERIFIED"},
    ],
)
def test_verdict_review_and_provenance_data_are_refused(field):
    """Handed to the contract alongside a valid statement, each is refused, not ignored."""
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class="CONTROLLED_TEST", label="GENUINE", **field)


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "GENUINE"},
        {"human_review": "GENUINE"},
        {"provenance_status": "PROVENANCE_PRESENT"},
    ],
)
def test_a_label_cannot_be_derived_from_verdict_review_or_provenance_alone(payload):
    """Without a label stated directly there is no Ground Truth to build."""
    with pytest.raises(ValidationError):
        GroundTruthContract(source_class="UNKNOWN", **payload)
