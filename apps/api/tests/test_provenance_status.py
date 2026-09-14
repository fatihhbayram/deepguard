"""The two provenance axes, and the two-way wall between provenance and manipulation.

Three things are pinned here, and the third is the reason the other two matter.

**The mapping** (`app.provenance_status.provenance_state`). Four inputs onto three states,
written out one row at a time rather than derived, so a test cannot agree with a defect by
recomputing it the same wrong way the mapping did. The row that earns the second axis its
existence is the pair at the bottom of `PROVENANCE_MATRIX`: a `SUCCESS` read with no
manifest and a `FAILED` read both answer `UNVERIFIED`, and only `availability` separates
"we looked, and this file carries no credentials" from "we could not look". Before R9-T6
they were one word, and a reader could not tell a finding from a gap.

**The vocabulary.** Two words per axis and no third. In particular nothing in this system
emits `VERIFIED_AUTHENTIC`, and `test_nothing_can_produce_a_verified_authentic_state`
sweeps every input to say so — not because such a state would be hard to reach, but
because there is no evidence any part of this pipeline reads that would justify one.

**Two-way orthogonality**, the invariant R9-T6 exists for. It is asserted in both
directions and neither direction is assumed from the other:

- provenance is not moved by a manipulation verdict — every one of the 36 rows of the v5
  decision matrix is replayed against every provenance input, and the axes come back
  identical each time (`test_no_manipulation_verdict_moves_the_provenance_axes`);
- the manipulation verdict is not moved by provenance — every one of those 36 rows is
  replayed against every provenance state, and the verdict and the rule that fired come
  back identical each time (`test_no_provenance_state_moves_any_manipulation_verdict`).

Both are structural before they are behavioural: `evaluate_v5` takes no provenance
parameter and `provenance_state` takes no verdict parameter, so there is no argument by
which either could reach the other. `test_neither_side_can_even_be_handed_the_other`
asserts that on the signatures themselves, because a signature is what a future change
would have to break first, and the replays above would keep passing right up until it did.

The decision matrix is imported from `test_risk_engine_v5.py` rather than restated. A
second copy would drift the moment a rule changed, and these tests would go on passing
while proving independence from a table nothing decides by any more.
"""

import dataclasses
import inspect

import pytest

from app import provenance_status
from app.api.analyses import ProvenanceSignal
from app.db.models import (
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
)
from app.provenance_status import (
    PROVENANCE_AVAILABILITIES,
    PROVENANCE_AVAILABLE,
    PROVENANCE_PRESENT,
    PROVENANCE_STATUSES,
    PROVENANCE_UNAVAILABLE,
    PROVENANCE_UNVERIFIED,
    ProvenanceState,
    provenance_state,
)
from app.risk_engine import evaluate_v5
from tests.test_risk_engine_v5 import (
    DECISION_MATRIX_V5,
    FACE_STATES,
    SVD_STATES,
)

# The C2PA SDK version the fixtures below report, and the manifest details a present
# manifest carries. None of it is read by the mapping; it is here so that the API-level
# assertions run against a signal shaped like a real one rather than a bare pair.
C2PA_SDK_VERSION = "0.90.14"


def signal(
    *,
    status=SIGNAL_STATUS_SUCCESS,
    manifest_exists=False,
    validation_state=None,
    claim_generator=None,
    signature_issuer=None,
    remote_manifest_url=None,
) -> ProvenanceSignal:
    """The persisted provenance signal as the API hands it back, unless varied.

    The default is the ordinary state of almost every file that reaches this system: it was
    read successfully and carries no Content Credentials at all.
    """
    return ProvenanceSignal(
        provider="c2pa",
        signal_type="provenance",
        status=status,
        provider_version=C2PA_SDK_VERSION,
        manifest_exists=manifest_exists,
        validation_state=validation_state,
        claim_generator=claim_generator,
        signature_issuer=signature_issuer,
        remote_manifest_url=remote_manifest_url,
    )


# --------------------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------------------


def test_each_axis_has_exactly_two_words():
    """Closed vocabularies. A third word on either axis is a contract change, not a detail."""
    assert PROVENANCE_STATUSES == ("UNVERIFIED", "PROVENANCE_PRESENT")
    assert PROVENANCE_AVAILABILITIES == ("AVAILABLE", "UNAVAILABLE")


def test_the_two_axes_share_no_word():
    """`UNVERIFIED` and `UNAVAILABLE` answer different questions and must never be swapped.

    They read alike, which is exactly why a renderer or a client that put one where the other
    belongs would look right. Disjoint vocabularies make that a mismatch rather than a typo.
    """
    assert set(PROVENANCE_STATUSES).isdisjoint(PROVENANCE_AVAILABILITIES)


# --------------------------------------------------------------------------------------
# The mapping — every input a persisted provenance signal can arrive in
# --------------------------------------------------------------------------------------

# The complete mapping, written out. `(signal status, manifest_exists)` onto the pair that
# must come back. The four rows at the bottom are the ones the second axis was added for.
PROVENANCE_MATRIX = {
    # The read ran and found a manifest. Present, and nothing beyond present.
    (SIGNAL_STATUS_SUCCESS, True): (PROVENANCE_PRESENT, PROVENANCE_AVAILABLE),
    # The read ran and found none. A finding about the media, and the ordinary one.
    (SIGNAL_STATUS_SUCCESS, False): (PROVENANCE_UNVERIFIED, PROVENANCE_AVAILABLE),
    # The read reports success but the stored record does not say what it found. Not
    # interpretable, and therefore not reported as a finding about the media.
    (SIGNAL_STATUS_SUCCESS, None): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
    # The read did not happen. Nothing is known either way, whatever the metadata holds.
    (SIGNAL_STATUS_FAILED, None): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
    (SIGNAL_STATUS_FAILED, False): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
    (SIGNAL_STATUS_FAILED, True): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
    (SIGNAL_STATUS_TIMEOUT, None): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
    (SIGNAL_STATUS_TIMEOUT, False): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
    (SIGNAL_STATUS_TIMEOUT, True): (PROVENANCE_UNVERIFIED, PROVENANCE_UNAVAILABLE),
}


def test_the_provenance_matrix_covers_every_combination():
    """Exhaustive over both inputs, not a sample of them."""
    assert set(PROVENANCE_MATRIX) == {
        (status, exists)
        for status in (SIGNAL_STATUS_SUCCESS, SIGNAL_STATUS_FAILED, SIGNAL_STATUS_TIMEOUT)
        for exists in (True, False, None)
    }


@pytest.mark.parametrize(
    ("signal_status", "manifest_exists", "status", "availability"),
    [
        (signal_status, manifest_exists, status, availability)
        for (signal_status, manifest_exists), (status, availability)
        in PROVENANCE_MATRIX.items()
    ],
)
def test_the_provenance_mapping(signal_status, manifest_exists, status, availability):
    state = provenance_state(signal_status, manifest_exists)

    assert state == ProvenanceState(status=status, availability=availability)


def test_a_failed_read_is_never_reported_as_an_absence():
    """The distinction R9-T6 was raised for, asserted on its own rather than inside a table.

    Both answer `UNVERIFIED`. Only `availability` says whether anybody looked, and a caller
    that dropped it would be telling a reader this file carries no credentials on the
    strength of a read that never completed.
    """
    looked = provenance_state(SIGNAL_STATUS_SUCCESS, False)
    could_not_look = provenance_state(SIGNAL_STATUS_FAILED, None)

    assert looked.status == could_not_look.status == PROVENANCE_UNVERIFIED
    assert looked.availability == PROVENANCE_AVAILABLE
    assert could_not_look.availability == PROVENANCE_UNAVAILABLE
    assert looked != could_not_look


@pytest.mark.parametrize("unknown_status", ["", "PENDING", "success", "RUNNING", None])
def test_an_unmapped_status_degrades_to_could_not_be_evaluated(unknown_status):
    """A status this mapping has never heard of must not become a finding about the media.

    A signal status added later and not yet routed through here lands on "we could not
    evaluate provenance", which is true of an unrecognised state and costs a reader nothing.
    The alternative default asserts something about the file on the strength of a word the
    mapping does not understand.
    """
    assert provenance_state(unknown_status, False) == ProvenanceState(
        status=PROVENANCE_UNVERIFIED, availability=PROVENANCE_UNAVAILABLE
    )


# --------------------------------------------------------------------------------------
# Presence is not proof, and absence is not evidence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "validation_state", ["Trusted", "Valid", "Invalid", "OtherError", None]
)
def test_a_present_manifest_reports_present_whatever_its_signature_says(validation_state):
    """`PROVENANCE_PRESENT` means a manifest exists. It never means the signature holds.

    An `Invalid` manifest is present provenance: the credentials are in the file and they do
    not verify, which is a finding worth surfacing loudly. Mapping it to `UNVERIFIED` would
    file a failed signature under the same word as the routine absence most media carries,
    and the failure would vanish from the high-level state entirely.
    """
    carrying = signal(manifest_exists=True, validation_state=validation_state)

    assert carrying.provenance_status == PROVENANCE_PRESENT
    assert carrying.provenance_availability == PROVENANCE_AVAILABLE
    # And the SDK's own word is still there beside it, untranslated.
    assert carrying.validation_state == validation_state


def test_the_axes_never_upgrade_a_signature_the_sdk_rejected():
    """Restated from the other side: no input makes an invalid signature read as valid.

    There is no word on either axis that asserts a signature holds, so the high-level state
    cannot be what tells a reader an `Invalid` manifest is fine. That is a property of the
    vocabulary and is asserted as one.
    """
    rejected = signal(manifest_exists=True, validation_state="Invalid")

    assert rejected.provenance_status in PROVENANCE_STATUSES
    assert "VALID" not in rejected.provenance_status
    assert "AUTHENTIC" not in rejected.provenance_status


def test_a_remote_manifest_is_not_a_present_one():
    """A file naming a manifest it does not contain carries no manifest in these bytes.

    The URL was recorded and deliberately never fetched, so nothing establishes that it
    resolves, let alone that it signs this file. `UNVERIFIED` / `AVAILABLE` is the honest
    reading, and the URL stays beside it as the fact it is.
    """
    remote = signal(
        manifest_exists=False, remote_manifest_url="https://example.invalid/manifest.c2pa"
    )

    assert remote.provenance_status == PROVENANCE_UNVERIFIED
    assert remote.provenance_availability == PROVENANCE_AVAILABLE
    assert remote.remote_manifest_url == "https://example.invalid/manifest.c2pa"


def test_nothing_can_produce_a_verified_authentic_state():
    """No provenance input, and no detector, emits a state that declares media authentic.

    R9-T6's hardest constraint, and it is asserted over the whole input space rather than on
    a chosen case: every signal status crossed with every answer to whether a manifest
    exists, none of which reaches a word this vocabulary does not contain.
    """
    for signal_status, manifest_exists in PROVENANCE_MATRIX:
        state = provenance_state(signal_status, manifest_exists)

        assert state.status in PROVENANCE_STATUSES
        assert state.availability in PROVENANCE_AVAILABILITIES
        assert state.status != "VERIFIED_AUTHENTIC"

    assert "VERIFIED_AUTHENTIC" not in PROVENANCE_STATUSES
    assert not any(
        getattr(provenance_status, name) == "VERIFIED_AUTHENTIC"
        for name in dir(provenance_status)
        if isinstance(getattr(provenance_status, name), str)
    )


# --------------------------------------------------------------------------------------
# Two-way orthogonality
# --------------------------------------------------------------------------------------

# Every state a provenance reading can be in, by the name this suite calls it. The axes each
# one must produce are `PROVENANCE_MATRIX`'s business; here they are the thing held fixed
# while the manipulation side is varied underneath them.
PROVENANCE_INPUTS = {
    "present": (SIGNAL_STATUS_SUCCESS, True),
    "absent": (SIGNAL_STATUS_SUCCESS, False),
    "unreadable": (SIGNAL_STATUS_SUCCESS, None),
    "failed": (SIGNAL_STATUS_FAILED, None),
    "timeout": (SIGNAL_STATUS_TIMEOUT, None),
}


@pytest.mark.parametrize("provenance_input", sorted(PROVENANCE_INPUTS))
@pytest.mark.parametrize(
    ("svd_state", "face_state"), sorted(DECISION_MATRIX_V5, key=str)
)
def test_no_manipulation_verdict_moves_the_provenance_axes(
    provenance_input, svd_state, face_state
):
    """180 comparisons: every provenance input against every row of the v5 decision matrix.

    Direction one of the wall. The verdict is computed for real on each row — a flag on
    either detector, a clean pair of readings, a run with no usable coverage at all — and the
    provenance axes are read beside it. They are identical every time, including on
    `MANIPULATION_DETECTED`, where the temptation to let a finding colour the provenance
    state is strongest and would be exactly wrong: a manipulated file's credentials are no
    more and no less present than an unremarkable one's.
    """
    signal_status, manifest_exists = PROVENANCE_INPUTS[provenance_input]
    expected = provenance_state(signal_status, manifest_exists)

    verdict = evaluate_v5(SVD_STATES[svd_state], FACE_STATES[face_state], None)
    carried = signal(status=signal_status, manifest_exists=manifest_exists)

    assert verdict.risk_level  # the verdict really was produced, not skipped
    assert carried.provenance_status == expected.status
    assert carried.provenance_availability == expected.availability


@pytest.mark.parametrize("provenance_input", sorted(PROVENANCE_INPUTS))
@pytest.mark.parametrize(
    ("svd_state", "face_state"), sorted(DECISION_MATRIX_V5, key=str)
)
def test_no_provenance_state_moves_any_manipulation_verdict(
    provenance_input, svd_state, face_state
):
    """The same 180 comparisons in the other direction, and it is not the same assertion.

    Direction two of the wall. The verdict this row must reach is the one the matrix already
    pins, and it is asserted against the table rather than against a second call to the
    engine — comparing the engine with itself would pass even if provenance had been wired
    into both sides.

    Present credentials do not soften a flag, and absent ones do not harden a quiet reading.
    The rule id is checked alongside the verdict because a verdict reached by a different
    sentence is a different decision wearing the same word.
    """
    expected_verdict, expected_rule = DECISION_MATRIX_V5[(svd_state, face_state)]
    signal_status, manifest_exists = PROVENANCE_INPUTS[provenance_input]

    state = provenance_state(signal_status, manifest_exists)
    decision = evaluate_v5(SVD_STATES[svd_state], FACE_STATES[face_state], None)

    assert state.status in PROVENANCE_STATUSES  # provenance really was evaluated
    assert decision.risk_level == expected_verdict
    assert decision.rule_id == expected_rule


def test_neither_side_can_even_be_handed_the_other():
    """The wall as a pair of signatures, which is what a future change breaks first.

    The replays above would keep passing on the day somebody added a `provenance` parameter
    to the engine with a default that changed nothing yet. This fails that day.
    """
    engine_parameters = set(inspect.signature(evaluate_v5).parameters)
    assert engine_parameters == {"svd", "face", "lip"}
    assert not any(
        "provenance" in name or "c2pa" in name or "manifest" in name
        for name in engine_parameters
    )

    mapping_parameters = set(inspect.signature(provenance_state).parameters)
    assert mapping_parameters == {"signal_status", "manifest_exists"}
    assert not any(
        "risk" in name or "verdict" in name or "score" in name
        for name in mapping_parameters
    )


def test_the_provenance_state_carries_no_room_for_a_verdict():
    """Two fields, both provenance. A third would be somewhere for a finding to be attached."""
    assert [field.name for field in dataclasses.fields(ProvenanceState)] == [
        "status",
        "availability",
    ]
