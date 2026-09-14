"""The `r9-v5.0.0` decision path, and the v4 path it was added beside rather than over.

Pure functions over three frozen dataclasses — no database, no clock, no configuration — so
every branch, boundary and degenerate input is reachable without staging anything. Its
centrepiece is `DECISION_MATRIX_V5`: the complete cross-product of what the two
decision-eligible detectors can each say, all 36 of them, written out one row at a time rather
than derived, so a test can never agree with a defect by recomputing it the same wrong way the
engine did.

The mouth-dynamics detector is not a third axis of that table, and its absence is the point.
Under v5 it is `evidence_only`: outside `D_total`, `D_usable` and `D_hits` entirely, not a zero
inside them. Putting it in the matrix would state that as 216 rows of coincidence; instead
`test_no_mouth_dynamics_reading_moves_any_verdict` replays all 36 rows against all six states
that detector can be in and asserts the decision is identical every time — 216 comparisons that
prove isolation rather than tabulate it (R9-T1 invariant 1).

**No threshold is mocked anywhere in this module**, for the reason the v4 suite gives: both are
calibrated constants with the 159-clip R4-T1 study behind them, not knobs.

**Two vocabularies, never merged.** `evaluate` still answers `HIGH`/`MEDIUM`/`UNKNOWN` under
`r7-v4.0.0` and every analysis already persisted keeps what it was decided under (R9-T1
invariant 4). The last section below asserts that separation directly: neither function can emit
the other's words, and no evidence makes v4 answer in the R9 vocabulary. `test_risk_engine.py`
remains the proof that v4 itself is unchanged.
"""

import dataclasses
import hashlib

import pytest

from app import risk_engine
from app.risk_engine import (
    D_TOTAL_V5,
    FACE_T_HIGH,
    SVD_T_HIGH,
    FaceEvidence,
    LipEvidence,
    RiskEngineError,
    SvdEvidence,
    UncalibratedEvidence,
    evaluate,
    evaluate_v5,
    face_threshold_reached,
    svd_threshold_reached,
)

# The deployments the calibrations bind to, restated rather than imported, on the same terms as
# the v4 suite: the binding is a fact about a measurement that was taken, not a value the code
# may choose, so an edit to a constant must fail here rather than be agreed with.
VALIDATED_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"
VALIDATED_FACE_CHECKPOINT = (
    "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
    "@4acc494f37eb63d7457166eff2acb45c5b04b9a6"
)
VALIDATED_LIP_MODEL = (
    "https://github.com/ahaliassos/LipForensics"
    "@d0bf5553bfb9676f1771d590472b26a3a76de894"
    "+4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
)

EXPECTED_RULES_VERSION_V5 = "r9-v5.0.0"
EXPECTED_RULES_VERSION_V4 = "r7-v4.0.0"

# The two artifacts a v5 decision stands on and the single identity it is stored under —
# character-for-character the ones v4 and v3 were decided under, because v5 adopts no new
# measurement and drops none.
EXPECTED_SVD_FACE_CALIBRATION_ID = (
    "cab2ea262bb7e41cb87e49bdb3dad53ecd0f02248035a993f9fcb033363afd1e"
)
EXPECTED_LIP_CALIBRATION_ID = (
    "85cb7484ab74d5821b1f4fa7ba917588dcaa98354f5f7a82c3080b624c9b8a29"
)
EXPECTED_CALIBRATION_ID = (
    "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"
)

MANIPULATION_DETECTED = "MANIPULATION_DETECTED"
NO_SIGNAL = "NO_CALIBRATED_MANIPULATION_SIGNAL"
INCONCLUSIVE = "INCONCLUSIVE"

TOTAL_CLIPS = 4
FRAMES_SCORED = 32
WINDOWS_SCORED = 6

# Four real clips from the R4-T1 corpus, the two detectors' scores on each. They are the
# measurement behind the rule that neither detector may be held back by the other: on both clips
# one detector is loud and the other is sitting near its floor.
#   sonic_en_03 (generated video) — NVIDIA 0.9961, face model 0.0053
#   ffpp_dev_Deepfakes_106_198 (face swap) — NVIDIA 0.1648, face model 0.9943
CORPUS_SYNTHETIC_SVD = 0.9961
CORPUS_SYNTHETIC_FACE = 0.0053
CORPUS_FACESWAP_SVD = 0.1648
CORPUS_FACESWAP_FACE = 0.9943


def svd(
    *,
    score=0.5,
    provider="nvidia",
    signal_type="synthetic_video",
    status="SUCCESS",
    provider_version=VALIDATED_FUNCTION_ID,
    total_clips=TOTAL_CLIPS,
) -> SvdEvidence:
    """Persisted synthetic-video evidence from the calibrated deployment, unless varied.

    Defaults are a healthy, eligible signal reading below its threshold, so every test below
    states only the one thing it is about and nothing else can drift underneath it.
    """
    return SvdEvidence(
        provider=provider,
        signal_type=signal_type,
        status=status,
        provider_version=provider_version,
        score=score,
        total_clips=total_clips,
    )


def face(
    *,
    score=0.5,
    provider="efficientnet-b7",
    signal_type="face_manipulation",
    status="SUCCESS",
    provider_version=VALIDATED_FACE_CHECKPOINT,
    frames_scored=FRAMES_SCORED,
) -> FaceEvidence:
    """Persisted face-manipulation evidence from the calibrated artifact, unless varied."""
    return FaceEvidence(
        provider=provider,
        signal_type=signal_type,
        status=status,
        provider_version=provider_version,
        score=score,
        frames_scored=frames_scored,
    )


def lip(
    *,
    score=0.05,
    provider="lipforensics",
    signal_type="lip_forensics",
    status="SUCCESS",
    provider_version=VALIDATED_LIP_MODEL,
    windows_scored=WINDOWS_SCORED,
) -> LipEvidence:
    """Persisted mouth-dynamics evidence from the calibrated model, unless varied.

    The default is not 0.5, because 0.5 is above this detector's own historical threshold and the
    three scales are unrelated. Under v5 nothing compares it against anything at all — the value
    is chosen for the tests that vary it, not for a branch it could take.
    """
    return LipEvidence(
        provider=provider,
        signal_type=signal_type,
        status=status,
        provider_version=provider_version,
        score=score,
        windows_scored=windows_scored,
    )


# --------------------------------------------------------------------------------------
# The ruleset, and the calibration it is bound to
# --------------------------------------------------------------------------------------


def test_the_v5_ruleset_names_itself_and_the_calibration_behind_it():
    assert risk_engine.RULES_VERSION_V5 == EXPECTED_RULES_VERSION_V5
    assert risk_engine.CALIBRATION_ID == EXPECTED_CALIBRATION_ID


def test_v5_is_a_different_ruleset_from_every_one_before_it():
    """A stored decision is only re-derivable if its version pins the logic exactly."""
    assert risk_engine.RULES_VERSION_V5 != risk_engine.RULES_VERSION
    assert risk_engine.RULES_VERSION_V5 not in {
        "p7-v1.0.0",
        "r4-v2.0.0",
        "r5-v3.0.0",
        "r7-v4.0.0",
    }


def test_v5_kept_the_calibration_v4_was_decided_under():
    """Because it changed no threshold, adopted no artifact and dropped none.

    `RULES_VERSION_V5` names the rules; `CALIBRATION_ID` names the measurements those rules read.
    v5 rewrites the first and leaves the second exactly where R4-T1 and R5-T3 put it, so minting
    a new id here would assert a measurement nobody took.
    """
    assert risk_engine.CALIBRATION_ID == EXPECTED_CALIBRATION_ID
    assert risk_engine.SVD_FACE_CALIBRATION_ID == EXPECTED_SVD_FACE_CALIBRATION_ID
    assert risk_engine.LIP_CALIBRATION_ID == EXPECTED_LIP_CALIBRATION_ID
    assert risk_engine.SVD_T_HIGH == 0.9550971388816833
    assert risk_engine.FACE_T_HIGH == 0.9867589175701141


def test_the_v5_calibration_id_is_still_derived_from_both_artifacts():
    """The construction, not just the literal — so it cannot change without notice."""
    derived = hashlib.sha256(
        f"{EXPECTED_SVD_FACE_CALIBRATION_ID}\n{EXPECTED_LIP_CALIBRATION_ID}".encode()
    ).hexdigest()

    assert derived == EXPECTED_CALIBRATION_ID
    assert risk_engine.CALIBRATION_ID == derived


def test_the_denominator_is_frozen_at_two_and_above_zero():
    """`D_total` is read from the ruleset, never from the rows that happen to exist.

    Two decision-eligible detectors, and above zero — which is what makes the contract's
    `D_total == 0` row unreachable here rather than silently answered by a branch. A denominator
    derived from the rows present would let a detector that was never invoked *improve* coverage
    by being absent.
    """
    assert D_TOTAL_V5 == 2
    assert D_TOTAL_V5 > 0


def test_the_v5_verdict_vocabulary_is_exactly_three_words():
    assert risk_engine.VERDICT_MANIPULATION_DETECTED == MANIPULATION_DETECTED
    assert risk_engine.VERDICT_NO_SIGNAL == NO_SIGNAL
    assert risk_engine.VERDICT_INCONCLUSIVE == INCONCLUSIVE


def test_no_v5_rule_wears_an_id_an_earlier_ruleset_already_used():
    """Reusing an id would rewrite what a stored row said.

    `app.risk_trace` still explains `p7-v1.0.0`, `r5-v3.0.0` and `r7-v4.0.0` rows through their
    own sentences, `R103` included — the id v3 used for a mouth-dynamics HIGH, retired and never
    re-let. The two tables are disjoint, and this is the assertion that keeps them so.
    """
    v5_ids = {
        value
        for name, value in vars(risk_engine).items()
        if name.startswith("RULE_V5_") and isinstance(value, str)
    }
    earlier_ids = {
        value
        for name, value in vars(risk_engine).items()
        if name.startswith("RULE_")
        and not name.startswith("RULE_V5_")
        and isinstance(value, str)
    }

    assert v5_ids == {"R9-100", "R9-101", "R9-102", "R9-200", "R9-300", "R9-301"}
    assert v5_ids & earlier_ids == set()
    assert "R103" not in v5_ids


# --------------------------------------------------------------------------------------
# The native comparators
# --------------------------------------------------------------------------------------


def test_each_detector_is_compared_against_its_own_operating_point():
    """There is no universal threshold in this module, and v5 does not invent one.

    The two operating points are different numbers on different scales, and a score is only ever
    handed to the comparator belonging to the detector that produced it.
    """
    assert SVD_T_HIGH != FACE_T_HIGH

    # 0.9861 is above NVIDIA's point and below the face model's. Same number, opposite answers —
    # which is what a universal comparator would have got wrong.
    assert svd_threshold_reached(svd(score=0.9861)) is True
    assert face_threshold_reached(face(score=0.9861)) is False


@pytest.mark.parametrize(
    ("score", "reached"),
    [
        (SVD_T_HIGH, True),
        (0.9550971388816832, False),
        (0.9550971388816834, True),
        (1.0, True),
        (0.0, False),
    ],
)
def test_the_synthetic_video_comparator_at_and_around_its_boundary(score, reached):
    """Equality reaches the threshold, and that is frozen in the comparator's definition."""
    assert svd_threshold_reached(svd(score=score)) is reached


@pytest.mark.parametrize(
    ("score", "reached"),
    [
        (FACE_T_HIGH, True),
        (0.986758917570114, False),
        (0.9867589175701142, True),
        (1.0, True),
        (0.0, False),
    ],
)
def test_the_face_comparator_at_and_around_its_boundary(score, reached):
    assert face_threshold_reached(face(score=score)) is reached


@pytest.mark.parametrize("index", range(0, 1001))
def test_the_v5_comparators_agree_with_v4_across_both_whole_scales(index):
    """v5 reuses v4's comparator rather than restating a rule that happens to match.

    Swept at every thousandth of both scales and pinned at both operating points: wherever v4
    calls a reading HIGH on one detector alone, v5 calls it a hit on that detector, and nowhere
    else. If either comparator is ever changed without the other, this fails on the score where
    they part.
    """
    score = index / 1000

    v4_svd_high = evaluate(svd(score=score), None, None).risk_level == "HIGH"
    v4_face_high = evaluate(None, face(score=score), None).risk_level == "HIGH"

    assert svd_threshold_reached(svd(score=score)) is v4_svd_high
    assert face_threshold_reached(face(score=score)) is v4_face_high


def test_the_v5_comparators_agree_with_v4_at_the_operating_points_themselves():
    """The one score neither sweep above can land on exactly."""
    assert svd_threshold_reached(svd(score=SVD_T_HIGH)) is (
        evaluate(svd(score=SVD_T_HIGH), None, None).risk_level == "HIGH"
    )
    assert face_threshold_reached(face(score=FACE_T_HIGH)) is (
        evaluate(None, face(score=FACE_T_HIGH), None).risk_level == "HIGH"
    )


# --------------------------------------------------------------------------------------
# The decision matrix — every combination of what the two deciding detectors can say
# --------------------------------------------------------------------------------------

# One evidence value per state a decision-eligible detector can be in.
#
#   flagged    — calibrated, readable, and its own comparator reports the operating point met
#   below      — the same, and the comparator reports it not met. A reading, not a clean finding
#   invalid    — the calibrated deployment answered and its figures cannot be read
#   ineligible — something answered, but not the deployment the thresholds were measured on
#   absent     — no row at all: the detector was never invoked (`no-row`)
#   abstained  — the detector declined to score, because a precondition of its own method was
#                absent. Persisted as a FAILED row with a null score, per R9-T1 section 3.3
#
# Only `flagged` and `below` are `usable_reading`s. The other four are unresolved coverage, and
# `abstained` is among them *by default* — it is not declared resolved for either detector here.
SVD_STATES = {
    "flagged": svd(score=0.9931),
    "below": svd(score=0.4646),
    "invalid": svd(score=None),
    "ineligible": svd(score=0.9931, provider_version="some-other-function"),
    "absent": None,
    "abstained": svd(score=None, status="FAILED"),
}

FACE_STATES = {
    "flagged": face(score=0.9931),
    "below": face(score=0.4646),
    "invalid": face(score=None),
    "ineligible": face(score=0.9931, provider_version="some-other-checkpoint"),
    "absent": None,
    "abstained": face(score=None, status="FAILED", frames_scored=0),
}

# Every state the evidence-only detector can be in. It is not an axis of the table below — under
# v5 it enters no arithmetic at all — but every one of these is replayed against every row of it.
LIP_STATES = {
    "flagged": lip(score=0.9931),
    "below": lip(score=0.0154),
    "invalid": lip(score=None),
    "ineligible": lip(score=0.9931, provider_version="some-other-checkpoint"),
    "absent": None,
    "abstained": lip(score=None, status="FAILED", windows_scored=0),
}

# What v5 must conclude for each of the 36 combinations, written out rather than computed, and
# grouped by the rule that has to fire. Reading a group is the fastest way to see the properties
# that matter most:
#
#   * every combination with a `flagged` in it is MANIPULATION_DETECTED. Nothing the other
#     detector says — quiet, broken, uncalibrated, abstaining or missing — softens a hit, and
#     incomplete coverage beside one is reported as coverage rather than as doubt;
#   * exactly one flag is attributed to the detector that produced it; both are `R9-100`, which
#     credits neither with the other's evidence;
#   * NO_CALIBRATED_MANIPULATION_SIGNAL has exactly one row. It takes both detectors reading
#     below their own thresholds and nothing less: one clean reading is not coverage;
#   * the remaining 24 rows are INCONCLUSIVE, split only by how much of the expected coverage was
#     obtained. `abstained` sits with `absent` and `ineligible` there, not with `below`.
DECISION_MATRIX_V5 = {
    # Both decision-eligible detectors reached their own thresholds independently.
    ("flagged", "flagged"): (MANIPULATION_DETECTED, "R9-100"),
    # The synthetic-video detector reached its threshold. The face model's state is named in no
    # condition that produced this verdict.
    ("flagged", "below"): (MANIPULATION_DETECTED, "R9-101"),
    ("flagged", "invalid"): (MANIPULATION_DETECTED, "R9-101"),
    ("flagged", "ineligible"): (MANIPULATION_DETECTED, "R9-101"),
    ("flagged", "absent"): (MANIPULATION_DETECTED, "R9-101"),
    ("flagged", "abstained"): (MANIPULATION_DETECTED, "R9-101"),
    # The face model reached its threshold, and the same holds in the other direction.
    ("below", "flagged"): (MANIPULATION_DETECTED, "R9-102"),
    ("invalid", "flagged"): (MANIPULATION_DETECTED, "R9-102"),
    ("ineligible", "flagged"): (MANIPULATION_DETECTED, "R9-102"),
    ("absent", "flagged"): (MANIPULATION_DETECTED, "R9-102"),
    ("abstained", "flagged"): (MANIPULATION_DETECTED, "R9-102"),
    # Complete coverage, no hit: both expected detectors produced a usable reading and neither
    # reached its operating point. This is the only row in the table that may say so.
    ("below", "below"): (NO_SIGNAL, "R9-200"),
    # Partial coverage, no hit — one usable reading out of the two the ruleset expects.
    ("below", "invalid"): (INCONCLUSIVE, "R9-300"),
    ("below", "ineligible"): (INCONCLUSIVE, "R9-300"),
    ("below", "absent"): (INCONCLUSIVE, "R9-300"),
    ("below", "abstained"): (INCONCLUSIVE, "R9-300"),
    ("invalid", "below"): (INCONCLUSIVE, "R9-300"),
    ("ineligible", "below"): (INCONCLUSIVE, "R9-300"),
    ("absent", "below"): (INCONCLUSIVE, "R9-300"),
    ("abstained", "below"): (INCONCLUSIVE, "R9-300"),
    # No coverage at all, and therefore no hit.
    ("invalid", "invalid"): (INCONCLUSIVE, "R9-301"),
    ("invalid", "ineligible"): (INCONCLUSIVE, "R9-301"),
    ("invalid", "absent"): (INCONCLUSIVE, "R9-301"),
    ("invalid", "abstained"): (INCONCLUSIVE, "R9-301"),
    ("ineligible", "invalid"): (INCONCLUSIVE, "R9-301"),
    ("ineligible", "ineligible"): (INCONCLUSIVE, "R9-301"),
    ("ineligible", "absent"): (INCONCLUSIVE, "R9-301"),
    ("ineligible", "abstained"): (INCONCLUSIVE, "R9-301"),
    ("absent", "invalid"): (INCONCLUSIVE, "R9-301"),
    ("absent", "ineligible"): (INCONCLUSIVE, "R9-301"),
    ("absent", "absent"): (INCONCLUSIVE, "R9-301"),
    ("absent", "abstained"): (INCONCLUSIVE, "R9-301"),
    ("abstained", "invalid"): (INCONCLUSIVE, "R9-301"),
    ("abstained", "ineligible"): (INCONCLUSIVE, "R9-301"),
    ("abstained", "absent"): (INCONCLUSIVE, "R9-301"),
    ("abstained", "abstained"): (INCONCLUSIVE, "R9-301"),
}


def test_the_v5_matrix_covers_every_combination():
    """The table is exhaustive over both deciding detectors' states, not a sample of them."""
    assert set(DECISION_MATRIX_V5) == {
        (svd_state, face_state) for svd_state in SVD_STATES for face_state in FACE_STATES
    }
    assert len(DECISION_MATRIX_V5) == 36


@pytest.mark.parametrize(
    ("svd_state", "face_state", "verdict", "rule_id"),
    [
        (svd_state, face_state, verdict, rule_id)
        for (svd_state, face_state), (verdict, rule_id) in DECISION_MATRIX_V5.items()
    ],
)
def test_the_v5_decision_matrix(svd_state, face_state, verdict, rule_id):
    decision = evaluate_v5(SVD_STATES[svd_state], FACE_STATES[face_state], None)

    assert decision.risk_level == verdict
    assert decision.rule_id == rule_id


# --------------------------------------------------------------------------------------
# Evidence-only isolation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("svd_state", "face_state"), sorted(DECISION_MATRIX_V5, key=str)
)
def test_no_mouth_dynamics_reading_moves_any_verdict(svd_state, face_state):
    """216 comparisons: every row of the table against every state that detector can be in.

    This is R9-T1 invariant 1 stated as an experiment rather than as a comment. The
    mouth-dynamics detector flagging, reading below, answering illegibly, running on an
    uncalibrated deployment, abstaining or never being invoked leaves the verdict, the rule id,
    the ruleset version and the calibration identity character-for-character identical — which is
    what "outside the arithmetic, not a zero inside it" has to mean if it means anything.
    """
    without_lip = evaluate_v5(SVD_STATES[svd_state], FACE_STATES[face_state], None)

    for lip_state in LIP_STATES:
        assert (
            evaluate_v5(
                SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
            )
            == without_lip
        )


def test_a_mouth_dynamics_crossing_cannot_complete_coverage():
    """The one case where v5 is arithmetically stricter than v4, and deliberately.

    v4 counted a readable mouth-dynamics signal toward the difference between `R200` and `R201`.
    v5 counts it nowhere: one deciding detector reading below its threshold is partial coverage,
    however loudly the evidence-only detector answered. A detector that may not decide may not
    complete coverage either.
    """
    decision = evaluate_v5(svd(score=0.4646), None, lip(score=0.9931))

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"


def test_a_calibrated_and_an_uncalibrated_mouth_dynamics_deployment_decide_alike():
    """Same SVD, same B7, and the evidence-only detector swapped underneath them.

    Calibrated and readable on one side, an uncalibrated deployment of the same model on the
    other, and every field of the decision identical — verdict, rule id, ruleset version and
    calibration identity. Under v4 the difference between these two is real: a readable
    mouth-dynamics signal is what separates an `R200` from an `R201`. Under v5 it is nothing at
    all, and that is the whole content of `evidence_only`.

    Asserted across every state of the deciding pair, so the claim is not a property of one
    convenient row.
    """
    calibrated = lip(score=0.9931)
    uncalibrated = lip(score=0.9931, provider_version="some-other-checkpoint")

    for svd_state in SVD_STATES:
        for face_state in FACE_STATES:
            evidence = (SVD_STATES[svd_state], FACE_STATES[face_state])

            assert evaluate_v5(*evidence, calibrated) == evaluate_v5(
                *evidence, uncalibrated
            )

    # And under v4 the same swap is not always a no-op, which is what makes the above a
    # statement about this ruleset rather than about these two evidence values.
    v4_calibrated = evaluate(svd(score=0.4646), face(score=0.4646), calibrated)
    v4_uncalibrated = evaluate(svd(score=0.4646), face(score=0.4646), uncalibrated)

    assert v4_calibrated.rule_id == "R200"
    assert v4_uncalibrated.rule_id == "R201"


def test_a_failed_mouth_dynamics_model_removes_no_coverage():
    """Complete coverage is complete with the evidence-only detector dead on the floor."""
    decision = evaluate_v5(
        svd(score=0.4646), face(score=0.4646), lip(score=None, status="FAILED")
    )

    assert decision.risk_level == NO_SIGNAL
    assert decision.rule_id == "R9-200"


# --------------------------------------------------------------------------------------
# The R9-T1 truth table, row by row
# --------------------------------------------------------------------------------------


def test_row_1_a_hit_with_complete_coverage_is_manipulation_detected():
    decision = evaluate_v5(svd(score=0.9931), face(score=0.4646), lip())

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-101"


def test_row_2_a_hit_with_incomplete_coverage_is_still_manipulation_detected():
    """Absolute precedence. Incomplete coverage beside a hit is reported, not weighed."""
    decision = evaluate_v5(svd(score=0.9931), None, lip())

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-101"


def test_row_3_complete_coverage_with_no_hit_is_no_calibrated_signal():
    decision = evaluate_v5(svd(score=0.4646), face(score=0.4646), lip())

    assert decision.risk_level == NO_SIGNAL
    assert decision.rule_id == "R9-200"


def test_row_4_incomplete_coverage_with_no_hit_is_inconclusive():
    decision = evaluate_v5(svd(score=0.4646), None, lip())

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"


def test_a_dual_failure_is_inconclusive():
    """Both decision-eligible detectors failed: `D_usable` is 0 against a frozen `D_total` of 2."""
    decision = evaluate_v5(
        svd(score=None, status="FAILED"), face(score=None, status="FAILED"), lip()
    )

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-301"


def test_one_detector_failing_and_the_other_hitting_is_manipulation_detected():
    decision = evaluate_v5(
        svd(score=None, status="TIMEOUT"), face(score=0.9931), lip()
    )

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-102"


def test_one_detector_failing_and_the_other_reading_below_is_inconclusive():
    """A single clean reading is not coverage."""
    decision = evaluate_v5(
        svd(score=None, status="TIMEOUT"), face(score=0.4646), lip()
    )

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"


def test_a_never_invoked_detector_stays_in_the_denominator():
    """`no-row` beside a reading below threshold is incomplete coverage, not complete coverage.

    This is what the frozen `D_total` buys. Derived from the rows present, the denominator would
    have been 1 here and this analysis would have reported complete coverage and
    NO_CALIBRATED_MANIPULATION_SIGNAL — a detector improving coverage by never running.
    """
    decision = evaluate_v5(None, face(score=0.4646), lip())

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"


def test_a_clean_run_on_an_uncalibrated_deployment_is_not_coverage():
    """Row 3 becomes row 4: the ruleset has a number it is not entitled to compare."""
    decision = evaluate_v5(
        svd(score=0.4646, provider_version="847b6e53-0133-452d-ab85-d7acf3ace723-preview"),
        face(score=0.4646),
        lip(),
    )

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"


def test_a_reading_exactly_at_both_operating_points_is_manipulation_detected():
    """The boundary is the comparator's, frozen per detector, and it is reached at equality."""
    decision = evaluate_v5(svd(score=SVD_T_HIGH), face(score=FACE_T_HIGH), lip())

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-100"


# --------------------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------------------


def test_an_abstaining_face_detector_is_unresolved_coverage_not_a_clean_finding():
    """No detector's abstention is declared resolved under v5, and that is the default.

    A clip with no face in any sampled frame is persisted as a FAILED row with a null score. It
    is a statement about the media, not a finding about it: R4-T1 recorded 8 abstentions and
    every one of them was on generated video. Treating it as a usable reading here would let a
    detector buy complete coverage by declining to look.
    """
    decision = evaluate_v5(
        svd(score=0.4646), face(score=None, status="FAILED", frames_scored=0), lip()
    )

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"
    assert decision.risk_level != NO_SIGNAL


def test_an_abstention_that_reached_a_success_row_is_still_not_a_usable_reading():
    """Guarding the stored figures rather than the live path.

    The detector raises before it can write a mean over zero crops today. If some future change
    writes one, a score aggregated over nothing is still not a reading, and coverage is still
    incomplete.
    """
    decision = evaluate_v5(svd(score=0.4646), face(score=0.99, frames_scored=0), lip())

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-300"


def test_both_detectors_abstaining_is_inconclusive_never_no_signal():
    decision = evaluate_v5(
        svd(score=None, status="FAILED", total_clips=0),
        face(score=None, status="FAILED", frames_scored=0),
        lip(),
    )

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-301"


def test_an_abstaining_detector_cannot_hold_back_the_other_detectors_hit():
    decision = evaluate_v5(
        svd(score=0.9931), face(score=None, status="FAILED", frames_scored=0), lip()
    )

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-101"


# --------------------------------------------------------------------------------------
# The disagreement policy, on the clips that measured it
# --------------------------------------------------------------------------------------


def test_a_quiet_face_model_cannot_hold_back_a_synthetic_video_finding():
    """`sonic_en_03` from the R4-T1 corpus: generated video, NVIDIA 0.9961, face model 0.0053."""
    decision = evaluate_v5(
        svd(score=CORPUS_SYNTHETIC_SVD), face(score=CORPUS_SYNTHETIC_FACE), lip()
    )

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-101"


def test_a_quiet_synthetic_video_detector_cannot_hold_back_a_face_finding():
    """`ffpp_dev_Deepfakes_106_198`: a face swap, NVIDIA 0.1648, face model 0.9943."""
    decision = evaluate_v5(
        svd(score=CORPUS_FACESWAP_SVD), face(score=CORPUS_FACESWAP_FACE), lip()
    )

    assert decision.risk_level == MANIPULATION_DETECTED
    assert decision.rule_id == "R9-102"


def test_requiring_the_detectors_to_agree_would_detect_nothing():
    """At their operating points the two never agreed once across 159 clips.

    Both real manipulations above are found by exactly one detector. A rule requiring
    corroboration would have returned NO_CALIBRATED_MANIPULATION_SIGNAL for both — a verdict
    about someone's media that the measurement says would be wrong.
    """
    both = [
        evaluate_v5(
            svd(score=CORPUS_SYNTHETIC_SVD), face(score=CORPUS_SYNTHETIC_FACE), lip()
        ),
        evaluate_v5(
            svd(score=CORPUS_FACESWAP_SVD), face(score=CORPUS_FACESWAP_FACE), lip()
        ),
    ]

    assert [decision.risk_level for decision in both] == [
        MANIPULATION_DETECTED,
        MANIPULATION_DETECTED,
    ]
    assert {decision.rule_id for decision in both} == {"R9-101", "R9-102"}


def test_no_arithmetic_combines_the_scores():
    """0.98 and 0.20 have the same mean either way round; these two decisions do not match."""
    svd_loud = evaluate_v5(svd(score=0.98), face(score=0.20), lip())
    face_loud = evaluate_v5(svd(score=0.20), face(score=0.98), lip())

    # 0.98 is above NVIDIA's operating point (0.9551) and below the face model's (0.9868).
    assert svd_loud.risk_level == MANIPULATION_DETECTED
    assert svd_loud.rule_id == "R9-101"
    assert face_loud.risk_level == NO_SIGNAL
    assert face_loud.rule_id == "R9-200"


# --------------------------------------------------------------------------------------
# Uncalibrated evidence
# --------------------------------------------------------------------------------------


def test_the_two_deciding_signals_cannot_be_supplied_in_each_others_place():
    """The guard covers the slots that decide, which is where an operating point can be reached."""
    with pytest.raises(UncalibratedEvidence):
        evaluate_v5(face(), None, None)  # type: ignore[arg-type]

    with pytest.raises(UncalibratedEvidence):
        evaluate_v5(None, svd(), None)  # type: ignore[arg-type]


def test_uncalibrated_evidence_is_refused_before_any_rule_is_read():
    """Not classified and then discarded — the rules never run at all."""
    with pytest.raises(UncalibratedEvidence) as raised:
        evaluate_v5(svd(score=0.9931), lip(), None)  # type: ignore[arg-type]

    assert "LipEvidence" in str(raised.value)
    assert "FaceEvidence" in str(raised.value)


def test_the_evidence_only_slot_is_not_guarded_because_a_guard_is_an_outcome():
    """Nothing in the evidence-only slot can stop v5 from producing a verdict.

    This is the last place the isolation could have leaked, and it is the subtler half of R9-T1
    invariant 1. A detector that cannot change the verdict can still change whether there *is*
    one, if the engine refuses to return on account of what arrived in its slot — and then that
    detector's deployment metadata is deciding the outcome after all, just by a different route.
    Under v5 `lip` is never read: an uncalibrated deployment, a shadow-mode subclass and an
    object of another type entirely are all carried through to the same decision the two
    deciding detectors produced on their own.

    The guard is untouched where it matters. `svd` and `face` are decision-eligible, a shadow
    observation reaching one of their operating points is the risk R6-T1 named, and the test
    above still proves both slots refuse.
    """

    class ShadowObservation(LipEvidence):
        pass

    shadow = ShadowObservation(
        provider="lipforensics",
        signal_type="lip_forensics",
        status="SUCCESS",
        provider_version=VALIDATED_LIP_MODEL,
        score=0.99,
        windows_scored=WINDOWS_SCORED,
    )
    expected = evaluate_v5(svd(score=0.4646), face(score=0.4646), None)

    assert evaluate_v5(svd(score=0.4646), face(score=0.4646), shadow) == expected
    # Not even a LipEvidence at all. The parameter is accepted for signature compatibility with
    # `evaluate` and the callers R9-T3 will write; the decision path does not look at it.
    assert evaluate_v5(svd(score=0.4646), face(score=0.4646), svd()) == expected  # type: ignore[arg-type]
    assert evaluate_v5(svd(score=0.4646), face(score=0.4646), object()) == expected  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Properties of the ruleset as a whole
# --------------------------------------------------------------------------------------


def test_the_v5_ruleset_emits_only_the_three_supported_verdicts():
    verdicts = {
        evaluate_v5(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).risk_level
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    assert verdicts == {MANIPULATION_DETECTED, NO_SIGNAL, INCONCLUSIVE}


def test_only_the_six_documented_v5_rules_can_fire():
    rules = {
        evaluate_v5(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).rule_id
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    assert rules == {"R9-100", "R9-101", "R9-102", "R9-200", "R9-300", "R9-301"}


def test_every_v5_rule_id_the_module_names_is_reachable():
    """No v5 rule constant is declared and left unfirable, and none fires that is not declared."""
    declared = {
        value
        for name, value in vars(risk_engine).items()
        if name.startswith("RULE_V5_") and isinstance(value, str)
    }
    fired = {
        evaluate_v5(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).rule_id
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    assert declared == fired


def test_every_v5_decision_carries_the_ruleset_and_calibration_that_made_it():
    """Including the INCONCLUSIVEs. A decision with no trace is not explainable later."""
    for svd_state in SVD_STATES:
        for face_state in FACE_STATES:
            decision = evaluate_v5(SVD_STATES[svd_state], FACE_STATES[face_state], lip())

            assert decision.rules_version == EXPECTED_RULES_VERSION_V5
            assert decision.calibration_id == EXPECTED_CALIBRATION_ID


def test_no_verdict_ever_says_the_media_is_authentic():
    """R9-T1 invariant 3, at the only place a verdict string is minted.

    A detector that found no manipulation has not found authenticity, and the verdict this engine
    emits must not be a word a renderer can shorten into one. The check is on the vocabulary
    itself, because every surface downstream takes its label from here.
    """
    forbidden = {"REAL", "GENUINE", "AUTHENTIC", "VERIFIED", "CLEAN", "SAFE", "FAKE"}

    verdicts = {
        evaluate_v5(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).risk_level
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    for verdict in verdicts:
        assert not forbidden & set(verdict.upper().split("_"))


def test_the_same_evidence_always_yields_the_same_v5_decision():
    """Stateless and deterministic: nothing accumulates between calls."""
    first = svd(score=0.9550971)
    second = face(score=0.9867588)
    third = lip(score=0.2296253)

    assert (
        evaluate_v5(first, second, third)
        == evaluate_v5(first, second, third)
        == evaluate_v5(
            svd(score=0.9550971), face(score=0.9867588), lip(score=0.2296253)
        )
    )


def test_a_v5_decision_is_immutable_once_made():
    decision = evaluate_v5(svd(score=0.99), None, None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.risk_level = INCONCLUSIVE


def test_evaluate_v5_defaults_every_signal_to_absent():
    decision = evaluate_v5()

    assert decision.risk_level == INCONCLUSIVE
    assert decision.rule_id == "R9-301"


def test_the_impossible_is_a_defect_and_not_a_default(monkeypatch):
    """A denominator out of step with the detectors counted raises rather than persisting.

    Unreachable while `D_TOTAL_V5` is 2 and two booleans are summed, which is why it is forced
    here. R9-T1 invariant 9: inventing a fall-through verdict would ship a bug as a claim about
    someone's media.
    """
    monkeypatch.setattr(risk_engine, "D_TOTAL_V5", 1)

    with pytest.raises(RiskEngineError):
        evaluate_v5(svd(score=0.4646), face(score=0.4646), lip())


# --------------------------------------------------------------------------------------
# Two vocabularies, never merged
# --------------------------------------------------------------------------------------


def test_v4_still_answers_in_its_own_vocabulary_under_its_own_ruleset():
    """`evaluate` is untouched by R9-T2 and keeps deciding every analysis persisted so far."""
    levels = set()
    versions = set()

    for svd_state in SVD_STATES:
        for face_state in FACE_STATES:
            for lip_state in LIP_STATES:
                decision = evaluate(
                    SVD_STATES[svd_state],
                    FACE_STATES[face_state],
                    LIP_STATES[lip_state],
                )
                levels.add(decision.risk_level)
                versions.add(decision.rules_version)

    assert levels == {"HIGH", "MEDIUM", "UNKNOWN"}
    assert versions == {EXPECTED_RULES_VERSION_V4}


def test_neither_engine_can_emit_the_others_words():
    """No evidence makes v4 speak R9, and none makes v5 speak in bands.

    R9 introduces no backfill, no re-labelling and no mapping of legacy levels onto the new
    verdicts (R9-T1 invariant 4). The two vocabularies are disjoint sets of strings, and a
    consumer resolves which one applies through `rules_version` on the row.
    """
    legacy = {"HIGH", "MEDIUM", "UNKNOWN", "LOW"}
    r9 = {MANIPULATION_DETECTED, NO_SIGNAL, INCONCLUSIVE}

    for svd_state in SVD_STATES:
        for face_state in FACE_STATES:
            for lip_state in LIP_STATES:
                evidence = (
                    SVD_STATES[svd_state],
                    FACE_STATES[face_state],
                    LIP_STATES[lip_state],
                )

                assert evaluate(*evidence).risk_level not in r9
                assert evaluate_v5(*evidence).risk_level not in legacy


def test_the_two_engines_disagree_exactly_where_v5_was_written_to():
    """The behavioural difference between the rulesets, stated as the two cases that differ.

    v4 calls both of these MEDIUM and separates them only by a rule id in the trace. v5 separates
    them by verdict, so a renderer that shows the level and drops the rule id cannot present
    "we could barely read anything" as the same finding as "we read everything and found
    nothing".
    """
    complete = (svd(score=0.4646), face(score=0.4646), lip())
    barely_anything = (svd(score=0.4646), None, None)

    assert evaluate(*complete).risk_level == "MEDIUM"
    assert evaluate(*barely_anything).risk_level == "MEDIUM"

    assert evaluate_v5(*complete).risk_level == NO_SIGNAL
    assert evaluate_v5(*barely_anything).risk_level == INCONCLUSIVE
