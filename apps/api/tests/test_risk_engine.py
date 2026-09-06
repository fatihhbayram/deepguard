"""The risk engine, and the orchestration that stores what it decided.

Two halves, and the split is deliberate.

The first half runs the rules directly. They are pure functions over three frozen dataclasses
— no database, no clock, no configuration — so every branch, boundary and degenerate input
is reachable without staging anything, and the tests read as the rule table they check. Its
centrepiece is `DECISION_MATRIX`: the complete cross-product of what the three detectors can
each say — all 125 of them — written out one row at a time rather than derived, so a test can
never agree with a defect by recomputing it the same wrong way the engine did.

The second half runs `conclude_job` against real PostgreSQL, because what is being checked
there is not arithmetic but a property of the stored row: that the decision, the ruleset,
the calibration and the rule that fired all survive the write, and that no amount of
changing, removing or failing the *uncalibrated* evidence moves the classification by a
band. Those tests write the signal rows themselves rather than driving the detectors, so the
evidence under test is exact and everything around it can be varied freely — which is the
whole point of the isolation check.

**No threshold is mocked anywhere in this module.** All three are calibrated constants with a
measurement behind them — two from the 159-clip R4-T1 study, one from the 40-clip R5-T3 study —
not knobs; a test that patched one would be checking that comparison operators work rather than
that DeepGuard classifies media the way it was calibrated to.

**Two of the three decide.** `r7-v4.0.0` withdrew the mouth-dynamics score from the HIGH rules
after R7-T5 measured a 7.17% false HIGH rate on 307 independent genuine lineages, 21 of the 22
false positives coming from `R103` alone. The score is still read, still stored and still banded
for a reader; what it may no longer do is conclude. A third section below replays the measured
readings of the benign lineage that produced the production regression and asserts exactly that,
from a fixture of numbers — never from the private media, which is not in this repository and
which no test may need.
"""

import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path

import pytest
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from app import risk_engine, worker
from app.db.models import (
    Analysis,
    AnalysisJob,
    AnalysisSignal,
)
from app.db.session import SessionLocal, engine
from app.risk_engine import (
    FaceEvidence,
    LipEvidence,
    RiskDecision,
    SvdEvidence,
    evaluate,
)

# The three deployments the calibrations bind to, restated rather than imported. If a constant
# in `app.risk_engine` is ever edited, these tests must fail rather than agree with the edit:
# the binding is a fact about a measurement that was taken, not a value the code may choose.
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
EXPECTED_RULES_VERSION = "r7-v4.0.0"

# The two artifacts a v3 decision stands on, and the single identity it is stored under. The
# composite is restated as a literal *and* recomputed from its parts below: the construction is
# what makes the stored id traceable to both measurements, so a test that only compared literals
# would let the construction change without anything noticing.
EXPECTED_SVD_FACE_CALIBRATION_ID = (
    "cab2ea262bb7e41cb87e49bdb3dad53ecd0f02248035a993f9fcb033363afd1e"
)
EXPECTED_LIP_CALIBRATION_ID = (
    "85cb7484ab74d5821b1f4fa7ba917588dcaa98354f5f7a82c3080b624c9b8a29"
)
EXPECTED_CALIBRATION_ID = "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"

# The operating points R4-T1 and R5-T3 selected, restated for the same reason.
EXPECTED_SVD_T_HIGH = 0.9550971388816833
EXPECTED_FACE_T_HIGH = 0.9867589175701141
EXPECTED_LIP_T_HIGH = 0.22962537594139576

# NVIDIA's aggregate figures for a call that succeeded, as `detect_synthetic_video` writes
# them into the signal's metadata.
TOTAL_CLIPS = 7
LOGIT = 1.9142135381698608

# How many face crops the classifier's stored mean was taken over, as
# `detect_face_manipulation` writes it.
FRAMES_SCORED = 8

# How many 25-frame runs the mouth-dynamics model's stored mean was taken over, as
# `detect_lip_forensics` writes it. One of the four sampled runs lost the face, which is the
# ordinary shape of a real reading.
WINDOWS_SCORED = 3

# Two real clips from the calibration corpus, and the clearest evidence in the whole study
# that neither detector may be allowed to quieten the other. Both are genuine manipulations;
# in each case one detector is emphatic and the other is at its floor.
#
#   sonic_en_03 (talking-head synthesis) — NVIDIA 0.9961, face model 0.0053
#   ffpp_dev_Deepfakes_106_198 (face swap) — NVIDIA 0.1648, face model 0.9943
CORPUS_SYNTHETIC_SVD = 0.9961
CORPUS_SYNTHETIC_FACE = 0.0053
CORPUS_FACESWAP_SVD = 0.1648
CORPUS_FACESWAP_FACE = 0.9943

# Two real clips from the R5-T3 corpus, at the two ends of the gap the mouth-dynamics threshold
# was placed in. `real_695` is the highest any genuine clip scored; `Deepfakes_220_219` is the
# lowest any face swap scored — and the clip the benchmark's 0.5 reporting convention would have
# missed, which is why R5-T3 did not select it.
CORPUS_LIP_GENUINE_MAX = 0.016813907772302628
CORPUS_LIP_FACESWAP_MIN = 0.4424368441104889


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

    Defaults are a healthy, eligible signal sitting in the indeterminate band, so every test
    below states only the one thing it is about and nothing else can drift underneath it.
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

    Like the two factories above, the default is a healthy, eligible signal sitting in the
    indeterminate band. It is not 0.5, because 0.5 is above this detector's threshold: the three
    scales are unrelated, and a default shared across the three factories would have been quiet
    for two detectors and a flag for this one.
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
# The calibration the rules are bound to
# --------------------------------------------------------------------------------------


def test_the_ruleset_names_itself_and_the_calibration_behind_it():
    """The two identifiers every stored decision is only readable through."""
    assert risk_engine.RULES_VERSION == EXPECTED_RULES_VERSION
    assert risk_engine.CALIBRATION_ID == EXPECTED_CALIBRATION_ID


def test_the_ruleset_moved_off_every_earlier_calibration():
    """Each measurement has its own identity, and a stored row must be able to say which.

    Rows written under `p7-v1.0.0` were classified from NVIDIA's score alone against a
    threshold of 0.98; rows written under `r4-v2.0.0` were classified from two detectors under
    R4-T1 alone. Reusing either identifier here would make those rows and these
    indistinguishable.
    """
    assert risk_engine.RULES_VERSION not in {"p7-v1.0.0", "r4-v2.0.0", "r5-v3.0.0"}
    assert risk_engine.CALIBRATION_ID not in {
        "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c",
        EXPECTED_SVD_FACE_CALIBRATION_ID,
        EXPECTED_LIP_CALIBRATION_ID,
    }


def test_the_ruleset_kept_the_calibration_v3_was_decided_under():
    """v4 changed the rules and not one measurement, so the identity is v3's, deliberately.

    The two columns answer different questions. `risk_rules_version` says which sentences could
    fire; `risk_calibration_id` says which artifacts the thresholds came from. R7-T6 withdrew
    `R103` on the strength of R7-T5's *observation* of the existing operating points — it
    re-derived nothing, adopted no artifact and dropped none — so the artifacts behind a v4
    decision are exactly the artifacts behind a v3 one.

    Minting a fresh id here would assert a measurement nobody took, and would leave every v4
    decision unresolvable against the two studies it actually rests on. The version string is
    what separates the two, and it does (above).
    """
    assert risk_engine.CALIBRATION_ID == (
        "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"
    )

    # The thresholds themselves, unchanged. This is the substance of the claim: the identity is
    # honest only because the numbers it names are the ones still in force.
    assert risk_engine.SVD_T_HIGH == EXPECTED_SVD_T_HIGH
    assert risk_engine.FACE_T_HIGH == EXPECTED_FACE_T_HIGH
    assert risk_engine.LIP_T_HIGH == EXPECTED_LIP_T_HIGH


def test_the_stored_calibration_id_is_derived_from_both_artifacts():
    """The one column holds one digest, and the decision rests on two measurements.

    The construction is what makes the stored id traceable back to both: anyone holding the two
    artifact identities can recompute it and confirm which measurements a stored verdict was
    taken under. Recomputed here rather than only compared as a literal, so a changed
    construction fails rather than passing under a new constant.
    """
    assert risk_engine.SVD_FACE_CALIBRATION_ID == EXPECTED_SVD_FACE_CALIBRATION_ID
    assert risk_engine.LIP_CALIBRATION_ID == EXPECTED_LIP_CALIBRATION_ID

    derived = hashlib.sha256(
        f"{EXPECTED_SVD_FACE_CALIBRATION_ID}\n{EXPECTED_LIP_CALIBRATION_ID}".encode()
    ).hexdigest()

    assert risk_engine.CALIBRATION_ID == derived == EXPECTED_CALIBRATION_ID


def test_the_synthetic_video_binding_is_the_measured_one():
    assert risk_engine.SVD_T_HIGH == EXPECTED_SVD_T_HIGH
    assert risk_engine.SVD_PROVIDER == "nvidia"
    assert risk_engine.SVD_SIGNAL_TYPE == "synthetic_video"
    assert risk_engine.SVD_PROVIDER_VERSION == VALIDATED_FUNCTION_ID


def test_the_face_manipulation_binding_is_the_measured_one():
    assert risk_engine.FACE_T_HIGH == EXPECTED_FACE_T_HIGH
    assert risk_engine.FACE_PROVIDER == "efficientnet-b7"
    assert risk_engine.FACE_SIGNAL_TYPE == "face_manipulation"
    assert risk_engine.FACE_PROVIDER_VERSION == VALIDATED_FACE_CHECKPOINT


def test_the_mouth_dynamics_binding_is_the_measured_one():
    """R5-T3's operating point, and the composite identity it was measured against."""
    assert risk_engine.LIP_T_HIGH == EXPECTED_LIP_T_HIGH
    assert risk_engine.LIP_PROVIDER == "lipforensics"
    assert risk_engine.LIP_SIGNAL_TYPE == "lip_forensics"
    assert risk_engine.LIP_PROVIDER_VERSION == VALIDATED_LIP_MODEL


def test_the_mouth_dynamics_operating_point_survives_unedited():
    """R5-T3's number is kept exactly, and v4 decides nothing with it.

    Two properties at once, and they are the whole of what R7-T6 did to this constant.

    It is not 0.5, the R5-T1 harness default fixed before any score existed, which R5-T3
    rejected by measurement — on its corpus it sits above `Deepfakes_220_219` at 0.4424. And it
    is not moved, raised or re-derived: R7-T5 measured that this operating point cannot carry a
    HIGH on independent genuine media, which is a reason to withdraw the rule and not a
    measurement of a better point. Editing it here would be inventing one.

    What changed is the rule. The same face swap that was a HIGH under v3 is no longer one on
    this evidence alone — a real loss of coverage, taken deliberately, and the reason `R103`'s
    withdrawal is recorded as a cost rather than a fix.
    """
    assert risk_engine.LIP_T_HIGH == EXPECTED_LIP_T_HIGH != 0.5

    decision = evaluate(lip=lip(score=CORPUS_LIP_FACESWAP_MIN))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


def test_each_detector_is_banded_on_its_own_threshold():
    """The three operating points are different numbers, and none is used for another.

    A score of 0.96 is above NVIDIA's threshold, below the face model's and far above the
    mouth-dynamics model's. If one constant were serving more than one detector, one of these
    three assertions would fail.
    """
    assert risk_engine.SVD_T_HIGH != risk_engine.FACE_T_HIGH
    assert risk_engine.LIP_T_HIGH not in {risk_engine.SVD_T_HIGH, risk_engine.FACE_T_HIGH}

    assert evaluate(svd(score=0.96), None).risk_level == "HIGH"
    assert evaluate(None, face(score=0.96)).risk_level == "MEDIUM"
    # 0.96 is far above the mouth-dynamics threshold and reaches nothing: under v4 that
    # detector is read and does not decide.
    assert evaluate(lip=lip(score=0.96)).risk_level == "MEDIUM"


def test_a_low_score_on_one_scale_is_not_a_low_score_on_another():
    """0.25 is at the floor of the two R4-T1 detectors and above the mouth-dynamics threshold.

    The three scales are unrelated, and this is the test that fails the moment someone reads
    them as comparable — by sharing a threshold, by ranking the scores, or by treating the
    lowest threshold as the most easily convinced detector. The scales stay unrelated under v4;
    the difference is that crossing the third one now reaches no level at all, so the band is
    the same on both lines for two quite different reasons.
    """
    assert evaluate(svd(score=0.25), face(score=0.25)).risk_level == "MEDIUM"
    assert evaluate(lip=lip(score=0.25)).risk_level == "MEDIUM"


def test_t_low_is_recorded_for_every_detector_but_is_not_a_boundary():
    """The measured values are part of the calibration identity and decide nothing.

    Both sides of each `T_LOW` must classify identically, because there is no LOW band. The
    mouth-dynamics `T_LOW` coincides with its `T_HIGH` — R5-T3 found one clean gap and both
    selection rules landed in it — and under v4 neither of them is a boundary in this module at
    all, so a score at that value classifies exactly as a score below it.
    """
    assert risk_engine.SVD_T_LOW == 0.030267621390521526
    assert risk_engine.FACE_T_LOW == 0.0047953922767192125
    assert risk_engine.LIP_T_LOW == EXPECTED_LIP_T_HIGH

    for below, above in [
        (evaluate(svd(score=0.02), None), evaluate(svd(score=0.04), None)),
        (evaluate(None, face(score=0.004)), evaluate(None, face(score=0.006))),
    ]:
        assert below.risk_level == above.risk_level == "MEDIUM"
        assert below.rule_id == above.rule_id == "R201"

    at_the_point = evaluate(lip=lip(score=EXPECTED_LIP_T_HIGH))
    under_it = evaluate(lip=lip(score=0.0154))

    assert at_the_point.risk_level == under_it.risk_level == "MEDIUM"
    assert at_the_point.rule_id == under_it.rule_id == "R201"


# --------------------------------------------------------------------------------------
# The decision matrix — every combination of what the three detectors can say
# --------------------------------------------------------------------------------------

# One evidence value per state a detector can be in, per detector.
#
#   flagged    — calibrated, readable, at or above its own threshold
#   below      — calibrated, readable, under its own threshold
#   invalid    — the calibrated deployment answered, and its figures cannot be read
#   ineligible — something answered, but not the deployment the thresholds were measured on
#   absent     — no row at all
SVD_STATES = {
    "flagged": svd(score=0.9931),
    "below": svd(score=0.4646),
    "invalid": svd(score=None),
    "ineligible": svd(score=0.9931, provider_version="some-other-function"),
    "absent": None,
}

FACE_STATES = {
    "flagged": face(score=0.9931),
    "below": face(score=0.4646),
    "invalid": face(score=None),
    "ineligible": face(score=0.9931, provider_version="some-other-checkpoint"),
    "absent": None,
}

# The flagged and below values here are not the ones above, and cannot be: 0.4646 is *above*
# this detector's threshold. Its scale is its own, which is the whole reason each detector
# carries its own operating point.
LIP_STATES = {
    "flagged": lip(score=0.9931),
    "invalid": lip(score=None),
    "below": lip(score=0.0154),
    "ineligible": lip(score=0.9931, provider_version="some-other-checkpoint"),
    "absent": None,
}

# What this ruleset must conclude for each of the 125 combinations, written out rather than
# computed, and grouped by the rule that has to fire. Reading a group is the fastest way to see
# the properties that matter most:
#
#   * every combination with a `flagged` in the first two positions is HIGH. Nothing the other
#     detectors say — quiet, broken, uncalibrated or missing — softens a finding.
#   * exactly one of those two flags is attributed to the detector that produced it; both are
#     `R102`, which credits neither with the other's evidence.
#   * a `flagged` in the third position reaches no level of its own and raises none. `R103` is
#     absent from this table because it is absent from the ruleset, and the 16 combinations it
#     used to own are MEDIUM here — one `R200`, fifteen `R201`.
#   * `R201` fills the band where some but not all detectors could be read. That is a MEDIUM
#     with less coverage than an `R200`, and the trace says so rather than hiding it.
#
# The 16 rows that moved out of `R103` and the 8 that moved off `R102` are the entire behavioural
# difference between v3 and v4. Everything else in this table is character-for-character what v3
# concluded.
DECISION_MATRIX = {
    # HIGH by R102 — both detectors this ruleset decides from reached their own thresholds,
    # independently. The rule does not name which; the signal rows do. A mouth-dynamics crossing
    # is not one of the two and never contributes to this count: `R103` re-entering through the
    # rule that reads two flags at once is exactly what R7-T6 had to prevent. (5 of 125)
    ("flagged", "flagged", "flagged"): ("HIGH", "R102"),
    ("flagged", "flagged", "below"): ("HIGH", "R102"),
    ("flagged", "flagged", "invalid"): ("HIGH", "R102"),
    ("flagged", "flagged", "ineligible"): ("HIGH", "R102"),
    ("flagged", "flagged", "absent"): ("HIGH", "R102"),
    # HIGH by R100 — NVIDIA alone among the deciding detectors. Nothing the other two say —
    # quiet, broken, missing, or a mouth-dynamics score at the top of its scale — softens it or
    # adds to it. The four rows ending `flagged` were `R102` under v3 and are `R100` here: the
    # level is taken from NVIDIA's finding, and a detector that may not decide alone may not be
    # credited with confirming one either. (20 of 125)
    ("flagged", "below", "flagged"): ("HIGH", "R100"),
    ("flagged", "below", "below"): ("HIGH", "R100"),
    ("flagged", "below", "invalid"): ("HIGH", "R100"),
    ("flagged", "below", "ineligible"): ("HIGH", "R100"),
    ("flagged", "below", "absent"): ("HIGH", "R100"),
    ("flagged", "invalid", "flagged"): ("HIGH", "R100"),
    ("flagged", "invalid", "below"): ("HIGH", "R100"),
    ("flagged", "invalid", "invalid"): ("HIGH", "R100"),
    ("flagged", "invalid", "ineligible"): ("HIGH", "R100"),
    ("flagged", "invalid", "absent"): ("HIGH", "R100"),
    ("flagged", "ineligible", "flagged"): ("HIGH", "R100"),
    ("flagged", "ineligible", "below"): ("HIGH", "R100"),
    ("flagged", "ineligible", "invalid"): ("HIGH", "R100"),
    ("flagged", "ineligible", "ineligible"): ("HIGH", "R100"),
    ("flagged", "ineligible", "absent"): ("HIGH", "R100"),
    ("flagged", "absent", "flagged"): ("HIGH", "R100"),
    ("flagged", "absent", "below"): ("HIGH", "R100"),
    ("flagged", "absent", "invalid"): ("HIGH", "R100"),
    ("flagged", "absent", "ineligible"): ("HIGH", "R100"),
    ("flagged", "absent", "absent"): ("HIGH", "R100"),
    # HIGH by R101 — the face classifier alone among the deciding detectors, on the same terms
    # and with the same four rows moved off `R102`. (20 of 125)
    ("below", "flagged", "flagged"): ("HIGH", "R101"),
    ("below", "flagged", "below"): ("HIGH", "R101"),
    ("below", "flagged", "invalid"): ("HIGH", "R101"),
    ("below", "flagged", "ineligible"): ("HIGH", "R101"),
    ("below", "flagged", "absent"): ("HIGH", "R101"),
    ("invalid", "flagged", "flagged"): ("HIGH", "R101"),
    ("invalid", "flagged", "below"): ("HIGH", "R101"),
    ("invalid", "flagged", "invalid"): ("HIGH", "R101"),
    ("invalid", "flagged", "ineligible"): ("HIGH", "R101"),
    ("invalid", "flagged", "absent"): ("HIGH", "R101"),
    ("ineligible", "flagged", "flagged"): ("HIGH", "R101"),
    ("ineligible", "flagged", "below"): ("HIGH", "R101"),
    ("ineligible", "flagged", "invalid"): ("HIGH", "R101"),
    ("ineligible", "flagged", "ineligible"): ("HIGH", "R101"),
    ("ineligible", "flagged", "absent"): ("HIGH", "R101"),
    ("absent", "flagged", "flagged"): ("HIGH", "R101"),
    ("absent", "flagged", "below"): ("HIGH", "R101"),
    ("absent", "flagged", "invalid"): ("HIGH", "R101"),
    ("absent", "flagged", "ineligible"): ("HIGH", "R101"),
    ("absent", "flagged", "absent"): ("HIGH", "R101"),
    # MEDIUM by R200 — all three detectors read this media and neither deciding detector reached
    # its threshold. The second row is the shape of the R7-T5 regression: the mouth-dynamics
    # score is above its own threshold, the reading is reported as the crossing it is, and the
    # level is MEDIUM because no detector entitled to conclude did. Under v3 that row was a HIGH
    # by `R103`. (2 of 125)
    ("below", "below", "flagged"): ("MEDIUM", "R200"),
    ("below", "below", "below"): ("MEDIUM", "R200"),
    # MEDIUM by R201 — one or two detectors read it, and neither deciding detector reached its
    # threshold. Less coverage than an `R200`, and the trace says so rather than hiding it. A
    # mouth-dynamics reading counts toward that coverage whether or not it crossed its threshold,
    # which is why the rows ending `flagged` are here rather than in a HIGH group. (51 of 125)
    ("below", "below", "invalid"): ("MEDIUM", "R201"),
    ("below", "below", "ineligible"): ("MEDIUM", "R201"),
    ("below", "below", "absent"): ("MEDIUM", "R201"),
    ("below", "invalid", "flagged"): ("MEDIUM", "R201"),
    ("below", "invalid", "below"): ("MEDIUM", "R201"),
    ("below", "invalid", "invalid"): ("MEDIUM", "R201"),
    ("below", "invalid", "ineligible"): ("MEDIUM", "R201"),
    ("below", "invalid", "absent"): ("MEDIUM", "R201"),
    ("below", "ineligible", "flagged"): ("MEDIUM", "R201"),
    ("below", "ineligible", "below"): ("MEDIUM", "R201"),
    ("below", "ineligible", "invalid"): ("MEDIUM", "R201"),
    ("below", "ineligible", "ineligible"): ("MEDIUM", "R201"),
    ("below", "ineligible", "absent"): ("MEDIUM", "R201"),
    ("below", "absent", "flagged"): ("MEDIUM", "R201"),
    ("below", "absent", "below"): ("MEDIUM", "R201"),
    ("below", "absent", "invalid"): ("MEDIUM", "R201"),
    ("below", "absent", "ineligible"): ("MEDIUM", "R201"),
    ("below", "absent", "absent"): ("MEDIUM", "R201"),
    ("invalid", "below", "flagged"): ("MEDIUM", "R201"),
    ("invalid", "below", "below"): ("MEDIUM", "R201"),
    ("invalid", "below", "invalid"): ("MEDIUM", "R201"),
    ("invalid", "below", "ineligible"): ("MEDIUM", "R201"),
    ("invalid", "below", "absent"): ("MEDIUM", "R201"),
    ("invalid", "invalid", "flagged"): ("MEDIUM", "R201"),
    ("invalid", "invalid", "below"): ("MEDIUM", "R201"),
    ("invalid", "ineligible", "flagged"): ("MEDIUM", "R201"),
    ("invalid", "ineligible", "below"): ("MEDIUM", "R201"),
    ("invalid", "absent", "flagged"): ("MEDIUM", "R201"),
    ("invalid", "absent", "below"): ("MEDIUM", "R201"),
    ("ineligible", "below", "flagged"): ("MEDIUM", "R201"),
    ("ineligible", "below", "below"): ("MEDIUM", "R201"),
    ("ineligible", "below", "invalid"): ("MEDIUM", "R201"),
    ("ineligible", "below", "ineligible"): ("MEDIUM", "R201"),
    ("ineligible", "below", "absent"): ("MEDIUM", "R201"),
    ("ineligible", "invalid", "flagged"): ("MEDIUM", "R201"),
    ("ineligible", "invalid", "below"): ("MEDIUM", "R201"),
    ("ineligible", "ineligible", "flagged"): ("MEDIUM", "R201"),
    ("ineligible", "ineligible", "below"): ("MEDIUM", "R201"),
    ("ineligible", "absent", "flagged"): ("MEDIUM", "R201"),
    ("ineligible", "absent", "below"): ("MEDIUM", "R201"),
    ("absent", "below", "flagged"): ("MEDIUM", "R201"),
    ("absent", "below", "below"): ("MEDIUM", "R201"),
    ("absent", "below", "invalid"): ("MEDIUM", "R201"),
    ("absent", "below", "ineligible"): ("MEDIUM", "R201"),
    ("absent", "below", "absent"): ("MEDIUM", "R201"),
    ("absent", "invalid", "flagged"): ("MEDIUM", "R201"),
    ("absent", "invalid", "below"): ("MEDIUM", "R201"),
    ("absent", "ineligible", "flagged"): ("MEDIUM", "R201"),
    ("absent", "ineligible", "below"): ("MEDIUM", "R201"),
    ("absent", "absent", "flagged"): ("MEDIUM", "R201"),
    ("absent", "absent", "below"): ("MEDIUM", "R201"),
    # UNKNOWN by R012 — no readable evidence, but a calibrated detector did answer unreadably. (19 of 125)
    ("invalid", "invalid", "invalid"): ("UNKNOWN", "R012"),
    ("invalid", "invalid", "ineligible"): ("UNKNOWN", "R012"),
    ("invalid", "invalid", "absent"): ("UNKNOWN", "R012"),
    ("invalid", "ineligible", "invalid"): ("UNKNOWN", "R012"),
    ("invalid", "ineligible", "ineligible"): ("UNKNOWN", "R012"),
    ("invalid", "ineligible", "absent"): ("UNKNOWN", "R012"),
    ("invalid", "absent", "invalid"): ("UNKNOWN", "R012"),
    ("invalid", "absent", "ineligible"): ("UNKNOWN", "R012"),
    ("invalid", "absent", "absent"): ("UNKNOWN", "R012"),
    ("ineligible", "invalid", "invalid"): ("UNKNOWN", "R012"),
    ("ineligible", "invalid", "ineligible"): ("UNKNOWN", "R012"),
    ("ineligible", "invalid", "absent"): ("UNKNOWN", "R012"),
    ("ineligible", "ineligible", "invalid"): ("UNKNOWN", "R012"),
    ("ineligible", "absent", "invalid"): ("UNKNOWN", "R012"),
    ("absent", "invalid", "invalid"): ("UNKNOWN", "R012"),
    ("absent", "invalid", "ineligible"): ("UNKNOWN", "R012"),
    ("absent", "invalid", "absent"): ("UNKNOWN", "R012"),
    ("absent", "ineligible", "invalid"): ("UNKNOWN", "R012"),
    ("absent", "absent", "invalid"): ("UNKNOWN", "R012"),
    # UNKNOWN by R010 — no validated reading was ever obtained. (8 of 125)
    ("ineligible", "ineligible", "ineligible"): ("UNKNOWN", "R010"),
    ("ineligible", "ineligible", "absent"): ("UNKNOWN", "R010"),
    ("ineligible", "absent", "ineligible"): ("UNKNOWN", "R010"),
    ("ineligible", "absent", "absent"): ("UNKNOWN", "R010"),
    ("absent", "ineligible", "ineligible"): ("UNKNOWN", "R010"),
    ("absent", "ineligible", "absent"): ("UNKNOWN", "R010"),
    ("absent", "absent", "ineligible"): ("UNKNOWN", "R010"),
    ("absent", "absent", "absent"): ("UNKNOWN", "R010"),
}


def test_the_matrix_covers_every_combination():
    """The table is exhaustive over all three detectors' states, not a sample of them."""
    assert set(DECISION_MATRIX) == {
        (svd_state, face_state, lip_state)
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }
    assert len(DECISION_MATRIX) == 125


@pytest.mark.parametrize(
    ("svd_state", "face_state", "lip_state", "risk_level", "rule_id"),
    [
        (svd_state, face_state, lip_state, risk_level, rule_id)
        for (svd_state, face_state, lip_state), (
            risk_level,
            rule_id,
        ) in DECISION_MATRIX.items()
    ],
)
def test_the_decision_matrix(svd_state, face_state, lip_state, risk_level, rule_id):
    decision = evaluate(
        SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
    )

    assert decision.risk_level == risk_level
    assert decision.rule_id == rule_id


# --------------------------------------------------------------------------------------
# Disagreement — the case R4-T1 found to be the normal one
# --------------------------------------------------------------------------------------


def test_a_quiet_face_model_cannot_hold_back_a_synthetic_video_finding():
    """`sonic_en_03`, as R4-T1 actually measured it.

    NVIDIA 0.9961, face model 0.0053. The face model is worse than chance on generated video
    (AUROC 0.3960 against genuine), so its silence here is a property of the detector and not
    a statement about the media. HIGH by `R100`, on NVIDIA's evidence alone.
    """
    decision = evaluate(
        svd(score=CORPUS_SYNTHETIC_SVD), face(score=CORPUS_SYNTHETIC_FACE)
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R100"


def test_a_quiet_synthetic_video_detector_cannot_hold_back_a_face_finding():
    """`ffpp_dev_Deepfakes_106_198`, as R4-T1 actually measured it.

    NVIDIA 0.1648, face model 0.9943. NVIDIA's detector is near-blind to face swaps (AUROC
    0.5422) and flagged 0 of the 50 in the corpus. HIGH by `R101`, on the classifier's
    evidence alone — this is the entire detection capability v1 did not have.
    """
    decision = evaluate(
        svd(score=CORPUS_FACESWAP_SVD), face(score=CORPUS_FACESWAP_FACE)
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R101"


def test_requiring_the_detectors_to_agree_would_detect_nothing():
    """The measured justification for OR, stated as a test.

    R4-T1 scored 151 clips with both detectors and recorded `both_flagged: 0`. Every one of
    the 47 detections came from exactly one detector. An AND rule would therefore have
    classified both real manipulations below as not-HIGH; both must be HIGH here.
    """
    synthetic = evaluate(
        svd(score=CORPUS_SYNTHETIC_SVD), face(score=CORPUS_SYNTHETIC_FACE)
    )
    faceswap = evaluate(svd(score=CORPUS_FACESWAP_SVD), face(score=CORPUS_FACESWAP_FACE))

    assert synthetic.risk_level == faceswap.risk_level == "HIGH"
    # And by different rules, because different evidence decided them.
    assert synthetic.rule_id != faceswap.rule_id


@pytest.mark.parametrize("other_face", [0.0, 0.0053, 0.4646, 0.9, EXPECTED_FACE_T_HIGH - 1e-9])
@pytest.mark.parametrize("other_lip", [0.0, 0.0154, 0.1, EXPECTED_LIP_T_HIGH - 1e-9, None])
def test_no_other_score_below_its_threshold_changes_a_synthetic_video_high(
    other_face, other_lip
):
    """Swept along the whole sub-threshold range of both other detectors, together.

    A HIGH from one detector is invariant under everything the others can say short of their
    own thresholds — including saying nothing at all, which `None` covers here. This is the
    test that fails the moment someone introduces a weighted average, a veto, a confidence
    discount or a tie-break.
    """
    decision = evaluate(
        svd(score=0.9931),
        face(score=other_face),
        None if other_lip is None else lip(score=other_lip),
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R100"


@pytest.mark.parametrize("other_svd", [0.0, 0.1648, 0.4646, 0.9, EXPECTED_SVD_T_HIGH - 1e-9])
@pytest.mark.parametrize("other_lip", [0.0, 0.0154, 0.1, EXPECTED_LIP_T_HIGH - 1e-9, None])
def test_no_other_score_below_its_threshold_changes_a_face_high(other_svd, other_lip):
    decision = evaluate(
        svd(score=other_svd),
        face(score=0.9931),
        None if other_lip is None else lip(score=other_lip),
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R101"


@pytest.mark.parametrize("other_svd", [0.0, 0.1648, 0.4646, 0.9, EXPECTED_SVD_T_HIGH - 1e-9])
@pytest.mark.parametrize("other_face", [0.0, 0.0053, 0.4646, 0.9, EXPECTED_FACE_T_HIGH - 1e-9])
def test_a_mouth_dynamics_crossing_reaches_no_level_whatever_the_others_say(
    other_svd, other_face
):
    """The R7-T6 withdrawal, swept across the deciding detectors' quiet range.

    A mouth-dynamics score of 0.9931 — four times its own threshold — against every quiet value
    the other two take, including each one a hair under its own operating point. Not one of
    these is a HIGH. Under `r5-v3.0.0` every one of them was, by `R103`, and R7-T5 measured what
    that cost on genuine media: 21 false HIGHs out of 22, on 307 independent genuine lineages.

    MEDIUM by `R200` rather than `R201` throughout, because all three detectors did read this
    media. The crossing is not discarded and not disbelieved; it is read, recorded and not
    permitted to conclude.
    """
    decision = evaluate(
        svd(score=other_svd), face(score=other_face), lip(score=0.9931)
    )

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R200"


@pytest.mark.parametrize("flagged_lip", [True, False])
def test_both_deciding_detectors_flagging_is_recorded_as_its_own_rule(flagged_lip):
    """R4-T1 never observed agreement on 159 clips. It is still a state the rules must name.

    Attributing a joint finding to `R100` or `R101` would credit one detector with evidence the
    other independently produced. `R102` deliberately does not say *which* two agreed: that is
    in the signal rows, and an id per subset would be a rule table nobody can hold in mind for
    no decision it does not already make.

    The mouth-dynamics score is varied across the whole width of its scale and changes nothing,
    which is the point of running it both ways: `R102` is about the two detectors this ruleset
    decides from, and a third reading neither completes it nor dilutes it.
    """
    decision = evaluate(
        svd(score=0.9931),
        face(score=0.9931),
        lip(score=0.9931 if flagged_lip else 0.0154),
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R102"


@pytest.mark.parametrize(
    ("flagged_svd", "expected_rule"), [(True, "R100"), (False, "R101")]
)
def test_a_mouth_dynamics_crossing_never_makes_a_single_finding_a_corroborated_one(
    flagged_svd, expected_rule
):
    """One deciding detector flagged and the mouth-dynamics score above its own threshold.

    Under `r5-v3.0.0` this was `R102` — two detectors, independently, and the trace said so.
    Under v4 it is the single rule of the detector that actually decided. The level is identical
    either way; what would not be identical is the record, and a trace claiming a corroborated
    finding on evidence that may not decide is `R103` re-entering by the back door.
    """
    decision = evaluate(
        svd(score=0.9931 if flagged_svd else 0.4646),
        face(score=0.4646 if flagged_svd else 0.9931),
        lip(score=0.9931),
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == expected_rule


def test_a_medium_says_whether_every_detector_or_only_some_were_behind_it():
    """`R200` and `R201` are the same band with materially different coverage.

    An `R200` means all three questions were asked of this media and none was answered above
    its threshold. An `R201` means at least one was never asked at all. A reader who cannot tell
    them apart would read the same reassurance into both.
    """
    every = evaluate(svd(score=0.4646), face(score=0.4646), lip(score=0.0154))
    two = evaluate(svd(score=0.4646), face(score=0.4646), None)
    only_svd = evaluate(svd(score=0.4646), None, None)
    only_face = evaluate(None, face(score=0.4646), None)
    only_lip = evaluate(None, None, lip(score=0.0154))

    assert (
        every.risk_level
        == two.risk_level
        == only_svd.risk_level
        == only_face.risk_level
        == only_lip.risk_level
        == "MEDIUM"
    )
    assert every.rule_id == "R200"
    assert two.rule_id == only_svd.rule_id == only_face.rule_id == only_lip.rule_id == "R201"


def test_an_abstaining_face_detector_is_not_a_finding_of_no_manipulation():
    """A clip with no face in it fails, and a `FAILED` row is silence, not evidence.

    R4-T1 hit this on 8 clips, every one of them generated video. Treating the abstention as
    a clean reading would have turned 8 unscored clips into 8 assertions.
    """
    decision = evaluate(
        svd(score=0.4646), face(score=None, status="FAILED"), lip(score=0.0154)
    )

    assert decision.risk_level == "MEDIUM"
    # R201, not R200: only two detectors actually read this media.
    assert decision.rule_id == "R201"


def test_an_abstaining_mouth_dynamics_model_cannot_hold_back_a_face_finding():
    """This model needs a face tracked through 25 consecutive frames and often has none.

    An abstention is the ordinary case on media the other two score without difficulty, and it
    is silence. A face score above its own threshold is a HIGH beside it, by `R101`.
    """
    decision = evaluate(
        svd(score=CORPUS_FACESWAP_SVD),
        face(score=CORPUS_FACESWAP_FACE),
        lip(score=None, status="FAILED"),
    )

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R101"


def test_the_mouth_dynamics_model_alone_decides_nothing_and_is_still_read():
    """The capability this ruleset withdrew, and the one it kept, in one analysis.

    Both R4-T1 detectors silent and the mouth-dynamics score at four times its own threshold.
    Under `r4-v2.0.0` this was `R010`; under `r5-v3.0.0` R5-T3's operating point made it a HIGH
    by `R103`; under v4 it is `R201` — MEDIUM, on partial coverage.

    `R201` and not `R010` is the half that matters as much as the level. The detector answered,
    its figures were readable, and the rules read them. What they did not do is conclude from
    them. A ruleset that had dropped the detector instead of the rule would report this analysis
    as having no evidence at all, which is a different and false statement.
    """
    decision = evaluate(
        svd(score=0.999, status="FAILED"),
        face(score=None, status="FAILED"),
        lip(score=0.9931),
    )

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


# --------------------------------------------------------------------------------------
# Boundaries — each detector against its own threshold
# --------------------------------------------------------------------------------------


def test_a_synthetic_video_score_exactly_at_its_threshold_is_high():
    """`>=`, not `>`. The measured operating point is inside the HIGH band."""
    decision = evaluate(svd(score=EXPECTED_SVD_T_HIGH), None)

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R100"


def test_a_face_score_exactly_at_its_threshold_is_high():
    decision = evaluate(None, face(score=EXPECTED_FACE_T_HIGH))

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R101"


def test_a_mouth_dynamics_score_exactly_at_its_threshold_is_not_high():
    """No boundary here at all any more: `LIP_T_HIGH` is compared against by nothing."""
    decision = evaluate(lip=lip(score=EXPECTED_LIP_T_HIGH))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


@pytest.mark.parametrize("score", [0.9551, 0.9704, 0.98, 0.9986, 1.0])
def test_a_synthetic_video_score_above_its_threshold_is_high(score):
    decision = evaluate(svd(score=score), None)

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R100"


@pytest.mark.parametrize("score", [0.9868, 0.9899, 0.9943, 1.0])
def test_a_face_score_above_its_threshold_is_high(score):
    decision = evaluate(None, face(score=score))

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R101"


@pytest.mark.parametrize("score", [0.9550971, 0.9541, 0.5123, 0.0248, 0.0])
def test_a_synthetic_video_score_below_its_threshold_is_medium(score):
    """Including 0.9541 — the highest score any genuine clip in the corpus reached."""
    decision = evaluate(svd(score=score), None)

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


@pytest.mark.parametrize("score", [0.9867589, 0.9865, 0.7949, 0.0043, 0.0])
def test_a_face_score_below_its_threshold_is_medium(score):
    """Including 0.9865 — the highest score any genuine clip reached on this detector."""
    decision = evaluate(None, face(score=score))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


@pytest.mark.parametrize("score", [0.2297, 0.4424, 0.5, 0.9612, 1.0])
def test_a_mouth_dynamics_score_above_its_threshold_is_not_high(score):
    """The whole span above R5-T3's operating point, up to the ceiling of the scale.

    Including 0.4424, the lowest score any face swap in that corpus reached — a real
    manipulation this ruleset no longer flags on this evidence alone. That is the coverage
    R7-T6 gave up, and it is asserted here rather than left implicit: the trade was made
    against a measured 7.17% false HIGH rate on genuine media, and pretending it was free would
    misdescribe it.
    """
    decision = evaluate(lip=lip(score=score))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


@pytest.mark.parametrize("score", [0.2296253, 0.0168, 0.01, 0.0001, 0.0])
def test_a_mouth_dynamics_score_below_its_threshold_is_medium(score):
    """Including 0.0168 — the highest score any genuine clip reached on this detector.

    Identical to the band above it under v4, and deliberately kept as its own test: the two
    sides of that threshold must classify the same, and a build where they diverge has put the
    comparison back.
    """
    decision = evaluate(lip=lip(score=score))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


def test_the_highest_genuine_score_r5_t3_observed_is_not_high():
    """`real_695`, as R5-T3 actually measured it: 0.0168, and 0.2128 clear of the threshold.

    R7-T5 then measured genuine media that lay outside that corpus and found the margin does
    not survive it, which is why the clip below and one four times over the threshold now
    classify alike.
    """
    decision = evaluate(lip=lip(score=CORPUS_LIP_GENUINE_MAX))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R201"


def test_the_bottom_of_each_scale_is_medium_not_unknown():
    """0.0 is a real reading at the floor of a sigmoid, not missing evidence.

    Classifying it UNKNOWN would confuse "the detector said the lowest thing it can say" with
    "the detector said nothing".
    """
    floors = evaluate(svd(score=0.0), face(score=0.0), lip(score=0.0))

    assert floors.risk_level == "MEDIUM"
    assert floors.rule_id == "R200"


# --------------------------------------------------------------------------------------
# Eligibility — R010, per detector
# --------------------------------------------------------------------------------------


def test_no_signals_at_all_is_unknown():
    decision = evaluate(None, None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


def test_evaluate_defaults_every_signal_to_absent():
    """Called with nothing, the engine classifies the absence rather than raising."""
    assert evaluate() == evaluate(None, None, None)


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT", "UNAVAILABLE", None])
def test_no_detector_counts_unless_it_succeeded(status):
    """A detector that refused, timed out or was unavailable produced no reading.

    Every score is set high on purpose: a stale or placeholder figure sitting beside a
    non-SUCCESS status must not be able to reach HIGH from any source.
    """
    decision = evaluate(
        svd(score=0.999, status=status),
        face(score=0.999, status=status),
        lip(score=0.999, status=status),
    )

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "provider_version",
    [
        None,
        "",
        "f286f937-05c4-454b-8312-fba67a2a6fa7",
        f"{VALIDATED_FUNCTION_ID}-preview",
        f"{VALIDATED_FUNCTION_ID} ",
        f" {VALIDATED_FUNCTION_ID}",
        f"nvcf:{VALIDATED_FUNCTION_ID}",
        f"prefix-{VALIDATED_FUNCTION_ID}-suffix",
        VALIDATED_FUNCTION_ID.upper(),
        VALIDATED_FUNCTION_ID[:-1],
    ],
)
def test_a_synthetic_video_deployment_that_is_not_the_calibrated_one_is_unknown(
    provider_version,
):
    """Exact equality, never substring, prefix, suffix or case-insensitive matching.

    Every value here contains or nearly is the validated function id and none of them names
    the deployment that was measured. The operating point clears the highest genuine score
    observed by 0.0010, so a different deployment answering behind a similar-looking
    identifier is precisely what `R010` exists to refuse.
    """
    decision = evaluate(svd(score=0.999, provider_version=provider_version), None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "provider_version",
    [
        None,
        "",
        "tomas-gajarsky/facetorch-deepfake-efficientnet-b7",
        "tomas-gajarsky/facetorch-deepfake-efficientnet-b7@main",
        f"{VALIDATED_FACE_CHECKPOINT}-preview",
        f" {VALIDATED_FACE_CHECKPOINT}",
        VALIDATED_FACE_CHECKPOINT.upper(),
        VALIDATED_FACE_CHECKPOINT[:-1],
    ],
)
def test_a_face_artifact_that_is_not_the_calibrated_one_is_unknown(provider_version):
    """This threshold clears the highest genuine score by 0.0003.

    A different revision of the weights is not a small difference in a measurement — it is a
    different measurement, with no operating point of its own.
    """
    decision = evaluate(None, face(score=0.999, provider_version=provider_version))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "provider_version",
    [
        None,
        "",
        "https://github.com/ahaliassos/LipForensics",
        "https://github.com/ahaliassos/LipForensics@d0bf5553bfb9676f1771d590472b26a3a76de894",
        "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d",
        f"{VALIDATED_LIP_MODEL}-preview",
        f" {VALIDATED_LIP_MODEL}",
        VALIDATED_LIP_MODEL.upper(),
        VALIDATED_LIP_MODEL[:-1],
        VALIDATED_LIP_MODEL.replace("+", "@"),
    ],
)
def test_a_mouth_dynamics_model_that_is_not_the_calibrated_one_is_unknown(provider_version):
    """The identity is a composite and is compared whole.

    The architecture revision and the weights digest each fix half of what produced a score,
    and the third and fourth entries here name exactly one half. The same checkpoint loaded
    into a different network is a different model with no operating point of its own.
    """
    decision = evaluate(lip=lip(score=0.999, provider_version=provider_version))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize("provider", ["aasist", "c2pa", "NVIDIA", "efficientnet-b7", "", None])
def test_a_synthetic_video_signal_from_another_provider_is_unknown(provider):
    decision = evaluate(svd(score=0.999, provider=provider), None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize("provider", ["aasist", "c2pa", "nvidia", "EfficientNet-B7", "", None])
def test_a_face_signal_from_another_provider_is_unknown(provider):
    decision = evaluate(None, face(score=0.999, provider=provider))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "signal_type", ["active_speaker", "audio_authenticity", "provenance", "face_manipulation", None]
)
def test_a_signal_answering_another_question_is_not_the_synthetic_video_one(signal_type):
    """NVIDIA answers two questions, and only one of them was calibrated here."""
    decision = evaluate(svd(score=0.999, signal_type=signal_type), None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "signal_type", ["active_speaker", "audio_authenticity", "provenance", "synthetic_video", None]
)
def test_a_signal_answering_another_question_is_not_the_face_one(signal_type):
    decision = evaluate(None, face(score=0.999, signal_type=signal_type))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize("provider", ["aasist", "c2pa", "nvidia", "efficientnet-b7", "", None])
def test_a_mouth_dynamics_signal_from_another_provider_is_unknown(provider):
    decision = evaluate(lip=lip(score=0.999, provider=provider))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "signal_type",
    ["active_speaker", "audio_authenticity", "provenance", "face_manipulation", "lip_sync", None],
)
def test_a_signal_answering_another_question_is_not_the_mouth_dynamics_one(signal_type):
    """`lip_sync` is in the list deliberately. This model is given no audio and reports nothing
    about whether a sound track matches a mouth; a row typed that way is not this signal."""
    decision = evaluate(lip=lip(score=0.999, signal_type=signal_type))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


def test_the_three_signals_cannot_be_supplied_in_each_others_place():
    """A face row handed in as the synthetic-video argument is not eligible, and so on around.

    The engine identifies each source by provider and signal type rather than by which
    parameter it arrived in, so crossing the wires produces UNKNOWN rather than a
    classification made against the wrong threshold.
    """
    crossed_svd = svd(
        score=0.999, provider="efficientnet-b7", signal_type="face_manipulation"
    )
    crossed_face = face(score=0.999, provider="lipforensics", signal_type="lip_forensics")
    crossed_lip = lip(score=0.999, provider="nvidia", signal_type="synthetic_video")

    decision = evaluate(crossed_svd, crossed_face, crossed_lip)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


# --------------------------------------------------------------------------------------
# Usable figures — R012, per detector
# --------------------------------------------------------------------------------------


def test_a_null_score_on_every_eligible_signal_is_unknown():
    """Eligible, so `R010` passes; then there is no number to compare, from any of them."""
    decision = evaluate(svd(score=None), face(score=None), lip(score=None))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("score", [-0.0001, -1.0, -1e-12, 1.0000001, 2.0, 1e6])
def test_a_score_outside_the_scale_is_unknown(score):
    """Every score comes off a sigmoid and cannot leave [0, 1]."""
    assert evaluate(svd(score=score), None).rule_id == "R012"
    assert evaluate(None, face(score=score)).rule_id == "R012"
    assert evaluate(lip=lip(score=score)).rule_id == "R012"


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_a_non_finite_score_is_unknown(score):
    """NaN compares false against every threshold, so it must be rejected explicitly.

    Left to fall through, the band would be decided by which comparison happened to be
    written first rather than by the evidence.
    """
    assert evaluate(svd(score=score), None).rule_id == "R012"
    assert evaluate(None, face(score=score)).rule_id == "R012"
    assert evaluate(lip=lip(score=score)).rule_id == "R012"


@pytest.mark.parametrize("score", ["0.99", None, [], True, {"p": 1}])
def test_a_score_that_is_not_a_number_is_unknown(score):
    """`True` is in the list because Python would let it pass `>= 0.98` while measuring
    nothing."""
    assert evaluate(svd(score=score), None).rule_id == "R012"
    assert evaluate(None, face(score=score)).rule_id == "R012"
    assert evaluate(lip=lip(score=score)).rule_id == "R012"


@pytest.mark.parametrize("total_clips", [0, -1, None])
def test_synthetic_video_evidence_aggregated_over_no_clips_is_unknown(total_clips):
    """The aggregate is a figure over an empty table, not a reading of the video.

    The score is set high so that a degenerate clip count cannot be overridden by it.
    """
    decision = evaluate(svd(score=0.999, total_clips=total_clips), None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("total_clips", ["7", 7.0, True, [], {"n": 7}])
def test_a_clip_count_that_is_not_a_whole_number_is_unknown(total_clips):
    """It comes from a JSON document, so no column guarantees its type."""
    decision = evaluate(svd(score=0.999, total_clips=total_clips), None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("frames_scored", [0, -1, None])
def test_face_evidence_averaged_over_no_crops_is_unknown(frames_scored):
    """A mean over no face crops is not a reading of the media."""
    decision = evaluate(None, face(score=0.999, frames_scored=frames_scored))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("frames_scored", ["8", 8.0, True, [], {"n": 8}])
def test_a_frame_count_that_is_not_a_whole_number_is_unknown(frames_scored):
    decision = evaluate(None, face(score=0.999, frames_scored=frames_scored))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("windows_scored", [0, -1, None])
def test_mouth_dynamics_evidence_averaged_over_no_runs_is_unknown(windows_scored):
    """A mean over no runs that held a trackable face is not a reading of the media."""
    decision = evaluate(lip=lip(score=0.999, windows_scored=windows_scored))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("windows_scored", ["3", 3.0, True, [], {"n": 3}])
def test_a_run_count_that_is_not_a_whole_number_is_unknown(windows_scored):
    decision = evaluate(lip=lip(score=0.999, windows_scored=windows_scored))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


def test_one_detectors_unreadable_figures_do_not_sink_the_others_reading():
    """`R012` is the last resort, not a poison pill.

    A broken synthetic-video row beside a face score above its threshold is still a HIGH: the
    face model's evidence is intact and was measured on its own. And two broken rows beside a
    readable mouth-dynamics score are not `R012` either — that reading is not a level, but it is
    a reading, and `R012` would report an analysis where nothing could be read.
    """
    decision = evaluate(svd(score=math.nan), face(score=0.9931))

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R101"

    on_the_third = evaluate(
        svd(score=math.nan), face(score=math.nan), lip(score=0.9931)
    )

    assert on_the_third.risk_level == "MEDIUM"
    assert on_the_third.rule_id == "R201"


# --------------------------------------------------------------------------------------
# Properties of the ruleset as a whole
# --------------------------------------------------------------------------------------


def test_low_is_never_emitted_anywhere_on_any_scale():
    """Swept across the whole of [0, 1] for all three detectors, including below every `T_LOW`.

    v3 has no LOW branch, and the sweep is what proves it rather than the absence of the word
    from the source.
    """
    levels = set()

    for index in range(0, 1001):
        score = index / 1000
        levels.add(evaluate(svd(score=score), None).risk_level)
        levels.add(evaluate(None, face(score=score)).risk_level)
        levels.add(evaluate(lip=lip(score=score)).risk_level)
        levels.add(
            evaluate(svd(score=score), face(score=score), lip(score=score)).risk_level
        )

    assert levels == {"MEDIUM", "HIGH"}
    assert "LOW" not in levels


def test_the_ruleset_emits_only_the_three_supported_levels():
    """Across every branch reachable in this module, including the degenerate ones."""
    levels = {
        evaluate(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).risk_level
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    assert levels == {"HIGH", "MEDIUM", "UNKNOWN"}


def test_only_the_eight_documented_rules_can_fire():
    rules = {
        evaluate(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).rule_id
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    assert rules == {
        "R010",
        "R012",
        "R100",
        "R101",
        "R102",
        "R200",
        "R201",
    }

    # The id v3 used for a mouth-dynamics HIGH. No arrangement of evidence can produce it here,
    # and no rule in this ruleset may ever wear it: `app.risk_trace` still explains stored
    # `r5-v3.0.0` rows through that sentence, and reusing the id would rewrite what they said.
    assert "R103" not in rules


def test_every_rule_id_the_module_names_is_reachable():
    """No rule constant is declared and left unfirable, and none fires that is not declared."""
    declared = {
        value
        for name, value in vars(risk_engine).items()
        if name.startswith("RULE_") and isinstance(value, str)
    }
    fired = {
        evaluate(
            SVD_STATES[svd_state], FACE_STATES[face_state], LIP_STATES[lip_state]
        ).rule_id
        for svd_state in SVD_STATES
        for face_state in FACE_STATES
        for lip_state in LIP_STATES
    }

    assert declared == fired


def test_every_decision_carries_the_ruleset_and_calibration_that_made_it():
    """Including the UNKNOWNs. A decision with no trace is not explainable later."""
    for svd_state in SVD_STATES:
        for face_state in FACE_STATES:
            for lip_state in LIP_STATES:
                decision = evaluate(
                    SVD_STATES[svd_state],
                    FACE_STATES[face_state],
                    LIP_STATES[lip_state],
                )

                assert decision.rules_version == EXPECTED_RULES_VERSION
                assert decision.calibration_id == EXPECTED_CALIBRATION_ID


def test_the_same_evidence_always_yields_the_same_decision():
    """Stateless and deterministic: nothing accumulates between calls."""
    first = svd(score=0.9550971)
    second = face(score=0.9867588)
    third = lip(score=0.2296253)

    assert (
        evaluate(first, second, third)
        == evaluate(first, second, third)
        == evaluate(
            svd(score=0.9550971), face(score=0.9867588), lip(score=0.2296253)
        )
    )


def test_a_decision_is_immutable_once_made():
    decision = evaluate(svd(score=0.99), None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.risk_level = "MEDIUM"


def test_no_arithmetic_combines_the_scores():
    """The scores never meet, so a fixed set of readings is invariant under moving which
    detector is the loud one — the same is not true of any average, sum or product.

    0.98 and 0.20 combine to the same mean either way round, but the arrangements here are
    decided by different rules on different thresholds, and one of them is not even HIGH.
    """
    quiet_lip = lip(score=0.0154)
    svd_loud = evaluate(svd(score=0.98), face(score=0.20), quiet_lip)
    face_loud = evaluate(svd(score=0.20), face(score=0.98), quiet_lip)

    assert svd_loud.rule_id == "R100"
    assert svd_loud.risk_level == "HIGH"
    # 0.98 is above NVIDIA's threshold (0.9551) and below the face model's (0.9868), so the
    # mirrored pair is not a mirrored decision. An engine that averaged, summed or otherwise
    # pooled the numbers could not tell these two analyses apart at all.
    assert face_loud.rule_id == "R200"
    assert face_loud.risk_level == "MEDIUM"

    # And a third arrangement no mean could separate from the second: the same two figures,
    # with 0.20 now on a scale where it is a flag rather than silence.
    lip_loud = evaluate(svd(score=0.98), face(score=0.98), lip(score=0.20))

    assert lip_loud.rule_id == "R100"
    assert lip_loud.risk_level == "HIGH"


# --------------------------------------------------------------------------------------
# Isolation, in the shape of the input itself
# --------------------------------------------------------------------------------------


def test_the_engine_has_nowhere_to_put_an_uncalibrated_signal():
    """The strongest isolation check there is: the fields do not exist.

    C2PA, active speaker and AASIST cannot influence a classification they cannot be passed
    to. This test fails the moment someone widens the input to admit one, which is exactly
    when the design decision behind rule 11 would be quietly reversed. The mouth-dynamics
    signal is here rather than among them because R5-T3 measured an operating point for it;
    nothing else has one.
    """
    assert {field.name for field in dataclasses.fields(SvdEvidence)} == {
        "provider",
        "signal_type",
        "status",
        "provider_version",
        "score",
        "total_clips",
    }
    assert {field.name for field in dataclasses.fields(FaceEvidence)} == {
        "provider",
        "signal_type",
        "status",
        "provider_version",
        "score",
        "frames_scored",
    }
    assert {field.name for field in dataclasses.fields(LipEvidence)} == {
        "provider",
        "signal_type",
        "status",
        "provider_version",
        "score",
        "windows_scored",
    }


def test_the_engine_takes_no_argument_but_the_three_calibrated_signals():
    """No session, no analysis id, no fourth signal: there is nothing else to pass it."""
    assert list(inspect.signature(evaluate).parameters) == ["svd", "face", "lip"]


def test_each_detector_keeps_its_own_degeneracy_count():
    """The three counts count different things and are not interchangeable.

    NVIDIA aggregates over clips it cut itself; the classifier averages over face crops it
    sampled; the mouth-dynamics model averages over runs of 25 consecutive frames in which the
    face stayed trackable. A shared field would make the numbers read as commensurable.
    """
    svd_fields = {field.name for field in dataclasses.fields(SvdEvidence)}
    face_fields = {field.name for field in dataclasses.fields(FaceEvidence)}
    lip_fields = {field.name for field in dataclasses.fields(LipEvidence)}

    assert "total_clips" in svd_fields and "total_clips" not in face_fields | lip_fields
    assert "frames_scored" in face_fields and "frames_scored" not in svd_fields | lip_fields
    assert "windows_scored" in lip_fields and "windows_scored" not in svd_fields | face_fields


# --------------------------------------------------------------------------------------
# The R7-T5 regression lineages, replayed from their measured readings
# --------------------------------------------------------------------------------------
#
# R7-T5 replayed the unchanged v3 rules over 307 independent genuine lineages and found 22 false
# HIGHs, 21 of them `R103`. Two of the 307 are private benign phone captures, and one of those
# two is the production regression that started this. Its readings — and its eight constructed
# degradations, and the second lineage's — are in the fixture below.
#
# **Numbers, and only numbers.** The clips are not in this repository and never will be: the
# fixture holds each lineage's scores, its aggregate counts and its detector statuses under the
# opaque identifier R7-T5 tracked it by, which is the same thing the published report holds. No
# media, no frames, no filenames.
#
# **Nothing is re-run and nothing is re-measured.** These readings came off the detectors once,
# during R7-T5, on the deployments the calibrations bind to. A test that re-ran inference would
# be measuring today's GPU rather than checking a rule, and a test that adjusted a reading to
# get a level it liked would be fabricating evidence. The fixture is replayed exactly as
# recorded and the *rules* are what is under test.

REGRESSION_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "r7t5_private_regression_readings.json"
)


def _regression_readings():
    """The R7-T5 readings, checked against the identity they were measured under.

    The guard is the point of the loader. These figures are only evidence about this ruleset if
    they came off the deployments its thresholds were measured on and were read against those
    thresholds; a fixture recorded under some other calibration would be a different study, and
    replaying it here would prove nothing about anything.
    """
    payload = json.loads(REGRESSION_FIXTURE.read_text(encoding="utf-8"))

    assert payload["recorded_under"]["calibration_id"] == EXPECTED_CALIBRATION_ID
    assert payload["recorded_under"]["rules_version"] == "r5-v3.0.0"
    assert payload["recorded_under"]["svd_t_high"] == EXPECTED_SVD_T_HIGH
    assert payload["recorded_under"]["face_t_high"] == EXPECTED_FACE_T_HIGH
    assert payload["recorded_under"]["lip_t_high"] == EXPECTED_LIP_T_HIGH
    assert payload["deployments"] == {
        "svd": VALIDATED_FUNCTION_ID,
        "face": VALIDATED_FACE_CHECKPOINT,
        "lip": VALIDATED_LIP_MODEL,
    }

    return payload["readings"]


def _as_evidence(reading):
    """One recorded reading as the three evidence rows the worker would have persisted."""
    scores, counts, statuses = (
        reading["scores"],
        reading["counts"],
        reading["statuses"],
    )
    return (
        svd(
            score=scores["svd"],
            status=statuses["svd"],
            total_clips=counts["svd_total_clips"],
        ),
        face(
            score=scores["face"],
            status=statuses["face"],
            frames_scored=counts["face_frames_scored"],
        ),
        lip(
            score=scores["lip"],
            status=statuses["lip"],
            windows_scored=counts["lip_windows_scored"],
        ),
    )


REGRESSION_READINGS = _regression_readings()
AS_CAPTURED = [r for r in REGRESSION_READINGS if r["derivation"] == "none"]


def test_the_fixture_holds_the_readings_r7_t5_reported_and_no_media():
    """A guard on the fixture itself, before anything is concluded from it.

    Two lineages as captured plus eight degradations each, every one of them recorded as a HIGH
    by `R103` under v3 with LipForensics as the sole responsible detector — which is exactly
    what §7.5 of the R7-T5 report says, and is why this fixture is the right evidence for the
    rule that was withdrawn. Three of the second lineage's variants fell to MEDIUM and are here
    too, unedited: a fixture trimmed to its convenient rows is not a replay.

    The last two assertions are the privacy contract. Nothing in this file is media, a path or a
    filename, and the lineage identifiers are the opaque ones the report published.
    """
    assert len(AS_CAPTURED) == 2
    assert {r["lineage"] for r in AS_CAPTURED} == {
        "private:regression-1",
        "private:regression-2",
    }

    observed = [r["observed_under_r5_v3"] for r in REGRESSION_READINGS]
    high_by_r103 = [o for o in observed if o["rule_id"] == "R103"]

    assert len(high_by_r103) == 15
    assert all(o["risk_level"] == "HIGH" for o in high_by_r103)
    assert all(o["responsible_detectors"] == ["lipforensics"] for o in high_by_r103)
    assert all(r["label"] == "real" for r in REGRESSION_READINGS)

    assert not any(
        key in reading for reading in REGRESSION_READINGS for key in ("path", "filename")
    )


def test_the_production_regression_no_longer_evaluates_high():
    """The clip this task exists for, replayed through the rules that replaced v3.

    `private:regression-1` as it was captured: a benign consumer phone video, LipForensics at
    0.6584 — nearly three times its operating point — and both R4-T1 detectors at their floor,
    0.0403 and 0.0484. Under `r5-v3.0.0` this was a HIGH by `R103`, and it is the false positive
    that was reported from production.

    Under v4 it is `R200`: all three detectors read the media, neither detector entitled to
    decide reached its threshold, and the level says the evidence did not support a
    classification. The mouth-dynamics score is unchanged, still above its threshold, and still
    on the analysis — what changed is that no rule concludes from it.
    """
    reading = next(r for r in AS_CAPTURED if r["lineage"] == "private:regression-1")
    svd_evidence, face_evidence, lip_evidence = _as_evidence(reading)

    # The evidence really is what made the production HIGH: the withdrawn rule's detector above
    # its own threshold, and neither other detector anywhere near its own.
    assert lip_evidence.score >= EXPECTED_LIP_T_HIGH
    assert svd_evidence.score < EXPECTED_SVD_T_HIGH
    assert face_evidence.score < EXPECTED_FACE_T_HIGH
    assert reading["observed_under_r5_v3"] == {
        "risk_level": "HIGH",
        "rule_id": "R103",
        "responsible_detectors": ["lipforensics"],
    }

    decision = evaluate(svd_evidence, face_evidence, lip_evidence)

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R200"
    assert decision.rules_version == EXPECTED_RULES_VERSION
    assert decision.calibration_id == EXPECTED_CALIBRATION_ID


@pytest.mark.parametrize(
    "reading", REGRESSION_READINGS, ids=lambda r: r["reading_id"]
)
def test_no_r7_t5_regression_reading_evaluates_high_under_the_new_ruleset(reading):
    """All 18 recorded readings, both lineages, every degradation.

    R7-T5 measured that the degradations make the mouth-dynamics score *worse*, not better —
    seven of `regression-1`'s eight scored above its as-captured value, one of them at 0.8278.
    Every one of them was a HIGH under v3 and none of them may be one here.

    Sweeping the whole fixture rather than the two as-captured rows is deliberate: a rule that
    happened to spare one clip and not its re-encode would not be a fix, and this is the test
    that would say so.
    """
    decision = evaluate(*_as_evidence(reading))

    assert decision.risk_level != "HIGH"
    assert decision.rule_id not in {"R100", "R101", "R102"}


def test_the_regression_readings_are_still_high_under_the_rules_that_flagged_them():
    """The other half of the claim: v3's arithmetic is reproduced, not disputed.

    R7-T6 did not find a bug in v3's comparisons — it found that the rule those comparisons fed
    was wrong for production. Re-deriving each recorded outcome from the frozen v3 thresholds
    here proves the fixture is the same evidence v3 saw, so the difference measured above is the
    ruleset and nothing else.

    The v3 conditions are spelled out rather than imported: `app.risk_engine` no longer holds
    them, and reaching for a historical rule through today's module is the thing the whole
    version-aware design exists to prevent.
    """
    for reading in REGRESSION_READINGS:
        svd_evidence, face_evidence, lip_evidence = _as_evidence(reading)
        observed = reading["observed_under_r5_v3"]

        flagged_under_v3 = {
            "nvidia": svd_evidence.score >= EXPECTED_SVD_T_HIGH,
            "efficientnet-b7": face_evidence.score >= EXPECTED_FACE_T_HIGH,
            "lipforensics": lip_evidence.score >= EXPECTED_LIP_T_HIGH,
        }
        responsible = sorted(name for name, hit in flagged_under_v3.items() if hit)

        assert responsible == sorted(observed["responsible_detectors"])
        assert (observed["risk_level"] == "HIGH") is bool(responsible)


def test_the_withdrawal_is_what_changed_the_regression_and_not_the_other_evidence():
    """The mouth-dynamics reading is the only thing standing between v3 and v4 here.

    Both R4-T1 detectors are replayed exactly as measured, and the mouth-dynamics reading is
    varied across its whole scale: from the top, through its operating point, to the floor. Not
    one of those readings changes the decision. Under v3 the same sweep crossed a band boundary
    at 0.2296.

    This is also the isolation check the task turns on. If some *other* detector had drifted
    into deciding this analysis, the level would move somewhere in this sweep, and it does not.
    """
    reading = next(r for r in AS_CAPTURED if r["lineage"] == "private:regression-1")
    svd_evidence, face_evidence, _ = _as_evidence(reading)

    decisions = {
        score: evaluate(svd_evidence, face_evidence, lip(score=score))
        for score in (1.0, 0.9931, 0.6584, 0.4424, EXPECTED_LIP_T_HIGH, 0.0168, 0.0)
    }

    assert {(d.risk_level, d.rule_id) for d in decisions.values()} == {
        ("MEDIUM", "R200")
    }


def test_an_unreadable_mouth_dynamics_row_does_not_disturb_the_regression_evidence():
    """A failed third detector costs coverage on this analysis and nothing else.

    The same two R4-T1 readings, with the mouth-dynamics detector failing, returning nothing
    readable, and coming from an uncalibrated deployment in turn. The band never moves; the rule
    id records that one of the three questions went unanswered, which is a smaller claim than
    `R200` and is the honest one.
    """
    reading = next(r for r in AS_CAPTURED if r["lineage"] == "private:regression-1")
    svd_evidence, face_evidence, _ = _as_evidence(reading)

    for broken in (
        lip(score=None, status="FAILED"),
        lip(score=0.6584, status="TIMEOUT"),
        lip(score=math.nan),
        lip(score=0.6584, windows_scored=0),
        lip(score=0.6584, provider_version="some-other-checkpoint"),
        None,
    ):
        decision = evaluate(svd_evidence, face_evidence, broken)

        assert decision.risk_level == "MEDIUM"
        assert decision.rule_id == "R201"


# --------------------------------------------------------------------------------------
# Orchestration, against real PostgreSQL
# --------------------------------------------------------------------------------------


@pytest.fixture
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def analysed(database):
    """An analysis whose evidence is already persisted and whose job is still `processing`.

    Exactly the state `conclude_job` is called in: the detectors have run, every signal row
    is committed, and the only thing left is the classification. The signals are written here
    rather than produced by driving the detectors, so the calibrated evidence under test is
    exact and the uncalibrated signals around it can be varied at will.
    """
    created = []

    def stage(
        *,
        svd_signal=True,
        face_signal=False,
        lip_signal=False,
        svd_kwargs=None,
        face_kwargs=None,
        lip_kwargs=None,
        context=(),
    ):
        with SessionLocal() as session:
            analysis = Analysis(status="queued")
            session.add(analysis)
            session.flush()

            if svd_signal:
                session.add(persisted_svd_signal(analysis.id, **(svd_kwargs or {})))
            if face_signal:
                session.add(persisted_face_signal(analysis.id, **(face_kwargs or {})))
            if lip_signal:
                session.add(persisted_lip_signal(analysis.id, **(lip_kwargs or {})))
            for signal in context:
                signal.analysis_id = analysis.id
                session.add(signal)

            session.add(AnalysisJob(analysis_id=analysis.id, status="processing"))
            session.commit()
            created.append(analysis.id)

            return worker.ClaimedJob(
                job_id=session.query(AnalysisJob)
                .filter_by(analysis_id=analysis.id)
                .one()
                .id,
                analysis_id=analysis.id,
                original_storage_key="originals/unused",
                normalization_required=False,
                frame_rate=30.0,
            )

    yield stage

    with SessionLocal() as session:
        for analysis_id in created:
            # Signals and the job go with it through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


def persisted_svd_signal(
    analysis_id,
    *,
    score=0.5,
    status="SUCCESS",
    provider_version=VALIDATED_FUNCTION_ID,
    total_clips=TOTAL_CLIPS,
) -> AnalysisSignal:
    """A synthetic-video row exactly as `detect_synthetic_video` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="nvidia",
        signal_type="synthetic_video",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={"logit": LOGIT, "total_clips": total_clips},
    )


def persisted_face_signal(
    analysis_id,
    *,
    score=0.5,
    status="SUCCESS",
    provider_version=VALIDATED_FACE_CHECKPOINT,
    frames_scored=FRAMES_SCORED,
) -> AnalysisSignal:
    """A face-manipulation row exactly as `detect_face_manipulation` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="efficientnet-b7",
        signal_type="face_manipulation",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={
            "frames_requested": 8,
            "frames_decoded": frames_scored if isinstance(frames_scored, int) else 8,
            "frames_scored": frames_scored,
            "frame_scores": [],
        },
    )


def persisted_lip_signal(
    analysis_id,
    *,
    score=0.05,
    status="SUCCESS",
    provider_version=VALIDATED_LIP_MODEL,
    windows_scored=WINDOWS_SCORED,
) -> AnalysisSignal:
    """A mouth-dynamics row exactly as `detect_lip_forensics` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="lipforensics",
        signal_type="lip_forensics",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={
            "windows_requested": 4,
            "windows_read": 4,
            "windows_scored": windows_scored,
            "window_logits": [],
        },
    )


def context_signals(*, provenance=True, active_speaker=True, audio=True, failed=False):
    """The three signals that must never touch a classification, in a stated arrangement.

    The face-manipulation signal is deliberately not among them any more, and neither is the
    mouth-dynamics one. Under ruleset v3 both are calibrated deciders rather than context —
    R4-T1 is the measurement that moved the first and R5-T3 the one that moved the second, and
    they are varied through `persisted_face_signal` and `persisted_lip_signal` instead.

    What is left has no calibration of any kind, which is the whole reason it is excluded.
    """
    status = "FAILED" if failed else "SUCCESS"
    signals = []

    if provenance:
        signals.append(
            AnalysisSignal(
                provider="c2pa",
                signal_type="provenance",
                status=status,
                provider_version="0.90.14",
                signal_metadata={
                    "manifest_exists": not failed,
                    "validation_state": "Valid" if not failed else None,
                    "signature_issuer": "Test Signing Cert" if not failed else None,
                },
            )
        )
    if active_speaker:
        signals.append(
            AnalysisSignal(
                provider="nvidia",
                signal_type="active_speaker",
                status=status,
                provider_version="f286f937-05c4-454b-8312-fba67a2a6fa7",
                signal_metadata={"total_speaking_segments": 0 if failed else 12},
            )
        )
    if audio:
        signals.append(
            AnalysisSignal(
                provider="aasist",
                signal_type="audio_authenticity",
                status=status,
                provider_version="SpeechAntiSpoofingBenchmarks/AASIST@16774d45",
                signal_metadata={"total_audio_windows": 0 if failed else 3},
            )
        )

    return signals


def read_analysis(analysis_id) -> Analysis:
    with SessionLocal() as reader:
        return reader.query(Analysis).filter_by(id=analysis_id).one()


def read_job(job_id) -> AnalysisJob:
    with SessionLocal() as reader:
        return reader.query(AnalysisJob).filter_by(id=job_id).one()


def read_signals(analysis_id) -> dict[str, AnalysisSignal]:
    with SessionLocal() as reader:
        rows = reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).all()

    return {row.signal_type: row for row in rows}


@pytest.mark.integration
def test_the_engine_reads_the_synthetic_video_evidence_the_database_holds(analysed):
    """Not the values a detector returned in memory — the committed row.

    A classification derived from anything else could disagree with the evidence stored
    beside it, and nothing in the database would say so.
    """
    claimed = analysed(svd_kwargs={"score": 0.9931, "total_clips": 1829})

    with SessionLocal() as session:
        evidence = worker.persisted_svd_evidence(session, claimed.analysis_id)

    assert evidence == SvdEvidence(
        provider="nvidia",
        signal_type="synthetic_video",
        status="SUCCESS",
        provider_version=VALIDATED_FUNCTION_ID,
        score=0.9931,
        total_clips=1829,
    )


@pytest.mark.integration
def test_the_engine_reads_the_face_evidence_the_database_holds(analysed):
    claimed = analysed(
        svd_signal=False,
        face_signal=True,
        face_kwargs={"score": 0.9943, "frames_scored": 6},
    )

    with SessionLocal() as session:
        evidence = worker.persisted_face_evidence(session, claimed.analysis_id)

    assert evidence == FaceEvidence(
        provider="efficientnet-b7",
        signal_type="face_manipulation",
        status="SUCCESS",
        provider_version=VALIDATED_FACE_CHECKPOINT,
        score=0.9943,
        frames_scored=6,
    )


@pytest.mark.integration
def test_the_engine_reads_the_mouth_dynamics_evidence_the_database_holds(analysed):
    claimed = analysed(
        svd_signal=False,
        lip_signal=True,
        lip_kwargs={"score": CORPUS_LIP_FACESWAP_MIN, "windows_scored": 2},
    )

    with SessionLocal() as session:
        evidence = worker.persisted_lip_evidence(session, claimed.analysis_id)

    assert evidence == LipEvidence(
        provider="lipforensics",
        signal_type="lip_forensics",
        status="SUCCESS",
        provider_version=VALIDATED_LIP_MODEL,
        score=CORPUS_LIP_FACESWAP_MIN,
        windows_scored=2,
    )


@pytest.mark.integration
def test_a_missing_signal_reads_back_as_absent(analysed):
    """Each reader answers about its own signal and does not see the others' rows."""
    claimed = analysed(
        svd_signal=False, face_signal=True, lip_signal=True, context=context_signals()
    )

    with SessionLocal() as session:
        assert worker.persisted_svd_evidence(session, claimed.analysis_id) is None
        assert worker.persisted_face_evidence(session, claimed.analysis_id) is not None
        assert worker.persisted_lip_evidence(session, claimed.analysis_id) is not None

    other = analysed(svd_signal=True, face_signal=False, lip_signal=False)

    with SessionLocal() as session:
        assert worker.persisted_face_evidence(session, other.analysis_id) is None
        assert worker.persisted_lip_evidence(session, other.analysis_id) is None
        assert worker.persisted_svd_evidence(session, other.analysis_id) is not None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("svd_kwargs", "face_kwargs", "lip_kwargs", "risk_level", "rule_id"),
    [
        # All three above their thresholds.
        ({"score": 0.9931}, {"score": 0.9931}, {"score": 0.9931}, "HIGH", "R102"),
        # Both deciding detectors, with the third quiet. The finding is not attributed to one
        # of them, and the trace does not pretend to say which two.
        ({"score": 0.9931}, {"score": 0.9931}, {"score": 0.0154}, "HIGH", "R102"),
        # One deciding detector with the mouth-dynamics score at the top of its scale. `R102`
        # under v3, and the single detector's own rule here: a reading that may not decide may
        # not corroborate either, and the stored rule id is where that would show.
        ({"score": 0.9931}, {"score": 0.4646}, {"score": 0.9931}, "HIGH", "R100"),
        ({"score": 0.4646}, {"score": 0.9931}, {"score": 0.9931}, "HIGH", "R101"),
        # Synthetic video alone, with the other two quiet — `sonic_en_03`.
        (
            {"score": CORPUS_SYNTHETIC_SVD},
            {"score": CORPUS_SYNTHETIC_FACE},
            {"score": CORPUS_LIP_GENUINE_MAX},
            "HIGH",
            "R100",
        ),
        # Face alone, with NVIDIA quiet — `ffpp_dev_Deepfakes_106_198`.
        (
            {"score": CORPUS_FACESWAP_SVD},
            {"score": CORPUS_FACESWAP_FACE},
            {"score": None, "status": "FAILED"},
            "HIGH",
            "R101",
        ),
        # Mouth dynamics alone, at the lowest score any face swap in R5-T3's corpus reached.
        # A HIGH by `R103` under v3 and a full-coverage MEDIUM here, through the database.
        (
            {"score": CORPUS_FACESWAP_SVD},
            {"score": 0.4646},
            {"score": CORPUS_LIP_FACESWAP_MIN},
            "MEDIUM",
            "R200",
        ),
        # All three readable, none flagged.
        ({"score": 0.9541}, {"score": 0.9865}, {"score": 0.0168}, "MEDIUM", "R200"),
        ({"score": 0.0}, {"score": 0.0}, {"score": 0.0}, "MEDIUM", "R200"),
        # Some readable, the rest failed or uncalibrated.
        (
            {"score": 0.4646},
            {"score": None, "status": "FAILED"},
            {"score": 0.0154},
            "MEDIUM",
            "R201",
        ),
        (
            {"score": 0.999, "status": "FAILED"},
            {"score": 0.4646},
            {"score": None, "status": "FAILED"},
            "MEDIUM",
            "R201",
        ),
        # None readable.
        (
            {"score": 0.999, "status": "FAILED"},
            {"score": None, "status": "FAILED"},
            {"score": None, "status": "FAILED"},
            "UNKNOWN",
            "R010",
        ),
        (
            {"score": 0.999, "provider_version": "other-function"},
            {"score": 0.999, "provider_version": "other-checkpoint"},
            {"score": 0.999, "provider_version": "other-model"},
            "UNKNOWN",
            "R010",
        ),
        ({"score": None}, {"score": None}, {"score": None}, "UNKNOWN", "R012"),
        (
            {"score": 0.999, "total_clips": 0},
            {"score": 0.999, "frames_scored": 0},
            {"score": 0.999, "windows_scored": 0},
            "UNKNOWN",
            "R012",
        ),
    ],
)
def test_the_decision_and_its_whole_trace_are_persisted(
    analysed, svd_kwargs, face_kwargs, lip_kwargs, risk_level, rule_id
):
    """Level, ruleset, calibration and the rule that fired — all four, exactly.

    The trace is what makes a decision explainable after the thresholds move on, so each
    branch is checked at the stored row rather than at the return value. In particular the
    rule id is what tells a later reader *which detector* concluded this, which is the whole
    reason a multi-source ruleset needs one.
    """
    claimed = analysed(
        face_signal=True,
        lip_signal=True,
        svd_kwargs=svd_kwargs,
        face_kwargs=face_kwargs,
        lip_kwargs=lip_kwargs,
    )

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == risk_level
    assert analysis.risk_rule_id == rule_id
    assert analysis.risk_rules_version == EXPECTED_RULES_VERSION
    assert analysis.risk_calibration_id == EXPECTED_CALIBRATION_ID


@pytest.mark.integration
def test_a_stored_face_finding_alone_completes_high(analysed):
    """The capability v1 did not have, end to end through the database.

    No synthetic-video row at all, a face score above its calibrated threshold, and the
    analysis classifies HIGH by `R101` with the trace to prove which detector said so.
    """
    claimed = analysed(
        svd_signal=False,
        face_signal=True,
        face_kwargs={"score": CORPUS_FACESWAP_FACE},
        context=context_signals(),
    )

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == "HIGH"
    assert analysis.risk_rule_id == "R101"
    assert analysis.risk_rules_version == EXPECTED_RULES_VERSION
    assert analysis.risk_calibration_id == EXPECTED_CALIBRATION_ID


@pytest.mark.integration
def test_an_analysis_with_no_calibrated_evidence_at_all_is_unknown(analysed):
    claimed = analysed(
        svd_signal=False, face_signal=False, lip_signal=False, context=context_signals()
    )

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"


@pytest.mark.integration
def test_a_stored_mouth_dynamics_finding_alone_does_not_complete_high(analysed):
    """The capability r7-v4.0.0 withdrew, end to end through the database.

    No synthetic-video row and no face row at all, a mouth-dynamics score above its calibrated
    threshold. Under `r5-v3.0.0` the stored decision was HIGH by `R103`; here it is MEDIUM by
    `R201`, and the row says so in all four columns.

    The signal is written either way, and the assertion below is on the *decision*, not on the
    evidence: what R7-T6 removed is a rule, and a build that removed the detector instead would
    pass a level check and fail this analysis's readers.
    """
    claimed = analysed(
        svd_signal=False,
        lip_signal=True,
        lip_kwargs={"score": CORPUS_LIP_FACESWAP_MIN},
        context=context_signals(),
    )

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == "MEDIUM"
    assert analysis.risk_rule_id == "R201"
    assert analysis.risk_rules_version == EXPECTED_RULES_VERSION
    assert analysis.risk_calibration_id == EXPECTED_CALIBRATION_ID


@pytest.mark.integration
def test_risk_evaluation_is_the_last_step_before_the_job_completes(analysed):
    """Nothing is classified while the job is still in progress, and nothing completes
    without a classification."""
    claimed = analysed(svd_kwargs={"score": 0.5})

    before = read_analysis(claimed.analysis_id)
    assert before.status == "queued"
    assert before.risk_level is None
    assert before.risk_rules_version is None
    assert before.risk_calibration_id is None
    assert before.risk_rule_id is None

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    assert read_job(claimed.job_id).status == "completed"
    assert read_analysis(claimed.analysis_id).status == "completed"
    assert read_analysis(claimed.analysis_id).risk_level == "MEDIUM"


@pytest.mark.integration
def test_the_forensic_signals_are_untouched_by_the_classification(analysed):
    """The decision goes on the analysis. No signal row is rewritten to carry a verdict."""
    claimed = analysed(
        svd_kwargs={"score": 0.99},
        face_signal=True,
        face_kwargs={"score": 0.99},
        lip_signal=True,
        lip_kwargs={"score": 0.99},
        context=context_signals(),
    )

    before = {
        signal_type: (row.status, row.score, row.provider_version, row.risk_level)
        for signal_type, row in read_signals(claimed.analysis_id).items()
    }

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    after = {
        signal_type: (row.status, row.score, row.provider_version, row.risk_level)
        for signal_type, row in read_signals(claimed.analysis_id).items()
    }

    assert after == before
    # In particular, the per-signal column stays null on every one of them: risk is a
    # decision about the analysis, taken under a named ruleset, not a label per provider.
    assert all(row.risk_level is None for row in read_signals(claimed.analysis_id).values())


# --------------------------------------------------------------------------------------
# Isolation, against stored evidence
# --------------------------------------------------------------------------------------

# Identical calibrated evidence, each entry a different arrangement of everything else.
# Removed, present, failed, and every combination in between.
CONTEXT_ARRANGEMENTS = {
    "all present and successful": context_signals(),
    "all present and failed": context_signals(failed=True),
    "none at all": [],
    "no provenance": context_signals(provenance=False),
    "no active speaker": context_signals(active_speaker=False),
    "no audio authenticity": context_signals(audio=False),
    "provenance only": context_signals(active_speaker=False, audio=False),
    "provenance failed": context_signals(active_speaker=False, audio=False, failed=True),
    "active speaker only": context_signals(provenance=False, audio=False),
    "active speaker failed": context_signals(provenance=False, audio=False, failed=True),
    "audio only": context_signals(provenance=False, active_speaker=False),
    "audio failed": context_signals(provenance=False, active_speaker=False, failed=True),
}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("svd_score", "face_score", "lip_score", "risk_level", "rule_id"),
    [
        (0.9931, 0.4646, 0.0154, "HIGH", "R100"),
        (0.4646, 0.9931, 0.0154, "HIGH", "R101"),
        (0.4646, 0.4646, 0.9931, "MEDIUM", "R200"),
        (0.4646, 0.4646, 0.0154, "MEDIUM", "R200"),
    ],
)
def test_no_arrangement_of_the_uncalibrated_signals_changes_the_classification(
    analysed, svd_score, face_score, lip_score, risk_level, rule_id
):
    """The isolation requirement, checked exhaustively against stored evidence.

    Every arrangement of C2PA, active speaker and AASIST — present, absent, successful,
    failed — over one unchanged set of calibrated scores. All of them must land on the same
    band by the same rule. Nothing here is averaged, voted on, weighted or combined, and this
    test is what would fail the moment something started to be.

    The three signals here have no calibration of any kind, which is the whole reason they
    are excluded. The face-manipulation and mouth-dynamics signals used to be in this table and
    are not any more: R4-T1 and R5-T3 measured operating points for them, so they are deciders
    now and are varied above rather than held inert here.
    """
    decisions = {}

    for name, context in CONTEXT_ARRANGEMENTS.items():
        claimed = analysed(
            svd_kwargs={"score": svd_score},
            face_signal=True,
            face_kwargs={"score": face_score},
            lip_signal=True,
            lip_kwargs={"score": lip_score},
            # Rebuilt per arrangement: an ORM instance cannot be attached twice.
            context=[
                AnalysisSignal(
                    provider=signal.provider,
                    signal_type=signal.signal_type,
                    status=signal.status,
                    score=signal.score,
                    provider_version=signal.provider_version,
                    signal_metadata=signal.signal_metadata,
                )
                for signal in context
            ],
        )

        with SessionLocal() as session:
            worker.conclude_job(session, claimed)

        analysis = read_analysis(claimed.analysis_id)
        decisions[name] = (
            analysis.risk_level,
            analysis.risk_rule_id,
            analysis.risk_rules_version,
            analysis.risk_calibration_id,
        )

    expected = (risk_level, rule_id, EXPECTED_RULES_VERSION, EXPECTED_CALIBRATION_ID)

    assert decisions == {name: expected for name in CONTEXT_ARRANGEMENTS}


@pytest.mark.integration
def test_context_signals_cannot_rescue_three_uncalibrated_detectors(analysed):
    """UNKNOWN is not a gap for other evidence to fill in.

    An analysis with a perfect provenance chain, a full speaker timeline and clean audio
    windows still classifies UNKNOWN when every scoring detector came from a deployment the
    thresholds were never measured against.
    """
    claimed = analysed(
        svd_kwargs={"score": 0.9999, "provider_version": f"{VALIDATED_FUNCTION_ID}-v2"},
        face_signal=True,
        face_kwargs={
            "score": 0.9999,
            "provider_version": f"{VALIDATED_FACE_CHECKPOINT}-v2",
        },
        lip_signal=True,
        lip_kwargs={"score": 0.9999, "provider_version": f"{VALIDATED_LIP_MODEL}-v2"},
        context=context_signals(),
    )

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "duplicated", ["synthetic_video", "face_manipulation", "lip_forensics"]
)
def test_duplicate_calibrated_evidence_raises_rather_than_classifying(
    analysed, duplicated
):
    """A defect in how evidence was written must not be papered over with a verdict.

    Two rows for one detector on one analysis means something wrote evidence twice. Picking
    one and classifying it would publish a decision over half the evidence and hide the
    defect; raising sends it to the worker's own handler, which logs it and fails the job
    with the forensic rows intact.
    """
    claimed = analysed(
        svd_kwargs={"score": 0.5},
        face_signal=True,
        face_kwargs={"score": 0.5},
        lip_signal=True,
        lip_kwargs={"score": 0.05},
    )

    with SessionLocal() as session:
        if duplicated == "synthetic_video":
            session.add(persisted_svd_signal(claimed.analysis_id, score=0.99))
        elif duplicated == "face_manipulation":
            session.add(persisted_face_signal(claimed.analysis_id, score=0.99))
        else:
            session.add(persisted_lip_signal(claimed.analysis_id, score=0.99))
        session.commit()

    with SessionLocal() as session:
        with pytest.raises(MultipleResultsFound):
            worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    # No fabricated classification, and no half-finished completion either.
    assert analysis.risk_level is None
    assert analysis.status == "queued"
    assert read_job(claimed.job_id).status == "processing"
    # The evidence that provoked it survives untouched.
    with SessionLocal() as reader:
        assert (
            reader.query(AnalysisSignal)
            .filter_by(analysis_id=claimed.analysis_id, signal_type=duplicated)
            .count()
            == 2
        )


@pytest.mark.integration
def test_a_failed_job_records_no_classification(analysed):
    """`fail_job` writes no risk columns. Null is the absence of a decision, and it is
    deliberately not `UNKNOWN`, which is a conclusion an explicit rule reached."""
    claimed = analysed(svd_kwargs={"score": 0.99})

    with SessionLocal() as session:
        worker.fail_job(session, claimed, RuntimeError("storage is unreachable"))

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.status == "failed"
    assert analysis.risk_level is None
    assert analysis.risk_rules_version is None
    assert analysis.risk_calibration_id is None
    assert analysis.risk_rule_id is None


def test_the_decision_type_is_the_only_thing_persisted():
    """Four fields, and no room for a provider figure or a prose reason to be copied
    alongside them.

    The signal rows remain the forensic record; a score duplicated onto the analysis could
    drift from the row it came from (rule 11). A stored reason string could drift from the
    rule that actually fired, which is why the rationale is derived from `rule_id` and
    `rules_version` at read time rather than persisted beside them.
    """
    fields = {field.name for field in dataclasses.fields(RiskDecision)}

    assert fields == {"risk_level", "rules_version", "calibration_id", "rule_id"}


def test_an_analysis_row_carries_no_detector_figures_and_no_reason_column():
    """The risk columns on `analyses` are a decision trace, not a copy of the evidence."""
    risk_columns = {
        column.name for column in Analysis.__table__.columns if column.name.startswith("risk")
    }

    assert risk_columns == {
        "risk_level",
        "risk_rules_version",
        "risk_calibration_id",
        "risk_rule_id",
    }
