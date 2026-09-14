"""What a persisted risk decision says, read back under the rules that produced it.

This module explains decisions; it does not take them. `app.risk_engine` classified the
analysis once, at the time it ran, and committed four columns — `risk_level`,
`risk_rule_id`, `risk_rules_version`, `risk_calibration_id` — beside the detector rows the
decision was taken from. Everything here is a *derived view* of those columns and that
already-persisted evidence, assembled at read time so the report and the API can show the
reasoning without a second source of truth for the reasoning.

Three properties are load-bearing, and each is a thing this module refuses to do.

**It never re-evaluates.** `risk_engine.evaluate` is not imported, not called, and cannot
be: nothing below constructs an evidence dataclass the engine would accept, and no branch
here can produce a `risk_level` or a `rule_id`. Both are read off the analysis row. A trace
builder that re-ran the rules would eventually disagree with the record it was supposed to
be explaining — the day a threshold moves, every historical analysis would silently acquire
a new explanation and, worse, a plausible one.

**It never uses today's numbers on yesterday's decision.** The thresholds and rule meanings
below are frozen literals, one set per ruleset version, transcribed from the engine as each
version stood. `app.risk_engine` is deliberately not imported, so editing a threshold there
— a recalibration, a fixture, a bug — cannot reach a historical trace at all. A `p7-v1.0.0`
decision is explained against 0.98 because that is what `T_HIGH` was when it was taken, and
it stays 0.98 here forever.

**It never converts silence into reassurance.** A detector that failed, abstained, never
ran, ran as an uncalibrated deployment, or answered with figures that cannot be read is
reported as `unavailable`, with the reason it is unavailable and with no score and no
threshold beside it — the decision could not use that reading, so the trace does not stage
it as though a comparison had been made. Such a detector is never reported as a score below
a threshold, never counted toward anything, and — this is the part that matters
forensically — never described as evidence that the media was not manipulated. Nor is
`threshold_not_reached`, which is what a detector reports for a manipulation family it is
blind to just as readily as for genuine media (see the complementarity measurements in
`app.risk_engine`). The vocabulary here is `threshold_reached`, `threshold_not_reached`,
`unavailable` and `not_interpreted`; `HIGH`, `MEDIUM` and `UNKNOWN` are the only levels, and
no word in this file asserts that anything is authentic, real, fake or manipulated.

Unresolvable metadata degrades rather than guesses. A `rules_version` this module has never
heard of, or a `calibration_id` that is not the one that version was measured under, yields
the persisted decision with its contributions left uninterpreted — never a trace assembled
from whatever thresholds happened to be nearest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The status a detector row carries when the provider actually answered. Spelled here rather
# than imported from `app.risk_engine`, which this module must not depend on: the point of
# the frozen tables below is that nothing in today's engine can reach a historical trace.
# This string is the persisted vocabulary of `analysis_signals.status` and has never changed.
SIGNAL_STATUS_SUCCESS = "SUCCESS"

SCORE_FLOOR = 0.0
SCORE_CEILING = 1.0

# How one detector stood in the decision, in the only four states the persisted evidence can
# support. None of them is a statement about the media's authenticity.
CONDITION_THRESHOLD_REACHED = "threshold_reached"
CONDITION_THRESHOLD_NOT_REACHED = "threshold_not_reached"
CONDITION_UNAVAILABLE = "unavailable"
# The detector's own figures are readable, but the historical threshold they would have to be
# read against could not be resolved. Deliberately distinct from `unavailable`, which is about
# the detector, and from the two threshold answers, which would be inventing one.
CONDITION_NOT_INTERPRETED = "not_interpreted"

# Why a detector contributed nothing. Only ever set alongside `unavailable`.
UNAVAILABLE_NO_READING = "no_reading"
UNAVAILABLE_DETECTOR_DID_NOT_REPORT = "detector_did_not_report"
UNAVAILABLE_UNCALIBRATED_DEPLOYMENT = "uncalibrated_deployment"
UNAVAILABLE_UNREADABLE_FIGURES = "unreadable_figures"
UNAVAILABLE_THRESHOLD_UNRESOLVED = "threshold_unresolved"

# What the fired rule made of this detector. `decisive` is only ever set on a detector that
# reached its own threshold under a rule that concluded HIGH *and* that its ruleset version could
# take a decision from — the HIGH rules are disjunctive, so such a detector is the reason for the
# level. Everything else was in scope of the ruleset and read by it, which is all `considered`
# claims, and it is all a `r7-v4.0.0` mouth-dynamics crossing may claim however high it scored.
ROLE_DECISIVE = "decisive"
ROLE_CONSIDERED = "considered"

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_UNKNOWN = "UNKNOWN"

# What `r9-v5.0.0` may conclude, transcribed here as the three levels above were: a frozen copy
# of the vocabulary that version decided in, not an import from the engine. The two vocabularies
# are never merged and never mapped onto one another — a stored `MEDIUM` is not an
# `INCONCLUSIVE` and is never re-labelled as one (R9-T1 invariant 4) — so a reader resolves
# which of the two a decision speaks through `rules_version` and nothing else.
#
# `NO_CALIBRATED_MANIPULATION_SIGNAL` says the decision-eligible detectors were read and none
# reached its operating point. It is not "Real", "Genuine", "Authentic" or "Clean", here as in
# the engine: this module already refuses to convert silence into reassurance, and it refuses to
# convert a below-threshold reading into it either.
VERDICT_MANIPULATION_DETECTED = "MANIPULATION_DETECTED"
VERDICT_NO_SIGNAL = "NO_CALIBRATED_MANIPULATION_SIGNAL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

# The two conditions in which a detector actually produced a reading the persisted decision
# could use. Reaching a threshold and not reaching it are both readings; `unavailable` is the
# absence of one, and `not_interpreted` is a reading this build cannot say was read against
# anything. This set is what `decision_coverage` counts, and it is deliberately the only
# definition of usability in this module: a second, looser one — "status was SUCCESS", say —
# would let the trace report coverage the engine never had.
USABLE_READING_CONDITIONS = frozenset(
    {CONDITION_THRESHOLD_REACHED, CONDITION_THRESHOLD_NOT_REACHED}
)


@dataclass(frozen=True)
class CalibratedSignal:
    """One detector as a ruleset version knew it, frozen at that version's identity.

    `provider`, `signal_type` and `provider_version` together are what the engine required
    before it would read a score at all: a different deployment of the same model is an
    uncalibrated one, because the operating point was measured against that deployment's
    distribution and nothing else. `count_key` names the metadata figure the engine demanded
    be positive — the provider's own statement that it aggregated something.

    `decisional` says whether *this version's rules* could take a HIGH from this detector. It is
    a property of the ruleset and not of the detector: LipForensics was decisional under
    `r5-v3.0.0` and is not under `r7-v4.0.0`, with the same threshold and the same deployment on
    both sides of that line, because R7-T5 measured what the threshold does on genuine media and
    the rules changed rather than the measurement. It defaults to true, which is what every
    detector in every ruleset up to v3 was, so no historical entry below is altered by its
    existence. Only `role` reads it: a detector that could not decide is never reported as having
    decided, however high it scored.
    """

    signal_type: str
    provider: str
    provider_version: str
    threshold: float
    count_key: str
    decisional: bool = True


@dataclass(frozen=True)
class Ruleset:
    """One immutable ruleset version: which detectors it read, and what its rules meant.

    `rules` is why this table exists at all. `R200` under `p7-v1.0.0` is one detector's score
    sitting below its threshold; under `r4-v2.0.0` it is two detectors both below theirs;
    under `r5-v3.0.0` it is three below theirs, and under `r7-v4.0.0` it is three readable with
    neither decisional one reaching its own — a band a stored mouth-dynamics score above 0.2296
    can now sit inside and could not before. `R102` is "both detectors" in v2, "two or more" in
    v3, and "both decisional detectors" in v4. The same four characters, four different
    statements — so an old decision read under today's meanings would be misreported while
    looking entirely well-formed.

    `R103` is the sharpest case and the reason nothing here is ever edited in place: it exists in
    `r5-v3.0.0` and in no other version. A stored v3 row naming it must keep reading as the
    sentence v3 wrote, and a v4 row can never name it, because v4 has no such rule.

    `decision_total` is the `D_total` of R9-T1 section 2.4: how many decision-eligible
    detectors this version *expected*, frozen as a literal on the version itself. It is None on
    every ruleset that predates the R9 coverage model, and a None here means this module reports
    no coverage for that version rather than a plausible-looking fraction. v4's `R200` is "all
    three were readable and neither deciding one reached its threshold", which is a statement
    about three detectors under a rule that never counted a denominator; rendering it as "2/2"
    would be a coverage claim v4 never made and R9-T1 forbids inventing.

    `decisive_level` is the conclusion under which a detector that reached its own threshold is
    the reason for the decision. It is `HIGH` for every version decided in the legacy vocabulary
    and `MANIPULATION_DETECTED` for v5, because the two vocabularies name the same position with
    different words. Reading a v5 decision against `HIGH` would report every detector that
    actually produced the verdict as merely `considered`.
    """

    rules_version: str
    calibration_id: str
    signals: tuple[CalibratedSignal, ...]
    rules: dict[str, str]
    decision_total: int | None = None
    decisive_level: str = RISK_HIGH

    def __post_init__(self) -> None:
        """Refuse a version whose frozen denominator and decision-eligible detectors disagree.

        `decision_total` is written as a literal, exactly as the engine writes `D_TOTAL_V5`, and
        is never `len()` of anything: a denominator derived from the detectors present would let
        a detector that was never listed improve coverage by being absent. What this checks is
        that the literal still describes the table beside it — if a later edit adds or demotes a
        decision-eligible detector without moving the number, coverage silently stops meaning
        what it says, and a loud failure at import is the only acceptable outcome (R9-T1
        invariant 9).
        """
        if self.decision_total is None:
            return

        declared = sum(1 for signal in self.signals if signal.decisional)
        if declared != self.decision_total:
            raise ValueError(
                f"{self.rules_version} freezes D_total at {self.decision_total} but lists "
                f"{declared} decision-eligible detectors; the denominator and the detectors "
                "it counts have gone out of step."
            )


# NVIDIA's synthetic-video deployment, unchanged across every ruleset. The threshold is
# not: P7 chose 0.98 by hand, R4-T1 measured 0.9550971388816833 in its place.
_SVD_PROVIDER = "nvidia"
_SVD_SIGNAL_TYPE = "synthetic_video"
_SVD_PROVIDER_VERSION = "847b6e53-0133-452d-ab85-d7acf3ace723"
_SVD_COUNT_KEY = "total_clips"

_FACE_PROVIDER = "efficientnet-b7"
_FACE_SIGNAL_TYPE = "face_manipulation"
_FACE_PROVIDER_VERSION = (
    "tomas-gajarsky/facetorch-deepfake-efficientnet-b7@4acc494f37eb63d7457166eff2acb45c5b04b9a6"
)
_FACE_COUNT_KEY = "frames_scored"

_LIP_PROVIDER = "lipforensics"
_LIP_SIGNAL_TYPE = "lip_forensics"
_LIP_PROVIDER_VERSION = (
    "https://github.com/ahaliassos/LipForensics"
    "@d0bf5553bfb9676f1771d590472b26a3a76de894"
    "+4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
)
_LIP_COUNT_KEY = "windows_scored"


RULESET_V1 = Ruleset(
    rules_version="p7-v1.0.0",
    calibration_id="3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c",
    signals=(
        CalibratedSignal(
            signal_type=_SVD_SIGNAL_TYPE,
            provider=_SVD_PROVIDER,
            provider_version=_SVD_PROVIDER_VERSION,
            # P7's own operating point. R4-T1 later measured a different one; that
            # measurement does not apply to a decision taken before it existed.
            threshold=0.98,
            count_key=_SVD_COUNT_KEY,
        ),
    ),
    rules={
        "R010": (
            "No calibrated evidence was available: the synthetic-video signal was absent, "
            "the detector did not report, or the deployment was not the calibrated one."
        ),
        "R012": (
            "The calibrated synthetic-video detector reported, but with figures that could "
            "not be read as a calibrated probability over a non-empty aggregate."
        ),
        "R100": "The calibrated synthetic-video score reached its measured threshold.",
        "R200": (
            "The calibrated synthetic-video score was readable and did not reach its "
            "threshold; the single available signal did not support a classification."
        ),
    },
)

RULESET_V2 = Ruleset(
    rules_version="r4-v2.0.0",
    calibration_id="cab2ea262bb7e41cb87e49bdb3dad53ecd0f02248035a993f9fcb033363afd1e",
    signals=(
        CalibratedSignal(
            signal_type=_SVD_SIGNAL_TYPE,
            provider=_SVD_PROVIDER,
            provider_version=_SVD_PROVIDER_VERSION,
            threshold=0.9550971388816833,
            count_key=_SVD_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_FACE_SIGNAL_TYPE,
            provider=_FACE_PROVIDER,
            provider_version=_FACE_PROVIDER_VERSION,
            threshold=0.9867589175701141,
            count_key=_FACE_COUNT_KEY,
        ),
    ),
    rules={
        "R010": (
            "No calibrated evidence was available from either detector: both signals were "
            "absent, did not report, or came from uncalibrated deployments."
        ),
        "R012": (
            "At least one calibrated detector reported with figures that could not be read."
        ),
        "R100": "The calibrated synthetic-video score reached its measured threshold.",
        "R101": "The calibrated face-manipulation score reached its measured threshold.",
        "R102": "Both calibrated detectors independently reached their measured thresholds.",
        "R200": (
            "Both calibrated detectors were readable and neither reached its threshold; the "
            "evidence did not support a classification."
        ),
        "R201": (
            "Exactly one calibrated detector was readable and it did not reach its "
            "threshold; the other contributed no reading."
        ),
    },
)

RULESET_V3 = Ruleset(
    rules_version="r5-v3.0.0",
    calibration_id="a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7",
    signals=(
        CalibratedSignal(
            signal_type=_SVD_SIGNAL_TYPE,
            provider=_SVD_PROVIDER,
            provider_version=_SVD_PROVIDER_VERSION,
            threshold=0.9550971388816833,
            count_key=_SVD_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_FACE_SIGNAL_TYPE,
            provider=_FACE_PROVIDER,
            provider_version=_FACE_PROVIDER_VERSION,
            threshold=0.9867589175701141,
            count_key=_FACE_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_LIP_SIGNAL_TYPE,
            provider=_LIP_PROVIDER,
            provider_version=_LIP_PROVIDER_VERSION,
            threshold=0.22962537594139576,
            count_key=_LIP_COUNT_KEY,
        ),
    ),
    rules={
        "R010": (
            "No calibrated evidence was available from any of the three detectors: every "
            "signal was absent, did not report, or came from an uncalibrated deployment."
        ),
        "R012": (
            "At least one calibrated detector reported with figures that could not be read."
        ),
        "R100": "The calibrated synthetic-video score reached its measured threshold.",
        "R101": "The calibrated face-manipulation score reached its measured threshold.",
        "R102": (
            "Two or more calibrated detectors independently reached their measured "
            "thresholds. The level is not raised by the agreement; there is no band above "
            "HIGH and no measurement that says two flags mean more than one."
        ),
        "R103": "The calibrated mouth-dynamics score reached its measured threshold.",
        "R200": (
            "All three calibrated detectors were readable and none reached its threshold; "
            "the evidence did not support a classification."
        ),
        "R201": (
            "At least one calibrated detector was readable and did not reach its threshold, "
            "while the remaining detectors contributed no reading."
        ),
    },
)

# The ruleset in force. It reads the same three detectors as v3, against the same three
# thresholds, under the same calibration identity — R7-T5 changed no measurement and adopted no
# artifact, so the id it was taken under is the id v3 was taken under, and a v4 row resolves
# against exactly the artifacts it rests on. What changed is the rules: `R103` is gone, and the
# mouth-dynamics detector is carried here as non-decisional evidence.
#
# v3 above is untouched by any of this, and must stay untouched. The two entries deliberately
# share `calibration_id` and differ in `rules_version`, which is the whole reason a decision
# persists both.
RULESET_V4 = Ruleset(
    rules_version="r7-v4.0.0",
    calibration_id="a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7",
    signals=(
        CalibratedSignal(
            signal_type=_SVD_SIGNAL_TYPE,
            provider=_SVD_PROVIDER,
            provider_version=_SVD_PROVIDER_VERSION,
            threshold=0.9550971388816833,
            count_key=_SVD_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_FACE_SIGNAL_TYPE,
            provider=_FACE_PROVIDER,
            provider_version=_FACE_PROVIDER_VERSION,
            threshold=0.9867589175701141,
            count_key=_FACE_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_LIP_SIGNAL_TYPE,
            provider=_LIP_PROVIDER,
            provider_version=_LIP_PROVIDER_VERSION,
            # R5-T3's operating point, unchanged and still the honest thing to band a stored
            # score against: it is where this detector's own study put it. What v4 withdrew is
            # the rule that concluded from the crossing, not the crossing's meaning as evidence.
            threshold=0.22962537594139576,
            count_key=_LIP_COUNT_KEY,
            decisional=False,
        ),
    ),
    rules={
        "R010": (
            "No calibrated evidence was available from any of the three detectors: every "
            "signal was absent, did not report, or came from an uncalibrated deployment."
        ),
        "R012": (
            "At least one calibrated detector reported with figures that could not be read."
        ),
        "R100": "The calibrated synthetic-video score reached its measured threshold.",
        "R101": "The calibrated face-manipulation score reached its measured threshold.",
        "R102": (
            "Both detectors this ruleset takes a decision from — synthetic-video and "
            "face-manipulation — independently reached their measured thresholds. The level is "
            "not raised by the agreement; there is no band above HIGH and no measurement that "
            "says two flags mean more than one."
        ),
        "R200": (
            "All three calibrated detectors were readable and neither detector this ruleset "
            "takes a decision from reached its threshold; the evidence did not support a "
            "classification. The mouth-dynamics score is reported beside its measured "
            "threshold as independent evidence and does not decide this level, whether or not "
            "it reached it (R7-T5)."
        ),
        "R201": (
            "At least one calibrated detector was readable and neither detector this ruleset "
            "takes a decision from reached its threshold, while the remaining detectors "
            "contributed no reading. The mouth-dynamics score is reported beside its measured "
            "threshold as independent evidence and does not decide this level, whether or not "
            "it reached it (R7-T5)."
        ),
    },
)


# --- `r9-v5.0.0`: the R9 verdict vocabulary, and the first version that counts coverage ------
#
# Added beside v4, never over it. Every entry above is untouched, and a decision persisted under
# `r7-v4.0.0` is still explained by v4's sentences against v4's thresholds — the two versions
# share a calibration identity and differ in `rules_version`, exactly as v3 and v4 already do.
#
# The detectors, their deployments and their operating points are transcribed from v4 unchanged,
# because R9-T2 moved no threshold: the same two detectors decide on the same two measured
# points, and the mouth-dynamics detector is `decisional=False` here as it is there. What is new
# is the vocabulary the rules conclude in and the coverage statement beside it.
#
# One thing is worth saying plainly, because the table cannot: under v5 the engine does not read
# the mouth-dynamics signal at all — it is outside the coverage arithmetic rather than a zero
# inside it, and its failure, absence or abstention removes no coverage and changes no verdict.
# It is listed here anyway, and banded against R5-T3's operating point, for the reason v4 lists
# it: a stored score a reader can see is better explained beside the point its own study
# measured than left unexplained. It lands in `supplementary_evidence`, never in
# `decision_eligible_detectors`, and `decision_coverage` never counts it.
RULESET_V5 = Ruleset(
    rules_version="r9-v5.0.0",
    # Unchanged from v4, deliberately. This column names the *measurements* a decision was taken
    # under and v5 adopted no artifact and dropped none; minting an identity here would assert a
    # measurement nobody took and leave every v5 decision unresolvable against the artifacts it
    # actually rests on.
    calibration_id="a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7",
    signals=(
        CalibratedSignal(
            signal_type=_SVD_SIGNAL_TYPE,
            provider=_SVD_PROVIDER,
            provider_version=_SVD_PROVIDER_VERSION,
            threshold=0.9550971388816833,
            count_key=_SVD_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_FACE_SIGNAL_TYPE,
            provider=_FACE_PROVIDER,
            provider_version=_FACE_PROVIDER_VERSION,
            threshold=0.9867589175701141,
            count_key=_FACE_COUNT_KEY,
        ),
        CalibratedSignal(
            signal_type=_LIP_SIGNAL_TYPE,
            provider=_LIP_PROVIDER,
            provider_version=_LIP_PROVIDER_VERSION,
            threshold=0.22962537594139576,
            count_key=_LIP_COUNT_KEY,
            decisional=False,
        ),
    ),
    # Two, frozen as a literal by this version and checked against the table above at import.
    # Not a count of the rows an analysis happens to carry: if the denominator came from the
    # evidence present, a detector that was never invoked would *improve* coverage by being
    # absent, which is the most dangerous defect a coverage model can have.
    decision_total=2,
    decisive_level=VERDICT_MANIPULATION_DETECTED,
    # The v5 rule ids are disjoint from every id above, by construction and not by coincidence.
    # An `R100` was stored against `r7-v4.0.0` and `R9-100` never can be, so no sentence here can
    # rewrite what a historical row said — the same reasoning that keeps `R103` retired.
    rules={
        "R9-100": (
            "Both detectors this ruleset takes a decision from — synthetic-video and "
            "face-manipulation — produced usable readings and independently reached their "
            "measured thresholds. The verdict is not strengthened by the agreement; there is "
            "no verdict above MANIPULATION_DETECTED and no measurement that says two "
            "detectors reaching their own thresholds mean more than one."
        ),
        "R9-101": (
            "The calibrated synthetic-video detector produced a usable reading and reached its "
            "measured threshold. The verdict stands whatever the other decision-eligible "
            "detector did: a detector that failed, abstained or read below its own threshold "
            "carries no information about the manipulation family this one is calibrated for, "
            "and any coverage it left incomplete is reported beside the verdict rather than "
            "folded into it."
        ),
        "R9-102": (
            "The calibrated face-manipulation detector produced a usable reading and reached "
            "its measured threshold. The verdict stands whatever the other decision-eligible "
            "detector did, on the same terms and for the same measured reason."
        ),
        "R9-200": (
            "Both detectors this ruleset takes a decision from produced usable readings and "
            "neither reached its measured threshold. This is not a finding that the media is "
            "authentic: a detector reports a score below its threshold for a manipulation "
            "family it is blind to as readily as for genuine media. The mouth-dynamics score "
            "is reported beside its measured threshold as independent evidence; it decides "
            "nothing here and is counted in no coverage, whether or not it reached it."
        ),
        "R9-300": (
            "Some but not all of the expected decision coverage was obtained, and no detector "
            "that did produce a usable reading reached its measured threshold. The assessment "
            "could not be completed; nothing about the media follows from that."
        ),
        "R9-301": (
            "None of the expected decision coverage was obtained: neither detector this "
            "ruleset takes a decision from produced a usable reading. The assessment could "
            "not be completed; nothing about the media follows from that."
        ),
    },
)


# Keyed by the string the decision persisted. A version absent from this table is a version
# this module cannot explain, and it says so rather than reaching for the nearest one.
RULESETS: dict[str, Ruleset] = {
    RULESET_V1.rules_version: RULESET_V1,
    RULESET_V2.rules_version: RULESET_V2,
    RULESET_V3.rules_version: RULESET_V3,
    RULESET_V4.rules_version: RULESET_V4,
    RULESET_V5.rules_version: RULESET_V5,
}


@dataclass(frozen=True)
class PersistedSignal:
    """One detector row as the database holds it, with nothing interpreted yet.

    `metadata` is the signal's JSON document, passed through whole rather than picked apart
    by the caller: which figure a detector had to report a positive count of is a property of
    the ruleset, so the ruleset's `count_key` is what reads it. It is typed `object` because
    JSON guarantees nothing about its shape — a string where a count belongs must degrade to
    unreadable figures, not raise.
    """

    provider: str | None
    signal_type: str | None
    status: str | None
    provider_version: str | None
    score: float | None
    metadata: object = None


@dataclass(frozen=True)
class SignalContribution:
    """What one detector contributed to a decision that has already been taken.

    Every field is either copied from the persisted row or read off the frozen ruleset the
    decision names. Nothing here is recomputed from current configuration, and `condition`
    is a statement about a number and a threshold — never about the media.
    """

    signal: str
    provider: str
    # The deployment that answered, as persisted. Null when nothing answered at all.
    provider_version: str | None
    # The provider's own figure, exactly as stored, when the persisted decision could use
    # it. Null when there is none — a missing score is never rendered as 0.0, which would be
    # a fabricated reading — and null on every `unavailable` contribution, where showing a
    # number beside a threshold would imply a comparison the engine never made.
    score: float | None
    # The operating point measured for this detector under *this* ruleset version. Null when
    # it could not be resolved, and null on every `unavailable` contribution, for the same
    # reason the score is: this detector's reading was not read against anything.
    threshold: float | None
    condition: str
    # Set only when `condition` is `unavailable` or `not_interpreted`, and read as the reason
    # this detector contributed nothing. Never a finding about the media.
    unavailable_reason: str | None
    role: str


@dataclass(frozen=True)
class DecisionCoverage:
    """How much of the decision coverage a ruleset version expected was actually obtained.

    `D_usable / D_total` of R9-T1 section 2.3, restated for a reader of an already-decided
    analysis. Both numbers are about the *decision-eligible* detectors only; an evidence-only
    detector is outside this arithmetic rather than a zero inside it, so nothing it did — or
    failed to do — moves either number (R9-T1 invariant 1).

    `total` is the version's frozen `decision_total`, never a count of the rows this analysis
    happens to carry. A detector that was never invoked stays in the denominator and shows up as
    missing coverage instead of vanishing from it.

    `usable` counts the decision-eligible detectors that produced a `usable_reading` — the
    calibrated deployment answered successfully *and* its figures can be read against that
    version's own operating point. `SUCCESS` alone is not enough and never has been: a row from
    an uncalibrated deployment, a score that is not a probability, and a mean taken over zero
    units all carry that status and none of them is a reading.

    This is a count of readings, not of findings. A detector that read below its threshold is as
    usable as one that read above it, and neither says anything here about the media.
    """

    usable: int
    total: int


@dataclass(frozen=True)
class RiskTrace:
    """The persisted decision, plus how each detector in scope stood when it was taken.

    The first four fields are the decision itself, copied from the analysis row without
    alteration; they are the source of truth and this object never contradicts them.
    `rule_summary`, `contributions` and the three R9 fields below are the derived part.

    `decision_eligible_detectors` and `supplementary_evidence` partition `contributions` by the
    `decisional` flag the *persisted* version set on each detector, so the split is that
    version's own and not today's: the mouth-dynamics detector is decision-eligible in a
    `r5-v3.0.0` trace and supplementary in a `r7-v4.0.0` one, with the same score on both sides.
    Every contribution appears in exactly one of the two, and the union is `contributions`
    unchanged — they exist so a consumer never has to reconstruct which detectors could decide,
    which is the reconstruction R9-T4 is here to make unnecessary.
    """

    risk_level: str
    rule_id: str | None
    rules_version: str | None
    calibration_id: str | None
    # What the fired rule meant *under the persisted ruleset version*. Null when the version
    # or the rule is not one this module knows — an old decision is left unexplained rather
    # than explained wrongly.
    rule_summary: str | None
    contributions: tuple[SignalContribution, ...]
    # False when the ruleset version or its calibration identity could not be resolved, so a
    # reader can tell "no detector contributed" from "this trace could not be interpreted".
    interpreted: bool
    # `D_usable / D_total` for the decision-eligible detectors, and null whenever this version
    # cannot honestly state it: every ruleset before `r9-v5.0.0` predates the coverage model and
    # took its decisions without a denominator, and an uninterpreted trace has read nothing
    # against anything. Null is "this decision makes no coverage claim", which is a different
    # fact from "coverage was zero" and must never be rendered as one.
    decision_coverage: DecisionCoverage | None = None
    # `contributions`, partitioned by the persisted version's own `decisional` flag. Both are
    # populated for every interpretable version, legacy included, because the flag is already
    # frozen per version and says exactly what that version's rules could decide from.
    decision_eligible_detectors: tuple[SignalContribution, ...] = ()
    supplementary_evidence: tuple[SignalContribution, ...] = ()


def _is_calibrated_probability(score: object) -> bool:
    """Whether a persisted score is a figure a threshold comparison can be made against.

    The same three refusals the engine made when it took the decision, restated because this
    module does not import it: `None` and `bool` are not scores, `NaN` compares false against
    every threshold, and anything outside [0, 1] cannot have come from the calibrated head.
    """
    if score is None or isinstance(score, bool):
        return False
    if not isinstance(score, float | int):
        return False
    if not math.isfinite(score):
        return False
    return SCORE_FLOOR <= score <= SCORE_CEILING


def _is_positive_count(count: object) -> bool:
    """Whether the provider stated it actually aggregated something."""
    if count is None or isinstance(count, bool):
        return False
    if not isinstance(count, int):
        return False
    return count > 0


def _find(
    signals: dict[str, PersistedSignal] | None, signal_type: str
) -> PersistedSignal | None:
    if not signals:
        return None
    return signals.get(signal_type)


def _contribution(
    calibrated: CalibratedSignal,
    persisted: PersistedSignal | None,
    threshold: float | None,
    risk_level: str,
    decisive_level: str,
) -> SignalContribution:
    """Read one detector's persisted row against one historical threshold.

    The order of the checks is the order in which a reading can fail to be a reading, and
    each failure is reported as its own reason: no row at all, a row from a detector that did
    not report, a row from a deployment the operating point was never measured on, and a row
    whose figures cannot be read. All four are `unavailable`. None of them is turned into a
    score, a threshold comparison, or any claim about the media.
    """

    def unavailable(reason: str) -> SignalContribution:
        """A detector that contributed no usable reading, with the figures withheld.

        `score` and `threshold` are both null here, whatever the signal row happens to hold.
        The trace describes the evidence the persisted decision could actually use, and a
        number shown beside an operating point reads as a comparison that was made — which
        is exactly what did not happen: the engine refused this reading and classified
        without it. The raw persisted value is still available in full on the analysis's own
        signal evidence, where it is a provider's output rather than a term in a decision.

        What remains is what is true: which detector, which signal, which deployment
        answered if one did, that it was unavailable, and why.
        """
        return SignalContribution(
            signal=calibrated.signal_type,
            provider=calibrated.provider,
            provider_version=persisted.provider_version if persisted else None,
            score=None,
            threshold=None,
            condition=CONDITION_UNAVAILABLE,
            unavailable_reason=reason,
            role=ROLE_CONSIDERED,
        )

    if persisted is None:
        return unavailable(UNAVAILABLE_NO_READING)

    if persisted.status != SIGNAL_STATUS_SUCCESS:
        # Failed, timed out, abstained — the detector produced no number. Silence, and
        # silence is not evidence of anything about the media.
        return unavailable(UNAVAILABLE_DETECTOR_DID_NOT_REPORT)

    if (
        persisted.provider != calibrated.provider
        or persisted.signal_type != calibrated.signal_type
        or persisted.provider_version != calibrated.provider_version
    ):
        # A different deployment than the one the threshold was measured on. The engine
        # refused to read it and so does this.
        return unavailable(UNAVAILABLE_UNCALIBRATED_DEPLOYMENT)

    metadata = persisted.metadata if isinstance(persisted.metadata, dict) else {}
    if not _is_calibrated_probability(persisted.score) or not _is_positive_count(
        metadata.get(calibrated.count_key)
    ):
        return unavailable(UNAVAILABLE_UNREADABLE_FIGURES)

    if threshold is None:
        # The figures are readable but the historical operating point is not resolvable, so
        # there is nothing honest to compare them with. Substituting today's threshold here
        # is exactly the silent reinterpretation this module exists to prevent.
        return SignalContribution(
            signal=calibrated.signal_type,
            provider=calibrated.provider,
            provider_version=persisted.provider_version,
            score=persisted.score,
            threshold=None,
            condition=CONDITION_NOT_INTERPRETED,
            unavailable_reason=UNAVAILABLE_THRESHOLD_UNRESOLVED,
            role=ROLE_CONSIDERED,
        )

    reached = persisted.score >= threshold  # type: ignore[operator]
    return SignalContribution(
        signal=calibrated.signal_type,
        provider=calibrated.provider,
        provider_version=persisted.provider_version,
        score=persisted.score,
        threshold=threshold,
        condition=(
            CONDITION_THRESHOLD_REACHED if reached else CONDITION_THRESHOLD_NOT_REACHED
        ),
        # A threshold comparison was made, so there is no reason for absence to report.
        unavailable_reason=None,
        # The HIGH rules are disjunctive and name one detector's own finding, so a detector
        # that reached its threshold under a HIGH decision is a reason for that level — provided
        # this version's rules could take a decision from it at all. Under `r7-v4.0.0` the
        # mouth-dynamics detector cannot, so a crossing of its threshold alongside a HIGH taken
        # from another detector is `considered`: the level was not reached on its evidence, and
        # `decisive` beside a score that could not have decided would be a false attribution.
        # Under any conclusion other than that one, no detector's threshold produced the
        # decision and nothing here is decisive.
        #
        # `decisive_level` rather than the literal `HIGH`, because v5 concludes in a different
        # vocabulary and names the same position `MANIPULATION_DETECTED`. Comparing a v5 verdict
        # against `HIGH` would report the very detector that produced it as merely `considered`,
        # which is the misattribution this field exists to prevent — in the opposite direction
        # from the one `decisional` guards.
        role=(
            ROLE_DECISIVE
            if reached and risk_level == decisive_level and calibrated.decisional
            else ROLE_CONSIDERED
        ),
    )


def _coverage(
    ruleset: Ruleset,
    decision_eligible: tuple[SignalContribution, ...],
    thresholds_resolved: bool,
) -> DecisionCoverage | None:
    """State how much of a version's expected decision coverage this analysis obtained.

    Null in the two cases where there is no honest statement to make, and both are silences
    rather than zeros:

    * the version declares no `decision_total` — every ruleset before `r9-v5.0.0`. Those
      decisions were taken without a denominator and none of their rules counted one, so a
      fraction here would be a coverage claim the decision never made. R9-T1 forbids retrofitting
      the coverage model onto them and this is where that refusal is enforced;
    * the thresholds could not be resolved, which means no contribution was read against an
      operating point at all. Every detector is `not_interpreted`, so a count of usable readings
      would be `0` — indistinguishable from the genuinely uncovered analysis that `R9-301`
      describes, and a far stronger statement than "this build could not interpret the trace".

    `usable` counts `USABLE_READING_CONDITIONS` over the decision-eligible contributions and
    nothing else, which is what keeps the one definition of usability in this module. Those
    conditions are set in `_contribution` by exactly the checks the engine made before it would
    read a score: a row exists, the detector reported `SUCCESS`, the deployment is the one the
    operating point was measured on, the score is a probability, and the provider's own count
    says it aggregated something. A `FAILED` or `TIMEOUT` row, an abstention, a missing row, an
    uncalibrated deployment and unreadable figures are all `unavailable` there, so none of them
    can be counted here — `SUCCESS` is necessary and was never sufficient.

    `total` is the frozen literal off the version, never `len(decision_eligible)`. The two agree
    by construction — `Ruleset.__post_init__` refuses a version where they do not — and the
    literal is still what is reported, because a denominator that counted the detectors present
    would be a different and much weaker guarantee.
    """
    if ruleset.decision_total is None or not thresholds_resolved:
        return None

    usable = sum(
        1
        for contribution in decision_eligible
        if contribution.condition in USABLE_READING_CONDITIONS
    )

    return DecisionCoverage(usable=usable, total=ruleset.decision_total)


def build_trace(
    *,
    risk_level: str | None,
    rule_id: str | None,
    rules_version: str | None,
    calibration_id: str | None,
    signals: dict[str, PersistedSignal] | None = None,
) -> RiskTrace | None:
    """Assemble the derived trace for one already-decided analysis.

    Returns `None` when `risk_level` is null, which is the absence of a decision — an
    analysis still queued or being worked on, or one completed before the engine existed.
    That is not `UNKNOWN`, which is a decision with a rule behind it and gets a trace like
    any other.

    Every level, rule id, ruleset version and calibration id in the result is the persisted
    one. Nothing here can produce them, and `app.risk_engine.evaluate` is never called — the
    module is not even imported.

    That holds for the R9 fields too, and it is the point of them. `decision_coverage` is read
    off the same persisted evidence the decision was taken from, under the same version's frozen
    identities and thresholds, so it *explains* the stored verdict rather than checking it: a
    coverage of 1/2 beside a `MANIPULATION_DETECTED` is the engine's own "a hit is never
    softened" rule showing through, not a disagreement. Nothing here compares the coverage it
    computed against the verdict it was given, and nothing here would change the verdict if the
    two ever looked odd together — an inconsistency in the record is something a reader must be
    able to see, and a trace that quietly corrected it would be the one place the record could
    be rewritten without anyone noticing.
    """
    if risk_level is None:
        return None

    ruleset = RULESETS.get(rules_version) if rules_version else None

    if ruleset is None:
        # A ruleset this build does not know. The decision is still reported in full; what is
        # withheld is the interpretation, because there is none to give that would not be a
        # guess about what those rules meant — including which of its detectors could decide
        # and how much coverage it expected, which is why all three R9 fields are empty here.
        return RiskTrace(
            risk_level=risk_level,
            rule_id=rule_id,
            rules_version=rules_version,
            calibration_id=calibration_id,
            rule_summary=None,
            contributions=(),
            interpreted=False,
            decision_coverage=None,
            decision_eligible_detectors=(),
            supplementary_evidence=(),
        )

    # The thresholds are only this version's if the decision was taken under this version's
    # calibration. A row naming a calibration identity that is not the one this ruleset was
    # measured under gets its detectors listed with no threshold rather than with numbers
    # from a measurement it did not use.
    thresholds_resolved = calibration_id == ruleset.calibration_id

    read = tuple(
        (
            calibrated,
            _contribution(
                calibrated,
                _find(signals, calibrated.signal_type),
                calibrated.threshold if thresholds_resolved else None,
                risk_level,
                ruleset.decisive_level,
            ),
        )
        for calibrated in ruleset.signals
    )

    contributions = tuple(contribution for _, contribution in read)
    decision_eligible = tuple(
        contribution for calibrated, contribution in read if calibrated.decisional
    )
    supplementary = tuple(
        contribution for calibrated, contribution in read if not calibrated.decisional
    )

    return RiskTrace(
        risk_level=risk_level,
        rule_id=rule_id,
        rules_version=rules_version,
        calibration_id=calibration_id,
        rule_summary=ruleset.rules.get(rule_id) if rule_id else None,
        contributions=contributions,
        interpreted=thresholds_resolved,
        decision_coverage=_coverage(ruleset, decision_eligible, thresholds_resolved),
        decision_eligible_detectors=decision_eligible,
        supplementary_evidence=supplementary,
    )
