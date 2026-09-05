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
# reached its own threshold under a rule that concluded HIGH — the rules are disjunctive, so a
# flagged detector is the reason for the level. Everything else was in scope of the ruleset and
# read by it, which is all `considered` claims.
ROLE_DECISIVE = "decisive"
ROLE_CONSIDERED = "considered"

RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CalibratedSignal:
    """One detector as a ruleset version knew it, frozen at that version's identity.

    `provider`, `signal_type` and `provider_version` together are what the engine required
    before it would read a score at all: a different deployment of the same model is an
    uncalibrated one, because the operating point was measured against that deployment's
    distribution and nothing else. `count_key` names the metadata figure the engine demanded
    be positive — the provider's own statement that it aggregated something.
    """

    signal_type: str
    provider: str
    provider_version: str
    threshold: float
    count_key: str


@dataclass(frozen=True)
class Ruleset:
    """One immutable ruleset version: which detectors it read, and what its rules meant.

    `rules` is why this table exists at all. `R200` under `p7-v1.0.0` is one detector's score
    sitting below its threshold; under `r4-v2.0.0` it is two detectors both below theirs;
    under `r5-v3.0.0` it is three. `R102` is "both detectors" in v2 and "two or more" in v3.
    The same four characters, three different statements — so an old decision read under
    today's meanings would be misreported while looking entirely well-formed.
    """

    rules_version: str
    calibration_id: str
    signals: tuple[CalibratedSignal, ...]
    rules: dict[str, str]


# NVIDIA's synthetic-video deployment, unchanged across all three rulesets. The threshold is
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

# Keyed by the string the decision persisted. A version absent from this table is a version
# this module cannot explain, and it says so rather than reaching for the nearest one.
RULESETS: dict[str, Ruleset] = {
    RULESET_V1.rules_version: RULESET_V1,
    RULESET_V2.rules_version: RULESET_V2,
    RULESET_V3.rules_version: RULESET_V3,
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
class RiskTrace:
    """The persisted decision, plus how each detector in scope stood when it was taken.

    The first four fields are the decision itself, copied from the analysis row without
    alteration; they are the source of truth and this object never contradicts them.
    `rule_summary` and `contributions` are the derived part.
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
        # that reached its threshold under a HIGH decision is a reason for that level. Under
        # any other level no detector reached one, and nothing here is decisive.
        role=(
            ROLE_DECISIVE if reached and risk_level == RISK_HIGH else ROLE_CONSIDERED
        ),
    )


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
    """
    if risk_level is None:
        return None

    ruleset = RULESETS.get(rules_version) if rules_version else None

    if ruleset is None:
        # A ruleset this build does not know. The decision is still reported in full; what is
        # withheld is the interpretation, because there is none to give that would not be a
        # guess about what those rules meant.
        return RiskTrace(
            risk_level=risk_level,
            rule_id=rule_id,
            rules_version=rules_version,
            calibration_id=calibration_id,
            rule_summary=None,
            contributions=(),
            interpreted=False,
        )

    # The thresholds are only this version's if the decision was taken under this version's
    # calibration. A row naming a calibration identity that is not the one this ruleset was
    # measured under gets its detectors listed with no threshold rather than with numbers
    # from a measurement it did not use.
    thresholds_resolved = calibration_id == ruleset.calibration_id

    contributions = tuple(
        _contribution(
            calibrated,
            _find(signals, calibrated.signal_type),
            calibrated.threshold if thresholds_resolved else None,
            risk_level,
        )
        for calibrated in ruleset.signals
    )

    return RiskTrace(
        risk_level=risk_level,
        rule_id=rule_id,
        rules_version=rules_version,
        calibration_id=calibration_id,
        rule_summary=ruleset.rules.get(rule_id) if rule_id else None,
        contributions=contributions,
        interpreted=thresholds_resolved,
    )
