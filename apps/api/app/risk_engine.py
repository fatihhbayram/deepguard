"""DeepGuard's own classification of one analysis, from one calibrated direct-risk signal.

This is the first layer in the codebase entitled to say anything about what a detector's
number *means*. Everything below it — `detection.py`, the provider integrations, the
segment rows — records what a provider said and refuses to interpret it. Here that
interpretation happens, once, against thresholds that were measured rather than chosen
(P7-T2), and the decision that comes out carries the identity of the measurement it was
made under so a stored verdict stays readable years after the thresholds move.

Stateless and deterministic. It opens no connection, reads no configuration, consults no
clock and holds nothing between calls: the same evidence always yields the same decision,
which is what makes a persisted decision reproducible from the persisted evidence.

**One signal decides, and only one.** The v1 rules read NVIDIA's `synthetic_video` score
and nothing else. C2PA provenance, the active-speaker timeline and AASIST's audio windows
are all persisted as independent forensic evidence and none of them can move this
classification by a single band — they are not averaged in, not weighted, not used as a
tie-break, not even read. That is rule 11 of AGENTS.md applied to the one place where the
temptation to combine is strongest, and it is what keeps HIGH meaning "a validated
direct-risk rule fired" rather than "several unrelated numbers looked bad together".

**LOW does not exist in v1.** P7-T2 measured `T_LOW = 0.05` and it is part of the
calibration artifact, but it is deliberately not activated as a boundary here: at that
operating point only 9.1 % of genuine media would earn LOW, while a face-swapped video is
*more* likely to fall below it than a genuine broadcast is (P7-T2 §5.3, §9.3). A band that
rare and that badly behaved on the manipulation families is not worth the reassurance
users would read into it. `T_LOW` is recorded in `CALIBRATION` because the calibration
identity has to match the artifact byte for byte, not because anything branches on it.
"""

import math
from dataclasses import dataclass

# The ruleset this module implements, as one immutable string. It is not derived from a
# package version or a git hash: it names *these rules*, and it changes when a rule,
# threshold or ordering changes — never when unrelated code around it does. A stored
# decision is only re-derivable if the version it names pins the logic exactly.
RULES_VERSION = "p7-v1.0.0"

# Identity of the measurement the thresholds below came from — the SHA-256 of the
# calibration artifact's identity fields (P7-T2 §11). Stored with every decision so a
# verdict can be traced to the corpus, the provider deployment and the error policy it was
# made under. A recalibration produces a different id, and old rows keep the old one.
CALIBRATION_ID = "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c"

# The single provider deployment these thresholds were measured against. NVIDIA exposes no
# model or weights version, so its NVCF function id is the only version handle there is
# (D017), and the calibration binds to this one exactly.
CALIBRATED_PROVIDER = "nvidia"
CALIBRATED_SIGNAL_TYPE = "synthetic_video"
CALIBRATED_PROVIDER_VERSION = "847b6e53-0133-452d-ab85-d7acf3ace723"

# Where HIGH begins. Measured, not chosen: it sits above the highest score any of 110
# genuine samples reached (0.9704) with margin, produced zero false HIGH across 11 sources
# and 8 languages, and the forced-normalization study moved the worst genuine sample away
# from it rather than toward it (P7-T2 §7.1, §F1.8).
T_HIGH = 0.98

# Measured and deliberately inert — see the module docstring. Nothing in this file compares
# against it.
T_LOW = 0.05

# The lower end of the provider's own scale. `probability = expit(logit)` cannot leave
# [0, 1], so a score outside it is not a low-risk reading — it is evidence that something
# other than the calibrated provider produced the row.
SCORE_FLOOR = 0.0
SCORE_CEILING = 1.0

# What this engine may conclude. LOW is absent by design and is never emitted.
RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_UNKNOWN = "UNKNOWN"

# The rules, in the order they are tried. The id is persisted with the decision, so reading
# a stored row tells you not only what was concluded but which sentence concluded it.
RULE_UNVALIDATED_PROVIDER = "R010"
RULE_INVALID_DIRECT_EVIDENCE = "R012"
RULE_CALIBRATED_HIGH = "R100"
RULE_INDETERMINATE_BAND = "R200"

# The provider status that makes a signal eligible at all. A detector that refused or timed
# out produced no reading, and a stale score sitting beside a failed status is not evidence.
SIGNAL_STATUS_SUCCESS = "SUCCESS"


class RiskEngineError(Exception):
    """The rules failed to classify evidence they are supposed to cover exhaustively.

    Unreachable by construction: `R012` leaves only a finite score in [0, 1], and `R100`
    and `R200` partition exactly that interval. It exists so the impossible case is a loud
    defect rather than a quiet default classification — inventing a fall-through band would
    let a bug in the rules above ship as a verdict about someone's media.
    """


@dataclass(frozen=True)
class SvdEvidence:
    """The persisted NVIDIA synthetic-video signal, as the database holds it.

    Built from a row that has been written, never from a value still in flight: the worker
    reads it back after persisting evidence so that what is classified is provably what a
    reader of the database will find behind the classification. A decision made from a
    pre-persistence value could disagree with the stored evidence and nothing would say so.

    Every field is optional or nullable exactly where the column is, because absence is one
    of the things the rules decide on. `total_clips` comes from the signal's metadata rather
    than a column of its own; a signal whose metadata never carried it arrives here as None
    and is treated as the degenerate evidence it is.
    """

    provider: str | None
    signal_type: str | None
    status: str | None
    provider_version: str | None
    score: float | None
    total_clips: int | None


@dataclass(frozen=True)
class RiskDecision:
    """One classification and the full trace of how it was reached.

    All four fields are persisted together. The level on its own would be unreadable in a
    year: it does not say which rules produced it, which measurement the thresholds came
    from, or which sentence fired. These four do, and they are what makes a historical
    decision explainable without re-running anything.
    """

    risk_level: str
    rules_version: str
    calibration_id: str
    rule_id: str


def _decision(risk_level: str, rule_id: str) -> RiskDecision:
    """Stamp a conclusion with the identity of the rules and calibration behind it."""
    return RiskDecision(
        risk_level=risk_level,
        rules_version=RULES_VERSION,
        calibration_id=CALIBRATION_ID,
        rule_id=rule_id,
    )


def is_eligible(evidence: SvdEvidence | None) -> bool:
    """Whether this signal is the calibrated deployment answering successfully.

    Four things must hold, and the version check is exact string equality on purpose. A
    prefix, suffix or substring match would accept a function id that merely *contains* the
    validated one — `847b6e53-...-d7acf3ace723-preview`, or the same uuid embedded in a
    longer identifier — and those are different deployments. The operating point was
    selected against 0.0096 of margin over observed genuine media (P7-T2 §11); a redeployed
    model could move the distribution by more than that with nothing visible to say so, so
    anything that is not character-for-character this deployment is uncalibrated.
    """
    return (
        evidence is not None
        and evidence.provider == CALIBRATED_PROVIDER
        and evidence.signal_type == CALIBRATED_SIGNAL_TYPE
        and evidence.status == SIGNAL_STATUS_SUCCESS
        and evidence.provider_version == CALIBRATED_PROVIDER_VERSION
    )


def is_usable_score(evidence: SvdEvidence) -> bool:
    """Whether an eligible signal's figures can carry a classification at all.

    A null score is a provider that returned no number. A non-finite one — NaN or either
    infinity — compares false against every threshold, so letting it through would decide
    the band by which comparison happened to be written first. A score outside [0, 1] cannot
    have come from `expit`. And `total_clips <= 0` is the provider reporting that it
    aggregated nothing, which makes the aggregate a figure over an empty table rather than a
    reading of the video — this is the degeneracy `D_MIN` was speculatively guarding against,
    caught on the provider's own output instead of on a guess about duration (P7-T2 §8).

    The clip count is type-checked because it does not come from a column. It is one field
    of the signal's JSON metadata, so nothing in the schema guarantees it is a number at
    all, and comparing a string against zero would raise where the rules are supposed to
    conclude `UNKNOWN`. `bool` is excluded for the same reason it is a number in Python and
    not one in evidence: `True > 0` would quietly pass a count nobody counted.
    """
    if evidence.score is None or evidence.total_clips is None:
        return False

    if not isinstance(evidence.total_clips, int) or isinstance(evidence.total_clips, bool):
        return False

    if not isinstance(evidence.score, float | int) or isinstance(evidence.score, bool):
        return False

    if not math.isfinite(evidence.score):
        return False

    if not SCORE_FLOOR <= evidence.score <= SCORE_CEILING:
        return False

    return evidence.total_clips > 0


def evaluate(evidence: SvdEvidence | None) -> RiskDecision:
    """Classify one analysis from its persisted direct-risk evidence.

    The rules are tried in order and the first that fires decides. They are exhaustive by
    construction rather than by a catch-all: `R012` leaves only a finite score inside
    [0, 1], and `R100` (`>= 0.98`) and `R200` (`0.0 <= score < 0.98`) partition that
    interval with nothing between them and nothing outside. There is no default branch, and
    the impossible remainder raises instead of classifying.

    `UNKNOWN` is not a hedge and not a low-confidence HIGH. It is the honest statement that
    no validated rule could be applied — the provider was not the calibrated one, did not
    answer, or answered with figures that cannot be read — and it is deliberately reachable
    from more conditions than the other two bands, because the error policy behind these
    thresholds prefers admitting ignorance to asserting risk (P7-T2 §6).
    """
    if not is_eligible(evidence):
        return _decision(RISK_UNKNOWN, RULE_UNVALIDATED_PROVIDER)

    # Narrowed by `is_eligible`, which rejects None before anything reads a field.
    assert evidence is not None

    if not is_usable_score(evidence):
        return _decision(RISK_UNKNOWN, RULE_INVALID_DIRECT_EVIDENCE)

    if evidence.score >= T_HIGH:
        return _decision(RISK_HIGH, RULE_CALIBRATED_HIGH)

    if SCORE_FLOOR <= evidence.score < T_HIGH:
        return _decision(RISK_MEDIUM, RULE_INDETERMINATE_BAND)

    raise RiskEngineError(
        f"No rule in {RULES_VERSION} classified an eligible score; the rules are supposed "
        "to be exhaustive over the validated range."
    )
