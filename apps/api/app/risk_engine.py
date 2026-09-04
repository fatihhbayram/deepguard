"""DeepGuard's own classification of one analysis, from two independently calibrated signals.

This is the first layer in the codebase entitled to say anything about what a detector's
number *means*. Everything below it — `detection.py`, the provider integrations, the
segment rows — records what a provider said and refuses to interpret it. Here that
interpretation happens, once, against thresholds that were measured rather than chosen
(R4-T1), and the decision that comes out carries the identity of the measurement it was
made under so a stored verdict stays readable years after the thresholds move.

Stateless and deterministic. It opens no connection, reads no configuration, consults no
clock and holds nothing between calls: the same evidence always yields the same decision,
which is what makes a persisted decision reproducible from the persisted evidence.

**Two signals decide, and each decides alone.** Ruleset v2 reads NVIDIA's `synthetic_video`
score and EfficientNet-B7's `face_manipulation` score. Each is compared against its own
measured threshold and against nothing else. The two numbers never meet: they are not
averaged, not weighted, not summed, not multiplied, and neither is ever used to adjust or
discount the other. What the rules combine is the two detectors' *separate decisions* about
the same media, which is the only combination R4-T1 measured and the only one rule 11 of
AGENTS.md permits.

C2PA provenance, the active-speaker timeline, AASIST's audio windows and — since R5-T2 —
LipForensics' mouth-dynamics score remain persisted as independent forensic evidence and still
cannot move this classification by a single band. They are not read here at all: `evaluate`
takes two arguments, this module imports nothing that would fetch a third signal, and an
analysis carrying a mouth-dynamics row is classified exactly as the same analysis without one. They
have no calibration, and a scale is not a calibration.

The mouth-dynamics signal is the one that will change, and saying so here is not the same as leaving
room for it. R5-T1 measured the detector; R5-T3 is where an operating point for it is measured
and a rule that reads it is written, with its own threshold, its own eligibility check bound to
its own artifacts, and its own place in the ordering above. Until that measurement exists there
is nothing for a rule to compare against, and a third detector admitted on the strength of its
scale rather than its calibration is exactly what this module refuses.

**Why the rule is OR, and why silence never dampens a flag.** R4-T1 measured both detectors
over one 159-clip corpus and found them strictly complementary. NVIDIA's detector separates
generated video (AUROC 0.9205 against genuine) and is near-blind to face swaps (0.5422);
EfficientNet-B7 separates face swaps (0.9304) and is worse than chance on generated video
(0.3960). Their scores are *negatively* rank-correlated (Spearman -0.2781). At the two
selected operating points the detectors never once agreed: of 159 clips, `both_flagged` was
0, and every one of the 47 detections came from exactly one detector while the other sat
near its floor — `ffpp_dev_Deepfakes_106_198` scored 0.1648 on NVIDIA and 0.9943 on the face
model; `sonic_en_03` scored 0.9961 on NVIDIA and 0.0053 on the face model.

Two things follow, and they are the whole design of this module:

1. A rule requiring both detectors to agree would have detected nothing whatsoever on the
   calibration corpus. Agreement is not available evidence, so it cannot be required.
2. A detector scoring below its threshold is not asserting that the media is genuine. On the
   manipulation family it is blind to, that is exactly what it does anyway. So a low score
   carries no information capable of contradicting the other detector's finding, and it is
   never allowed to soften, downgrade or veto it.

There is therefore no tie-break, because there is no tie: `R100` and `R101` fire on their own
detector's evidence irrespective of what the other detector said or whether it said anything.
`R102` exists for the case the corpus never produced — both above threshold at once — so that
the trace can record it honestly rather than attributing a joint finding to one source.

**The false-HIGH budget is spent per detector, and that is deliberate.** Each threshold was
placed at the lowest point with zero observed false positives on the 54 genuine clips, giving
each a 95% upper bound of 0.0540 on its own false-positive rate. Reading them under OR means
the bound on the *combined* rule is not 0.0540; with no genuine clip flagged by either
detector the joint rate is still 0 observed over 54, but the honest upper bound for the
disjunction is looser than for either alone. That cost is accepted because the alternative —
raising both thresholds to buy back the difference — would forfeit face-swap coverage
entirely, and R4-T1's error policy gives up detection rate to protect genuine media only
where the trade is real. It is not real here: no genuine clip in the corpus came near either
boundary under either detector.

**LOW does not exist in v2 either.** R4-T1 measured `T_LOW` for both detectors and both are
useless as a reassurance: NVIDIA's would cover 1.85% of genuine media, EfficientNet-B7's the
same 1.85%. A band that rare is not worth the certainty a reader would take from it. Both are
recorded below because the calibration identity has to match the artifact, not because
anything branches on them.
"""

import math
from dataclasses import dataclass

# The ruleset this module implements, as one immutable string. It is not derived from a
# package version or a git hash: it names *these rules*, and it changes when a rule,
# threshold or ordering changes — never when unrelated code around it does. A stored
# decision is only re-derivable if the version it names pins the logic exactly.
RULES_VERSION = "r4-v2.0.0"

# Identity of the measurement the thresholds below came from — the SHA-256 of the R4-T1
# calibration artifact's identity fields. Stored with every decision so a verdict can be
# traced to the corpus, both detector deployments and the error policy it was made under. A
# recalibration produces a different id, and rows written under p7-v1.0.0 keep the old one.
CALIBRATION_ID = "cab2ea262bb7e41cb87e49bdb3dad53ecd0f02248035a993f9fcb033363afd1e"

# --- The calibrated synthetic-video signal -------------------------------------------------
#
# The single provider deployment these thresholds were measured against. NVIDIA exposes no
# model or weights version, so its NVCF function id is the only version handle there is
# (D017), and the calibration binds to this one exactly.
SVD_PROVIDER = "nvidia"
SVD_SIGNAL_TYPE = "synthetic_video"
SVD_PROVIDER_VERSION = "847b6e53-0133-452d-ab85-d7acf3ace723"

# Where HIGH begins for this detector. Measured, not chosen: the lowest threshold with zero
# false positives across 54 genuine clips, placed at the midpoint between the highest genuine
# score observed (0.9541) and the lowest score above it (0.9561). At that point it flags
# 54.55% of generated video, 0% of face swaps and 0% of genuine media.
SVD_T_HIGH = 0.9550971388816833

# Measured and deliberately inert — see the module docstring. Nothing compares against it.
SVD_T_LOW = 0.030267621390521526

# --- The calibrated face-manipulation signal -----------------------------------------------
#
# The exact classifier artifact the thresholds were measured against, in the form
# `app.detection` writes into `provider_version`: repository at revision. A different
# revision is a different measurement, so the check is exact string equality.
FACE_PROVIDER = "efficientnet-b7"
FACE_SIGNAL_TYPE = "face_manipulation"
FACE_PROVIDER_VERSION = (
    "tomas-gajarsky/facetorch-deepfake-efficientnet-b7@4acc494f37eb63d7457166eff2acb45c5b04b9a6"
)

# Where HIGH begins for this detector, derived by the same rule against the same 54 genuine
# clips: the midpoint between the highest genuine score (0.9865) and the lowest score above
# it (0.9870). At that point it flags 44% of face swaps, 0% of generated video and 0% of
# genuine media — the mirror image of the detector above, which is the point of reading both.
FACE_T_HIGH = 0.9867589175701141

# Measured and deliberately inert, for the same reason as `SVD_T_LOW`.
FACE_T_LOW = 0.0047953922767192125

# The lower and upper ends of both providers' scales. Both scores are probabilities — NVIDIA's
# is `expit(mean clip logit)`, the classifier's is a mean of sigmoids — so neither can leave
# [0, 1], and a score outside it is not a low-risk reading. It is evidence that something
# other than the calibrated detector produced the row.
SCORE_FLOOR = 0.0
SCORE_CEILING = 1.0

# What this engine may conclude. LOW is absent by design and is never emitted.
RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_UNKNOWN = "UNKNOWN"

# The rules, in the order they are tried. The id is persisted with the decision, so reading a
# stored row tells you not only what was concluded but which sentence concluded it, and — now
# that there are two sources — which source or sources concluded it.
#
# The two UNKNOWN ids are carried over from p7-v1.0.0 with their meanings intact: `R010` is
# "no validated reading was ever obtained", `R012` is "one was obtained and its figures could
# not be read". They are not sub-divided per detector; a nine-way cross-product of two
# detectors' failure modes would be a rule table nobody can hold in mind, and the per-signal
# rows already record exactly which detector failed and how.
RULE_NO_CALIBRATED_EVIDENCE = "R010"
RULE_INVALID_CALIBRATED_EVIDENCE = "R012"
RULE_HIGH_SYNTHETIC_VIDEO = "R100"
RULE_HIGH_FACE_MANIPULATION = "R101"
RULE_HIGH_BOTH_SOURCES = "R102"
RULE_INDETERMINATE_BOTH_SOURCES = "R200"
RULE_INDETERMINATE_SINGLE_SOURCE = "R201"

# The provider status that makes a signal eligible at all. A detector that refused or timed
# out produced no reading, and a stale score sitting beside a failed status is not evidence.
# A face classifier that found no face in the clip is a `FAILED` row and lands here too:
# abstention is not a finding of no manipulation, and R4-T1 kept the two distinct for the same
# reason (8 of its clips were abstentions, every one of them on generated video).
SIGNAL_STATUS_SUCCESS = "SUCCESS"


class RiskEngineError(Exception):
    """The rules failed to classify evidence they are supposed to cover exhaustively.

    Unreachable by construction — see `evaluate`, which enumerates the partition. It exists so
    the impossible case is a loud defect rather than a quiet default classification: inventing
    a fall-through band would let a bug in the rules above ship as a verdict about someone's
    media.
    """


@dataclass(frozen=True)
class SvdEvidence:
    """The persisted NVIDIA synthetic-video signal, as the database holds it.

    Built from a row that has been written, never from a value still in flight: the worker
    reads it back after persisting evidence so that what is classified is provably what a
    reader of the database will find behind the classification. A decision made from a
    pre-persistence value could disagree with the stored evidence and nothing would say so.

    Every field is optional or nullable exactly where the column is, because absence is one of
    the things the rules decide on. `total_clips` comes from the signal's metadata rather than
    a column of its own; a signal whose metadata never carried it arrives here as None and is
    treated as the degenerate evidence it is.
    """

    provider: str | None
    signal_type: str | None
    status: str | None
    provider_version: str | None
    score: float | None
    total_clips: int | None


@dataclass(frozen=True)
class FaceEvidence:
    """The persisted EfficientNet-B7 face-manipulation signal, as the database holds it.

    Deliberately a separate type from `SvdEvidence` rather than a shared one with a renamed
    count. The two detectors aggregate different things — NVIDIA over clips it cut itself,
    this classifier over face crops sampled from frames — and the field that says a reading
    is degenerate is named after what was actually counted. Collapsing them into one
    `unit_count` would make the rules read as though the two numbers were commensurable, which
    is the exact confusion this module exists to prevent.

    `frames_scored` comes from the signal's metadata, where `app.detection` records how many
    face crops the mean was actually taken over.
    """

    provider: str | None
    signal_type: str | None
    status: str | None
    provider_version: str | None
    score: float | None
    frames_scored: int | None


@dataclass(frozen=True)
class RiskDecision:
    """One classification and the full trace of how it was reached.

    All four fields are persisted together. The level on its own would be unreadable in a
    year: it does not say which rules produced it, which measurement the thresholds came from,
    or which sentence fired. These four do, and they are what makes a historical decision
    explainable without re-running anything.

    There is deliberately no free-text reason field. `rule_id` names the sentence that fired
    and the sentence is fixed by `rules_version`, so the rationale is derivable from the trace
    rather than duplicated beside it — a stored prose reason could drift from the rule that
    actually decided, and then the row would contradict itself with nothing to say which half
    was right.
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


def _is_calibrated_probability(score: object) -> bool:
    """Whether a stored score can be compared against a calibrated threshold at all.

    Shared by both detectors because it is one question about one kind of value — a
    probability that came off a sigmoid — and not a shared abstraction over the detectors
    themselves. Their eligibility, their thresholds and their degeneracy counts all stay
    separate.

    A null score is a detector that returned no number. A non-finite one — NaN or either
    infinity — compares false against every threshold, so letting it through would decide the
    band by which comparison happened to be written first. A score outside [0, 1] cannot have
    come from a sigmoid. And `bool` is excluded for the same reason it is a number in Python
    and not one in evidence: `True >= 0.98` would quietly pass a score nobody measured.
    """
    if score is None or isinstance(score, bool):
        return False

    if not isinstance(score, float | int):
        return False

    if not math.isfinite(score):
        return False

    return SCORE_FLOOR <= score <= SCORE_CEILING


def _is_positive_count(count: object) -> bool:
    """Whether a metadata count reports that the detector actually aggregated something.

    Type-checked because it does not come from a column. It is one field of the signal's JSON
    metadata, so nothing in the schema guarantees it is a number at all, and comparing a
    string against zero would raise where the rules are supposed to conclude `UNKNOWN`. `bool`
    is excluded so `True > 0` cannot pass a count nobody counted.
    """
    if count is None or isinstance(count, bool):
        return False

    if not isinstance(count, int):
        return False

    return count > 0


def is_eligible_svd(evidence: SvdEvidence | None) -> bool:
    """Whether the synthetic-video signal is the calibrated deployment answering successfully.

    Four things must hold, and the version check is exact string equality on purpose. A
    prefix, suffix or substring match would accept a function id that merely *contains* the
    validated one — `847b6e53-...-d7acf3ace723-preview`, or the same uuid embedded in a longer
    identifier — and those are different deployments. The operating point clears the highest
    genuine score observed by 0.0010; a redeployed model could move the distribution by more
    than that with nothing visible to say so, so anything that is not character-for-character
    this deployment is uncalibrated.
    """
    return (
        evidence is not None
        and evidence.provider == SVD_PROVIDER
        and evidence.signal_type == SVD_SIGNAL_TYPE
        and evidence.status == SIGNAL_STATUS_SUCCESS
        and evidence.provider_version == SVD_PROVIDER_VERSION
    )


def is_eligible_face(evidence: FaceEvidence | None) -> bool:
    """Whether the face signal is the calibrated classifier artifact answering successfully.

    The same four checks, against this detector's own identity. Exact equality matters at
    least as much here: this threshold clears the highest genuine score by 0.0003, so a
    different revision of the weights is not a small difference in a measurement — it is a
    different measurement with no operating point of its own.
    """
    return (
        evidence is not None
        and evidence.provider == FACE_PROVIDER
        and evidence.signal_type == FACE_SIGNAL_TYPE
        and evidence.status == SIGNAL_STATUS_SUCCESS
        and evidence.provider_version == FACE_PROVIDER_VERSION
    )


def is_usable_svd(evidence: SvdEvidence) -> bool:
    """Whether an eligible synthetic-video signal's figures can carry a classification.

    Beyond the score itself, `total_clips <= 0` is the provider reporting that it aggregated
    nothing, which makes the aggregate a figure over an empty table rather than a reading of
    the video.
    """
    return _is_calibrated_probability(evidence.score) and _is_positive_count(
        evidence.total_clips
    )


def is_usable_face(evidence: FaceEvidence) -> bool:
    """Whether an eligible face signal's figures can carry a classification.

    `frames_scored <= 0` is a mean taken over no face crops at all. In practice the detector
    raises before it can write such a row — a clip with no face in it fails rather than
    scoring — so this guards the stored figures rather than the live path, which is the point:
    the rules classify what the database holds, including a row some future change writes
    differently.
    """
    return _is_calibrated_probability(evidence.score) and _is_positive_count(
        evidence.frames_scored
    )


def evaluate(
    svd: SvdEvidence | None = None, face: FaceEvidence | None = None
) -> RiskDecision:
    """Classify one analysis from its two persisted calibrated signals.

    Each detector is reduced to one of three states, entirely on its own evidence and its own
    threshold, without either detector's state depending on the other's:

    - **flagged** — the calibrated deployment answered, its figures are readable, and its score
      is at or above the threshold measured for it;
    - **below** — the same, but the score is under that threshold. Not a finding of authentic
      media: on the manipulation family this detector is blind to, `below` is what it reports
      for a manipulation as readily as for a genuine clip;
    - **silent** — no reading to use, whether because the signal is missing, the detector
      failed or abstained, the deployment is not the calibrated one, or its figures cannot be
      read. `_invalid` distinguishes the last of those, which is the only silence where a
      validated detector did answer.

    The rules are then tried in order and the first that fires decides. They are exhaustive
    over the nine combinations by construction rather than by a catch-all: any flagged
    detector lands on `R102`, `R100` or `R101`; with none flagged, two `below` land on `R200`
    and exactly one on `R201`; with none flagged and none `below` both detectors are silent,
    which is `R012` if either answered unreadably and `R010` otherwise. There is no default
    branch, and the impossible remainder raises instead of classifying.

    A `below` detector appears in no HIGH rule's condition. That is the disagreement policy
    stated as code: `R100` and `R101` do not mention the other detector at all, so a face model
    reporting 0.0053 cannot hold back a synthetic-video score of 0.9961, and NVIDIA reporting
    0.1648 cannot hold back a face score of 0.9943. Both of those are real clips from the
    calibration corpus, both are genuine manipulations, and a rule that let the quiet detector
    speak would have missed both.

    `UNKNOWN` is not a hedge and not a low-confidence HIGH. It is the honest statement that no
    validated rule could be applied — neither detector was the calibrated one, neither
    answered, or both answered with figures that cannot be read — and it is deliberately
    reachable from more conditions than the other two bands, because the error policy behind
    these thresholds prefers admitting ignorance to asserting risk.
    """
    svd_eligible = is_eligible_svd(svd)
    face_eligible = is_eligible_face(face)

    # Narrowed by the eligibility checks, which reject None before anything reads a field.
    svd_usable = svd_eligible and is_usable_svd(svd)  # type: ignore[arg-type]
    face_usable = face_eligible and is_usable_face(face)  # type: ignore[arg-type]

    svd_flagged = svd_usable and svd.score >= SVD_T_HIGH  # type: ignore[union-attr]
    face_flagged = face_usable and face.score >= FACE_T_HIGH  # type: ignore[union-attr]

    svd_below = svd_usable and not svd_flagged
    face_below = face_usable and not face_flagged

    # An eligible detector whose figures could not be read. Kept apart from an ineligible or
    # absent one so `R012` keeps the meaning it had in v1.
    svd_invalid = svd_eligible and not svd_usable
    face_invalid = face_eligible and not face_usable

    if svd_flagged and face_flagged:
        return _decision(RISK_HIGH, RULE_HIGH_BOTH_SOURCES)

    if svd_flagged:
        return _decision(RISK_HIGH, RULE_HIGH_SYNTHETIC_VIDEO)

    if face_flagged:
        return _decision(RISK_HIGH, RULE_HIGH_FACE_MANIPULATION)

    if svd_below and face_below:
        return _decision(RISK_MEDIUM, RULE_INDETERMINATE_BOTH_SOURCES)

    if svd_below or face_below:
        return _decision(RISK_MEDIUM, RULE_INDETERMINATE_SINGLE_SOURCE)

    if svd_invalid or face_invalid:
        return _decision(RISK_UNKNOWN, RULE_INVALID_CALIBRATED_EVIDENCE)

    if not svd_usable and not face_usable:
        return _decision(RISK_UNKNOWN, RULE_NO_CALIBRATED_EVIDENCE)

    raise RiskEngineError(
        f"No rule in {RULES_VERSION} classified the evidence; the rules are supposed to be "
        "exhaustive over both detectors' states."
    )
