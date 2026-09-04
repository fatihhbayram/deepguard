"""DeepGuard's own classification of one analysis, from three independently calibrated signals.

This is the first layer in the codebase entitled to say anything about what a detector's
number *means*. Everything below it — `detection.py`, the provider integrations, the
segment rows — records what a provider said and refuses to interpret it. Here that
interpretation happens, once, against thresholds that were measured rather than chosen
(R4-T1 for two of them, R5-T3 for the third), and the decision that comes out carries the
identity of the measurements it was made under so a stored verdict stays readable years
after the thresholds move.

Stateless and deterministic. It opens no connection, reads no configuration, consults no
clock and holds nothing between calls: the same evidence always yields the same decision,
which is what makes a persisted decision reproducible from the persisted evidence.

**Three signals decide, and each decides alone.** Ruleset v3 reads NVIDIA's `synthetic_video`
score, EfficientNet-B7's `face_manipulation` score and LipForensics' `lip_forensics`
mouth-dynamics score. Each is compared against its own measured threshold and against nothing
else. The three numbers never meet: they are not averaged, not weighted, not summed, not
multiplied, not voted on, and none is ever used to adjust or discount another. What the rules
combine is the three detectors' *separate decisions* about the same media, which is the only
combination R4-T1 and R5-T3 measured and the only one rule 11 of AGENTS.md permits.

C2PA provenance, the active-speaker timeline and AASIST's audio windows remain persisted as
independent forensic evidence and still cannot move this classification by a single band. They
are not read here at all: `evaluate` takes three arguments, this module imports nothing that
would fetch a fourth signal, and an analysis carrying a provenance row is classified exactly as
the same analysis without one. They have no calibration, and a scale is not a calibration.

**What changed from v2, and why it was allowed to.** Under v2 the mouth-dynamics score was
recorded and unread, for exactly one reason: nothing had measured an operating point for it,
and a third detector admitted on the strength of its scale rather than its calibration is what
this module refuses. R5-T1 benchmarked the detector and R5-T3 measured the operating point,
under the two selection rules R4-T1 stated, applied unchanged. That measurement — and not the
detector's plausibility, its provenance or its agreement with anything else — is what promotes
it here from evidence to decider.

**The third detector's operating point, and its honest limits.** R5-T3 scored 40 clips of
FaceForensics++ C23 (20 genuine, 20 face swaps) and placed `LIP_T_HIGH` by the same rule as the
other two: the lowest threshold with zero observed false positives on genuine media, at the
midpoint between the highest genuine score (0.0168) and the lowest score above it (0.4424). At
that point it flagged 20 of 20 face swaps and 0 of 20 genuine clips.

Two things about that measurement are stated here rather than left to the artifact, because
they bound what this rule may be read as saying:

1. The corpus is small and drawn from one dataset that lies inside the model's training
   distribution. Zero false positives over 20 genuine clips is a 95% upper bound of 0.1391 on
   this detector's own false-positive rate — looser than the 0.0540 each R4-T1 detector carries
   over 54 genuine clips. What makes the point defensible is not that bound but the margin
   underneath it: the threshold clears the highest genuine score by 0.2128, where the R4-T1
   thresholds clear theirs by 0.0010 and 0.0003.
2. A perfect separation on one dataset is a property of that corpus, not a claim about media in
   general. Nothing here treats this detector as more reliable than the other two because its
   study happened to separate cleanly, and no rule below gives its finding more weight.

**Why the rule is still OR, and why silence never dampens a flag.** R4-T1 measured its two
detectors over one 159-clip corpus and found them strictly complementary. NVIDIA's detector
separates generated video (AUROC 0.9205 against genuine) and is near-blind to face swaps
(0.5422); EfficientNet-B7 separates face swaps (0.9304) and is worse than chance on generated
video (0.3960). Their scores are *negatively* rank-correlated (Spearman -0.2781). At the two
selected operating points the detectors never once agreed: of 159 clips, `both_flagged` was 0,
and every one of the 47 detections came from exactly one detector while the other sat near its
floor — `ffpp_dev_Deepfakes_106_198` scored 0.1648 on NVIDIA and 0.9943 on the face model;
`sonic_en_03` scored 0.9961 on NVIDIA and 0.0053 on the face model.

Two things follow, and they are the whole design of this module:

1. A rule requiring detectors to agree would have detected nothing whatsoever on the R4-T1
   corpus. Agreement is not available evidence, so it cannot be required.
2. A detector scoring below its threshold is not asserting that the media is genuine. On the
   manipulation family it is blind to, that is exactly what it does anyway. So a low score
   carries no information capable of contradicting another detector's finding, and it is never
   allowed to soften, downgrade or veto it.

There is therefore no tie-break, because there is no tie: `R100`, `R101` and `R103` fire on
their own detector's evidence irrespective of what the others said or whether they said
anything. `R102` exists for the case where more than one detector reached its own threshold
independently, so that the trace can record a corroborated finding honestly rather than
attributing it to one source.

**The face model and the mouth-dynamics model ask different questions about the same family.**
This is the one overlap the v2 pair did not have, and it is not a licence to combine them.
EfficientNet-B7 judges the *appearance* of a face crop and never sees motion; LipForensics
judges how the mouth *moves* across 25 consecutive frames and is never shown a still. Both were
calibrated on face swaps, on different corpora, with separate operating points. Two consequences
are load-bearing below: neither may stand in for the other when one is silent (a missing
mouth-dynamics reading does not weaken a face finding, and vice versa), and the two agreeing is
recorded as `R102` without the level being raised, because there is no band above HIGH and no
measurement that says two flags mean more than one.

**The false-HIGH budget is spent per detector, and that is deliberate.** Each threshold was
placed at the lowest point with zero observed false positives on its own genuine clips. Reading
all three under OR means the bound on the *combined* rule is looser than any of them alone; no
genuine clip in either corpus was flagged by any detector, so the joint rate is still 0
observed, but the honest upper bound for a three-way disjunction is looser than for a two-way
one, which was already looser than for either detector alone. That cost is accepted because the
alternative — raising the thresholds to buy the difference back — would forfeit detection
coverage entirely, and the error policy behind these calibrations gives up detection rate to
protect genuine media only where the trade is real. It is not real here: no genuine clip in
either corpus came near any boundary under any detector.

**LOW does not exist in v3 either.** R4-T1 measured `T_LOW` for its two detectors and both are
useless as a reassurance: each would cover 1.85% of genuine media. R5-T3's `T_LOW` for the third
landed on the same value as its `T_HIGH` — the two classes were separated by a single gap and
both selection rules landed inside it — which says that corpus supports no ambiguous band at
all, not that the detector has none. All three are recorded below because the calibration
identity has to match the artifacts, not because anything branches on them.
"""

import math
from dataclasses import dataclass

# The ruleset this module implements, as one immutable string. It is not derived from a
# package version or a git hash: it names *these rules*, and it changes when a rule,
# threshold or ordering changes — never when unrelated code around it does. A stored
# decision is only re-derivable if the version it names pins the logic exactly.
RULES_VERSION = "r5-v3.0.0"

# The two calibration artifacts these rules stand on, by the identity each one computed for
# itself: the SHA-256 of its own identity fields. R4-T1 measured the synthetic-video and
# face-manipulation operating points over a 159-clip corpus; R5-T3 measured the mouth-dynamics
# operating point over a 40-clip corpus, under R4-T1's selection rules applied unchanged.
SVD_FACE_CALIBRATION_ID = (
    "cab2ea262bb7e41cb87e49bdb3dad53ecd0f02248035a993f9fcb033363afd1e"
)
LIP_CALIBRATION_ID = "85cb7484ab74d5821b1f4fa7ba917588dcaa98354f5f7a82c3080b624c9b8a29"

# Identity of the measurement behind a v3 decision, stored with every decision so a verdict
# can be traced to the corpora, the detector deployments and the error policies it was made
# under. A recalibration of either artifact produces a different id, and rows written under
# r4-v2.0.0 or p7-v1.0.0 keep theirs.
#
# Derived rather than read off an artifact, because a v3 decision rests on two of them and
# `analyses.risk_calibration_id` holds one 64-character digest. The construction is fixed and
# reproducible by anyone holding the two ids above:
#
#     sha256(f"{SVD_FACE_CALIBRATION_ID}\n{LIP_CALIBRATION_ID}".encode()).hexdigest()
#
# It is written out as a literal rather than computed at import, for the same reason every
# threshold below is: this is a fact about which measurements were adopted, and a value the
# code computes is a value the code can silently change.
CALIBRATION_ID = "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"

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

# --- The calibrated mouth-dynamics signal --------------------------------------------------
#
# The exact model this threshold was measured against, in the form `app.detection` writes into
# `provider_version`: upstream repository at revision, plus the digest of the forgery weights.
# It takes both, and the check is exact string equality on the whole string — the architecture
# is executed from source rather than shipped as one artifact, so the same checkpoint loaded
# into a different network is a different model with no operating point of its own.
#
# The device is deliberately not part of the binding. R5-T3 measured the same corpus on CPU and
# on CUDA: scores moved by at most 0.0715, no clip's decision at the selected threshold flipped,
# and the selection rule re-run on the other device landed at 0.1939 — inside the same gap. The
# stored signal records which device produced it either way (`app.detection`).
LIP_PROVIDER = "lipforensics"
LIP_SIGNAL_TYPE = "lip_forensics"
LIP_PROVIDER_VERSION = (
    "https://github.com/ahaliassos/LipForensics"
    "@d0bf5553bfb9676f1771d590472b26a3a76de894"
    "+4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
)

# Where HIGH begins for this detector, derived by the same rule as the two above against 20
# genuine clips: the midpoint between the highest genuine score (0.0168) and the lowest score
# above it (0.4424). At that point it flags 100% of the face swaps in its corpus and 0% of
# genuine media. The margin over the highest genuine score is 0.2128 — see the module docstring
# for what that margin does and does not buy.
#
# Not 0.5. That figure appears in the R5-T1 benchmark as a reporting convention fixed before any
# score existed; on this corpus it sits above one real face swap and buys no reduction in
# observed false positives, so R5-T3 rejected it by measurement rather than by assertion.
LIP_T_HIGH = 0.22962537594139576

# Measured and deliberately inert, like the two above. It coincides with `LIP_T_HIGH` because
# both selection rules landed inside one clean gap between the classes; that is a fact about the
# corpus and not a boundary, and nothing compares against it.
LIP_T_LOW = 0.22962537594139576

# The lower and upper ends of all three providers' scales. Every score is a probability —
# NVIDIA's is `expit(mean clip logit)`, the classifier's is a mean of sigmoids, LipForensics' is
# `sigmoid(mean window logit)` — so none can leave [0, 1], and a score outside it is not a
# low-risk reading. It is evidence that something other than the calibrated detector produced
# the row.
SCORE_FLOOR = 0.0
SCORE_CEILING = 1.0

# What this engine may conclude. LOW is absent by design and is never emitted.
RISK_HIGH = "HIGH"
RISK_MEDIUM = "MEDIUM"
RISK_UNKNOWN = "UNKNOWN"

# The rules, in the order they are tried. The id is persisted with the decision, so reading a
# stored row tells you not only what was concluded but which sentence concluded it, and — now
# that there are three sources — which source or sources concluded it.
#
# There are eight, and there are deliberately not more. Three detectors have 125 distinguishable
# input states and 7 non-empty subsets that could flag; enumerating a rule per subset would name
# conditions that decide identically and would leave a rule table nobody can hold in mind. What
# earns an id here is a genuinely different decision condition:
#
#   * which single detector's evidence produced a HIGH, because that is what fixes the coverage
#     claim behind it — generated video, face appearance, or mouth motion;
#   * that more than one detector reached its own threshold independently, because attributing a
#     corroborated finding to one source would credit it with evidence another produced;
#   * whether a MEDIUM was reached with every detector reporting or with only some of them,
#     because those are the same band with materially different coverage;
#   * and the two ways no rule could be applied at all.
#
# `R102` does not say *which* detectors agreed and `R201` does not say *which* were readable.
# Neither needs to: the per-signal rows record exactly what each detector did, and the report
# renders them beside the trace. An id per subset would move that detail into the rule table
# without adding a decision.
#
# The two UNKNOWN ids are carried over from p7-v1.0.0 with their meanings intact: `R010` is
# "no validated reading was ever obtained", `R012` is "one was obtained and its figures could
# not be read". They are not sub-divided per detector, for the same reason.
RULE_NO_CALIBRATED_EVIDENCE = "R010"
RULE_INVALID_CALIBRATED_EVIDENCE = "R012"
RULE_HIGH_SYNTHETIC_VIDEO = "R100"
RULE_HIGH_FACE_MANIPULATION = "R101"
RULE_HIGH_MULTIPLE_SOURCES = "R102"
RULE_HIGH_MOUTH_DYNAMICS = "R103"
RULE_INDETERMINATE_ALL_SOURCES = "R200"
RULE_INDETERMINATE_PARTIAL_SOURCES = "R201"

# The provider status that makes a signal eligible at all. A detector that refused or timed
# out produced no reading, and a stale score sitting beside a failed status is not evidence.
# A face classifier that found no face in the clip is a `FAILED` row and lands here too, as is
# a mouth-dynamics reading in which no run held a trackable face: abstention is not a finding of
# no manipulation, and R4-T1 kept the two distinct for the same reason (8 of its clips were
# abstentions, every one of them on generated video).
SIGNAL_STATUS_SUCCESS = "SUCCESS"


class RiskEngineError(Exception):
    """The rules failed to classify evidence they are supposed to cover exhaustively.

    Unreachable by construction — see `evaluate`, which enumerates the partition. It exists so
    the impossible case is a loud defect rather than a quiet default classification: inventing
    a fall-through band would let a bug in the rules above ship as a verdict about someone's
    media.
    """


class UncalibratedEvidence(TypeError):
    """Something that is not a calibrated production signal was offered to the rules (R6-T1).

    Shadow mode runs uncalibrated experimental workloads on live traffic, and the one thing
    those workloads may never do is influence a verdict about someone's media. That is already
    true structurally — their observations are written to `shadow_runs`, the three readers in
    `app.worker` select `analysis_signals` by provider and signal type, and nothing turns a
    shadow row into one of the three types below — so this guard is the second lock rather than
    the first.

    It exists because the first lock is an absence, and an absence is invisible to whoever
    later adds a caller. `evaluate` is a public function with three optional parameters and no
    verdict is a small change away from being taken on the wrong evidence; a caller that tries
    now gets a `TypeError` naming what it passed instead of a classification.

    A `TypeError` and not a `RiskEngineError`, because it is not a failure of the rules. The
    rules never ran: they were handed something that is not their input.
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
    count. The detectors aggregate different things — NVIDIA over clips it cut itself, this
    classifier over face crops sampled from frames — and the field that says a reading is
    degenerate is named after what was actually counted. Collapsing them into one `unit_count`
    would make the rules read as though the numbers were commensurable, which is the exact
    confusion this module exists to prevent.

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
class LipEvidence:
    """The persisted LipForensics mouth-dynamics signal, as the database holds it.

    A third separate type, for the third time and the same reason. This detector aggregates over
    runs of 25 consecutive frames it sampled itself, which is neither a clip NVIDIA cut nor a
    face crop the classifier scored, and `windows_scored` is named after the thing it counts.

    It is emphatically not a second reading of `FaceEvidence`. That model judges the appearance
    of a still crop; this one judges how a mouth moves over time. The two are never compared,
    averaged or reconciled anywhere below.

    `windows_scored` comes from the signal's metadata, where `app.detection` records how many
    runs actually held a trackable face and therefore how many logits the stored mean was taken
    over.
    """

    provider: str | None
    signal_type: str | None
    status: str | None
    provider_version: str | None
    score: float | None
    windows_scored: int | None


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

    Shared by all three detectors because it is one question about one kind of value — a
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


def is_eligible_lip(evidence: LipEvidence | None) -> bool:
    """Whether the mouth-dynamics signal is the calibrated model answering successfully.

    The same four checks again, against this detector's own identity — which is a composite of
    two things rather than one, and is compared whole. The architecture revision and the weights
    digest each fix half of what produced a score, and a row naming one of them is not the model
    R5-T3 measured.
    """
    return (
        evidence is not None
        and evidence.provider == LIP_PROVIDER
        and evidence.signal_type == LIP_SIGNAL_TYPE
        and evidence.status == SIGNAL_STATUS_SUCCESS
        and evidence.provider_version == LIP_PROVIDER_VERSION
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


def is_usable_lip(evidence: LipEvidence) -> bool:
    """Whether an eligible mouth-dynamics signal's figures can carry a classification.

    `windows_scored <= 0` is a mean taken over no runs at all. As with the face classifier the
    detector raises before it can write such a row — a clip in which no run holds a trackable
    face fails rather than scoring — so this guards the stored figures rather than the live
    path.
    """
    return _is_calibrated_probability(evidence.score) and _is_positive_count(
        evidence.windows_scored
    )


def _reject_uncalibrated(parameter: str, evidence: object, expected: type) -> None:
    """Refuse anything that is not the calibrated evidence type this parameter is for.

    `type(...) is not` rather than `isinstance`, so a subclass is refused too. A type that
    inherited `SvdEvidence` to carry an experimental workload's number would satisfy an
    `isinstance` check and be classified against a threshold measured for NVIDIA's detector,
    which is precisely the substitution this guard is here to make impossible.

    `None` is admitted: a missing signal is evidence the rules decide on, not a caller error.

    The message names the type it was given and never the value. An observation is not a
    secret, but a rejected object is untrusted input and rendering it into a log line or an
    exception is how untrusted input ends up somewhere it was never reviewed for.
    """
    if evidence is None or type(evidence) is expected:
        return

    raise UncalibratedEvidence(
        f"The risk engine was offered {type(evidence).__name__} as {parameter} evidence; "
        f"only {expected.__name__} is calibrated for these rules. Uncalibrated evidence — "
        "shadow-mode observations included — cannot reach a verdict."
    )


def evaluate(
    svd: SvdEvidence | None = None,
    face: FaceEvidence | None = None,
    lip: LipEvidence | None = None,
) -> RiskDecision:
    """Classify one analysis from its three persisted calibrated signals.

    Each detector is reduced to one of three states, entirely on its own evidence and its own
    threshold, without any detector's state depending on another's:

    - **flagged** — the calibrated deployment answered, its figures are readable, and its score
      is at or above the threshold measured for it;
    - **below** — the same, but the score is under that threshold. Not a finding of authentic
      media: on the manipulation family this detector is blind to, `below` is what it reports
      for a manipulation as readily as for a genuine clip;
    - **silent** — no reading to use, whether because the signal is missing, the detector
      failed or abstained, the deployment is not the calibrated one, or its figures cannot be
      read. `_invalid` distinguishes the last of those, which is the only silence where a
      validated detector did answer.

    The rules are then tried in order and the first that fires decides. They are exhaustive over
    the 125 combinations by construction rather than by a catch-all: two or more flagged land on
    `R102` and exactly one on `R100`, `R101` or `R103`; with none flagged, three `below` land on
    `R200` and one or two on `R201`; with none flagged and none `below` every detector is silent,
    which is `R012` if any answered unreadably and `R010` otherwise. There is no default branch,
    and the impossible remainder raises instead of classifying.

    A `below` detector appears in no HIGH rule's condition. That is the disagreement policy
    stated as code: `R100`, `R101` and `R103` do not mention the other detectors at all, so a
    face model reporting 0.0053 cannot hold back a synthetic-video score of 0.9961, NVIDIA
    reporting 0.1648 cannot hold back a face score of 0.9943, and neither of them reporting
    anything can hold back a mouth-dynamics score of 1.0. The first two are real clips from the
    R4-T1 corpus, both are genuine manipulations, and a rule that let the quiet detector speak
    would have missed both.

    Nor may a *missing* detector hold one back, which is the case R5-T3's corpus makes concrete:
    LipForensics needs a trackable face through 25 consecutive frames, so it abstains on media
    the other two score without difficulty. An abstention is silence, and silence decides
    nothing here.

    `UNKNOWN` is not a hedge and not a low-confidence HIGH. It is the honest statement that no
    validated rule could be applied — no detector was the calibrated one, none answered, or
    those that answered did so with figures that cannot be read — and it is deliberately
    reachable from more conditions than the other two bands, because the error policy behind
    these thresholds prefers admitting ignorance to asserting risk.

    Nothing uncalibrated is admitted, and that is checked before a single rule is read (R6-T1).
    Only the three types above may reach these rules; anything else — a shadow-mode observation
    above all — raises `UncalibratedEvidence` rather than being classified against a threshold
    that was never measured for it.
    """
    # Before any rule is consulted: these are the three types the rules are calibrated for and
    # nothing else is admitted (R6-T1). See `UncalibratedEvidence` for why the structural
    # separation is not considered enough on its own.
    _reject_uncalibrated("svd", svd, SvdEvidence)
    _reject_uncalibrated("face", face, FaceEvidence)
    _reject_uncalibrated("lip", lip, LipEvidence)

    svd_eligible = is_eligible_svd(svd)
    face_eligible = is_eligible_face(face)
    lip_eligible = is_eligible_lip(lip)

    # Narrowed by the eligibility checks, which reject None before anything reads a field.
    svd_usable = svd_eligible and is_usable_svd(svd)  # type: ignore[arg-type]
    face_usable = face_eligible and is_usable_face(face)  # type: ignore[arg-type]
    lip_usable = lip_eligible and is_usable_lip(lip)  # type: ignore[arg-type]

    svd_flagged = svd_usable and svd.score >= SVD_T_HIGH  # type: ignore[union-attr]
    face_flagged = face_usable and face.score >= FACE_T_HIGH  # type: ignore[union-attr]
    lip_flagged = lip_usable and lip.score >= LIP_T_HIGH  # type: ignore[union-attr]

    svd_below = svd_usable and not svd_flagged
    face_below = face_usable and not face_flagged
    lip_below = lip_usable and not lip_flagged

    # An eligible detector whose figures could not be read. Kept apart from an ineligible or
    # absent one so `R012` keeps the meaning it had in v1.
    svd_invalid = svd_eligible and not svd_usable
    face_invalid = face_eligible and not face_usable
    lip_invalid = lip_eligible and not lip_usable

    # Counted rather than enumerated. Each detector's flag was reached on its own evidence
    # against its own threshold, and what the rules need from the three of them is how many
    # fired and — when exactly one did — which. Nothing is summed that a reader could mistake
    # for a score: these are counts of separate decisions, not a pooled number.
    flagged = sum((svd_flagged, face_flagged, lip_flagged))
    readable = sum((svd_usable, face_usable, lip_usable))

    if flagged >= 2:
        return _decision(RISK_HIGH, RULE_HIGH_MULTIPLE_SOURCES)

    if svd_flagged:
        return _decision(RISK_HIGH, RULE_HIGH_SYNTHETIC_VIDEO)

    if face_flagged:
        return _decision(RISK_HIGH, RULE_HIGH_FACE_MANIPULATION)

    if lip_flagged:
        return _decision(RISK_HIGH, RULE_HIGH_MOUTH_DYNAMICS)

    if svd_below and face_below and lip_below:
        return _decision(RISK_MEDIUM, RULE_INDETERMINATE_ALL_SOURCES)

    if readable >= 1:
        return _decision(RISK_MEDIUM, RULE_INDETERMINATE_PARTIAL_SOURCES)

    if svd_invalid or face_invalid or lip_invalid:
        return _decision(RISK_UNKNOWN, RULE_INVALID_CALIBRATED_EVIDENCE)

    if readable == 0:
        return _decision(RISK_UNKNOWN, RULE_NO_CALIBRATED_EVIDENCE)

    raise RiskEngineError(
        f"No rule in {RULES_VERSION} classified the evidence; the rules are supposed to be "
        "exhaustive over all three detectors' states."
    )
