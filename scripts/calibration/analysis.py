"""The measurements R4-T1 selects thresholds from. Pure functions over plain data.

Nothing here reads a file, a clock, a model or a network. The same scores always produce
the same tables, which is what lets a calibration artifact be recomputed from the per-clip
records it carries — and lets the parts that decide a threshold be tested without a GPU or
a provider account.

**Two detectors are measured, never combined.** Every distribution, sweep and operating
point below belongs to exactly one detector. Where the two appear together it is as a
*joint table of their separate decisions* — how often they agree, and on which families
they disagree — and never as a blended score. Averaging two detectors that answer
different questions would fabricate a number neither of them reported (AGENTS.md rule 11),
and the disagreements are the evidence R4-T2 needs rather than noise to be smoothed away.

**Thresholds are derived, not chosen.** `select_high_threshold` and `select_low_threshold`
implement one stated rule each, and the rule is a function of the observed scores alone:
no grid of round numbers is consulted and no value is entered by hand. Both can return
`None`, which is the honest answer when the distributions overlap so completely that no
threshold supports the claim the band would make.

**Undefined is `None`, never `0.0`** — the same convention `benchmark.metrics` holds to.
An AUROC over an empty class, a rate over an empty denominator and a threshold that does
not exist are all absences, and reporting them as zero would invent a measurement.
"""

import math
from statistics import fmean, quantiles

# Confidence level for the binomial bounds reported beside every observed rate. A count of
# zero false positives over a hundred samples is not a false positive rate of zero, and the
# upper bound is the only honest statement of what such a count establishes.
CONFIDENCE_ALPHA = 0.05

# The detectors' shared score range. Both report a probability in [0, 1] — NVIDIA's is
# `expit` of its own aggregate logit, EfficientNet-B7's a mean of sigmoids — so a threshold
# above the highest possible score, or below the lowest, cannot exist and is reported as
# absent rather than as the endpoint.
SCORE_FLOOR = 0.0
SCORE_CEILING = 1.0


def score_summary(values: list[float]) -> dict:
    """Where a set of scores sits: count, quartiles, extremes, mean.

    Quartiles by the inclusive method, so on the small per-family groups in this corpus
    the reported q1 and q3 stay inside the observed range instead of being extrapolated
    past the samples that were actually measured.
    """
    if not values:
        return {
            "count": 0, "min": None, "q1": None, "median": None,
            "q3": None, "max": None, "mean": None,
        }
    ordered = sorted(values)
    if len(ordered) == 1:
        only = ordered[0]
        q1 = median = q3 = only
    else:
        q1, median, q3 = quantiles(ordered, n=4, method="inclusive")
    return {
        "count": len(ordered),
        "min": ordered[0],
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": ordered[-1],
        "mean": fmean(ordered),
    }


def auroc(positive: list[float], negative: list[float]) -> float | None:
    """Probability a random manipulated clip outscores a random genuine one.

    The rank form of the Mann-Whitney statistic, with tied scores sharing an average rank
    so that a detector returning the same number for both classes measures 0.5 rather than
    whichever class happened to sort first. Threshold-free on purpose: it says how much the
    two distributions overlap, which is the quantity that decides whether *any* pair of
    operating points can carve out clean bands, before a particular pair is argued about.
    """
    if not positive or not negative:
        return None
    combined = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])
    ranks = [0.0] * len(combined)
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        shared = (index + end) / 2 + 1  # ranks are 1-based
        for position in range(index, end + 1):
            ranks[position] = shared
        index = end + 1
    positive_rank_sum = sum(
        rank for rank, (_, group) in zip(ranks, combined) if group == 1
    )
    n_positive, n_negative = len(positive), len(negative)
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """P(X <= successes) for X ~ Binomial(trials, probability), summed exactly."""
    return sum(
        math.comb(trials, i) * probability**i * (1 - probability) ** (trials - i)
        for i in range(successes + 1)
    )


def clopper_pearson_upper(
    successes: int, trials: int, alpha: float = CONFIDENCE_ALPHA
) -> float | None:
    """Exact upper confidence bound on a rate, at `1 - alpha`.

    "Zero false positives in 55 genuine clips" bounds the true rate near 5 %, not at 0 %,
    and every zero count in this calibration is reported with that bound beside it. The
    bound is found by bisection on the exact binomial CDF rather than by a normal
    approximation, which is unusable at the counts and sample sizes here.
    """
    if trials <= 0:
        return None
    if successes >= trials:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        if _binomial_cdf(successes, trials, middle) > alpha:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _split(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Genuine scores and manipulated scores, from rows carrying `label` and `score`."""
    genuine = [row["score"] for row in rows if row["label"] == "real"]
    manipulated = [row["score"] for row in rows if row["label"] != "real"]
    return genuine, manipulated


def sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """Measured trade-off at each threshold: what it costs on genuine media, what it catches.

    One row per threshold, with the flag rate broken out per stratum. The breakdown is the
    point: a pooled detection rate over a corpus that is part face swap and part generated
    video describes neither, and the whole reason two detectors are being calibrated
    together is that they succeed on different halves of it.
    """
    genuine, manipulated = _split(rows)
    strata = sorted({row["stratum"] for row in rows})
    table = []
    for threshold in thresholds:
        false_positives = sum(1 for score in genuine if score >= threshold)
        true_positives = sum(1 for score in manipulated if score >= threshold)
        table.append(
            {
                "threshold": threshold,
                "false_positives": false_positives,
                "genuine_n": len(genuine),
                "false_positive_rate": (
                    false_positives / len(genuine) if genuine else None
                ),
                "false_positive_rate_upper95": clopper_pearson_upper(
                    false_positives, len(genuine)
                ),
                "true_positives": true_positives,
                "manipulated_n": len(manipulated),
                "true_positive_rate": (
                    true_positives / len(manipulated) if manipulated else None
                ),
                "flag_rate_by_stratum": {
                    stratum: _flag_rate(rows, stratum, threshold) for stratum in strata
                },
            }
        )
    return table


def _flag_rate(rows: list[dict], stratum: str, threshold: float) -> float | None:
    group = [row["score"] for row in rows if row["stratum"] == stratum]
    if not group:
        return None
    return sum(1 for score in group if score >= threshold) / len(group)


def sweep_grid(rows: list[dict], steps: int = 20) -> list[float]:
    """A reporting grid drawn from the observed scores, not from round numbers.

    Evenly spaced quantiles of everything measured, deduplicated. It exists so the
    trade-off table covers the range the data actually occupies — a fixed 0.1/0.2/0.3 grid
    on a detector whose scores all sit above 0.9 would show one meaningful row. The
    selected operating points come from `select_high_threshold` and
    `select_low_threshold`, never from this grid.
    """
    values = sorted(row["score"] for row in rows)
    if not values:
        return []
    grid = {
        values[min(len(values) - 1, round(index * (len(values) - 1) / steps))]
        for index in range(steps + 1)
    }
    return sorted(grid)


def select_high_threshold(rows: list[dict]) -> dict:
    """The lowest threshold at which no genuine clip in this corpus is flagged.

    **The rule, stated before the number it produces.** DeepGuard's adopted error policy
    (P7-T2 §6) is to strongly avoid a false HIGH on legitimate media, accepting lost
    detection as the price. Applied to a measured corpus that rule has exactly one
    solution: the operating point must sit above every genuine score observed, and among
    the points that do, the lowest one detects the most manipulation. So:

        T_HIGH = midpoint between the highest genuine score and the lowest score above it

    The midpoint rather than the genuine maximum itself, so the boundary falls in the gap
    between two measurements instead of on top of one — a threshold placed exactly on an
    observed score is decided by the `>=` in `metrics.predict` rather than by evidence.
    Nothing here is chosen: the corpus determines the value, and a different corpus
    determines a different one.

    Returns `threshold: None` when the highest genuine score is the ceiling of the
    detector's range, i.e. when genuine media reached the top of the scale and no
    threshold can exclude it. That is a refusal, not a failure — see the calibration
    report's reading of it.
    """
    genuine, manipulated = _split(rows)
    if not genuine:
        return {"threshold": None, "reason": "no genuine clips in the corpus"}
    genuine_max = max(genuine)
    above = sorted(
        score for score in (genuine + manipulated) if score > genuine_max
    )
    if genuine_max >= SCORE_CEILING:
        return {
            "threshold": None,
            "reason": (
                f"genuine media reached the top of the detector's range "
                f"({genuine_max}); no threshold excludes it"
            ),
            "genuine_max": genuine_max,
        }
    upper = above[0] if above else SCORE_CEILING
    threshold = (genuine_max + upper) / 2
    top_genuine = sorted(
        (
            {"clip_id": row.get("clip_id"), "score": row["score"], "family": row.get("family")}
            for row in rows
            if row["label"] == "real"
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[:5]
    return {
        "threshold": threshold,
        # The five highest genuine scores, because the margin above is a distance to *one*
        # sample and the reader is entitled to see how isolated that sample is. A tail that
        # falls away steeply behind the maximum and one that crowds up against it support
        # very different confidence in the same threshold.
        "genuine_tail": top_genuine,
        "rule": (
            "lowest threshold with zero observed false positives on genuine media, "
            "placed at the midpoint between the highest genuine score and the lowest "
            "score above it"
        ),
        "genuine_max": genuine_max,
        "next_score_above_genuine_max": above[0] if above else None,
        "margin_over_genuine_max": threshold - genuine_max,
    }


def select_low_threshold(rows: list[dict]) -> dict:
    """The highest threshold below which no manipulated clip in this corpus falls.

    The mirror of `select_high_threshold`, under the mirror of the same policy: a LOW band
    may only make a reassuring statement about media no manipulated sample in the corpus
    would have received. Below the lowest manipulated score, placed at the midpoint
    between it and the highest score beneath it.

    `coverage` is what the band would be worth — the share of genuine media that would
    actually earn it. A threshold that excludes every manipulated clip but admits almost
    no genuine one is a band nobody reaches, and P7-T2 §7.2 already found exactly that for
    NVIDIA's detector; the figure is returned so the report can say so with a number.
    """
    genuine, manipulated = _split(rows)
    if not manipulated:
        return {"threshold": None, "reason": "no manipulated clips in the corpus"}
    manipulated_min = min(manipulated)
    if manipulated_min <= SCORE_FLOOR:
        return {
            "threshold": None,
            "reason": (
                f"manipulated media reached the bottom of the detector's range "
                f"({manipulated_min}); no threshold excludes it"
            ),
            "manipulated_min": manipulated_min,
        }
    below = sorted(
        score for score in (genuine + manipulated) if score < manipulated_min
    )
    lower = below[-1] if below else SCORE_FLOOR
    threshold = (manipulated_min + lower) / 2
    return {
        "threshold": threshold,
        "rule": (
            "highest threshold with zero observed manipulated clips beneath it, placed "
            "at the midpoint between the lowest manipulated score and the highest score "
            "below it"
        ),
        "manipulated_min": manipulated_min,
        "previous_score_below": below[-1] if below else None,
        "coverage_genuine": (
            sum(1 for score in genuine if score < threshold) / len(genuine)
            if genuine
            else None
        ),
    }


def youden_point(rows: list[dict]) -> dict:
    """The threshold with the best measured TPR - FPR, reported and *not* selected.

    Included because a calibration that shows only the operating point its policy picked
    hides the trade-off that policy made. This is where the corpus says the detector
    separates the two classes best; the distance between its false-positive rate and the
    selected point's zero is the price the error policy pays, in the corpus's own numbers.
    """
    genuine, manipulated = _split(rows)
    if not genuine or not manipulated:
        return {"threshold": None, "reason": "both classes are required"}
    best = None
    for candidate in sorted({row["score"] for row in rows}):
        tpr = sum(1 for score in manipulated if score >= candidate) / len(manipulated)
        fpr = sum(1 for score in genuine if score >= candidate) / len(genuine)
        if best is None or tpr - fpr > best["youden_j"]:
            best = {
                "threshold": candidate,
                "youden_j": tpr - fpr,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
            }
    return best


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Rank correlation between two detectors' scores over the same clips.

    Rank rather than linear correlation because the two scales are not comparable —
    NVIDIA's aggregate probability and a mean of per-face sigmoids are different
    quantities — and only their *ordering* of the same clips can be honestly compared. A
    low value is the evidence that the two carry independent information, which is the
    premise R4-T2's disagreement rules rest on. It is a description of the pair, never an
    input to a combined score.
    """
    if len(pairs) < 2:
        return None
    left = _ranks([pair[0] for pair in pairs])
    right = _ranks([pair[1] for pair in pairs])
    mean_left, mean_right = fmean(left), fmean(right)
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    spread_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    spread_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if spread_left == 0 or spread_right == 0:
        return None
    return covariance / (spread_left * spread_right)


def _ranks(values: list[float]) -> list[float]:
    """Ranks with ties sharing their average, in the order the values were given."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2 + 1
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


# What the joint table can say about one clip. `abstained` is deliberately its own state
# and never folded into "not flagged": EfficientNet-B7 raises when it finds no face, which
# is a refusal to answer, and recording it as an unflagged clip would turn "I did not see a
# face" into "I saw no manipulation" — a verdict the detector never gave.
AGREE_FLAGGED = "both_flagged"
AGREE_CLEAR = "neither_flagged"
DISAGREE_SVD_ONLY = "synthetic_video_only"
DISAGREE_FACE_ONLY = "face_manipulation_only"
ABSTAINED_SVD = "synthetic_video_abstained"
ABSTAINED_FACE = "face_manipulation_abstained"
ABSTAINED_BOTH = "both_abstained"


def joint_states(
    svd_rows: list[dict],
    face_rows: list[dict],
    svd_threshold: float | None,
    face_threshold: float | None,
) -> list[dict]:
    """One row per clip: what each detector said, and whether they agree.

    Clips are joined by `clip_id` across the two runs, and a clip either detector could
    not score is carried through as an abstention rather than dropped — a corpus where one
    detector answers and the other cannot is precisely the case R4-T2 has to write a rule
    for, and dropping it would hide how often it happens.

    A `None` threshold means that detector has no supportable operating point on this
    corpus, and every clip is then recorded as unflagged *by that detector* with the reason
    carried in the calibration artifact, not silently treated as clear.
    """
    face_by_id = {row["clip_id"]: row for row in face_rows}
    joint = []
    for svd in svd_rows:
        face = face_by_id.get(svd["clip_id"])
        if face is None:
            continue
        svd_flag = _flag(svd["score"], svd_threshold)
        face_flag = _flag(face["score"], face_threshold)
        joint.append(
            {
                "clip_id": svd["clip_id"],
                "label": svd["label"],
                "family": svd["family"],
                "stratum": svd["stratum"],
                "synthetic_video_score": svd["score"],
                "face_manipulation_score": face["score"],
                "synthetic_video_flagged": svd_flag,
                "face_manipulation_flagged": face_flag,
                "state": _state(svd_flag, face_flag),
            }
        )
    return joint


def _flag(score: float | None, threshold: float | None) -> bool | None:
    """Whether this score clears the operating point. `None` is an abstention."""
    if score is None:
        return None
    if threshold is None:
        return False
    return score >= threshold


def _state(svd_flag: bool | None, face_flag: bool | None) -> str:
    if svd_flag is None and face_flag is None:
        return ABSTAINED_BOTH
    if svd_flag is None:
        return ABSTAINED_SVD
    if face_flag is None:
        return ABSTAINED_FACE
    if svd_flag and face_flag:
        return AGREE_FLAGGED
    if svd_flag:
        return DISAGREE_SVD_ONLY
    if face_flag:
        return DISAGREE_FACE_ONLY
    return AGREE_CLEAR


def disagreement_summary(joint: list[dict]) -> dict:
    """How often the two detectors agree, broken down by stratum and by family.

    The overall counts answer "how often does this happen", and the breakdowns answer the
    question that actually shapes a rule: *on what kind of media*. A `face_manipulation_only`
    row over a face swap and one over genuine footage carry opposite meanings, and only the
    breakdown separates them.
    """
    states = (
        AGREE_FLAGGED, AGREE_CLEAR, DISAGREE_SVD_ONLY, DISAGREE_FACE_ONLY,
        ABSTAINED_SVD, ABSTAINED_FACE, ABSTAINED_BOTH,
    )
    def tally(rows: list[dict]) -> dict:
        counts = {state: 0 for state in states}
        for row in rows:
            counts[row["state"]] += 1
        return {"n": len(rows), **counts}

    by_stratum = {
        stratum: tally([row for row in joint if row["stratum"] == stratum])
        for stratum in sorted({row["stratum"] for row in joint})
    }
    by_family = {
        family: tally([row for row in joint if row["family"] == family])
        for family in sorted({row["family"] for row in joint})
    }
    scored = [
        (row["synthetic_video_score"], row["face_manipulation_score"])
        for row in joint
        if row["synthetic_video_score"] is not None
        and row["face_manipulation_score"] is not None
    ]
    return {
        "overall": tally(joint),
        "by_stratum": by_stratum,
        "by_family": by_family,
        "score_rank_correlation": spearman(scored),
        "clips_scored_by_both": len(scored),
    }
