"""Binary classification and performance metrics for a benchmark run.

Pure functions over plain data. Nothing here reads a file, a clock or a model — the
same inputs always produce the same output, which is what lets a `results.json` be
recomputed from the per-clip records it carries.

**Positive class.** The positive class is *manipulated media*. A detector's job here
is to find manipulation, so a true positive is a manipulated clip correctly flagged,
and a false positive is genuine media wrongly flagged. This orientation decides what
FPR and FNR mean and it is fixed for the whole framework: FPR is the rate at which
genuine media is accused, FNR the rate at which manipulation is missed. Those two are
the numbers a forensic product is actually judged on, and swapping the convention
silently would invert both.

**Undefined metrics are `None`, never `0.0`.** Precision over zero predicted positives
is not zero, it is undefined, and a benchmark that reports `0.0` there has fabricated a
measurement. Every ratio below returns `None` when its denominator is empty, and the
JSON artifact carries `null`. That is AGENTS.md rule 11 applied to arithmetic.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts of the four outcomes, with *manipulated* as the positive class."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "total": self.total}


def confusion_matrix(pairs: list[tuple[bool, bool]]) -> ConfusionMatrix:
    """Count outcomes for `(is_manipulated, predicted_manipulated)` pairs.

    Both elements are booleans by the time they arrive: thresholding a raw score is
    the caller's decision (see `predict`), not a concern of the counting.
    """
    tp = fp = tn = fn = 0
    for truth, predicted in pairs:
        if truth and predicted:
            tp += 1
        elif truth and not predicted:
            fn += 1
        elif not truth and predicted:
            fp += 1
        else:
            tn += 1
    return ConfusionMatrix(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def predict(score: float, threshold: float) -> bool:
    """Decide `manipulated` from a raw score.

    Inclusive at the threshold: a score exactly equal to it counts as manipulated.
    Stated here rather than left to the reader of a `>=` because the boundary is
    part of what a reported number means, and R4 will move this threshold around.
    """
    return score >= threshold


def _ratio(numerator: int, denominator: int) -> float | None:
    """A rate, or `None` when nothing was measured."""
    if denominator == 0:
        return None
    return numerator / denominator


def accuracy(matrix: ConfusionMatrix) -> float | None:
    """Share of all clips classified correctly.

    Reported because it is expected, and read with care: on a corpus that is mostly
    genuine media, a detector that flags nothing scores well here while being useless.
    FPR and FNR are the honest pair.
    """
    return _ratio(matrix.true_positives + matrix.true_negatives, matrix.total)


def precision(matrix: ConfusionMatrix) -> float | None:
    """Of the clips flagged as manipulated, the share that really were."""
    return _ratio(
        matrix.true_positives, matrix.true_positives + matrix.false_positives
    )


def recall(matrix: ConfusionMatrix) -> float | None:
    """Of the manipulated clips, the share that were flagged. Also 1 - FNR."""
    return _ratio(
        matrix.true_positives, matrix.true_positives + matrix.false_negatives
    )


def false_positive_rate(matrix: ConfusionMatrix) -> float | None:
    """Share of *genuine* clips wrongly flagged as manipulated."""
    return _ratio(
        matrix.false_positives, matrix.false_positives + matrix.true_negatives
    )


def false_negative_rate(matrix: ConfusionMatrix) -> float | None:
    """Share of *manipulated* clips that were missed."""
    return _ratio(
        matrix.false_negatives, matrix.false_negatives + matrix.true_positives
    )


def classification_metrics(matrix: ConfusionMatrix) -> dict[str, float | None]:
    """The five reported rates, in one dict shaped for the JSON artifact."""
    return {
        "accuracy": accuracy(matrix),
        "precision": precision(matrix),
        "recall": recall(matrix),
        "false_positive_rate": false_positive_rate(matrix),
        "false_negative_rate": false_negative_rate(matrix),
    }


def latency_summary(latencies_ms: list[float]) -> dict[str, float | None]:
    """Per-clip latency distribution.

    Mean is what the roadmap asks for; median and max come with it because a mean
    alone hides the two failure modes that matter operationally — a long tail, and one
    pathological clip. `p95` is the sorted-sample nearest-rank value, which on the small
    corpora this framework is built for is a real observed measurement rather than an
    interpolation between two clips that were never that slow.
    """
    if not latencies_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "total_ms": None,
        }
    ordered = sorted(latencies_ms)
    count = len(ordered)
    middle = count // 2
    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    # Nearest-rank: the smallest observed value at or above the 95th percentile.
    p95_index = min(count - 1, -(-95 * count // 100) - 1)
    return {
        "count": count,
        "mean_ms": sum(ordered) / count,
        "median_ms": median,
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "total_ms": sum(ordered),
    }
