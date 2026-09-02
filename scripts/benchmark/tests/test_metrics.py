"""Metric arithmetic, including the cases where a metric is not defined."""

import pytest

from benchmark.metrics import (
    ConfusionMatrix,
    accuracy,
    classification_metrics,
    confusion_matrix,
    false_negative_rate,
    false_positive_rate,
    latency_summary,
    precision,
    predict,
    recall,
)

# (is_manipulated, predicted_manipulated)
MIXED = [
    (True, True),    # TP
    (True, True),    # TP
    (True, False),   # FN
    (False, False),  # TN
    (False, False),  # TN
    (False, False),  # TN
    (False, True),   # FP
]


def test_confusion_matrix_counts_each_outcome():
    matrix = confusion_matrix(MIXED)
    assert (
        matrix.true_positives,
        matrix.false_negatives,
        matrix.true_negatives,
        matrix.false_positives,
    ) == (2, 1, 3, 1)
    assert matrix.total == 7


def test_confusion_matrix_of_nothing_is_all_zeros():
    matrix = confusion_matrix([])
    assert matrix.as_dict() == {
        "true_positives": 0,
        "false_positives": 0,
        "true_negatives": 0,
        "false_negatives": 0,
        "total": 0,
    }


def test_rates_on_a_mixed_matrix():
    matrix = confusion_matrix(MIXED)
    assert accuracy(matrix) == pytest.approx(5 / 7)
    assert precision(matrix) == pytest.approx(2 / 3)
    assert recall(matrix) == pytest.approx(2 / 3)
    assert false_positive_rate(matrix) == pytest.approx(1 / 4)
    assert false_negative_rate(matrix) == pytest.approx(1 / 3)


def test_recall_and_false_negative_rate_are_complements():
    matrix = confusion_matrix(MIXED)
    assert recall(matrix) + false_negative_rate(matrix) == pytest.approx(1.0)


def test_a_perfect_detector():
    matrix = ConfusionMatrix(
        true_positives=5, false_positives=0, true_negatives=5, false_negatives=0
    )
    assert classification_metrics(matrix) == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
    }


def test_a_detector_that_is_wrong_every_time():
    matrix = ConfusionMatrix(
        true_positives=0, false_positives=5, true_negatives=0, false_negatives=5
    )
    assert classification_metrics(matrix) == {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "false_positive_rate": 1.0,
        "false_negative_rate": 1.0,
    }


def test_a_detector_that_flags_nothing_scores_well_on_a_genuine_heavy_corpus():
    # 90 genuine, 10 manipulated, nothing flagged. Accuracy looks respectable and the
    # detector is useless; FNR is the number that says so. Precision is undefined
    # because nothing was predicted positive.
    matrix = ConfusionMatrix(
        true_positives=0, false_positives=0, true_negatives=90, false_negatives=10
    )
    assert accuracy(matrix) == pytest.approx(0.9)
    assert false_negative_rate(matrix) == 1.0
    assert false_positive_rate(matrix) == 0.0
    assert precision(matrix) is None


def test_undefined_metrics_are_none_not_zero():
    empty = confusion_matrix([])
    assert classification_metrics(empty) == {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
    }

    # A corpus with no manipulated media can measure FPR but not recall or FNR.
    genuine_only = ConfusionMatrix(
        true_positives=0, false_positives=2, true_negatives=8, false_negatives=0
    )
    assert recall(genuine_only) is None
    assert false_negative_rate(genuine_only) is None
    assert false_positive_rate(genuine_only) == pytest.approx(0.2)

    # And the mirror image: no genuine media means no FPR.
    manipulated_only = ConfusionMatrix(
        true_positives=8, false_positives=0, true_negatives=0, false_negatives=2
    )
    assert false_positive_rate(manipulated_only) is None
    assert recall(manipulated_only) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "score,threshold,expected",
    [
        (0.5, 0.5, True),  # inclusive at the boundary
        (0.49999, 0.5, False),
        (1.0, 0.5, True),
        (0.0, 0.0, True),
        (0.97, 0.98, False),
    ],
)
def test_predict_is_inclusive_at_the_threshold(score, threshold, expected):
    assert predict(score, threshold) is expected


def test_latency_summary_over_an_odd_sample():
    summary = latency_summary([30.0, 10.0, 20.0])
    assert summary["count"] == 3
    assert summary["mean_ms"] == pytest.approx(20.0)
    assert summary["median_ms"] == pytest.approx(20.0)
    assert summary["min_ms"] == pytest.approx(10.0)
    assert summary["max_ms"] == pytest.approx(30.0)
    assert summary["total_ms"] == pytest.approx(60.0)
    assert summary["p95_ms"] == pytest.approx(30.0)


def test_latency_summary_median_of_an_even_sample_is_the_midpoint():
    assert latency_summary([10.0, 20.0, 30.0, 40.0])["median_ms"] == pytest.approx(25.0)


def test_latency_p95_is_an_observed_value():
    summary = latency_summary([float(n) for n in range(1, 101)])
    assert summary["p95_ms"] == pytest.approx(95.0)


def test_latency_summary_of_nothing_reports_none_not_zero():
    summary = latency_summary([])
    assert summary["count"] == 0
    assert summary["mean_ms"] is None
    assert summary["max_ms"] is None
