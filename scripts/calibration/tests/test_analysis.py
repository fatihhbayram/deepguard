"""The measurements a threshold is derived from, checked against hand-computable cases."""

import math

import pytest

from calibration import analysis


def rows(*triples):
    """`(clip_id, label, score)` triples as the rows the analysis functions take."""
    return [
        {
            "clip_id": clip_id,
            "label": label,
            "score": score,
            "family": "family_" + label,
            "stratum": "genuine_face" if label == "real" else "manipulated",
        }
        for clip_id, label, score in triples
    ]


def test_score_summary_reports_quartiles_and_extremes():
    summary = analysis.score_summary([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary["count"] == 5
    assert summary["min"] == pytest.approx(0.1)
    assert summary["median"] == pytest.approx(0.3)
    assert summary["max"] == pytest.approx(0.5)
    assert summary["mean"] == pytest.approx(0.3)


def test_score_summary_of_nothing_is_absent_not_zero():
    summary = analysis.score_summary([])
    assert summary["count"] == 0
    assert summary["median"] is None
    assert summary["mean"] is None


def test_auroc_is_one_when_separated_and_half_when_identical():
    assert analysis.auroc([0.8, 0.9], [0.1, 0.2]) == pytest.approx(1.0)
    assert analysis.auroc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)
    # Every score tied: neither class outranks the other, and ties must not be broken by
    # sort order — a detector that returns one constant discriminates nothing.
    assert analysis.auroc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_needs_both_classes():
    assert analysis.auroc([0.5], []) is None
    assert analysis.auroc([], [0.5]) is None


def test_clopper_pearson_bounds_a_zero_count():
    # Zero events in n trials bounds the rate at 1 - alpha**(1/n), which is the closed form
    # of the general bisection for k = 0.
    assert analysis.clopper_pearson_upper(0, 55) == pytest.approx(
        1 - 0.05 ** (1 / 55), abs=1e-9
    )
    assert analysis.clopper_pearson_upper(0, 110) == pytest.approx(0.0269, abs=1e-4)
    assert analysis.clopper_pearson_upper(3, 3) == 1.0
    assert analysis.clopper_pearson_upper(0, 0) is None


def test_high_threshold_sits_between_the_genuine_max_and_the_score_above_it():
    table = rows(
        ("g1", "real", 0.10),
        ("g2", "real", 0.40),
        ("m1", "face_swap", 0.30),
        ("m2", "synthetic", 0.60),
    )
    selected = analysis.select_high_threshold(table)
    assert selected["threshold"] == pytest.approx(0.50)
    assert selected["genuine_max"] == pytest.approx(0.40)
    # Zero observed false positives is the property the rule exists to guarantee.
    measured = analysis.sweep(table, [selected["threshold"]])[0]
    assert measured["false_positives"] == 0
    assert measured["true_positive_rate"] == pytest.approx(0.5)


def test_high_threshold_is_absent_when_genuine_media_reaches_the_ceiling():
    selected = analysis.select_high_threshold(
        rows(("g1", "real", 1.0), ("m1", "synthetic", 0.9))
    )
    assert selected["threshold"] is None
    assert "no threshold excludes it" in selected["reason"]


def test_low_threshold_sits_below_every_manipulated_score():
    table = rows(
        ("g1", "real", 0.05),
        ("g2", "real", 0.40),
        ("m1", "face_swap", 0.20),
        ("m2", "synthetic", 0.60),
    )
    selected = analysis.select_low_threshold(table)
    assert selected["threshold"] == pytest.approx(0.125)
    assert selected["coverage_genuine"] == pytest.approx(0.5)
    assert not [
        row for row in table
        if row["label"] != "real" and row["score"] < selected["threshold"]
    ]


def test_sweep_reports_rates_per_stratum_and_bounds_the_false_positive_rate():
    table = rows(
        ("g1", "real", 0.10),
        ("g2", "real", 0.90),
        ("m1", "synthetic", 0.95),
    )
    row = analysis.sweep(table, [0.5])[0]
    assert row["false_positives"] == 1
    assert row["false_positive_rate"] == pytest.approx(0.5)
    assert row["false_positive_rate_upper95"] > 0.5
    assert row["flag_rate_by_stratum"]["manipulated"] == pytest.approx(1.0)
    assert row["flag_rate_by_stratum"]["genuine_face"] == pytest.approx(0.5)


def test_sweep_grid_is_drawn_from_observed_scores():
    table = rows(("a", "real", 0.91), ("b", "real", 0.95), ("c", "synthetic", 0.99))
    grid = analysis.sweep_grid(table, steps=4)
    assert grid  # non-empty
    assert set(grid) <= {0.91, 0.95, 0.99}


def test_youden_point_finds_the_best_separation_without_selecting_it():
    table = rows(
        ("g1", "real", 0.10),
        ("g2", "real", 0.20),
        ("m1", "synthetic", 0.80),
        ("m2", "synthetic", 0.90),
    )
    best = analysis.youden_point(table)
    assert best["true_positive_rate"] == pytest.approx(1.0)
    assert best["false_positive_rate"] == pytest.approx(0.0)
    assert best["threshold"] == pytest.approx(0.80)


def test_spearman_is_one_for_the_same_ordering_and_minus_one_for_the_reverse():
    assert analysis.spearman([(1, 10), (2, 20), (3, 30)]) == pytest.approx(1.0)
    assert analysis.spearman([(1, 30), (2, 20), (3, 10)]) == pytest.approx(-1.0)
    assert analysis.spearman([(1, 1)]) is None
    # A constant column has no ordering to correlate with, which is undefined and not zero.
    assert analysis.spearman([(1, 5), (2, 5), (3, 5)]) is None


def test_joint_states_separates_disagreement_from_abstention():
    svd = [
        {"clip_id": "a", "label": "synthetic", "family": "t2v", "stratum": "generated",
         "score": 0.99},
        {"clip_id": "b", "label": "face_swap", "family": "swap", "stratum": "face_swap",
         "score": 0.10},
        {"clip_id": "c", "label": "real", "family": "real", "stratum": "genuine_face",
         "score": 0.10},
    ]
    face = [
        # No face in a text-to-video landscape clip: a refusal, never a "looks genuine".
        {"clip_id": "a", "label": "synthetic", "family": "t2v", "stratum": "generated",
         "score": None},
        {"clip_id": "b", "label": "face_swap", "family": "swap", "stratum": "face_swap",
         "score": 0.99},
        {"clip_id": "c", "label": "real", "family": "real", "stratum": "genuine_face",
         "score": 0.10},
    ]
    joint = analysis.joint_states(svd, face, 0.5, 0.5)
    states = {row["clip_id"]: row["state"] for row in joint}
    assert states["a"] == analysis.ABSTAINED_FACE
    assert states["b"] == analysis.DISAGREE_FACE_ONLY
    assert states["c"] == analysis.AGREE_CLEAR

    summary = analysis.disagreement_summary(joint)
    assert summary["overall"]["n"] == 3
    assert summary["overall"][analysis.DISAGREE_FACE_ONLY] == 1
    assert summary["overall"][analysis.ABSTAINED_FACE] == 1
    assert summary["clips_scored_by_both"] == 2
    assert summary["by_stratum"]["face_swap"][analysis.DISAGREE_FACE_ONLY] == 1


def test_joint_states_treats_an_absent_threshold_as_flagging_nothing():
    svd = [{"clip_id": "a", "label": "real", "family": "r", "stratum": "genuine_face",
            "score": 0.99}]
    face = [{"clip_id": "a", "label": "real", "family": "r", "stratum": "genuine_face",
             "score": 0.99}]
    joint = analysis.joint_states(svd, face, None, 0.5)
    assert joint[0]["synthetic_video_flagged"] is False
    assert joint[0]["face_manipulation_flagged"] is True
    assert joint[0]["state"] == analysis.DISAGREE_FACE_ONLY


def test_no_analysis_function_produces_a_combined_score():
    """Rule 11, enforced on this module's own surface.

    Every public name here either describes one detector or tabulates the two detectors'
    separate decisions. Nothing returns a fused score, and this test fails if a function
    that does is added later.
    """
    forbidden = {"combined_score", "fused_score", "average_score", "blend", "mean_score"}
    assert forbidden.isdisjoint(dir(analysis))
    joint = analysis.joint_states(
        [{"clip_id": "a", "label": "real", "family": "r", "stratum": "s", "score": 0.2}],
        [{"clip_id": "a", "label": "real", "family": "r", "stratum": "s", "score": 0.8}],
        0.5,
        0.5,
    )
    # The joint row carries both scores side by side and no third number derived from them.
    numeric = {
        key for key, value in joint[0].items()
        if isinstance(value, float) and not math.isnan(value)
    }
    assert numeric == {"synthetic_video_score", "face_manipulation_score"}
