"""The three rules the analysis must not break: lineages, separate targets, honest abstentions."""

import pytest

from eval import analyze
from eval.corpus import SPLIT_CALIBRATION, SPLIT_EVALUATION


def record(clip_id, lineage, score, label="real", family="mavos_real_en", status="ok",
           split=SPLIT_EVALUATION, derivation="none", stratum="genuine_in_the_wild"):
    return {
        "clip_id": clip_id,
        "split": split,
        "label": label,
        "family": family,
        "stratum": stratum,
        "derivation": derivation,
        "is_base": derivation == "none",
        "private": False,
        "source_lineage_id": lineage,
        "status": status,
        "score": score,
        "units_scored": 4,
        "error": None,
    }


LIP = next(detector for detector in analyze.DETECTORS if detector.key == "lip")
SVD = next(detector for detector in analyze.DETECTORS if detector.key == "svd")


def test_a_lineage_counts_once_however_many_derivatives_it_has():
    """Nine transcodes of one recording are one observation, not nine.

    Without this the false-positive denominator is decided by the derivative budget rather
    than by how many independent recordings were actually tested.
    """
    records = [record("a", "l:1", 0.9)] + [
        record(f"a__{index}", "l:1", 0.9, derivation=f"t{index}") for index in range(8)
    ]
    rate = analyze._lineage_rate(records, lambda r: r["score"] >= 0.5)
    assert rate.numerator == 1
    assert rate.denominator == 1


def test_one_flagged_derivative_makes_its_lineage_a_false_positive():
    records = [
        record("a", "l:1", 0.1),
        record("a__t", "l:1", 0.9, derivation="social_transcode_shorts"),
    ]
    rate = analyze._lineage_rate(records, lambda r: r["score"] >= 0.5)
    assert rate.numerator == 1
    assert rate.denominator == 1


def test_abstentions_leave_both_sides_of_the_false_positive_rate():
    """A clip the detector could not read may not quietly count as a correct negative."""
    records = [
        record("a", "l:1", 0.1),
        record("b", "l:2", None, status="abstained"),
        record("c", "l:3", 0.9),
    ]
    report = analyze.detector_report(LIP, records, 0.5, SPLIT_EVALUATION)
    false_positives = report["genuine"]["false_positive_rate_by_lineage"]
    assert false_positives["n"] == 2
    assert false_positives["k"] == 1
    abstentions = report["genuine"]["abstention_rate_by_lineage"]
    assert abstentions["k"] == 1
    assert abstentions["n"] == 3


def test_primary_task_and_cross_family_are_never_pooled():
    """LipForensics finding no text-to-video is a scope statement, not a missed detection."""
    records = [
        record("swap", "l:1", 0.99, label="face_swap", family="faceswap_roop",
               stratum="face_swap"),
        record("t2v", "l:2", 0.01, label="synthetic", family="t2v_veo3", stratum="generated"),
    ]
    report = analyze.detector_report(LIP, records, 0.5, SPLIT_EVALUATION)
    assert report["primary_task"]["detection_rate_by_lineage"]["k"] == 1
    assert report["primary_task"]["detection_rate_by_lineage"]["n"] == 1
    assert report["cross_family"]["detection_rate_by_lineage"]["k"] == 0
    assert report["cross_family"]["detection_rate_by_lineage"]["n"] == 1


def test_each_detector_gets_its_own_target_family():
    """The same clip is primary evidence for one detector and cross-family for another."""
    records = [
        record("t2v", "l:2", 0.99, label="synthetic", family="t2v_veo3", stratum="generated")
    ]
    assert (
        analyze.detector_report(SVD, records, 0.5, SPLIT_EVALUATION)["primary_task"][
            "detection_rate_by_lineage"
        ]["n"]
        == 1
    )
    assert (
        analyze.detector_report(LIP, records, 0.5, SPLIT_EVALUATION)["cross_family"][
            "detection_rate_by_lineage"
        ]["n"]
        == 1
    )


def test_a_false_positive_rate_carries_a_bound_and_never_a_bare_zero():
    records = [record(f"c{i}", f"l:{i}", 0.01) for i in range(50)]
    report = analyze.detector_report(LIP, records, 0.5, SPLIT_EVALUATION)
    rate = report["genuine"]["false_positive_rate_by_lineage"]
    assert rate["k"] == 0
    assert rate["observed"] == 0.0
    assert rate["upper_95_one_sided"] > 0.0


def test_private_clips_are_named_by_lineage_only():
    """No private clip id reaches an example table, whatever else the record carries."""
    private = record("private_regression-1", "private:regression-1", 0.9)
    private["private"] = True
    example = analyze._example(private)
    assert example["clip_id"] is None
    assert example["source_lineage_id"] == "private:regression-1"


def test_threshold_candidates_come_from_the_calibration_split_only():
    """A candidate derived on the evaluation split would have read the hold-out."""
    records = [
        record("cal_real", "l:1", 0.10, split=SPLIT_CALIBRATION),
        record("cal_swap", "l:2", 0.80, label="face_swap", family="ffpp_deepfakes",
               split=SPLIT_CALIBRATION, stratum="face_swap"),
        record("eval_real", "l:3", 0.99, split=SPLIT_EVALUATION),
    ]
    candidates = analyze.threshold_candidates(records, 0.5)
    assert candidates["genuine_n"] == 1
    assert candidates["highest_genuine_score"] == 0.10
    assert candidates["candidate_midpoint"]["threshold"] == pytest.approx(0.45)
    assert candidates["candidate_midpoint"]["genuine_flagged"] == 0


def test_quantiles_return_observed_values_rather_than_interpolations():
    ordered = [0.1, 0.2, 0.3, 0.4]
    assert analyze._quantile(ordered, 0.0) == 0.1
    assert analyze._quantile(ordered, 1.0) == 0.4
    assert analyze._quantile(ordered, 0.5) in ordered
