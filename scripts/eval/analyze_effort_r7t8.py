"""Derive, freeze and then spend the R7-T8 operating threshold, in that order.

Two subcommands, and the order between them is the study design rather than a convenience:

    # 1. Derive the candidate on the new calibration split and freeze it. Runs before any
    #    validation clip has been scored, and refuses to overwrite an existing freeze.
    PYTHONPATH=scripts python3 scripts/eval/analyze_effort_r7t8.py freeze \
        --corpus ../deepguard-corpus/r7t8/corpus.json \
        --calibration-run ../deepguard-corpus/runs/r7t8-effort-calibration/results.json \
        --protocol scripts/eval/r7t8_protocol.json \
        --output ../deepguard-corpus/r7t8/threshold_frozen.json

    # 2. Score the frozen threshold against the validation split, once.
    PYTHONPATH=scripts python3 scripts/eval/analyze_effort_r7t8.py analyze \
        --corpus ../deepguard-corpus/r7t8/corpus.json \
        --calibration-run ../deepguard-corpus/runs/r7t8-effort-calibration/results.json \
        --validation-run ../deepguard-corpus/runs/r7t8-effort-validation/results.json \
        --frozen-threshold ../deepguard-corpus/r7t8/threshold_frozen.json \
        --independence ../deepguard-corpus/r7t8/independence.json \
        --freeze scripts/eval/effort_freeze.json \
        --protocol scripts/eval/r7t8_protocol.json \
        --r7t7-analysis ../deepguard-corpus/r7t5/analysis-effort/effort_analysis.json \
        --output-dir ../deepguard-corpus/r7t8/analysis

Every measurement rule is imported from `analyze_effort` rather than restated, so that R7-T7's
numbers and this task's are produced by the same arithmetic and a difference between them is a
difference in the data. The threshold selection rule is imported for the same reason: it is
`analyze_effort.calibration_candidate`, unmodified, applied to a different calibration split.

**Three things this module adds to R7-T7's analysis.**

*The regression lineages leave the headline.* R7-T7 counted its two private lineages inside the
genuine evaluation denominator, correctly, because they were independent of its calibration
split. Here they are the one declared re-use in the corpus, so counting them would put two
lineages R7-T7 already measured into a figure whose entire claim is independence. They are
excluded from the primary genuine denominator by family name and reported in their own section.

*The power requirement is a gate, not a footnote.* `readable_genuine_n` is compared against the
predeclared 299 and the comparison decides whether Recommendation A is reachable at all, before
any false-positive count is looked at.

*The decision gate is evaluated in code.* A1 to A5 are read from the protocol, evaluated
mechanically, and the recommendation falls out of them. A criterion cannot be quietly softened in
prose if the artifact says which ones passed.

Threshold drift is computed and reported and is deliberately absent from the gate, as
`threshold_stability_handling` in the protocol predeclared (Option B, descriptive only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import stats
from eval.analyze_effort import (
    OOD_LABEL,
    PRIMARY_LABEL,
    _distribution,
    _lineage_rate,
    _readable,
    abstention_structure,
    auroc,
    calibration_candidate,
    compute_cost,
    derivative_robustness,
    failure_report,
    genuine_primary,
    join_run,
    manipulated_report,
    regression_report,
    threshold_sensitivity,
)
from eval.build_corpus_r7t8 import REGRESSION_FAMILY
from eval.corpus import (
    LABEL_REAL,
    SPLIT_CALIBRATION,
    SPLIT_EVALUATION,
    read_corpus,
)

# The R7-T7 candidate this study exists to test. Used for descriptive comparison only; no
# criterion anywhere in this module is stated in terms of proximity to it.
R7T7_CANDIDATE_THRESHOLD = 0.96612

# The lip-sync family carries the face-manipulation label because it is a manipulation of a real
# face, and it is not a swap. It is reported inside the primary task and broken out beside it, so
# neither figure is a silent average of the two.
LIPSYNC_FAMILY = "lipsync_wav2lip"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_genuine(records: list[dict]) -> list[dict]:
    """Genuine records excluding the declared-reuse regression lineages.

    The regression lineages are benign genuine media and they are re-used by design. Every
    headline genuine figure in this task is a claim about *independent* lineages, so they are
    removed here and reported in `regression`. Removing them costs two observations and buys the
    only thing the study is for.
    """
    return [record for record in records if record["family"] != REGRESSION_FAMILY]


def freeze_threshold(argv: argparse.Namespace) -> int:
    items = read_corpus(argv.corpus)
    run = json.loads(argv.calibration_run.read_text(encoding="utf-8"))
    records = join_run(items, run)

    splits = sorted({record["split"] for record in records})
    if splits != [SPLIT_CALIBRATION]:
        print(
            f"refusing to derive a threshold: the calibration run covers splits {splits}, "
            "which is not the calibration split alone",
            file=sys.stderr,
        )
        return 1
    if argv.output.exists():
        print(
            f"{argv.output} already exists. A frozen threshold is written once; delete it "
            "deliberately if the calibration split is genuinely being rebuilt.",
            file=sys.stderr,
        )
        return 1

    candidate = calibration_candidate(records)
    if not candidate.get("available") or candidate.get("midpoint_candidate") is None:
        print(
            "the calibration split does not support the predeclared selection rule: no "
            "manipulated score lies above the highest genuine score",
            file=sys.stderr,
        )
        json.dumps(candidate)
        return 1

    threshold = candidate["midpoint_candidate"]
    genuine = _readable(
        _independent_genuine(
            [r for r in records if r["label"] == LABEL_REAL and r["is_base"]]
        )
    )
    manipulated = _readable([r for r in records if r["is_manipulated"] and r["is_base"]])

    payload = {
        "schema_version": "r7-t8-frozen-threshold-1",
        "task": "R7-T8",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_before_validation_split_was_scored": True,
        "selection_rule": (
            "midpoint between the highest readable genuine base-clip score on the calibration "
            "split and the lowest readable manipulated base-clip score above it"
        ),
        "rule_implementation": "eval.analyze_effort.calibration_candidate, unmodified",
        "derived_on": {
            "split": SPLIT_CALIBRATION,
            "corpus": str(argv.corpus),
            "calibration_run": str(argv.calibration_run),
            "calibration_run_sha256": _digest(argv.calibration_run),
            "genuine_base_lineages_readable": len(
                {r["source_lineage_id"] for r in genuine}
            ),
            "manipulated_base_lineages_readable": len(
                {r["source_lineage_id"] for r in manipulated}
            ),
        },
        "candidate": candidate,
        "threshold": threshold,
        "genuine_calibration_score_distribution": _distribution(
            [r["score"] for r in genuine]
        ),
        "not_anchored": (
            "Derived from this calibration split alone by the predeclared rule. It was not "
            f"compared to the R7-T7 candidate ({R7T7_CANDIDATE_THRESHOLD}) before being fixed, "
            "and no acceptance criterion is stated in terms of the distance between them."
        ),
    }
    argv.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {argv.output}")
    print(f"highest genuine calibration score: {candidate['highest_genuine_score']}")
    print(
        "lowest manipulated score above it: "
        f"{candidate['lowest_manipulated_score_above_it']}"
    )
    print(f"FROZEN THRESHOLD: {threshold}")
    return 0


def _sub_aggregate(records: list[dict], families: set[str], threshold: float) -> dict:
    """A detection rate over a named subset of the primary families, at lineage level."""
    base = [
        record
        for record in records
        if record["label"] == PRIMARY_LABEL
        and record["is_base"]
        and record["family"] in families
    ]
    readable = _readable(base)
    detected = _lineage_rate(
        readable, lambda r: r["score"] is not None and r["score"] > threshold
    )
    return {
        "families": sorted(families),
        "lineages_total": len({r["source_lineage_id"] for r in base}),
        "lineages_readable": len({r["source_lineage_id"] for r in readable}),
        "detected": detected.as_dict(),
        "detected_summary": detected.describe(),
        "score_distribution": _distribution([r["score"] for r in readable]),
    }


def threshold_stability(
    calibration: list[dict],
    validation: list[dict],
    frozen: dict,
    r7t7_analysis: dict | None,
) -> dict:
    """What moved, reported and not judged.

    Predeclared as Option B in the protocol: threshold drift is descriptive. This function
    therefore computes the movement, the two order statistics on each side that produced it, the
    genuine score distributions, and the cross-tabulation of what each corpus does at each
    threshold - and returns no verdict, because it was declared in advance that there would not
    be one.
    """
    threshold = frozen["threshold"]
    drift = threshold - R7T7_CANDIDATE_THRESHOLD

    genuine_validation = _readable(
        _independent_genuine(
            [r for r in validation if r["label"] == LABEL_REAL and r["is_base"]]
        )
    )

    def crossings(records: list[dict], value: float) -> dict:
        rate = _lineage_rate(
            records, lambda r: r["score"] is not None and r["score"] > value
        )
        return {"threshold": value, **rate.as_dict(), "summary": rate.describe()}

    r7t7_genuine = None
    if r7t7_analysis is not None:
        r7t7_genuine = {
            "evaluation_score_distribution": r7t7_analysis["evaluation"][
                "genuine_primary"
            ]["score_distribution"],
            "false_positives_at_r7t7_candidate": r7t7_analysis.get("candidate_threshold_view"),
            "candidate_threshold": r7t7_analysis.get("evaluation", {})
            .get("calibration_derived", {})
            .get("candidate_threshold"),
        }

    return {
        "handling": "Option B, descriptive only, predeclared in r7t8_protocol.json",
        "not_a_criterion": (
            "Threshold movement is not a pass/fail criterion in this task and no tolerance was "
            "declared. The measured quantity that carries the safety claim is the false-positive "
            "rate at the frozen threshold on independent data, reported separately."
        ),
        "r7t7_candidate_threshold": R7T7_CANDIDATE_THRESHOLD,
        "r7t8_frozen_threshold": threshold,
        "signed_drift": drift,
        "absolute_drift": abs(drift),
        "what_produced_each_midpoint": {
            "r7t8": {
                "highest_genuine_calibration_score": frozen["candidate"][
                    "highest_genuine_score"
                ],
                "lowest_manipulated_score_above_it": frozen["candidate"][
                    "lowest_manipulated_score_above_it"
                ],
                "genuine_calibration_lineages": frozen["candidate"]["genuine_lineages"],
            },
            "r7t7": (
                r7t7_analysis.get("evaluation", {}).get("calibration_derived")
                if r7t7_analysis
                else None
            ),
        },
        "genuine_score_distribution": {
            "r7t8_calibration": frozen["genuine_calibration_score_distribution"],
            "r7t8_validation_independent": _distribution(
                [r["score"] for r in genuine_validation]
            ),
            "r7t7_evaluation": (
                r7t7_genuine["evaluation_score_distribution"] if r7t7_genuine else None
            ),
        },
        "cross_tabulation": {
            "note": (
                "The R7-T8 validation genuine lineages, counted at both operating points. The "
                "row at the R7-T7 candidate is descriptive: it is what the previous task's "
                "threshold would have done to this corpus, and it is not the figure any "
                "criterion is stated over."
            ),
            "r7t8_validation_at_r7t8_frozen": crossings(
                genuine_validation, threshold
            ),
            "r7t8_validation_at_r7t7_candidate": crossings(
                genuine_validation, R7T7_CANDIDATE_THRESHOLD
            ),
        },
    }


def evaluate_gate(
    protocol: dict,
    genuine: dict,
    primary: dict,
    regression: list[dict],
    independence: dict,
) -> dict:
    """A1 to A5, then the C conditions, mechanically, from the predeclared protocol."""
    required_n = protocol["statistical_power"]["required_readable_genuine_n"]
    target = protocol["statistical_power"]["target"]["threshold"]

    readable_n = genuine["lineages_readable"]
    upper = genuine["false_positives"]["upper_95_one_sided"]
    observed_k = genuine["false_positives"]["k"]

    detected = primary["detected"]
    detection_lower = detected.get("lower_95_one_sided")
    detection_upper = detected.get("upper_95_one_sided")
    detection_observed = detected.get("observed")

    regression_observed = [
        entry
        for entry in regression
        if entry["as_acquired"] is not None and entry["as_acquired"]["status"] == "ok"
    ]
    regression_flagged = [
        entry for entry in regression_observed if entry["as_acquired"]["flagged"]
    ]

    criteria = {
        "A1": {
            "criterion": (
                "one-sided 95% upper bound on genuine FPR at the frozen threshold "
                f"<= {target}"
            ),
            "measured": upper,
            "passed": upper is not None and upper <= target,
        },
        "A2": {
            "criterion": f"readable genuine lineage n in the validation split >= {required_n}",
            "measured": readable_n,
            "passed": readable_n >= required_n,
        },
        "A3": {
            "criterion": "gating independence checks pass with zero findings",
            "measured": independence.get("gating_checks_passed"),
            "passed": bool(independence.get("gating_checks_passed")),
        },
        "A4": {
            "criterion": (
                "neither benign regression lineage crosses the frozen threshold as acquired; "
                "an abstention is a non-observation and cannot satisfy this"
            ),
            "measured": {
                "lineages": len(regression),
                "observed": len(regression_observed),
                "flagged": len(regression_flagged),
            },
            "passed": bool(regression_observed) and not regression_flagged,
        },
        "A5": {
            "criterion": (
                "face-manipulation detection at the frozen threshold is non-zero and its "
                "one-sided 95% lower bound is above 0.05"
            ),
            "measured": {
                "observed": detection_observed,
                "lower_95_one_sided": detection_lower,
            },
            "passed": bool(detection_observed) and (detection_lower or 0.0) > 0.05,
        },
    }

    failed = [name for name, entry in criteria.items() if not entry["passed"]]

    reject_conditions = {
        "genuine_fpr_upper_bound_above_0.05": upper is not None and upper > 0.05,
        "primary_detection_indistinguishable_from_zero": (
            detection_upper is not None and detection_upper < 0.10
        ),
        "independence_defect": not bool(independence.get("gating_checks_passed")),
    }
    reject = [name for name, hit in reject_conditions.items() if hit]

    if not failed:
        recommendation = "A"
        label = "ELIGIBILITY STUDY PASSED"
    elif reject:
        recommendation = "C"
        label = "REJECT"
    else:
        recommendation = "B"
        label = "CONTINUE SHADOW"

    return {
        "criteria": criteria,
        "criteria_failed": failed,
        "reject_conditions": reject_conditions,
        "reject_conditions_met": reject,
        "recommendation": recommendation,
        "recommendation_label": label,
        "threshold_drift_is_not_a_criterion": True,
        "power_gate": {
            "required_readable_genuine_n": required_n,
            "readable_genuine_n": readable_n,
            "sufficient": readable_n >= required_n,
            "observed_false_positives": observed_k,
            "recommendation_a_reachable": readable_n >= required_n,
        },
    }


def run_analysis(argv: argparse.Namespace) -> int:
    items = read_corpus(argv.corpus)
    calibration_run = json.loads(argv.calibration_run.read_text(encoding="utf-8"))
    validation_run = json.loads(argv.validation_run.read_text(encoding="utf-8"))
    frozen = json.loads(argv.frozen_threshold.read_text(encoding="utf-8"))
    independence = json.loads(argv.independence.read_text(encoding="utf-8"))
    freeze = json.loads(argv.freeze.read_text(encoding="utf-8"))
    protocol = json.loads(argv.protocol.read_text(encoding="utf-8"))
    r7t7_analysis = (
        json.loads(argv.r7t7_analysis.read_text(encoding="utf-8"))
        if argv.r7t7_analysis
        else None
    )

    threshold = frozen["threshold"]
    calibration = join_run(items, calibration_run)
    validation = join_run(items, validation_run)

    split_integrity = {
        "calibration_run_contains_only": sorted({r["split"] for r in calibration}),
        "validation_run_contains_only": sorted({r["split"] for r in validation}),
        "frozen_threshold_derived_on": frozen["derived_on"]["split"],
        "calibration_run_digest_matches_freeze": (
            _digest(argv.calibration_run) == frozen["derived_on"]["calibration_run_sha256"]
        ),
    }
    split_integrity["clean"] = (
        split_integrity["calibration_run_contains_only"] == [SPLIT_CALIBRATION]
        and split_integrity["validation_run_contains_only"] == [SPLIT_EVALUATION]
        and split_integrity["calibration_run_digest_matches_freeze"]
    )

    independent_validation = _independent_genuine(validation)
    genuine = genuine_primary(independent_validation, threshold)
    primary = manipulated_report(validation, PRIMARY_LABEL, threshold)
    ood = manipulated_report(validation, OOD_LABEL, threshold)
    regression = regression_report(validation, threshold)

    swap_families = {
        family
        for family in primary["by_family"]
        if family != LIPSYNC_FAMILY
    }
    primary["sub_aggregates"] = {
        "swap_families_only": _sub_aggregate(validation, swap_families, threshold),
        "lipsync_only": _sub_aggregate(validation, {LIPSYNC_FAMILY}, threshold),
        "note": (
            "Lip-sync manipulates a real face without replacing it. It carries the face "
            "manipulation label and is reported inside the primary task, and it is broken out "
            "here so neither figure is a silent average of two different manipulation types."
        ),
    }

    genuine_scores = [
        r["score"]
        for r in _readable(
            _independent_genuine(
                [r for r in validation if r["label"] == LABEL_REAL and r["is_base"]]
            )
        )
    ]
    primary_scores = [
        r["score"]
        for r in _readable(
            [r for r in validation if r["label"] == PRIMARY_LABEL and r["is_base"]]
        )
    ]
    ood_scores = [
        r["score"]
        for r in _readable(
            [r for r in validation if r["label"] == OOD_LABEL and r["is_base"]]
        )
    ]

    gate = evaluate_gate(protocol, genuine, primary, regression, independence)

    analysis = {
        "schema_version": "r7-t8-effort-analysis-1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "R7-T8",
        "protocol": {
            "path": str(argv.protocol),
            "sha256": _digest(argv.protocol),
            "threshold_stability_handling": protocol["threshold_stability_handling"][
                "option_taken"
            ],
        },
        "candidate": {
            "detector": "effort",
            "variant": freeze["candidate"]["checkpoint"]["variant"],
            "checkpoint_sha256": freeze["candidate"]["checkpoint"]["sha256"],
            "upstream_revision": freeze["candidate"]["upstream_revision"],
            "preprocessing_and_inference": "unchanged from R7-T7",
        },
        "frozen_threshold": {
            "value": threshold,
            "sha256": _digest(argv.frozen_threshold),
            "frozen_at": frozen["frozen_at"],
            "derived_on": frozen["derived_on"],
            "selection_rule": frozen["selection_rule"],
        },
        "independence": {
            "gating_checks_passed": independence.get("gating_checks_passed"),
            "shared_lineage_ids": independence["cross_task_identity"]["shared_lineage_ids"],
            "shared_sha256": independence["cross_task_identity"]["shared_sha256"],
            "ffpp_reachability_violations": independence["cross_task_identity"][
                "ffpp_reachability_violations"
            ],
            "declared_reuse": independence["cross_task_identity"]["reused_as_declared"],
            "within_corpus_leaked": independence["within_corpus_split_integrity"]["leaked"],
            "perceptual_cross_task_pairs": independence["cross_task_perceptual"].get(
                "pair_count"
            ),
        },
        "split_integrity": split_integrity,
        "validation": {
            "genuine_primary_independent": genuine,
            "primary_task_face_manipulation": primary,
            "cross_family_ood_generated": ood,
            "auroc": {
                "genuine_vs_face_manipulation": auroc(primary_scores, genuine_scores),
                "genuine_vs_generated_ood": auroc(ood_scores, genuine_scores),
                "note": (
                    "Threshold-free separation. Reported for the primary task and the "
                    "out-of-distribution task separately and never pooled."
                ),
            },
            "derivative_robustness": derivative_robustness(
                independent_validation, threshold
            ),
            "abstention_by_frames_with_face": abstention_structure(validation, threshold),
            "failures": failure_report(validation),
            "threshold_sensitivity": threshold_sensitivity(
                independent_validation,
                [0.5, 0.9, R7T7_CANDIDATE_THRESHOLD, threshold, 0.99],
            ),
        },
        "regression": {
            "note": (
                "The two benign real-world lineages, the one declared re-use in this corpus. "
                "Excluded from every headline denominator above; reported here at the frozen "
                "threshold. An abstention is recorded as a non-observation, not a pass."
            ),
            "lineages": regression,
        },
        "threshold_stability": threshold_stability(
            calibration, validation, frozen, r7t7_analysis
        ),
        "decision_gate": gate,
        "calibration_reference": {
            "note": (
                "The calibration split, for reference only. No criterion is stated over it and "
                "the threshold was derived here before the validation split was opened."
            ),
            "genuine_primary_independent": genuine_primary(
                _independent_genuine(calibration), threshold
            ),
            "primary_task_face_manipulation": manipulated_report(
                calibration, PRIMARY_LABEL, threshold
            ),
            "cross_family_ood_generated": manipulated_report(
                calibration, OOD_LABEL, threshold
            ),
            "failures": failure_report(calibration),
        },
        "compute": {
            "calibration": compute_cost(calibration_run),
            "validation": compute_cost(validation_run),
        },
    }

    argv.output_dir.mkdir(parents=True, exist_ok=True)
    destination = argv.output_dir / "effort_r7t8_analysis.json"
    destination.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    print(f"frozen threshold: {threshold}")
    print(f"split integrity clean: {split_integrity['clean']}")
    print(f"independence gating passed: {independence.get('gating_checks_passed')}")
    print(
        "genuine (independent, lineage-level): "
        f"{genuine['false_positive_summary']}"
    )
    print(
        f"readable genuine n: {genuine['lineages_readable']} "
        f"(required {protocol['statistical_power']['required_readable_genuine_n']})"
    )
    print(f"face manipulation detected: {primary['detected_summary']}")
    print(f"generated OOD detected: {ood['detected_summary']}")
    print(
        f"threshold drift vs R7-T7 candidate: "
        f"{analysis['threshold_stability']['signed_drift']:+.5f} (descriptive only)"
    )
    print(f"criteria failed: {gate['criteria_failed'] or 'none'}")
    print(f"RECOMMENDATION: {gate['recommendation']} - {gate['recommendation_label']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_effort_r7t8",
        description="Derive, freeze and spend the R7-T8 operating threshold.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="derive the candidate and freeze it")
    freeze.add_argument("--corpus", required=True, type=Path)
    freeze.add_argument("--calibration-run", required=True, type=Path)
    freeze.add_argument("--protocol", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    analyze = sub.add_parser("analyze", help="score the frozen threshold, once")
    analyze.add_argument("--corpus", required=True, type=Path)
    analyze.add_argument("--calibration-run", required=True, type=Path)
    analyze.add_argument("--validation-run", required=True, type=Path)
    analyze.add_argument("--frozen-threshold", required=True, type=Path)
    analyze.add_argument("--independence", required=True, type=Path)
    analyze.add_argument("--freeze", required=True, type=Path)
    analyze.add_argument("--protocol", required=True, type=Path)
    analyze.add_argument("--r7t7-analysis", type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        return freeze_threshold(args)
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
