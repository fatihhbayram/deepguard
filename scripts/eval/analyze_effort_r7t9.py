"""Derive, freeze and then spend the R7-T9 operating threshold, in that order.

    # 1. Derive the candidate on the new calibration split and freeze it, before any
    #    validation clip has been scored. Refuses to overwrite an existing freeze.
    PYTHONPATH=scripts python3 scripts/eval/analyze_effort_r7t9.py freeze \
        --corpus ../deepguard-corpus/r7t9/corpus.json \
        --calibration-run ../deepguard-corpus/runs/r7t9-effort-calibration/results.json \
        --protocol scripts/eval/r7t9_protocol.json \
        --output ../deepguard-corpus/r7t9/threshold_frozen.json

    # 2. Score the frozen threshold against the validation split, once.
    PYTHONPATH=scripts python3 scripts/eval/analyze_effort_r7t9.py analyze \
        --corpus ../deepguard-corpus/r7t9/corpus.json \
        --calibration-run ../deepguard-corpus/runs/r7t9-effort-calibration/results.json \
        --validation-run ../deepguard-corpus/runs/r7t9-effort-validation/results.json \
        --frozen-threshold ../deepguard-corpus/r7t9/threshold_frozen.json \
        --independence ../deepguard-corpus/r7t9/independence.json \
        --freeze scripts/eval/effort_freeze.json \
        --protocol scripts/eval/r7t9_protocol.json \
        --r7t8-analysis ../deepguard-corpus/r7t8/analysis/effort_r7t8_analysis.json \
        --output-dir ../deepguard-corpus/r7t9/analysis

**What is different here from R7-T8, and why.** R7-T8 selected its operating point with the
midpoint rule inherited from R4-T1: halfway between the highest genuine calibration score and
the lowest manipulated score above it. That rule wedged a threshold into a gap of 0.000075
between two order statistics and bought a detection rate of 3.6%. Its failure was not the
detector's ranking ability but the selection rule, so this task replaces the rule and nothing
else. Model, checkpoint, preprocessing, frame count, aggregation and abstention semantics are
inherited verbatim; only the arithmetic that turns calibration scores into a threshold changes.

**The candidate set is exact, not a grid.** With the frozen decision rule `score > threshold`,
two thresholds are decision-equivalent on a finite sample when no observed score separates
them. The observed readable calibration base-clip scores `s_1 < ... < s_m` therefore partition
the real line into exactly `m + 1` equivalence classes, and this module enumerates one
representative of each: `nextafter(s_1, -inf)` for the class that flags everything, and `s_i`
for the class `[s_i, s_{i+1})`. That is the complete set of distinct classification outcomes
available on this calibration sample, with no decimal quantisation anywhere and no arbitrary
increment to defend.

**The selection rule is a safety constraint first.** Every candidate is scored for its exact
one-sided 95% Clopper-Pearson upper bound on the genuine false-positive rate; a candidate whose
bound exceeds the predeclared 1% is discarded before detection is looked at. Among what
survives, the highest observed intended-target TPR wins, ties broken by lower observed genuine
FPR and then by higher threshold. If nothing survives, no threshold is invented and the
eligibility path fails on the safety constraint — the validation split stays closed.

Every measurement helper is imported from `analyze_effort` rather than restated, so a difference
between R7-T8's numbers and these is a difference in the data or in the selection rule and never
in the arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import stats
from eval.analyze_effort import (
    OOD_LABEL,
    PRIMARY_LABEL,
    _distribution,
    _flagged,
    _lineage_rate,
    _readable,
    abstention_structure,
    auroc,
    compute_cost,
    derivative_robustness,
    failure_report,
    genuine_primary,
    join_run,
    manipulated_report,
    regression_report,
    threshold_sensitivity,
)
from eval.corpus import (
    LABEL_REAL,
    SPLIT_CALIBRATION,
    SPLIT_EVALUATION,
    read_corpus,
)

# The family name that carries the declared-reuse benign regression lineages. Keyed on rather
# than on `private`, because a future private lineage that is not a regression fixture must not
# inherit the exemption.
REGRESSION_FAMILY = "real_world_regression"

# Prior operating points, carried for descriptive comparison only. No criterion in this module is
# stated in terms of proximity to either.
R7T7_CANDIDATE_THRESHOLD = 0.96612
R7T8_FROZEN_THRESHOLD = 0.9874524883925915


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent(records: list[dict]) -> list[dict]:
    """Records excluding the declared-reuse regression lineages.

    Every headline figure in this task is a claim about lineages this study has not seen before.
    The regression fixtures are deliberately re-read media; they are removed from every
    denominator here and reported in their own section.
    """
    return [record for record in records if record["family"] != REGRESSION_FAMILY]


def _base(records: list[dict], label: str | None = None) -> list[dict]:
    """Base clips only, optionally of one label. Derivatives never enter a headline rate."""
    return [
        record
        for record in records
        if record["is_base"] and (label is None or record["label"] == label)
    ]


# --- The exact candidate set ---------------------------------------------------------------


def candidate_thresholds(scores: list[float]) -> list[float]:
    """One representative of every decision-equivalent threshold class on `scores`.

    Under `score > threshold`, thresholds `t` and `t'` classify every observation identically
    exactly when no observed score lies between them. So the `m` distinct observed scores induce
    `m + 1` classes, and this returns one representative of each, ascending:

    - `nextafter(s_1, -inf)`: the class `t < s_1`, which flags every observation. Represented by
      the largest float below the minimum rather than by an invented round number, so the value
      is derived from the data and nothing is quantised.
    - `s_i` for each observed score: the class `[s_i, s_{i+1})`, which flags everything strictly
      above `s_i`. The lower endpoint is the canonical representative and is itself an observed
      value, so a frozen threshold is always a number this corpus produced.

    The class above the maximum is `[s_m, inf)`, represented by `s_m`, which flags nothing; it is
    included because "flag nothing" is a possible outcome the selection rule has to be able to
    see and reject on usefulness rather than never consider.
    """
    distinct = sorted(set(scores))
    if not distinct:
        return []
    return [math.nextafter(distinct[0], -math.inf), *distinct]


def evaluate_candidates(
    genuine: list[dict],
    primary: list[dict],
    candidates: list[float],
) -> list[dict]:
    """Every candidate, with exact one-sided 95% bounds on both rates it decides.

    Genuine and intended-target rates are counted over distinct lineages, base clips only,
    abstentions already excluded by the caller. The same `_flagged` predicate the frozen protocol
    applies everywhere decides both.
    """
    rows = []
    for threshold in candidates:
        false_positives = _lineage_rate(genuine, lambda r: _flagged(r, threshold))
        detected = _lineage_rate(primary, lambda r: _flagged(r, threshold))
        rows.append(
            {
                "threshold": threshold,
                "genuine_false_positives": false_positives.numerator,
                "genuine_readable_lineages": false_positives.denominator,
                "genuine_fpr_observed": false_positives.observed,
                "genuine_fpr_upper_95": false_positives.upper_95,
                "primary_detected": detected.numerator,
                "primary_readable_lineages": detected.denominator,
                "primary_tpr_observed": detected.observed,
                "primary_tpr_lower_95": detected.lower_95,
                "primary_tpr_upper_95": detected.upper_95,
            }
        )
    return rows


def select_operating_point(rows: list[dict], fpr_upper_bound_max: float) -> dict:
    """The predeclared FPR-constrained selection, applied deterministically.

    1. Discard every candidate whose exact one-sided 95% upper bound on the genuine FPR exceeds
       the safety target. This happens before any detection number is consulted, so a threshold
       can never be bought with false positives.
    2. Among the survivors take the highest observed intended-target TPR.
    3. Tie-break on the lower observed genuine FPR, then on the higher threshold.

    Returns the selection and the full reasoning, including the empty case. An empty surviving
    set is a result, not an error to route around: it means the safety constraint cannot be met
    on this calibration sample and no threshold may be invented.
    """
    surviving = [
        row
        for row in rows
        if row["genuine_fpr_upper_95"] is not None
        and row["genuine_fpr_upper_95"] <= fpr_upper_bound_max
    ]
    if not surviving:
        return {
            "constraint": f"genuine FPR one-sided 95% upper bound <= {fpr_upper_bound_max}",
            "candidates_evaluated": len(rows),
            "candidates_surviving": 0,
            "selected": None,
            "why": (
                "No candidate operating point on this calibration split satisfies the "
                "predeclared genuine false-positive safety constraint. The constraint is not "
                "weakened, no threshold is invented, and the final validation split is not "
                "opened to search for one."
            ),
        }

    chosen = min(
        surviving,
        key=lambda row: (
            -(row["primary_tpr_observed"] or 0.0),
            row["genuine_fpr_observed"] if row["genuine_fpr_observed"] is not None else 1.0,
            -row["threshold"],
        ),
    )

    best_tpr = chosen["primary_tpr_observed"]
    tied_on_tpr = [row for row in surviving if row["primary_tpr_observed"] == best_tpr]
    tied_on_fpr = [
        row
        for row in tied_on_tpr
        if row["genuine_fpr_observed"] == chosen["genuine_fpr_observed"]
    ]

    return {
        "constraint": f"genuine FPR one-sided 95% upper bound <= {fpr_upper_bound_max}",
        "candidates_evaluated": len(rows),
        "candidates_surviving": len(surviving),
        "surviving_threshold_range": {
            "min": min(row["threshold"] for row in surviving),
            "max": max(row["threshold"] for row in surviving),
        },
        "selected": chosen,
        "tie_break": {
            "candidates_at_max_tpr": len(tied_on_tpr),
            "of_those_at_min_observed_fpr": len(tied_on_fpr),
            "rule": (
                "maximum observed intended-target TPR, then minimum observed genuine FPR, then "
                "highest threshold"
            ),
            "decided_by": (
                "TPR alone"
                if len(tied_on_tpr) == 1
                else ("observed FPR" if len(tied_on_fpr) == 1 else "highest threshold")
            ),
        },
        "why": (
            "Highest observed intended-target TPR among the candidates whose exact one-sided "
            "95% genuine FPR upper bound satisfies the predeclared safety constraint."
        ),
    }


# --- The abstention / core coverage gate ---------------------------------------------------


def core_coverage(records: list[dict], protocol: dict) -> dict:
    """Per-stratum genuine coverage against the predeclared core acquisition strata.

    An abstention is neither a false positive nor a true negative; it is the detector declining
    to answer. A false-positive bound computed only over the media the detector consented to read
    is a claim about that media, so this reports what fraction of each acquisition domain it
    consented to read, and gates on the two thresholds the protocol fixed in advance.
    """
    gate = protocol["abstention_gate"]
    core_map: dict[str, str] = gate["core_strata"]
    minimum_n = gate["minimum_independent_lineages_per_core_stratum"]
    maximum_abstention = gate["maximum_abstention_in_a_core_stratum"]

    genuine = _base(_independent(records), LABEL_REAL)
    by_stratum: dict[str, list[dict]] = {}
    for record in genuine:
        by_stratum.setdefault(record["stratum"], []).append(record)

    strata = {}
    for stratum in sorted(by_stratum):
        clips = by_stratum[stratum]
        total = len({r["source_lineage_id"] for r in clips})
        readable = len({r["source_lineage_id"] for r in _readable(clips)})
        abstained = total - readable
        strata[stratum] = {
            "core_stratum": core_map.get(stratum),
            "lineages_total": total,
            "lineages_readable": readable,
            "lineages_abstained": abstained,
            "abstention_rate": (abstained / total) if total else None,
        }

    core: dict[str, dict] = {}
    for name in sorted(set(core_map.values())):
        members = [s for s, mapped in core_map.items() if mapped == name]
        present = [strata[s] for s in members if s in strata]
        total = sum(entry["lineages_total"] for entry in present)
        readable = sum(entry["lineages_readable"] for entry in present)
        abstained = total - readable
        rate = (abstained / total) if total else None
        core[name] = {
            "strata": sorted(members),
            "lineages_total": total,
            "lineages_readable": readable,
            "lineages_abstained": abstained,
            "abstention_rate": rate,
            "meets_minimum_n": total >= minimum_n,
            "abstention_within_limit": (
                None if total < minimum_n else (rate is not None and rate <= maximum_abstention)
            ),
        }

    short = [name for name, entry in core.items() if not entry["meets_minimum_n"]]
    over = [name for name, entry in core.items() if entry["abstention_within_limit"] is False]

    return {
        "minimum_independent_lineages_per_core_stratum": minimum_n,
        "maximum_abstention_in_a_core_stratum": maximum_abstention,
        "core_strata_declared": sorted(set(core_map.values())),
        "core": core,
        "by_stratum": strata,
        "core_strata_below_minimum_n": short,
        "core_strata_over_abstention_limit": over,
        "passed": not short and not over,
        "note": (
            "Abstentions are excluded from both the numerator and the denominator of every rate "
            "in this study. They are counted here instead, because the size of the population "
            "the detector declined to read is the scope limit on every claim made about the "
            "population it did read."
        ),
    }


# --- The decision gate ---------------------------------------------------------------------


def evaluate_gate(
    protocol: dict,
    genuine: dict,
    primary: dict,
    regression: list[dict],
    independence: dict,
    coverage: dict,
) -> dict:
    """A1 to A6, then the reject conditions, mechanically, from the predeclared protocol."""
    power = protocol["statistical_power"]
    required_n = power["required_readable_genuine_n"]
    fpr_target = power["target"]["threshold"]
    usefulness = protocol["decision_gate"]["A"]["usefulness_constraint"]
    reject = protocol["decision_gate"]["C"]["reject_conditions"]

    readable_n = genuine["lineages_readable"]
    upper = genuine["false_positives"]["upper_95_one_sided"]
    observed_k = genuine["false_positives"]["k"]

    detected = primary["detected"]
    tpr_lower = detected.get("lower_95_one_sided")
    tpr_upper = detected.get("upper_95_one_sided")
    tpr_observed = detected.get("observed")

    observed_regression = [
        entry
        for entry in regression
        if entry["as_acquired"] is not None and entry["as_acquired"]["status"] == "ok"
    ]
    flagged_regression = [
        entry for entry in observed_regression if entry["as_acquired"]["flagged"]
    ]

    criteria = {
        "A1": {
            "criterion": (
                "final-validation one-sided 95% Clopper-Pearson upper bound on genuine FPR at "
                f"the frozen threshold <= {fpr_target}"
            ),
            "measured": upper,
            "passed": upper is not None and upper <= fpr_target,
        },
        "A2": {
            "criterion": (
                f"readable independent genuine lineages in final validation >= {required_n}"
            ),
            "measured": readable_n,
            "passed": readable_n >= required_n,
        },
        "A3": {
            "criterion": "all gating independence / leakage checks pass with zero findings",
            "measured": independence.get("gating_checks_passed"),
            "passed": bool(independence.get("gating_checks_passed")),
        },
        "A4": {
            "criterion": (
                "no known benign regression lineage produces a candidate-positive result when "
                "readable; an abstention is a non-observation and cannot satisfy this"
            ),
            "measured": {
                "lineages": len(regression),
                "readable": len(observed_regression),
                "flagged": len(flagged_regression),
            },
            "passed": bool(observed_regression) and not flagged_regression,
        },
        "A5": {
            "criterion": (
                "intended face-manipulation usefulness: one-sided 95% TPR lower bound > "
                f"{usefulness['tpr_lower_bound_above']} AND one-sided 95% TPR upper bound >= "
                f"{usefulness['tpr_upper_bound_at_least']}"
            ),
            "measured": {
                "observed": tpr_observed,
                "lower_95_one_sided": tpr_lower,
                "upper_95_one_sided": tpr_upper,
            },
            "passed": (
                tpr_lower is not None
                and tpr_upper is not None
                and tpr_lower > usefulness["tpr_lower_bound_above"]
                and tpr_upper >= usefulness["tpr_upper_bound_at_least"]
            ),
        },
        "A6": {
            "criterion": (
                "every predeclared core genuine acquisition stratum carries at least "
                f"{coverage['minimum_independent_lineages_per_core_stratum']} independent "
                "lineages and none exceeds "
                f"{coverage['maximum_abstention_in_a_core_stratum']:.0%} abstention"
            ),
            "measured": {
                "below_minimum_n": coverage["core_strata_below_minimum_n"],
                "over_abstention_limit": coverage["core_strata_over_abstention_limit"],
                "core": {
                    name: {
                        "n": entry["lineages_total"],
                        "readable": entry["lineages_readable"],
                        "abstention_rate": entry["abstention_rate"],
                    }
                    for name, entry in coverage["core"].items()
                },
            },
            "passed": coverage["passed"],
        },
    }

    failed = [name for name, entry in criteria.items() if not entry["passed"]]

    reject_conditions = {
        "genuine_fpr_upper_bound_above_limit": (
            upper is not None and upper > reject["genuine_fpr_upper_bound_above"]
        ),
        "primary_detection_indistinguishable_from_zero": (
            tpr_upper is not None and tpr_upper < reject["primary_tpr_upper_bound_below"]
        ),
        "independence_defect": not bool(independence.get("gating_checks_passed")),
        "no_safe_calibration_threshold_existed": False,
    }
    reject_met = [name for name, hit in reject_conditions.items() if hit]

    if not failed:
        recommendation, label = "A", "ELIGIBILITY STUDY PASSED"
    elif reject_met:
        recommendation, label = "C", "REJECT EFFORT ELIGIBILITY"
    else:
        recommendation, label = "B", "CONTINUE SHADOW"

    return {
        "criteria": criteria,
        "criteria_failed": failed,
        "reject_conditions": reject_conditions,
        "reject_conditions_met": reject_met,
        "recommendation": recommendation,
        "recommendation_label": label,
        "power_gate": {
            "required_readable_genuine_n": required_n,
            "readable_genuine_n": readable_n,
            "sufficient": readable_n >= required_n,
            "observed_false_positives": observed_k,
            "recommendation_a_reachable": readable_n >= required_n,
        },
        "evaluated_from": (
            "the criteria frozen in the protocol before the validation split was scored; this "
            "function reads them and does not restate them"
        ),
    }


# --- Subcommands ---------------------------------------------------------------------------


def freeze_threshold(argv: argparse.Namespace) -> int:
    items = read_corpus(argv.corpus)
    protocol = json.loads(argv.protocol.read_text(encoding="utf-8"))
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

    independent = _independent(records)
    genuine = _readable(_base(independent, LABEL_REAL))
    primary = _readable(_base(independent, PRIMARY_LABEL))
    inducing = _readable(_base(independent))

    if not genuine or not primary:
        print(
            "the calibration split carries no readable genuine or no readable intended-target "
            "lineages; no operating point can be derived from it",
            file=sys.stderr,
        )
        return 1

    candidates = candidate_thresholds([record["score"] for record in inducing])
    rows = evaluate_candidates(genuine, primary, candidates)
    fpr_target = protocol["statistical_power"]["target"]["threshold"]
    selection = select_operating_point(rows, fpr_target)

    required_n = protocol["statistical_power"]["required_readable_genuine_n"]
    readable_genuine_lineages = len({r["source_lineage_id"] for r in genuine})

    payload = {
        "schema_version": "r7-t9-frozen-threshold-1",
        "task": "R7-T9",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_before_validation_split_was_scored": True,
        "selection_rule": (
            "on the calibration split only: enumerate every decision-equivalent operating "
            "threshold induced by the observed readable base-clip scores under `score > "
            "threshold`; discard every candidate whose exact one-sided 95% Clopper-Pearson "
            "upper bound on the genuine false-positive rate exceeds the predeclared target; "
            "among the survivors take the highest observed intended-target TPR, breaking ties "
            "on lower observed genuine FPR and then on higher threshold"
        ),
        "rule_implementation": (
            "eval.analyze_effort_r7t9.candidate_thresholds, .evaluate_candidates and "
            ".select_operating_point"
        ),
        "comparison_semantic": "score > threshold, from eval.analyze_effort._flagged",
        "bound_method": "exact one-sided 95% Clopper-Pearson, eval.stats, unchanged",
        "derived_on": {
            "split": SPLIT_CALIBRATION,
            "corpus": str(argv.corpus),
            "calibration_run": str(argv.calibration_run),
            "calibration_run_sha256": _digest(argv.calibration_run),
            "protocol": str(argv.protocol),
            "protocol_sha256": _digest(argv.protocol),
            "genuine_base_lineages_readable": readable_genuine_lineages,
            "genuine_base_lineages_total": len(
                {r["source_lineage_id"] for r in _base(independent, LABEL_REAL)}
            ),
            "primary_base_lineages_readable": len(
                {r["source_lineage_id"] for r in primary}
            ),
            "scores_inducing_the_candidate_set": len(inducing),
            "distinct_scores": len({r["score"] for r in inducing}),
        },
        "calibration_power": {
            "required_readable_genuine_n": required_n,
            "readable_genuine_n": readable_genuine_lineages,
            "sufficient": readable_genuine_lineages >= required_n,
        },
        "candidate_set": {
            "size": len(candidates),
            "construction": (
                "one representative of each decision-equivalent class: nextafter(min, -inf) "
                "for the class below the smallest observed score, and each distinct observed "
                "score for the class it opens. No numeric grid and no decimal quantisation."
            ),
            "min": candidates[0] if candidates else None,
            "max": candidates[-1] if candidates else None,
        },
        "selection": selection,
        "threshold": (selection["selected"] or {}).get("threshold"),
        "genuine_calibration_score_distribution": _distribution(
            [r["score"] for r in genuine]
        ),
        "primary_calibration_score_distribution": _distribution(
            [r["score"] for r in primary]
        ),
        "not_anchored": (
            "Derived from this calibration split alone by the predeclared rule. The historical "
            f"candidates ({R7T7_CANDIDATE_THRESHOLD} from R7-T7, {R7T8_FROZEN_THRESHOLD} from "
            "R7-T8) were not used as a starting point, were not compared to this value before "
            "it was fixed, and no acceptance criterion is stated in terms of either."
        ),
        "candidate_table": rows,
    }

    if selection["selected"] is None:
        payload["outcome"] = (
            "NO SAFE OPERATING POINT. The predeclared genuine false-positive constraint is not "
            "satisfiable on this calibration split. Recommendation A is impossible and the "
            "final validation split is not opened."
        )
        argv.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {argv.output}")
        print(f"candidates evaluated: {len(rows)}; surviving the FPR constraint: 0")
        print("NO SAFE THRESHOLD EXISTS ON THIS CALIBRATION SPLIT")
        return 2

    argv.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    chosen = selection["selected"]
    print(f"wrote {argv.output}")
    print(f"readable genuine calibration lineages: {readable_genuine_lineages}")
    print(f"candidates evaluated: {len(rows)}")
    print(f"candidates surviving FPR upper bound <= {fpr_target}: {selection['candidates_surviving']}")
    print(
        f"selected: TPR {chosen['primary_detected']}/{chosen['primary_readable_lineages']} "
        f"= {chosen['primary_tpr_observed']:.4%}, genuine FP "
        f"{chosen['genuine_false_positives']}/{chosen['genuine_readable_lineages']}, "
        f"FPR upper bound {chosen['genuine_fpr_upper_95']:.4%}"
    )
    print(f"tie-break decided by: {selection['tie_break']['decided_by']}")
    print(f"FROZEN THRESHOLD: {chosen['threshold']}")
    return 0


def operating_point_context(
    validation: list[dict], frozen: dict, r7t8_analysis: dict | None
) -> dict:
    """Where the frozen point sits relative to the previous tasks' points. Descriptive only."""
    threshold = frozen["threshold"]
    genuine = _readable(_base(_independent(validation), LABEL_REAL))

    def crossings(value: float) -> dict:
        rate = _lineage_rate(genuine, lambda r: _flagged(r, value))
        return {"threshold": value, **rate.as_dict(), "summary": rate.describe()}

    return {
        "handling": (
            "descriptive only. No criterion in this task is stated over the distance between "
            "this operating point and any earlier one; the quantity that carries the safety "
            "claim is the false-positive rate at the frozen threshold on independent data."
        ),
        "r7t7_candidate_threshold": R7T7_CANDIDATE_THRESHOLD,
        "r7t8_frozen_threshold": R7T8_FROZEN_THRESHOLD,
        "r7t9_frozen_threshold": threshold,
        "signed_drift_vs_r7t8": threshold - R7T8_FROZEN_THRESHOLD,
        "signed_drift_vs_r7t7": threshold - R7T7_CANDIDATE_THRESHOLD,
        "r7t9_validation_genuine_at": {
            "r7t9_frozen": crossings(threshold),
            "r7t8_frozen": crossings(R7T8_FROZEN_THRESHOLD),
            "r7t7_candidate": crossings(R7T7_CANDIDATE_THRESHOLD),
        },
        "r7t8_reference": (
            {
                "validation_genuine_false_positives": r7t8_analysis["validation"][
                    "genuine_primary_independent"
                ]["false_positive_summary"],
                "validation_primary_detected": r7t8_analysis["validation"][
                    "primary_task_face_manipulation"
                ]["detected_summary"],
                "recommendation": r7t8_analysis["decision_gate"]["recommendation"],
            }
            if r7t8_analysis
            else None
        ),
    }


def run_analysis(argv: argparse.Namespace) -> int:
    items = read_corpus(argv.corpus)
    calibration_run = json.loads(argv.calibration_run.read_text(encoding="utf-8"))
    validation_run = json.loads(argv.validation_run.read_text(encoding="utf-8"))
    frozen = json.loads(argv.frozen_threshold.read_text(encoding="utf-8"))
    independence = json.loads(argv.independence.read_text(encoding="utf-8"))
    freeze = json.loads(argv.freeze.read_text(encoding="utf-8"))
    protocol = json.loads(argv.protocol.read_text(encoding="utf-8"))
    r7t8_analysis = (
        json.loads(argv.r7t8_analysis.read_text(encoding="utf-8"))
        if argv.r7t8_analysis
        else None
    )

    threshold = frozen["threshold"]
    if threshold is None:
        print(
            "the frozen artifact carries no threshold: the calibration split produced no "
            "candidate satisfying the safety constraint, and the validation split must not be "
            "scored",
            file=sys.stderr,
        )
        return 1

    calibration = join_run(items, calibration_run)
    validation = join_run(items, validation_run)

    split_integrity = {
        "calibration_run_contains_only": sorted({r["split"] for r in calibration}),
        "validation_run_contains_only": sorted({r["split"] for r in validation}),
        "frozen_threshold_derived_on": frozen["derived_on"]["split"],
        "calibration_run_digest_matches_freeze": (
            _digest(argv.calibration_run) == frozen["derived_on"]["calibration_run_sha256"]
        ),
        "protocol_digest_matches_freeze": (
            _digest(argv.protocol) == frozen["derived_on"]["protocol_sha256"]
        ),
        "frozen_before_validation_was_scored": frozen[
            "frozen_before_validation_split_was_scored"
        ],
    }
    split_integrity["clean"] = (
        split_integrity["calibration_run_contains_only"] == [SPLIT_CALIBRATION]
        and split_integrity["validation_run_contains_only"] == [SPLIT_EVALUATION]
        and split_integrity["calibration_run_digest_matches_freeze"]
        and split_integrity["protocol_digest_matches_freeze"]
    )

    independent_validation = _independent(validation)
    genuine = genuine_primary(independent_validation, threshold)
    primary = manipulated_report(independent_validation, PRIMARY_LABEL, threshold)
    ood = manipulated_report(independent_validation, OOD_LABEL, threshold)
    regression = regression_report(validation, threshold)
    coverage = core_coverage(validation, protocol)

    genuine_scores = [r["score"] for r in _readable(_base(independent_validation, LABEL_REAL))]
    primary_scores = [r["score"] for r in _readable(_base(independent_validation, PRIMARY_LABEL))]
    ood_scores = [r["score"] for r in _readable(_base(independent_validation, OOD_LABEL))]

    gate = evaluate_gate(protocol, genuine, primary, regression, independence, coverage)

    analysis = {
        "schema_version": "r7-t9-effort-analysis-1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "R7-T9",
        "protocol": {
            "path": str(argv.protocol),
            "sha256": _digest(argv.protocol),
            "selection_rule": protocol["threshold_derivation"]["rule"],
        },
        "candidate": {
            "detector": "effort",
            "variant": freeze["candidate"]["checkpoint"]["variant"],
            "checkpoint_sha256": freeze["candidate"]["checkpoint"]["sha256"],
            "upstream_revision": freeze["candidate"]["upstream_revision"],
            "preprocessing_and_inference": "unchanged from R7-T7 and R7-T8",
        },
        "frozen_threshold": {
            "value": threshold,
            "sha256": _digest(argv.frozen_threshold),
            "frozen_at": frozen["frozen_at"],
            "derived_on": frozen["derived_on"],
            "selection": frozen["selection"],
            "candidate_set_size": frozen["candidate_set"]["size"],
            "calibration_power": frozen["calibration_power"],
        },
        "independence": {
            "gating_checks_passed": independence.get("gating_checks_passed"),
            "checks": independence.get("gate_summary"),
        },
        "split_integrity": split_integrity,
        "validation": {
            "genuine_primary_independent": genuine,
            "core_coverage": coverage,
            "primary_task_face_manipulation": primary,
            "cross_family_ood_generated": ood,
            "auroc": {
                "genuine_vs_face_manipulation": auroc(primary_scores, genuine_scores),
                "genuine_vs_generated_ood": auroc(ood_scores, genuine_scores),
                "note": (
                    "Threshold-free separation, reported for the primary task and the "
                    "out-of-distribution task separately and never pooled."
                ),
            },
            "derivative_robustness": derivative_robustness(independent_validation, threshold),
            "abstention_by_frames_with_face": abstention_structure(
                independent_validation, threshold
            ),
            "failures": failure_report(validation),
            "threshold_sensitivity": threshold_sensitivity(
                independent_validation,
                sorted({0.5, 0.9, R7T7_CANDIDATE_THRESHOLD, threshold, R7T8_FROZEN_THRESHOLD, 0.99}),
            ),
        },
        "regression": {
            "note": (
                "The benign real-world lineages, the one declared re-use in this corpus. "
                "Excluded from every headline denominator above and from threshold selection; "
                "reported here at the frozen operating point. An abstention is recorded as a "
                "non-observation, never as a pass."
            ),
            "lineages": regression,
        },
        "operating_point_context": operating_point_context(validation, frozen, r7t8_analysis),
        "decision_gate": gate,
        "calibration_reference": {
            "note": (
                "The calibration split, for reference only. No criterion is stated over it. The "
                "threshold was derived here and frozen before the validation split was opened."
            ),
            "genuine_primary_independent": genuine_primary(
                _independent(calibration), threshold
            ),
            "core_coverage": core_coverage(calibration, protocol),
            "primary_task_face_manipulation": manipulated_report(
                _independent(calibration), PRIMARY_LABEL, threshold
            ),
            "cross_family_ood_generated": manipulated_report(
                _independent(calibration), OOD_LABEL, threshold
            ),
            "failures": failure_report(calibration),
        },
        "compute": {
            "calibration": compute_cost(calibration_run),
            "validation": compute_cost(validation_run),
        },
    }

    argv.output_dir.mkdir(parents=True, exist_ok=True)
    destination = argv.output_dir / "effort_r7t9_analysis.json"
    destination.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {destination}")
    print(f"frozen threshold: {threshold}")
    print(f"split integrity clean: {split_integrity['clean']}")
    print(f"independence gating passed: {independence.get('gating_checks_passed')}")
    print(f"genuine (independent, lineage-level): {genuine['false_positive_summary']}")
    print(
        f"readable genuine n: {genuine['lineages_readable']} "
        f"(required {protocol['statistical_power']['required_readable_genuine_n']})"
    )
    for name, entry in coverage["core"].items():
        print(
            f"  core stratum {name}: n={entry['lineages_total']} "
            f"readable={entry['lineages_readable']} "
            f"abstention={0.0 if entry['abstention_rate'] is None else entry['abstention_rate']:.2%}"
        )
    print(f"face manipulation detected: {primary['detected_summary']}")
    print(
        "  TPR bounds: lower "
        f"{primary['detected']['lower_95_one_sided']}, upper "
        f"{primary['detected']['upper_95_one_sided']}"
    )
    print(f"generated OOD detected: {ood['detected_summary']}")
    print(f"criteria failed: {gate['criteria_failed'] or 'none'}")
    print(f"RECOMMENDATION: {gate['recommendation']} - {gate['recommendation_label']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_effort_r7t9",
        description="Derive, freeze and spend the R7-T9 FPR-constrained operating point.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="derive the candidate operating point and freeze it")
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
    analyze.add_argument("--r7t8-analysis", type=Path)
    analyze.add_argument("--output-dir", required=True, type=Path)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "freeze":
        return freeze_threshold(args)
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
