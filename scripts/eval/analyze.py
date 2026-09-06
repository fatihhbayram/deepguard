"""Turn three detector runs and one rule replay into the measurements R7-T5 has to report.

    PYTHONPATH=scripts python3 scripts/eval/analyze.py \
        --corpus ../deepguard-corpus/r7t5/corpus.json \
        --lip-run ../deepguard-corpus/runs/r7t5-lip/results.json \
        --face-run ../deepguard-corpus/runs/r7t5-face/results.json \
        --svd-run ../deepguard-corpus/runs/r7t5-svd/results.json \
        --replay ../deepguard-corpus/r7t5/replay.json \
        --output-dir ../deepguard-corpus/r7t5/analysis

Three rules govern everything below, and they are the three the task is most easily got wrong on.

**Rates are counted over lineages, never over files.** `_lineage_rate` reduces any set of clips
to the distinct recordings behind them and asks how many of those recordings produced the
outcome. A corpus that is one third constructed transcodes would otherwise report a
false-positive rate in which one bad lineage counts nine times, and the acceptance target — which
is stated over independent genuine lineages — would be met or missed by the arithmetic of the
derivative budget.

**Every detector is measured against its own target, and nothing is pooled.** A single
"manipulated vs genuine" accuracy over this corpus would average LipForensics' behaviour on face
swaps, which it was trained for, with its behaviour on text-to-video, which it was not, and
report the mean as though it described either. `DETECTORS` names each detector's primary family
set explicitly; everything else it meets is reported separately as cross-family behaviour, and
neither figure is ever added to the other.

**An abstention is not a negative.** A clip the detector could not read is removed from both the
numerator and the denominator of every rate, and counted in its own. Folding abstentions into
the negatives would let a detector improve its false-positive rate by failing to run.

Nothing here is a threshold decision. Candidate operating points are derived on the calibration
split and reported as candidates; the production constants are read from the risk engine and used
exactly as found.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import stats
from eval.corpus import (
    LABEL_REAL,
    SPLIT_CALIBRATION,
    SPLIT_EVALUATION,
    CorpusItem,
    read_corpus,
)
from eval.replay import load_risk_engine


@dataclass(frozen=True)
class Detector:
    """One detector, and the question it is entitled to be scored on.

    `primary_families` is the manipulation this detector was built and calibrated to find.
    `primary_label` says the same thing in words, for the report. Everything manipulated that is
    not in `primary_families` is cross-family: the detector meeting evidence outside its
    coverage claim, where a miss is a statement about scope rather than about sensitivity.
    """

    key: str
    name: str
    primary_label: str
    primary_families: tuple[str, ...]
    threshold_attribute: str


DETECTORS = (
    Detector(
        "svd",
        "NVIDIA synthetic-video",
        "generated / synthetic video",
        (
            "talkinghead_echomimic",
            "talkinghead_sonic",
            "talkinghead_memo",
            "reenactment_liveportrait",
            "t2v_veo3",
            "t2v_sora2",
            "t2v_kling",
        ),
        "SVD_T_HIGH",
    ),
    Detector(
        "face",
        "EfficientNet-B7",
        "face manipulation / face swap",
        ("faceswap_inswapper", "faceswap_roop", "ffpp_deepfakes", "ffpp_faceswap"),
        "FACE_T_HIGH",
    ),
    Detector(
        "lip",
        "LipForensics",
        "facial-forgery mouth dynamics",
        ("faceswap_inswapper", "faceswap_roop", "ffpp_deepfakes", "ffpp_faceswap"),
        "LIP_T_HIGH",
    ),
)

# How many extreme clips to name when illustrating a failure. Enough to see a pattern, few
# enough that a reader reads them.
EXAMPLES = 8


def _lineage_rate(records: list[dict], predicate) -> stats.Rate:
    """The rate at which distinct recordings produce an outcome, with exact bounds.

    A lineage counts once and counts as positive if any of its clips in this set satisfies the
    predicate. For base-clip sets that is one clip per lineage and the reduction is a no-op;
    for a derivative stratum it is what stops one recording's eight degradations from being
    eight observations of one fact.
    """
    by_lineage: dict[str, bool] = {}
    for record in records:
        lineage = record["source_lineage_id"]
        by_lineage[lineage] = by_lineage.get(lineage, False) or bool(predicate(record))
    return stats.rate(sum(by_lineage.values()), len(by_lineage))


def _distribution(values: list[float]) -> dict:
    """Where a set of scores actually sits, without a histogram nobody can read in Markdown."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": statistics.median(ordered),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _quantile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank quantile: an observed value, never an interpolation between two."""
    if not ordered:
        raise ValueError("no values")
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def join_run(items: list[CorpusItem], run: dict | None, counts_key: str) -> list[dict]:
    """One record per clip: what the corpus says about it and what the detector said.

    `status` is the detector's, reduced to three states a rate can be computed over: `ok` for a
    reading, `abstained` for a clip it could not read, and `absent` for a clip it was never
    asked about. The third is not a detector property and is reported separately so a partial
    run cannot look like an unreliable model.
    """
    if run is None:
        by_id: dict[str, dict] = {}
        provenance: dict = {}
    else:
        by_id = {clip["clip_id"]: clip for clip in run["clips"]}
        provenance = run["run"].get("model_provenance") or {}
    counts = provenance.get(counts_key) or {}

    records = []
    for item in items:
        clip = by_id.get(item.clip_id)
        if clip is None:
            status, score = "absent", None
        elif clip.get("status") == "ok":
            status, score = "ok", clip.get("score")
        else:
            status, score = "abstained", None
        records.append(
            {
                "clip_id": item.clip_id,
                "split": item.split,
                "label": item.label,
                "family": item.family,
                "stratum": item.stratum_primary,
                "derivation": item.derivation,
                "is_base": item.is_base,
                "private": item.private,
                "source_lineage_id": item.source_lineage_id,
                "status": status,
                "score": score,
                "units_scored": counts.get(item.clip_id),
                "error": (clip or {}).get("error"),
            }
        )
    return records


def detector_report(
    detector: Detector, records: list[dict], threshold: float, split: str
) -> dict:
    """Everything this detector's section of the report needs, for one split."""
    scoped = [r for r in records if r["split"] == split]
    base = [r for r in scoped if r["is_base"]]
    genuine_base = [r for r in base if r["label"] == LABEL_REAL]
    manipulated_base = [r for r in base if r["label"] != LABEL_REAL]

    def flagged(record: dict) -> bool:
        return record["status"] == "ok" and record["score"] is not None and (
            record["score"] >= threshold
        )

    def readable(records_in: list[dict]) -> list[dict]:
        return [r for r in records_in if r["status"] == "ok"]

    primary = [r for r in manipulated_base if r["family"] in detector.primary_families]
    cross_family = [
        r for r in manipulated_base if r["family"] not in detector.primary_families
    ]

    report = {
        "detector": detector.name,
        "key": detector.key,
        "split": split,
        "threshold": threshold,
        "threshold_source": f"risk_engine.{detector.threshold_attribute}",
        "score_semantics": (
            "a model output on this detector's own scale; the provider defines no calibrated "
            "confidence semantics for it"
        ),
        "genuine": {
            "false_positive_rate_by_lineage": _lineage_rate(
                readable(genuine_base), flagged
            ).as_dict(),
            "abstention_rate_by_lineage": _lineage_rate(
                genuine_base, lambda r: r["status"] == "abstained"
            ).as_dict(),
            "unavailable_rate_by_lineage": _lineage_rate(
                genuine_base, lambda r: r["status"] == "absent"
            ).as_dict(),
            "score_distribution": _distribution(
                [r["score"] for r in readable(genuine_base)]
            ),
            "highest_scores": _top(readable(genuine_base), EXAMPLES),
            "false_positives": _top(
                [r for r in readable(genuine_base) if flagged(r)], EXAMPLES
            ),
        },
        "primary_task": {
            "families": list(detector.primary_families),
            "target": detector.primary_label,
            "detection_rate_by_lineage": _lineage_rate(readable(primary), flagged).as_dict(),
            "abstention_rate_by_lineage": _lineage_rate(
                primary, lambda r: r["status"] == "abstained"
            ).as_dict(),
            "score_distribution": _distribution([r["score"] for r in readable(primary)]),
            "false_negatives": _bottom(
                [r for r in readable(primary) if not flagged(r)], EXAMPLES
            ),
        },
        "cross_family": {
            "detection_rate_by_lineage": _lineage_rate(
                readable(cross_family), flagged
            ).as_dict(),
            "abstention_rate_by_lineage": _lineage_rate(
                cross_family, lambda r: r["status"] == "abstained"
            ).as_dict(),
            "score_distribution": _distribution([r["score"] for r in readable(cross_family)]),
            "note": (
                "Media outside this detector's coverage claim. A miss here is a statement "
                "about scope, not about sensitivity, and is never added to the primary-task "
                "rate."
            ),
        },
        "by_family": {},
        "by_stratum": {},
    }

    for family in sorted({r["family"] for r in scoped}):
        subset = [r for r in scoped if r["family"] == family and r["is_base"]]
        if not subset:
            continue
        report["by_family"][family] = {
            "label": subset[0]["label"],
            "in_primary_target": family in detector.primary_families,
            "flag_rate_by_lineage": _lineage_rate(readable(subset), flagged).as_dict(),
            "abstention_rate_by_lineage": _lineage_rate(
                subset, lambda r: r["status"] == "abstained"
            ).as_dict(),
            "score_distribution": _distribution([r["score"] for r in readable(subset)]),
        }

    for stratum in sorted({r["stratum"] for r in scoped}):
        subset = [r for r in scoped if r["stratum"] == stratum]
        report["by_stratum"][stratum] = {
            "label": Counter(r["label"] for r in subset).most_common(1)[0][0],
            "constructed": any(r["derivation"] != "none" for r in subset),
            "flag_rate_by_lineage": _lineage_rate(readable(subset), flagged).as_dict(),
            "abstention_rate_by_lineage": _lineage_rate(
                subset, lambda r: r["status"] == "abstained"
            ).as_dict(),
            "score_distribution": _distribution([r["score"] for r in readable(subset)]),
        }

    return report


def _top(records: list[dict], count: int) -> list[dict]:
    ordered = sorted(
        (r for r in records if r["score"] is not None),
        key=lambda r: r["score"],
        reverse=True,
    )
    return [_example(r) for r in ordered[:count]]


def _bottom(records: list[dict], count: int) -> list[dict]:
    ordered = sorted(
        (r for r in records if r["score"] is not None), key=lambda r: r["score"]
    )
    return [_example(r) for r in ordered[:count]]


def _example(record: dict) -> dict:
    """One clip, named the way it may safely be named.

    A private clip is identified by its lineage id alone. Its `clip_id` is derived from that
    lineage id and carries nothing else, but the rule here is that nothing outside the lineage
    id, the stratum and the measured numbers travels — so the clip id is dropped rather than
    reasoned about.
    """
    return {
        "clip_id": None if record["private"] else record["clip_id"],
        "source_lineage_id": record["source_lineage_id"],
        "private": record["private"],
        "family": record["family"],
        "stratum": record["stratum"],
        "derivation": record["derivation"],
        "score": record["score"],
        "units_scored": record["units_scored"],
    }


def threshold_candidates(records: list[dict], production_threshold: float) -> dict:
    """What operating points the calibration split alone would support, as candidates only.

    Derived on the calibration split and reported for the Architect to decide on. Two rules,
    the same two R4-T1 and R5-T3 used, so a candidate is comparable with the constant it would
    replace:

    - the lowest threshold at which no genuine calibration lineage is flagged, placed at the
      midpoint between the highest genuine score and the lowest score above it;
    - the same, with the midpoint replaced by the highest genuine score itself, which is the
      tightest threshold that still clears the observed genuine media and shows how much of the
      first figure is margin rather than measurement.

    Also reported: what each candidate would cost in primary-task detection on the calibration
    split. A threshold that flags nothing is always safe and always useless, and the trade has
    to be visible next to the number.
    """
    calibration = [
        r
        for r in records
        if r["split"] == SPLIT_CALIBRATION and r["is_base"] and r["status"] == "ok"
    ]
    genuine = sorted(
        r["score"] for r in calibration if r["label"] == LABEL_REAL and r["score"] is not None
    )
    manipulated = sorted(
        r["score"] for r in calibration if r["label"] != LABEL_REAL and r["score"] is not None
    )
    if not genuine:
        return {"available": False, "reason": "no readable genuine calibration clips"}

    highest_genuine = genuine[-1]
    above = [score for score in manipulated if score > highest_genuine]
    midpoint = (highest_genuine + min(above)) / 2 if above else None

    def coverage(threshold: float | None) -> dict | None:
        if threshold is None:
            return None
        return {
            "threshold": threshold,
            "genuine_flagged": sum(1 for score in genuine if score >= threshold),
            "genuine_n": len(genuine),
            "manipulated_flagged": sum(1 for score in manipulated if score >= threshold),
            "manipulated_n": len(manipulated),
        }

    return {
        "available": True,
        "split": SPLIT_CALIBRATION,
        "genuine_n": len(genuine),
        "manipulated_n": len(manipulated),
        "highest_genuine_score": highest_genuine,
        "lowest_manipulated_above_genuine": min(above) if above else None,
        "candidate_midpoint": coverage(midpoint),
        "candidate_at_highest_genuine": coverage(highest_genuine),
        "production_threshold": coverage(production_threshold),
        "note": (
            "Candidates only. Nothing here is adopted, and adopting one is a separate "
            "Architect-approved task."
        ),
    }


def system_report(replay: dict) -> dict:
    """The current v3 rules' observed behaviour, split by what a reader has to decide on."""
    evaluation = [d for d in replay["decisions"] if d["split"] == SPLIT_EVALUATION]
    genuine = [d for d in evaluation if d["label"] == LABEL_REAL]
    genuine_base = [d for d in genuine if d["derivation"] == "none"]

    def high(record: dict) -> bool:
        return record["risk_level"] == "HIGH"

    by_stratum = {}
    for stratum in sorted({d["stratum"] for d in genuine}):
        subset = [d for d in genuine if d["stratum"] == stratum]
        by_stratum[stratum] = {
            "constructed": any(d["derivation"] != "none" for d in subset),
            "high_rate_by_lineage": _lineage_rate(subset, high).as_dict(),
            "rules_fired": dict(sorted(Counter(d["rule_id"] for d in subset).items())),
        }

    genuine_highs = [d for d in genuine if high(d)]
    return {
        "rules": replay["rules"],
        "eligibility": replay["eligibility"],
        "evaluation_clips": len(evaluation),
        "genuine": {
            "base_lineage_high_rate": _lineage_rate(genuine_base, high).as_dict(),
            "all_clip_high_rate_by_lineage": _lineage_rate(genuine, high).as_dict(),
            "rules_fired": dict(sorted(Counter(d["rule_id"] for d in genuine).items())),
            "levels": dict(sorted(Counter(d["risk_level"] for d in genuine).items())),
            "by_stratum": by_stratum,
            "high_cases": [
                {
                    "clip_id": None if d["private"] else d["clip_id"],
                    "source_lineage_id": d["source_lineage_id"],
                    "private": d["private"],
                    "stratum": d["stratum"],
                    "derivation": d["derivation"],
                    "rule_id": d["rule_id"],
                    "responsible_detectors": d["responsible_detectors"],
                    "scores": d["scores"],
                    "counts": d["counts"],
                }
                for d in genuine_highs
            ],
            "responsible_detector_counts": dict(
                sorted(
                    Counter(
                        detector
                        for d in genuine_highs
                        for detector in d["responsible_detectors"]
                    ).items()
                )
            ),
        },
        "manipulated": {
            "high_rate_by_lineage": _lineage_rate(
                [d for d in evaluation if d["label"] != LABEL_REAL], high
            ).as_dict(),
            "rules_fired": dict(
                sorted(
                    Counter(
                        d["rule_id"] for d in evaluation if d["label"] != LABEL_REAL
                    ).items()
                )
            ),
        },
    }


def regression_report(replay: dict, joined: dict[str, list[dict]]) -> list[dict]:
    """The private lineages, by lineage id, measured outputs and outcome, and nothing else."""
    cases = []
    for decision in replay["decisions"]:
        if not decision["private"]:
            continue
        cases.append(
            {
                "source_lineage_id": decision["source_lineage_id"],
                "derivation": decision["derivation"],
                "stratum": decision["stratum"],
                "risk_level": decision["risk_level"],
                "rule_id": decision["rule_id"],
                "responsible_detectors": decision["responsible_detectors"],
                "scores": decision["scores"],
                "counts": decision["counts"],
                "statuses": decision["statuses"],
            }
        )
    return sorted(cases, key=lambda case: (case["source_lineage_id"], case["derivation"]))


def quarantined_lineages(verification: dict | None) -> list[str]:
    """Evaluation genuine lineages the pair audit could not clear of sharing a recording.

    The identity checks passed: no lineage id and no byte-identical clip crosses the split
    boundary. These are the lineages the *content* screen flagged and the finer adjudication
    then failed to dismiss as a screening artefact — a shared source recording that no field in
    the manifest could have revealed, because MAVOS-DD publishes no mapping from its clips back
    to the videos they were cut from.

    They are not removed from the corpus and not moved between splits. They drive a sensitivity
    pass: every headline genuine figure is reported twice, once over all evaluation lineages and
    once with these excluded, so a reader can see whether the conclusion depends on them. It
    does not, and showing that is worth more than quietly dropping three lineages would be.
    """
    if not verification:
        return []
    return sorted(
        {
            entry["evaluation_lineage"]
            for entry in verification.get("results", [])
            if entry["evaluation_label"] == LABEL_REAL
            and entry["verdict"].startswith(("likely shared", "unresolved"))
        }
    )


def sensitivity(replay: dict, excluded: list[str]) -> dict:
    """The system-level genuine figures recomputed without the lineages under suspicion."""
    kept = [
        decision
        for decision in replay["decisions"]
        if decision["split"] == SPLIT_EVALUATION
        and decision["label"] == LABEL_REAL
        and decision["derivation"] == "none"
        and decision["source_lineage_id"] not in excluded
    ]
    return {
        "excluded_lineages": excluded,
        "excluded_count": len(excluded),
        "base_lineage_high_rate": _lineage_rate(
            kept, lambda d: d["risk_level"] == "HIGH"
        ).as_dict(),
        "note": (
            "The identity leakage checks passed; these lineages were flagged by the content "
            "screen and not dismissed by the finer adjudication. Reported so the headline "
            "figure can be seen not to depend on them."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze", description="Compute the R7-T5 robustness measurements."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--lip-run", type=Path)
    parser.add_argument("--face-run", type=Path)
    parser.add_argument("--svd-run", type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument(
        "--pair-verification",
        type=Path,
        help="verify_pairs output; its unresolved genuine lineages drive a sensitivity pass",
    )
    parser.add_argument("--acceptance-target", type=Path,
                        default=Path(__file__).resolve().parent / "acceptance_target.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _load(path: Path | None) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = load_risk_engine()
    items = read_corpus(args.corpus)
    replay = _load(args.replay)
    target = _load(args.acceptance_target)

    runs = {
        "svd": (_load(args.svd_run), "total_clips_by_clip_id"),
        "face": (_load(args.face_run), "frames_scored_by_clip_id"),
        "lip": (_load(args.lip_run), "windows_scored_by_clip_id"),
    }
    joined = {key: join_run(items, run, counts) for key, (run, counts) in runs.items()}

    detectors = {}
    for detector in DETECTORS:
        threshold = getattr(engine, detector.threshold_attribute)
        records = joined[detector.key]
        detectors[detector.key] = {
            SPLIT_EVALUATION: detector_report(
                detector, records, threshold, SPLIT_EVALUATION
            ),
            SPLIT_CALIBRATION: detector_report(
                detector, records, threshold, SPLIT_CALIBRATION
            ),
            "threshold_candidates": threshold_candidates(records, threshold),
        }

    verification = _load(args.pair_verification)
    excluded = quarantined_lineages(verification)
    system = system_report(replay)
    genuine_bound = system["genuine"]["base_lineage_high_rate"]
    declared = target["primary_target"]["threshold"] if target else None
    met = (
        genuine_bound["upper_95_one_sided"] is not None
        and declared is not None
        and genuine_bound["upper_95_one_sided"] <= declared
    )

    analysis = {
        "schema_version": "r7-t5-analysis-1",
        "analysed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "acceptance_target": target,
        "acceptance_outcome": {
            "system_level_genuine_high": genuine_bound,
            "declared_upper_bound": declared,
            "required_n_for_zero_observed": stats.required_n_for_zero_observed(declared)
            if declared
            else None,
            "target_met": met,
            "note": (
                "Met means the measured one-sided 95% upper bound is at or below the bound "
                "declared before the evaluation split was scored. It is not a claim that the "
                "true rate is zero."
            ),
        },
        "detectors": detectors,
        "system": system,
        "lineage_independence_sensitivity": sensitivity(replay, excluded),
        "regression_cases": regression_report(replay, joined),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )

    print(f"acceptance target met: {met}")
    print(f"system-level genuine HIGH: "
          f"{stats.Rate(**_rate_kwargs(genuine_bound)).describe()}")
    for key, entry in detectors.items():
        fpr = entry[SPLIT_EVALUATION]["genuine"]["false_positive_rate_by_lineage"]
        print(f"{key}: genuine FPR {stats.Rate(**_rate_kwargs(fpr)).describe()}")
    return 0


def _rate_kwargs(payload: dict) -> dict:
    return {
        "numerator": payload["k"],
        "denominator": payload["n"],
        "observed": payload["observed"],
        "upper_95": payload["upper_95_one_sided"],
        "lower_95": payload["lower_95_one_sided"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
