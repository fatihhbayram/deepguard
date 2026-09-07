"""Turn the two Effort shadow runs into the measurements R7-T7 has to report.

    PYTHONPATH=scripts python3 scripts/eval/analyze_effort.py \
        --corpus ../deepguard-corpus/r7t5/corpus.json \
        --calibration-run ../deepguard-corpus/runs/r7t7-effort-calibration/results.json \
        --evaluation-run ../deepguard-corpus/runs/r7t7-effort-evaluation/results.json \
        --freeze scripts/eval/effort_freeze.json \
        --output-dir ../deepguard-corpus/r7t5/analysis-effort

Four rules govern everything below. Three are inherited unchanged from `analyze.py`, because a
candidate measured under different rules than the incumbent cannot be compared with it; the fourth
is this task's.

**Rates are counted over lineages, never over files.** `_lineage_rate` reduces a set of clips to
the distinct recordings behind them. A third of this corpus is constructed degradations of other
clips in it, and a file-counted false-positive rate would be decided by the derivative budget
rather than by how many independent recordings were tested.

**The primary genuine figure uses base clips only.** A lineage does not become a false positive
because one of its constructed degradations crossed the threshold. Derivatives are reported in
their own section, at clip level, with the number of distinct lineages behind them stated, and
they never enter the headline denominator.

**An abstention is not a negative.** A clip the detector could not read leaves both the numerator
and the denominator and is counted in its own rate. Folding abstentions into the negatives would
let a detector improve its false-positive rate by failing to run.

**Nothing is pooled across detection targets.** Effort's face-deepfake checkpoint was trained on
FaceForensics++ face manipulations. Face swaps are its primary task; generated and audio-driven
video is out-of-distribution. A single "manipulated vs genuine" accuracy over this corpus would
average the two and describe neither, so the two are computed separately and never added.

Every threshold and aggregation this module applies is read from the freeze document rather than
chosen here, and the calibration-derived candidate is computed on calibration records only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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

# Effort's primary task in this corpus, and the families that are outside it. Declared in
# `effort_freeze.json` before scoring and restated here as the thing the code actually keys on.
PRIMARY_LABEL = "face_swap"
OOD_LABEL = "synthetic"


def _lineage_rate(records: list[dict], predicate) -> stats.Rate:
    """The rate at which distinct recordings produce an outcome, with exact bounds.

    A lineage counts once, and counts as positive if any of its clips in this set satisfies the
    predicate. For base-clip sets that is one clip per lineage and the reduction is a no-op; for a
    derivative stratum it is what stops one recording's eight degradations from being eight
    observations of one fact.
    """
    by_lineage: dict[str, bool] = {}
    for record in records:
        lineage = record["source_lineage_id"]
        by_lineage[lineage] = by_lineage.get(lineage, False) or bool(predicate(record))
    return stats.rate(sum(by_lineage.values()), len(by_lineage))


def _quantile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank quantile: an observed value, never an interpolation between two."""
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _distribution(values: list[float]) -> dict:
    """Where a set of scores actually sits. The tail is the part that decides a threshold."""
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


def auroc(positive: list[float], negative: list[float]) -> float | None:
    """Area under the ROC curve, by the Mann-Whitney U identity, with ties counted as halves.

    Computed from ranks rather than by integrating a sampled ROC curve, because the rank form is
    exact: it is the probability that a random positive outscores a random negative, plus half the
    probability that they tie. Ties matter here — a saturating detector returns exactly 1.0 on many
    clips — and a trapezoidal integration over a coarse threshold grid would silently round them.

    `None` when either class is empty, where the quantity is undefined rather than zero.
    """
    if not positive or not negative:
        return None

    combined = sorted([(value, 1) for value in positive] + [(value, 0) for value in negative])

    # Average ranks within each tied block, which is what makes ties contribute exactly one half.
    ranks: list[float] = [0.0] * len(combined)
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1

    positive_rank_sum = sum(
        rank for rank, (_, label) in zip(ranks, combined) if label == 1
    )
    n_positive, n_negative = len(positive), len(negative)
    u_statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return u_statistic / (n_positive * n_negative)


def join_run(items: list[CorpusItem], run: dict) -> list[dict]:
    """One record per clip: what the corpus says about it and what Effort said.

    `status` is reduced to two states a rate can be computed over. `ok` is a score; `abstained` is
    the detector declining to answer, whether because no sampled frame held a face or because the
    clip would not decode. Both arrive from the harness as an error record — the harness has one
    failure channel — and the distinction is preserved in `error` for the failure table.
    """
    by_id = {item.clip_id: item for item in items}
    detail = (run.get("run", {}).get("model_provenance") or {}).get(
        "frame_detail_by_clip_id", {}
    )

    joined = []
    for record in run["clips"]:
        item = by_id.get(record["clip_id"])
        if item is None:
            continue
        frames = detail.get(record["clip_id"], {})
        joined.append(
            {
                "clip_id": item.clip_id,
                "label": item.label,
                "family": item.family,
                "stratum": item.stratum_primary,
                "split": item.split,
                "source_lineage_id": item.source_lineage_id,
                "is_base": item.is_base,
                "is_manipulated": item.is_manipulated,
                "private": item.private,
                "status": "ok" if record["status"] == "ok" else "abstained",
                "score": record["score"],
                "error": record.get("error"),
                "latency_ms": record.get("latency_ms"),
                "frames_requested": frames.get("frames_requested"),
                "frames_decoded": frames.get("frames_decoded"),
                "frames_with_face": frames.get("frames_with_face"),
                "frame_scores": frames.get("frame_scores"),
            }
        )
    return joined


def _readable(records: list[dict]) -> list[dict]:
    return [record for record in records if record["status"] == "ok"]


def _flagged(record: dict, threshold: float) -> bool:
    """The frozen decision rule: strictly greater than the operating threshold.

    Strict, because that is what upstream's `get_video_metrics` does — `(video_pred > 0.5)` — and
    what a two-class `argmax` amounts to. The harness's own `--threshold` is an at-or-above
    comparison; on a continuous score the two differ only on an exact tie, but the rule that
    decides every figure in the report is applied here, from the freeze, rather than inherited.
    """
    return record["score"] is not None and record["score"] > threshold


def genuine_primary(records: list[dict], threshold: float) -> dict:
    """The headline: false positives over independent genuine lineages, base clips only.

    This is the only genuine figure the acceptance target is stated over. One observation per
    lineage, taken from the clip as it was acquired, with no constructed degradation anywhere in
    the denominator or the numerator.
    """
    base = [
        record
        for record in records
        if record["label"] == LABEL_REAL and record["is_base"]
    ]
    readable = _readable(base)

    false_positive_rate = _lineage_rate(readable, lambda r: _flagged(r, threshold))
    abstention = _lineage_rate(base, lambda r: r["status"] == "abstained")

    flagged = sorted(
        (r for r in readable if _flagged(r, threshold)),
        key=lambda r: r["score"],
        reverse=True,
    )

    return {
        "unit": "one independent genuine source_lineage_id, represented by its base clip",
        "lineages_total": len({r["source_lineage_id"] for r in base}),
        "lineages_readable": len({r["source_lineage_id"] for r in readable}),
        "false_positives": false_positive_rate.as_dict(),
        "false_positive_summary": false_positive_rate.describe(),
        "abstention": abstention.as_dict(),
        "score_distribution": _distribution([r["score"] for r in readable]),
        "flagged_lineages": [
            {
                "source_lineage_id": r["source_lineage_id"],
                "clip_id": None if r["private"] else r["clip_id"],
                "stratum": r["stratum"],
                "score": r["score"],
                "frames_with_face": r["frames_with_face"],
            }
            for r in flagged
        ],
        "highest_scoring": [
            {
                "source_lineage_id": r["source_lineage_id"],
                "clip_id": None if r["private"] else r["clip_id"],
                "stratum": r["stratum"],
                "score": r["score"],
                "frames_with_face": r["frames_with_face"],
            }
            for r in sorted(readable, key=lambda r: r["score"], reverse=True)[:10]
        ],
    }


def manipulated_report(records: list[dict], label: str, threshold: float) -> dict:
    """Detection over independent manipulated lineages of one label, base clips only.

    Reported per family as well as in total, because a detection rate averaged over families the
    checkpoint saw in training and families it never met describes neither of them.
    """
    base = [record for record in records if record["label"] == label and record["is_base"]]
    readable = _readable(base)

    detected = _lineage_rate(readable, lambda r: _flagged(r, threshold))
    abstention = _lineage_rate(base, lambda r: r["status"] == "abstained")

    families: dict[str, dict] = {}
    for family in sorted({record["family"] for record in base}):
        family_readable = [r for r in readable if r["family"] == family]
        family_all = [r for r in base if r["family"] == family]
        families[family] = {
            "detected": _lineage_rate(
                family_readable, lambda r: _flagged(r, threshold)
            ).as_dict(),
            "abstention": _lineage_rate(
                family_all, lambda r: r["status"] == "abstained"
            ).as_dict(),
        }

    missed = sorted(
        (r for r in readable if not _flagged(r, threshold)), key=lambda r: r["score"]
    )

    return {
        "label": label,
        "lineages_total": len({r["source_lineage_id"] for r in base}),
        "lineages_readable": len({r["source_lineage_id"] for r in readable}),
        "detected": detected.as_dict(),
        "detected_summary": detected.describe(),
        "false_negative_rate": (
            None if detected.observed is None else 1.0 - detected.observed
        ),
        "abstention": abstention.as_dict(),
        "score_distribution": _distribution([r["score"] for r in readable]),
        "by_family": families,
        "lowest_scoring_missed": [
            {
                "clip_id": r["clip_id"],
                "family": r["family"],
                "score": r["score"],
                "frames_with_face": r["frames_with_face"],
            }
            for r in missed[:10]
        ],
    }


def derivative_robustness(records: list[dict], threshold: float) -> dict:
    """Constructed degradations: robustness evidence, never independent false positives.

    Every figure here is explicitly clip-level, and every stratum states how many distinct
    lineages sit behind it. The seven constructed genuine strata apply one transform each to the
    same small set of lineages, so their rates are paired repeated measurements: they may be
    compared with one another and must not be added, and none of them adds evidence to the
    headline.
    """
    derivatives = [
        record
        for record in records
        if record["label"] == LABEL_REAL and not record["is_base"]
    ]
    readable = _readable(derivatives)
    flagged = [record for record in readable if _flagged(record, threshold)]

    strata: dict[str, dict] = {}
    for stratum in sorted({record["stratum"] for record in derivatives}):
        in_stratum = [r for r in derivatives if r["stratum"] == stratum]
        stratum_readable = _readable(in_stratum)
        stratum_flagged = [r for r in stratum_readable if _flagged(r, threshold)]
        lineage_view = _lineage_rate(stratum_readable, lambda r: _flagged(r, threshold))
        strata[stratum] = {
            "clips": len(in_stratum),
            "clips_readable": len(stratum_readable),
            "clips_flagged": len(stratum_flagged),
            "distinct_lineages": len({r["source_lineage_id"] for r in in_stratum}),
            "within_stratum_lineage_rate": lineage_view.as_dict(),
            "within_stratum_lineage_summary": lineage_view.describe(),
            "abstention": _lineage_rate(
                in_stratum, lambda r: r["status"] == "abstained"
            ).as_dict(),
            "score_distribution": _distribution([r["score"] for r in stratum_readable]),
        }

    # How many lineages pass as they arrive but fail once degraded. This is the fragility measure:
    # it is about the same recordings already counted in the headline, so it is reported here and
    # not there.
    base_by_lineage = {
        record["source_lineage_id"]: record
        for record in records
        if record["label"] == LABEL_REAL and record["is_base"]
    }
    flipped = []
    for record in flagged:
        base = base_by_lineage.get(record["source_lineage_id"])
        if base is not None and base["status"] == "ok" and not _flagged(base, threshold):
            flipped.append(record["source_lineage_id"])

    return {
        "unit": "clip (constructed derivative), reported as robustness evidence only",
        "note": (
            "These clips are repeated measurements on lineages already counted in the primary "
            "evaluation. They are never counted as independent false positives."
        ),
        "derivative_clips": len(derivatives),
        "derivative_clips_readable": len(readable),
        "derivative_clips_flagged": len(flagged),
        "distinct_lineages_behind_flagged_clips": len(
            {record["source_lineage_id"] for record in flagged}
        ),
        "lineages_carrying_derivatives": len(
            {record["source_lineage_id"] for record in derivatives}
        ),
        "lineages_clean_as_acquired_but_flagged_when_degraded": len(set(flipped)),
        "by_stratum": strata,
    }


def abstention_structure(records: list[dict], threshold: float) -> dict:
    """Whether readings built on fewer usable frames are the ones that go wrong.

    R7-T5 found exactly this for LipForensics: a score averaged over one window was three times as
    likely to be a false positive as one averaged over four. Effort's analogue is how many of the
    eight sampled frames yielded a face, so the same question is asked of it. Clip-level, over
    genuine clips, and therefore robustness evidence rather than an independent rate.
    """
    genuine = _readable([r for r in records if r["label"] == LABEL_REAL])
    table: dict[str, dict] = {}
    for record in genuine:
        key = str(record["frames_with_face"])
        entry = table.setdefault(key, {"clips": 0, "flagged": 0, "rate": None})
        entry["clips"] += 1
        if _flagged(record, threshold):
            entry["flagged"] += 1
    for entry in table.values():
        entry["rate"] = entry["flagged"] / entry["clips"] if entry["clips"] else None
    return dict(sorted(table.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else -1))


def threshold_sensitivity(records: list[dict], thresholds: list[float]) -> list[dict]:
    """What the genuine and primary-task counts would be at other operating points.

    Descriptive only. Reported so a reader can see the shape of the trade-off, exactly as R7-T5
    reported LipForensics' candidates; nothing here is adopted, and the split each row is computed
    over is named in the caller.
    """
    genuine = _readable(
        [r for r in records if r["label"] == LABEL_REAL and r["is_base"]]
    )
    primary = _readable(
        [r for r in records if r["label"] == PRIMARY_LABEL and r["is_base"]]
    )
    ood = _readable([r for r in records if r["label"] == OOD_LABEL and r["is_base"]])

    rows = []
    for threshold in thresholds:
        genuine_rate = _lineage_rate(genuine, lambda r: _flagged(r, threshold))
        rows.append(
            {
                "threshold": threshold,
                "genuine_lineages_flagged": genuine_rate.numerator,
                "genuine_lineages": genuine_rate.denominator,
                "genuine_fpr": genuine_rate.observed,
                "genuine_fpr_upper_95": genuine_rate.upper_95,
                "face_swap_lineages_detected": _lineage_rate(
                    primary, lambda r: _flagged(r, threshold)
                ).numerator,
                "face_swap_lineages": len({r["source_lineage_id"] for r in primary}),
                "synthetic_lineages_detected": _lineage_rate(
                    ood, lambda r: _flagged(r, threshold)
                ).numerator,
                "synthetic_lineages": len({r["source_lineage_id"] for r in ood}),
            }
        )
    return rows


def calibration_candidate(records: list[dict]) -> dict:
    """A candidate operating point derived from calibration records alone.

    The same selection rule R4-T1 and R5-T3 used, and that R7-T5 re-ran: place the threshold just
    above the highest genuine score observed, at the midpoint between it and the next distinct
    score above it. Reported for context — the headline uses upstream's 0.5 — and computed here
    only over records the caller has already restricted to the calibration split.
    """
    genuine = _readable([r for r in records if r["label"] == LABEL_REAL and r["is_base"]])
    if not genuine:
        return {"available": False}

    genuine_scores = sorted(record["score"] for record in genuine)
    highest_genuine = genuine_scores[-1]

    manipulated = _readable([r for r in records if r["is_manipulated"] and r["is_base"]])
    above = sorted(
        score
        for score in (record["score"] for record in manipulated)
        if score > highest_genuine
    )
    midpoint = (highest_genuine + above[0]) / 2.0 if above else None

    return {
        "available": True,
        "derived_on": SPLIT_CALIBRATION,
        "genuine_lineages": len({r["source_lineage_id"] for r in genuine}),
        "highest_genuine_score": highest_genuine,
        "lowest_manipulated_score_above_it": above[0] if above else None,
        "midpoint_candidate": midpoint,
        "note": (
            "A candidate, not an adoption. The report's headline figures use the frozen upstream "
            "threshold of 0.5. This number is computed on calibration records only and never "
            "touches the evaluation split."
        ),
    }


def compute_cost(run: dict) -> dict:
    """What the run cost, from the harness's own timing and the model's own GPU accounting."""
    provenance = run.get("run", {}).get("model_provenance") or {}
    performance = run.get("performance", {})
    return {
        "clips": run.get("dataset", {}).get("clip_count"),
        "duration_s": run.get("run", {}).get("duration_s"),
        "latency_ms": performance.get("latency"),
        "process_memory_mb": performance.get("memory"),
        "gpu": provenance.get("gpu"),
        "device": (provenance.get("load") or {}).get("device"),
        "torch_version": provenance.get("torch_version"),
    }


def failure_report(records: list[dict]) -> dict:
    """Every clip that produced no reading, and why, grouped by the reason it gave."""
    abstained = [record for record in records if record["status"] == "abstained"]
    reasons: dict[str, int] = {}
    for record in abstained:
        error = record["error"] or "unknown"
        kind = (
            "no_face_detected"
            if "no face detected" in error
            else "decode_failure"
            if "could not be opened" in error
            else "other"
        )
        reasons[kind] = reasons.get(kind, 0) + 1

    overall = stats.rate(len(abstained), len(records))
    return {
        "abstained_clips": len(abstained),
        "clips_total": len(records),
        "clip_level_rate": overall.as_dict(),
        "by_reason": dict(sorted(reasons.items())),
        "by_label": {
            label: stats.rate(
                sum(
                    1
                    for r in records
                    if r["label"] == label and r["status"] == "abstained"
                ),
                sum(1 for r in records if r["label"] == label),
            ).as_dict()
            for label in sorted({record["label"] for record in records})
        },
    }


def regression_report(records: list[dict], threshold: float) -> list[dict]:
    """The known benign regression lineages, by opaque identifier and measured output only.

    No filename, no path, no personal metadata and no raw media reaches this structure. The base
    clip is the lineage-level observation and is counted once in the primary evaluation like any
    other genuine lineage; its constructed variants appear beneath it as robustness evidence and
    are counted nowhere else.
    """
    private = [record for record in records if record["private"]]
    reports = []
    for lineage in sorted({record["source_lineage_id"] for record in private}):
        clips = [record for record in private if record["source_lineage_id"] == lineage]
        base = next((record for record in clips if record["is_base"]), None)
        variants = [record for record in clips if not record["is_base"]]

        def view(record: dict | None) -> dict | None:
            if record is None:
                return None
            return {
                "stratum": record["stratum"],
                "status": record["status"],
                "score": record["score"],
                "flagged": _flagged(record, threshold) if record["status"] == "ok" else None,
                "frames_with_face": record["frames_with_face"],
            }

        readable_variants = [v for v in variants if v["status"] == "ok"]
        reports.append(
            {
                "source_lineage_id": lineage,
                "as_acquired": view(base),
                "counted_once_at_lineage_level": True,
                "variants_are_robustness_evidence_only": True,
                "variant_count": len(variants),
                "variants_flagged": sum(
                    1 for v in readable_variants if _flagged(v, threshold)
                ),
                "variants": sorted(
                    (view(variant) for variant in variants),
                    key=lambda entry: (entry["score"] is None, -(entry["score"] or 0)),
                ),
            }
        )
    return reports


def build_analysis(
    items: list[CorpusItem],
    calibration_run: dict,
    evaluation_run: dict,
    freeze: dict,
) -> dict:
    threshold = freeze["frozen_protocol"]["7_operating_threshold"]["value"]

    calibration = join_run(items, calibration_run)
    evaluation = join_run(items, evaluation_run)

    # Guard the whole point of the exercise. If a record from one split turned up in the other's
    # run, every hold-out claim below would be void, so it is asserted rather than assumed.
    splits_seen = {
        SPLIT_CALIBRATION: {record["split"] for record in calibration},
        SPLIT_EVALUATION: {record["split"] for record in evaluation},
    }

    genuine_base_eval = _readable(
        [r for r in evaluation if r["label"] == LABEL_REAL and r["is_base"]]
    )
    primary_base_eval = _readable(
        [r for r in evaluation if r["label"] == PRIMARY_LABEL and r["is_base"]]
    )
    ood_base_eval = _readable(
        [r for r in evaluation if r["label"] == OOD_LABEL and r["is_base"]]
    )

    return {
        "schema_version": "r7-t7-effort-analysis-1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": {
            "detector": "effort",
            "variant": freeze["candidate"]["checkpoint"]["variant"],
            "checkpoint_sha256": freeze["candidate"]["checkpoint"]["sha256"],
            "upstream_revision": freeze["candidate"]["upstream_revision"],
        },
        "frozen_protocol": freeze["frozen_protocol"],
        "operating_threshold": threshold,
        "split_integrity": {
            "calibration_run_contains_only": sorted(splits_seen[SPLIT_CALIBRATION]),
            "evaluation_run_contains_only": sorted(splits_seen[SPLIT_EVALUATION]),
            "clean": splits_seen[SPLIT_CALIBRATION] == {SPLIT_CALIBRATION}
            and splits_seen[SPLIT_EVALUATION] == {SPLIT_EVALUATION},
        },
        "evaluation": {
            "genuine_primary": genuine_primary(evaluation, threshold),
            "primary_task_face_swap": manipulated_report(
                evaluation, PRIMARY_LABEL, threshold
            ),
            "cross_family_ood_synthetic": manipulated_report(
                evaluation, OOD_LABEL, threshold
            ),
            "auroc": {
                "note": (
                    "Computed only where both classes are semantically comparable: genuine "
                    "against the detector's primary task, and genuine against the "
                    "out-of-distribution families, never both pooled together."
                ),
                "genuine_vs_face_swap": auroc(
                    [r["score"] for r in primary_base_eval],
                    [r["score"] for r in genuine_base_eval],
                ),
                "genuine_vs_synthetic_ood": auroc(
                    [r["score"] for r in ood_base_eval],
                    [r["score"] for r in genuine_base_eval],
                ),
                "n_genuine": len(genuine_base_eval),
                "n_face_swap": len(primary_base_eval),
                "n_synthetic": len(ood_base_eval),
            },
            "derivative_robustness": derivative_robustness(evaluation, threshold),
            "abstention_structure_genuine": abstention_structure(evaluation, threshold),
            "failures": failure_report(evaluation),
            "threshold_sensitivity": threshold_sensitivity(
                evaluation, [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            ),
            "regression_lineages": regression_report(evaluation, threshold),
            "compute": compute_cost(evaluation_run),
        },
        "calibration": {
            "genuine_primary": genuine_primary(calibration, threshold),
            "primary_task_face_swap": manipulated_report(
                calibration, PRIMARY_LABEL, threshold
            ),
            "cross_family_ood_synthetic": manipulated_report(
                calibration, OOD_LABEL, threshold
            ),
            "candidate_threshold": calibration_candidate(calibration),
            "failures": failure_report(calibration),
            "compute": compute_cost(calibration_run),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_effort",
        description="Measure the Effort shadow runs under the frozen R7-T7 protocol.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--calibration-run", required=True, type=Path)
    parser.add_argument("--evaluation-run", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    items = read_corpus(args.corpus)
    calibration_run = json.loads(args.calibration_run.read_text(encoding="utf-8"))
    evaluation_run = json.loads(args.evaluation_run.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))

    analysis = build_analysis(items, calibration_run, evaluation_run, freeze)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "effort_analysis.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    genuine = analysis["evaluation"]["genuine_primary"]
    primary = analysis["evaluation"]["primary_task_face_swap"]
    ood = analysis["evaluation"]["cross_family_ood_synthetic"]

    print(f"wrote {output}")
    print(f"split integrity clean: {analysis['split_integrity']['clean']}")
    print(f"operating threshold: {analysis['operating_threshold']}")
    print(f"genuine (primary, lineage-level): {genuine['false_positive_summary']}")
    print(f"face swaps detected: {primary['detected_summary']}")
    print(f"synthetic (OOD) detected: {ood['detected_summary']}")
    print(f"AUROC genuine vs face_swap: {analysis['evaluation']['auroc']['genuine_vs_face_swap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
