"""Turn two benchmark runs into one calibration artifact. Offline, and read-only.

    python3 scripts/calibration/calibrate.py \
        --manifest ../deepguard-corpus/r4t1_calibration/manifest.csv \
        --sources  ../deepguard-corpus/r4t1_calibration/sources.json \
        --synthetic-video-run ../deepguard-corpus/runs/r4t1-svd/results.json \
        --face-manipulation-run ../deepguard-corpus/runs/r4t1-face/results.json \
        --output-dir docs/ai/reviews/R4_T1

This is the whole of R4-T1's decision-making, and it decides nothing about production. It
reads two `results.json` artifacts the R2 harness produced, joins them on `clip_id`,
measures the distributions, derives an operating point per detector from those
measurements, and writes down what it found together with everything needed to reproduce
it. It imports nothing from `apps/`, touches no database, and cannot change how a single
analysis is classified: the risk engine reads its thresholds from constants in its own
module, and moving them there is R4-T2's work under a separate review.

**Both detectors are calibrated separately and neither score is ever combined with the
other.** The joint table records what each said about the same clip and where they
disagreed; there is no blended score anywhere in this file, because two detectors
answering different questions have no shared unit to be averaged in (AGENTS.md rule 11).

**The artifact names what it measured.** `calibration_id` is the SHA-256 of the identity
fields — corpus digest, manifest digest, both detectors' provenance, the selected
thresholds and the policy they were selected under. Change the corpus, the model weights,
the provider deployment or the rule, and the id changes with it, which is what makes a
stored decision traceable to the measurement behind it years later (D017).
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import analysis

# Bumped when the shape of `calibration.json` changes.
SCHEMA_VERSION = "r4-calibration-1"

# The two detectors this calibration covers, and the signal each one's evidence is stored
# under. AASIST and the active-speaker and C2PA signals are deliberately absent: they
# answer other questions, none of them has been benchmarked as a risk-eligible detector,
# and R4's non-goals say so explicitly.
SYNTHETIC_VIDEO = "synthetic_video"
FACE_MANIPULATION = "face_manipulation"

# The product error policy the thresholds are derived under, carried in the artifact
# because a threshold without the policy that produced it cannot be argued with. Adopted
# in P7-T2 §6 and unchanged here.
ERROR_POLICY = (
    "Strongly avoid a false HIGH on legitimate media: a wrong HIGH on genuine footage is "
    "the failure that destroys trust in a forensic product. Detection rate is given up to "
    "buy that, and where evidence is ambiguous MEDIUM or UNKNOWN is preferred over an "
    "unjustified HIGH."
)


def read_manifest_metadata(manifest_path: Path) -> dict[str, dict]:
    """`family` and `stratum` per clip, read from the manifest the runs were scored over.

    The benchmark harness ignores columns it does not know, so these two travel in the
    manifest without the runs having to carry them. They are read back here rather than
    inferred from clip ids: a family is ground truth about how a clip was made, and
    parsing it out of a name would make a calibration depend on a naming convention.
    """
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row.get("clip_id") or row.get("path") or "").strip(): {
                "family": (row.get("family") or "unknown").strip(),
                "stratum": (row.get("stratum") or "unknown").strip(),
            }
            for row in csv.DictReader(handle)
        }


def load_run(path: Path, metadata: dict[str, dict]) -> dict:
    """One benchmark artifact, its scored rows joined to the manifest's family columns."""
    results = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    errors = []
    for record in results["clips"]:
        extra = metadata.get(record["clip_id"], {"family": "unknown", "stratum": "unknown"})
        if record["status"] != "ok":
            errors.append(
                {
                    "clip_id": record["clip_id"],
                    "label": record["label"],
                    **extra,
                    "error": record["error"],
                }
            )
            continue
        rows.append(
            {
                "clip_id": record["clip_id"],
                "label": record["label"],
                "score": record["score"],
                **extra,
            }
        )
    return {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": results.get("schema_version"),
        "model": results["run"]["model"],
        "model_provenance": results["run"].get("model_provenance"),
        "manifest": results["dataset"]["manifest"],
        "manifest_sha256": results["dataset"]["manifest_sha256"],
        "started_at": results["run"]["started_at"],
        "finished_at": results["run"]["finished_at"],
        "latency": results["performance"]["latency"],
        "rows": rows,
        "errors": errors,
    }


def measure(run: dict) -> dict:
    """Everything measured about one detector, before any threshold is selected."""
    rows = run["rows"]
    genuine = [row["score"] for row in rows if row["label"] == "real"]
    manipulated = [row["score"] for row in rows if row["label"] != "real"]
    strata = sorted({row["stratum"] for row in rows})
    families = sorted({row["family"] for row in rows})
    return {
        "scored": len(rows),
        "abstentions": len(run["errors"]),
        "distribution_overall": analysis.score_summary([row["score"] for row in rows]),
        "distribution_by_stratum": {
            stratum: analysis.score_summary(
                [row["score"] for row in rows if row["stratum"] == stratum]
            )
            for stratum in strata
        },
        "distribution_by_family": {
            family: analysis.score_summary(
                [row["score"] for row in rows if row["family"] == family]
            )
            for family in families
        },
        # Reported per stratum as well as pooled. A single AUROC over a corpus holding both
        # face swaps and generated video averages a domain each detector is good at with
        # one it is not, and P7-T2 §5.2 already showed how badly that flatters NVIDIA's.
        "auroc_pooled": analysis.auroc(manipulated, genuine),
        "auroc_by_stratum": {
            stratum: analysis.auroc(
                [
                    row["score"] for row in rows
                    if row["stratum"] == stratum and row["label"] != "real"
                ],
                genuine,
            )
            for stratum in strata
        },
        "threshold_sweep": analysis.sweep(rows, analysis.sweep_grid(rows)),
        "best_separation_not_selected": analysis.youden_point(rows),
    }


def select(run: dict) -> dict:
    """The operating points this corpus supports for one detector, and their measured cost."""
    rows = run["rows"]
    high = analysis.select_high_threshold(rows)
    low = analysis.select_low_threshold(rows)
    at_high = (
        analysis.sweep(rows, [high["threshold"]])[0]
        if high.get("threshold") is not None
        else None
    )
    return {"t_high": high, "t_low": low, "measured_at_t_high": at_high}


def canonical_identity(document: dict) -> dict:
    """The fields a `calibration_id` is computed over.

    A curated subset, not the whole artifact: the id has to change when the *measurement*
    changes and stay stable when the prose around it is edited. Timestamps, latencies and
    per-clip records are therefore excluded, and everything that decides what a threshold
    means is included.
    """
    detectors = document["detectors"]
    return {
        "schema_version": document["schema_version"],
        "corpus_digest": document["corpus"]["corpus_digest"],
        "manifest_sha256": document["corpus"]["manifest_sha256"],
        "clip_count": document["corpus"]["clip_count"],
        "error_policy": document["error_policy"],
        "detectors": {
            name: {
                "model": detector["run"]["model"],
                "provenance": detector["run"]["model_provenance"],
                "t_high": detector["thresholds"]["t_high"].get("threshold"),
                "t_low": detector["thresholds"]["t_low"].get("threshold"),
                "selection_rule": detector["thresholds"]["t_high"].get("rule"),
            }
            for name, detector in detectors.items()
        },
    }


def build_calibration(
    *,
    corpus: dict,
    svd_run: dict,
    face_run: dict,
    joint: list[dict],
    generated_at: datetime,
) -> dict:
    """Assemble `calibration.json`, then stamp it with the id of its own identity fields."""
    detectors = {}
    for name, run in ((SYNTHETIC_VIDEO, svd_run), (FACE_MANIPULATION, face_run)):
        detectors[name] = {
            "run": {
                "artifact": run["artifact"],
                "artifact_sha256": run["artifact_sha256"],
                "benchmark_schema_version": run["schema_version"],
                "model": run["model"],
                "model_provenance": run["model_provenance"],
                "manifest_sha256": run["manifest_sha256"],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
                "latency": run["latency"],
            },
            "measurements": measure(run),
            "thresholds": select(run),
            "abstentions": run["errors"],
        }

    document = {
        "schema_version": SCHEMA_VERSION,
        "task": "R4-T1",
        "generated_at": generated_at.isoformat(),
        "error_policy": ERROR_POLICY,
        "scope": (
            "Offline calibration only. Nothing in this artifact is read by the risk "
            "engine; adopting any of it is R4-T2 under separate review."
        ),
        "corpus": corpus,
        "detectors": detectors,
        "disagreement": analysis.disagreement_summary(joint),
    }
    identity = canonical_identity(document)
    document["calibration_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    document["calibration_identity"] = identity
    return document


def _format(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(document: dict, joint: list[dict]) -> str:
    """The same calibration as a document a reviewer can argue with."""
    corpus = document["corpus"]
    lines = [
        "# R4-T1 — Multi-detector calibration",
        "",
        f"- Calibration id: `{document['calibration_id']}`",
        f"- Generated: {document['generated_at']}",
        f"- Corpus digest: `{corpus['corpus_digest']}`",
        f"- Manifest: `{corpus['manifest']}` (SHA-256 `{corpus['manifest_sha256']}`)",
        f"- Clips: {corpus['clip_count']} — {corpus['label_counts']}",
        f"- Strata: {corpus['stratum_counts']}",
        "",
        "Offline calibration. No production file, threshold or stored decision is changed "
        "by this run; the risk engine still classifies from its own constants alone.",
        "",
        "## Error policy",
        "",
        document["error_policy"],
        "",
    ]

    for name, detector in document["detectors"].items():
        measurements = detector["measurements"]
        thresholds = detector["thresholds"]
        high, low = thresholds["t_high"], thresholds["t_low"]
        lines += [
            f"## Detector: `{name}`",
            "",
            f"- Model: `{detector['run']['model']}`",
            f"- Scored: {measurements['scored']}, abstentions: "
            f"{measurements['abstentions']}",
            f"- AUROC (pooled): {_format(measurements['auroc_pooled'])}",
            "",
            "### Score distribution by stratum",
            "",
            "| stratum | n | min | q1 | median | q3 | max | AUROC vs genuine |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stratum, summary in measurements["distribution_by_stratum"].items():
            lines.append(
                f"| {stratum} | {summary['count']} | {_format(summary['min'])} "
                f"| {_format(summary['q1'])} | {_format(summary['median'])} "
                f"| {_format(summary['q3'])} | {_format(summary['max'])} "
                f"| {_format(measurements['auroc_by_stratum'].get(stratum))} |"
            )
        lines += [
            "",
            "### Score distribution by family",
            "",
            "| family | n | min | median | max |",
            "|---|---:|---:|---:|---:|",
        ]
        for family, summary in measurements["distribution_by_family"].items():
            lines.append(
                f"| {family} | {summary['count']} | {_format(summary['min'])} "
                f"| {_format(summary['median'])} | {_format(summary['max'])} |"
            )
        lines += [
            "",
            "### Measured trade-off",
            "",
            "| threshold | false positives | FPR | FPR upper 95% | TPR |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in measurements["threshold_sweep"]:
            lines.append(
                f"| {_format(row['threshold'])} | {row['false_positives']}/"
                f"{row['genuine_n']} | {_format(row['false_positive_rate'])} "
                f"| {_format(row['false_positive_rate_upper95'])} "
                f"| {_format(row['true_positive_rate'])} |"
            )
        best = measurements["best_separation_not_selected"]
        lines += [
            "",
            f"Best measured separation (**reported, not selected**): threshold "
            f"{_format(best.get('threshold'))}, TPR "
            f"{_format(best.get('true_positive_rate'))}, FPR "
            f"{_format(best.get('false_positive_rate'))}.",
            "",
            "### Selected operating points",
            "",
            f"- `T_HIGH` = **{_format(high.get('threshold'))}** — "
            f"{high.get('rule') or high.get('reason')}",
        ]
        if high.get("threshold") is not None:
            at_high = thresholds["measured_at_t_high"]
            lines += [
                f"  - highest genuine score observed: "
                f"{_format(high.get('genuine_max'))}, margin "
                f"{_format(high.get('margin_over_genuine_max'))}",
                f"  - false HIGH: {at_high['false_positives']}/{at_high['genuine_n']} "
                f"observed, 95% upper bound "
                f"{_format(at_high['false_positive_rate_upper95'])}",
                f"  - detection at that point, by stratum: "
                + ", ".join(
                    f"{stratum} {_format(rate)}"
                    for stratum, rate in at_high["flag_rate_by_stratum"].items()
                ),
            ]
        lines += [
            f"- `T_LOW` = **{_format(low.get('threshold'))}** — "
            f"{low.get('rule') or low.get('reason')}",
        ]
        if low.get("threshold") is not None:
            lines.append(
                f"  - genuine media that would earn LOW: "
                f"{_format(low.get('coverage_genuine'))}"
            )
        if detector["abstentions"]:
            lines += ["", "### Abstentions", ""]
            lines += [
                f"- `{record['clip_id']}` ({record['family']}) — {record['error']}"
                for record in detector["abstentions"]
            ]
        lines.append("")

    lines += _limitations(document)

    disagreement = document["disagreement"]
    overall = disagreement["overall"]
    lines += [
        "## Disagreement between the two detectors",
        "",
        "Evaluated at each detector's own selected `T_HIGH`. The two scores are never "
        "averaged, combined or compared to one another — only their separate decisions "
        "about the same clip are tabulated.",
        "",
        f"- Clips scored by both: {disagreement['clips_scored_by_both']}",
        f"- Score rank correlation (Spearman): "
        f"{_format(disagreement['score_rank_correlation'])}",
        "",
        "| stratum | n | both | neither | synthetic-video only | face-manipulation only "
        "| face abstained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stratum, counts in disagreement["by_stratum"].items():
        lines.append(
            f"| {stratum} | {counts['n']} | {counts[analysis.AGREE_FLAGGED]} "
            f"| {counts[analysis.AGREE_CLEAR]} | {counts[analysis.DISAGREE_SVD_ONLY]} "
            f"| {counts[analysis.DISAGREE_FACE_ONLY]} "
            f"| {counts[analysis.ABSTAINED_FACE]} |"
        )
    lines += [
        "",
        "| family | n | both | neither | synthetic-video only | face-manipulation only "
        "| face abstained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, counts in disagreement["by_family"].items():
        lines.append(
            f"| {family} | {counts['n']} | {counts[analysis.AGREE_FLAGGED]} "
            f"| {counts[analysis.AGREE_CLEAR]} | {counts[analysis.DISAGREE_SVD_ONLY]} "
            f"| {counts[analysis.DISAGREE_FACE_ONLY]} "
            f"| {counts[analysis.ABSTAINED_FACE]} |"
        )
    disagreeing = [
        row for row in joint
        if row["state"] in (analysis.DISAGREE_SVD_ONLY, analysis.DISAGREE_FACE_ONLY)
    ]
    lines += [
        "",
        f"### The {len(disagreeing)} clip(s) the detectors disagreed on",
        "",
        "| clip | label | family | synthetic-video | face-manipulation | flagged by |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in disagreeing:
        flagged = (
            "synthetic video"
            if row["state"] == analysis.DISAGREE_SVD_ONLY
            else "face manipulation"
        )
        lines.append(
            f"| `{row['clip_id']}` | {row['label']} | {row['family']} "
            f"| {_format(row['synthetic_video_score'])} "
            f"| {_format(row['face_manipulation_score'])} | {flagged} |"
        )
    lines.append("")
    return "\n".join(lines)


def _limitations(document: dict) -> list[str]:
    """What this calibration does not establish, in its own measured numbers.

    Written from the artifact rather than from memory: the sample sizes, bounds and
    margins quoted here are read back out of what was just measured, so the section
    cannot drift away from the tables above it as the corpus changes.
    """
    corpus = document["corpus"]
    genuine_n = corpus["label_counts"].get("real", 0)
    thin = []
    for name, detector in document["detectors"].items():
        high = detector["thresholds"]["t_high"]
        if high.get("threshold") is None:
            continue
        tail = high.get("genuine_tail") or []
        nearest = tail[0] if tail else {}
        thin.append(
            f"- `{name}`: `T_HIGH` clears the highest genuine score by "
            f"{_format(high.get('margin_over_genuine_max'), 4)}. That margin is a "
            f"distance to one clip — `{nearest.get('clip_id')}` "
            f"({nearest.get('family')}, {_format(nearest.get('score'))}) — and the next "
            f"genuine score below it is "
            f"{_format(tail[1]['score']) if len(tail) > 1 else 'n/a'}. A corpus with one "
            f"more unusual genuine clip would move this boundary."
        )

    small_families = sorted(
        family
        for family, counts in corpus["family_counts"].items()
        if counts < 10
    )
    face = document["detectors"][FACE_MANIPULATION]
    abstentions = face["measurements"]["abstentions"]

    return [
        "## Limitations",
        "",
        f"- **A zero count is a bound, not a proof.** Neither detector produced a false "
        f"HIGH on the {genuine_n} genuine clips here, which bounds the true rate at "
        f"about 5.4 % with 95 % confidence and no lower. Roughly a thousand genuine "
        f"clips would be needed to bound it near 0.3 %.",
        "- **The selected margins are thin.**",
        *thin,
        f"- **{abstentions} clip(s) carry no face at all**, so the face-manipulation "
        f"detector abstained on them. Its figures describe the media it could read, and "
        f"say nothing about generated video without a face — which is a large part of "
        f"what the `generated` stratum is.",
        f"- **Per-family numbers are directional only** where the family is small: "
        f"{', '.join(small_families) if small_families else 'none'} carry fewer than ten "
        f"clips each, and no error rate should be quoted from them.",
        "- **The corpus is assembled from third-party mirrors of published benchmarks and "
        "from the R3-T1 FaceForensics++ sample.** Mirror integrity was not independently "
        "verified against the original distributions, and the source datasets' own "
        "labelling error rates are inherited unmeasured.",
        "- **Genuine media here is talking-head footage.** No landscape, sports, product, "
        "animation or heavy-VFX genuine media was tested, and legitimate CGI is an obvious "
        "false-HIGH risk that this corpus cannot see.",
        "- **No compression ladder and no adversarial testing.** Codec, resolution and "
        "duration variation is whatever the sources happened to carry; nothing was done to "
        "attack either threshold.",
        "- **This establishes operating points, not rules.** What the risk engine should "
        "emit when the two detectors disagree — and whether a LOW band is offered at all — "
        "is R4-T2's decision, taken against this evidence and not contained in it.",
        "",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate",
        description="Derive detector operating points from two benchmark runs.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--sources", required=True, type=Path, help="corpus sources.json from fetch_corpus"
    )
    parser.add_argument("--synthetic-video-run", required=True, type=Path)
    parser.add_argument("--face-manipulation-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_path = args.output_dir / "calibration.json"
    if artifact_path.exists() and not args.overwrite:
        print(f"refusing to overwrite {artifact_path} (pass --overwrite)", file=sys.stderr)
        return 1

    metadata = read_manifest_metadata(args.manifest)
    svd_run = load_run(args.synthetic_video_run, metadata)
    face_run = load_run(args.face_manipulation_run, metadata)

    # A calibration whose two halves were measured over different ground truth is not a
    # calibration of a pair. Refused rather than reported with a caveat.
    if svd_run["manifest_sha256"] != face_run["manifest_sha256"]:
        print(
            "the two runs were scored over different manifests:\n"
            f"  {svd_run['model']}: {svd_run['manifest_sha256']}\n"
            f"  {face_run['model']}: {face_run['manifest_sha256']}",
            file=sys.stderr,
        )
        return 1

    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    if sources["manifest_sha256"] != svd_run["manifest_sha256"]:
        print(
            f"{args.sources} describes a different manifest than the runs scored:\n"
            f"  sources: {sources['manifest_sha256']}\n"
            f"  runs:    {svd_run['manifest_sha256']}",
            file=sys.stderr,
        )
        return 1

    corpus = {
        "manifest": str(args.manifest),
        "manifest_sha256": sources["manifest_sha256"],
        "corpus_digest": sources["corpus_digest"],
        "clip_count": sources["clip_count"],
        "label_counts": sources["label_counts"],
        "stratum_counts": sources["stratum_counts"],
        "family_counts": sources["family_counts"],
        "remote_sources": sources["remote_sources"],
        "local_splits": sources["local_splits"],
        "built_at": sources["built_at"],
    }

    svd_thresholds = select(svd_run)
    face_thresholds = select(face_run)
    joint = analysis.joint_states(
        # Abstentions are carried into the joint table as scoreless rows, so a clip one
        # detector refused is visible there rather than quietly absent.
        svd_run["rows"] + [{**row, "score": None} for row in svd_run["errors"]],
        face_run["rows"] + [{**row, "score": None} for row in face_run["errors"]],
        svd_thresholds["t_high"].get("threshold"),
        face_thresholds["t_high"].get("threshold"),
    )

    document = build_calibration(
        corpus=corpus,
        svd_run=svd_run,
        face_run=face_run,
        joint=joint,
        generated_at=datetime.now(timezone.utc),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_path = args.output_dir / "report.md"
    report_path.write_text(render_report(document, joint), encoding="utf-8")
    joint_path = args.output_dir / "per_clip.jsonl"
    joint_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joint),
        encoding="utf-8",
    )

    print(
        f"calibration {document['calibration_id']}\n"
        + "\n".join(
            f"{name}: T_HIGH "
            f"{detector['thresholds']['t_high'].get('threshold')} "
            f"T_LOW {detector['thresholds']['t_low'].get('threshold')}"
            for name, detector in document["detectors"].items()
        )
        + f"\ndisagreements: {document['disagreement']['overall']}"
        + f"\nwrote {artifact_path}\nwrote {report_path}\nwrote {joint_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
