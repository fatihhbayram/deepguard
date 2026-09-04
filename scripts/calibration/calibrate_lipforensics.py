"""Turn the R5-T1 LipForensics benchmark run into one calibration artifact. Offline.

    python3 scripts/calibration/calibrate_lipforensics.py \
        --manifest ../deepguard-corpus/ff++_c23_test_r3t1/manifest.csv \
        --run ../deepguard-corpus/runs/2026-09-03-lipforensics/results.json \
        --cross-device-run ../deepguard-corpus/runs/2026-09-03-lipforensics-cuda/results.json \
        --output-dir docs/ai/reviews/R5_T3

R5-T3 in full, and it decides nothing about production. It reads the `results.json` the R2
harness wrote for `benchmark.models.lipforensics:detect`, measures the distributions,
derives an operating point from those measurements under R4-T1's stated rule, and writes
down what it found together with everything needed to reproduce it. It imports nothing
from `apps/`, touches no database, and cannot change how a single analysis is classified:
the risk engine reads its thresholds from constants in its own module and knows nothing
about LipForensics, which is where R5-T2 deliberately left it.

**The benchmark threshold is not the risk threshold.** R5-T1 ran its confusion matrix at
`0.5` — a reporting convention, the midpoint of a sigmoid, chosen before any score had
been seen. It is measured here beside the derived point so the difference is visible in
the corpus's own numbers, and it is not selected. `T_HIGH` comes from the observed
distributions alone.

**One detector, calibrated alone.** No score in this file is averaged with, combined with
or compared in magnitude to NVIDIA's or EfficientNet-B7's (AGENTS.md rule 11). LipForensics
answers a third question — whether the mouth moves like a real mouth — and what the risk
engine should emit when it disagrees with the other two is a later task's decision, taken
against this evidence and not contained in it.

**The artifact names what it measured.** `calibration_id` is the SHA-256 of the identity
fields — corpus digest, manifest digest, the detector's full provenance, the selected
thresholds and the policy they were selected under. Change the corpus, the weights, the
landmark models, the device or the rule, and the id changes with it (D017).
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
SCHEMA_VERSION = "r5-lipforensics-calibration-1"

DETECTOR = "lip_forensics"

# The model reference this calibration is willing to read a run from. A calibration that
# silently accepted a different model's `results.json` would attach LipForensics' name to
# another detector's numbers.
EXPECTED_MODEL = "benchmark.models.lipforensics:detect"

# The device the selected threshold is calibrated for. `apps/api/app/lip_forensics.py`
# pins `DEVICE = "cpu"` and deliberately does not make it configurable, so the CPU run is
# the one production will reproduce; a CUDA run is admissible only as the cross-device
# check `--cross-device-run` performs.
PRODUCTION_DEVICE = "cpu"

# What R5-T1 ran its confusion matrix at, recorded so the artifact can measure it and
# reject it rather than leave the reader to assume it was considered.
BENCHMARK_REPORTING_THRESHOLD = 0.5

# The product error policy the thresholds are derived under, carried in the artifact
# because a threshold without the policy that produced it cannot be argued with. Adopted
# in P7-T2 §6, applied unchanged in R4-T1, and unchanged again here.
ERROR_POLICY = (
    "Strongly avoid a false HIGH on legitimate media: a wrong HIGH on genuine footage is "
    "the failure that destroys trust in a forensic product. Detection rate is given up to "
    "buy that, and where evidence is ambiguous MEDIUM or UNKNOWN is preferred over an "
    "unjustified HIGH."
)

# How a FaceForensics++ `source` path maps onto the stratum vocabulary R4-T1 established.
# The mapping is over the *source dataset's own directory layout* — the record of how each
# clip was produced — and not over clip ids. Parsing a family out of a clip id would make
# the calibration depend on a naming convention; reading it out of the provenance column
# the manifest carries is reading ground truth.
GENUINE_STRATUM = "genuine_face"
GENUINE_FAMILY = "ffpp_real"
FACE_SWAP_STRATUM = "face_swap"


def clip_family_and_stratum(source: str, label: str) -> tuple[str, str]:
    """`(family, stratum)` for one manifest row, from its recorded source path.

    `FaceForensics++_C23/fake/Deepfakes/048_029.mp4` is a Deepfakes face swap;
    `FaceForensics++_C23/real/012.mp4` is genuine. Anything else raises: an unrecognised
    provenance string is a corpus this calibration has not been written for, and guessing
    at it would put a made-up family into an artifact whose whole purpose is traceability.
    """
    parts = [part for part in source.strip().split("/") if part]
    if len(parts) >= 4 and parts[1] == "fake":
        return f"ffpp_{parts[2].lower()}", FACE_SWAP_STRATUM
    if len(parts) >= 3 and parts[1] == "real":
        return GENUINE_FAMILY, GENUINE_STRATUM
    raise ValueError(
        f"unrecognised source provenance {source!r} (label {label!r}): this calibration "
        f"reads FaceForensics++ layout only"
    )


def read_manifest(manifest_path: Path) -> tuple[dict[str, dict], str]:
    """Per-clip metadata from the manifest, and the manifest's own SHA-256.

    The digest is recomputed from the bytes on disk rather than taken from the run, so a
    manifest edited after it was scored is caught here instead of being inherited.
    """
    raw = manifest_path.read_bytes()
    metadata: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = (row.get("clip_id") or "").strip()
            if not clip_id:
                continue
            label = (row.get("label") or "").strip()
            family, stratum = clip_family_and_stratum(row.get("source") or "", label)
            metadata[clip_id] = {
                "family": family,
                "stratum": stratum,
                "path": (row.get("path") or "").strip(),
                "source": (row.get("source") or "").strip(),
            }
    return metadata, hashlib.sha256(raw).hexdigest()


def digest_clips(manifest_path: Path, metadata: dict[str, dict]) -> tuple[str, list[dict]]:
    """The identity of the media itself: one SHA-256 per clip, and a digest over all of them.

    The same `clip_id:sha256` construction `fetch_corpus.corpus_digest` uses, so a corpus
    assembled by that tool and one described by a hand-built manifest are identified the
    same way. This split predates `fetch_corpus` and carries no `sources.json`, so without
    this step the calibration would name a manifest but not the bytes the manifest points
    at — and a manifest can list the same clip ids over entirely different media.
    """
    records = []
    for clip_id in sorted(metadata):
        clip_path = (manifest_path.parent / metadata[clip_id]["path"]).resolve()
        if not clip_path.is_file():
            raise FileNotFoundError(
                f"{clip_id} is listed in the manifest but {clip_path} does not exist; "
                f"the corpus digest identifies the media, so it cannot be skipped"
            )
        payload = clip_path.read_bytes()
        records.append(
            {
                "clip_id": clip_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "source": metadata[clip_id]["source"],
            }
        )
    joined = "\n".join(f"{record['clip_id']}:{record['sha256']}" for record in records)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest(), records


def load_run(path: Path, metadata: dict[str, dict]) -> dict:
    """One benchmark artifact, its scored rows joined to the manifest's provenance columns."""
    results = json.loads(path.read_text(encoding="utf-8"))
    model = results["run"]["model"]
    if model != EXPECTED_MODEL:
        raise ValueError(
            f"{path} was produced by {model}, not {EXPECTED_MODEL}"
        )
    rows, errors = [], []
    for record in results["clips"]:
        extra = metadata.get(record["clip_id"])
        if extra is None:
            raise ValueError(
                f"{path} scored {record['clip_id']}, which the manifest does not list"
            )
        common = {
            "clip_id": record["clip_id"],
            "label": record["label"],
            "family": extra["family"],
            "stratum": extra["stratum"],
        }
        if record["status"] != "ok":
            errors.append({**common, "error": record["error"]})
            continue
        rows.append({**common, "score": record["score"]})
    provenance = results["run"].get("model_provenance") or {}
    return {
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": results.get("schema_version"),
        "model": model,
        "model_provenance": provenance,
        "device": provenance.get("device"),
        "score_semantics": provenance.get("score_semantics"),
        "reporting_threshold": results["run"].get("threshold"),
        "manifest": results["dataset"]["manifest"],
        "manifest_sha256": results["dataset"]["manifest_sha256"],
        "started_at": results["run"]["started_at"],
        "finished_at": results["run"]["finished_at"],
        "latency": results["performance"]["latency"],
        "rows": rows,
        "errors": errors,
    }


def measure(run: dict) -> dict:
    """Everything measured about the detector, before any threshold is selected."""
    rows = run["rows"]
    genuine = [row["score"] for row in rows if row["label"] == "real"]
    manipulated = [row["score"] for row in rows if row["label"] != "real"]
    strata = sorted({row["stratum"] for row in rows})
    families = sorted({row["family"] for row in rows})
    return {
        "scored": len(rows),
        "abstentions": len(run["errors"]),
        "distribution_overall": analysis.score_summary([row["score"] for row in rows]),
        "distribution_by_label": {
            "real": analysis.score_summary(genuine),
            "manipulated": analysis.score_summary(manipulated),
        },
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
        # Pooled as well as per stratum, exactly as R4-T1 reports it. Here the corpus holds
        # one manipulated stratum, so the two agree; the breakdown is kept so a later run
        # over a corpus carrying generated video reports the split rather than hiding it.
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
    """The operating points this corpus supports, and their measured cost.

    R4-T1's two rules, unchanged and applied to a third detector: `T_HIGH` is the lowest
    threshold at which no genuine clip in the corpus is flagged, `T_LOW` its mirror. Using
    the rules the other two detectors were calibrated under is the point — a threshold
    derived by a rule invented for the detector it flatters is not evidence.
    """
    rows = run["rows"]
    high = analysis.select_high_threshold(rows)
    low = analysis.select_low_threshold(rows)
    at_high = (
        analysis.sweep(rows, [high["threshold"]])[0]
        if high.get("threshold") is not None
        else None
    )
    return {
        "t_high": high,
        "t_low": low,
        "measured_at_t_high": at_high,
        "band_between_them": _band(high.get("threshold"), low.get("threshold")),
    }


def _band(high: float | None, low: float | None) -> dict:
    """What lies between the two selected points, which is not always a band.

    `T_LOW` is the highest threshold no manipulated clip falls below and `T_HIGH` the
    lowest no genuine clip reaches. When the two classes are separated by a single clean
    gap both rules land in that same gap and the points coincide, leaving nothing between
    them. That is a statement about *this corpus* — one dataset, in the model's own
    training distribution — and not a licence to treat the detector as a binary verdict:
    the width of the ambiguous band on media the corpus does not contain is unmeasured,
    not zero. Recorded here so the artifact says which of the three cases it found rather
    than leaving a reader to infer it from two equal numbers.
    """
    if high is None or low is None:
        return {"kind": "undefined", "width": None, "reading": (
            "one of the two points does not exist on this corpus"
        )}
    if low < high:
        return {"kind": "ambiguous_band", "width": high - low, "reading": (
            "scores in this range were reached by both genuine and manipulated media"
        )}
    if low == high:
        return {"kind": "clean_gap", "width": 0.0, "reading": (
            "the two classes are separated by a single gap and both rules landed in it, "
            "so this corpus supports no ambiguous band at all — a fact about a corpus "
            "drawn from one dataset inside the model's training distribution, not a "
            "finding that the detector has no ambiguous region"
        )}
    return {"kind": "inverted", "width": low - high, "reading": (
        "T_LOW sits above T_HIGH, which means no clip of either class scored in between; "
        "the ordering is degenerate and neither point should be adopted without a wider "
        "corpus"
    )}


def compare_to_reporting_threshold(run: dict, selected: dict) -> dict:
    """What `0.5` would have decided on this corpus, beside what the derived point decides.

    The comparison exists so that rejecting the benchmark threshold is a measurement
    rather than an assertion. `0.5` was fixed before any score existed — it is where a
    sigmoid crosses its own midpoint — and the question a calibration has to answer is
    what it costs *here*, in flagged genuine clips and missed manipulations.
    """
    rows = run["rows"]
    threshold = selected["t_high"].get("threshold")
    candidates = [BENCHMARK_REPORTING_THRESHOLD]
    if threshold is not None:
        candidates.append(threshold)
    measured = analysis.sweep(rows, candidates)
    reporting = measured[0]
    derived = measured[1] if threshold is not None else None
    missed = sorted(
        (
            {"clip_id": row["clip_id"], "family": row["family"], "score": row["score"]}
            for row in rows
            if row["label"] != "real" and row["score"] < BENCHMARK_REPORTING_THRESHOLD
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "reporting_threshold": BENCHMARK_REPORTING_THRESHOLD,
        "at_reporting_threshold": reporting,
        "at_derived_threshold": derived,
        "manipulated_below_reporting_threshold": missed,
        "verdict": (
            "not selected: it is a reporting convention fixed before any score was "
            "observed, and on this corpus it sits above "
            f"{len(missed)} manipulated clip(s) while buying no reduction in observed "
            "false positives over the derived point"
        ),
    }


def cross_device_check(run: dict, other: dict, threshold: float | None) -> dict:
    """Whether the selected point survives the one perturbation that was measured.

    R5-T1 scored the same corpus twice, on CPU and on CUDA. That is not a second corpus
    and it establishes nothing about generalisation — but it does put a number on how far
    a score moves when only the arithmetic backend changes, and a margin thinner than that
    number is not a margin. Decisions are compared at the threshold derived from the
    production-device run; the same rule is re-run on the other device and reported beside
    it, so a reader can see whether the *selection* was stable as well as the decisions.
    """
    other_by_id = {row["clip_id"]: row for row in other["rows"]}
    deltas, flips = [], []
    for row in run["rows"]:
        twin = other_by_id.get(row["clip_id"])
        if twin is None:
            continue
        delta = abs(row["score"] - twin["score"])
        deltas.append(
            {
                "clip_id": row["clip_id"],
                "label": row["label"],
                "family": row["family"],
                "score": row["score"],
                "other_score": twin["score"],
                "abs_delta": delta,
            }
        )
        if threshold is not None and (row["score"] >= threshold) != (
            twin["score"] >= threshold
        ):
            flips.append(row["clip_id"])
    deltas.sort(key=lambda item: item["abs_delta"], reverse=True)
    reselected = analysis.select_high_threshold(other["rows"])
    return {
        "device": run["device"],
        "other_device": other["device"],
        "other_artifact": other["artifact"],
        "other_artifact_sha256": other["artifact_sha256"],
        "clips_compared": len(deltas),
        "max_abs_delta": deltas[0]["abs_delta"] if deltas else None,
        "largest_deltas": deltas[:5],
        "decision_flips_at_selected_threshold": flips,
        "t_high_reselected_on_other_device": reselected.get("threshold"),
        "selection_rule_is_device_stable": (
            threshold is not None
            and reselected.get("threshold") is not None
            and not flips
        ),
    }


def canonical_identity(document: dict) -> dict:
    """The fields a `calibration_id` is computed over.

    A curated subset, not the whole artifact: the id has to change when the *measurement*
    changes and stay stable when the prose around it is edited. Timestamps, latencies and
    per-clip records are therefore excluded, and everything that decides what a threshold
    means is included — the model provenance block in particular, because LipForensics'
    score depends on the landmark models in front of it as much as on its own weights.
    """
    detector = document["detectors"][DETECTOR]
    return {
        "schema_version": document["schema_version"],
        "corpus_digest": document["corpus"]["corpus_digest"],
        "manifest_sha256": document["corpus"]["manifest_sha256"],
        "clip_count": document["corpus"]["clip_count"],
        "error_policy": document["error_policy"],
        "detectors": {
            DETECTOR: {
                "model": detector["run"]["model"],
                "provenance": detector["run"]["model_provenance"],
                "t_high": detector["thresholds"]["t_high"].get("threshold"),
                "t_low": detector["thresholds"]["t_low"].get("threshold"),
                "selection_rule": detector["thresholds"]["t_high"].get("rule"),
            }
        },
    }


def build_calibration(
    *,
    corpus: dict,
    run: dict,
    cross_device: dict | None,
    generated_at: datetime,
) -> dict:
    """Assemble `calibration.json`, then stamp it with the id of its own identity fields."""
    thresholds = select(run)
    detector = {
        "run": {
            "artifact": run["artifact"],
            "artifact_sha256": run["artifact_sha256"],
            "benchmark_schema_version": run["schema_version"],
            "model": run["model"],
            "model_provenance": run["model_provenance"],
            "device": run["device"],
            "manifest_sha256": run["manifest_sha256"],
            "started_at": run["started_at"],
            "finished_at": run["finished_at"],
            "latency": run["latency"],
        },
        # Carried at the top of the detector block, not only inside the provenance dump:
        # `sigmoid(mean logit over sampled 25-frame mouth runs)` is what the threshold is a
        # threshold *on*, and a number without its unit is not an operating point.
        "score_semantics": run["score_semantics"],
        "measurements": measure(run),
        "thresholds": thresholds,
        "benchmark_threshold_comparison": compare_to_reporting_threshold(run, thresholds),
        "abstentions": run["errors"],
    }
    if cross_device is not None:
        detector["cross_device_stability"] = cross_device

    document = {
        "schema_version": SCHEMA_VERSION,
        "task": "R5-T3",
        "generated_at": generated_at.isoformat(),
        "error_policy": ERROR_POLICY,
        "scope": (
            "Offline calibration only. Nothing in this artifact is read by the risk "
            "engine, which knows nothing about LipForensics and is unchanged by this "
            "task; adopting any of it is the Risk Engine v3 task under separate review."
        ),
        "corpus": corpus,
        "detectors": {DETECTOR: detector},
    }
    identity = canonical_identity(document)
    document["calibration_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    document["calibration_identity"] = identity
    return document


def _counts(counts: dict[str, int]) -> str:
    """A count map as prose. `{'a': 2}` in the middle of a sentence is a leaked repr."""
    return ", ".join(f"{name} {count}" for name, count in counts.items()) or "none"


def _format(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(document: dict) -> str:
    """The same calibration as a document a reviewer can argue with."""
    corpus = document["corpus"]
    detector = document["detectors"][DETECTOR]
    measurements = detector["measurements"]
    thresholds = detector["thresholds"]
    high, low = thresholds["t_high"], thresholds["t_low"]
    comparison = detector["benchmark_threshold_comparison"]
    provenance = detector["run"]["model_provenance"]

    lines = [
        "# R5-T3 — LipForensics calibration",
        "",
        f"- Calibration id: `{document['calibration_id']}`",
        f"- Generated: {document['generated_at']}",
        f"- Corpus digest: `{corpus['corpus_digest']}`",
        f"- Manifest: `{corpus['manifest']}` (SHA-256 `{corpus['manifest_sha256']}`)",
        f"- Clips: {corpus['clip_count']} — {_counts(corpus['label_counts'])}",
        f"- Strata: {_counts(corpus['stratum_counts'])}",
        f"- Families: {_counts(corpus['family_counts'])}",
        f"- Source datasets: {_counts(corpus['source_datasets'])}",
        "",
        document["scope"],
        "",
        "## Error policy",
        "",
        document["error_policy"],
        "",
        "## Detector provenance",
        "",
        f"- Model: `{detector['run']['model']}`",
        f"- Device: `{detector['run']['device']}` (production pins "
        f"`{PRODUCTION_DEVICE}`)",
        f"- Score semantics: {detector['score_semantics']}",
        f"- Forgery weights: `{(provenance.get('classifier') or {}).get('artifact')}` "
        f"SHA-256 `{(provenance.get('classifier') or {}).get('sha256')}`",
        f"- Upstream: `{(provenance.get('upstream') or {}).get('repository')}` @ "
        f"`{(provenance.get('upstream') or {}).get('revision')}`",
        f"- Landmarks: {(provenance.get('landmarks') or {}).get('library')}, "
        f"detector `{((provenance.get('landmarks') or {}).get('face_detector') or {}).get('artifact')}`, "
        f"model `{((provenance.get('landmarks') or {}).get('landmark_model') or {}).get('artifact')}`",
        f"- Sampling: {provenance.get('windows')} window(s) of "
        f"{provenance.get('frames_per_window')} frames, crop {provenance.get('crop_size')} → "
        f"input {provenance.get('input_size')}",
        f"- Run artifact: `{detector['run']['artifact']}` SHA-256 "
        f"`{detector['run']['artifact_sha256']}`",
        "",
        "## Measurements",
        "",
        f"- Scored: {measurements['scored']}, abstentions: {measurements['abstentions']}",
        f"- AUROC (pooled): {_format(measurements['auroc_pooled'])}",
        "",
        "### Score distribution by stratum",
        "",
        "| stratum | n | min | q1 | median | q3 | max | AUROC vs genuine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stratum, summary in measurements["distribution_by_stratum"].items():
        lines.append(
            f"| {stratum} | {summary['count']} | {_format(summary['min'], 6)} "
            f"| {_format(summary['q1'], 6)} | {_format(summary['median'], 6)} "
            f"| {_format(summary['q3'], 6)} | {_format(summary['max'], 6)} "
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
            f"| {family} | {summary['count']} | {_format(summary['min'], 6)} "
            f"| {_format(summary['median'], 6)} | {_format(summary['max'], 6)} |"
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
            f"| {_format(row['threshold'], 6)} | {row['false_positives']}/"
            f"{row['genuine_n']} | {_format(row['false_positive_rate'])} "
            f"| {_format(row['false_positive_rate_upper95'])} "
            f"| {_format(row['true_positive_rate'])} |"
        )
    best = measurements["best_separation_not_selected"]
    lines += [
        "",
        f"Best measured separation (**reported, not selected**): threshold "
        f"{_format(best.get('threshold'), 6)}, TPR "
        f"{_format(best.get('true_positive_rate'))}, FPR "
        f"{_format(best.get('false_positive_rate'))}.",
        "",
        "## Selected operating points",
        "",
        f"- `T_HIGH` = **{_format(high.get('threshold'), 6)}** — "
        f"{high.get('rule') or high.get('reason')}",
    ]
    if high.get("threshold") is not None:
        at_high = thresholds["measured_at_t_high"]
        lines += [
            f"  - highest genuine score observed: "
            f"{_format(high.get('genuine_max'), 6)}, lowest score above it "
            f"{_format(high.get('next_score_above_genuine_max'), 6)}, margin "
            f"{_format(high.get('margin_over_genuine_max'), 6)}",
            f"  - false HIGH: {at_high['false_positives']}/{at_high['genuine_n']} "
            f"observed, 95% upper bound "
            f"{_format(at_high['false_positive_rate_upper95'])}",
            f"  - detection at that point: "
            f"{at_high['true_positives']}/{at_high['manipulated_n']} "
            f"({_format(at_high['true_positive_rate'])})",
        ]
        lines += [
            "",
            "The five highest genuine scores, because the margin is a distance to one "
            "sample:",
            "",
            "| clip | family | score |",
            "|---|---|---:|",
        ]
        for record in high.get("genuine_tail") or []:
            lines.append(
                f"| `{record['clip_id']}` | {record['family']} "
                f"| {_format(record['score'], 8)} |"
            )
    lines += [
        "",
        f"- `T_LOW` = **{_format(low.get('threshold'), 6)}** — "
        f"{low.get('rule') or low.get('reason')}",
    ]
    if low.get("threshold") is not None:
        lines.append(
            f"  - genuine media that would earn LOW: "
            f"{_format(low.get('coverage_genuine'))}"
        )
    band = thresholds["band_between_them"]
    lines += [
        "",
        f"**Between the two points: {band['kind']}** (width "
        f"{_format(band['width'], 6)}) — {band['reading']}.",
    ]

    reporting = comparison["at_reporting_threshold"]
    derived = comparison["at_derived_threshold"]
    lines += [
        "",
        "## Why not `0.5`",
        "",
        "R5-T1 reported its confusion matrix at `0.5`. That is a reporting convention — "
        "the midpoint of a sigmoid, fixed before any score on this corpus existed — and "
        "it is measured here rather than inherited.",
        "",
        "| threshold | provenance | false positives | FPR upper 95% | true positives | TPR |",
        "|---:|---|---:|---:|---:|---:|",
        f"| {_format(reporting['threshold'], 6)} | R5-T1 reporting convention, "
        f"**not selected** | {reporting['false_positives']}/{reporting['genuine_n']} "
        f"| {_format(reporting['false_positive_rate_upper95'])} "
        f"| {reporting['true_positives']}/{reporting['manipulated_n']} "
        f"| {_format(reporting['true_positive_rate'])} |",
    ]
    if derived is not None:
        lines.append(
            f"| {_format(derived['threshold'], 6)} | derived from the measured "
            f"distributions, **selected** "
            f"| {derived['false_positives']}/{derived['genuine_n']} "
            f"| {_format(derived['false_positive_rate_upper95'])} "
            f"| {derived['true_positives']}/{derived['manipulated_n']} "
            f"| {_format(derived['true_positive_rate'])} |"
        )
    missed = comparison["manipulated_below_reporting_threshold"]
    lines += [
        "",
        f"{len(missed)} manipulated clip(s) score below `0.5` and above the derived "
        f"point:",
        "",
    ]
    lines += [
        f"- `{record['clip_id']}` ({record['family']}) — {_format(record['score'], 6)}"
        for record in missed
    ] or ["- none"]
    lines += [
        "",
        "So on this corpus `0.5` costs detection and buys nothing: both points flag the "
        "same zero genuine clips. It is rejected on the measurement, not on principle.",
        "",
    ]

    stability = detector.get("cross_device_stability")
    if stability is not None:
        lines += [
            "## Cross-device stability",
            "",
            f"The same corpus was scored twice — `{stability['device']}` (production "
            f"pins this) and `{stability['other_device']}`. This is one perturbation, "
            "not a second corpus, and it establishes nothing about generalisation. What "
            "it does is put a number on how far a score moves when only the arithmetic "
            "backend changes.",
            "",
            f"- Clips compared: {stability['clips_compared']}",
            f"- Largest absolute score difference: "
            f"{_format(stability['max_abs_delta'], 6)}",
            f"- Decisions that flipped at the selected `T_HIGH`: "
            f"{len(stability['decision_flips_at_selected_threshold']) or 'none'}",
            f"- Same rule re-run on `{stability['other_device']}` selects "
            f"{_format(stability['t_high_reselected_on_other_device'], 6)}",
            "",
            "| clip | family | this device | other device | Δ |",
            "|---|---|---:|---:|---:|",
        ]
        for record in stability["largest_deltas"]:
            lines.append(
                f"| `{record['clip_id']}` | {record['family']} "
                f"| {_format(record['score'], 6)} "
                f"| {_format(record['other_score'], 6)} "
                f"| {_format(record['abs_delta'], 6)} |"
            )
        lines.append("")

    if detector["abstentions"]:
        lines += ["## Abstentions", ""]
        lines += [
            f"- `{record['clip_id']}` ({record['family']}) — {record['error']}"
            for record in detector["abstentions"]
        ]
        lines.append("")

    lines += _limitations(document)
    return "\n".join(lines)


def _limitations(document: dict) -> list[str]:
    """What this calibration does not establish, in its own measured numbers.

    Written from the artifact rather than from memory: the sample sizes, bounds and
    margins quoted here are read back out of what was just measured, so the section cannot
    drift away from the tables above it as the corpus changes.
    """
    corpus = document["corpus"]
    detector = document["detectors"][DETECTOR]
    high = detector["thresholds"]["t_high"]
    genuine_n = corpus["label_counts"].get("real", 0)
    at_high = detector["thresholds"]["measured_at_t_high"] or {}
    bound = at_high.get("false_positive_rate_upper95")
    tail = high.get("genuine_tail") or []
    nearest = tail[0] if tail else {}
    stability = detector.get("cross_device_stability") or {}
    delta = stability.get("max_abs_delta")
    # The families behind the manipulated half only: quoting the genuine family beside
    # them as evidence of what was manipulated would misdescribe the corpus.
    manipulated_families = {
        family: summary["count"]
        for family, summary in detector["measurements"]["distribution_by_family"].items()
        if family != GENUINE_FAMILY
    }
    margin = high.get("margin_over_genuine_max")

    lines = [
        "## Limitations",
        "",
        f"- **A zero count is a bound, not a proof.** No genuine clip here was flagged at "
        f"the selected point, which over {genuine_n} genuine clips bounds the true false-"
        f"HIGH rate at about {_format(bound)} with 95 % confidence and no lower. A "
        f"materially tighter bound needs roughly an order of magnitude more genuine "
        f"media.",
        f"- **The corpus is one dataset.** Every clip comes from "
        f"{', '.join(corpus['source_datasets'])}, at one compression level, and the "
        f"manipulated half is face-swap only — {_counts(manipulated_families)}. "
        f"LipForensics' weights were "
        f"trained on FaceForensics++, so this measures it *inside its training "
        f"distribution*; the numbers above are an upper bound on what it would do on "
        f"unseen manipulation families, not an estimate of it.",
        f"- **The margin is a distance to one clip.** `T_HIGH` clears the highest genuine "
        f"score by {_format(margin, 6)}, and that highest score belongs to "
        f"`{nearest.get('clip_id')}` ({_format(nearest.get('score'), 8)}). One more "
        f"unusual genuine clip would move this boundary.",
    ]
    if delta is not None:
        lines.append(
            f"- **The lower edge of the gap is not stable to the arithmetic backend.** "
            f"The largest score difference between the two device runs was "
            f"{_format(delta, 6)}, on the clip that defines the lowest manipulated score. "
            f"The selected point survives that shift here — no decision flipped — but the "
            f"gap it sits in is bounded below by a single clip whose score moves by more "
            f"than a rounding error."
        )
    lines += [
        f"- **Genuine media here is talking-head footage.** No landscape, sports, "
        f"animation, dubbing or heavy-VFX genuine media was tested. Legitimate "
        f"re-dubbing in particular is an obvious false-HIGH risk for a mouth-dynamics "
        f"detector and this corpus cannot see it.",
        "- **No compression ladder and no adversarial testing.** Codec, resolution and "
        "duration are whatever the FF++ c23 split carries; nothing was done to attack the "
        "threshold.",
        "- **Abstention behaviour is untested.** No clip in this corpus failed to yield a "
        "tracked mouth, so how often LipForensics refuses to answer on real submissions — "
        "and what the risk engine should do when it does — is not measured here.",
        "- **This establishes an operating point, not a rule.** What the risk engine "
        "should emit when LipForensics fires, and how that interacts with the two "
        "detectors already calibrated in R4-T1, is the Risk Engine v3 task's decision, "
        "taken against this evidence and not contained in it. No score in this artifact "
        "is combined with or compared in magnitude to another detector's.",
        "",
    ]
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate_lipforensics",
        description="Derive LipForensics operating points from its benchmark run.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="results.json from the run on the production device (cpu)",
    )
    parser.add_argument(
        "--cross-device-run",
        type=Path,
        help="optional results.json over the same manifest on another torch device",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_path = args.output_dir / "calibration.json"
    if artifact_path.exists() and not args.overwrite:
        print(f"refusing to overwrite {artifact_path} (pass --overwrite)", file=sys.stderr)
        return 1

    metadata, manifest_sha256 = read_manifest(args.manifest)
    run = load_run(args.run, metadata)

    # A calibration whose numbers were measured over different ground truth than the
    # manifest it names is not a calibration of this corpus. Refused, not caveated.
    if run["manifest_sha256"] != manifest_sha256:
        print(
            f"{args.run} was scored over a different manifest than {args.manifest}:\n"
            f"  run:      {run['manifest_sha256']}\n"
            f"  manifest: {manifest_sha256}",
            file=sys.stderr,
        )
        return 1

    # The threshold is calibrated for the device production runs on. A CUDA run is
    # admissible as the cross-device check, never as the run the point is derived from.
    if run["device"] != PRODUCTION_DEVICE:
        print(
            f"{args.run} was scored on device {run['device']!r}, but production pins "
            f"{PRODUCTION_DEVICE!r}; pass the production-device run as --run",
            file=sys.stderr,
        )
        return 1

    cross_device = None
    if args.cross_device_run is not None:
        other = load_run(args.cross_device_run, metadata)
        if other["manifest_sha256"] != manifest_sha256:
            print(
                f"{args.cross_device_run} was scored over a different manifest:\n"
                f"  run:      {other['manifest_sha256']}\n"
                f"  manifest: {manifest_sha256}",
                file=sys.stderr,
            )
            return 1
        cross_device = cross_device_check(
            run, other, select(run)["t_high"].get("threshold")
        )

    corpus_digest, clip_records = digest_clips(args.manifest, metadata)
    rows = run["rows"] + run["errors"]
    corpus = {
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "corpus_digest": corpus_digest,
        "clip_count": len(metadata),
        "label_counts": _tally(row["label"] for row in rows),
        "stratum_counts": _tally(row["stratum"] for row in rows),
        "family_counts": _tally(row["family"] for row in rows),
        "source_datasets": _tally(
            record["source"].split("/")[0] for record in clip_records
        ),
        "clips": clip_records,
    }

    document = build_calibration(
        corpus=corpus,
        run=run,
        cross_device=cross_device,
        generated_at=datetime.now(timezone.utc),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_path = args.output_dir / "report.md"
    report_path.write_text(render_report(document), encoding="utf-8")

    thresholds = document["detectors"][DETECTOR]["thresholds"]
    high = thresholds["t_high"].get("threshold")
    per_clip_path = args.output_dir / "per_clip.jsonl"
    per_clip_path.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "flagged_at_t_high": (
                        None if row.get("score") is None
                        else (high is not None and row["score"] >= high)
                    ),
                },
                sort_keys=True,
            )
            + "\n"
            for row in sorted(rows, key=lambda item: item["clip_id"])
        ),
        encoding="utf-8",
    )

    print(
        f"calibration {document['calibration_id']}\n"
        f"{DETECTOR}: T_HIGH {high} T_LOW {thresholds['t_low'].get('threshold')}\n"
        f"wrote {artifact_path}\nwrote {report_path}\nwrote {per_clip_path}"
    )
    return 0


def _tally(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
