"""Run a detector over a labelled corpus and write the evaluation artifacts.

    python3 scripts/benchmark/cli.py \
        --manifest corpus/manifest.csv \
        --model mock \
        --output-dir runs/2026-09-02-mock

Offline by construction. This entrypoint imports `dataset` and `metrics` and nothing
else from the repository: no FastAPI app, no SQLAlchemy session, no worker, no storage
client. A model under evaluation has not been adopted yet, and the harness that decides
whether it should be must not be able to touch what is already in production.

**Models.** `--model` is either `mock` (below) or a dotted reference
`package.module:attribute` to a callable taking one `dataset.Clip` and returning a
float score, where higher means *more likely manipulated*. That is the entire contract
— no base class, no registry, no factory. When R3 brings a real face-manipulation
detector it supplies one function, and the framework is unchanged.

**What a model may say about itself.** Optionally, one more function: a module defining
`provenance()` has it called after scoring and its dict recorded in `results.json`. The
dotted reference names code, not bytes, and a run whose weights or provider deployment
cannot be named afterwards cannot be the basis of a threshold (R4-T1).

**What a per-clip failure does.** It is recorded and the run continues. A model that
crashes on three clips out of two hundred still yields a measurement over the other
hundred and ninety-seven, and the artifact says plainly how many were excluded and why.
Failed clips never enter the confusion matrix: counting a crash as a wrong answer would
quietly turn a broken decoder into a detector's accuracy problem.

**Reproducibility.** Same manifest, same model, same threshold gives the same metrics
and the same per-clip records in the same order. The artifacts additionally carry the
manifest's SHA-256, the threshold, and the interpreter/platform, so a number can be
traced to the ground truth and the machine it came from. Wall-clock timestamps and the
latency figures are the parts that legitimately differ between two runs.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark import metrics
from benchmark.dataset import (
    Clip,
    Dataset,
    ManifestError,
    label_counts,
    load_manifest,
)

# Bumped when the shape of `results.json` changes, so a later reader can tell whether it
# understands an artifact it did not produce.
SCHEMA_VERSION = "r2-benchmark-2"

Model = Callable[[Clip], float]


def mock_model(clip: Clip) -> float:
    """A stand-in detector that carries no forensic signal whatsoever.

    Its score is the first four bytes of `sha256(clip_id)` mapped onto `[0, 1)`: a
    fixed, evenly spread, label-blind number. It exists to exercise the pipeline —
    ingestion, timing, metrics, artifacts — and being deterministic means the harness
    can be verified end to end without a real model and without a random seed to
    remember. Its accuracy is meaningless by design and should be read as such.
    """
    digest = hashlib.sha256(clip.clip_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def model_provenance(reference: str) -> dict | None:
    """Whatever the model's module says identifies it, or `None` if it says nothing.

    A model module may define a zero-argument `provenance()` returning a JSON-serializable
    dict. It is called *after* scoring, so a wrapper can report what actually answered —
    the provider deployment that served the calls, the digests of the weights it loaded —
    rather than what its configuration hoped for. `--model` names a dotted reference and
    nothing else, and that string does not pin bytes: R4 binds thresholds to a specific
    detector build, and a threshold whose model cannot be named is not traceable.

    Optional, because `mock` and any one-function detector have nothing to declare. A
    failure to produce it is recorded rather than raised: it arrives at the end of a run
    that may have taken half an hour on a paid provider, and losing those measurements to
    a broken metadata call would be the more expensive mistake.
    """
    if ":" not in reference:
        return None
    module_name, _, _ = reference.partition(":")
    describe = getattr(sys.modules.get(module_name), "provenance", None)
    if not callable(describe):
        return None
    try:
        return describe()
    except Exception as error:  # noqa: BLE001 - metadata must never lose a run
        return {"error": f"{type(error).__name__}: {error}"}


def resolve_model(reference: str) -> Model:
    """Turn `--model` into a callable, or raise `ValueError` explaining why not."""
    if reference == "mock":
        return mock_model
    if ":" not in reference:
        raise ValueError(
            f"unknown model {reference!r}: expected 'mock' or "
            f"'package.module:attribute'"
        )
    module_name, _, attribute = reference.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"cannot import module {module_name!r}: {error}") from error
    try:
        candidate = getattr(module, attribute)
    except AttributeError as error:
        raise ValueError(
            f"module {module_name!r} has no attribute {attribute!r}"
        ) from error
    if not callable(candidate):
        raise ValueError(f"{reference!r} is not callable")
    return candidate


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MiB.

    `ru_maxrss` is a high-water mark the kernel maintains, which is what makes it the
    right instrument here: a model's memory cost is its worst moment, not its average,
    and unlike `tracemalloc` this counts the native allocations that ONNX Runtime and
    torch actually make — where a detector's memory in fact goes. Linux reports it in
    kibibytes; on macOS the same field is bytes, and this framework targets the Linux
    hosts the product runs on.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_benchmark(
    clips: list[Clip], model: Model, threshold: float
) -> tuple[list[dict], list[float]]:
    """Score every clip, timing each call. Returns per-clip records and latencies.

    Only successful calls contribute a latency: timing a traceback would report how
    fast the model failed as though it were how fast it works.
    """
    records: list[dict] = []
    latencies_ms: list[float] = []

    for clip in clips:
        started = time.perf_counter()
        try:
            raw_score = model(clip)
            score = float(raw_score)
            if not math.isfinite(score):
                raise ValueError(f"model returned a non-finite score: {raw_score!r}")
        except Exception as error:  # noqa: BLE001 - a model may fail any way it likes
            records.append(
                {
                    "clip_id": clip.clip_id,
                    "label": clip.label,
                    "is_manipulated": clip.is_manipulated,
                    "status": "error",
                    "score": None,
                    "predicted_manipulated": None,
                    "correct": None,
                    "latency_ms": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)
        predicted = metrics.predict(score, threshold)
        records.append(
            {
                "clip_id": clip.clip_id,
                "label": clip.label,
                "is_manipulated": clip.is_manipulated,
                "status": "ok",
                "score": score,
                "predicted_manipulated": predicted,
                "correct": predicted == clip.is_manipulated,
                "latency_ms": latency_ms,
                "error": None,
            }
        )

    return records, latencies_ms


def per_label_breakdown(records: list[dict]) -> dict[str, dict]:
    """Flag rate within each ground-truth label.

    For a manipulation family this is recall on that family; for `real` it is the false
    positive rate. Kept separate from the headline numbers because a corpus is never
    balanced across families, and an aggregate hides exactly the weakness that decides
    whether a detector is worth adopting.
    """
    breakdown: dict[str, dict] = {}
    for record in records:
        entry = breakdown.setdefault(
            record["label"],
            {"total": 0, "scored": 0, "errors": 0, "flagged": 0, "flagged_rate": None},
        )
        entry["total"] += 1
        if record["status"] != "ok":
            entry["errors"] += 1
            continue
        entry["scored"] += 1
        if record["predicted_manipulated"]:
            entry["flagged"] += 1
    for entry in breakdown.values():
        if entry["scored"]:
            entry["flagged_rate"] = entry["flagged"] / entry["scored"]
    return dict(sorted(breakdown.items()))


def build_results(
    *,
    dataset: Dataset,
    model_reference: str,
    provenance: dict | None,
    threshold: float,
    records: list[dict],
    latencies_ms: list[float],
    baseline_rss_mb: float,
    peak_mb: float,
    started_at: datetime,
    finished_at: datetime,
) -> dict:
    """Assemble the `results.json` document."""
    clips = dataset.clips
    scored = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    matrix = metrics.confusion_matrix(
        [(r["is_manipulated"], r["predicted_manipulated"]) for r in scored]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "model": model_reference,
            "model_provenance": provenance,
            "threshold": threshold,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": (finished_at - started_at).total_seconds(),
        },
        "dataset": {
            "manifest": str(dataset.manifest_path),
            # Captured when the manifest was read, not recomputed here: this digest
            # names the bytes these very records came from.
            "manifest_sha256": dataset.manifest_sha256,
            "clip_count": len(clips),
            "label_counts": label_counts(clips),
            "genuine": sum(1 for c in clips if not c.is_manipulated),
            "manipulated": sum(1 for c in clips if c.is_manipulated),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "counts": {
            "scored": len(scored),
            "errors": len(failed),
        },
        "confusion_matrix": matrix.as_dict(),
        "metrics": metrics.classification_metrics(matrix),
        "per_label": per_label_breakdown(records),
        "performance": {
            "latency": metrics.latency_summary(latencies_ms),
            "memory": {
                "baseline_rss_mb": baseline_rss_mb,
                "peak_rss_mb": peak_mb,
                "delta_rss_mb": peak_mb - baseline_rss_mb,
            },
        },
        "clips": records,
    }


def _format(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(results: dict) -> str:
    """A short markdown summary of a run, for reading and for pasting into a review."""
    run = results["run"]
    dataset = results["dataset"]
    matrix = results["confusion_matrix"]
    scores = results["metrics"]
    latency = results["performance"]["latency"]
    memory = results["performance"]["memory"]

    lines = [
        "# Detector benchmark run",
        "",
        f"- Model: `{run['model']}`",
        f"- Model provenance: `{json.dumps(run.get('model_provenance'))}`",
        f"- Threshold: `{run['threshold']}` (score >= threshold means manipulated)",
        f"- Manifest: `{dataset['manifest']}`",
        f"- Manifest SHA-256: `{dataset['manifest_sha256']}`",
        f"- Clips: {dataset['clip_count']} "
        f"({dataset['genuine']} genuine / {dataset['manipulated']} manipulated)",
        f"- Scored: {results['counts']['scored']}, "
        f"errors: {results['counts']['errors']}",
        f"- Finished: {run['finished_at']}",
        f"- Python {results['environment']['python']} on "
        f"{results['environment']['platform']}",
        "",
        "## Classification",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Accuracy | {_format(scores['accuracy'])} |",
        f"| Precision | {_format(scores['precision'])} |",
        f"| Recall | {_format(scores['recall'])} |",
        f"| False positive rate | {_format(scores['false_positive_rate'])} |",
        f"| False negative rate | {_format(scores['false_negative_rate'])} |",
        "",
        "| | predicted manipulated | predicted genuine |",
        "|---|---|---|",
        f"| **actually manipulated** | {matrix['true_positives']} "
        f"| {matrix['false_negatives']} |",
        f"| **actually genuine** | {matrix['false_positives']} "
        f"| {matrix['true_negatives']} |",
        "",
        "## Per label",
        "",
        "| label | clips | scored | flagged | flagged rate | errors |",
        "|---|---|---|---|---|---|",
    ]
    for label, entry in results["per_label"].items():
        lines.append(
            f"| {label} | {entry['total']} | {entry['scored']} | {entry['flagged']} "
            f"| {_format(entry['flagged_rate'])} | {entry['errors']} |"
        )
    lines += [
        "",
        "## Performance",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Mean latency | {_format(latency['mean_ms'], 2)} ms |",
        f"| Median latency | {_format(latency['median_ms'], 2)} ms |",
        f"| p95 latency | {_format(latency['p95_ms'], 2)} ms |",
        f"| Max latency | {_format(latency['max_ms'], 2)} ms |",
        f"| Peak RSS | {memory['peak_rss_mb']:.1f} MiB |",
        f"| RSS growth during run | {memory['delta_rss_mb']:.1f} MiB |",
        "",
    ]
    errors = [r for r in results["clips"] if r["status"] != "ok"]
    if errors:
        lines += ["## Errors", ""]
        lines += [f"- `{r['clip_id']}` — {r['error']}" for r in errors]
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Evaluate a detector against a labelled corpus, offline.",
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="ground-truth CSV manifest"
    )
    parser.add_argument(
        "--model",
        default="mock",
        help="'mock', or 'package.module:attribute' naming a callable "
        "that takes a Clip and returns a score",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for results.json and report.md",
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
        help="score at or above which a clip counts as manipulated (default: 0.5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing results.json in the output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    results_path = args.output_dir / "results.json"
    if results_path.exists() and not args.overwrite:
        print(
            f"refusing to overwrite {results_path} (pass --overwrite)",
            file=sys.stderr,
        )
        return 1

    try:
        dataset = load_manifest(args.manifest)
    except ManifestError as error:
        print(error, file=sys.stderr)
        return 1

    try:
        model = resolve_model(args.model)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"{len(dataset.clips)} clip(s) from {args.manifest}, model {args.model}")

    baseline_rss_mb = peak_rss_mb()
    started_at = datetime.now(timezone.utc)
    records, latencies_ms = run_benchmark(dataset.clips, model, args.threshold)
    finished_at = datetime.now(timezone.utc)
    peak_mb = peak_rss_mb()

    results = build_results(
        dataset=dataset,
        model_reference=args.model,
        provenance=model_provenance(args.model),
        threshold=args.threshold,
        records=records,
        latencies_ms=latencies_ms,
        baseline_rss_mb=baseline_rss_mb,
        peak_mb=peak_mb,
        started_at=started_at,
        finished_at=finished_at,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report_path = args.output_dir / "report.md"
    report_path.write_text(render_report(results), encoding="utf-8")

    scores = results["metrics"]
    print(
        f"scored {results['counts']['scored']}, errors {results['counts']['errors']}\n"
        f"accuracy {_format(scores['accuracy'])}  "
        f"precision {_format(scores['precision'])}  "
        f"recall {_format(scores['recall'])}\n"
        f"FPR {_format(scores['false_positive_rate'])}  "
        f"FNR {_format(scores['false_negative_rate'])}\n"
        f"mean latency {_format(results['performance']['latency']['mean_ms'], 2)} ms  "
        f"peak RSS {peak_mb:.1f} MiB\n"
        f"wrote {results_path}\nwrote {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
