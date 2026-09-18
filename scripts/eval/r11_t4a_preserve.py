#!/usr/bin/env python3
"""R11-T4A: raw output log collection and data preservation.

Freezes the authoritative R11-T3A artifact set. This script performs no inference: it
reads the frozen artifacts, cross-reconciles the JSON against the CSV, copies the
authoritative bytes into the durable corpus, and writes a metadata-only manifest.

Invariants enforced here:
  * inference is never run, and `r11_t3a_baseline.py` is never imported or executed;
  * the canonical artifacts are never re-serialized — preservation copies are byte-for-byte
    and verified by SHA-256 after the copy;
  * the manifest carries hashes, counts and lineage only, never raw logits;
  * any reconciliation mismatch halts immediately, with no repair and no rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "deepguard/r11-t4a-preservation-manifest/1"
TASK = "R11-T4A"

EXPECTED_RECORD_COUNT = 12
EXPECTED_WINDOW_TOTAL = 3466

CSV_COLUMNS = (
    "recording_id",
    "panel",
    "person",
    "derivative_relative_path",
    "derivative_sha256",
    "window_index",
    "start_sample",
    "end_sample",
    "padded_samples",
    "logit_0",
    "logit_1",
    "bona_fide_logit",
)

# Facts observed in R11-T3A and carried through verbatim. Not re-derived here.
REC02_L_P04_LINEAGE = {
    "derivative_relative_path": "wav/rec02__L_p04.wav",
    "observed_in": "R11-T3A",
    "windows_returned": 255,
    "counterpart": "rec02__R_p06",
    "counterpart_windows_returned": 255,
    "observed_facts": {
        "total_samples": 16414264,
        "counterpart_total_samples": 16446880,
        "final_window_padded_samples": 58736,
        "counterpart_final_window_padded_samples": 26120,
        "window_padding_scheme": "repeat-tile",
    },
    "summary": (
        "Shorter decoded duration than its rec02 counterpart. The shortfall is smaller than one "
        "window, so it lands entirely inside the final window and surfaces as increased "
        "padded_samples there. The window count is unchanged at 255."
    ),
    "note": (
        "Recorded as lineage metadata only, carried verbatim from R11-T3A observables. "
        "R11-T4A did not re-analyse, re-measure or reinterpret this record."
    ),
}


class PreservationHalt(RuntimeError):
    """Raised on any integrity failure. Execution stops; nothing is repaired or rerun."""


# --- Hashing ---------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_entry(path: Path, root: Path) -> dict:
    if not path.is_file():
        raise PreservationHalt(f"authoritative artifact missing: {path}")
    try:
        relative = str(path.relative_to(root))
    except ValueError:
        # Artifact lives outside `root` (an overridden --source-dir, say). Name it plainly
        # rather than inventing a relative path that would not resolve.
        relative = path.name
    return {
        "relative_path": relative,
        "absolute_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


# --- Cross-reconciliation --------------------------------------------------------------------


def window_key(recording_id: str, panel: str, person: str, window_index: int) -> tuple:
    return (recording_id, panel, person, window_index)


def load_json_windows(report: dict) -> tuple[dict, dict]:
    """Flatten the JSON report into {window_key: row} plus {file_key: window count}."""
    rows: dict[tuple, dict] = {}
    per_file: dict[tuple, int] = {}
    duplicates: list[tuple] = []

    for record in report["records"]:
        file_key = (record["recording_id"], record["panel"], record["person"])
        seen_indices = Counter()
        for window in record["windows"]:
            key = window_key(*file_key, window["window_index"])
            if key in rows:
                duplicates.append(key)
            seen_indices[window["window_index"]] += 1
            rows[key] = {
                "derivative_relative_path": record["derivative"]["relative_path"],
                "derivative_sha256": record["derivative"]["sha256_verified"],
                "start_sample": window["start_sample"],
                "end_sample": window["end_sample"],
                "padded_samples": window["padded_samples"],
                "logit_0": repr(window["logits"][0]),
                "logit_1": repr(window["logits"][1]),
                "bona_fide_logit": repr(window["bona_fide_logit"]),
            }
        per_file[file_key] = len(record["windows"])

        expected = list(range(len(record["windows"])))
        actual = [window["window_index"] for window in record["windows"]]
        if actual != expected:
            raise PreservationHalt(
                f"JSON window_index sequence broken for {file_key}: "
                f"expected 0..{len(expected) - 1} in order, got {actual[:5]}..."
            )

    if duplicates:
        raise PreservationHalt(f"duplicate windows in JSON: {duplicates[:5]}")
    return rows, per_file


def load_csv_windows(csv_bytes: bytes) -> tuple[dict, dict, list]:
    """Flatten the CSV into {window_key: row}, {file_key: count}, and the raw key order."""
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8"), newline=""))
    header = next(reader)
    if tuple(header) != CSV_COLUMNS:
        raise PreservationHalt(f"unexpected CSV header: {header}")

    rows: dict[tuple, dict] = {}
    per_file: Counter = Counter()
    order: list[tuple] = []
    duplicates: list[tuple] = []

    for line_number, row in enumerate(reader, start=2):
        if len(row) != len(CSV_COLUMNS):
            raise PreservationHalt(f"malformed CSV row at line {line_number}: {len(row)} fields")
        field = dict(zip(CSV_COLUMNS, row))
        file_key = (field["recording_id"], field["panel"], field["person"])
        key = window_key(*file_key, int(field["window_index"]))
        if key in rows:
            duplicates.append(key)
        rows[key] = {
            "derivative_relative_path": field["derivative_relative_path"],
            "derivative_sha256": field["derivative_sha256"],
            "start_sample": int(field["start_sample"]),
            "end_sample": int(field["end_sample"]),
            "padded_samples": int(field["padded_samples"]),
            "logit_0": field["logit_0"],
            "logit_1": field["logit_1"],
            "bona_fide_logit": field["bona_fide_logit"],
        }
        per_file[file_key] += 1
        order.append(key)

    if duplicates:
        raise PreservationHalt(f"duplicate windows in CSV: {duplicates[:5]}")
    return rows, dict(per_file), order


def reconcile(report: dict, csv_bytes: bytes) -> dict:
    json_rows, json_per_file = load_json_windows(report)
    csv_rows, csv_per_file, csv_order = load_csv_windows(csv_bytes)

    declared = report["record_count"]
    if declared != EXPECTED_RECORD_COUNT:
        raise PreservationHalt(
            f"JSON declares record_count={declared}, expected {EXPECTED_RECORD_COUNT}"
        )
    if len(report["records"]) != EXPECTED_RECORD_COUNT:
        raise PreservationHalt(
            f"JSON holds {len(report['records'])} records, declares {declared}"
        )

    if set(json_per_file) != set(csv_per_file):
        only_json = sorted(set(json_per_file) - set(csv_per_file))
        only_csv = sorted(set(csv_per_file) - set(json_per_file))
        raise PreservationHalt(
            f"orphan files — JSON only: {only_json}, CSV only: {only_csv}"
        )
    if len(csv_per_file) != EXPECTED_RECORD_COUNT:
        raise PreservationHalt(
            f"CSV holds {len(csv_per_file)} files, expected {EXPECTED_RECORD_COUNT}"
        )

    for file_key in sorted(json_per_file):
        if json_per_file[file_key] != csv_per_file[file_key]:
            raise PreservationHalt(
                f"window count mismatch for {file_key}: "
                f"JSON {json_per_file[file_key]}, CSV {csv_per_file[file_key]}"
            )

    json_total = sum(json_per_file.values())
    csv_total = sum(csv_per_file.values())
    if json_total != EXPECTED_WINDOW_TOTAL or csv_total != EXPECTED_WINDOW_TOTAL:
        raise PreservationHalt(
            f"window total mismatch: JSON {json_total}, CSV {csv_total}, "
            f"expected {EXPECTED_WINDOW_TOTAL}"
        )

    orphan_json = sorted(set(json_rows) - set(csv_rows))
    orphan_csv = sorted(set(csv_rows) - set(json_rows))
    if orphan_json or orphan_csv:
        raise PreservationHalt(
            f"orphan windows — JSON only: {orphan_json[:5]}, CSV only: {orphan_csv[:5]}"
        )

    # CSV row order must reproduce the JSON record-then-window traversal exactly.
    json_order = [
        window_key(record["recording_id"], record["panel"], record["person"], window["window_index"])
        for record in report["records"]
        for window in record["windows"]
    ]
    if json_order != csv_order:
        first = next(
            index for index, (a, b) in enumerate(zip(json_order, csv_order)) if a != b
        )
        raise PreservationHalt(
            f"window order diverges at position {first}: "
            f"JSON {json_order[first]}, CSV {csv_order[first]}"
        )

    mismatches: list[str] = []
    for key in json_order:
        left, right = json_rows[key], csv_rows[key]
        for column in sorted(left):
            if left[column] != right[column]:
                mismatches.append(f"{key} {column}: JSON {left[column]!r}, CSV {right[column]!r}")
                if len(mismatches) >= 5:
                    break
        if len(mismatches) >= 5:
            break
    if mismatches:
        raise PreservationHalt("JSON/CSV field mismatch:\n  " + "\n  ".join(mismatches))

    return {
        "record_count": EXPECTED_RECORD_COUNT,
        "window_total": EXPECTED_WINDOW_TOTAL,
        "duplicate_windows": 0,
        "orphan_windows": 0,
        "window_order_matches": True,
        "logits_match_exactly": True,
        "logit_comparison": "repr() string equality on logit_0, logit_1 and bona_fide_logit",
        "fields_compared": [
            "derivative_relative_path",
            "derivative_sha256",
            "start_sample",
            "end_sample",
            "padded_samples",
            "logit_0",
            "logit_1",
            "bona_fide_logit",
        ],
        "per_file_window_counts": {
            f"{file_key[0]}__{file_key[1]}_{file_key[2]}": json_per_file[file_key]
            for file_key in sorted(json_per_file)
        },
        "passed": True,
    }


# --- Preservation ----------------------------------------------------------------------------


def preserve(entries: list[dict], destination: Path) -> list[dict]:
    """Byte-for-byte copy, verified by re-hashing the destination. No re-serialization."""
    destination.mkdir(parents=True, exist_ok=True)
    preserved: list[dict] = []

    for entry in entries:
        source = Path(entry["absolute_path"])
        target = destination / source.name
        shutil.copyfile(source, target)

        target_sha = sha256_file(target)
        target_size = target.stat().st_size
        if target_sha != entry["sha256"] or target_size != entry["size_bytes"]:
            raise PreservationHalt(
                f"preservation copy is not byte-identical for {source.name}: "
                f"source {entry['sha256']}/{entry['size_bytes']}, "
                f"copy {target_sha}/{target_size}"
            )

        preserved.append(
            {
                "filename": source.name,
                "source_path": entry["absolute_path"],
                "source_relative_path": entry["relative_path"],
                "preserved_path": str(target),
                "sha256": target_sha,
                "size_bytes": target_size,
                "byte_identical": True,
            }
        )
    return preserved


# --- Manifest --------------------------------------------------------------------------------


def build_manifest(
    *,
    report: dict,
    determinism: dict,
    reconciliation: dict,
    preserved: list[dict],
    upstream: dict,
    destination: Path,
    source_dir: Path,
) -> dict:
    runtime = report["runtime"]
    model = runtime["model"]

    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": (
            "Metadata-only preservation index for the frozen R11-T3A AASIST raw-logit "
            "artifacts. Contains hashes, counts, provenance and lineage. No logits, no "
            "classification, no calibration, no verdicts."
        ),
        "contains_raw_logits": False,
        "inference_performed_by_this_task": False,
        "artifacts_re_serialized": False,
        "preservation_root": str(destination),
        "authoritative_source": {
            "task": "R11-T3A",
            "directory": str(source_dir),
        },
        "preserved_artifacts": preserved,
        "counts": {
            "record_count": reconciliation["record_count"],
            "window_total": reconciliation["window_total"],
            "per_file_window_counts": reconciliation["per_file_window_counts"],
        },
        "cross_reconciliation": {
            key: value
            for key, value in reconciliation.items()
            if key != "per_file_window_counts"
        },
        "upstream": upstream,
        "determinism": {
            "source": "R11-T3A r11_t3a_determinism.json, carried through verbatim",
            "method": determinism["method"],
            "inference_runs": determinism["inference_runs"],
            "json_sha256_pass_1": determinism["json_sha256_pass_1"],
            "json_sha256_pass_2": determinism["json_sha256_pass_2"],
            "json_byte_identical": determinism["json_byte_identical"],
            "csv_sha256_pass_1": determinism["csv_sha256_pass_1"],
            "csv_sha256_pass_2": determinism["csv_sha256_pass_2"],
            "csv_byte_identical": determinism["csv_byte_identical"],
            "schema_version": determinism["schema_version"],
        },
        "runtime": {
            "inference_runtime": runtime["inference_runtime"],
            "onnxruntime_version": runtime.get("onnxruntime_version"),
            "execution_provider_used": runtime["execution_provider_used"],
            "execution_providers_available": runtime["execution_providers_available"],
            "device": runtime["device"],
            "cuda_used": runtime["cuda"]["used"],
            "detector_source": runtime["detector_source"],
            "numpy_version": runtime.get("numpy_version"),
            "torch": runtime.get("torch"),
            "python_version": runtime.get("python_version"),
            "platform": runtime.get("platform"),
        },
        "model": {
            "repository": model["repository"],
            "revision": model["revision"],
            "filename": model["filename"],
            "path": model["path"],
            "checkpoint_sha256": model["sha256_on_disk"],
            "checkpoint_sha256_expected": model["sha256_expected"],
            "checkpoint_sha256_matches_expected": (
                model["sha256_on_disk"] == model["sha256_expected"]
            ),
            "window_samples_expected": model["window_samples_expected"],
            "sample_rate": model["sample_rate"],
            "channels": model["channels"],
        },
        "dataset": report["dataset"],
        "source_schema_version": report["schema_version"],
        "semantic_flags": report["semantic_flags"],
        "lineage": {
            "rec02__L_p04": REC02_L_P04_LINEAGE,
        },
    }


# --- Entrypoint ------------------------------------------------------------------------------


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="R11-T4A preservation (no inference).")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "docs/ai/reviews/R11_T3A",
        help="Directory holding the authoritative R11-T3A artifacts.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=repo_root / "scripts/eval/r11_t3a_baseline.py",
        help="R11-T3A runner script, hashed for provenance. Never executed.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("/home/adentechio/deepguard-corpus/external-genuine-real-turnturk"),
        help="Durable corpus root for the Real-TurnTurk dataset.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Durable preservation directory. Defaults to <corpus-root>/evaluation/r11_t3a.",
    )
    args = parser.parse_args()

    destination = args.destination or (args.corpus_root / "evaluation" / "r11_t3a")

    json_path = args.source_dir / "r11_t3a_aasist_baseline.json"
    csv_path = args.source_dir / "r11_t3a_aasist_windows.csv"
    determinism_path = args.source_dir / "r11_t3a_determinism.json"
    report_path = args.source_dir / "r11-t3a-baseline-execution-report.md"

    authoritative = [json_path, csv_path, determinism_path, report_path, args.runner]
    entries = [hashed_entry(path, repo_root) for path in authoritative]
    print(f"authoritative set: {len(entries)} files", flush=True)
    for entry in entries:
        print(f"  {entry['relative_path']}  {entry['sha256']}  {entry['size_bytes']}B", flush=True)

    # Read for validation only. The bytes on disk stay untouched and are never rewritten.
    report = json.loads(json_path.read_bytes().decode("utf-8"))
    csv_bytes = csv_path.read_bytes()
    determinism = json.loads(determinism_path.read_bytes().decode("utf-8"))

    json_sha, csv_sha = entries[0]["sha256"], entries[1]["sha256"]
    if determinism["json_sha256_pass_1"] != json_sha:
        raise PreservationHalt(
            f"JSON hash disagrees with R11-T3A determinism record: "
            f"on disk {json_sha}, recorded {determinism['json_sha256_pass_1']}"
        )
    if determinism["csv_sha256_pass_1"] != csv_sha:
        raise PreservationHalt(
            f"CSV hash disagrees with R11-T3A determinism record: "
            f"on disk {csv_sha}, recorded {determinism['csv_sha256_pass_1']}"
        )
    print("determinism hashes match the artifacts on disk", flush=True)

    reconciliation = reconcile(report, csv_bytes)
    print(
        f"cross-reconciliation passed: {reconciliation['record_count']} files, "
        f"{reconciliation['window_total']} windows, 0 duplicates, 0 orphans",
        flush=True,
    )

    preserved = preserve(entries, destination)
    print(f"preserved {len(preserved)} artifacts byte-identically to {destination}", flush=True)

    upstream_manifest_path = args.corpus_root / report["upstream_manifest"]["relative_path"]
    upstream = {
        "r11_t1": {
            "manifest_json": hashed_entry(args.corpus_root / "manifest.json", args.corpus_root),
            "manifest_csv": hashed_entry(args.corpus_root / "manifest.csv", args.corpus_root),
            "lineage_md": hashed_entry(args.corpus_root / "LINEAGE.md", args.corpus_root),
        },
        "r11_t2a": {
            "execution_manifest": hashed_entry(upstream_manifest_path, args.corpus_root),
            "recorded_in_r11_t3a": report["upstream_manifest"],
            "sha256_agrees_with_r11_t3a": (
                sha256_file(upstream_manifest_path) == report["upstream_manifest_sha256"]
            ),
        },
        "r11_t3a": {
            "baseline_json_sha256": json_sha,
            "windows_csv_sha256": csv_sha,
            "determinism_json_sha256": entries[2]["sha256"],
            "execution_report_sha256": entries[3]["sha256"],
            "runner_script": {
                "relative_path": entries[4]["relative_path"],
                "sha256": entries[4]["sha256"],
                "size_bytes": entries[4]["size_bytes"],
                "executed_by_this_task": False,
            },
        },
    }
    if not upstream["r11_t2a"]["sha256_agrees_with_r11_t3a"]:
        raise PreservationHalt(
            "R11-T2A execution manifest on disk does not match the hash recorded in R11-T3A"
        )

    manifest = build_manifest(
        report=report,
        determinism=determinism,
        reconciliation=reconciliation,
        preserved=preserved,
        upstream=upstream,
        destination=destination,
        source_dir=args.source_dir,
    )

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if b'"logits"' in manifest_bytes:
        raise PreservationHalt("manifest contains raw logits; it must be metadata only")

    for out_dir in (destination, args.source_dir.parent / "R11_T4A"):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "r11_t4a_manifest.json").write_bytes(manifest_bytes)
        print(
            f"manifest {out_dir / 'r11_t4a_manifest.json'} "
            f"sha256={hashlib.sha256(manifest_bytes).hexdigest()}",
            flush=True,
        )

    print("R11-T4A preservation complete. Inference was not run.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PreservationHalt as error:
        print(f"HALT: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
