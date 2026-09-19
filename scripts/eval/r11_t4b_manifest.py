#!/usr/bin/env python3
"""R11-T4B: deterministic visual execution manifest and hash freeze.

Prepares the 80 raw VCD1 `th` + `th-m` clips for a later deterministic SVD/B7 visual
regression run. This script runs no inference, decodes no video and extracts no frames:
it re-hashes the frozen MP4 bytes from disk, asserts an exact 80/80 match against the
R11-T3B intake manifest, and emits a strictly ordered execution manifest.

Invariants enforced here:
  * SVD, EfficientNet-B7 and every other detector are neither imported nor executed;
  * no pixels are touched — all stream metadata is carried verbatim from the R11-T3B
    `ffprobe` records, and ffprobe is not re-run;
  * any hash, count or field mismatch halts immediately, with no repair and no rerun;
  * the manifest carries no timestamp, host-transient field or random value, so building
    it twice yields byte-identical JSON;
  * the execution contract defers to the existing production path — this task invents no
    FPS, resize or clipping rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "deepguard/r11-t4b-execution-manifest/1"
TASK = "R11-T4B"

VCD1_REVISION = "e4c6598cb349bcf2f2cd5df55ef76cf58e947e77"
SELECTED_SUBSETS = ("th", "th-m")
EXPECTED_FILE_COUNT = 80
EXPECTED_SUBJECT_COUNT = 80

DEFAULT_CORPUS_ROOT = Path("/home/adentechio/deepguard-corpus/external-genuine-vcd1")
DEFAULT_INTAKE_MANIFEST = DEFAULT_CORPUS_ROOT / "vcd1_intake_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs/ai/reviews/R11_T4B"

# The production Fast Decision Path this regression will execute against. Named, not
# reimplemented: R11-T4B defines no preprocessing of its own.
EXECUTION_CONTRACT = {
    "path": "apps/api/app/worker.py :: Fast Decision Path (NVIDIA SVD + EfficientNet-B7)",
    "path_detail": {
        "contract_document": "docs/architecture/R10_EXECUTION_CONTRACT.md",
        "svd_entrypoint": "apps.api.app.detection.detect_synthetic_video",
        "b7_entrypoint": "apps.api.app.detection.detect_face_manipulation",
        "invoked_by_this_task": False,
    },
    "input_media": "original_frozen_mp4",
    "preprocessing": "existing_pipeline_contract",
    "preprocessing_note": (
        "Whatever the production path already does to an MP4 before SVD and B7 see it is "
        "the contract. R11-T4B specifies no FPS, resize, crop or clipping rule of its own."
    ),
}

# How the two identifier fields are derived from the R11-T3B intake record. Both are
# transcriptions, not new facts.
RECORD_FIELD_DERIVATIONS = {
    "object_id": "R11-T3B `upstream_object`, the VCD1 blob object key, verbatim",
    "file_id": "basename stem of `relative_path` (the VCD1 filename identifier)",
    "subject_id": "R11-T3B `subject_id`, verbatim",
    "intake_sha256": "R11-T3B `sha256_local_freeze`, re-verified against disk by this task",
    "container_codec": "R11-T3B `probe` container/codec fields, verbatim",
    "width_height": "R11-T3B `probe` width/height, verbatim",
    "fps": "R11-T3B `probe` fps, verbatim (ffprobe not re-run)",
    "duration_frame_count": "R11-T3B `probe` nb_frames/duration_sec, verbatim",
}


class ManifestHalt(RuntimeError):
    """Raised on any integrity failure. Execution stops; nothing is repaired or rerun."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_intake(path: Path) -> dict:
    if not path.is_file():
        raise ManifestHalt(f"R11-T3B intake manifest not found: {path}")
    intake = json.loads(path.read_text(encoding="utf-8"))
    if intake.get("task") != "R11-T3B":
        raise ManifestHalt(f"intake manifest is not R11-T3B: task={intake.get('task')!r}")
    if intake.get("dataset_revision") != VCD1_REVISION:
        raise ManifestHalt(
            "intake manifest pins a different VCD1 revision: "
            f"{intake.get('dataset_revision')!r} != {VCD1_REVISION!r}"
        )
    if tuple(intake.get("selected_subsets", ())) != SELECTED_SUBSETS:
        raise ManifestHalt(
            f"intake manifest subsets {intake.get('selected_subsets')!r} != {list(SELECTED_SUBSETS)!r}"
        )
    records = intake.get("file_records", [])
    if len(records) != EXPECTED_FILE_COUNT:
        raise ManifestHalt(
            f"intake manifest holds {len(records)} file records, expected {EXPECTED_FILE_COUNT}"
        )
    if intake.get("file_record_count") != EXPECTED_FILE_COUNT:
        raise ManifestHalt(
            f"intake file_record_count={intake.get('file_record_count')!r}, "
            f"expected {EXPECTED_FILE_COUNT}"
        )
    subjects = {record["subject_id"] for record in records}
    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        raise ManifestHalt(
            f"intake manifest holds {len(subjects)} distinct subjects, "
            f"expected {EXPECTED_SUBJECT_COUNT}"
        )
    return intake


def verify_hashes(records: list[dict], corpus_root: Path) -> None:
    """Re-hash every frozen MP4 from disk and halt unless all 80 match R11-T3B."""
    mismatches: list[str] = []
    missing: list[str] = []
    verified = 0
    for record in records:
        relative_path = record["relative_path"]
        path = corpus_root / relative_path
        if not path.is_file():
            missing.append(relative_path)
            continue
        if path.stat().st_size != record["byte_size"]:
            mismatches.append(f"{relative_path}: byte_size on disk differs from R11-T3B")
            continue
        if sha256_file(path) != record["sha256_local_freeze"]:
            mismatches.append(f"{relative_path}: sha256 on disk differs from R11-T3B")
            continue
        verified += 1
    if missing or mismatches:
        detail = "; ".join(missing + mismatches)
        raise ManifestHalt(
            f"hash freeze failed: {verified}/{EXPECTED_FILE_COUNT} verified. {detail}"
        )
    if verified != EXPECTED_FILE_COUNT:
        raise ManifestHalt(f"hash freeze failed: {verified}/{EXPECTED_FILE_COUNT} verified")


def build_records(records: list[dict]) -> list[dict]:
    """One execution record per intake record, sorted by `relative_path` ascending."""
    ordered = sorted(records, key=lambda record: record["relative_path"])
    paths = [record["relative_path"] for record in ordered]
    if len(set(paths)) != len(paths):
        raise ManifestHalt("intake manifest holds duplicate relative_path values")
    if paths != sorted(paths):
        raise ManifestHalt("record sort did not produce lexical ascending relative_path order")

    built = []
    for index, record in enumerate(ordered, start=1):
        probe = record["probe"]
        built.append(
            {
                "execution_order_index": index,
                "object_id": record["upstream_object"],
                "file_id": Path(record["relative_path"]).stem,
                "subject_id": record["subject_id"],
                "subset": record["subset"],
                "relative_path": record["relative_path"],
                "byte_size": record["byte_size"],
                "intake_sha256": record["sha256_local_freeze"],
                "container_codec": {
                    "container_format_name": probe["container_format_name"],
                    "video_codec_name": probe["video_codec_name"],
                    "video_profile": probe["video_profile"],
                    "pix_fmt": probe["pix_fmt"],
                },
                "width_height": {"width": probe["width"], "height": probe["height"]},
                "fps": probe["fps"],
                "duration_frame_count": {
                    "nb_frames": probe["nb_frames"],
                    "duration_sec": probe["duration_sec"],
                },
            }
        )
    return built


def build_manifest(
    *, intake_sha256: str, corpus_root: Path, execution_records: list[dict]
) -> dict:
    subsets = sorted({record["subset"] for record in execution_records})
    if subsets != sorted(SELECTED_SUBSETS):
        raise ManifestHalt(f"execution records cover subsets {subsets!r}, expected both")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK,
        "vcd1_revision": VCD1_REVISION,
        "r11_t3b_intake_manifest_sha256": intake_sha256,
        "selected_subsets": list(SELECTED_SUBSETS),
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_subject_count": EXPECTED_SUBJECT_COUNT,
        "corpus_root": str(corpus_root),
        "execution_contract": EXECUTION_CONTRACT,
        "record_order": "relative_path lexical ascending",
        "record_field_derivations": RECORD_FIELD_DERIVATIONS,
        "hash_freeze": {
            "recomputed_from_disk": True,
            "files_verified": EXPECTED_FILE_COUNT,
            "source_of_truth": "R11-T3B sha256_local_freeze",
        },
        "invariants_observed": {
            "inference_executed": False,
            "video_decoded": False,
            "frames_extracted": False,
            "ffprobe_re_run": False,
            "timestamps_in_manifest": False,
        },
        "record_count": len(execution_records),
        "records": execution_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R11-T4B execution manifest builder")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--intake-manifest", type=Path, default=DEFAULT_INTAKE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    corpus_root = args.corpus_root.resolve()
    intake_path = args.intake_manifest.resolve()

    intake = load_intake(intake_path)
    intake_sha256 = sha256_file(intake_path)

    recorded_root = Path(intake["storage_root"]).resolve()
    if recorded_root != corpus_root:
        raise ManifestHalt(
            f"corpus root {corpus_root} does not match R11-T3B storage_root {recorded_root}"
        )

    verify_hashes(intake["file_records"], corpus_root)
    print(
        f"hash freeze: {EXPECTED_FILE_COUNT}/{EXPECTED_FILE_COUNT} files re-verified from disk",
        flush=True,
    )

    execution_records = build_records(intake["file_records"])
    if len(execution_records) != EXPECTED_FILE_COUNT:
        raise ManifestHalt(
            f"built {len(execution_records)} records, expected {EXPECTED_FILE_COUNT}"
        )

    manifest = build_manifest(
        intake_sha256=intake_sha256,
        corpus_root=corpus_root,
        execution_records=execution_records,
    )
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "r11_t4b_execution_manifest.json"
    out_path.write_bytes(manifest_bytes)
    print(f"manifest {out_path} sha256={hashlib.sha256(manifest_bytes).hexdigest()}", flush=True)
    print("R11-T4B manifest complete. No inference, decode or frame extraction was run.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestHalt as error:
        print(f"HALT: {error}", file=sys.stderr, flush=True)
        sys.exit(2)
