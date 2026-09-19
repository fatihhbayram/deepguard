#!/usr/bin/env python3
"""R11-T5B: visual genuine regression execution over the 80 VCD1 `th` + `th-m` clips.

This script measures. It does not judge.

It re-hashes the 80 frozen MP4s from disk against the R11-T4B execution manifest, halts
unless the match is a perfect 80/80, then calls the two unmodified production detector
entrypoints once per clip, in the manifest's `execution_order_index` order:

    apps.api.app.detection.detect_synthetic_video   (NVIDIA SVD, remote NIM)
    apps.api.app.detection.detect_face_manipulation (local EfficientNet-B7)

Whatever those two functions return is written down exactly as returned. This file defines
no threshold, no band, no label, no verdict and no classification, and none may be added to
it. A "no-face" B7 result is a `FAILED` signal *by the detector's own contract* — R3-T1's
semantics, recorded verbatim as the detector emitted it — and this script neither reinterprets
it nor invents a category for it.

Invariants enforced here:
  * the production worker path, `evaluate_v5`, the risk engine and verdict generation are
    neither imported nor executed — only the two entrypoints above are called;
  * `app.normalization` is not invoked: the input media is the original frozen MP4, which is
    what the R11-T4B execution contract froze as `input_media`;
  * every one of the 80 inputs is attempted and its outcome recorded, success or failure;
    nothing is skipped, retried or dropped;
  * no model, weight, ruleset or production schema is touched;
  * wall-clock latency — the one non-deterministic measurement here — is written to a
    separate telemetry artifact, so the canonical evidence artifact serializes byte-identically;
  * the canonical structure is serialized twice from the same captured objects, without
    rerunning inference, and the two byte strings are asserted equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

SCHEMA_VERSION = "deepguard/r11-t5b-visual-baseline/1"
TELEMETRY_SCHEMA_VERSION = "deepguard/r11-t5b-telemetry/1"
TASK = "R11-T5B"
MANIFEST_TASK = "R11-T4B"

EXPECTED_FILE_COUNT = 80
HASH_CHUNK_BYTES = 1024 * 1024

DEFAULT_MANIFEST = Path("/app/evidence/r11_t4b_execution_manifest.json")
DEFAULT_CORPUS_ROOT = Path("/corpus")
DEFAULT_OUTPUT_DIR = Path("/out")

# The production sources whose bytes produced every number below. Hashed, not described: a
# different revision of any of them is a different measurement.
DETECTOR_SOURCES = (
    "app/detection.py",
    "app/face_detector.py",
    "app/nvidia_video.py",
    "app/limits.py",
    "app/db/models.py",
)

# Columns on the returned ORM objects that carry persistence identity rather than detector
# evidence. They are `None` here because nothing was ever flushed to a session, and they are
# dropped rather than recorded as nulls so the canonical record holds evidence only.
PERSISTENCE_COLUMNS = frozenset({"id", "analysis_id", "signal_id", "created_at"})


class BaselineHalt(RuntimeError):
    """A frozen hash did not match, or an invariant this task must not violate was broken."""


# --- Hashing -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Manifest ----------------------------------------------------------------------------


def load_manifest(path: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise BaselineHalt(f"{MANIFEST_TASK} execution manifest not found: {path}")
    manifest_sha256 = sha256_file(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("task_id") != MANIFEST_TASK:
        raise BaselineHalt(
            f"expected a {MANIFEST_TASK} manifest, got task_id={manifest.get('task_id')!r}"
        )
    records = manifest.get("records", [])
    if len(records) != EXPECTED_FILE_COUNT:
        raise BaselineHalt(
            f"manifest holds {len(records)} records, expected {EXPECTED_FILE_COUNT}"
        )
    if manifest.get("record_count") != EXPECTED_FILE_COUNT:
        raise BaselineHalt(
            f"manifest record_count={manifest.get('record_count')!r}, "
            f"expected {EXPECTED_FILE_COUNT}"
        )
    indices = [record["execution_order_index"] for record in records]
    if indices != list(range(1, EXPECTED_FILE_COUNT + 1)):
        raise BaselineHalt("manifest execution_order_index is not a contiguous 1..80 sequence")
    return manifest, manifest_sha256


def verify_hashes(records: list[dict], corpus_root: Path) -> dict[int, str]:
    """Re-hash all 80 MP4s from disk. Halt unless every one matches `intake_sha256`.

    Returns the independently computed digest per `execution_order_index`, so the record
    written later carries the hash this function actually read off the disk rather than the
    manifest's expectation copied forward.
    """
    verified: dict[int, str] = {}
    failures: list[str] = []
    for record in records:
        relative_path = record["relative_path"]
        path = corpus_root / relative_path
        if not path.is_file():
            failures.append(f"{relative_path}: absent from disk")
            continue
        size = path.stat().st_size
        if size != record["byte_size"]:
            failures.append(
                f"{relative_path}: byte_size {size} != manifest {record['byte_size']}"
            )
            continue
        digest = sha256_file(path)
        if digest != record["intake_sha256"]:
            failures.append(f"{relative_path}: sha256 on disk differs from manifest")
            continue
        verified[record["execution_order_index"]] = digest
    if failures or len(verified) != EXPECTED_FILE_COUNT:
        detail = "; ".join(failures) or "count mismatch"
        raise BaselineHalt(
            f"pre-inference disk verification failed: {len(verified)}/{EXPECTED_FILE_COUNT} "
            f"verified. {detail}"
        )
    return verified


# --- Detector output capture ---------------------------------------------------------------


def capture_columns(row) -> dict:
    """Every mapped column on a returned ORM object, verbatim, minus persistence identity.

    Read off the SQLAlchemy mapper rather than a hand-written field list, so a column added to
    the contract appears here without this script being taught about it.

    `mapper.column_attrs` rather than `__table__.columns`: the two disagree wherever a mapped
    attribute is named differently from its DB column, and `AnalysisSignal.signal_metadata`
    over the `metadata` column is exactly that case — `__table__.columns` would name it
    `metadata`, and reading that attribute off the instance returns the declarative base's
    `MetaData` object instead of the detector's evidence.
    """
    from sqlalchemy import inspect as sa_inspect

    captured = {}
    for attribute in sa_inspect(row).mapper.column_attrs:
        if attribute.key in PERSISTENCE_COLUMNS:
            continue
        captured[attribute.key] = getattr(row, attribute.key)
    return captured


def capture_svd(detection, path: Path) -> tuple[dict, dict]:
    """One `detect_synthetic_video` call. Returns (evidence, state)."""
    import asyncio

    try:
        signal, segments = asyncio.run(detection.detect_synthetic_video(path))
    except BaseException as error:  # noqa: BLE001 - recorded, never swallowed
        return (
            {},
            {
                "attempted": True,
                "returned": False,
                "raised": type(error).__name__,
                "raised_module": type(error).__module__,
            },
        )
    evidence = {
        "signal": capture_columns(signal),
        "segments": [capture_columns(segment) for segment in segments],
        "segment_count": len(segments),
    }
    return evidence, {"attempted": True, "returned": True, "raised": None}


def capture_b7(detection, path: Path) -> tuple[dict, dict]:
    """One `detect_face_manipulation` call. Returns (evidence, state).

    The entrypoint's docstring states it never raises: every `FaceDetectorError` becomes a
    `FAILED` signal. The `except` below is therefore a recorder for the case the contract says
    cannot happen, not an expectation that it will.
    """
    try:
        signal = detection.detect_face_manipulation(path)
    except BaseException as error:  # noqa: BLE001 - recorded, never swallowed
        return (
            {},
            {
                "attempted": True,
                "returned": False,
                "raised": type(error).__name__,
                "raised_module": type(error).__module__,
            },
        )
    return {"signal": capture_columns(signal)}, {
        "attempted": True,
        "returned": True,
        "raised": None,
    }


# --- Identity headers ------------------------------------------------------------------------


def detector_source_hashes(app_root: Path) -> dict:
    hashes = {}
    for relative in DETECTOR_SOURCES:
        source = app_root / relative
        if not source.is_file():
            raise BaselineHalt(f"production detector source not found: {source}")
        hashes[relative] = sha256_file(source)
    return hashes


def svd_identity(nvidia_video, limits) -> dict:
    """What answered for SVD. The function ID is NVIDIA's public catalog identifier."""
    function_id = os.getenv("NVIDIA_SVD_FUNCTION_ID", "").strip()
    if not function_id:
        raise BaselineHalt("NVIDIA_SVD_FUNCTION_ID is not configured; SVD identity is unknown")
    return {
        "detector": "nvidia_synthetic_video_detection",
        "execution": "remote_nim",
        "endpoint_target": nvidia_video.PREVIEW_TARGET,
        "nvcf_function_id_configured": function_id,
        "provider_constant": "NVIDIA",
        "model_revision_reported_by_provider": (
            "none — NVIDIA publishes no model version for this function; the NVCF function "
            "ID returned per call as `provider_version` is what pins the deployed detector"
        ),
        "timeout_seconds": limits.nvidia_svd_timeout_seconds(),
    }


def b7_identity(face_detector) -> dict:
    """What answered for B7, by digest. Both artifacts are re-hashed off the disk they load from."""
    model_dir = face_detector._model_dir()
    classifier = model_dir / face_detector.CLASSIFIER_FILENAME
    locator = model_dir / face_detector.LOCATOR_FILENAME
    for artifact in (classifier, locator):
        if not artifact.is_file():
            raise BaselineHalt(f"B7 model artifact not found: {artifact}")
    classifier_disk = sha256_file(classifier)
    locator_disk = sha256_file(locator)
    if classifier_disk != face_detector.CLASSIFIER_SHA256:
        raise BaselineHalt("B7 classifier on disk does not match its pinned digest")
    if locator_disk != face_detector.LOCATOR_SHA256:
        raise BaselineHalt("B7 locator on disk does not match its pinned digest")
    return {
        "detector": "efficientnet_b7_face_manipulation",
        "execution": "local_in_process",
        "model_dir": str(model_dir),
        "classifier": {
            "repository": face_detector.CLASSIFIER_REPOSITORY,
            "revision": face_detector.CLASSIFIER_REVISION,
            "filename": face_detector.CLASSIFIER_FILENAME,
            "pinned_sha256": face_detector.CLASSIFIER_SHA256,
            "verified_on_disk_sha256": classifier_disk,
        },
        "locator": {
            "repository": face_detector.LOCATOR_REPOSITORY,
            "revision": face_detector.LOCATOR_REVISION,
            "filename": face_detector.LOCATOR_FILENAME,
            "pinned_sha256": face_detector.LOCATOR_SHA256,
            "verified_on_disk_sha256": locator_disk,
        },
        "frame_samples_configured": face_detector.frame_samples(),
        "frame_samples_default": face_detector.DEFAULT_FRAME_SAMPLES,
        "provider_constant": "efficientnet-b7",
    }


def runtime_identity() -> dict:
    import cv2
    import grpc
    import numpy
    import torch

    return {
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "opencv_version": cv2.__version__,
        "grpcio_version": grpc.__version__,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


# --- Serialization ---------------------------------------------------------------------------


def canonical_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


# --- Main --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="R11-T5B visual genuine regression execution")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--app-root", type=Path, default=Path("/app"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="smoke-test only: run the first N clips. 0 runs all 80. A limited run writes "
        "its artifacts under a `SMOKE_` prefix and is never the baseline.",
    )
    args = parser.parse_args()

    manifest, manifest_sha256 = load_manifest(args.manifest.resolve())
    records = manifest["records"]
    print(f"{MANIFEST_TASK} manifest sha256={manifest_sha256}", flush=True)

    # Step 1. Every hash, before any detector is imported, let alone called.
    disk_hashes = verify_hashes(records, args.corpus_root.resolve())
    print(
        f"pre-inference disk verification: {len(disk_hashes)}/{EXPECTED_FILE_COUNT} "
        "SHA-256 matched the R11-T4B manifest",
        flush=True,
    )

    # Step 2. Only now are the detectors imported. Nothing above this line can reach a model.
    sys.path.insert(0, str(args.app_root))
    from app import detection, face_detector, limits, nvidia_video  # noqa: E402

    header = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK,
        "r11_t4b_manifest_sha256": manifest_sha256,
        "vcd1_revision": manifest["vcd1_revision"],
        "corpus_root": str(args.corpus_root.resolve()),
        "selected_subsets": manifest["selected_subsets"],
        "record_order": "R11-T4B execution_order_index ascending (relative_path lexical)",
        "input_media": manifest["execution_contract"]["input_media"],
        "entrypoints_invoked": {
            "svd": "apps.api.app.detection.detect_synthetic_video",
            "b7": "apps.api.app.detection.detect_face_manipulation",
        },
        "svd_identity": svd_identity(nvidia_video, limits),
        "b7_identity": b7_identity(face_detector),
        "runtime": runtime_identity(),
        "detector_source_sha256": detector_source_hashes(args.app_root),
        "capture_contract": {
            "source": "every mapped column on the returned ORM objects, verbatim",
            "persistence_columns_dropped": sorted(PERSISTENCE_COLUMNS),
            "persistence_columns_note": (
                "None on every row here — nothing was flushed to a session — and dropped so "
                "the record carries detector evidence only"
            ),
            "interpretation_added": "none",
        },
        "invariants_observed": {
            "production_worker_path_invoked": False,
            "evaluate_v5_invoked": False,
            "risk_engine_invoked": False,
            "verdict_generated": False,
            "normalization_invoked": False,
            "media_transcoded": False,
            "production_code_modified": False,
            "timestamps_in_canonical_artifact": False,
            "latency_in_canonical_artifact": False,
        },
    }

    # Step 3. Inference, in manifest order, one clip at a time.
    limit = args.limit if args.limit > 0 else len(records)
    evidence_records: list[dict] = []
    telemetry_records: list[dict] = []
    started = time.time()

    for record in records[:limit]:
        index = record["execution_order_index"]
        path = args.corpus_root.resolve() / record["relative_path"]

        svd_started = time.perf_counter()
        svd_evidence, svd_state = capture_svd(detection, path)
        svd_seconds = time.perf_counter() - svd_started

        b7_started = time.perf_counter()
        b7_evidence, b7_state = capture_b7(detection, path)
        b7_seconds = time.perf_counter() - b7_started

        evidence_records.append(
            {
                "execution_order_index": index,
                "object_id": record["object_id"],
                "file_id": record["file_id"],
                "subject_id": record["subject_id"],
                "subset": record["subset"],
                "relative_path": record["relative_path"],
                "expected_t4b_sha256": record["intake_sha256"],
                "independently_verified_disk_sha256": disk_hashes[index],
                "svd": svd_evidence,
                "b7": b7_evidence,
                "detector_state": {"svd": svd_state, "b7": b7_state},
            }
        )
        telemetry_records.append(
            {
                "execution_order_index": index,
                "file_id": record["file_id"],
                "relative_path": record["relative_path"],
                "svd_wall_seconds": round(svd_seconds, 6),
                "b7_wall_seconds": round(b7_seconds, 6),
            }
        )

        svd_status = (svd_evidence.get("signal") or {}).get("status") or svd_state["raised"]
        b7_status = (b7_evidence.get("signal") or {}).get("status") or b7_state["raised"]
        print(
            f"[{index:>2}/{limit}] {record['relative_path']} "
            f"svd={svd_status} ({svd_seconds:.1f}s) b7={b7_status} ({b7_seconds:.1f}s)",
            flush=True,
        )

    total_seconds = time.time() - started

    if limit == EXPECTED_FILE_COUNT and len(evidence_records) != EXPECTED_FILE_COUNT:
        raise BaselineHalt(
            f"{len(evidence_records)} records produced, expected {EXPECTED_FILE_COUNT}"
        )

    baseline = dict(header)
    baseline["record_count"] = len(evidence_records)
    baseline["inputs_attempted"] = len(evidence_records)
    baseline["records"] = evidence_records

    # Step 4. Byte-identical serialization, from the same captured objects, no rerun.
    first = canonical_bytes(baseline)
    second = canonical_bytes(baseline)
    if first != second:
        raise BaselineHalt("canonical serialization is not byte-identical across two passes")
    baseline_sha256 = hashlib.sha256(first).hexdigest()

    telemetry = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "task_id": TASK,
        "authority": "non_authoritative",
        "note": (
            "Wall-clock latency only. Non-deterministic by nature and deliberately kept out "
            "of the canonical evidence artifact so its SHA-256 stays reproducible."
        ),
        "canonical_artifact_sha256": baseline_sha256,
        "total_wall_seconds": round(total_seconds, 3),
        "record_count": len(telemetry_records),
        "records": telemetry_records,
    }

    prefix = "" if limit == EXPECTED_FILE_COUNT else "SMOKE_"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / f"{prefix}r11_t5b_visual_baseline.json"
    telemetry_path = args.output_dir / f"{prefix}r11_t5b_telemetry.json"
    baseline_path.write_bytes(first)
    telemetry_path.write_bytes(
        (json.dumps(telemetry, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    )

    print(f"double-serialization byte-identical: True ({len(first)} bytes)", flush=True)
    print(f"baseline  {baseline_path} sha256={baseline_sha256}", flush=True)
    print(f"telemetry {telemetry_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaselineHalt as halt:
        print(f"HALT: {halt}", file=sys.stderr, flush=True)
        raise SystemExit(2) from halt
