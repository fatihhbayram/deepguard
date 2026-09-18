"""R11-T3A: AASIST baseline execution over the 12 frozen Real-TurnTurk derivatives.

This script measures. It does not judge.

It verifies the R11-T2A frozen hashes, runs each derivative through
`app.audio_detector.analyze_audio_authenticity(...)` exactly as that module exists, and
persists the raw logits as evidence. No threshold, calibration, vote, classification metric
or risk-engine rule is applied anywhere in this file, and none may be added to it: the moment
a number here is compared against a cut-off it stops being a measurement.

Fields the detector contract does not expose are recorded as
`not_observable_from_current_detector_contract` rather than derived. In particular the
detector reports the windows it returned; it has no notion of a window attempted, failed or
skipped, so those counts are not inferred from durations.

Runs inside the API image, which is the only environment carrying the pinned checkpoint at
/models/aasist.onnx and the onnxruntime build the production path uses. See the R11-T3A
report for the exact invocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "deepguard/r11-t3a-aasist-baseline/1"
TASK = "R11-T3A"
UPSTREAM_TASK = "R11-T2A"

# The detector contract exposes the windows it returned. It carries no counter for a window
# attempted, failed or skipped, and this run refuses to reconstruct one from durations.
NOT_OBSERVABLE = "not_observable_from_current_detector_contract"
UNEXPOSED_WINDOW_FIELDS = ("windows_attempted", "windows_failed", "windows_skipped")

HASH_CHUNK_BYTES = 1024 * 1024


class BaselineHalt(RuntimeError):
    """An input did not match its frozen hash, or a determinism assertion failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --- Serialization ---------------------------------------------------------------------------
#
# Both writers are pure functions of the in-memory result structure and are called twice on the
# identical object to prove the artifact is deterministic. Nothing below may read the clock,
# the filesystem or a set's iteration order.


def serialize_json(report: dict) -> bytes:
    return (
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def serialize_csv(report: dict) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for record in report["records"]:
        for window in record["windows"]:
            writer.writerow(
                [
                    record["recording_id"],
                    record["panel"],
                    record["person"],
                    record["derivative"]["relative_path"],
                    record["derivative"]["sha256_verified"],
                    window["window_index"],
                    window["start_sample"],
                    window["end_sample"],
                    window["padded_samples"],
                    repr(window["logits"][0]),
                    repr(window["logits"][1]),
                    repr(window["bona_fide_logit"]),
                ]
            )
    return buffer.getvalue().encode("utf-8")


# --- Runtime identity ------------------------------------------------------------------------


def load_detector(detector_path: Path):
    """Import the detector straight from its file, without importing the `app` package.

    `app/__init__.py` validates the service's database and object-store configuration at import
    time, which this measurement neither needs nor should depend on. `audio_detector.py` imports
    nothing from its own package, so loading the file by path gives the identical module object
    while keeping the baseline free of unrelated service configuration. The file's SHA-256 goes
    into the report, which is a stronger statement than "was not edited".
    """
    spec = importlib.util.spec_from_file_location("deepguard_audio_detector", detector_path)
    if spec is None or spec.loader is None:
        raise BaselineHalt(f"could not load the detector at {detector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_metadata(detector, model_path: Path, detector_path: Path, detector_sha256: str) -> dict:
    import numpy
    import onnxruntime

    # The production path pins CPUExecutionProvider in `_load_session`. Recorded as the device
    # actually used, alongside what the installed runtime could have offered, so a later run on
    # a different host is comparable rather than merely plausible.
    metadata = {
        "device": "cpu",
        "execution_provider_used": "CPUExecutionProvider",
        "execution_providers_available": sorted(onnxruntime.get_available_providers()),
        "inference_runtime": "onnxruntime",
        "numpy_version": numpy.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    # PyTorch and CUDA are named in the R11-T3A acceptance criteria, but neither is on this
    # execution path: the checkpoint is ONNX and the session is CPU-pinned by the detector.
    # Recording the fact is more useful than recording a version that did not compute anything.
    try:
        import torch
    except ImportError:
        metadata["torch"] = {
            "installed": False,
            "role": "not_in_execution_path",
        }
    else:
        metadata["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "role": "not_in_execution_path",
        }

    metadata["cuda"] = {
        "used": False,
        "reason": "detector pins CPUExecutionProvider in _load_session; no CUDA provider requested",
    }

    metadata["detector_source"] = {
        "path": str(detector_path),
        "sha256": detector_sha256,
        "entrypoint": "analyze_audio_authenticity",
        "modified_by_this_task": False,
    }

    metadata["model"] = {
        "repository": detector.MODEL_REPOSITORY,
        "revision": detector.MODEL_REVISION,
        "filename": detector.MODEL_FILENAME,
        "sha256_expected": detector.MODEL_SHA256,
        "sha256_on_disk": sha256_file(model_path),
        "path": str(model_path),
        "window_samples_expected": detector.EXPECTED_WINDOW_SAMPLES,
        "sample_rate": detector.MODEL_SAMPLE_RATE,
        "channels": detector.MODEL_CHANNELS,
    }
    return metadata


# --- Input verification ----------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedInput:
    record: dict
    wav_path: Path
    sha256_verified: str


def verify_inputs(manifest: dict, manifest_root: Path) -> list[VerifiedInput]:
    """Re-hash every derivative against its frozen R11-T2A hash. Halt on the first mismatch.

    `manifest_root` is the directory holding the manifest: R11-T2A records each derivative's
    path relative to that file, not to the corpus root.
    """
    verified: list[VerifiedInput] = []
    for record in sorted(manifest["records"], key=lambda item: item["derivative"]["relative_path"]):
        derivative = record["derivative"]
        wav_path = manifest_root / derivative["relative_path"]
        if not wav_path.is_file():
            raise BaselineHalt(f"frozen derivative missing: {wav_path}")

        actual = sha256_file(wav_path)
        if actual != derivative["sha256"]:
            raise BaselineHalt(
                f"hash mismatch for {derivative['relative_path']}: "
                f"manifest {derivative['sha256']}, on disk {actual}"
            )

        size = wav_path.stat().st_size
        if size != derivative["size_bytes"]:
            raise BaselineHalt(
                f"size mismatch for {derivative['relative_path']}: "
                f"manifest {derivative['size_bytes']}, on disk {size}"
            )

        verified.append(VerifiedInput(record=record, wav_path=wav_path, sha256_verified=actual))

    if len(verified) != manifest["record_count"]:
        raise BaselineHalt(
            f"manifest declares {manifest['record_count']} records, verified {len(verified)}"
        )
    return verified


# --- Measurement -----------------------------------------------------------------------------


def measure(entry: VerifiedInput, detector, model_path: Path) -> dict:
    evidence = detector.analyze_audio_authenticity(entry.wav_path, model_path=model_path)

    record = entry.record
    derivative = record["derivative"]
    parity = record["duration_parity"]

    return {
        "recording_id": record["recording_id"],
        "panel": record["panel"],
        "person": record["person"],
        "source": {
            "relative_path": record["source"]["relative_path"],
            "sha256": record["source"]["sha256"],
            "container_duration_seconds": record["source"]["duration_seconds"],
        },
        "derivative": {
            "relative_path": derivative["relative_path"],
            "sha256_manifest": derivative["sha256"],
            "sha256_verified": entry.sha256_verified,
            "sha256_matches_manifest": entry.sha256_verified == derivative["sha256"],
            "size_bytes": derivative["size_bytes"],
            "manifest_duration_seconds": derivative["duration_seconds"],
        },
        "upstream_duration_parity": {
            "within_tolerance": parity["within_tolerance"],
            "delta_seconds": parity["delta_seconds"],
            "note": parity["note"],
        },
        # Everything under `observed` is read straight off the detector's return value.
        "observed": {
            "total_samples": evidence.total_samples,
            "sample_rate": evidence.sample_rate,
            "channels": evidence.channels,
            "window_samples": evidence.window_samples,
            "window_padding_scheme": evidence.window_padding_scheme,
            "windows_returned": len(evidence.windows),
            "model_repository": evidence.model_repository,
            "model_revision": evidence.model_revision,
            "model_sha256": evidence.model_sha256,
        },
        # Exact arithmetic on two observed integers, labelled so it is never mistaken for a
        # field the detector reported.
        "derived_from_observed": {
            "decoded_duration_seconds": evidence.total_samples / evidence.sample_rate,
            "derivation": "total_samples / sample_rate",
        },
        "not_observable": {field: NOT_OBSERVABLE for field in UNEXPOSED_WINDOW_FIELDS},
        "windows": [
            {
                "window_index": window.window_index,
                "start_sample": window.start_sample,
                "end_sample": window.end_sample,
                "padded_samples": window.padded_samples,
                "logits": list(window.logits),
                "bona_fide_logit": window.bona_fide_logit,
            }
            for window in evidence.windows
        ],
    }


def build_report(manifest: dict, manifest_sha256: str, records: list[dict], metadata: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "purpose": (
            "Raw AASIST logits for the 12 frozen Real-TurnTurk derivatives, preserved as "
            "measurement evidence. No classification, calibration, threshold, vote or verdict "
            "is present in this artifact."
        ),
        "upstream_manifest": {
            "task": UPSTREAM_TASK,
            "relative_path": "r11_t2a/r11_t2a_execution_manifest.json",
            "schema_version": manifest["schema_version"],
            "sha256": manifest_sha256,
        },
        "upstream_manifest_sha256": manifest_sha256,
        "dataset": manifest["dataset"],
        "semantic_flags": {
            "classification_performed": False,
            "calibration_applied": False,
            "threshold_applied": False,
            "risk_engine_invoked": False,
            "voting_applied": False,
            "verdict_produced": False,
            "inference_performed": True,
            "audio_padded_or_trimmed_by_this_task": False,
            "detector_source_modified": False,
        },
        "runtime": metadata,
        "record_count": len(records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R11-T3A AASIST baseline execution")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("/app/app/audio_detector.py"),
        help="The unmodified audio_detector.py to measure through.",
    )
    args = parser.parse_args()

    detector = load_detector(args.detector)
    detector_sha256 = sha256_file(args.detector)

    manifest_path = args.manifest.resolve()
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("task") != UPSTREAM_TASK:
        raise BaselineHalt(f"expected an {UPSTREAM_TASK} manifest, got task {manifest.get('task')!r}")

    # Hash gate closes before a single window is inferred.
    verified = verify_inputs(manifest, manifest_path.parent)
    print(f"verified {len(verified)} frozen derivatives against {UPSTREAM_TASK} hashes", flush=True)

    metadata = runtime_metadata(detector, args.model_path, args.detector, detector_sha256)

    records = []
    for index, entry in enumerate(verified, start=1):
        print(f"[{index}/{len(verified)}] {entry.record['derivative']['relative_path']}", flush=True)
        records.append(measure(entry, detector, args.model_path))

    report = build_report(manifest, manifest_sha256, records, metadata)

    # Determinism is a property of the serializer over one captured structure, so the same
    # object is written twice. Re-running inference would test the model, not the artifact.
    json_first, json_second = serialize_json(report), serialize_json(report)
    csv_first, csv_second = serialize_csv(report), serialize_csv(report)
    if json_first != json_second:
        raise BaselineHalt("JSON serialization is not deterministic over one result structure")
    if csv_first != csv_second:
        raise BaselineHalt("CSV serialization is not deterministic over one result structure")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "r11_t3a_aasist_baseline.json"
    csv_path = args.out_dir / "r11_t3a_aasist_windows.csv"
    json_path.write_bytes(json_first)
    csv_path.write_bytes(csv_first)

    determinism = {
        "schema_version": SCHEMA_VERSION,
        "method": "double_serialization_of_identical_in_memory_result_structure",
        "inference_runs": 1,
        "json_sha256_pass_1": sha256_bytes(json_first),
        "json_sha256_pass_2": sha256_bytes(json_second),
        "json_byte_identical": True,
        "csv_sha256_pass_1": sha256_bytes(csv_first),
        "csv_sha256_pass_2": sha256_bytes(csv_second),
        "csv_byte_identical": True,
    }
    (args.out_dir / "r11_t3a_determinism.json").write_bytes(
        (json.dumps(determinism, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )

    print(f"json {json_path} sha256={determinism['json_sha256_pass_1']}", flush=True)
    print(f"csv  {csv_path} sha256={determinism['csv_sha256_pass_1']}", flush=True)
    print("determinism: double-serialization byte-identical", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
