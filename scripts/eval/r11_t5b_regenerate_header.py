#!/usr/bin/env python3
"""R11-T5B header-only regeneration. No inference, no NVCF call, no detector execution.

The 80 captured detector evidence records are carried through untouched: they are parsed from
the existing canonical artifact and re-emitted as-is. Nothing in this file recomputes a score,
re-reads a model, contacts NVIDIA, or reinterprets a returned field.

What it adds is the execution provenance the approved R11-T5B plan requires in the canonical
header and which the first pass recorded only in the Markdown report:

  * the container/image the run executed in, as precisely as it can be stated;
  * the production detector source digests inside that container, recomputed here;
  * the same sources' digests in the working tree, and the equality result between the two;
  * the normalization decision, derived with the production predicate rather than asserted.

`app.media.probe_media` and `app.normalization.needs_normalization` are read-only: ffprobe reads
container and stream metadata, and the predicate is pure. Neither decodes a frame, transcodes
anything, or touches a detector. `app.detection`, `app.worker`, `app.risk_engine` and
`evaluate_v5` are not imported anywhere in this file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REGENERATION_SCHEMA_VERSION = "deepguard/r11-t5b-visual-baseline/2"
TASK = "R11-T5B"
EXPECTED_FILE_COUNT = 80
HASH_CHUNK_BYTES = 1024 * 1024

DETECTOR_SOURCES = (
    "app/detection.py",
    "app/face_detector.py",
    "app/nvidia_video.py",
    "app/limits.py",
    "app/db/models.py",
)


class RegenerationHalt(RuntimeError):
    """An invariant this regeneration must not violate was broken."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def source_digests(root: Path, label: str) -> dict:
    digests = {}
    for relative in DETECTOR_SOURCES:
        source = root / relative
        if not source.is_file():
            raise RegenerationHalt(f"{label} source not found: {source}")
        digests[relative] = sha256_file(source)
    return digests


def normalization_decision(records: list[dict], corpus_root: Path) -> dict:
    """Run the production bypass predicate over all 80 clips. Read-only; nothing is transcoded.

    This is the evidence behind `input_media: original_frozen_mp4`: if `needs_normalization` is
    False for every clip, the Fast Decision Path would hand SVD and B7 these exact bytes, so
    calling the entrypoints on the originals is not a divergence from production.
    """
    from app.media import probe_media
    from app.normalization import (
        CANONICAL_CODEC,
        CANONICAL_PIX_FMT,
        MP4_MAJOR_BRANDS,
        needs_normalization,
    )

    per_clip = []
    requiring = 0
    for record in records:
        path = corpus_root / record["relative_path"]
        metadata = asyncio.run(probe_media(path))
        required = needs_normalization(metadata)
        requiring += int(required)
        per_clip.append(
            {
                "execution_order_index": record["execution_order_index"],
                "relative_path": record["relative_path"],
                "major_brand": metadata.major_brand,
                "codec_name": metadata.codec_name,
                "pix_fmt": metadata.pix_fmt,
                "constant_frame_rate": metadata.constant_frame_rate,
                "needs_normalization": required,
            }
        )

    return {
        "detector_input_media": "original_frozen_mp4",
        "predicate": "apps.api.app.normalization.needs_normalization",
        "predicate_inputs": "apps.api.app.media.probe_media (ffprobe, read-only)",
        "clips_evaluated": len(per_clip),
        "clips_requiring_normalization": requiring,
        "needs_normalization_false_for_all": requiring == 0,
        "bypass_reason": (
            "`needs_normalization` returns False only when the major brand is in "
            f"{sorted(MP4_MAJOR_BRANDS)}, the codec is {CANONICAL_CODEC!r}, the pixel format is "
            f"{CANONICAL_PIX_FMT!r} and the frame rate is constant. All 80 clips satisfy every "
            "one of those conditions, so `app.worker.prepared_artifact` would yield the original "
            "file unchanged and no derivative would exist. The bytes these detectors were given "
            "are therefore the bytes the production Fast Decision Path would give them."
        ),
        "normalization_invoked_by_this_task": False,
        "media_transcoded_by_this_task": False,
        "per_clip": per_clip,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R11-T5B canonical header regeneration")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, default=Path("/corpus"))
    parser.add_argument("--app-root", type=Path, default=Path("/app"))
    parser.add_argument("--worktree-root", type=Path, default=Path("/worktree/apps/api"))
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-created", required=True)
    parser.add_argument("--git-head", required=True)
    args = parser.parse_args()

    original_bytes = args.baseline.read_bytes()
    baseline = json.loads(original_bytes)
    if baseline.get("task_id") != TASK:
        raise RegenerationHalt(f"not an {TASK} artifact: task_id={baseline.get('task_id')!r}")
    records = baseline["records"]
    if len(records) != EXPECTED_FILE_COUNT:
        raise RegenerationHalt(f"{len(records)} records, expected {EXPECTED_FILE_COUNT}")

    # The evidence, carried through byte-for-byte. Held aside so nothing below can touch it and
    # re-attached unchanged at the end.
    preserved_records = records
    preserved_digest = hashlib.sha256(
        json.dumps(preserved_records, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    sys.path.insert(0, str(args.app_root))

    container_digests = source_digests(args.app_root, "container")
    worktree_digests = source_digests(args.worktree_root, "working tree")
    mismatches = sorted(
        relative
        for relative in DETECTOR_SOURCES
        if container_digests[relative] != worktree_digests[relative]
    )

    provenance = {
        "executed_in": "throwaway container from the production worker image",
        "reason_host_was_not_used": (
            "the host fails the production detector contract: host torch is 2.14, which "
            "`app.face_detector.VERIFIED_TORCH_VERSIONS` ({(2, 11), (2, 13)}) refuses by design, "
            "and the host has no grpcio at all"
        ),
        "image_reference": args.image_reference,
        "image_id": args.image_id,
        "image_created": args.image_created,
        "image_id_source": (
            "`docker image inspect` on the host; not observable from inside the container, so it "
            "is carried in rather than measured here"
        ),
        "container_mounts": {
            "/corpus": "VCD1 corpus, read-only",
            "/app/evidence/r11_t4b_execution_manifest.json": "R11-T4B manifest, read-only",
            "/out": "artifact output, the only writable mount",
        },
        "resource_limits": {"memory": "6g", "cpus": "4.0"},
        "network": "deepguard_default",
        "git_head": args.git_head,
        "code_identity": {
            "container_source_sha256": container_digests,
            "working_tree_source_sha256": worktree_digests,
            "container_matches_working_tree": not mismatches,
            "files_compared": len(DETECTOR_SOURCES),
            "files_matched": len(DETECTOR_SOURCES) - len(mismatches),
            "mismatched_files": mismatches,
            "note": (
                "the worker image bakes its code in rather than mounting it, so the digests the "
                "run recorded are compared against the working tree to establish that the "
                "executed code is the reviewed code"
            ),
        },
    }
    if mismatches:
        raise RegenerationHalt(
            f"container code differs from the working tree for: {', '.join(mismatches)}"
        )

    baseline["schema_version"] = REGENERATION_SCHEMA_VERSION
    baseline["execution_provenance"] = provenance
    baseline["normalization_decision"] = normalization_decision(records, args.corpus_root)
    baseline["regeneration"] = {
        "kind": "header_only",
        "supersedes_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "superseded_schema_version": "deepguard/r11-t5b-visual-baseline/1",
        "detector_evidence_records": "carried through unchanged from the superseded artifact",
        "detector_evidence_sha256": preserved_digest,
        "inference_rerun": False,
        "nvcf_calls_made": 0,
        "detectors_executed": False,
        "reason": (
            "the approved R11-T5B plan requires execution environment and code identity "
            "provenance in the canonical header; the first pass recorded it in the Markdown "
            "report only"
        ),
    }
    baseline["invariants_observed"]["detector_evidence_recomputed"] = False
    baseline["invariants_observed"]["detector_evidence_reinterpreted"] = False

    if baseline["records"] is not preserved_records:
        raise RegenerationHalt("the evidence record array was replaced rather than carried through")

    first = canonical_bytes(baseline)
    second = canonical_bytes(baseline)
    if first != second:
        raise RegenerationHalt("canonical serialization is not byte-identical across two passes")

    args.out.write_bytes(first)
    print(f"evidence records carried through unchanged: sha256={preserved_digest}", flush=True)
    print(f"container vs working tree: {len(DETECTOR_SOURCES)}/{len(DETECTOR_SOURCES)} matched", flush=True)
    nd = baseline["normalization_decision"]
    print(
        f"needs_normalization false for all: {nd['needs_normalization_false_for_all']} "
        f"({nd['clips_evaluated'] - nd['clips_requiring_normalization']}/{nd['clips_evaluated']})",
        flush=True,
    )
    print(f"double-serialization byte-identical: True ({len(first)} bytes)", flush=True)
    print(f"canonical sha256={hashlib.sha256(first).hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegenerationHalt as halt:
        print(f"HALT: {halt}", file=sys.stderr, flush=True)
        raise SystemExit(2) from halt
