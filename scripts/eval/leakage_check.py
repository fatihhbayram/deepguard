"""Prove the two splits are independent, by identity and by content.

    PYTHONPATH=scripts ~/.venvs/deepguard-lipforensics/bin/python \
        scripts/eval/leakage_check.py \
            --corpus ../deepguard-corpus/r7t5/corpus.json \
            --output ../deepguard-corpus/r7t5/leakage.json

Exit status is `0` when nothing leaked and `1` when something did, so this can gate a run.

**Two questions, because the first one can only be answered on trust.** `corpus.leakage_findings`
asks whether any `source_lineage_id` or any `sha256` appears on both sides. That is the check
the task requires and it is exact — but it can only catch a shared lineage that the build
*labelled* as shared. If two clips came from one recording and the corpus gave them different
lineage ids, no comparison of those ids will ever say so.

So this module asks the same question of the pixels. It samples frames from every base clip,
reduces each to a 64-bit difference hash, and compares every evaluation clip against every
calibration clip. Two clips that share a recording — even across a resolution change, a
recompression or a frame-rate change, which is exactly what a difference hash is insensitive to
— come back with a small Hamming distance, and the pair is reported with the distance that
found it. A perceptual near-duplicate is not proof of shared lineage on its own (two news clips
of one press conference are genuinely different recordings of one event), so this reports pairs
for a reader to judge rather than failing the run on them, and it is the *identity* check that
gates.

Cost is bounded by design: base clips only (a derivative is a near-duplicate of its base by
construction and comparing them would report every one of them), eight frames per clip, and the
comparison is a bitwise population count over 64-bit integers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpus import (
    SPLIT_CALIBRATION,
    SPLIT_EVALUATION,
    CorpusItem,
    leakage_findings,
    read_corpus,
)

# Frames sampled per clip for the perceptual pass. Evenly spaced over the decodable range, so a
# clip whose first second is a title card is still compared on its content.
FRAME_SAMPLES = 8

# Below this Hamming distance between two 64-bit difference hashes, two frames are reported as
# perceptually near-identical. 10 of 64 bits is the conventional operating point for dHash and
# is deliberately generous here: this pass is meant to over-report and be read, not to decide.
NEAR_DUPLICATE_BITS = 10


def frame_hashes(path: Path) -> list[int]:
    """Difference hashes of frames sampled across one clip, or an empty list if unreadable.

    A difference hash records whether each pixel is brighter than its right-hand neighbour on a
    9x8 greyscale reduction. It survives rescaling, re-encoding and moderate brightness change
    — the transformations that separate a clip from its own transcode — which is precisely why
    it is the right instrument for finding a lineage the manifest failed to declare.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        indices = np.linspace(0, max(total - 1, 0), FRAME_SAMPLES).astype(int)
        hashes = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            read, frame = capture.read()
            if not read:
                continue
            small = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA
            ).astype(np.int16)
            bits = (small[:, 1:] > small[:, :-1]).flatten()
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            hashes.append(value)
        return hashes
    finally:
        capture.release()


def _resolve(item: CorpusItem, corpus_dir: Path) -> Path:
    path = Path(item.path)
    return path if path.is_absolute() else corpus_dir / item.path


def perceptual_pairs(items: list[CorpusItem], corpus_dir: Path) -> dict:
    """Every cross-split pair of base clips whose sampled frames come back near-identical."""
    calibration = [
        item for item in items if item.split == SPLIT_CALIBRATION and item.is_base
    ]
    evaluation = [item for item in items if item.split == SPLIT_EVALUATION and item.is_base]

    digests: dict[str, list[int]] = {}
    unreadable: list[str] = []
    for item in (*calibration, *evaluation):
        hashes = frame_hashes(_resolve(item, corpus_dir))
        if not hashes:
            unreadable.append(item.clip_id)
        digests[item.clip_id] = hashes

    pairs = []
    for left in evaluation:
        for right in calibration:
            best = min(
                (
                    (bin(a ^ b).count("1"), a, b)
                    for a in digests[left.clip_id]
                    for b in digests[right.clip_id]
                ),
                default=None,
            )
            if best is None or best[0] > NEAR_DUPLICATE_BITS:
                continue
            pairs.append(
                {
                    "evaluation_clip": left.clip_id,
                    "evaluation_lineage": left.source_lineage_id,
                    "calibration_clip": right.clip_id,
                    "calibration_lineage": right.source_lineage_id,
                    "min_hamming_distance": best[0],
                }
            )

    return {
        "frames_per_clip": FRAME_SAMPLES,
        "near_duplicate_bits": NEAR_DUPLICATE_BITS,
        "base_clips_compared": {
            SPLIT_CALIBRATION: len(calibration),
            SPLIT_EVALUATION: len(evaluation),
        },
        "comparisons": len(calibration) * len(evaluation),
        "unreadable_clips": unreadable,
        "near_duplicate_pairs": sorted(
            pairs, key=lambda entry: entry["min_hamming_distance"]
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leakage_check",
        description="Prove zero lineage overlap between the R7-T5 splits.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--skip-perceptual",
        action="store_true",
        help="run the identity checks only (no decoding, no OpenCV)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = read_corpus(args.corpus)
    corpus_dir = args.corpus.parent

    report = {
        "schema_version": "r7-t5-leakage-1",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(args.corpus),
        "identity": leakage_findings(items),
    }
    if not args.skip_perceptual:
        report["perceptual"] = perceptual_pairs(items, corpus_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    identity = report["identity"]
    print(f"lineages: {identity['lineages_calibration']} calibration, "
          f"{identity['lineages_evaluation']} evaluation")
    print(f"shared lineage ids: {len(identity['shared_lineage_ids'])}")
    print(f"cross-split identical bytes: {len(identity['cross_split_sha256'])}")
    print(f"derivatives split from their base: {len(identity['derivatives_split_from_base'])}")
    if "perceptual" in report:
        pairs = report["perceptual"]["near_duplicate_pairs"]
        print(f"perceptual near-duplicate cross-split pairs: {len(pairs)} "
              f"over {report['perceptual']['comparisons']} comparisons")
        for pair in pairs[:10]:
            print(f"  {pair['evaluation_clip']} ~ {pair['calibration_clip']} "
                  f"(distance {pair['min_hamming_distance']})")

    if identity["leaked"]:
        print("LEAKED: the splits are not independent", file=sys.stderr)
        return 1
    print("clean: no lineage and no byte-identical clip appears in both splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
