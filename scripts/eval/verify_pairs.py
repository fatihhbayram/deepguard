"""Re-examine the near-duplicate pairs the leakage audit flagged, at a resolution that can decide.

    PYTHONPATH=scripts ~/.venvs/deepguard-lipforensics/bin/python scripts/eval/verify_pairs.py \
        --corpus ../deepguard-corpus/r7t5/corpus.json \
        --leakage ../deepguard-corpus/r7t5/leakage.json \
        --max-distance 4 \
        --output ../deepguard-corpus/r7t5/pair_verification.json

`leakage_check`'s perceptual pass is deliberately blunt: a 64-bit difference hash over an 8x9
greyscale reduction, at a generous threshold, so that it over-reports rather than misses. That
makes it a screen, not a verdict — and a screen's hits have to be adjudicated by something
sharper before they appear in a report as either a leak or a false alarm.

**The two ways a 64-bit dHash produces a small distance on unrelated media** are both testable:

- *Low-entropy frames.* A frame that is nearly flat — a dark interior, a letterboxed bar, a
  studio backdrop — produces a hash whose bits are almost all one value. Two such hashes agree
  by construction and say nothing about content. `bit_balance` measures this directly: a hash
  with 32 of 64 bits set carries a full bit of information per bit, one with 2 set carries
  almost none.
- *Coincidental global layout.* Two different talking-head clips framed the same way agree on
  the coarse light/dark structure an 8x9 reduction keeps. A finer reduction separates them; the
  same recording stays together.

So each flagged pair is re-measured three ways: a 256-bit dHash on a 17x16 reduction, a
greyscale-histogram correlation, and the bit balance of the matching frames. Genuinely shared
content stays close under all three. A screening artefact separates under at least one, and the
output says which, so the report can state what was checked rather than that nothing was found.

This decides nothing on its own and moves no clip between splits. It produces evidence for the
report's limitations section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpus import CorpusItem, read_corpus

# The finer reduction. 17x16 greyscale yields 16 rows of 16 comparisons: 256 bits, sixteen times
# the spatial detail of the screening hash, which is enough to separate two clips that merely
# share a framing convention.
FINE_WIDTH, FINE_HEIGHT = 17, 16
FINE_BITS = (FINE_WIDTH - 1) * FINE_HEIGHT

# Frames sampled per clip. More than the screen uses, because this pass runs over tens of pairs
# rather than tens of thousands.
FRAME_SAMPLES = 16

# Below this fraction of differing bits on the fine hash, two frames are still near-identical.
# 12% of 256 bits is the same generosity as the screen's 10 of 64, held at higher resolution.
FINE_DISTANCE_FRACTION = 0.12

# A hash with fewer than this fraction of its bits set (or unset) came off a frame with almost
# no structure, and its agreement with another such hash is not evidence about content.
LOW_ENTROPY_BALANCE = 0.15


def _frames(path: Path):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        collected = []
        for index in np.linspace(0, max(total - 1, 0), FRAME_SAMPLES).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            read, frame = capture.read()
            if read:
                collected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return collected
    finally:
        capture.release()


def _signatures(path: Path) -> dict:
    """Fine hashes and a normalised luminance histogram for one clip."""
    import cv2
    import numpy as np

    hashes, balances, histograms = [], [], []
    for frame in _frames(path):
        small = cv2.resize(
            frame, (FINE_WIDTH, FINE_HEIGHT), interpolation=cv2.INTER_AREA
        ).astype(np.int16)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        hashes.append(value)
        set_bits = int(bits.sum())
        balances.append(min(set_bits, FINE_BITS - set_bits) / FINE_BITS)
        histogram = cv2.calcHist([frame], [0], None, [64], [0, 256]).flatten()
        total = histogram.sum()
        histograms.append(histogram / total if total else histogram)
    return {"hashes": hashes, "balances": balances, "histograms": histograms}


def _compare(left: dict, right: dict) -> dict:
    """The closest frame pairing between two clips, under all three measures."""
    import numpy as np

    if not left["hashes"] or not right["hashes"]:
        return {"comparable": False}

    best = None
    for i, a in enumerate(left["hashes"]):
        for j, b in enumerate(right["hashes"]):
            distance = bin(a ^ b).count("1")
            if best is None or distance < best[0]:
                best = (distance, i, j)
    distance, i, j = best

    correlation = float(
        np.corrcoef(left["histograms"][i], right["histograms"][j])[0, 1]
    )
    balance = min(left["balances"][i], right["balances"][j])
    return {
        "comparable": True,
        "fine_bits": FINE_BITS,
        "fine_distance": distance,
        "fine_distance_fraction": distance / FINE_BITS,
        "histogram_correlation": correlation,
        "min_bit_balance": balance,
        "low_entropy_frames": balance < LOW_ENTROPY_BALANCE,
        "near_identical_at_fine_resolution": (
            distance / FINE_BITS <= FINE_DISTANCE_FRACTION
        ),
    }


def _verdict(comparison: dict) -> str:
    """What the three measures together support, stated in words a report can quote."""
    if not comparison.get("comparable"):
        return "undecidable: one clip could not be decoded"
    if comparison["low_entropy_frames"]:
        return (
            "screening artefact: the matching frames carry almost no structure, so their "
            "agreement is not evidence about content"
        )
    if not comparison["near_identical_at_fine_resolution"]:
        return (
            "screening artefact: the pair separates at four times the spatial resolution, "
            "so the coarse match was a shared framing convention rather than shared content"
        )
    if comparison["histogram_correlation"] < 0.9:
        return (
            "unresolved: still close spatially but the luminance distributions disagree; "
            "review by eye before drawing a conclusion"
        )
    return (
        "likely shared underlying content: the pair stays near-identical at higher "
        "resolution and the luminance distributions agree"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_pairs",
        description="Adjudicate the near-duplicate pairs flagged by the leakage screen.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--leakage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-distance",
        type=int,
        default=4,
        help="only adjudicate screened pairs at or below this screening distance",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = {item.clip_id: item for item in read_corpus(args.corpus)}
    corpus_dir = args.corpus.parent
    screened = json.loads(args.leakage.read_text(encoding="utf-8"))["perceptual"][
        "near_duplicate_pairs"
    ]
    selected = [
        pair for pair in screened if pair["min_hamming_distance"] <= args.max_distance
    ]

    def resolve(item: CorpusItem) -> Path:
        path = Path(item.path)
        return path if path.is_absolute() else corpus_dir / item.path

    cache: dict[str, dict] = {}

    def signatures(clip_id: str) -> dict:
        if clip_id not in cache:
            cache[clip_id] = _signatures(resolve(items[clip_id]))
        return cache[clip_id]

    results = []
    for pair in selected:
        left, right = pair["evaluation_clip"], pair["calibration_clip"]
        comparison = _compare(signatures(left), signatures(right))
        verdict = _verdict(comparison)
        results.append(
            {
                **pair,
                "evaluation_label": items[left].label,
                "calibration_label": items[right].label,
                "evaluation_family": items[left].family,
                "calibration_family": items[right].family,
                "comparison": comparison,
                "verdict": verdict,
            }
        )
        print(f"{left} ~ {right} (screen {pair['min_hamming_distance']}): {verdict}")

    unresolved = [
        entry
        for entry in results
        if entry["verdict"].startswith(("likely shared", "unresolved"))
    ]
    payload = {
        "schema_version": "r7-t5-pair-verification-1",
        "screening_pairs_total": len(screened),
        "adjudicated_at_or_below_distance": args.max_distance,
        "adjudicated": len(results),
        "unresolved_or_shared": len(unresolved),
        "genuine_side_unresolved_or_shared": [
            entry
            for entry in unresolved
            if entry["evaluation_label"] == "real" or entry["calibration_label"] == "real"
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n{len(results)} adjudicated, {len(unresolved)} not dismissed as screening artefacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
