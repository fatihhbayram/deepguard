"""Write a manifest holding a chosen subset of a corpus, for the runs that cannot afford all of it.

    PYTHONPATH=scripts python3 scripts/eval/subset_manifest.py \
        --corpus ../deepguard-corpus/r7t5/corpus.json \
        --split evaluation --base-clips-only \
        --output ../deepguard-corpus/r7t5/manifest_svd.csv

Two of the three detectors run locally and cost only wall-clock. NVIDIA's runs against a
metered remote deployment, so this task scores the evaluation split's base clips with it rather
than the whole corpus, and says so wherever an SVD figure is reported.

That is a coverage decision, not a metric decision: the clips it selects are selected by split
and by derivative status, both of which are fixed before any score exists. Nothing here can
select on a score, and a subset chosen by outcome would not be a subset, it would be a result.

The output is an ordinary benchmark manifest with paths rewritten relative to its own location,
so it can sit beside the corpus or be copied into a container next to the clips.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpus import SPLIT_CALIBRATION, SPLIT_EVALUATION, read_corpus, write_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="subset_manifest",
        description="Write a benchmark manifest for part of an R7-T5 corpus.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split", choices=(SPLIT_CALIBRATION, SPLIT_EVALUATION), action="append"
    )
    parser.add_argument(
        "--base-clips-only",
        action="store_true",
        help="exclude constructed derivatives, keeping one clip per lineage",
    )
    parser.add_argument(
        "--include-private-derivatives",
        action="store_true",
        help="keep the private lineages' derivatives even under --base-clips-only",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = read_corpus(args.corpus)

    chosen = []
    for item in items:
        if args.split and item.split not in args.split:
            continue
        if args.base_clips_only and not item.is_base:
            if not (args.include_private_derivatives and item.private):
                continue
        chosen.append(item)

    if not chosen:
        print("the selection is empty; no manifest written", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_manifest(chosen, args.output)
    lineages = len({item.source_lineage_id for item in chosen})
    print(f"{len(chosen)} clips, {lineages} lineages -> {args.output}")
    print(f"manifest sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
