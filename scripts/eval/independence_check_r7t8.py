"""Prove the R7-T8 corpus is independent of R7-T7, and of itself across the split boundary.

    PYTHONPATH=scripts ~/.venvs/deepguard-effort/bin/python \
        scripts/eval/independence_check_r7t8.py \
            --corpus ../deepguard-corpus/r7t8/corpus.json \
            --r7t7-corpus ../deepguard-corpus/r7t5/corpus.json \
            --protocol scripts/eval/r7t8_protocol.json \
            --output ../deepguard-corpus/r7t8/independence.json

Exit status is `0` when nothing leaked and `1` when something did, so this gates the run.

R7-T5's `leakage_check.py` asks whether the two splits of one corpus are independent. This asks
that question too, and one more that R7-T5 never had to: whether this corpus is independent of a
*previous task's* corpus. A stability study whose "new" data overlaps the data the threshold was
derived on measures nothing, and the overlap can arrive by four different routes, so all four are
checked and all four are recorded - including the ones that come back empty.

**Four gating checks.**

- *Lineage identity.* No `source_lineage_id` of this corpus may appear in R7-T7's, except the two
  regression lineages the protocol declares as re-used in advance. The whitelist is read from the
  protocol rather than written here, so the exemption cannot widen without editing the
  predeclaration.
- *Content identity.* No `sha256` of this corpus may appear in R7-T7's. This catches a shared
  recording that the two builds happened to name differently, which lineage comparison alone
  never could.
- *FaceForensics++ reachability.* Five swap families here are built on FaceForensics++ originals,
  and R7-T7 held forty FaceForensics++ swaps and thirty-nine reals. A new *method* over an old
  *original* is a new file over an old recording. Every FaceForensics++-derived lineage and every
  recorded member name is re-screened against the ids read out of R7-T7's own corpus - the same
  set the build refused on, checked again here against the finished artifact, because a build-time
  filter that silently stopped working would leave no other trace.
- *Within-corpus split integrity.* `corpus.leakage_findings`, unchanged, over the finished splits.

**One reporting check.** A difference-hash pass compares this corpus's base clips against R7-T7's,
and across its own split boundary. A difference hash survives rescaling, recompression and
frame-rate change, so it finds a shared recording that no identifier declared. It reports pairs
for a reader to judge rather than failing the run: two independent recordings of one event can be
perceptually close and that is not a defect. Identity is what gates.

Private lineages are compared like any other and are reported by lineage id and digest only. No
filename, no path and no container detail from them reaches the artifact.
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
from eval.leakage_check import FRAME_SAMPLES, NEAR_DUPLICATE_BITS, frame_hashes

# The namespace the FaceForensics++-derived swap families share. Its lineage key is the target
# original id, which is the thing that has to stay disjoint from R7-T7.
FFPP_NAMESPACE = "ffpp-orig"

# A difference hash of a near-uniform frame - a fade-in, a black leader, a title card - carries
# almost no information, and every such frame in the corpus collides with every other at Hamming
# distance 0. The first run of this check returned two hundred pairs, of which the overwhelming
# majority were one black first frame matching another, and the cap on the reported list then hid
# the pairs that meant something behind them. Frames whose hash is nearly all zeros or nearly all
# ones are therefore not compared. The threshold is generous: a real frame of a real scene lands
# far inside it, and this only removes hashes that could not distinguish anything anyway.
MIN_INFORMATIVE_BITS = 8
MAX_INFORMATIVE_BITS = 56


def _informative(value: int) -> bool:
    """Whether a difference hash carries enough set bits to identify anything."""
    return MIN_INFORMATIVE_BITS <= bin(value).count("1") <= MAX_INFORMATIVE_BITS


def _ffpp_ids(items: list[CorpusItem]) -> set[str]:
    """Every FaceForensics++ original id an R7-T7 corpus record touches."""
    found: set[str] = set()
    for item in items:
        lineage = item.source_lineage_id
        if not lineage.startswith("ffpp_"):
            continue
        for token in lineage.split(":", 1)[1].split("_"):
            if token.isdigit():
                found.add(token.zfill(3))
    return found


def identity_checks(
    items: list[CorpusItem],
    r7t7: list[CorpusItem],
    declared_reuse: set[str],
) -> dict:
    """The four gating questions, each answered over the finished artifacts."""
    r7t7_lineages = {item.source_lineage_id for item in r7t7}
    r7t7_digests = {item.sha256: item.clip_id for item in r7t7}
    refused_ffpp = _ffpp_ids(r7t7)

    shared_lineages = sorted(
        {
            item.source_lineage_id
            for item in items
            if item.source_lineage_id in r7t7_lineages
            and item.source_lineage_id not in declared_reuse
        }
    )
    reused_as_declared = sorted(
        {
            item.source_lineage_id
            for item in items
            if item.source_lineage_id in declared_reuse
        }
    )

    shared_digests = [
        {
            "sha256": item.sha256,
            "r7t8_lineage": item.source_lineage_id,
            "r7t7_clip": r7t7_digests[item.sha256],
        }
        for item in sorted(items, key=lambda entry: entry.clip_id)
        if item.sha256 in r7t7_digests and item.source_lineage_id not in declared_reuse
    ]

    ffpp_violations = []
    for item in items:
        if item.source_lineage_id.startswith(f"{FFPP_NAMESPACE}:"):
            key = item.source_lineage_id.split(":", 1)[1]
            if key.zfill(3) in refused_ffpp:
                ffpp_violations.append(
                    {
                        "lineage": item.source_lineage_id,
                        "clip_id": item.clip_id,
                        "reason": "target original was used by R7-T7",
                    }
                )
                continue
        # The recorded member name carries both halves of a `<target>_<source>` pair, so the
        # source original is screened here even though the lineage is keyed on the target.
        if item.is_base and "#" in item.source and item.family.startswith("faceswap_"):
            stem = Path(item.source.rsplit(":", 1)[-1]).stem
            parts = stem.split("_")
            if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
                for part in parts[:2]:
                    if part.zfill(3) in refused_ffpp:
                        ffpp_violations.append(
                            {
                                "lineage": item.source_lineage_id,
                                "clip_id": item.clip_id,
                                "reason": f"member names FaceForensics++ original {part}, "
                                          "which R7-T7 used",
                            }
                        )

    return {
        "r7t7_lineages": len(r7t7_lineages),
        "r7t8_lineages": len({item.source_lineage_id for item in items}),
        "declared_reuse_whitelist": sorted(declared_reuse),
        "reused_as_declared": reused_as_declared,
        "shared_lineage_ids": shared_lineages,
        "shared_sha256": shared_digests,
        "ffpp_original_ids_used_by_r7t7": len(refused_ffpp),
        "ffpp_reachability_violations": ffpp_violations,
        "clean": not (shared_lineages or shared_digests or ffpp_violations),
    }


def _resolve(item: CorpusItem, corpus_dir: Path) -> Path:
    path = Path(item.path)
    return path if path.is_absolute() else corpus_dir / item.path


def _hash_bases(items: list[CorpusItem], corpus_dir: Path, label: str) -> dict[str, list[int]]:
    """Difference hashes for every base clip, keyed by clip id."""
    table: dict[str, list[int]] = {}
    bases = [item for item in items if item.is_base]
    for index, item in enumerate(bases, start=1):
        table[item.clip_id] = frame_hashes(_resolve(item, corpus_dir))
        if index % 100 == 0:
            print(f"  {label}: hashed {index}/{len(bases)}", file=sys.stderr)
    return table


def _near_pairs(
    left: dict[str, list[int]],
    right: dict[str, list[int]],
    left_by_id: dict[str, CorpusItem],
    right_by_id: dict[str, CorpusItem],
    limit: int = 500,
) -> list[dict]:
    """Pairs of clips whose closest sampled frames are within the dHash threshold."""
    import numpy as np

    popcount = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)

    def matrix(table: dict[str, list[int]]) -> tuple[list[str], "np.ndarray"]:
        ids, rows = [], []
        for clip_id, hashes in table.items():
            for value in hashes:
                if not _informative(value):
                    continue
                ids.append(clip_id)
                rows.append(value)
        if not rows:
            return [], np.zeros((0, 8), dtype=np.uint8)
        packed = np.frombuffer(
            b"".join(int(value).to_bytes(8, "big") for value in rows), dtype=np.uint8
        ).reshape(-1, 8)
        return ids, packed

    left_ids, left_rows = matrix(left)
    right_ids, right_rows = matrix(right)
    if not left_ids or not right_ids:
        return []

    best: dict[tuple[str, str], int] = {}
    for start in range(0, len(left_rows), 256):
        block = left_rows[start : start + 256]
        distances = popcount[
            np.bitwise_xor(block[:, None, :], right_rows[None, :, :])
        ].sum(axis=2)
        hits = np.argwhere(distances <= NEAR_DUPLICATE_BITS)
        for row, column in hits:
            key = (left_ids[start + int(row)], right_ids[int(column)])
            distance = int(distances[row, column])
            if best.get(key, 65) > distance:
                best[key] = distance

    def describe(clip_id: str, table: dict[str, CorpusItem]) -> dict:
        item = table[clip_id]
        return {
            "clip_id": None if item.private else clip_id,
            "lineage": item.source_lineage_id,
            "split": item.split,
            "private": item.private,
        }

    pairs = [
        {
            "left": describe(left_id, left_by_id),
            "right": describe(right_id, right_by_id),
            "hamming": distance,
        }
        for (left_id, right_id), distance in sorted(best.items(), key=lambda kv: kv[1])
    ]
    return pairs[:limit]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="independence_check_r7t8",
        description="Gate the R7-T8 corpus on independence from R7-T7 and from itself.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--r7t7-corpus", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--skip-perceptual",
        action="store_true",
        help="run the gating identity checks only; the perceptual pass decodes every base clip",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = read_corpus(args.corpus)
    r7t7 = read_corpus(args.r7t7_corpus)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    declared_reuse = set(
        protocol["independence_requirement"]["declared_reuse_of_regression_lineages"]["lineages"]
    )

    identity = identity_checks(items, r7t7, declared_reuse)
    within = leakage_findings(items)

    perceptual_cross_task: dict = {"ran": False}
    perceptual_within: dict = {"ran": False}
    if not args.skip_perceptual:
        corpus_dir = args.corpus.parent
        r7t7_dir = args.r7t7_corpus.parent
        print("perceptual pass: hashing base clips", file=sys.stderr)
        here = _hash_bases(items, corpus_dir, "r7t8")
        there = _hash_bases(r7t7, r7t7_dir, "r7t7")
        by_id = {item.clip_id: item for item in items}
        r7t7_by_id = {item.clip_id: item for item in r7t7}
        unreadable = sorted(clip for clip, values in here.items() if not values)
        pairs = _near_pairs(here, there, by_id, r7t7_by_id)
        perceptual_cross_task = {
            "ran": True,
            "frames_per_clip": FRAME_SAMPLES,
            "near_duplicate_bits": NEAR_DUPLICATE_BITS,
            "uninformative_frames_excluded": (
                f"hashes with fewer than {MIN_INFORMATIVE_BITS} or more than "
                f"{MAX_INFORMATIVE_BITS} set bits are not compared; a near-uniform frame "
                "collides with every other one and tells a reader nothing"
            ),
            "r7t8_base_clips_hashed": sum(1 for v in here.values() if v),
            "r7t7_base_clips_hashed": sum(1 for v in there.values() if v),
            "r7t8_base_clips_unreadable": [
                clip for clip in unreadable if not by_id[clip].private
            ],
            "pairs": pairs,
            "pair_count": len(pairs),
            "gating": False,
            "note": (
                "Reported, not gated. A small Hamming distance is evidence for a reader to "
                "weigh, not proof of a shared recording."
            ),
        }
        calibration = {k: v for k, v in here.items() if by_id[k].split == SPLIT_CALIBRATION}
        validation = {k: v for k, v in here.items() if by_id[k].split == SPLIT_EVALUATION}
        within_pairs = _near_pairs(validation, calibration, by_id, by_id)
        perceptual_within = {
            "ran": True,
            "pairs": within_pairs,
            "pair_count": len(within_pairs),
            "gating": False,
        }

    leaked = not identity["clean"] or within["leaked"]
    payload = {
        "schema_version": "r7-t8-independence-1",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(args.corpus),
        "r7t7_corpus": str(args.r7t7_corpus),
        "protocol": str(args.protocol),
        "cross_task_identity": identity,
        "within_corpus_split_integrity": within,
        "cross_task_perceptual": perceptual_cross_task,
        "within_corpus_perceptual": perceptual_within,
        "gating_checks_passed": not leaked,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"cross-task identity clean: {identity['clean']}")
    print(f"  shared lineages: {len(identity['shared_lineage_ids'])}")
    print(f"  shared digests: {len(identity['shared_sha256'])}")
    print(f"  ffpp reachability violations: {len(identity['ffpp_reachability_violations'])}")
    print(f"  declared re-use present: {identity['reused_as_declared']}")
    print(f"within-corpus split leaked: {within['leaked']}")
    if perceptual_cross_task["ran"]:
        print(f"perceptual cross-task pairs (reported, not gating): "
              f"{perceptual_cross_task['pair_count']}")
        print(f"perceptual within-corpus pairs (reported, not gating): "
              f"{perceptual_within['pair_count']}")
    print(f"GATING CHECKS {'PASSED' if not leaked else 'FAILED'}")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
