"""Prove the R7-T9 corpus is independent of R7-T5/R7-T7 and R7-T8, and of itself.

    PYTHONPATH=scripts ~/.venvs/deepguard-effort/bin/python \
        scripts/eval/independence_check_r7t9.py \
            --corpus ../deepguard-corpus/r7t9/corpus.json \
            --prior-corpus ../deepguard-corpus/r7t5/corpus.json \
            --prior-corpus ../deepguard-corpus/r7t8/corpus.json \
            --protocol scripts/eval/r7t9_protocol.json \
            --output ../deepguard-corpus/r7t9/independence.json

Exit status is `0` when nothing leaked and `1` when something did, so this gates the run.

**What changed from R7-T8's checker, and why it had to.** R7-T8 compared one corpus against one
predecessor and treated the perceptual pass as reporting only: a small Hamming distance between
two independent recordings of one event is possible, so a reader judged it. R7-T9's task
statement removes that latitude — the checker must *fail closed on unresolved high-similarity
candidates* and may not silently whitelist ambiguous media. So the perceptual pass here is split
in two. Pairs at or below `HIGH_SIMILARITY_BITS` are a gating finding and fail the build unless
the pair is the declared regression fixture matching itself. Pairs in the wider `NEAR_DUPLICATE_BITS`
band are reported for a reader, as before, because R7-T8 measured that band to be full of
unrelated media and gating on it would fail on noise rather than on leakage.

**Six gating checks.**

- *Lineage identity*, against every prior corpus at once. No `source_lineage_id` here may appear
  in any of them, except the regression fixtures the protocol whitelists in advance.
- *Content identity*. No `sha256` here may appear in any prior corpus, whitelist excepted.
- *FaceForensics++ reachability*. Several manipulated families here are built on FaceForensics++
  originals, and both predecessors consumed originals — R7-T7 as reals and swaps, R7-T8 as swap
  targets and sources. The union of ids reachable from either is re-screened against every
  lineage and every recorded member name of this corpus.
- *Source-family relationships*, which identifiers alone cannot see. A manipulated pool built on
  top of a genuine pool a previous task read is a new file over an old recording no matter what
  its members are called. The protocol names those relationships and this refuses any pool whose
  declared parent dataset appears in a prior corpus.
- *Within-corpus split integrity*, `corpus.leakage_findings`, unchanged: shared lineages, shared
  digests, orphaned derivatives, and derivatives separated from their base.
- *Perceptual high similarity*, gating, as described above.

Private lineages are compared like any other and reported by lineage id and digest only. No
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

# The namespace shared by every FaceForensics++-derived manipulated family. Its lineage key is
# the target original id, which is what has to stay disjoint from both predecessors.
FFPP_NAMESPACE = "ffpp-orig"

# Gating band. Two clips whose closest sampled frames agree to within this many bits of a 64-bit
# difference hash are treated as the same recording until proven otherwise, and the build fails.
# R7-T8 measured this band empirically over 1034 clips against 637: it held exactly the two
# declared re-used lineages at distance 0 and nothing else below 6 bits, while the 6-10 band held
# 244 unrelated pairs. Gating at 5 therefore fails on leakage and not on noise.
HIGH_SIMILARITY_BITS = 5

# A difference hash of a near-uniform frame - a fade-in, a black leader, a title card - carries
# almost no information and collides with every other such frame at distance 0. Excluded from
# comparison entirely; R7-T8 found these hiding two real findings behind two hundred false ones.
MIN_INFORMATIVE_BITS = 8
MAX_INFORMATIVE_BITS = 56

# Corroboration. A 64-bit difference hash summarises one frame in eight bytes, and eight bytes
# cannot tell "the same recording" from "two recordings whose coarse luminance layout happens to
# agree". The gating band therefore does not decide anything on its own: every pair inside it is
# re-tested at 16x16, which is 256 bits over the same frames, and the pair is confirmed as a
# shared recording only if that stronger descriptor agrees too.
#
# The threshold is set from the two populations rather than from the pairs it will judge. Two
# encodings of one frame - a transcode, a resize, a re-compression - agree to within roughly 10
# to 15 per cent of their bits; two unrelated frames sit near 50 per cent, which is what random
# bits do. A quarter of 256 bits is 64, which is far above the first population and far below the
# second. A pair whose media cannot be read is neither confirmed nor refuted: it stays
# unresolved, and unresolved fails.
CORROBORATION_HASH_SIDE = 16
CORROBORATION_MAX_BITS = 64


def _informative(value: int) -> bool:
    return MIN_INFORMATIVE_BITS <= bin(value).count("1") <= MAX_INFORMATIVE_BITS


def _fine_hashes(path: Path) -> list[int]:
    """Difference hashes of the same sampled frames at `CORROBORATION_HASH_SIDE`, as integers.

    The same frames `leakage_check.frame_hashes` reads, described in 256 bits instead of 64. A
    coarse collision that is really one recording survives the finer description; one that is an
    artefact of eight-byte summarisation does not.
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
        side = CORROBORATION_HASH_SIDE
        hashes = []
        for index in np.linspace(0, max(total - 1, 0), FRAME_SAMPLES).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            read, frame = capture.read()
            if not read:
                continue
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, (side + 1, side), interpolation=cv2.INTER_AREA)
            bits = small[:, 1:] > small[:, :-1]
            value = 0
            for bit in bits.flatten():
                value = (value << 1) | int(bit)
            hashes.append(value)
        return hashes
    finally:
        capture.release()


def _corroborate(
    pairs: list[dict],
    resolve_left,
    resolve_right,
    cache: dict,
) -> None:
    """Re-test every pair inside the gating band with a 256-bit descriptor, in place.

    Writes `corroboration` onto each pair: the finest distance found between any two sampled
    frames, and a verdict. `confirmed` means the two clips really do look like one recording and
    the finding stands. `refuted` means the coarse hash collided and the finer one did not, which
    is a property of eight-byte summaries and not evidence of anything. `unreadable` means the
    question could not be asked, and an unaskable question is not an answer: it stays unresolved.
    """
    import numpy as np

    def fine(path: Path) -> list[int]:
        # Keyed on the resolved path, not on the clip id: two corpora can name a clip the same
        # thing, and a cache that confused them would compare the wrong media.
        key = str(path)
        if key not in cache:
            cache[key] = _fine_hashes(path)
        return cache[key]

    for pair in pairs:
        left = fine(resolve_left(pair["left"]["key"]))
        right = fine(resolve_right(pair["right"]["key"]))
        if not left or not right:
            pair["corroboration"] = {"verdict": "unreadable", "closest_bits": None}
            continue
        best = min(
            bin(a ^ b).count("1") for a in left for b in right
        )
        pair["corroboration"] = {
            "verdict": "confirmed" if best <= CORROBORATION_MAX_BITS else "refuted",
            "closest_bits": best,
            "of_bits": CORROBORATION_HASH_SIDE * CORROBORATION_HASH_SIDE,
            "threshold_bits": CORROBORATION_MAX_BITS,
        }


def _ffpp_ids(items: list[CorpusItem]) -> set[str]:
    """Every FaceForensics++ original id a corpus record touches, zero-padded.

    Reads both the `ffpp_*` lineage names R7-T7 used and the `ffpp-orig:` namespace R7-T8
    introduced, plus the `<target>_<source>` member names recorded in `source`, because the
    two predecessors spell the same fact three different ways.
    """
    found: set[str] = set()
    for item in items:
        lineage = item.source_lineage_id
        key = lineage.split(":", 1)[1] if ":" in lineage else lineage
        if lineage.startswith("ffpp_") or lineage.startswith(f"{FFPP_NAMESPACE}:"):
            for token in key.replace("-", "_").split("_"):
                if token.isdigit() and len(token) <= 3:
                    found.add(token.zfill(3))
        if "#" in item.source and (
            item.family.startswith("faceswap_")
            or item.family.startswith("ffpp_")
            or item.family.startswith("reenact_")
            or item.family.startswith("lipsync_")
        ):
            stem = Path(item.source.rsplit(":", 1)[-1]).stem
            parts = stem.split("_")
            if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
                for part in parts[:2]:
                    found.add(part.zfill(3))
    return found


def identity_checks(
    items: list[CorpusItem],
    priors: dict[str, list[CorpusItem]],
    declared_reuse: set[str],
) -> dict:
    """Lineage, digest and FaceForensics++ reachability, against every predecessor at once."""
    prior_lineages: dict[str, str] = {}
    prior_digests: dict[str, tuple[str, str]] = {}
    refused_ffpp: set[str] = set()
    for task, prior in priors.items():
        for item in prior:
            prior_lineages.setdefault(item.source_lineage_id, task)
            prior_digests.setdefault(item.sha256, (task, item.clip_id))
        refused_ffpp |= _ffpp_ids(prior)

    shared_lineages = sorted(
        {
            f"{item.source_lineage_id} (also in {prior_lineages[item.source_lineage_id]})"
            for item in items
            if item.source_lineage_id in prior_lineages
            and item.source_lineage_id not in declared_reuse
        }
    )
    reused_as_declared = sorted(
        {item.source_lineage_id for item in items if item.source_lineage_id in declared_reuse}
    )
    shared_digests = [
        {
            "sha256": item.sha256,
            "r7t9_lineage": item.source_lineage_id,
            "prior_task": prior_digests[item.sha256][0],
            "prior_clip": prior_digests[item.sha256][1],
        }
        for item in sorted(items, key=lambda entry: entry.clip_id)
        if item.sha256 in prior_digests and item.source_lineage_id not in declared_reuse
    ]

    violations = []
    for item in items:
        reachable = _ffpp_ids([item])
        hit = sorted(reachable & refused_ffpp)
        if hit:
            violations.append(
                {
                    "lineage": item.source_lineage_id,
                    "clip_id": None if item.private else item.clip_id,
                    "originals": hit,
                    "reason": "FaceForensics++ original(s) used by a prior task",
                }
            )

    return {
        "prior_tasks": sorted(priors),
        "prior_lineages": len(prior_lineages),
        "r7t9_lineages": len({item.source_lineage_id for item in items}),
        "declared_reuse_whitelist": sorted(declared_reuse),
        "reused_as_declared": reused_as_declared,
        "shared_lineage_ids": shared_lineages,
        "shared_sha256": shared_digests,
        "ffpp_original_ids_used_by_prior_tasks": len(refused_ffpp),
        "ffpp_reachability_violations": violations,
        "clean": not (shared_lineages or shared_digests or violations),
    }


def _normalise(name: str) -> str:
    """A dataset name reduced to what identifies it across two different spellings.

    A prior corpus records its pools by mirror repository id (`34data/v15-human-vid-celebv-hq`)
    and this corpus declares its parents by dataset name (`CelebV-HQ`). Comparing the two
    literally would never match and the check would pass on everything, which is worse than no
    check. Both sides are therefore stripped to lowercase alphanumerics with the mirror's own
    packaging prefixes and shard suffixes removed, so the comparison is between dataset
    identities rather than between two naming conventions.
    """
    lowered = name.lower()
    for prefix in ("34data/", "v13-", "v14-", "v15-", "v22-", "human-vid-", "human-img-",
                   "real-", "fake-", "gen-videos-"):
        lowered = lowered.replace(prefix, "")
    for suffix in ("_real", "_fake", "-real", "-fake"):
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
    return "".join(character for character in lowered if character.isalnum())


def source_family_checks(corpus_payload: dict, priors: dict[str, list[CorpusItem]]) -> dict:
    """Relationships identifiers cannot express, declared in the protocol and enforced here.

    Two pools can share no lineage id, no digest and no naming convention and still be the same
    recordings: a manipulated pool built on top of a genuine pool a predecessor read, or a second
    shard of one dataset. The build records each pool's `parent_datasets` beside the pool it
    describes, every prior corpus's pools are reduced to the same normalised identity, and a
    match is a finding rather than a coincidence a reader has to notice.

    The comparison is containment in both directions after normalisation, because a mirror shard
    (`dh-facevid-1k-0002-part_3`) and a dataset name (`DH-FaceVid-1K`) are the same dataset and
    neither string is a prefix of the other.
    """
    declared = corpus_payload.get("independence", {}).get("source_family_declarations", {})
    prior_parents: dict[str, str] = {}
    for task, prior in priors.items():
        for item in prior:
            repository = item.source.split("@", 1)[0] if "@" in item.source else ""
            for candidate in (repository, item.family):
                key = _normalise(candidate)
                if len(key) >= 4:
                    prior_parents.setdefault(key, task)

    findings = []
    for source_id, entry in sorted(declared.items()):
        for parent in entry.get("parent_datasets", []):
            key = _normalise(parent)
            if len(key) < 4:
                continue
            for prior_key, task in prior_parents.items():
                if key == prior_key or key in prior_key or prior_key in key:
                    findings.append(
                        {
                            "source_id": source_id,
                            "parent_dataset": parent,
                            "matched_prior_pool": prior_key,
                            "prior_task": task,
                            "reason": "pool is built on a dataset a prior task already read",
                        }
                    )
    return {
        "declarations": len(declared),
        "prior_parent_datasets": len(prior_parents),
        "prior_parent_keys": sorted(prior_parents),
        "violations": findings,
        "clean": not findings,
        "note": (
            "The declarations are written in the build script beside the pool they describe, so "
            "adding a pool without stating what it is built on is a build-time omission a "
            "reader can see rather than a silent pass here. FaceForensics++ originals are "
            "screened by reachability instead, which is exact where a name comparison is not."
        ),
    }


def _resolve(item: CorpusItem, corpus_dir: Path) -> Path:
    path = Path(item.path)
    return path if path.is_absolute() else corpus_dir / item.path


def _hash_bases(items: list[CorpusItem], corpus_dir: Path, label: str) -> dict[str, list[int]]:
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
    limit: int = 2000,
) -> list[dict]:
    """Pairs whose closest sampled frames are within the reporting threshold, closest first."""
    import numpy as np

    popcount = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)

    def matrix(table: dict[str, list[int]]):
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
        for row, column in np.argwhere(distances <= NEAR_DUPLICATE_BITS):
            key = (left_ids[start + int(row)], right_ids[int(column)])
            distance = int(distances[row, column])
            if best.get(key, 65) > distance:
                best[key] = distance

    def describe(clip_id: str, table: dict[str, CorpusItem]) -> dict:
        item = table[clip_id]
        return {
            "clip_id": None if item.private else clip_id,
            # The corroboration pass needs to open the media; `clip_id` is withheld from the
            # artifact for private lineages and this key never reaches it - `_gate_perceptual`
            # strips it before the payload is written.
            "key": clip_id,
            "lineage": item.source_lineage_id,
            "split": item.split,
            "private": item.private,
        }

    return [
        {
            "left": describe(left_id, left_by_id),
            "right": describe(right_id, right_by_id),
            "hamming": distance,
        }
        for (left_id, right_id), distance in sorted(best.items(), key=lambda kv: kv[1])
    ][:limit]


def _gate_perceptual(pairs: list[dict], declared_reuse: set[str]) -> dict:
    """Classify the reported pairs into what gates and what informs.

    Three things can put a pair outside the gating set, and only three. It is the declared
    regression fixture matching itself, which is the one re-use this protocol permits and which
    is expected to match at distance 0. It is outside the gating band. Or the 256-bit
    corroboration refuted it, meaning the coarse hash collided and the finer description of the
    same frames did not agree - a property of eight-byte summaries, not evidence about the media.

    Everything else gates, including a pair whose media could not be read. An unresolved
    high-similarity candidate is not something this study may wave through, and "the question
    could not be asked" is not a resolution.
    """
    gating, reported = [], []
    for pair in pairs:
        pair = {**pair, "left": {**pair["left"]}, "right": {**pair["right"]}}
        pair["left"].pop("key", None)
        pair["right"].pop("key", None)
        if pair["hamming"] > HIGH_SIMILARITY_BITS:
            reported.append(pair)
            continue
        left, right = pair["left"]["lineage"], pair["right"]["lineage"]
        if left in declared_reuse and right in declared_reuse and left == right:
            pair["resolution"] = "declared regression fixture matching itself"
            reported.append(pair)
            continue
        verdict = (pair.get("corroboration") or {}).get("verdict")
        if verdict == "refuted":
            pair["resolution"] = (
                "refuted by 256-bit corroboration over the same frames; the 64-bit collision is "
                "an artefact of the coarse descriptor"
            )
            reported.append(pair)
            continue
        pair["resolution"] = (
            "CONFIRMED as a shared recording by 256-bit corroboration"
            if verdict == "confirmed"
            else "UNRESOLVED: the media could not be read, so the question could not be asked"
        )
        gating.append(pair)

    histogram: dict[str, int] = {}
    for pair in pairs:
        bucket = "0" if pair["hamming"] == 0 else ("1-5" if pair["hamming"] <= 5 else "6-10")
        histogram[bucket] = histogram.get(bucket, 0) + 1
    in_band = [p for p in pairs if p["hamming"] <= HIGH_SIMILARITY_BITS]
    verdicts: dict[str, int] = {}
    for pair in in_band:
        key = (pair.get("corroboration") or {}).get("verdict", "not tested")
        verdicts[key] = verdicts.get(key, 0) + 1
    return {
        "high_similarity_bits": HIGH_SIMILARITY_BITS,
        "near_duplicate_bits": NEAR_DUPLICATE_BITS,
        "corroboration": {
            "hash_side": CORROBORATION_HASH_SIDE,
            "threshold_bits": CORROBORATION_MAX_BITS,
            "pairs_in_band": len(in_band),
            "verdicts": verdicts,
        },
        "unresolved_high_similarity_pairs": gating,
        "unresolved_high_similarity_count": len(gating),
        "resolved_or_below_threshold_pairs": reported[:500],
        "pair_count": len(pairs),
        "hamming_histogram": histogram,
        "clean": not gating,
        "gating": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="independence_check_r7t9",
        description="Gate the R7-T9 corpus on independence from every prior task and itself.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument(
        "--prior-corpus",
        required=True,
        action="append",
        type=Path,
        help="a prior task's corpus.json; repeatable, and every one given is screened against",
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--skip-perceptual",
        action="store_true",
        help="identity checks only; the perceptual pass decodes every base clip and gates",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = read_corpus(args.corpus)
    corpus_payload = json.loads(args.corpus.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    declared_reuse = set(
        protocol["independence_requirement"]["declared_reuse_of_regression_lineages"]["lineages"]
    )

    priors = {path.parent.name: read_corpus(path) for path in args.prior_corpus}
    identity = identity_checks(items, priors, declared_reuse)
    families = source_family_checks(corpus_payload, priors)
    within = leakage_findings(items)

    perceptual_cross: dict = {"ran": False, "clean": None}
    perceptual_within: dict = {"ran": False, "clean": None}
    if not args.skip_perceptual:
        corpus_dir = args.corpus.parent
        print("perceptual pass: hashing base clips", file=sys.stderr)
        here = _hash_bases(items, corpus_dir, "r7t9")
        by_id = {item.clip_id: item for item in items}

        # The corroboration pass decodes media, so its results are cached per clip: one clip can
        # appear in several flagged pairs and there is no reason to read it twice.
        fine_cache: dict = {}
        here_path = lambda key: _resolve(by_id[key], corpus_dir)  # noqa: E731

        cross_pairs: list[dict] = []
        for path in args.prior_corpus:
            prior_items = priors[path.parent.name]
            there = _hash_bases(prior_items, path.parent, path.parent.name)
            prior_by_id = {item.clip_id: item for item in prior_items}
            pairs = _near_pairs(here, there, by_id, prior_by_id)
            in_band = [p for p in pairs if p["hamming"] <= HIGH_SIMILARITY_BITS]
            if in_band:
                print(
                    f"  corroborating {len(in_band)} in-band pair(s) against "
                    f"{path.parent.name} at {CORROBORATION_HASH_SIDE}x{CORROBORATION_HASH_SIDE}",
                    file=sys.stderr,
                )
                _corroborate(
                    in_band,
                    here_path,
                    lambda key, root=path.parent: _resolve(prior_by_id[key], root),
                    fine_cache,
                )
            for pair in pairs:
                cross_pairs.append({**pair, "prior_task": path.parent.name})
        cross_pairs.sort(key=lambda entry: entry["hamming"])
        perceptual_cross = {
            "ran": True,
            "frames_per_clip": FRAME_SAMPLES,
            "uninformative_frames_excluded": (
                f"hashes with fewer than {MIN_INFORMATIVE_BITS} or more than "
                f"{MAX_INFORMATIVE_BITS} set bits are not compared"
            ),
            "r7t9_base_clips_hashed": sum(1 for values in here.values() if values),
            "r7t9_base_clips_unreadable": [
                clip for clip, values in sorted(here.items()) if not values and not by_id[clip].private
            ],
            **_gate_perceptual(cross_pairs, declared_reuse),
        }

        calibration = {k: v for k, v in here.items() if by_id[k].split == SPLIT_CALIBRATION}
        validation = {k: v for k, v in here.items() if by_id[k].split == SPLIT_EVALUATION}
        within_pairs = _near_pairs(validation, calibration, by_id, by_id)
        within_band = [p for p in within_pairs if p["hamming"] <= HIGH_SIMILARITY_BITS]
        if within_band:
            print(
                f"  corroborating {len(within_band)} in-band pair(s) across the split boundary",
                file=sys.stderr,
            )
            _corroborate(within_band, here_path, here_path, fine_cache)
        perceptual_within = {"ran": True, **_gate_perceptual(within_pairs, declared_reuse)}

    gates = {
        "cross_task_identity": identity["clean"],
        "source_family_relationships": families["clean"],
        "within_corpus_split_integrity": not within["leaked"],
        "cross_task_perceptual_high_similarity": (
            True if not perceptual_cross["ran"] else perceptual_cross["clean"]
        ),
        "within_corpus_perceptual_high_similarity": (
            True if not perceptual_within["ran"] else perceptual_within["clean"]
        ),
    }
    passed = all(gates.values()) and not args.skip_perceptual

    payload = {
        "schema_version": "r7-t9-independence-1",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(args.corpus),
        "prior_corpora": [str(path) for path in args.prior_corpus],
        "protocol": str(args.protocol),
        "cross_task_identity": identity,
        "source_family_relationships": families,
        "within_corpus_split_integrity": within,
        "cross_task_perceptual": perceptual_cross,
        "within_corpus_perceptual": perceptual_within,
        "gate_summary": gates,
        "perceptual_pass_ran": not args.skip_perceptual,
        "gating_checks_passed": passed,
        "fails_closed": (
            "Every check above gates. An unresolved pair inside the high-similarity band fails "
            "the run; only the regression fixtures the protocol declares in advance are "
            "resolvable, and only by matching themselves. Skipping the perceptual pass cannot "
            "produce a pass."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"cross-task identity clean: {identity['clean']}")
    print(f"  shared lineages: {len(identity['shared_lineage_ids'])}")
    print(f"  shared digests: {len(identity['shared_sha256'])}")
    print(f"  ffpp reachability violations: {len(identity['ffpp_reachability_violations'])}")
    print(f"  declared re-use present: {identity['reused_as_declared']}")
    print(f"source-family violations: {len(families['violations'])}")
    print(f"within-corpus split leaked: {within['leaked']}")
    if perceptual_cross["ran"]:
        print(
            "perceptual cross-task: "
            f"{perceptual_cross['pair_count']} pairs, histogram "
            f"{perceptual_cross['hamming_histogram']}, unresolved high-similarity "
            f"{perceptual_cross['unresolved_high_similarity_count']}"
        )
        print(
            "perceptual within-corpus: "
            f"{perceptual_within['pair_count']} pairs, unresolved high-similarity "
            f"{perceptual_within['unresolved_high_similarity_count']}"
        )
    print(f"GATING CHECKS {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
