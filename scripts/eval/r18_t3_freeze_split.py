"""R18-T3: freeze the R18 verified dataset and split it DEV / VAL / LOCKED_TEST without leakage.

    PYTHONPATH=scripts python3 scripts/eval/r18_t3_freeze_split.py split \
        --anchors scripts/eval/specs/r18_t2_verified_anchors.json \
        --adjudications scripts/eval/specs/r18_t3_perceptual_adjudications.json
    PYTHONPATH=scripts python3 scripts/eval/r18_t3_freeze_split.py audit-leakage \
        --anchors scripts/eval/specs/r18_t2_verified_anchors.json \
        --adjudications scripts/eval/specs/r18_t3_perceptual_adjudications.json \
        --corpus ~/deepguard-corpus
    PYTHONPATH=scripts python3 scripts/eval/r18_t3_freeze_split.py freeze \
        --anchors scripts/eval/specs/r18_t2_verified_anchors.json \
        --adjudications scripts/eval/specs/r18_t3_perceptual_adjudications.json \
        --freeze-timestamp 2026-10-21T16:00:00Z \
        --freeze-out scripts/eval/specs/r18_t3_freeze.json \
        --manifest-out scripts/eval/specs/r18_t3_split_manifest.csv
    PYTHONPATH=scripts python3 scripts/eval/r18_t3_freeze_split.py verify \
        --freeze scripts/eval/specs/r18_t3_freeze.json \
        --manifest scripts/eval/specs/r18_t3_split_manifest.csv

Applies `r18_t3_implementation_plan.md` (rev 3) and decides nothing beyond it.

The partition unit is a connected component of clips that share a `source_cluster_id`, a known
(normalised, non-null) `subject_id`, a SHA-256, or an adjudicated SHARED_SOURCE pair. Units are
never split. A seeded local search assigns whole units to DEV (~50 %), VAL (~20 %) and
LOCKED_TEST (~30 %): the Tier 1 invariants (no SHA, cluster or known-subject overlap; all seven AI
generator families in every split) are hard, while case share, AI/REAL, face presence, REAL
capture domain and source collection are soft, proportional targets. No cluster is pinned to a
split in advance. `source_collection_id` is a soft balance axis only: collections may span splits.

The perceptual audit hashes eight evenly spaced frames per clip (64-bit dHash on a 9x8 grayscale
copy). Every pair of clips from different source clusters whose closest frame pair is within
10 bits is a near-duplicate candidate, recorded in the tracked adjudication registry. The
candidate universe does not depend on the split, so a SHARED_SOURCE merge and re-split never
creates or removes a candidate. Only CROSS_SPLIT candidates must be adjudicated; the freeze
refuses while any of them is UNRESOLVED. SAME_SPLIT candidates are reported as diagnostics.

The freeze is a pure function of the anchors, the registry, the seed and the explicit
`--freeze-timestamp`: no clock is read, and a re-run rewrites both outputs byte for byte. No
detector, model, production pipeline or network is used; corpus media are only read.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from eval import r18_t1_inventory as t1

REPO = Path(__file__).resolve().parents[2]
SCHEMA_ANCHOR = "r18-t2-verified-anchors-1"
SCHEMA_FREEZE = "r18-t3-freeze-v1"
SCHEMA_ADJUDICATIONS = "r18-t3-perceptual-adjudications-1"
DATASET_NAME = "r18_eval_v1"
AUTHORITY = "r18_t3_implementation_plan.md"
DEFAULT_SEED = 20261021

VERIFIED_AI, VERIFIED_REAL = "VERIFIED_AI", "VERIFIED_REAL"
DEV, VAL, LOCKED_TEST = "DEV", "VAL", "LOCKED_TEST"
SPLITS = (DEV, VAL, LOCKED_TEST)
FRACTIONS = {DEV: 0.5, VAL: 0.2, LOCKED_TEST: 0.3}
# Plan §4.1.4: every family must appear in every split.
GENERATORS = ("Runway", "Kling", "Luma", "Hailuo/MiniMax", "Pika", "Veo", "Sora")
# Plan §4.2: reported against the plan's tolerances; never enforced at the cost of Tier 1.
AI_SHARE_TOLERANCE = (0.46, 0.54)
FACE_SHARE_TOLERANCE = (0.42, 0.50)

GOVERNANCE_NOTICE = ("LOCKED_TEST is source-cluster- and subject-isolated, but not collection- "
                     "or publisher-independent. T4/T5 tooling is restricted to DEV and VAL; "
                     "LOCKED_TEST is reserved exclusively for T6.")
LOCKED_TEST_PURPOSE = "R18-T6"

INDEPENDENT, SHARED_SOURCE, UNRESOLVED = "INDEPENDENT", "SHARED_SOURCE", "UNRESOLVED"
NOT_REQUIRED_SAME_SPLIT = "NOT_REQUIRED_SAME_SPLIT"
STATES = (INDEPENDENT, SHARED_SOURCE, UNRESOLVED, NOT_REQUIRED_SAME_SPLIT)
DECIDED = (INDEPENDENT, SHARED_SOURCE)
# Architect ruling (R18-T3 adjudication scope): every different-cluster pair is a candidate, but
# only a CROSS_SPLIT candidate must be adjudicated and can block the freeze; a SAME_SPLIT
# candidate is diagnostic and may stay NOT_REQUIRED_SAME_SPLIT.
CROSS_SPLIT, SAME_SPLIT = "CROSS_SPLIT", "SAME_SPLIT"
SCOPE_STATES = {CROSS_SPLIT: (INDEPENDENT, SHARED_SOURCE, UNRESOLVED),
                SAME_SPLIT: (INDEPENDENT, SHARED_SOURCE, NOT_REQUIRED_SAME_SPLIT)}
DISTANCE_BUCKETS = ((0, 2), (3, 5), (6, 8), (9, 10))
DHASH_SPEC = {
    "algorithm": "dHash (difference hash), 64 bits",
    "frames_per_clip": 8,
    "frame_times": "t_i = duration * (2i + 1) / 16 for i = 0..7 (evenly spaced, no endpoints)",
    "decoder": "ffmpeg -ss t -i clip -frames:v 1 (PNG pipe)",
    "preprocessing": "BGR -> grayscale (cv2.COLOR_BGR2GRAY), resize to 9x8 (cv2.INTER_AREA)",
    "bit_rule": "bit = pixel[r][c + 1] > pixel[r][c], row-major, MSB first",
    "pair_distance": "minimum Hamming distance over all 8 x 8 frame pairs",
    "candidate_threshold_bits": 10,
    "candidate_scope": "every pair of clips with different source_cluster_id (a split-"
                       "independent superset of the cross-split comparison)",
}
CANDIDATE_THRESHOLD = DHASH_SPEC["candidate_threshold_bits"]
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
NULL_SUBJECTS = frozenset({"", "unknown", "null", "none", "n/a"})

MANIFEST_COLUMNS = ("sha256", "canonical_path", "split", "ground_truth", "generator_family",
                    "capture_domain", "face_present", "source_cluster_id", "source_collection_id")

# Soft-objective weights: case share first, then the two binary strata, then the multi-valued.
WEIGHTS = {"cases": 4.0, "truth": 2.0, "face": 2.0, "gen": 1.0, "domain": 1.0,
           "collection": 1.0}
GENERATOR_PENALTY = 1e6
RESTARTS = 48
MAX_PASSES = 200


class Refusal(SystemExit):
    """A stop condition: nothing is written."""


# --- small helpers ----------------------------------------------------------------------------

def write_json(path, payload):
    t1.write_json(path, payload)


def canonical_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def validate_timestamp(text):
    """An explicit UTC ISO-8601 instant `YYYY-MM-DDTHH:MM:SSZ`; the clock is never read."""
    if not isinstance(text, str) or not TIMESTAMP_RE.match(text):
        raise Refusal(f"--freeze-timestamp must be UTC ISO 'YYYY-MM-DDTHH:MM:SSZ', got {text!r}")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise Refusal(f"--freeze-timestamp is not a valid instant: {exc}") from None
    return text


def normalise_subject(raw):
    """Known subject IDs are stripped and lower-cased; null-like values are no subject at all."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return None if text in NULL_SUBJECTS else text


def pair_id(sha_a, sha_b):
    a, b = sorted((sha_a, sha_b))
    return sha256_bytes(f"{a}:{b}".encode())[:16]


# --- cases and partition units ----------------------------------------------------------------

def load_anchors(path):
    anchor = json.loads(Path(path).read_text())
    if anchor.get("schema") != SCHEMA_ANCHOR:
        raise Refusal(f"anchor schema {anchor.get('schema')!r} != {SCHEMA_ANCHOR!r}")
    records = anchor.get("records") or []
    if not records:
        raise Refusal("anchor holds no records")
    return anchor, records


def case_view(rec):
    """The flat, freeze-relevant view of one anchored record (ground truth is copied, not made)."""
    gt = rec["ground_truth_provenance"]
    ann = rec.get("derived_annotations") or {}
    face = ann.get("face_present") or {}
    speech = ann.get("speech_talking_head") or {}
    truth = gt["ground_truth"]
    if truth not in (VERIFIED_AI, VERIFIED_REAL):
        raise Refusal(f"{rec['sha256']}: ground truth {truth!r} is not admitted")
    if not rec.get("inspectable"):
        raise Refusal(f"{rec['sha256']}: record is not inspectable")
    if not gt.get("source_cluster_id"):
        raise Refusal(f"{rec['sha256']}: no source_cluster_id")
    return {
        "sha256": rec["sha256"],
        "canonical_path": rec["canonical_path"],
        "ground_truth": truth,
        "generator_family": gt.get("generator_family"),
        "generator_model": gt.get("generator_model"),
        "capture_domain": gt.get("capture_domain"),
        "face_present": face.get("value") if face.get("status") == "DETERMINED" else None,
        "speech_talking_head": speech.get("status"),
        "source_cluster_id": gt["source_cluster_id"],
        "source_collection_id": gt.get("source_collection_id"),
        "subject_id": normalise_subject(gt.get("subject_id")),
    }


def load_cases(path):
    anchor, records = load_anchors(path)
    return anchor, sorted((case_view(r) for r in records), key=lambda c: c["sha256"])


class _Union:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_units(cases, shared_pairs=()):
    """Partition units: clips joined by cluster, known subject, SHA-256 or SHARED_SOURCE pair."""
    u = _Union(len(cases))
    first = {}
    for i, c in enumerate(cases):
        for key in (("cluster", c["source_cluster_id"]), ("subject", c["subject_id"]),
                    ("sha", c["sha256"])):
            if key[1] is None:
                continue
            if key in first:
                u.join(first[key], i)
            else:
                first[key] = i
    index = {c["sha256"]: i for i, c in enumerate(cases)}
    for a, b in shared_pairs:
        if a not in index or b not in index:
            raise Refusal(f"SHARED_SOURCE pair names a clip outside the anchor: {a} / {b}")
        u.join(index[a], index[b])
    groups = defaultdict(list)
    for i in range(len(cases)):
        groups[u.find(i)].append(i)
    units = []
    for members in groups.values():
        clusters = sorted({cases[i]["source_cluster_id"] for i in members})
        units.append({"unit_id": "+".join(clusters), "members": sorted(members)})
    return sorted(units, key=lambda x: x["unit_id"])


def features(case):
    f = Counter({("cases", None): 1})
    f[("truth", case["ground_truth"])] += 1
    f[("face", case["face_present"])] += 1
    if case["generator_family"]:
        f[("gen", case["generator_family"])] += 1
    if case["capture_domain"]:
        f[("domain", case["capture_domain"])] += 1
    f[("collection", case["source_collection_id"])] += 1
    return f


# --- seeded constrained partitioner -----------------------------------------------------------

class _State:
    """Per-split feature counts and the soft cost (weighted squared deviation from proportion)."""

    def __init__(self, unit_vecs, totals):
        self.unit_vecs, self.totals = unit_vecs, totals
        self.counts = {s: Counter() for s in SPLITS}

    def _term(self, split, key, count):
        target = FRACTIONS[split] * self.totals[key]
        return WEIGHTS[key[0]] * (count - target) ** 2 / max(self.totals[key], 1)

    def _penalty(self, split, key, count):
        return GENERATOR_PENALTY if key[0] == "gen" and count <= 0 else 0.0

    def cost(self):
        total = 0.0
        for s in SPLITS:
            for key in sorted(self.totals, key=repr):
                n = self.counts[s][key]
                total += self._term(s, key, n) + self._penalty(s, key, n)
        return total

    def move_delta(self, unit, src, dst):
        delta = 0.0
        for key, n in self.unit_vecs[unit]:
            for split, sign in ((src, -1), (dst, 1)):
                if split is None:
                    continue
                old = self.counts[split][key]
                new = old + sign * n
                delta += (self._term(split, key, new) + self._penalty(split, key, new)
                          - self._term(split, key, old) - self._penalty(split, key, old))
        return delta

    def apply(self, unit, src, dst):
        for key, n in self.unit_vecs[unit]:
            if src is not None:
                self.counts[src][key] -= n
            self.counts[dst][key] += n


def _solve_once(unit_vecs, totals, sizes, rng):
    order = list(range(len(unit_vecs)))
    rng.shuffle(order)
    order.sort(key=lambda i: -sizes[i])  # stable: the shuffle breaks ties between equal sizes
    state = _State(unit_vecs, totals)
    assign = [None] * len(unit_vecs)
    for i in order:
        # Greedy start; the generator penalty is ignored until every unit is placed.
        best = min(SPLITS, key=lambda s: (sum(
            state._term(s, k, state.counts[s][k] + n) - state._term(s, k, state.counts[s][k])
            for k, n in unit_vecs[i]), SPLITS.index(s)))
        state.apply(i, None, best)
        assign[i] = best
    for _ in range(MAX_PASSES):
        improved = False
        sweep = list(range(len(unit_vecs)))
        rng.shuffle(sweep)
        for i in sweep:
            for dst in SPLITS:
                if dst != assign[i] and state.move_delta(i, assign[i], dst) < -1e-9:
                    state.apply(i, assign[i], dst)
                    assign[i] = dst
                    improved = True
            partners = rng.sample(range(len(unit_vecs)), min(24, len(unit_vecs)))
            for j in partners:
                a, b = assign[i], assign[j]
                if a == b:
                    continue
                d1 = state.move_delta(i, a, b)
                state.apply(i, a, b)
                d2 = state.move_delta(j, b, a)
                if d1 + d2 < -1e-9:
                    state.apply(j, b, a)
                    assign[i], assign[j] = b, a
                    improved = True
                    break
                state.apply(i, b, a)
        if not improved:
            break
    return state.cost(), assign, state


def solve(cases, units, seed=DEFAULT_SEED, restarts=None):
    """Deterministic for a given seed: the best of `restarts` seeded greedy + local-search runs."""
    restarts = restarts or RESTARTS
    totals = Counter()
    unit_vecs, sizes = [], []
    for unit in units:
        vec = Counter()
        for i in unit["members"]:
            vec.update(features(cases[i]))
        unit_vecs.append(sorted(vec.items(), key=repr))
        sizes.append(len(unit["members"]))
        totals.update(vec)
    for g in GENERATORS:
        totals[("gen", g)] += 0  # a family absent from the data still has to be present
    rng = random.Random(seed)
    best = None
    for _ in range(restarts):
        run_rng = random.Random(rng.getrandbits(64))
        cost, assign, state = _solve_once(unit_vecs, totals, sizes, run_rng)
        key = (round(cost, 9), tuple(SPLITS.index(s) for s in assign))
        if best is None or key < best[0]:
            best = (key, assign, state)
    _, assign, state = best
    missing = [(s, g) for s in SPLITS for g in GENERATORS if state.counts[s][("gen", g)] <= 0]
    if missing:
        raise Refusal("constraint conflict: no split with seed {} places every generator family "
                      "in every split; missing {}".format(seed, missing))
    split_of = {}
    for unit, s in zip(units, assign):
        for i in unit["members"]:
            split_of[cases[i]["sha256"]] = s
    return split_of


# --- identity audit ---------------------------------------------------------------------------

def identity_audit(cases, split_of):
    seen = {"hash": defaultdict(set), "cluster": defaultdict(set), "subject": defaultdict(set)}
    for c in cases:
        s = split_of[c["sha256"]]
        seen["hash"][c["sha256"]].add(s)
        seen["cluster"][c["source_cluster_id"]].add(s)
        if c["subject_id"] is not None:
            seen["subject"][c["subject_id"]].add(s)
    overlaps = {k: sorted(x for x, splits in v.items() if len(splits) > 1)
                for k, v in seen.items()}
    gen_counts = {s: Counter(c["generator_family"] for c in cases
                             if split_of[c["sha256"]] == s and c["generator_family"])
                  for s in SPLITS}
    missing = sorted([s, g] for s in SPLITS for g in GENERATORS if gen_counts[s][g] <= 0)
    return {
        "hash_overlap_count": len(overlaps["hash"]),
        "cluster_overlap_count": len(overlaps["cluster"]),
        "subject_overlap_count": len(overlaps["subject"]),
        "overlapping": overlaps,
        "missing_generator_families": missing,
        "known_subjects": len(seen["subject"]),
        "null_subject_cases": sum(1 for c in cases if c["subject_id"] is None),
    }


# --- perceptual dHash audit -------------------------------------------------------------------

def frame_times(duration, n=DHASH_SPEC["frames_per_clip"]):
    return [duration * (2 * i + 1) / (2 * n) for i in range(n)]


def dhash_frame(bgr, cv2_module=None):
    if cv2_module is None:
        import cv2 as cv2_module
    gray = bgr if bgr.ndim == 2 else cv2_module.cvtColor(bgr, cv2_module.COLOR_BGR2GRAY)
    small = cv2_module.resize(gray, (9, 8), interpolation=cv2_module.INTER_AREA)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | int(small[r, c + 1] > small[r, c])
    return f"{bits:016x}"


def grab_frame(path, t):
    from eval import r18_t2_source as t2
    return t2.grab_frame(path, t)


def clip_dhashes(path, grab=grab_frame):
    probe, err = t1.run_ffprobe(path)
    status, reason, profile = t1.technical_profile(probe, err)
    if status != t1.INSPECTABLE:
        raise Refusal(f"{path}: not inspectable for the perceptual audit: {reason}")
    duration = profile["duration_s"]
    frames = []
    for t in frame_times(duration):
        img = grab(path, round(t, 3))
        frames.append(None if img is None else dhash_frame(img))
    if not any(frames):
        raise Refusal(f"{path}: no frame decoded for the perceptual audit")
    return {"duration_s": duration, "frames": frames}


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def min_distance(frames_a, frames_b):
    return min(hamming(a, b) for a in frames_a if a for b in frames_b if b)


def find_candidates(cases, frame_hashes, threshold=CANDIDATE_THRESHOLD):
    """Pairs of clips from different source clusters whose closest frame pair is <= threshold."""
    import numpy as np
    shas = [c["sha256"] for c in cases]
    clusters = [c["source_cluster_id"] for c in cases]
    n = len(shas)
    width = DHASH_SPEC["frames_per_clip"]
    vals = np.zeros((n, width), dtype=np.uint64)
    valid = np.zeros((n, width), dtype=bool)
    for i, sha in enumerate(shas):
        for k, h in enumerate(frame_hashes[sha]["frames"]):
            if h:
                vals[i, k] = np.uint64(int(h, 16))
                valid[i, k] = True
    out = []
    for i in range(n - 1):
        x = vals[i][None, :, None] ^ vals[i + 1:][:, None, :]
        d = np.bitwise_count(x).astype(np.int64)
        mask = valid[i][None, :, None] & valid[i + 1:][:, None, :]
        d = np.where(mask, d, 65).reshape(n - i - 1, -1).min(axis=1)
        for off in np.nonzero(d <= threshold)[0]:
            j = i + 1 + int(off)
            if clusters[i] != clusters[j]:
                out.append((shas[i], shas[j], int(d[off])))
    return out


def shared_pairs_of(registry):
    return sorted((e["clip_a_sha256"], e["clip_b_sha256"]) for e in registry.get("candidates", [])
                  if e["adjudication_state"] == SHARED_SOURCE)


def build_registry(cases, frame_hashes, previous, split_of, seed):
    """The adjudication registry for the current hashes; earlier decisions are kept by pair_id."""
    by_sha = {c["sha256"]: c for c in cases}
    prior = {e["pair_id"]: e for e in (previous or {}).get("candidates", [])}
    entries = []
    for a, b, d in find_candidates(cases, frame_hashes):
        a, b = sorted((a, b))
        pid = pair_id(a, b)
        old = prior.get(pid, {})
        state = old.get("adjudication_state", UNRESOLVED)
        if state not in STATES:
            raise Refusal(f"{pid}: unknown adjudication state {state!r}")
        scope = CROSS_SPLIT if split_of[a] != split_of[b] else SAME_SPLIT
        if state not in DECIDED:
            # A decision is kept wherever the pair lands; anything else follows the scope, so a
            # pair moved across splits by a re-split must be adjudicated again.
            state = UNRESOLVED if scope == CROSS_SPLIT else NOT_REQUIRED_SAME_SPLIT
        notes = old.get("adjudication_notes", "") if state in DECIDED else ""
        ca, cb = by_sha[a], by_sha[b]
        entries.append({
            "pair_id": pid,
            "clip_a_sha256": a, "clip_a_path": ca["canonical_path"],
            "clip_a_cluster": ca["source_cluster_id"], "clip_a_split": split_of[a],
            "clip_b_sha256": b, "clip_b_path": cb["canonical_path"],
            "clip_b_cluster": cb["source_cluster_id"], "clip_b_split": split_of[b],
            "scope": scope,
            "min_hamming_distance": d,
            "adjudication_state": state,
            "adjudication_notes": notes,
        })
    entries.sort(key=lambda e: e["pair_id"])
    dropped = sorted(set(prior) - {e["pair_id"] for e in entries})
    return {
        "schema": SCHEMA_ADJUDICATIONS,
        "method": DHASH_SPEC,
        "split_seed": seed,
        "rule": "INDEPENDENT: similar but independently recorded, may cross splits. "
                "SHARED_SOURCE: confirmed shared or tightly derived recording; the two clusters "
                "are merged into one partition unit and the dataset is re-split. UNRESOLVED: "
                "a CROSS_SPLIT candidate not yet investigated; the freeze refuses. "
                "NOT_REQUIRED_SAME_SPLIT: a SAME_SPLIT candidate, reported as a diagnostic; it "
                "does not block the freeze and becomes UNRESOLVED if a re-split separates it.",
        "summary": perceptual_summary(entries),
        "candidates": entries,
        "frame_hashes": {sha: {"canonical_path": by_sha[sha]["canonical_path"], **frame_hashes[sha]}
                         for sha in sorted(frame_hashes)},
    }, dropped


def bucket_of(distance):
    for lo, hi in DISTANCE_BUCKETS:
        if lo <= distance <= hi:
            return f"{lo}-{hi}"
    raise ValueError(f"distance {distance} outside the candidate threshold")


def perceptual_summary(entries):
    by_scope = {scope: Counter(e["adjudication_state"] for e in entries if e["scope"] == scope)
                for scope in SCOPE_STATES}
    buckets = {}
    for lo, hi in DISTANCE_BUCKETS:
        rows = [e for e in entries if lo <= e["min_hamming_distance"] <= hi]
        cross = [e for e in rows if e["scope"] == CROSS_SPLIT]
        buckets[f"{lo}-{hi}"] = {
            "candidates": len(rows),
            "cross_split": len(cross),
            "same_split": len(rows) - len(cross),
            "cross_split_independent": sum(1 for e in cross
                                           if e["adjudication_state"] == INDEPENDENT),
            "confirmed_shared_source": sum(1 for e in rows
                                           if e["adjudication_state"] == SHARED_SOURCE),
            "cross_split_unresolved": sum(1 for e in cross
                                          if e["adjudication_state"] == UNRESOLVED),
        }
    cross_unresolved = by_scope[CROSS_SPLIT].get(UNRESOLVED, 0)
    return {
        "perceptual_candidate_pair_count": len(entries),
        "cross_split_candidate_pair_count": sum(by_scope[CROSS_SPLIT].values()),
        "same_split_candidate_pair_count": sum(by_scope[SAME_SPLIT].values()),
        "adjudication_states_by_scope": {scope: {s: by_scope[scope].get(s, 0) for s in allowed}
                                         for scope, allowed in SCOPE_STATES.items()},
        "confirmed_shared_source_count": sum(1 for e in entries
                                             if e["adjudication_state"] == SHARED_SOURCE),
        "confirmed_cross_split_shared_source_count": by_scope[CROSS_SPLIT].get(SHARED_SOURCE, 0),
        "cross_split_unresolved_count": cross_unresolved,
        "unresolved_perceptual_candidate_count": cross_unresolved,
        "distance_buckets": buckets,
    }


def load_registry(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise Refusal(f"adjudication registry not found: {path} (run audit-leakage first)")
        return None
    reg = json.loads(path.read_text())
    if reg.get("schema") != SCHEMA_ADJUDICATIONS:
        raise Refusal(f"registry schema {reg.get('schema')!r} != {SCHEMA_ADJUDICATIONS!r}")
    for e in reg.get("candidates", []):
        allowed = SCOPE_STATES.get(e.get("scope"))
        if allowed is None:
            raise Refusal(f"{e.get('pair_id')}: unknown scope {e.get('scope')!r}")
        if e.get("adjudication_state") not in allowed:
            raise Refusal(f"{e.get('pair_id')}: adjudication state "
                          f"{e.get('adjudication_state')!r} is not allowed for {e['scope']}")
        if e["adjudication_state"] in DECIDED and not str(e.get("adjudication_notes")).strip():
            raise Refusal(f"{e['pair_id']}: {e['adjudication_state']} needs a rationale")
    return reg


# --- summaries --------------------------------------------------------------------------------

def _share(n, d):
    return round(n / d, 4) if d else None


def split_summary(cases, split_of):
    total = len(cases)
    out = {}
    for s in SPLITS:
        rows = [c for c in cases if split_of[c["sha256"]] == s]
        n = len(rows)
        ai = sum(1 for c in rows if c["ground_truth"] == VERIFIED_AI)
        face = sum(1 for c in rows if c["face_present"] is True)
        collections = Counter(c["source_collection_id"] for c in rows)
        top = max(collections.items(), key=lambda kv: (kv[1], kv[0])) if collections else None
        clusters = Counter(c["source_cluster_id"] for c in rows)
        out[s] = {
            "total_cases": n,
            "case_share": _share(n, total),
            "target_case_share": FRACTIONS[s],
            "ai_cases": ai,
            "real_cases": n - ai,
            "ai_share": _share(ai, n),
            "total_clusters": len(clusters),
            "max_cluster_share": _share(max(clusters.values()), n) if clusters else None,
            "known_subjects": len({c["subject_id"] for c in rows if c["subject_id"]}),
            "face_present_count": face,
            "faceless_count": sum(1 for c in rows if c["face_present"] is False),
            "face_unknown_count": sum(1 for c in rows if c["face_present"] is None),
            "face_share": _share(face, n),
            "generator_breakdown": dict(sorted(Counter(
                c["generator_family"] for c in rows if c["generator_family"]).items())),
            "generator_model_breakdown": dict(sorted(Counter(
                c["generator_model"] for c in rows if c["generator_model"]).items())),
            "capture_domain_breakdown": dict(sorted(Counter(
                c["capture_domain"] for c in rows if c["capture_domain"]).items())),
            "collection_breakdown": dict(sorted(collections.items())),
            "max_collection": top[0] if top else None,
            "max_collection_share": _share(top[1], n) if top else None,
            "soft_targets": {
                "ai_share_within_tolerance":
                    AI_SHARE_TOLERANCE[0] <= ai / n <= AI_SHARE_TOLERANCE[1] if n else False,
                "face_share_within_tolerance":
                    FACE_SHARE_TOLERANCE[0] <= face / n <= FACE_SHARE_TOLERANCE[1] if n else False,
                "ai_share_tolerance": list(AI_SHARE_TOLERANCE),
                "face_share_tolerance": list(FACE_SHARE_TOLERANCE),
            },
        }
    return out


def collection_spread(cases, split_of):
    spread = defaultdict(Counter)
    for c in cases:
        spread[c["source_collection_id"]][split_of[c["sha256"]]] += 1
    return {k: {s: v.get(s, 0) for s in SPLITS} for k, v in sorted(spread.items())}


def plan_split(anchors, registry, seed):
    _, cases = load_cases(anchors)
    shared = shared_pairs_of(registry) if registry else []
    units = build_units(cases, shared)
    return cases, units, solve(cases, units, seed)


# --- freeze -----------------------------------------------------------------------------------

def freeze_record(c, split, unit_of):
    return {
        "sha256": c["sha256"],
        "canonical_path": c["canonical_path"],
        "split": split,
        "ground_truth": c["ground_truth"],
        "generator_family": c["generator_family"],
        "generator_model": c["generator_model"],
        "capture_domain": c["capture_domain"],
        "face_present": c["face_present"],
        "speech_talking_head": c["speech_talking_head"],
        "source_cluster_id": c["source_cluster_id"],
        "source_collection_id": c["source_collection_id"],
        "subject_id": c["subject_id"],
        "partition_unit_id": unit_of[c["sha256"]],
    }


def manifest_digest(records):
    return sha256_bytes(canonical_bytes(sorted(records, key=lambda r: r["sha256"])))


def manifest_csv(records):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MANIFEST_COLUMNS)
    for r in sorted(records, key=lambda r: r["sha256"]):
        row = []
        for col in MANIFEST_COLUMNS:
            v = r[col]
            row.append("" if v is None else ("true" if v is True else "false" if v is False
                                             else v))
        w.writerow(row)
    return buf.getvalue()


def build_freeze(anchors_path, registry_path, seed, timestamp):
    """Everything the freeze writes, or a Refusal. Pure: no clock, no media, no network."""
    timestamp = validate_timestamp(timestamp)
    registry = load_registry(registry_path)
    _, cases = load_cases(anchors_path)
    shas = {c["sha256"] for c in cases}
    hashed = set(registry.get("frame_hashes", {}))
    if hashed != shas:
        raise Refusal("registry frame hashes do not cover exactly the anchored clips "
                      f"({len(hashed - shas)} extra, {len(shas - hashed)} missing); "
                      "re-run audit-leakage")
    if registry.get("method") != DHASH_SPEC:
        raise Refusal("registry dHash method differs from the frozen method; re-run audit-leakage")
    expected = {pair_id(a, b): d for a, b, d in find_candidates(cases, registry["frame_hashes"])}
    recorded = {e["pair_id"]: e["min_hamming_distance"] for e in registry["candidates"]}
    if expected != recorded:
        raise Refusal("registry candidates are stale against its frame hashes; re-run "
                      "audit-leakage")
    unresolved = [e["pair_id"] for e in registry["candidates"]
                  if e["scope"] == CROSS_SPLIT and e["adjudication_state"] == UNRESOLVED]
    if unresolved:
        raise Refusal(f"{len(unresolved)} cross-split perceptual candidate(s) UNRESOLVED: "
                      f"{unresolved[:10]}")
    units = build_units(cases, shared_pairs_of(registry))
    split_of = solve(cases, units, seed)
    if registry.get("split_seed") != seed or any(
            e["clip_a_split"] != split_of[e["clip_a_sha256"]]
            or e["clip_b_split"] != split_of[e["clip_b_sha256"]]
            or (e["scope"] == CROSS_SPLIT) != (e["clip_a_split"] != e["clip_b_split"])
            for e in registry["candidates"]):
        raise Refusal("registry split fields are stale for this seed; re-run audit-leakage")
    audit = identity_audit(cases, split_of)
    if audit["hash_overlap_count"] or audit["cluster_overlap_count"] or \
            audit["subject_overlap_count"] or audit["missing_generator_families"]:
        raise Refusal(f"Tier 1 invariant violated: {audit}")
    perceptual = perceptual_summary(registry["candidates"])
    if perceptual["confirmed_cross_split_shared_source_count"]:
        raise Refusal("a confirmed SHARED_SOURCE pair crosses splits")
    unit_of = {cases[i]["sha256"]: u["unit_id"] for u in units for i in u["members"]}
    records = [freeze_record(c, split_of[c["sha256"]], unit_of) for c in cases]
    csv_text = manifest_csv(records)
    freeze = {
        "schema": SCHEMA_FREEZE,
        "dataset_name": DATASET_NAME,
        "freeze_timestamp": timestamp,
        "seed": seed,
        "authority": AUTHORITY,
        "partition_topology": "DEV_VAL_LOCKED_TEST",
        "partition_fractions": FRACTIONS,
        "total_records": len(records),
        "manifest_sha256": manifest_digest(records),
        "manifest_sha256_rule": "SHA-256 of the records list sorted by sha256, serialised as "
                                "JSON with sorted keys, separators (',', ':'), UTF-8",
        "manifest_csv_sha256": sha256_bytes(csv_text.encode()),
        "governance_notice": GOVERNANCE_NOTICE,
        "locked_test_access": f"LOCKED_TEST case-level records may be loaded only for purpose "
                              f"{LOCKED_TEST_PURPOSE!r} (load_partition refuses otherwise)",
        "inputs": {
            "anchors_path": display_path(anchors_path),
            "anchors_sha256": t1.sha256_file(anchors_path),
            "adjudications_path": display_path(registry_path),
            "adjudications_sha256": t1.sha256_file(registry_path),
        },
        "partition_units": {"count": len(units),
                            "merged_by_shared_source": sorted(
                                u["unit_id"] for u in units if "+" in u["unit_id"])},
        "split_summary": split_summary(cases, split_of),
        "collection_spread": collection_spread(cases, split_of),
        "leakage_verification": {
            "cluster_overlap_count": audit["cluster_overlap_count"],
            "subject_overlap_count": audit["subject_overlap_count"],
            "hash_overlap_count": audit["hash_overlap_count"],
            "known_subjects": audit["known_subjects"],
            "null_subject_cases": audit["null_subject_cases"],
            "missing_generator_families": audit["missing_generator_families"],
            **perceptual,
            "adjudications_path": display_path(registry_path),
            "verification_status": "CLEAN",
        },
        "records": records,
    }
    return freeze, csv_text


def verify_freeze(freeze, csv_text=None):
    """Problems found in a freeze (an empty list means intact)."""
    problems = []
    records = freeze.get("records", [])
    if manifest_digest(records) != freeze.get("manifest_sha256"):
        problems.append("manifest_sha256 does not match the records")
    if len(records) != freeze.get("total_records"):
        problems.append("total_records does not match the records")
    if csv_text is not None:
        if sha256_bytes(csv_text.encode()) != freeze.get("manifest_csv_sha256"):
            problems.append("CSV manifest digest does not match")
        if csv_text != manifest_csv(records):
            problems.append("CSV manifest does not match the records")
    if any(r.get("split") not in SPLITS for r in records):
        problems.append("a record has an unknown split")
    cases = [{**r, "subject_id": normalise_subject(r.get("subject_id"))} for r in records]
    try:
        audit = identity_audit(cases, {r["sha256"]: r["split"] for r in records})
    except KeyError as exc:
        return problems + [f"a record lacks {exc}"]
    for key in ("hash_overlap_count", "cluster_overlap_count", "subject_overlap_count"):
        if audit[key]:
            problems.append(f"{key} = {audit[key]}")
    if audit["missing_generator_families"]:
        problems.append(f"missing generator families: {audit['missing_generator_families']}")
    return problems


def load_partition(freeze_path, split, purpose=None):
    """Case records of one split of an intact freeze. LOCKED_TEST only for R18-T6."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == LOCKED_TEST and purpose != LOCKED_TEST_PURPOSE:
        raise PermissionError("LOCKED_TEST is reserved for R18-T6; T4/T5 tooling may load only "
                              "DEV and VAL")
    freeze = json.loads(Path(freeze_path).read_text())
    problems = verify_freeze(freeze)
    if problems:
        raise ValueError(f"freeze failed verification: {problems}")
    return [r for r in freeze["records"] if r["split"] == split]


# --- subcommands ------------------------------------------------------------------------------

def cmd_split(args):
    registry = load_registry(args.adjudications, required=False) if args.adjudications else None
    cases, units, split_of = plan_split(args.anchors, registry, args.seed)
    audit = identity_audit(cases, split_of)
    summary = split_summary(cases, split_of)
    print(json.dumps({"seed": args.seed, "partition_units": len(units),
                      "identity_audit": {k: v for k, v in audit.items() if k != "overlapping"},
                      "splits": {s: {k: summary[s][k] for k in (
                          "total_cases", "ai_cases", "real_cases", "total_clusters",
                          "face_present_count", "generator_breakdown")} for s in SPLITS}},
                     indent=1))


def cmd_audit_leakage(args):
    previous = load_registry(args.adjudications, required=False)
    _, cases = load_cases(args.anchors)
    corpus = Path(args.corpus).expanduser()
    cached = (previous or {}).get("frame_hashes", {}) \
        if (previous or {}).get("method") == DHASH_SPEC and not args.rehash else {}
    frame_hashes = {}
    for k, c in enumerate(cases, 1):
        path = corpus / c["canonical_path"]
        if not path.is_file():
            raise Refusal(f"missing media: {path}")
        digest = t1.sha256_file(path)
        if digest != c["sha256"]:
            raise Refusal(f"{path}: sha256 {digest} != anchored {c['sha256']}")
        if c["sha256"] in cached:
            frame_hashes[c["sha256"]] = {"duration_s": cached[c["sha256"]]["duration_s"],
                                         "frames": cached[c["sha256"]]["frames"]}
        else:
            frame_hashes[c["sha256"]] = clip_dhashes(path)
        if k % 50 == 0:
            print(f"hashed {k}/{len(cases)}", file=sys.stderr)
    shared = shared_pairs_of(previous) if previous else []
    split_of = solve(cases, build_units(cases, shared), args.seed)
    registry, dropped = build_registry(cases, frame_hashes, previous, split_of, args.seed)
    write_json(args.adjudications, registry)
    audit = identity_audit(cases, split_of)
    print(json.dumps({"identity_audit": {k: v for k, v in audit.items() if k != "overlapping"},
                      "perceptual": registry["summary"],
                      "dropped_stale_adjudications": dropped}, indent=1))


def cmd_freeze(args):
    freeze, csv_text = build_freeze(args.anchors, args.adjudications, args.seed,
                                    args.freeze_timestamp)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).write_text(csv_text)
    write_json(args.freeze_out, freeze)
    print(json.dumps({"total_records": freeze["total_records"],
                      "manifest_sha256": freeze["manifest_sha256"],
                      "manifest_csv_sha256": freeze["manifest_csv_sha256"],
                      "splits": {s: freeze["split_summary"][s]["total_cases"] for s in SPLITS},
                      "leakage_verification": freeze["leakage_verification"]}, indent=1))


def cmd_verify(args):
    freeze = json.loads(Path(args.freeze).read_text())
    csv_text = Path(args.manifest).read_text() if args.manifest else None
    problems = verify_freeze(freeze, csv_text)
    print(json.dumps({"intact": not problems, "problems": problems}, indent=1))
    if problems:
        raise Refusal(1)


# --- CLI --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("split")
    a.add_argument("--anchors", required=True)
    a.add_argument("--adjudications")
    a.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a.set_defaults(func=cmd_split)
    b = sub.add_parser("audit-leakage")
    b.add_argument("--anchors", required=True)
    b.add_argument("--adjudications", required=True)
    b.add_argument("--corpus", required=True)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--rehash", action="store_true",
                   help="decode every clip again instead of reusing the registry's frame hashes")
    b.set_defaults(func=cmd_audit_leakage)
    c = sub.add_parser("freeze")
    c.add_argument("--anchors", required=True)
    c.add_argument("--adjudications", required=True)
    c.add_argument("--freeze-timestamp", required=True)
    c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c.add_argument("--freeze-out", required=True)
    c.add_argument("--manifest-out", required=True)
    c.set_defaults(func=cmd_freeze)
    d = sub.add_parser("verify")
    d.add_argument("--freeze", required=True)
    d.add_argument("--manifest")
    d.set_defaults(func=cmd_verify)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
