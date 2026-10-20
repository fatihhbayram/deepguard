"""R18-T1: catalogue the existing corpus read-only and measure it against the R18 target.

    PYTHONPATH=scripts python3 scripts/eval/r18_t1_inventory.py \
        --corpus ~/deepguard-corpus \
        --inventory scripts/eval/baselines/r18_t1_corpus_inventory.json \
        --gap scripts/eval/baselines/r18_t1_gap_analysis.json

Applies `r18_t1_implementation_plan.md` (rev 2) and decides nothing beyond it. Every file under
the corpus is opened read-only: media are SHA-256'd and `ffprobe`d, manifests are parsed. No
file is written, moved or re-encoded there, nothing is downloaded, and no detector, vision
model or production analysis runs — `ffprobe` is the only subprocess.

Ground truth comes only from the explicit allowlist below, and only where a git-tracked trust
anchor matches; a digest recorded only in git-excluded docs/ai/ is reported as context and
grants nothing, and a folder name is never evidence. Identical bytes collapse to one canonical entry (the
lexicographically smallest relative path); contradictory truth claims on one digest are
quarantined as AMBIGUOUS, never resolved. Technical inspectability is recorded separately from
ground-truth confidence, and no dataset admission is decided here.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R17_DIGESTS = REPO / "scripts/eval/specs/r17_t2_expected_digests.json"
SCHEMA_INVENTORY = "r18-t1-corpus-inventory-1"
SCHEMA_GAP = "r18-t1-gap-analysis-1"

VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
# Media containers the plan's extension list does not cover: listed, never hashed or classified.
OTHER_MEDIA_EXTS = frozenset({".flv", ".mpg", ".mpeg", ".m4v", ".ts", ".3gp", ".wmv"})
# Plan §4 / D1. `r17_t3_work` is deliberately absent: the 45 frozen R17 clips exist nowhere
# else on disk, and the plan's own record schema names `r17_t3_work/...B001.mp4` as an instance.
EXCLUDED_TOP = frozenset({"models", "reports", "runs", "r17_t3_phase_a", "r17_t3_phase_b",
                          "r17_t3_runlog", "r17_t3_session", "r17_t3_blind_map", "r17_t4_work",
                          "r17_t5_diagnostic"})

INSPECTABLE, UNREADABLE = "INSPECTABLE", "UNREADABLE"
VERIFIED_AI, VERIFIED_REAL = "VERIFIED_AI", "VERIFIED_REAL"
INSUFFICIENT, UNLABELED, AMBIGUOUS = "LABEL_PRESENT_BUT_GT_INSUFFICIENT", "UNLABELED", "AMBIGUOUS"
CONFIDENCE_ORDER = (VERIFIED_AI, VERIFIED_REAL, INSUFFICIENT, UNLABELED, AMBIGUOUS)

# Normalised truth claims. Coarse classes decide a conflict: REAL against anything not REAL is
# contradictory; AI_GENERATED against FACE_SWAP is a vocabulary difference, not a contradiction.
REAL, AI, FACE_SWAP, MANIPULATED, RENDERED = ("REAL", "AI_GENERATED", "FACE_SWAP",
                                               "MANIPULATED_OTHER", "RENDERED")
COARSE = {REAL: "REAL", AI: "NOT_REAL", FACE_SWAP: "NOT_REAL", MANIPULATED: "NOT_REAL",
          RENDERED: "RENDERED"}
LABELS = {"real": REAL, "genuine": REAL, "ai": AI, "ai_generated": AI, "synthetic": AI,
          "face_swap": FACE_SWAP, "manipulated": MANIPULATED, "audio_manipulation": MANIPULATED,
          "other_manipulation": MANIPULATED}

# The trust-anchor allowlist (plan §5.1 / D2). An allowlisted corpus yields VERIFIED_* only
# where its expected digests sit in a git-tracked file (Architect, R18-T1 D-5). A digest that is
# recorded only in git-excluded docs/ai/ is supplemental context: it is checked and reported,
# but it grants nothing. A corpus file cannot vouch for itself either (SEAL.json, manifest.json).
R16_ROOTS_SCRIPT = REPO / "scripts/eval/r16_t6_pilot_dataset.py"
TURNTURK_MANIFEST_SHA256 = "9cdbc0bba99a5cd0c363319a3a8835f906c45d93536aacb5368be80ea30cd711"
R16_GT_SHA256 = "147eff948d393f698a9d92c3c56cee86b79fe8043f77a2b27cc32e8f0e38eda3"
R16_EXECUTION_SHA256 = "7f1d64dc648686e8f292c33fda37e48db3941c1bcc725318a8201f47c60945a3"
R15_MANIFEST_SHA256 = "01f33af9f45db3ae2884514da15b6d55c4db37b94d594462387f54a31458d5bc"
# Not allowlisted: checked so the Architect can see the anchor holds, but it grants nothing.
VCD1_MANIFEST_SHA256 = "2f44995f0fb2be0bce5ba7a61b72066ce30f5b8c36c8cdad75978184437b1476"

ANCHOR_SOURCES = {
    "r17-ai-visual-v1": "scripts/eval/specs/r17_t2_expected_digests.json (git-tracked, 40a5c1b)",
    "r16_pilot owner originals": "scripts/eval/r16_t6_pilot_dataset.py ROOTS (git-tracked, "
                                 "afb8be9)",
    "external-genuine-real-turnturk": "docs/ai/reviews/R11_PHASE_CLOSURE.md, R11_T1 intake report "
                                      "(git-excluded)",
    "r16_pilot": "docs/ai/r16_t6_report.md, docs/ai/r16_t7_freeze.json (git-excluded)",
    "r15_oos_manifest.json": "docs/ai/r15_t7_freeze.json (git-excluded)",
    "external-genuine-vcd1": "docs/ai/reviews/R11_PHASE_CLOSURE.md, docs/ai/r15_t7_freeze.json "
                             "(git-excluded)",
}
UNTRACKED_BASIS = ("digest matches a git-excluded docs/ai/ record only; not a sufficient trust "
                   "anchor for VERIFIED (Architect R18-T1 D-5)")

# Plan §7.8 and phase rule 4: the target and the categories it names.
TARGET_MIN, TARGET_MAX = 300, 500
DESIRED_GENERATORS = ("Sora", "Veo", "Runway", "Kling", "Hailuo/MiniMax", "Luma", "Pika")
DESIRED_REAL_DOMAINS = ("smartphone", "webcam", "interview/talking-head", "podcast",
                        "street/outdoor", "nature", "vehicle", "product", "faceless",
                        "face-present")
# Which desired REAL domain a documented capture domain satisfies. Nothing else maps.
DOMAIN_COVERS = {"smartphone_handheld_owner": ("smartphone",),
                 "spacecraft_onboard_camera": ()}
FACE_FIELDS = ("face_present", "talking_head")


class Refusal(SystemExit):
    """A stop condition: the inventory is not written."""


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def normalise_label(raw):
    if raw is None:
        return None
    return LABELS.get(str(raw).strip().lower())


def subcorpus_of(rel):
    return rel.split("/", 1)[0]


# --- discovery ------------------------------------------------------------------------------

def discover(corpus):
    """Walk the corpus without following links. Returns media, other-media and symlink lists."""
    corpus = Path(corpus)
    media, other, links = [], [], []
    for root, dirs, files in os.walk(corpus, followlinks=False):
        rel_root = Path(root).relative_to(corpus).as_posix()
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in EXCLUDED_TOP]
        for name in files + [d for d in dirs if (Path(root) / d).is_symlink()]:
            p = Path(root) / name
            rel = p.relative_to(corpus).as_posix()
            ext = p.suffix.lower()
            if p.is_symlink():
                if ext in VIDEO_EXTS | OTHER_MEDIA_EXTS:
                    links.append({"path": rel, "target": os.readlink(p)})
            elif ext in VIDEO_EXTS:
                media.append(rel)
            elif ext in OTHER_MEDIA_EXTS:
                other.append(rel)
    return sorted(media), sorted(other), sorted(links, key=lambda x: x["path"])


# --- physical inspection --------------------------------------------------------------------

def run_ffprobe(path):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired:
        return None, "ffprobe timeout"
    if out.returncode != 0:
        return None, (out.stderr.strip().splitlines() or ["ffprobe failed"])[-1][:300]
    try:
        return json.loads(out.stdout), None
    except json.JSONDecodeError:
        return None, "ffprobe output not JSON"


def _rate(text):
    if not text or text in ("0/0", "0"):
        return None
    num, _, den = str(text).partition("/")
    try:
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return round(value, 3) if value > 0 else None


def _num(text, kind=float):
    try:
        return kind(text)
    except (TypeError, ValueError):
        return None


def technical_profile(probe, error=None, size_bytes=None):
    """INSPECTABLE needs a parsed container and a video stream with codec, size and duration."""
    if size_bytes == 0:
        return UNREADABLE, "zero-byte file", None
    if probe is None:
        return UNREADABLE, error or "container did not parse", None
    fmt = probe.get("format") or {}
    streams = probe.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not (s.get("disposition") or {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = _num(fmt.get("duration")) or _num((video or {}).get("duration"))
    profile = {
        "container": fmt.get("format_name"),
        "duration_s": round(duration, 3) if duration else duration,
        "video": None if video is None else {
            "codec": video.get("codec_name"), "width": video.get("width"),
            "height": video.get("height"),
            "fps": _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")),
            "frames": _num(video.get("nb_frames"), int)},
        "has_audio": audio is not None,
        "audio": None if audio is None else {
            "codec": audio.get("codec_name"), "sample_rate": _num(audio.get("sample_rate"), int),
            "channels": audio.get("channels")},
    }
    if video is None:
        return UNREADABLE, "no video stream", profile
    if not video.get("codec_name"):
        return UNREADABLE, "video stream has no codec", profile
    if not (video.get("width") or 0) > 0 or not (video.get("height") or 0) > 0:
        return UNREADABLE, "video stream has no resolution", profile
    if not duration or duration <= 0:
        return UNREADABLE, "no positive duration", profile
    return INSPECTABLE, None, profile


# --- metadata records -----------------------------------------------------------------------

def record(subcorpus, source, claim, raw_label, trusted, trust_basis, **extra):
    return {"subcorpus": subcorpus, "source_manifest": source, "claimed_truth": claim,
            "raw_label": raw_label, "trusted": trusted, "trust_basis": trust_basis,
            **{k: v for k, v in extra.items() if v is not None}}


def _hash_or_none(path):
    return sha256_file(path) if path.is_file() else None


def anchor_check(name, expected, actual_files, tracked):
    """`expected`/`actual_files`: {filename: digest}. All must be present and equal.

    `verified` is the digest match; `grants_verified` also needs the anchor to be git-tracked.
    """
    observed = {f: actual_files.get(f) for f in expected}
    ok = bool(expected) and all(observed[f] == expected[f] for f in expected)
    return {"corpus": name, "anchor_source": ANCHOR_SOURCES.get(name), "expected": expected,
            "observed": observed, "verified": ok, "tracked": tracked,
            "grants_verified": ok and tracked}


def tracked_r16_roots(script=None):
    """The owner originals pinned by the tracked R16-T6 build. Read with `ast`, never executed."""
    import ast
    path = Path(script or R16_ROOTS_SCRIPT)
    if not path.is_file():
        return {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ROOTS"
                                                for t in node.targets):
            return {digest: name for name, (digest, _notes) in ast.literal_eval(node.value).items()}
    return {}


def r17_records(corpus, digests_path=None):
    base = Path(corpus) / "r17-ai-visual-v1"
    if not base.is_dir():
        return [], [], None
    expected = json.loads(Path(digests_path or R17_DIGESTS).read_text())["sha256"]
    anchor = anchor_check("r17-ai-visual-v1", expected,
                          {f: _hash_or_none(base / f) for f in expected}, tracked=True)
    trusted = anchor["grants_verified"]
    basis = "r17-ai-visual-v1 frozen manifest; FREEZE/manifest digests match the committed anchor"
    cases = json.loads((base / "manifest.json").read_text())
    by_hash = []
    for c in cases:
        claim = normalise_label(c.get("truth"))
        validated = c.get("validation_status") == "VALIDATED"
        domain = "spacecraft_onboard_camera" if claim == REAL else None
        by_hash.append((c["clip_sha256"], record(
            "r17-ai-visual-v1", "r17-ai-visual-v1/manifest.json", claim, c.get("truth"),
            trusted and validated, basis if trusted and validated else
            "anchor mismatch or case not VALIDATED", case_id=c.get("case_id"),
            generator_or_source=c.get("generator_or_source"), capture_domain=domain,
            source_cluster_id=c.get("source_cluster_id"),
            provenance_ref=c.get("truth_evidence_url"))))
    # Whole source files T3 kept beside the clips. The freeze validated 8-s clips, not these.
    sums = Path(corpus) / "r17_t3_work/source_sha256.txt"
    clusters = json.loads((base / "FREEZE.json").read_text()).get("source_clusters", [])
    if sums.is_file():
        for line in sums.read_text().splitlines():
            digest, _, name = line.strip().partition("  ")
            stem = Path(name).stem.split("~")[0]
            cl = next((c for c in clusters if stem and stem in c.get("source_url", "")), None)
            if cl is None:
                continue
            by_hash.append((digest, record(
                "r17_t3_work", "r17_t3_work/source_sha256.txt", normalise_label(cl["truth"]),
                cl["truth"], False, "whole source file; the R17 freeze validated only 8-s clips "
                "cut from it, not the full upload", generator_or_source=cl["generator_or_source"],
                source_cluster_id=cl["source_cluster_id"], provenance_ref=cl["source_url"])))
    return by_hash, [], anchor


def turnturk_records(corpus):
    base = Path(corpus) / "external-genuine-real-turnturk"
    manifest = base / "manifest.json"
    if not manifest.is_file():
        return [], [], None
    anchor = anchor_check("external-genuine-real-turnturk",
                          {"manifest.json": TURNTURK_MANIFEST_SHA256},
                          {"manifest.json": sha256_file(manifest)}, tracked=False)
    data = json.loads(manifest.read_text())
    by_hash = []
    for f in data.get("files", []):
        role, modality = f.get("file_role", ""), f.get("modality")
        if modality == "audio":
            claim, domain, raw = REAL, "conversational_video_call_audio", "genuine audio"
        elif "video" in (modality or "") and not f.get("carries_original_human_video_pixels",
                                                         True):
            claim, domain, raw = RENDERED, None, role
        else:
            continue
        by_hash.append((f["sha256"], record(
            "external-genuine-real-turnturk", "external-genuine-real-turnturk/manifest.json",
            claim, raw, anchor["grants_verified"],
            UNTRACKED_BASIS if anchor["verified"] else "anchor mismatch",
            case_id=f.get("path"), generator_or_source="Real-TurnTurk " + role,
            capture_domain=domain, anchor_matched_untracked=anchor["verified"] or None)))
    return by_hash, [], anchor


def r16_records(corpus):
    base = Path(corpus) / "r16_pilot"
    gt, execution, seal = (base / "sealed/pilot_ground_truth.json",
                           base / "pilot_manifest.json", base / "SEAL.json")
    if not gt.is_file():
        return [], [], None
    actual = {"sealed/pilot_ground_truth.json": sha256_file(gt),
              "pilot_manifest.json": _hash_or_none(execution)}
    anchor = anchor_check("r16_pilot", {"sealed/pilot_ground_truth.json": R16_GT_SHA256,
                                        "pilot_manifest.json": R16_EXECUTION_SHA256}, actual,
                          tracked=False)
    if seal.is_file():
        s = json.loads(seal.read_text())
        agrees = (s.get("gt_manifest_sha256") == actual["sealed/pilot_ground_truth.json"]
                  and s.get("execution_manifest_sha256") == actual["pilot_manifest.json"])
        anchor["seal_agrees"] = agrees
        anchor["verified"] = anchor["verified"] and agrees
    # Only the camera originals have a tracked anchor: their bytes are pinned in ROOTS.
    roots = tracked_r16_roots()
    roots_anchor = {"corpus": "r16_pilot owner originals",
                    "anchor_source": ANCHOR_SOURCES["r16_pilot owner originals"],
                    "tracked": True, "pinned_digests": len(roots)}
    by_hash = []
    for c in json.loads(gt.read_text())["cases"]:
        manip = c.get("manipulation") or {}
        claim = normalise_label(c.get("truth"))
        original = c.get("source_class") == "OWNER_KNOWN"
        trusted = original and claim == REAL and c["sha256"] in roots
        basis = (f"camera original {roots[c['sha256']]} pinned by sha256 in git-tracked "
                 "scripts/eval/r16_t6_pilot_dataset.py ROOTS (afb8be9); owner first-person "
                 "attestation in r16_pilot_raw/genuine_source/SOURCES.txt") if trusted else (
            UNTRACKED_BASIS if anchor["verified"] else "anchor mismatch")
        by_hash.append((c["sha256"], record(
            "r16_pilot", "r16_pilot/sealed/pilot_ground_truth.json", claim, c.get("label"),
            trusted, basis, anchor_matched_untracked=(anchor["verified"] and not trusted) or None,
            case_id=c.get("case_id"), generator_or_source=manip.get("family"),
            capture_domain="smartphone_handheld_owner" if claim == REAL else None,
            derivative=None if original or claim != REAL else manip.get("encode_path_desc")
            or manip.get("family"), stratum=c.get("stratum"),
            source_cluster_id=(c.get("lineage") or {}).get("owner_lineage"))))
    roots_anchor["matched_in_corpus"] = sum(1 for d, r in by_hash if r["trusted"])
    roots_anchor["verified"] = roots_anchor["grants_verified"] = roots_anchor[
        "matched_in_corpus"] > 0
    anchor["tracked_sub_anchor"] = roots_anchor
    return by_hash, [], anchor


def r15_records(corpus):
    manifest = Path(corpus) / "r15_oos_manifest.json"
    if not manifest.is_file():
        return [], [], None
    anchor = anchor_check("r15_oos_manifest.json", {"r15_oos_manifest.json": R15_MANIFEST_SHA256},
                          {"r15_oos_manifest.json": sha256_file(manifest)}, tracked=False)
    data = json.loads(manifest.read_text())
    by_hash, outside = [], []
    for group in ("primary_oos", "auxiliary_known_family"):
        for items in (data.get(group) or {}).values():
            for it in items:
                if not isinstance(it, dict) or "file_hash" not in it:
                    continue
                path = it.get("file_path") or ""
                if Path(path).suffix.lower() not in VIDEO_EXTS:
                    continue
                claim = normalise_label(it.get("label"))
                rec = record("r15_oos", "r15_oos_manifest.json", claim, it.get("label"),
                             anchor["grants_verified"],
                             UNTRACKED_BASIS if anchor["verified"] else "anchor mismatch",
                             anchor_matched_untracked=anchor["verified"] or None,
                             capture_domain="smartphone_handheld_owner" if claim == REAL else None,
                             derivative=None if group == "primary_oos" else "auxiliary transcode",
                             file_path=path)
                by_hash.append((it["file_hash"], rec))
                if not path.startswith(str(Path(corpus)) + "/"):
                    outside.append({"path": path, "sha256": it["file_hash"], "group": group,
                                    "label": it.get("label")})
    return by_hash, outside, anchor


def vcd1_records(corpus):
    manifest = Path(corpus) / "external-genuine-vcd1/vcd1_intake_manifest.json"
    if not manifest.is_file():
        return [], [], None
    anchor = anchor_check("external-genuine-vcd1", {"vcd1_intake_manifest.json":
                                                    VCD1_MANIFEST_SHA256},
                          {"vcd1_intake_manifest.json": sha256_file(manifest)}, tracked=False)
    anchor["allowlisted"] = False
    data = json.loads(manifest.read_text())
    domains = {"th": "webcam (VCD1 th, documented)", "th-m": "handheld phone (VCD1 th-m, documented)"}
    by_hash = [(r["sha256_local_freeze"], record(
        "external-genuine-vcd1", "external-genuine-vcd1/vcd1_intake_manifest.json", REAL, "GENUINE",
        False, "R11-T3B intake digest matches a git-excluded record only, and VCD1 is not on "
        "the T1 trust allowlist (Architect R18-T1 D-2)",
        case_id=r.get("subject_id"), generator_or_source="Microsoft VCD1",
        documented_capture_domain=domains.get(r.get("subset")), allowlist_candidate=True))
        for r in data.get("file_records", [])]
    return by_hash, [], anchor


def known_generated_records(corpus):
    by_hash = []
    for meta in sorted((Path(corpus) / "external-known-generated").glob("*/metadata.json")):
        data = json.loads(meta.read_text())
        prov = data.get("ground_truth_provenance") or {}
        for name, f in (data.get("files") or {}).items():
            by_hash.append((f["sha256"], record(
                "external-known-generated", meta.relative_to(corpus).as_posix(),
                normalise_label(data.get("ground_truth")), data.get("ground_truth"), False,
                "owner assertion only; generator and method not established",
                case_id=data.get("id"), generator_or_source=prov.get("generator"),
                family=data.get("family"), role=f.get("role"))))
    return by_hash


SPECIFIC_MANIFESTS = frozenset({"r17-ai-visual-v1", "external-genuine-real-turnturk"})


def csv_manifest_records(corpus):
    """Path-keyed labels from every other `*manifest*.csv`. Never trusted (plan §5.1 item 4)."""
    corpus = Path(corpus)
    by_path, unresolved, outside = [], Counter(), Counter()
    for f in sorted(corpus.rglob("*manifest*.csv")):
        rel = f.relative_to(corpus).as_posix()
        top = subcorpus_of(rel)
        if top in EXCLUDED_TOP or top in SPECIFIC_MANIFESTS or "/clips/" in rel:
            continue
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            raw = row.get("label") or row.get("ground_truth") or row.get("truth")
            if not row.get("path") or raw is None:
                continue
            target = os.path.normpath(f.parent / row["path"])
            if not target.startswith(str(corpus) + os.sep):
                outside[rel] += 1
                continue
            if not Path(target).is_file():
                unresolved[rel] += 1
                continue
            by_path.append((Path(target).relative_to(corpus).as_posix(), record(
                top, rel, normalise_label(raw), raw, False,
                "label in an unreviewed benchmark manifest; no allowlisted anchor",
                case_id=row.get("clip_id"), family=row.get("family"),
                stratum=row.get("stratum"), generator_or_source=row.get("family"),
                source_cluster_id=row.get("source_lineage_id"))))
    return by_path, {"missing_inside_root": dict(unresolved),
                     "pointing_outside_root": dict(outside)}


# --- classification -------------------------------------------------------------------------

def classify(records):
    """One confidence class per digest from its records. Returns (class, rationale, conflict)."""
    claims = [r for r in records if r.get("claimed_truth")]
    coarse = {COARSE[r["claimed_truth"]] for r in claims}
    if len(coarse) > 1:
        return AMBIGUOUS, "contradictory truth claims on identical bytes: " + "; ".join(
            sorted({f"{r['source_manifest']}={r['claimed_truth']}" for r in claims})), True
    trusted = [r for r in claims if r.get("trusted")]
    kinds = {r["claimed_truth"] for r in trusted}
    if kinds == {AI}:
        return VERIFIED_AI, trusted[0]["trust_basis"], False
    if kinds == {REAL}:
        return VERIFIED_REAL, trusted[0]["trust_basis"], False
    # What the content is, not how far it is trusted, makes these ambiguous (D-3, D-4).
    every = {r["claimed_truth"] for r in claims}
    if RENDERED in every:
        return AMBIGUOUS, ("rendered face-vector animation; the intake record states it "
                           "carries no camera pixels and is not genuine footage"), False
    if MANIPULATED in every:
        return AMBIGUOUS, ("truth outside the AI/REAL binary (MANIPULATED_OTHER): real camera "
                           "pixels with a documented non-AI edit or synthetic audio"), False
    if claims:
        return INSUFFICIENT, "label present without an allowlisted trust anchor: " + ", ".join(
            sorted({r["trust_basis"] for r in claims})), False
    return UNLABELED, "no manifest or metadata record carries a truth claim", False


def trusted_attributes(records):
    """Only explicit fields of trusted records. No model ever fills these (plan D3)."""
    out = {f: None for f in FACE_FIELDS}
    for r in records:
        if r.get("trusted"):
            for f in FACE_FIELDS:
                if r.get(f) is not None and out[f] is None:
                    out[f] = r[f]
    return out


def canonical_path(paths):
    return min(Path(p).as_posix() for p in paths)


# --- assembly -------------------------------------------------------------------------------

def collect_metadata(corpus):
    anchors, hash_records, outside = [], defaultdict(list), []
    for fn in (r17_records, turnturk_records, r16_records, r15_records, vcd1_records):
        recs, out, anchor = fn(corpus)
        if anchor is not None:
            anchors.append(anchor)
        for digest, rec in recs:
            hash_records[digest].append(rec)
        outside.extend(out)
    for digest, rec in known_generated_records(corpus):
        hash_records[digest].append(rec)
    path_recs, unresolved = csv_manifest_records(corpus)
    path_records = defaultdict(list)
    for rel, rec in path_recs:
        path_records[rel].append(rec)
    return anchors, hash_records, path_records, outside, unresolved


def build_inventory(media, hashes, sizes, probes, hash_records, path_records):
    groups = defaultdict(list)
    for rel in media:
        groups[hashes[rel]].append(rel)
    items = []
    for digest, paths in groups.items():
        paths = sorted(paths)
        canon = canonical_path(paths)
        records = list(hash_records.get(digest, []))
        for p in paths:
            records.extend(path_records.get(p, []))
        records = sorted({json.dumps(r, sort_keys=True): r for r in records}.values(),
                         key=lambda r: json.dumps(r, sort_keys=True))
        status, reason, profile = probes[digest]
        confidence, rationale, conflict = classify(records)
        items.append({
            "canonical_path": canon, "subcorpus": subcorpus_of(canon), "sha256": digest,
            "size_bytes": sizes[canon], "technical_status": status, "technical_reason": reason,
            "format": profile, "duplicate_instances": paths,
            "associated_metadata_records": records, "confidence_classification": confidence,
            "classification_rationale": rationale, "has_metadata_conflict": conflict,
            "trusted_attributes": trusted_attributes(records)})
    return sorted(items, key=lambda x: x["canonical_path"])


def _counter(values):
    return dict(sorted(Counter(values).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def gap_analysis(items, media, sizes, anchors, other_media, links, outside, unresolved):
    dup_paths = sum(len(i["duplicate_instances"]) - 1 for i in items)
    dup_bytes = sum(i["size_bytes"] * (len(i["duplicate_instances"]) - 1) for i in items)
    usable = [i for i in items if i["technical_status"] == INSPECTABLE]
    v_ai = [i for i in usable if i["confidence_classification"] == VERIFIED_AI]
    v_real = [i for i in usable if i["confidence_classification"] == VERIFIED_REAL]

    def trusted_field(item, field):
        return next((r.get(field) for r in item["associated_metadata_records"]
                     if r.get("trusted") and r.get(field)), None)

    gens = _counter(trusted_field(i, "generator_or_source") for i in v_ai)
    domains = _counter(trusted_field(i, "capture_domain") for i in v_real)
    clusters_ai = sorted({trusted_field(i, "source_cluster_id") or "?" for i in v_ai})
    clusters_real = sorted({trusted_field(i, "source_cluster_id") or "?" for i in v_real})
    derivatives_real = sum(1 for i in v_real if trusted_field(i, "derivative"))
    present_gens = {g for g in DESIRED_GENERATORS
                    if any(g.split("/")[0].lower() in str(k).lower() for k in gens)}
    covered = {d for dom in domains for d in DOMAIN_COVERS.get(dom, ())}
    n_usable = len(v_ai) + len(v_real)

    label_present = defaultdict(Counter)
    for i in items:
        if i["confidence_classification"] == INSUFFICIENT:
            r = next(r for r in i["associated_metadata_records"] if r.get("claimed_truth"))
            label_present[i["technical_status"]][
                f"{r['claimed_truth']} | {r.get('family') or r.get('generator_or_source')}"] += 1
    candidates = [i for i in items if any(r.get("allowlist_candidate")
                                          for r in i["associated_metadata_records"])]
    face = {f: _counter(str(i["trusted_attributes"][f]) for i in items) for f in FACE_FIELDS}

    return {
        "schema": SCHEMA_GAP,
        "trust_anchors": anchors,
        "stock_discovery": {
            "media_paths": len(media), "unique_sha256": len(items),
            "duplicate_paths": dup_paths, "duplicate_bytes": dup_bytes,
            "total_bytes": sum(sizes[p] for p in media),
            "paths_by_subcorpus": _counter(subcorpus_of(p) for p in media),
            "unique_by_canonical_subcorpus": _counter(i["subcorpus"] for i in items),
            "excluded_top_level": sorted(EXCLUDED_TOP),
            "out_of_extension_media_not_classified": other_media,
            "symlinks_not_followed": links,
            "manifest_rows_with_unresolved_paths": unresolved},
        "technical_inspectability": {
            "unique": _counter(i["technical_status"] for i in items),
            "unreadable_reasons": _counter(i["technical_reason"] for i in items
                                           if i["technical_status"] == UNREADABLE)},
        "ground_truth_confidence": {
            "unique": {c: sum(1 for i in items if i["confidence_classification"] == c)
                       for c in CONFIDENCE_ORDER},
            "by_technical_status": {s: {c: sum(1 for i in items if i["technical_status"] == s
                                               and i["confidence_classification"] == c)
                                        for c in CONFIDENCE_ORDER}
                                    for s in (INSPECTABLE, UNREADABLE)},
            "ambiguous_reasons": _counter(i["classification_rationale"].split(":")[0]
                                          for i in items
                                          if i["confidence_classification"] == AMBIGUOUS)},
        "metadata_conflict_registry": [
            {"sha256": i["sha256"], "canonical_path": i["canonical_path"],
             "instances": i["duplicate_instances"],
             "claims": sorted({f"{r['source_manifest']}: {r['raw_label']}"
                               for r in i["associated_metadata_records"]
                               if r.get("claimed_truth")})}
            for i in items if i["has_metadata_conflict"]],
        "verified_ai_by_generator": {"inspectable": gens, "source_clusters": clusters_ai},
        "verified_real_by_capture_domain": {"inspectable": domains,
                                            "source_clusters_or_lineages": clusters_real,
                                            "of_which_documented_derivatives": derivatives_real},
        "facial_semantic_metadata": {
            "fields": face, "detector_or_vision_model_run": False,
            "note": "populated only from explicit fields of trusted manifests; none exists"},
        "label_present_candidate_stock": {s: dict(sorted(c.items(), key=lambda kv: -kv[1]))
                                          for s, c in sorted(label_present.items())},
        "allowlist_candidates_not_granted": {
            "external-genuine-vcd1": {
                "unique": len(candidates),
                "inspectable": sum(1 for i in candidates if i["technical_status"] == INSPECTABLE),
                "by_documented_domain": _counter(
                    next(r.get("documented_capture_domain") for r in i["associated_metadata_records"]
                         if r.get("allowlist_candidate")) for i in candidates)}},
        "untracked_anchor_not_granted": {
            "rule": UNTRACKED_BASIS,
            "by_subcorpus_and_class": _counter(
                f"{i['subcorpus']} | {i['confidence_classification']} | {i['technical_status']}"
                for i in items if any(r.get("anchor_matched_untracked")
                                      for r in i["associated_metadata_records"]))},
        "known_stock_outside_scan_root": outside,
        "gap_vs_r18_target": {
            "target_cases": [TARGET_MIN, TARGET_MAX],
            "verified_and_inspectable": {"AI": len(v_ai), "REAL": len(v_real), "total": n_usable},
            "shortfall_total": [max(0, TARGET_MIN - n_usable), max(0, TARGET_MAX - n_usable)],
            "illustrative_even_split_shortfall": {
                "assumption": "half AI / half REAL; the plan names no ratio",
                "AI": [max(0, TARGET_MIN // 2 - len(v_ai)), max(0, TARGET_MAX // 2 - len(v_ai))],
                "REAL": [max(0, TARGET_MIN // 2 - len(v_real)),
                         max(0, TARGET_MAX // 2 - len(v_real))]},
            "desired_generators_present": sorted(present_gens),
            "desired_generators_missing": [g for g in DESIRED_GENERATORS if g not in present_gens],
            "desired_real_domains_present": sorted(covered),
            "desired_real_domains_missing": [d for d in DESIRED_REAL_DOMAINS if d not in covered],
            "face_or_talking_head_documented": sum(
                1 for i in v_ai + v_real if any(v is not None
                                                for v in i["trusted_attributes"].values()))},
    }


def probe_unique(corpus, groups, sizes, workers):
    def one(item):
        digest, rel = item
        probe, err = run_ffprobe(Path(corpus) / rel)
        return digest, technical_profile(probe, err, sizes[rel])
    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, sorted(groups.items())))


def run(args):
    corpus = Path(args.corpus).expanduser().resolve()
    if not corpus.is_dir():
        raise Refusal(f"corpus not found: {corpus}")
    media, other, links = discover(corpus)
    sizes = {p: (corpus / p).stat().st_size for p in media}
    with ThreadPoolExecutor(args.workers) as pool:
        hashes = dict(zip(media, pool.map(lambda p: sha256_file(corpus / p), media)))
    # Identical bytes probe identically, so each digest is probed once, at its canonical path.
    groups = {}
    for p in media:
        groups[hashes[p]] = canonical_path([groups.get(hashes[p], p), p])
    probes = probe_unique(corpus, groups, sizes, args.workers)
    anchors, hash_records, path_records, outside, unresolved = collect_metadata(corpus)
    items = build_inventory(media, hashes, sizes, probes, hash_records, path_records)
    version = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True).stdout
    inventory = {"schema": SCHEMA_INVENTORY, "corpus_root": str(corpus),
                 "ffprobe": version.splitlines()[0] if version else None,
                 "unique_items": len(items), "items": items}
    gap = gap_analysis(items, media, sizes, anchors, other, links, outside, unresolved)
    write_json(args.inventory, inventory)
    write_json(args.gap, gap)
    print(json.dumps({"media_paths": len(media), "unique": len(items),
                      **gap["ground_truth_confidence"]["unique"],
                      **gap["technical_inspectability"]["unique"]}, indent=1))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True)
    p.add_argument("--inventory", required=True)
    p.add_argument("--gap", required=True)
    p.add_argument("--workers", type=int, default=4)
    return p


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
