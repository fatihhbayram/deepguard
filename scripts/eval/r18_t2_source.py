"""R18-T2: source new evaluation media, verify its ground truth, and anchor what is admitted.

    PYTHONPATH=scripts python3 scripts/eval/r18_t2_source.py author-spec \
        --spec scripts/eval/specs/r18_t2_sources.json
    PYTHONPATH=scripts python3 scripts/eval/r18_t2_source.py audit-local \
        --corpus ~/deepguard-corpus --intake ~/deepguard-corpus/r18_t2_intake \
        --t1-inventory scripts/eval/baselines/r18_t1_corpus_inventory.json
    PYTHONPATH=scripts python3 scripts/eval/r18_t2_source.py ingest-external \
        --spec scripts/eval/specs/r18_t2_sources.json --intake ~/deepguard-corpus/r18_t2_intake
    PYTHONPATH=scripts python3 scripts/eval/r18_t2_source.py generate-anchor \
        --spec scripts/eval/specs/r18_t2_sources.json --intake ~/deepguard-corpus/r18_t2_intake \
        --t1-inventory scripts/eval/baselines/r18_t1_corpus_inventory.json \
        --out scripts/eval/specs/r18_t2_verified_anchors.json

Applies `r18_t2_implementation_plan.md` (rev 4) and decides nothing beyond it.

Ground truth comes from primary provenance only: a publisher's own statement of what a file is
(a dataset protocol, a per-file generation record, a per-clip licence line), bound to the bytes
by a digest the publisher serves (Git LFS SHA-256, a GCS object MD5) or, for a corpus already on
disk, by re-downloading the upstream object and comparing. A file inside a corpus never vouches
for that corpus. Every external entry carries the seven mandatory fields; an entry whose usage
rights or provenance are not independently established stays a candidate and is never
downloaded. No detector score, no production pipeline and no audio or speech model is used.

Derived annotations are kept apart from ground truth. Face presence comes from the frozen YuNet
specification below and nothing else; if the runtime or the model digest differs, the detector
is not run and the annotation is ERROR while the ground truth stands. Talking-head status is
DETERMINED only from a trusted primary-source annotation, NOT_APPLICABLE when no face is
present, and UNKNOWN otherwise.

All media are written under `--intake` (outside the repository). Pre-existing corpus files are
only read. The anchor is a pure function of the spec, the intake ledgers and the T1 inventory:
re-running `generate-anchor` rewrites it byte for byte.
"""

import argparse
import base64
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import r18_t1_inventory as t1

REPO = Path(__file__).resolve().parents[2]
SCHEMA_SPEC = "r18-t2-sources-1"
SCHEMA_ANCHOR = "r18-t2-verified-anchors-1"
SCHEMA_LEDGER = "r18-t2-ingest-ledger-1"
SCHEMA_AUDIT = "r18-t2-local-audit-1"
LEDGER_NAME, AUDIT_NAME = "ingest_ledger.json", "local_audit.json"
USER_AGENT = "deepguard-r18-t2-sourcing/1"

VERIFIED_AI, VERIFIED_REAL = "VERIFIED_AI", "VERIFIED_REAL"
TRUTHS = (VERIFIED_AI, VERIFIED_REAL)
ELIGIBLE, CANDIDATE_ONLY, AMBIGUOUS = "ELIGIBLE", "CANDIDATE_ONLY", "AMBIGUOUS"

# Plan §4.1: the seven fields every external entry must carry.
MANDATORY_FIELDS = ("source_uri", "publisher_or_owner", "provenance_evidence",
                    "license_or_usage_basis", "download_method", "source_cluster",
                    "expected_truth")
# Usage bases that are independently established grants of use. Anything else — and in
# particular official promotional publication or a fair-use argument (Architect Fix 4) — leaves
# the entry a candidate.
ACCEPTED_LICENSES = frozenset({"Public-Domain", "CC0-1.0", "CC-BY-3.0", "CC-BY-4.0",
                               "CC-BY-SA-4.0", "Apache-2.0", "CDLA-Permissive-2.0"})
NOT_A_LICENSE = ("promotional", "fair use", "fair-use", "unknown", "unclear", "not established")

# Plan §5.2 / D-3: the frozen face-annotation tooling. The model digest is the verified SHA-256
# of opencv_zoo's face_detection_yunet_2023mar.onnx (Git LFS oid; the same pin as
# apps/api/app/face_detector.py); the plan's original value matched no YuNet artifact and was
# corrected by Architect ruling D-A (R18-T2 PASS WITH FIXES).
FACE_SPEC = {
    "tool": "opencv_dnn_yunet",
    "model_artifact": "face_detection_yunet_2023mar.onnx",
    "model_sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "opencv_version": "5.0.0",
    "backend": "DNN_BACKEND_OPENCV",
    "target": "DNN_TARGET_CPU",
    "score_threshold": 0.80,
    "nms_threshold": 0.3,
    "top_k": 5000,
    "sampling": "t = k * 1.0 s for integer k >= 0, t <= duration",
    "presence_rule": "face_present = true iff a face scoring >= 0.80 is found in >= 30% of the "
                     "decoded sampled frames",
    "presence_fraction": 0.30,
    # Architect D-H: YuNet misses faces far larger than its input scale (a close-up phone face
    # at 1080x1920 scores nothing; the same frame at long side 640 scores 0.93). Every sampled
    # frame is also scored on a copy resized to long side 640 when the native long side exceeds
    # it; the frame is face-positive if either scale clears the threshold.
    "preprocessing": "native + long-side-640 copy (INTER_AREA, aspect preserved, short side "
                     "rounded half-up) when native long side > 640; face-positive if either "
                     "scale scores >= threshold",
    "secondary_long_side": 640,
}
FACE_MODEL_DEFAULT = Path.home() / ".cache/deepguard/face_manipulation/yunet" / \
    FACE_SPEC["model_artifact"]

# Edit screen for REAL footage whose source does not document a single unedited take (plan
# §4.2: cuts are conventional edits and may not be labelled raw camera). FFmpeg's scene score is
# a signal-processing measure; it can only quarantine a clip, never admit one or assign truth.
SCENE_CUT_THRESHOLD = 0.4


class Refusal(SystemExit):
    """A stop condition: nothing is written."""


# --- small helpers ----------------------------------------------------------------------------

def write_json(path, payload):
    t1.write_json(path, payload)


def read_json(path):
    return json.loads(Path(path).read_text())


def short_hash(text, n=12):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def http_get(url, timeout=120, retries=4):
    """GET with retries and a polite back-off. Returns (bytes, headers)."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), resp.headers
        except Exception as exc:  # network errors are retried, then reported
            last = exc
            time.sleep(2 * (attempt + 1))
    raise Refusal(f"GET failed after {retries} attempts: {url}: {last}")


def http_json(url):
    return json.loads(http_get(url)[0])


def download(url, dest, expected=None, retries=3):
    """Stream `url` to `dest` while hashing it; verify against the publisher digest if given.

    `expected` is {"algorithm": "sha256" | "md5_base64", "value": ...}. On a mismatch the
    partial file is removed and the status says so: a publisher digest that does not match is
    never overridden.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    error = None
    for attempt in range(retries):
        sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as resp, open(part, "wb") as out:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    out.write(block)
                    sha.update(block)
                    md5.update(block)
                    size += len(block)
            error = None
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(3 * (attempt + 1))
    if error:
        part.unlink(missing_ok=True)
        return {"status": "DOWNLOAD_FAILED", "error": error}
    got = {"sha256": sha.hexdigest(), "md5_base64": base64.b64encode(md5.digest()).decode(),
           "size_bytes": size}
    if expected and got[expected["algorithm"]] != expected["value"]:
        part.unlink(missing_ok=True)
        return {"status": "DIGEST_MISMATCH", **got,
                "error": f"{expected['algorithm']} {got[expected['algorithm']]} != publisher "
                         f"{expected['value']}"}
    part.replace(dest)
    return {"status": "OK", **got}


# --- provenance and licensing validation (plan §4.1) ------------------------------------------

def validate_entry(entry):
    """Return (ELIGIBLE | CANDIDATE_ONLY, reason). Only ELIGIBLE entries are ever downloaded."""
    missing = [f for f in MANDATORY_FIELDS if not str(entry.get(f) or "").strip()]
    if missing:
        return CANDIDATE_ONLY, "missing mandatory field(s): " + ", ".join(missing)
    if entry["expected_truth"] not in TRUTHS:
        return CANDIDATE_ONLY, f"expected_truth {entry['expected_truth']!r} is not a verified class"
    basis = entry["license_or_usage_basis"].lower()
    if entry.get("license_id") not in ACCEPTED_LICENSES or any(m in basis for m in NOT_A_LICENSE):
        return CANDIDATE_ONLY, ("usage rights not independently established: "
                                f"{entry.get('license_id')!r} / {entry['license_or_usage_basis']!r}")
    if entry.get("provenance_status") != "ESTABLISHED":
        return CANDIDATE_ONLY, f"provenance {entry.get('provenance_status')!r}, not ESTABLISHED"
    if not entry.get("download_url") or not entry.get("expected_digest"):
        return CANDIDATE_ONLY, "no pinned download URL with a publisher digest"
    return ELIGIBLE, None


# --- derived annotations (plan §5) ------------------------------------------------------------

def face_guard(model_path, cv2_module=None):
    """None when the frozen runtime and artifact are exactly in place, else the reason."""
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError:
            return "opencv not importable"
    version = getattr(cv2_module, "__version__", None)
    if version != FACE_SPEC["opencv_version"]:
        return f"OpenCV runtime {version} != pinned {FACE_SPEC['opencv_version']}"
    model_path = Path(model_path)
    if model_path.name != FACE_SPEC["model_artifact"]:
        return f"model artifact {model_path.name} != pinned {FACE_SPEC['model_artifact']}"
    if not model_path.is_file():
        return f"model artifact not found: {model_path}"
    digest = t1.sha256_file(model_path)
    if digest != FACE_SPEC["model_sha256"]:
        return f"model sha256 {digest} != pinned {FACE_SPEC['model_sha256']}"
    return None


def sample_timestamps(duration):
    if not duration or duration <= 0:
        return []
    return [float(k) for k in range(int(math.floor(duration)) + 1) if k * 1.0 <= duration]


def grab_frame(path, t):
    """The frame presented at time `t`, decoded by FFmpeg, as a BGR array (or None)."""
    import cv2
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    if out.returncode != 0 or not out.stdout:
        return None
    return cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)


def _face_base():
    return {"value": None, "method": "offline_detector", "tool": FACE_SPEC["tool"],
            "version": FACE_SPEC["opencv_version"], "model_artifact": FACE_SPEC["model_artifact"],
            "model_sha256": FACE_SPEC["model_sha256"], "threshold": FACE_SPEC["score_threshold"],
            "sampling": FACE_SPEC["sampling"], "preprocessing": FACE_SPEC["preprocessing"],
            "confidence": None}


def secondary_size(w, h, long_side=None):
    """(w, h) of the long-side-640 copy, or None when the native long side is not above it."""
    long_side = long_side or FACE_SPEC["secondary_long_side"]
    if max(w, h) <= long_side:
        return None
    scale = long_side / max(w, h)
    short = int(math.floor(min(w, h) * scale + 0.5))
    return (long_side, short) if w >= h else (short, long_side)


def face_annotation(path, duration, model_path, cv2_module=None, grab=grab_frame):
    """Frozen YuNet face presence. Never raises: any failure is an ERROR/UNKNOWN annotation."""
    base = _face_base()
    reason = face_guard(model_path, cv2_module)
    if reason:
        return {**base, "status": "ERROR", "error": "annotation halted: " + reason}
    if cv2_module is None:
        import cv2 as cv2_module
    times = sample_timestamps(duration)
    if not times:
        return {**base, "status": "UNKNOWN", "error": "no positive duration to sample"}
    decoded = hits = native_hits = secondary_hits = 0
    best = 0.0
    threshold = FACE_SPEC["score_threshold"]
    try:
        detector = None

        def scores_of(image):
            h, w = image.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(image)
            return [] if faces is None else [float(f[-1]) for f in faces]

        for t in times:
            frame = grab(path, t)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if detector is None:
                detector = cv2_module.FaceDetectorYN.create(
                    str(model_path), "", (w, h), threshold,
                    FACE_SPEC["nms_threshold"], FACE_SPEC["top_k"],
                    cv2_module.dnn.DNN_BACKEND_OPENCV, cv2_module.dnn.DNN_TARGET_CPU)
            native = scores_of(frame)
            size = secondary_size(w, h)
            secondary = [] if size is None else scores_of(
                cv2_module.resize(frame, size, interpolation=cv2_module.INTER_AREA))
            decoded += 1
            n_hit = bool(native) and max(native) >= threshold
            s_hit = bool(secondary) and max(secondary) >= threshold
            native_hits += n_hit
            secondary_hits += s_hit
            hits += n_hit or s_hit
            best = max([best] + native + secondary)
    except Exception as exc:
        return {**base, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    if decoded == 0:
        return {**base, "status": "UNKNOWN", "error": "no sampled frame decoded"}
    fraction = hits / decoded
    return {**base, "value": fraction >= FACE_SPEC["presence_fraction"], "status": "DETERMINED",
            "confidence": round(fraction, 4), "frames_sampled": len(times),
            "frames_decoded": decoded, "frames_with_face": hits,
            "frames_with_face_native": native_hits, "frames_with_face_secondary": secondary_hits,
            "max_score": round(best, 4)}


def speech_annotation(face, primary=None):
    """Plan §5.3: trusted primary source -> DETERMINED; no face -> NOT_APPLICABLE; else UNKNOWN."""
    base = {"value": None, "method": "source_annotation", "tool": "none", "version": "1.0.0",
            "confidence": None}
    if primary is not None:
        return {**base, "value": bool(primary["value"]), "status": "DETERMINED",
                "evidence": primary["evidence"]}
    if face.get("status") == "DETERMINED" and face.get("value") is False:
        return {**base, "status": "NOT_APPLICABLE"}
    return {**base, "status": "UNKNOWN"}


def scene_cuts(path, threshold=SCENE_CUT_THRESHOLD):
    """Number of frames whose FFmpeg scene score exceeds `threshold`, or None if it failed."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:v:0", "-an",
           "-vf", f"select='gt(scene,{threshold})',metadata=print", "-f", "null", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0:
        return None
    return out.stderr.count("lavfi.scene_score=")


def inspect(path):
    probe, error = t1.run_ffprobe(path)
    status, reason, profile = t1.technical_profile(probe, error, Path(path).stat().st_size)
    return status, reason, profile


# --- author-spec: pinned upstream listings -> the seven-field source specification ------------

RAPIDATA_T2V = {"repo": "Rapidata/text-2-video-human-preferences-moonvalley-marey",
                "revision": "cdbf62f223abf5864d3d7cf9b6afeb3391fcc771", "dir": "Videos",
                "mode": "text_to_video"}
RAPIDATA_I2V = {"repo": "Rapidata/image-to-video-human-preference-seedance-1-pro",
                "revision": "987399c7d471f15d556598f79108c4c4f8c0b74a", "dir": "Videos/hailuo-02",
                "mode": "image_to_video"}
# Rapidata's model token -> (family, model as the publisher names it). Names are taken from the
# dataset cards at the pinned revisions ("Pika 2.0", "Runway ML Alpha", "Luma Ray 2", ...).
RAPIDATA_MODELS = {
    "sora": ("Sora", "Sora (OpenAI)"),
    "veo3": ("Veo", "Veo 3 (Google)"),
    "veo2": ("Veo", "Veo 2 (Google)"),
    "alpha": ("Runway", "Runway ML Alpha"),
    "kling-v2.1-master-25-7-25": ("Kling", "Kling v2.1 Master (Kuaishou)"),
    "ray2": ("Luma", "Luma Ray 2"),
    "hailuo-02": ("Hailuo/MiniMax", "Hailuo 02 (MiniMax)"),
    "pika": ("Pika", "Pika 2.0"),
    "pika2.2": ("Pika", "Pika 2.2"),
}
# Plan §7 quotas for new AI intake, split across a family's model versions.
RAPIDATA_QUOTAS = (("sora", 10), ("veo3", 8), ("veo2", 7), ("alpha", 25),
                   ("kling-v2.1-master-25-7-25", 25), ("ray2", 25), ("pika", 10),
                   ("pika2.2", 10), ("hailuo-02", 25))

UGC_BUCKET = "ugc-dataset"
UGC_PREFIX = "original_videos_h264/"
UGC_RAW_PREFIX = "original_videos/"
UGC_RESOLUTIONS = ("720P", "1080P")
# Category -> (capture domain, candidate quota). Animation, Gaming, LyricVideo, MusicVideo, VR,
# HDR, CoverSong, LiveMusic, NewsClip and TelevisionClip are not taken: rendered, composited,
# stitched or produced footage is not camera-real evidence.
# Quotas were raised once after the first screen passed 41 of 76 (plan §7: REAL to ~175).
UGC_CATEGORIES = (("Lecture", "ugc_lecture", 30), ("Vlog", "ugc_vlog", 45),
                  ("VerticalVideo", "ugc_vertical_smartphone", 40),
                  ("Sports", "ugc_sports_outdoor", 35), ("HowTo", "ugc_howto", 30))
UGC_ATTRIBUTION_RE = re.compile(
    r"^(\S+) is an? (.*?)excerpt from '(.*)' by '(.*)' licensed under (.*)$")
UGC_LICENSES = {"CC BY 4.0": "CC-BY-4.0", "CC BY 3.0": "CC-BY-3.0"}


def hf_tree(repo, revision, directory):
    """Every file under `directory` at `revision`, with its Git LFS SHA-256 (paginated)."""
    url = (f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/"
           f"{urllib.parse.quote(directory)}?recursive=true")
    files = []
    while url:
        body, headers = http_get(url)
        files += [f for f in json.loads(body) if f.get("type") == "file"]
        match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link") or "")
        url = match.group(1) if match else None
    return files


def hf_rows(repo):
    """file_name -> (model, prompt) from the dataset viewer's rows of the train split."""
    first = http_json(f"https://datasets-server.huggingface.co/rows?dataset={repo}"
                      "&config=default&split=train&offset=0&length=1")
    total, rows = first["num_rows_total"], {}
    for offset in range(0, total, 100):
        page = http_json(f"https://datasets-server.huggingface.co/rows?dataset={repo}"
                         f"&config=default&split=train&offset={offset}&length=100")
        for r in page["rows"]:
            x = r["row"]
            for side in ("1", "2"):
                rows.setdefault(x["file_name" + side], (x["model" + side], x["prompt"].strip()))
    return rows


def rapidata_entries(source, rows, tree, quotas, usage):
    """Seven-field entries for one Rapidata repo. One video per prompt per model; prompts are
    spread across models so that as few families as possible share a prompt cluster."""
    license_basis = (f"Apache-2.0 — license field of the {source['repo']} dataset card at "
                     f"revision {source['revision']}, granted by the party that generated and "
                     "published the videos")
    by_model = defaultdict(dict)
    for f in tree:
        name = f["path"].rsplit("/", 1)[-1]
        if not name.endswith(".mp4") or not f.get("lfs"):
            continue
        attributed = rows.get(name)
        # The publisher's row names the model and the file name must start with the same token.
        # `_ui` variants are undocumented by the card and are left out.
        if attributed is None or attributed[0] not in quotas \
                or not name.startswith(attributed[0] + "_") or name.endswith("_ui.mp4"):
            continue
        token, prompt = attributed
        by_model[token].setdefault(prompt, (name, f))
        if name < by_model[token][prompt][0]:
            by_model[token][prompt] = (name, f)
    entries = []
    for token, quota in quotas.items():
        family, model = RAPIDATA_MODELS[token]
        ranked = sorted(by_model[token].items(),
                        key=lambda kv: (usage[kv[0]], short_hash(kv[1][0], 64)))
        for prompt, (name, f) in ranked[:quota]:
            usage[prompt] += 1
            cluster = f"rapidata_prompt_{short_hash(prompt)}"
            entries.append({
                "item_id": f"rapidata:{source['repo'].split('/')[1]}:{name}",
                "source_uri": f"https://huggingface.co/datasets/{source['repo']}/blob/"
                              f"{source['revision']}/{f['path']}",
                "publisher_or_owner": "Rapidata AG (generated with the named model and published "
                                      "on Hugging Face)",
                "provenance_evidence": (
                    f"Publisher's per-file generation record: dataset {source['repo']} @ "
                    f"{source['revision']} lists '{name}' as an output of model token "
                    f"'{token}' ({model}) for the prompt below; the file name starts with the "
                    "same token. Bytes bound by the publisher's Git LFS SHA-256."),
                "license_or_usage_basis": license_basis,
                "download_method": "direct_http_urllib",
                "source_cluster": cluster,
                "expected_truth": VERIFIED_AI,
                "license_id": "Apache-2.0",
                "provenance_status": "ESTABLISHED",
                "provenance_tier": "third_party_benchmark_generation_record",
                "download_url": f"https://huggingface.co/datasets/{source['repo']}/resolve/"
                                f"{source['revision']}/{urllib.parse.quote(f['path'])}",
                "expected_digest": {"algorithm": "sha256", "value": f["lfs"]["oid"]},
                "expected_size": f["lfs"]["size"],
                "source_collection_id": source["repo"],
                "generator_family": family,
                "generator_model": model,
                "generation_mode": source["mode"],
                "prompt": prompt,
                "capture_domain": None,
                "subject_id": None,
                "speech_primary_annotation": None,
                "edit_screen_required": False,
            })
    return entries


def gcs_list(prefix):
    items, token = [], None
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{UGC_BUCKET}/o?prefix="
               f"{urllib.parse.quote(prefix)}&maxResults=1000&fields=nextPageToken,"
               "items(name,size,md5Hash,generation,timeCreated)")
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        page = http_json(url)
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items


def ugc_stem(name):
    return name.rsplit("/", 1)[-1].split("_crf_")[0].rsplit(".", 1)[0]


def ugc_entries(attribution_text, attribution_meta, h264, raw):
    """One CC-BY clip per YouTube creator across the selected categories."""
    attribution = {}
    for line in attribution_text.splitlines():
        m = UGC_ATTRIBUTION_RE.match(line.strip())
        if m:
            attribution[m.group(1)] = {"title": m.group(3), "creator": m.group(4),
                                       "license": m.group(5)}
    raw_created = {ugc_stem(i["name"]): i["timeCreated"] for i in raw if i["name"].endswith(".mkv")}
    by_category = defaultdict(list)
    for obj in h264:
        stem = ugc_stem(obj["name"])
        category, _, rest = stem.partition("_")
        if rest.split("-")[0] in UGC_RESOLUTIONS and stem in attribution:
            by_category[category].append((short_hash(stem, 64), stem, obj))
    used_creators, entries = set(), []
    for category, domain, quota in UGC_CATEGORIES:
        taken = 0
        for _, stem, obj in sorted(by_category[category]):
            att = attribution[stem]
            if taken == quota:
                break
            if att["creator"] in used_creators or att["license"] not in UGC_LICENSES:
                continue
            used_creators.add(att["creator"])
            taken += 1
            created = raw_created.get(stem)
            entries.append({
                "item_id": f"ytugc:{stem}",
                "source_uri": f"gs://{UGC_BUCKET}/{obj['name']}#generation={obj['generation']}",
                "publisher_or_owner": f"'{att['creator']}' (YouTube creator, licensor); "
                                      "excerpted and published by Google in the YouTube UGC "
                                      "dataset",
                "provenance_evidence": (
                    f"YouTube UGC dataset (Wang, Inguva, Adsumilli, MMSP 2019), category "
                    f"{category}: ATTRIBUTION line \"{stem} is an audio-removed excerpt from "
                    f"'{att['title']}' by '{att['creator']}'\". Raw original object "
                    f"{UGC_RAW_PREFIX}…/{stem}.mkv created {created} — the footage existed "
                    "before photorealistic text-to-video generators were publicly available. "
                    "Bytes bound by the publisher's GCS MD5 at a pinned object generation."),
                "license_or_usage_basis": (
                    f"{att['license']} — per-clip licence in gs://{UGC_BUCKET}/ATTRIBUTION "
                    f"(generation {attribution_meta['generation']}), licensed by the creator"),
                "download_method": "direct_http_urllib",
                "source_cluster": f"ytugc_creator_{short_hash(att['creator'])}",
                "expected_truth": VERIFIED_REAL,
                "license_id": UGC_LICENSES[att["license"]],
                "provenance_status": "ESTABLISHED" if created else "UNCLEAR",
                "provenance_tier": "published_dataset_with_per_clip_attribution",
                "download_url": (f"https://storage.googleapis.com/download/storage/v1/b/"
                                 f"{UGC_BUCKET}/o/{urllib.parse.quote(obj['name'], safe='')}"
                                 f"?generation={obj['generation']}&alt=media"),
                "expected_digest": {"algorithm": "md5_base64", "value": obj["md5Hash"]},
                "expected_size": int(obj["size"]),
                "source_collection_id": "google/youtube-ugc",
                "generator_family": None,
                "generator_model": None,
                "generation_mode": None,
                "prompt": None,
                "capture_domain": domain,
                "subject_id": f"ytugc_creator_{short_hash(att['creator'])}",
                "speech_primary_annotation": None,
                "edit_screen_required": True,
            })
    return entries


# Sources considered and not admitted. They keep the seven fields so the refusal is auditable.
CANDIDATE_ONLY_SOURCES = (
    {"item_id": "candidate:kling-official-showcase", "source_uri": "https://klingai.com/",
     "publisher_or_owner": "Kuaishou Technology",
     "provenance_evidence": "Official provider showcase; generation provenance plausible.",
     "license_or_usage_basis": "official promotional media — no licence grant found",
     "download_method": "not_downloaded", "source_cluster": "kling_official_showcase",
     "expected_truth": VERIFIED_AI, "license_id": None, "provenance_status": "ESTABLISHED"},
    {"item_id": "candidate:luma-official-showcase", "source_uri": "https://lumalabs.ai/",
     "publisher_or_owner": "Luma AI", "provenance_evidence": "Official provider showcase.",
     "license_or_usage_basis": "official promotional media — no licence grant found",
     "download_method": "not_downloaded", "source_cluster": "luma_official_showcase",
     "expected_truth": VERIFIED_AI, "license_id": None, "provenance_status": "ESTABLISHED"},
    {"item_id": "candidate:hailuo-official-showcase", "source_uri": "https://hailuoai.video/",
     "publisher_or_owner": "MiniMax", "provenance_evidence": "Official provider showcase.",
     "license_or_usage_basis": "official promotional media — no licence grant found",
     "download_method": "not_downloaded", "source_cluster": "hailuo_official_showcase",
     "expected_truth": VERIFIED_AI, "license_id": None, "provenance_status": "ESTABLISHED"},
    {"item_id": "candidate:pika-official-showcase", "source_uri": "https://pika.art/",
     "publisher_or_owner": "Pika Labs", "provenance_evidence": "Official provider showcase.",
     "license_or_usage_basis": "official promotional media — no licence grant found",
     "download_method": "not_downloaded", "source_cluster": "pika_official_showcase",
     "expected_truth": VERIFIED_AI, "license_id": None, "provenance_status": "ESTABLISHED"},
    {"item_id": "candidate:hf-nyuuzyou-klingai",
     "source_uri": "https://huggingface.co/datasets/nyuuzyou/klingai",
     "publisher_or_owner": "third-party scraper (nyuuzyou) of the klingai.com public gallery",
     "provenance_evidence": "Gallery scrape; the generating users are not the publisher.",
     "license_or_usage_basis": "CC0-1.0 declared by the scraper, not by the rights holders — "
                               "unclear",
     "download_method": "not_downloaded", "source_cluster": "hf_nyuuzyou_klingai",
     "expected_truth": VERIFIED_AI, "license_id": None, "provenance_status": "UNCLEAR"},
    {"item_id": "candidate:hf-bitmind-klingai-videos",
     "source_uri": "https://huggingface.co/datasets/bitmind/klingai-videos",
     "publisher_or_owner": "BitMind (third-party collector)",
     "provenance_evidence": "Collected outputs; per-file generation record not published.",
     "license_or_usage_basis": "Apache-2.0 declared by the collector; upstream rights unclear",
     "download_method": "not_downloaded", "source_cluster": "hf_bitmind_klingai",
     "expected_truth": VERIFIED_AI, "license_id": None, "provenance_status": "UNCLEAR"},
    {"item_id": "deferred:nasa-interview-reels",
     "source_uri": "https://images-api.nasa.gov/search?q=interview&media_type=video",
     "publisher_or_owner": "NASA (US federal government)",
     "provenance_evidence": "Official institutional recordings with per-item metadata.",
     "license_or_usage_basis": "Public-Domain (17 U.S.C. §105; NASA media guidelines) — "
                               "deferred: produced reels with cuts and lower-third graphics, and "
                               "the publisher serves no content digest",
     "download_method": "not_downloaded", "source_cluster": "nasa_interview_reels",
     "expected_truth": VERIFIED_REAL, "license_id": "Public-Domain",
     "provenance_status": "ESTABLISHED"},
)


def author_spec(args):
    spec_path = Path(args.spec)
    previous = read_json(spec_path) if spec_path.is_file() else {}
    usage = Counter()
    t2v_tree = hf_tree(RAPIDATA_T2V["repo"], RAPIDATA_T2V["revision"], RAPIDATA_T2V["dir"])
    i2v_tree = hf_tree(RAPIDATA_I2V["repo"], RAPIDATA_I2V["revision"], RAPIDATA_I2V["dir"])
    t2v_rows, i2v_rows = hf_rows(RAPIDATA_T2V["repo"]), hf_rows(RAPIDATA_I2V["repo"])
    quotas = dict(RAPIDATA_QUOTAS)
    t2v_quotas = {k: v for k, v in quotas.items() if k != "hailuo-02"}
    entries = rapidata_entries(RAPIDATA_T2V, t2v_rows, t2v_tree, t2v_quotas, usage)
    i2v_usage = Counter()
    entries += rapidata_entries(RAPIDATA_I2V, i2v_rows, i2v_tree, {"hailuo-02": quotas["hailuo-02"]},
                                i2v_usage)
    for e in entries:  # the i2v benchmark has its own prompt set: keep its clusters distinct
        if e["generation_mode"] == "image_to_video":
            e["source_cluster"] = e["source_cluster"].replace("rapidata_prompt_",
                                                              "rapidata_i2v_prompt_")
    attribution_obj = next(o for o in gcs_list("ATTRIBUTION") if o["name"] == "ATTRIBUTION")
    attribution_text = http_get(
        f"https://storage.googleapis.com/download/storage/v1/b/{UGC_BUCKET}/o/ATTRIBUTION"
        f"?generation={attribution_obj['generation']}&alt=media")[0].decode("utf-8")
    entries += ugc_entries(attribution_text, attribution_obj, gcs_list(UGC_PREFIX),
                           gcs_list(UGC_RAW_PREFIX))
    entries.sort(key=lambda e: e["item_id"])
    spec = {
        "schema": SCHEMA_SPEC,
        "plan": "r18_t2_implementation_plan.md rev 4 (Architect PASS)",
        "mandatory_fields": list(MANDATORY_FIELDS),
        "accepted_license_ids": sorted(ACCEPTED_LICENSES),
        "frozen_face_annotation": FACE_SPEC,
        "edit_screen": {"tool": "ffmpeg select='gt(scene,T)'", "threshold": SCENE_CUT_THRESHOLD,
                        "use": "quarantine only; applies where edit_screen_required"},
        "pinned_sources": {
            "rapidata_t2v": RAPIDATA_T2V, "rapidata_i2v": RAPIDATA_I2V,
            "youtube_ugc": {"bucket": UGC_BUCKET, "prefix": UGC_PREFIX,
                            "attribution_generation": attribution_obj["generation"],
                            "attribution_md5_base64": attribution_obj["md5Hash"],
                            "categories": [list(c) for c in UGC_CATEGORIES],
                            "resolutions": list(UGC_RESOLUTIONS)},
        },
        "selection_rules": [
            "Rapidata: one video per (model, prompt); the publisher's row and the file name must "
            "name the same model; prompts least used so far first, ties by SHA-256 of the file "
            "name; quotas per plan §7.",
            "YouTube UGC: 720P/1080P H.264 originals with an ATTRIBUTION line; one clip per "
            "creator across categories in the listed order; within a category by SHA-256 of the "
            "clip name; quotas are candidate counts before the edit and visual screens.",
        ],
        "entries": entries,
        "candidate_only_sources": list(CANDIDATE_ONLY_SOURCES),
        # Visual content-type screen (camera footage vs screencast/CGI/graphics). It can only
        # quarantine; it never assigns truth. Kept across re-authoring.
        "visual_review": previous.get("visual_review", {}),
    }
    write_json(spec_path, spec)
    print(json.dumps(dict(Counter(f"{e['expected_truth']}|{e.get('generator_model') or e.get('capture_domain')}"
                                  for e in entries)), indent=1))


# --- audit-local: internal candidates against primary provenance (plan §3.1) ------------------

VCD1_DIR = "external-genuine-vcd1"
VCD1_COMMIT = "e4c6598cb349bcf2f2cd5df55ef76cf58e947e77"
VCD1_REPO = "microsoft/VCD"
VCD1_PAPER = ("Naderi et al., 'VCD: A Video Conferencing Dataset for Video Compression', "
              "ICASSP 2024, pp. 3970-3974, arXiv:2309.07376")
VCD1_UPSTREAM_FILES = {"LICENSE-DATA": "lineage/upstream_LICENSE-DATA",
                       "README.md": "lineage/upstream_README.md",
                       "download/mp4_list.txt": "lineage/upstream_mp4_list.txt"}
VCD1_DOMAINS = {"th": "webcam_talking_head_scenario",
                "th-m": "smartphone_handheld_talking_head_scenario"}
BUILDER_SCRIPTS = ("scripts/eval/build_corpus.py", "scripts/eval/build_corpus_r7t8.py",
                   "scripts/eval/build_corpus_r7t9.py", "scripts/calibration/fetch_corpus.py")
MIRROR_RE = re.compile(r'"(34data/[^"]+)",\s*"([0-9a-f]{40})"')


def vcd1_audit(corpus, intake, model_path):
    root = Path(corpus) / VCD1_DIR
    manifest = read_json(root / "vcd1_intake_manifest.json")
    commit = http_json(f"https://api.github.com/repos/{VCD1_REPO}/commits/{VCD1_COMMIT}")
    upstream, bodies = {}, {}
    for rel, local in VCD1_UPSTREAM_FILES.items():
        body = http_get(f"https://raw.githubusercontent.com/{VCD1_REPO}/{VCD1_COMMIT}/{rel}")[0]
        upstream[rel] = {"upstream_sha256": hashlib.sha256(body).hexdigest(),
                         "local_copy_sha256": t1.sha256_file(root / local)}
        bodies[rel] = body
    listed = set(bodies["download/mp4_list.txt"].decode().split())
    repo_checks = {
        "commit_resolves": commit.get("sha") == VCD1_COMMIT,
        "commit_date": (commit.get("commit") or {}).get("committer", {}).get("date"),
        "upstream_files": upstream,
        "upstream_files_match_local_copies": all(v["upstream_sha256"] == v["local_copy_sha256"]
                                                 for v in upstream.values()),
        "license_matches_manifest": upstream["LICENSE-DATA"]["upstream_sha256"]
        == manifest["license"]["license_file_sha256"],
    }
    repo_ok = all(v for k, v in repo_checks.items() if isinstance(v, bool))

    def one(rec):
        local = root / rec["relative_path"]
        local_sha = t1.sha256_file(local)
        cluster = f"vcd1_subject_{rec['subject_id']}"
        dest = Path(intake) / "vcd1_upstream" / f"{cluster}_{rec['sha256_local_freeze'][:12]}.mp4"
        if dest.is_file() and t1.sha256_file(dest) == rec["sha256_local_freeze"]:
            fetched = {"status": "OK", "sha256": rec["sha256_local_freeze"]}
        else:
            fetched = download(rec["upstream_url"], dest,
                               {"algorithm": "sha256", "value": rec["sha256_local_freeze"]})
        checks = {"listed_in_upstream_mp4_list": rec["upstream_object"] in listed,
                  "local_bytes_match_r11_freeze": local_sha == rec["sha256_local_freeze"],
                  "upstream_redownload_matches": fetched["status"] == "OK"}
        failed = [k for k, v in checks.items() if not v]
        status, reason, profile = inspect(local)
        face = face_annotation(local, (profile or {}).get("duration_s"), model_path)
        return {
            "item_id": f"vcd1:{rec['upstream_object']}",
            "sha256": local_sha,
            "canonical_path": f"{VCD1_DIR}/{rec['relative_path']}",
            "intake_copy": dest.relative_to(intake).as_posix() if dest.is_file() else None,
            "technical_status": status, "technical_reason": reason, "format": profile,
            "checks": checks,
            "decision": VERIFIED_REAL if repo_ok and not failed else CANDIDATE_ONLY,
            "decision_reason": None if repo_ok and not failed else
            "failed: " + ", ".join(failed + ([] if repo_ok else ["upstream repository checks"])),
            "ground_truth_provenance": {
                "ground_truth": VERIFIED_REAL, "generator_family": None, "generator_model": None,
                "source_type": "published_dataset_protocol",
                "source_uri": rec["upstream_url"],
                "publisher_or_owner": "Microsoft (VCD dataset); recorded by consenting "
                                      "participants",
                "provenance_evidence": (
                    f"{VCD1_PAPER}; github.com/{VCD1_REPO} @ {VCD1_COMMIT[:12]} README: "
                    f"scenario '{rec['subset']}' is "
                    + ("webcam talking-head recordings by each participant 'without further "
                       "processing'" if rec["subset"] == "th" else
                       "handheld mobile talking-head recordings, indoor and outdoor")
                    + "; the TH-OB/TH-BB scenarios with Teams background effects are excluded. "
                      "Upstream object re-downloaded in T2 and byte-identical to the local copy."),
                "license_or_usage_basis": "CDLA-Permissive-2.0 (LICENSE-DATA @ "
                                          f"{VCD1_COMMIT[:12]}, sha256 "
                                          f"{manifest['license']['license_file_sha256'][:12]}…)",
                "primary_anchor_type": "upstream_redownload_hash_match",
                "source_cluster_id": f"vcd1_subject_{rec['subject_id']}",
                "subject_id": rec["subject_id"],
                "capture_domain": VCD1_DOMAINS[rec["subset"]],
                "source_collection_id": "microsoft/VCD",
            },
            "face_present": face,
            # README: "at least 30% of clips featuring active speaking" — not a per-clip
            # designation, and the release carries no audio stream.
            "speech_primary_annotation": None,
        }

    with ThreadPoolExecutor(4) as pool:
        clips = list(pool.map(one, manifest["file_records"]))
    return {"repository_checks": repo_checks, "paper": VCD1_PAPER,
            "clips": sorted(clips, key=lambda c: c["item_id"])}


def mirror_audit():
    """The benchmark mirrors the tracked corpus builders pinned, read as text, never executed."""
    pinned = set()
    for rel in BUILDER_SCRIPTS:
        text = (REPO / rel).read_text()
        pinned |= set(MIRROR_RE.findall(text))  # `\s*` spans the line break some use
    rows = []
    for repo, rev in sorted(pinned):
        info = http_json(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}")
        tags = [t for t in info.get("tags", []) if t.startswith("license:")]
        card = (info.get("cardData") or {})
        has_readme = any(s.get("rfilename") == "README.md" for s in info.get("siblings", []))
        rows.append({"repo": repo, "revision": rev, "license_tags": tags,
                     "card_license": card.get("license"), "dataset_card_present": has_readme,
                     "decision": CANDIDATE_ONLY,
                     "reason": "no licence and no dataset card at the pinned revision; the "
                               "mirror is not the original publisher and states no generation "
                               "or capture provenance" if not (tags or has_readme) else
                               "mirror of a third-party dataset; provenance of the upstream "
                               "labels not established by the mirror"})
    return rows


CODE_EXTS = frozenset({"py", "ipynb", "js", "ts", "sh", "r", "c", "cpp", "java", "go", "rs"})


def rapidata_audit():
    """Architect D-B: Apache-2.0 must cover the media and each file needs a generation record.

    Per repository at its pinned revision: the dataset card's licence field, the absence of any
    code or other licence file (so the grant cannot be read as covering code only), and that
    `main` is still the pinned revision (so the dataset viewer's rows are the pinned rows). Per
    file: the publisher row naming its model, kept for `generate-anchor` to check."""
    out = {}
    for source in (RAPIDATA_T2V, RAPIDATA_I2V):
        repo, rev = source["repo"], source["revision"]
        info = http_json(f"https://huggingface.co/api/datasets/{repo}/revision/{rev}")
        main = http_json(f"https://huggingface.co/api/datasets/{repo}")["sha"]
        card = http_get(f"https://huggingface.co/datasets/{repo}/raw/{rev}/README.md")[0].decode()
        front = card.split("---", 2)[1] if card.startswith("---") else ""
        licence = [ln.split(":", 1)[1].strip() for ln in front.splitlines()
                   if ln.startswith("license:")]
        files = [s["rfilename"] for s in info.get("siblings", [])]
        code = sorted(f for f in files if f.rsplit(".", 1)[-1].lower() in CODE_EXTS)
        other_licences = sorted(f for f in files if "licen" in f.lower())
        exts = Counter(f.rsplit(".", 1)[-1].lower() for f in files)
        checks = {"main_is_pinned_revision": main == rev,
                  "card_license_field_apache_2_0": licence == ["apache-2.0"],
                  "repository_holds_data_only": not code,
                  "no_other_licence_file": not other_licences}
        rows = hf_rows(repo)
        out[repo] = {"revision": rev, "main_revision": main, "card_license_field": licence,
                     "file_extensions": dict(sorted(exts.items())), "code_files": code,
                     "other_licence_files": other_licences, "checks": checks,
                     "accepted": all(checks.values()),
                     "file_model_rows": {name: model for name, (model, _) in sorted(rows.items())}}
    return out


def t1_carry_forward_annotations(corpus, t1_inventory, model_path):
    out = {}
    for item in read_json(t1_inventory)["items"]:
        if item["confidence_classification"] in TRUTHS:
            path = Path(corpus) / item["canonical_path"]
            out[item["sha256"]] = face_annotation(path, (item.get("format") or {}).get(
                "duration_s"), model_path)
    return out


def audit_local(args):
    intake = Path(args.intake).expanduser()
    corpus = Path(args.corpus).expanduser()
    intake.mkdir(parents=True, exist_ok=True)
    audit = {
        "schema": SCHEMA_AUDIT,
        "face_guard": face_guard(args.face_model) or "OK",
        "vcd1": vcd1_audit(corpus, intake, args.face_model),
        "benchmark_mirrors": mirror_audit(),
        "ffpp_local": {"subcorpora": ["ff++_c23_r3t1", "ff++_c23_test_r3t1"],
                       "decision": CANDIDATE_ONLY,
                       "reason": "FaceForensics++ terms of use (research only, no "
                                 "redistribution) are not an accepted usage basis; the source "
                                 "videos are third-party YouTube uploads without a per-clip "
                                 "licence"},
        "t1_carry_forward_face": t1_carry_forward_annotations(corpus, args.t1_inventory,
                                                              args.face_model),
        "rapidata": rapidata_audit(),
    }
    write_json(intake / AUDIT_NAME, audit)
    vcd = audit["vcd1"]
    print(json.dumps({"face_guard": audit["face_guard"],
                      "vcd1_repository_ok": all(v for v in vcd["repository_checks"].values()
                                                if isinstance(v, bool)),
                      "vcd1": dict(Counter(c["decision"] for c in vcd["clips"])),
                      "mirrors": len(audit["benchmark_mirrors"]),
                      "rapidata_accepted": {k: v["accepted"]
                                            for k, v in audit["rapidata"].items()}}, indent=1))


# --- ingest-external --------------------------------------------------------------------------

def ingest_one(entry, intake, model_path, previous):
    group = re.sub(r"[^A-Za-z0-9_.-]+", "_", entry["source_collection_id"])
    ext = Path(urllib.parse.urlparse(entry["download_url"]).path).suffix or ".mp4"
    old = previous.get(entry["item_id"])
    if old and old.get("ingest_status") == "OK" and (intake / old["path"]).is_file() \
            and t1.sha256_file(intake / old["path"]) == old["sha256"]:
        # The media and every ground-truth input are reused; only an annotation produced under
        # another pin, or one that failed, is recomputed.
        face = old.get("face_present") or {}
        if face.get("model_sha256") != FACE_SPEC["model_sha256"] or face.get("status") == "ERROR" \
                or face.get("preprocessing") != FACE_SPEC["preprocessing"]:
            old = {**old, "face_present": face_annotation(
                intake / old["path"], (old.get("format") or {}).get("duration_s"), model_path)}
        return old
    tmp = intake / group / ".partial" / (short_hash(entry["item_id"]) + ext)
    got = download(entry["download_url"], tmp, entry["expected_digest"])
    record = {"item_id": entry["item_id"], "ingest_status": got["status"],
              "error": got.get("error"), "sha256": got.get("sha256"),
              "md5_base64": got.get("md5_base64"), "size_bytes": got.get("size_bytes")}
    if got["status"] != "OK":
        return record
    final = intake / group / f"{entry['source_cluster']}_{got['sha256'][:12]}{ext}"
    tmp.replace(final)
    status, reason, profile = inspect(final)
    record.update({
        "path": final.relative_to(intake).as_posix(),
        "technical_status": status, "technical_reason": reason, "format": profile,
        "scene_cuts": scene_cuts(final) if entry["edit_screen_required"] and
        status == t1.INSPECTABLE else None,
        "face_present": face_annotation(final, (profile or {}).get("duration_s"), model_path),
    })
    return record


def ingest_external(args):
    intake = Path(args.intake).expanduser()
    spec = read_json(args.spec)
    ledger_path = intake / LEDGER_NAME
    previous = read_json(ledger_path)["items"] if ledger_path.is_file() else {}
    eligible = [e for e in spec["entries"] if validate_entry(e)[0] == ELIGIBLE]
    with ThreadPoolExecutor(args.workers) as pool:
        records = list(pool.map(lambda e: ingest_one(e, intake, args.face_model, previous),
                                eligible))
    for part in intake.glob("*/.partial"):
        shutil.rmtree(part, ignore_errors=True)
    write_json(ledger_path, {"schema": SCHEMA_LEDGER,
                             "face_guard": face_guard(args.face_model) or "OK",
                             "items": {r["item_id"]: r for r in records}})
    print(json.dumps({"eligible": len(eligible),
                      "status": dict(Counter(r["ingest_status"] for r in records)),
                      "technical": dict(Counter(r.get("technical_status") for r in records))},
                     indent=1))


# --- annotate: derived annotations only (Architect D-H) --------------------------------------

def annotate(args):
    """Recompute `face_present` for every ledger and audit item under the current FACE_SPEC.

    Reads media already on disk; no download, no network, and no ground-truth or provenance
    field is read for a decision or written."""
    intake, corpus = Path(args.intake).expanduser(), Path(args.corpus).expanduser()
    ledger_doc, audit = read_json(intake / LEDGER_NAME), read_json(intake / AUDIT_NAME)

    def face(path, fmt):
        return face_annotation(path, (fmt or {}).get("duration_s"), args.face_model)

    jobs = [("ledger", k, intake / r["path"], r.get("format"))
            for k, r in ledger_doc["items"].items() if r.get("ingest_status") == "OK"]
    jobs += [("vcd1", i, corpus / c["canonical_path"], c.get("format"))
             for i, c in enumerate(audit["vcd1"]["clips"])]
    formats = {i["sha256"]: (i["canonical_path"], i.get("format"))
               for i in read_json(args.t1_inventory)["items"]}
    jobs += [("t1", sha, corpus / formats[sha][0], formats[sha][1])
             for sha in audit["t1_carry_forward_face"]]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: face(j[2], j[3]), jobs))
    for (kind, key, _, _), result in zip(jobs, results):
        if kind == "ledger":
            ledger_doc["items"][key]["face_present"] = result
        elif kind == "vcd1":
            audit["vcd1"]["clips"][key]["face_present"] = result
        else:
            audit["t1_carry_forward_face"][key] = result
    ledger_doc["face_guard"] = audit["face_guard"] = face_guard(args.face_model) or "OK"
    write_json(intake / LEDGER_NAME, ledger_doc)
    write_json(intake / AUDIT_NAME, audit)
    print(json.dumps(dict(Counter(f"{r['status']}:{r['value']}" for r in results)), indent=1))


# --- generate-anchor --------------------------------------------------------------------------

def anchor_record(sha256, path, inspectable, provenance, face, speech_primary, origin):
    return {"sha256": sha256, "canonical_filename": Path(path).name, "canonical_path": path,
            "inspectable": inspectable, "origin": origin,
            "ground_truth_provenance": provenance,
            "derived_annotations": {"face_present": face,
                                    "speech_talking_head": speech_annotation(face,
                                                                             speech_primary)}}


T1_USAGE = {
    "r16_pilot": ("owner first-person consent for internal forensic evaluation; no public "
                  "distribution (r16_pilot_raw/genuine_source/SOURCES.txt)", "ESTABLISHED"),
    "nasa": ("Public-Domain (NASA, 17 U.S.C. §105)", "ESTABLISHED"),
    "ai": ("NOT ESTABLISHED in T2 — official provider promotional upload; carried forward from "
           "R18-T1 FINAL PASS, where usage rights were not part of the admission rule",
           "NOT_ESTABLISHED"),
}


T1_COLLECTIONS = {"r16_pilot": "r16-owner-pilot-2026-10-07",
                  "nasa": "nasa/artemis-ii-r17-frozen", "ai": "r17-ai-visual-v1"}


def t1_carry_forward(t1_inventory, faces):
    """Anchor records for the T1 verified set. Architect D-C: admission in R18-T2 needs an
    independently established usage basis, so records without one are returned separately —
    their historical ground truth (R17 frozen evidence) is not altered."""
    out, excluded = [], []
    for item in read_json(t1_inventory)["items"]:
        truth = item["confidence_classification"]
        if truth not in TRUTHS:
            continue
        rec = next(r for r in item["associated_metadata_records"] if r.get("trusted"))
        family = rec.get("generator_or_source")
        kind = ("r16_pilot" if item["subcorpus"] == "r16_pilot" else
                "nasa" if truth == VERIFIED_REAL else "ai")
        usage = T1_USAGE[kind]
        provenance = {
            "ground_truth": truth,
            "generator_family": None if truth == VERIFIED_REAL else
            re.sub(r"\s+Gen-\d+$", "", family),
            "generator_model": None if truth == VERIFIED_REAL else family,
            "source_type": "r18_t1_verified_carry_forward",
            "source_uri": rec.get("provenance_ref"),
            "publisher_or_owner": None,
            "provenance_evidence": item["classification_rationale"],
            "license_or_usage_basis": usage[0],
            "usage_rights_status": usage[1],
            "primary_anchor_type": "tracked_t1_anchor",
            "source_cluster_id": rec.get("source_cluster_id"),
            "subject_id": None,
            "capture_domain": rec.get("capture_domain"),
            "source_collection_id": T1_COLLECTIONS[kind],
        }
        face = faces.get(item["sha256"]) or {**_face_base(), "status": "UNKNOWN",
                                             "error": "not annotated"}
        if usage[1] != "ESTABLISHED":
            excluded.append({"item_id": f"t1:{item['canonical_path']}",
                             "class": "EXCLUDED_USAGE_NOT_ESTABLISHED",
                             "reason": "historical GT kept (R18-T1); usage basis not "
                                       "independently established (Architect D-C)"})
            continue
        out.append(anchor_record(item["sha256"], item["canonical_path"],
                                 item["technical_status"] == t1.INSPECTABLE, provenance, face,
                                 None, "r18_t1_carry_forward"))
    return out, excluded


def rapidata_decision(entry, audit):
    """None when the D-B guard holds for this Rapidata entry, else the reason it does not."""
    evidence = (audit.get("rapidata") or {}).get(entry["source_collection_id"])
    if evidence is None:
        return "no Rapidata collection evidence recorded"
    if not evidence["accepted"]:
        failed = [k for k, v in evidence["checks"].items() if not v]
        return "collection licence/provenance checks failed: " + ", ".join(failed)
    name = entry["item_id"].rsplit(":", 1)[1]
    model = evidence["file_model_rows"].get(name)
    if model is None or not name.startswith(model + "_") or \
            RAPIDATA_MODELS.get(model, (None, None))[1] != entry["generator_model"]:
        return f"no per-file generation record naming {entry['generator_model']!r} for {name}"
    return None


def external_decision(entry, led, review):
    """(truth or None, reason) for one validated external entry."""
    if led is None:
        return None, "not ingested"
    if led["ingest_status"] != "OK":
        return None, f"ingest {led['ingest_status']}: {led.get('error')}"
    if led["technical_status"] != t1.INSPECTABLE:
        return None, f"not inspectable: {led['technical_reason']}"
    if entry["expected_truth"] == VERIFIED_REAL and entry["edit_screen_required"]:
        if led.get("scene_cuts") is None:
            return None, "edit screen did not run"
        if led["scene_cuts"] > 0:
            return AMBIGUOUS, f"edited sequence: {led['scene_cuts']} scene cut(s) above " \
                              f"{SCENE_CUT_THRESHOLD}"
        if review is None:
            return None, "visual content-type screen pending"
        if review["verdict"] != "PASS":
            return AMBIGUOUS, "visual screen: " + review["reason"]
    return entry["expected_truth"], None


def concentration(clusters):
    counts = Counter(clusters)
    total = sum(counts.values())
    top = counts.most_common(5)
    return {"clusters": len(counts), "cases": total,
            "max_cluster": top[0][0] if top else None,
            "max_cluster_share": round(top[0][1] / total, 4) if total else None,
            "top5_share": round(sum(n for _, n in top) / total, 4) if total else None,
            "top5": [{"cluster": c, "cases": n} for c, n in top]}


def summarise(records):
    def gtp(r, k):
        return r["ground_truth_provenance"].get(k)
    truths = [gtp(r, "ground_truth") for r in records]
    by_truth = {t: [r for r in records if gtp(r, "ground_truth") == t] for t in TRUTHS}
    return {
        "cases": len(records),
        "by_truth": dict(Counter(truths)),
        "by_origin": dict(Counter(r["origin"] for r in records)),
        "by_generator_family": dict(Counter(gtp(r, "generator_family")
                                            for r in by_truth[VERIFIED_AI])),
        "by_generator_model": dict(Counter(gtp(r, "generator_model")
                                           for r in by_truth[VERIFIED_AI])),
        "by_real_capture_domain": dict(Counter(gtp(r, "capture_domain")
                                               for r in by_truth[VERIFIED_REAL])),
        "cluster_diversity": {"all": concentration([gtp(r, "source_cluster_id") for r in records]),
                              **{t: concentration([gtp(r, "source_cluster_id")
                                                   for r in by_truth[t]]) for t in TRUTHS}},
        "collection_diversity": {
            "all": concentration([gtp(r, "source_collection_id") for r in records]),
            **{t: concentration([gtp(r, "source_collection_id") for r in by_truth[t]])
               for t in TRUTHS}},
        "face_present": dict(Counter(
            f"{r['derived_annotations']['face_present']['status']}:"
            f"{r['derived_annotations']['face_present']['value']}" for r in records)),
        "speech_talking_head": dict(Counter(
            f"{r['derived_annotations']['speech_talking_head']['status']}:"
            f"{r['derived_annotations']['speech_talking_head']['value']}" for r in records)),
    }


def generate_anchor(args):
    intake = Path(args.intake).expanduser()
    spec = read_json(args.spec)
    ledger = read_json(intake / LEDGER_NAME)["items"]
    audit = read_json(intake / AUDIT_NAME)
    records, excluded = t1_carry_forward(args.t1_inventory, audit["t1_carry_forward_face"])
    for clip in audit["vcd1"]["clips"]:
        if clip["decision"] == VERIFIED_REAL and clip["technical_status"] == t1.INSPECTABLE:
            records.append(anchor_record(clip["sha256"], clip["canonical_path"], True,
                                         clip["ground_truth_provenance"], clip["face_present"],
                                         clip["speech_primary_annotation"],
                                         "r18_t2_internal_audit"))
        else:
            excluded.append({"item_id": clip["item_id"], "class": CANDIDATE_ONLY,
                             "reason": clip["decision_reason"] or clip["technical_reason"]})
    for entry in spec["entries"] + spec["candidate_only_sources"]:
        status, reason = validate_entry(entry)
        if status != ELIGIBLE:
            excluded.append({"item_id": entry["item_id"], "class": CANDIDATE_ONLY,
                             "reason": reason})
            continue
        if entry["source_collection_id"].startswith("Rapidata/"):
            reason = rapidata_decision(entry, audit)
            if reason:
                excluded.append({"item_id": entry["item_id"], "class": CANDIDATE_ONLY,
                                 "reason": "D-B guard: " + reason})
                continue
        led = ledger.get(entry["item_id"])
        truth, reason = external_decision(entry, led, spec["visual_review"].get(entry["item_id"]))
        if truth not in TRUTHS:
            excluded.append({"item_id": entry["item_id"], "class": truth or "NOT_ADMITTED",
                             "reason": reason})
            continue
        provenance = {
            "ground_truth": truth,
            "generator_family": entry["generator_family"],
            "generator_model": entry["generator_model"],
            "generation_mode": entry["generation_mode"],
            "source_type": entry["provenance_tier"],
            "source_uri": entry["source_uri"],
            "publisher_or_owner": entry["publisher_or_owner"],
            "provenance_evidence": entry["provenance_evidence"],
            "license_or_usage_basis": entry["license_or_usage_basis"],
            "usage_rights_status": "ESTABLISHED",
            "primary_anchor_type": "publisher_digest_match_" + entry["expected_digest"]["algorithm"],
            "source_cluster_id": entry["source_cluster"],
            "subject_id": entry["subject_id"],
            "capture_domain": entry["capture_domain"],
            "source_collection_id": entry["source_collection_id"],
            "prompt": entry["prompt"],
        }
        records.append(anchor_record(led["sha256"], "r18_t2_intake/" + led["path"], True,
                                     provenance, led["face_present"],
                                     entry["speech_primary_annotation"], "r18_t2_external"))
    missing = [r["sha256"] for r in records if not all(
        r["ground_truth_provenance"].get(k) for k in ("source_cluster_id", "source_collection_id"))]
    if missing:
        raise Refusal(f"admitted records without cluster/collection id: {missing[:5]}")
    seen = Counter(r["sha256"] for r in records)
    dupes = sorted(s for s, n in seen.items() if n > 1)
    if dupes:
        raise Refusal(f"the same bytes would be anchored twice: {dupes[:5]}")
    records.sort(key=lambda r: (r["ground_truth_provenance"]["ground_truth"],
                                r["ground_truth_provenance"]["source_cluster_id"] or "",
                                r["sha256"]))
    excluded.sort(key=lambda x: x["item_id"])
    anchor = {
        "schema": SCHEMA_ANCHOR,
        "corpus_root": "/home/adentechio/deepguard-corpus",
        "rule": "Records verified ground truth established by primary provenance; it does not "
                "create ground truth. Derived annotations never affect ground_truth_provenance.",
        "frozen_face_annotation": FACE_SPEC,
        "face_guard": audit["face_guard"],
        "edit_screen": spec["edit_screen"],
        "summary": summarise(records),
        "targets": {"cases": 350, "independent_source_clusters": 60},
        "records": records,
        "not_admitted": excluded,
        "not_admitted_summary": dict(Counter(x["class"] for x in excluded)),
        "benchmark_mirrors_audited": len(audit["benchmark_mirrors"]),
    }
    write_json(args.out, anchor)
    s = anchor["summary"]
    print(json.dumps({"cases": s["cases"], "by_truth": s["by_truth"],
                      "clusters": s["cluster_diversity"]["all"]["clusters"],
                      "collections": s["collection_diversity"]["all"]["clusters"],
                      "not_admitted": anchor["not_admitted_summary"]}, indent=1))


# --- CLI --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("author-spec")
    a.add_argument("--spec", required=True)
    a.set_defaults(func=author_spec)
    b = sub.add_parser("audit-local")
    b.add_argument("--corpus", required=True)
    b.add_argument("--intake", required=True)
    b.add_argument("--t1-inventory", required=True)
    b.add_argument("--face-model", default=str(FACE_MODEL_DEFAULT))
    b.set_defaults(func=audit_local)
    c = sub.add_parser("ingest-external")
    c.add_argument("--spec", required=True)
    c.add_argument("--intake", required=True)
    c.add_argument("--face-model", default=str(FACE_MODEL_DEFAULT))
    c.add_argument("--workers", type=int, default=4)
    c.set_defaults(func=ingest_external)
    e = sub.add_parser("annotate")
    e.add_argument("--intake", required=True)
    e.add_argument("--corpus", required=True)
    e.add_argument("--t1-inventory", required=True)
    e.add_argument("--face-model", default=str(FACE_MODEL_DEFAULT))
    e.add_argument("--workers", type=int, default=6)
    e.set_defaults(func=annotate)
    d = sub.add_parser("generate-anchor")
    d.add_argument("--spec", required=True)
    d.add_argument("--intake", required=True)
    d.add_argument("--t1-inventory", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(func=generate_anchor)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
