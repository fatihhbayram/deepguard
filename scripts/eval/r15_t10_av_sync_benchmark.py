"""R15-T10: offline benchmark of a pre-frozen audio-visual synchronisation (lip-sync) score.

The model, the face detector, the landmark-to-crop tracking, the audio front end, the temporal
window, the clip-level reading, the score direction, the threshold, the unscorable policy and the
corpus are fixed in `docs/ai/r15_t10_freeze.md` and pinned by `docs/ai/r15_t10_freeze.json`
before any corpus file is scored. The runner refuses to start if its own bytes, the upstream
source files it executes, the checkpoint, the landmark weights or the R15-T5 manifest differ from
the hashes recorded there. Exactly one configuration is run.

Model: ByteDance LatentSync StableSyncNet (`stable_syncnet.pt`, LatentSync-1.6 release, config
`syncnet_16_pixel_attn.yaml`). The architecture, the mel front end and the 3-point affine aligner
are executed from the pinned upstream checkout (`/ls`), not retyped.

Face detector and landmarks: the S3FD + 2D-FAN pair DeepGuard already pins for LipForensics
(`app.lip_forensics.SFD_SHA256` / `FAN_SHA256`). Upstream aligns on InsightFace 106-point
landmarks, whose weights are non-commercial and are not used; the three alignment points are
taken from the 68-point layout instead. That is a DeepGuard divergence, declared in the freeze.

Window: 16 frames at 25 fps (0.64 s) against 52 mel columns starting at `int(80 * frame / 25)`,
exactly `SyncNetDataset.crop_audio_window`. Windows are non-overlapping from frame 0.

Clip reading: the mean cosine similarity over valid windows. Upstream calls a window in sync iff
cosine similarity > 0.5 (`eval/eval_syncnet_acc.py`); a clip is FLAGGED as an AV-sync anomaly iff
its mean is <= 0.5. Direction is declared, not learned: higher = more in sync. AUROC is
P(manipulated score < genuine score) and is never flipped.

Unscorable is not negative: a clip that cannot be read is listed with its reason and enters no
rate, no AUROC and no score summary.

Nothing here touches the database, the API or the risk engine. Output is one JSON file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from eval.stats import rate

METHOD_ID = "STABLESYNCNET-LS16-FAN68-W16"
UPSTREAM = Path("/ls")
CHECKPOINT = Path("/home/adentechio/deepguard-corpus/models/stable_syncnet/stable_syncnet.pt")
LANDMARK_ROOT = Path("/home/adentechio/.cache/deepguard/lipforensics/face-alignment")
SFD_FILE = LANDMARK_ROOT / "checkpoints" / "s3fd-619a316812.pth"
FAN_FILE = LANDMARK_ROOT / "checkpoints" / "2DFAN4-11f355bf06.pth.tar"
R15_MANIFEST = Path("/home/adentechio/deepguard-corpus/r15_oos_manifest.json")
CORPUS_ROOT = Path("/home/adentechio/deepguard-corpus")
UPSTREAM_FILES = (
    "latentsync/models/stable_syncnet.py",
    "latentsync/models/attention.py",
    "latentsync/utils/audio.py",
    "latentsync/utils/affine_transform.py",
    "configs/audio.yaml",
    "configs/syncnet/syncnet_16_pixel_attn.yaml",
)

# The positive families: audio-driven mouth synthesis only. LAV-DF is excluded (no per-clip
# modality metadata in the mirror, so AV-sync semantics cannot be verified per clip).
POSITIVE_FAMILIES = {
    "r7t8": ("lipsync_wav2lip",),
    "r7t5": ("talkinghead_echomimic", "talkinghead_memo", "talkinghead_sonic"),
}
GENUINE, MANIPULATED = "GENUINE", "AV_SYNC_MANIPULATED"

FPS = 25
NUM_FRAMES = 16
MEL_WINDOW = 52                 # math.ceil(16 / 5 * 16), SyncNetDataset.mel_window_length
RESOLUTION = 256
LANDMARK_MAX_SIDE = 640         # frames are downscaled to this longer side for S3FD + 2D-FAN only
THRESHOLD = 0.5                 # upstream eval/eval_syncnet_acc.py: in sync iff sim > 0.5
MIN_VALID_WINDOWS = 3
TRACK_MIN_IOU = 0.3
SILENCE_PEAK = 1e-4
# 68-point iBUG layout: image-left brow 17-21, image-right brow 22-26, nose 27-35. Upstream takes
# the two brow centres and a nose centre from InsightFace's 106-point layout.
LEFT_BROW, RIGHT_BROW, NOSE = range(17, 22), range(22, 27), range(27, 36)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_freeze(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    if freeze["method_id"] != METHOD_ID:
        sys.exit("refusing to run: method id differs from the freeze record")
    pinned = {
        Path(__file__).resolve(): freeze["runner_sha256"],
        CHECKPOINT: freeze["checkpoint"]["sha256"],
        SFD_FILE: freeze["face_detector"]["sha256"],
        FAN_FILE: freeze["landmark_model"]["sha256"],
        R15_MANIFEST: freeze["manifest_sha256"],
    }
    pinned.update({UPSTREAM / name: digest for name, digest in freeze["upstream_files"].items()})
    drift = [str(p) for p, h in pinned.items() if sha256_file(p) != h]
    if drift:
        sys.exit(f"refusing to run: differs from the freeze record: {drift}")
    return freeze


def load_population(expected: dict) -> tuple[list[dict], dict]:
    """The 50 base positives and the 6 R15 genuine videos, with the leakage checks."""
    population = []
    for corpus, families in POSITIVE_FAMILIES.items():
        for item in json.loads((CORPUS_ROOT / corpus / "corpus.json").read_text())["items"]:
            if item["family"] in families and item["derivation"] == "none":
                partition = "CALIBRATION" if item["split"] == "calibration" else "HOLDOUT"
                lineage = item["source_lineage_id"]
                population.append({
                    "id": f"{corpus}:{item['clip_id']}", "path": str(CORPUS_ROOT / corpus / item["path"]),
                    "file_hash": item["sha256"], "label": MANIPULATED, "family": item["family"],
                    "partition": partition, "lineage": lineage,
                    # FaceForensics++ donor id for Wav2Lip; the clip's own lineage otherwise.
                    "donor": lineage, "subject": lineage,
                })
    manifest = json.loads(R15_MANIFEST.read_text())
    for r in manifest["primary_oos"]["video"]:
        population.append({
            "id": r["recording_identity"], "path": r["file_path"], "file_hash": r["file_hash"],
            "label": GENUINE, "family": r["benchmark_family"], "partition": r["dataset_split"],
            "lineage": r["source_lineage_id"], "donor": r["source_lineage_id"],
            "subject": r["subject_id"],
        })

    excluded = {h for hashes in manifest["excluded_must_not_enter"].values() for h in hashes}
    auxiliary = {r["file_hash"] for g in manifest["auxiliary_known_family"].values() for r in g}
    hashes = [r["file_hash"] for r in population]

    def keys(field: str, partition: str) -> set:
        return {r[field] for r in population if r["partition"] == partition}

    counts = Counter(f"{r['partition']}/{r['label']}/{r['family']}" for r in population)
    checks = {
        "files_by_partition_label_family": dict(sorted(counts.items())),
        "duplicate_hashes": len(hashes) - len(set(hashes)),
        "excluded_must_not_enter_hashes_loaded": len(set(hashes) & excluded),
        "auxiliary_known_family_hashes_loaded": len(set(hashes) & auxiliary),
        "lineage_overlap_calibration_holdout":
            sorted(keys("lineage", "CALIBRATION") & keys("lineage", "HOLDOUT")),
        "donor_overlap_calibration_holdout":
            sorted(keys("donor", "CALIBRATION") & keys("donor", "HOLDOUT")),
        "subject_overlap_calibration_holdout":
            sorted(keys("subject", "CALIBRATION") & keys("subject", "HOLDOUT")),
        "family_overlap_calibration_holdout":
            sorted(keys("family", "CALIBRATION") & keys("family", "HOLDOUT")),
        "genuine_subjects": sorted({r["subject"] for r in population if r["label"] == GENUINE}),
        "positive_donor_in_genuine_set": sorted(
            {r["donor"] for r in population if r["label"] == MANIPULATED}
            & {r["donor"] for r in population if r["label"] == GENUINE}),
    }
    failures = [name for name, bad in [
        ("duplicate hash", checks["duplicate_hashes"]),
        ("excluded hash", checks["excluded_must_not_enter_hashes_loaded"]),
        ("auxiliary hash", checks["auxiliary_known_family_hashes_loaded"]),
        ("lineage overlap", checks["lineage_overlap_calibration_holdout"]),
        ("donor overlap", checks["donor_overlap_calibration_holdout"]),
        ("subject overlap", checks["subject_overlap_calibration_holdout"]),
        ("family overlap", checks["family_overlap_calibration_holdout"]),
        ("donor shared with genuine", checks["positive_donor_in_genuine_set"]),
        ("counts", checks["files_by_partition_label_family"] != expected),
    ] if bad]
    if failures:
        sys.exit(f"refusing to run: corpus checks failed: {failures} {checks}")
    return population, checks


# --- the frozen pipeline -------------------------------------------------------------------

_STATE: dict = {}


def _init_worker(threads: int) -> None:
    import torch
    torch.set_num_threads(threads)
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)  # latentsync/utils/audio.py reads configs/audio.yaml relative to cwd
    import face_alignment
    from omegaconf import OmegaConf
    from latentsync.models.stable_syncnet import StableSyncNet

    config = OmegaConf.load(UPSTREAM / "configs/syncnet/syncnet_16_pixel_attn.yaml")
    model = StableSyncNet(OmegaConf.to_container(config.model))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)
    model.eval()
    torch.hub.set_dir(str(LANDMARK_ROOT))
    landmarker = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False, compile=False)
    _STATE.update(model=model, landmarker=landmarker)


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _largest_face(frame_rgb: np.ndarray):
    """(bbox, 68 landmarks) in full-frame pixels for the largest face passing upstream's filters."""
    height, width = frame_rgb.shape[:2]
    scale = min(1.0, LANDMARK_MAX_SIDE / max(height, width))
    small = frame_rgb if scale == 1.0 else cv2.resize(
        frame_rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    result = _STATE["landmarker"].get_landmarks_from_image(small, return_bboxes=True)
    landmarks, _, boxes = result if result is not None else (None, None, None)
    if not landmarks:
        return None, None
    best, best_size = None, 0.0
    for points, box in zip(landmarks, boxes):
        x1, y1, x2, y2, score = (float(v) for v in box[:5])
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
        w, h = x2 - x1, y2 - y1
        # Upstream face_detector.py filters, applied at full-frame scale.
        if w < 50 or h < 80 or not (0.2 <= w / h <= 1.5) or score < 0.5:
            continue
        if w * h > best_size:
            best, best_size = ((x1, y1, x2, y2), np.asarray(points, dtype=np.float64) / scale), w * h
    return best if best is not None else (None, None)


def decode_audio(path: str, scratch: str) -> np.ndarray:
    raw = Path(scratch) / "audio.f32"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-nostdin", "-y", "-i", path, "-vn",
                    "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", str(raw)],
                   capture_output=True, check=True)
    return np.fromfile(raw, dtype=np.float32)


def decode_video_25fps(path: str, scratch: str) -> str:
    # Verbatim upstream latentsync/utils/util.read_video(change_fps=True) re-encode.
    out = str(Path(scratch) / "video.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", path, "-r", "25",
                    "-crf", "18", out], capture_output=True, check=True)
    return out


def score_clip(record: dict) -> dict:
    import torch
    from latentsync.utils.affine_transform import AlignRestore
    from latentsync.utils.audio import melspectrogram

    def unscorable(reason: str, **extra) -> dict:
        return {"score": None, "unscorable": reason, "windows": [], **extra}

    if sha256_file(Path(record["path"])) != record["file_hash"]:
        return unscorable("U0_hash_mismatch")
    with tempfile.TemporaryDirectory() as scratch:
        try:
            audio = decode_audio(record["path"], scratch)
        except subprocess.CalledProcessError:
            return unscorable("U1_no_audio_stream_or_decode_failure")
        if audio.size == 0:
            return unscorable("U1_no_audio_stream_or_decode_failure")
        if float(np.max(np.abs(audio))) < SILENCE_PEAK:
            return unscorable("U2_silent_audio")
        try:
            video = decode_video_25fps(record["path"], scratch)
        except subprocess.CalledProcessError:
            return unscorable("U3_video_decode_failure")

        aligner = AlignRestore(resolution=RESOLUTION, device="cpu", dtype=torch.float32)
        crops, valid, previous_box, faces_found = [], [], None, 0
        capture = cv2.VideoCapture(video)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box, points = _largest_face(rgb)
            if box is None:
                crops.append(None); valid.append(False); previous_box = None
                continue
            faces_found += 1
            continuous = previous_box is None or _iou(box, previous_box) >= TRACK_MIN_IOU
            previous_box = box
            landmarks3 = np.round([points[list(LEFT_BROW)].mean(0), points[list(RIGHT_BROW)].mean(0),
                                   points[list(NOSE)].mean(0)])
            face, _ = aligner.align_warp_face(rgb.copy(), landmarks3=landmarks3, smooth=True)
            face = cv2.resize(face, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LANCZOS4)
            crops.append(face); valid.append(continuous)
        capture.release()

    total_frames = len(crops)
    if total_frames < NUM_FRAMES:
        return unscorable("U4_too_short", frames_25fps=total_frames)
    if faces_found == 0:
        return unscorable("U5_no_face_detected", frames_25fps=total_frames)

    mel = torch.from_numpy(melspectrogram(audio))
    model = _STATE["model"]
    windows = []
    for start in range(0, total_frames - NUM_FRAMES + 1, NUM_FRAMES):
        mel_start = int(80.0 * (start / float(FPS)))
        mel_window = mel[:, mel_start:mel_start + MEL_WINDOW]
        frames_ok = all(valid[start:start + NUM_FRAMES])
        entry = {"start_frame": start, "face_tracked": frames_ok,
                 "audio_complete": mel_window.shape[-1] == MEL_WINDOW, "sim": None}
        if frames_ok and entry["audio_complete"]:
            pixels = torch.from_numpy(np.stack(crops[start:start + NUM_FRAMES])).float()
            pixels = (pixels.permute(0, 3, 1, 2) / 255.0 - 0.5) / 0.5            # (16, 3, 256, 256)
            pixels = pixels.reshape(1, NUM_FRAMES * 3, RESOLUTION, RESOLUTION)    # b (f c) h w
            pixels = pixels[:, :, RESOLUTION // 2:, :]                            # lower half
            with torch.no_grad():
                vision, audio_embed = model(pixels, mel_window.unsqueeze(0).unsqueeze(0).float())
            entry["sim"] = float(torch.nn.functional.cosine_similarity(vision, audio_embed)[0])
        windows.append(entry)

    sims = [w["sim"] for w in windows if w["sim"] is not None]
    base = {"windows": windows, "frames_25fps": total_frames, "frames_with_face": faces_found,
            "valid_windows": len(sims), "total_windows": len(windows)}
    if not sims:
        return {**base, "score": None, "unscorable": "U6_face_tracking_failure"}
    if len(sims) < MIN_VALID_WINDOWS:
        return {**base, "score": None, "unscorable": "U7_insufficient_tracked_windows"}
    return {**base, "score": float(np.mean(sims)), "unscorable": None}


# --- metrics -------------------------------------------------------------------------------

def auroc(genuine: list[float], manipulated: list[float]) -> float | None:
    """P(genuine score > manipulated score), ties = 1/2. Direction never flipped."""
    if not genuine or not manipulated:
        return None
    wins = sum((g > m) + 0.5 * (g == m) for g in genuine for m in manipulated)
    return wins / (len(genuine) * len(manipulated))


def average_precision(genuine: list[float], manipulated: list[float]) -> float | None:
    """AP with manipulated positive, ranked by ascending score; ties as one block."""
    if not manipulated:
        return None
    scored = sorted([(s, 1) for s in manipulated] + [(s, 0) for s in genuine])
    ap, tp, fp, recall_prev, i = 0.0, 0, 0, 0.0, 0
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        tp += sum(f for _, f in scored[i:j])
        fp += sum(1 - f for _, f in scored[i:j])
        recall = tp / len(manipulated)
        ap += (recall - recall_prev) * (tp / (tp + fp))
        recall_prev = recall
        i = j
    return ap


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "min": min(values), "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)), "p75": float(np.percentile(values, 75)),
            "max": max(values)}


def stratum(records: list[dict]) -> dict:
    scored = [r for r in records if r["score"] is not None]
    flagged = sum(1 for r in scored if r["score"] <= THRESHOLD)
    return {
        "n": len(records), "scorable": len(scored), "unscorable": len(records) - len(scored),
        "unscorable_reasons": dict(Counter(r["unscorable"] for r in records if r["unscorable"])),
        "coverage": rate(len(scored), len(records)).as_dict(),
        "flagged": rate(flagged, len(scored)).as_dict(),
        "scores": summary([r["score"] for r in scored]),
    }


def analyse(population: list[dict]) -> dict:
    genuine = [r for r in population if r["label"] == GENUINE]
    manipulated = [r for r in population if r["label"] == MANIPULATED]
    g = stratum(genuine)
    gen_scores = [r["score"] for r in genuine if r["score"] is not None]

    def separation(records):
        scores = [r["score"] for r in records if r["score"] is not None]
        return {"auroc": auroc(gen_scores, scores), "ap": average_precision(gen_scores, scores),
                "n_genuine": len(gen_scores), "n_manipulated": len(scores)}

    families = sorted({r["family"] for r in manipulated})
    partitions = ("CALIBRATION", "HOLDOUT")
    return {
        "r15_genuine_line": f"{g['scorable']}/6 scorable, {g['flagged']['k']}/{g['scorable']} "
                            f"flagged, {g['unscorable']} unscorable",
        "r15_genuine": g,
        "manipulated_pooled": stratum(manipulated),
        "manipulated_by_family": {f: stratum([r for r in manipulated if r["family"] == f])
                                  for f in families},
        "manipulated_by_partition": {p: stratum([r for r in manipulated if r["partition"] == p])
                                     for p in partitions},
        "separation_pooled": separation(manipulated),
        "separation_by_family": {f: separation([r for r in manipulated if r["family"] == f])
                                 for f in families},
        "separation_by_partition": {p: separation([r for r in manipulated if r["partition"] == p])
                                    for p in partitions},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sanity", nargs="+", metavar="FILE",
                        help="score these non-corpus files only (pre-run pipeline check)")
    args = parser.parse_args()

    freeze = check_freeze(args.freeze)
    if args.sanity:
        population = [{"id": Path(p).name, "path": p, "file_hash": sha256_file(Path(p)),
                       "label": "SANITY", "family": "sanity", "partition": "NONE"}
                      for p in args.sanity]
        checks = {"mode": "sanity: non-corpus files, no metric"}
    else:
        population, checks = load_population(freeze["expected_population"])

    started = datetime.now(timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.threads,)) as pool:
        for record, result in zip(population, pool.map(score_clip, population)):
            record.update(result)
            print(f"{record['id']}: score={record['score']} unscorable={record['unscorable']}",
                  flush=True)

    import face_alignment
    import torch
    output = {
        "task": "R15-T10", "method_id": METHOD_ID, "mode": "sanity" if args.sanity else "benchmark",
        "freeze_record": str(args.freeze), "freeze_record_sha256": sha256_file(args.freeze),
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                 text=True).stdout.splitlines()[0],
        "torch": torch.__version__, "numpy": np.__version__,
        "face_alignment": face_alignment.__version__,
        "corpus_checks": checks,
        "analysis": None if args.sanity else analyse(population),
        "files": population,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
