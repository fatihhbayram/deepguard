"""R15-T7: offline benchmark of one pre-frozen temporal-frequency video anomaly score.

The method, its parameters, the corpus and the threshold rule are fixed in
`docs/ai/r15_t7_freeze.md` and pinned by `docs/ai/r15_t7_freeze.json` before any clip is
scored. This runner refuses to start if its own bytes or any input manifest differ from the
hashes recorded there, so the configuration that produced the numbers is the one that was
declared, and only one configuration is ever run.

Method TFA-1 (temporal high-frequency energy ratio):

    decode the first 64 frames -> luma, area-downscaled to 128x128 (ffmpeg, no audio)
    split into non-overlapping 32-frame windows (frames 0-31, 32-63)
    per window: subtract each pixel's temporal mean, apply a 32-point Hann taper along time,
                rFFT along time, average |F|^2 over all pixels -> 17 bins (k = 0..16)
    window score = sum(P[8..16]) / sum(P[1..16])   (energy at >= 0.25 cycles/frame, DC excluded)
    clip score   = unweighted mean of the window scores

Direction is declared, not learned: a higher score means more frame-to-frame high-frequency
change (flicker), and is read as more likely manipulated. An AUROC below 0.5 is reported as
measured and is never flipped.

The threshold is taken from the R7-T9 calibration split's genuine base clips alone (no positive
is consulted): the smallest observed genuine score that leaves at most 5% of them above it under
`score > threshold`. Everything else is scored once and only read.

Nothing here touches the database, the API or the risk engine. Output is one JSON file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.stats import rate

METHOD_ID = "TFA-1"
FRAMES = 64
WINDOW = 32
SIDE = 128
HIGH_BAND_FIRST_BIN = 8
MIN_POWER = 1e-10
CALIBRATION_FPR_TARGET = 0.05
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261001

HOME = Path("/home/adentechio")
CORPUS = HOME / "deepguard-corpus"
R7T9_DIR = CORPUS / "r7t9"
R7T9_MANIFEST = R7T9_DIR / "manifest.csv"
R7T9_CORPUS = R7T9_DIR / "corpus.json"
VCD1_MANIFEST = CORPUS / "external-genuine-vcd1/vcd1_intake_manifest.json"
R15_MANIFEST = CORPUS / "r15_oos_manifest.json"
POSITIVE_LABELS = {"face_swap", "synthetic"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_freeze(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    pinned = {Path(__file__).resolve(): freeze["runner_sha256"]}
    pinned.update({Path(p): h for p, h in freeze["input_sha256"].items()})
    drift = [str(p) for p, h in pinned.items() if sha256_file(p) != h]
    if drift:
        sys.exit(f"refusing to run: differs from the freeze record: {drift}")
    if freeze["method_id"] != METHOD_ID:
        sys.exit("refusing to run: method id differs from the freeze record")
    return freeze


def decode_luma(path: Path) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-an",
        "-frames:v", str(FRAMES), "-vf", f"scale={SIDE}:{SIDE}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    frames = len(raw) // (SIDE * SIDE)
    return np.frombuffer(raw[: frames * SIDE * SIDE], dtype=np.uint8).reshape(frames, SIDE, SIDE)


def window_score(window: np.ndarray) -> float | None:
    signal = window.astype(np.float64) / 255.0
    signal -= signal.mean(axis=0, keepdims=True)
    signal *= np.hanning(WINDOW)[:, None, None]
    power = (np.abs(np.fft.rfft(signal, axis=0)) ** 2).mean(axis=(1, 2))
    total = power[1:].sum()
    if total < MIN_POWER:
        return None
    return float(power[HIGH_BAND_FIRST_BIN:].sum() / total)


def score_clip(path: str) -> dict:
    try:
        frames = decode_luma(Path(path))
    except subprocess.CalledProcessError as error:
        return {"frames": 0, "windows": [], "score": None,
                "unscorable": f"decode failed: {error.stderr.decode(errors='replace')[-200:]}"}
    windows = [window_score(frames[i:i + WINDOW])
               for i in range(0, len(frames) - WINDOW + 1, WINDOW)]
    usable = [w for w in windows if w is not None]
    if not usable:
        reason = "fewer than 32 frames" if len(frames) < WINDOW else "no temporal energy"
        return {"frames": len(frames), "windows": windows, "score": None, "unscorable": reason}
    return {"frames": len(frames), "windows": windows,
            "score": sum(usable) / len(usable), "unscorable": None}


def load_population() -> list[dict]:
    items = {i["clip_id"]: i for i in json.loads(R7T9_CORPUS.read_text())["items"]}
    population = []
    for row in csv.DictReader(R7T9_MANIFEST.open()):
        item = items[row["clip_id"]]
        population.append({
            "id": f"r7t9:{row['clip_id']}",
            "path": str(R7T9_DIR / row["path"]),
            "source": "R7-T9",
            "split": row["split"],
            "positive": row["label"] in POSITIVE_LABELS,
            "label": row["label"],
            "family": row["family"],
            "stratum": row["stratum"],
            "base": item["derivation"] == "none",
            "lineage": row["source_lineage_id"],
        })
    vcd1 = json.loads(VCD1_MANIFEST.read_text())
    for record in vcd1["file_records"]:
        population.append({
            "id": f"vcd1:{record['subject_id']}",
            "path": str(VCD1_MANIFEST.parent / record["relative_path"]),
            "source": "R11-T3B VCD1", "split": "external_genuine", "positive": False,
            "label": "real", "family": f"vcd1_{record['subset']}", "stratum": "genuine_videoconf",
            "base": True, "lineage": f"vcd1:{record['subject_id']}",
        })
    for record in json.loads(R15_MANIFEST.read_text())["primary_oos"]["video"]:
        population.append({
            "id": record["recording_identity"], "path": record["file_path"],
            "source": "R15-T5 primary_oos video", "split": "r15_oos_genuine", "positive": False,
            "label": "real", "family": record["benchmark_family"],
            "stratum": record["stratum_primary"], "base": True,
            "lineage": record["source_lineage_id"], "file_hash": record["file_hash"],
        })
    return population


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """Mann-Whitney AUROC with ties counted as one half."""
    if not positives or not negatives:
        return None
    ordered = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        mid_rank = (i + 1 + j) / 2.0
        rank_sum += mid_rank * sum(flag for _, flag in ordered[i:j])
        i = j
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(positives: list[float], negatives: list[float]) -> float | None:
    """Step-wise AP over distinct thresholds, ties resolved as one block."""
    if not positives:
        return None
    scored = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives], reverse=True)
    ap, tp, fp, recall_prev, i = 0.0, 0, 0, 0.0, 0
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        block = scored[i:j]
        tp += sum(f for _, f in block)
        fp += sum(1 - f for _, f in block)
        recall = tp / len(positives)
        ap += (recall - recall_prev) * (tp / (tp + fp))
        recall_prev = recall
        i = j
    return ap


def bootstrap_auroc(positives: list[float], negatives: list[float]) -> list[float] | None:
    if not positives or not negatives:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    draws = sorted(
        auroc([rng.choice(positives) for _ in positives], [rng.choice(negatives) for _ in negatives])
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return [draws[int(0.025 * BOOTSTRAP_RESAMPLES)], draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]]


def calibration_threshold(genuine: list[float]) -> float:
    ordered = sorted(genuine)
    return ordered[math.ceil((1.0 - CALIBRATION_FPR_TARGET) * len(ordered)) - 1]


def discrimination(positives: list[float], negatives: list[float]) -> dict:
    return {
        "n_positive": len(positives), "n_negative": len(negatives),
        "auroc": auroc(positives, negatives),
        "auroc_bootstrap_95": bootstrap_auroc(positives, negatives),
        "average_precision": average_precision(positives, negatives),
        "prevalence": len(positives) / (len(positives) + len(negatives)) if positives else None,
    }


def flagged(records: list[dict], threshold: float) -> dict:
    k = sum(1 for r in records if r["score"] > threshold)
    return rate(k, len(records)).as_dict()


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record[key], []).append(record)
    return dict(sorted(groups.items()))


def analyse(population: list[dict]) -> dict:
    scored = [r for r in population if r["score"] is not None]

    def select(**conditions):
        return [r for r in scored if all(r[k] == v for k, v in conditions.items())]

    cal_gen = select(source="R7-T9", split="calibration", base=True, positive=False)
    cal_pos = select(source="R7-T9", split="calibration", base=True, positive=True)
    eva_gen = select(source="R7-T9", split="evaluation", base=True, positive=False)
    eva_pos = select(source="R7-T9", split="evaluation", base=True, positive=True)
    derived = [r for r in scored if r["source"] == "R7-T9" and not r["base"]]
    vcd1 = select(source="R11-T3B VCD1")
    r15 = select(source="R15-T5 primary_oos video")
    threshold = calibration_threshold([r["score"] for r in cal_gen])
    scores = lambda records: [r["score"] for r in records]  # noqa: E731

    cal_sorted = sorted(scores(cal_gen))
    return {
        "threshold": {
            "value": threshold,
            "rule": "smallest observed calibration genuine base-clip score leaving <= 5% of them "
                    "above it under score > threshold; positives not consulted",
            "derived_from_n_genuine": len(cal_gen),
            "calibration_genuine_flagged": flagged(cal_gen, threshold),
        },
        "unscorable": [{"id": r["id"], "source": r["source"], "split": r["split"],
                        "positive": r["positive"], "reason": r["unscorable"], "frames": r["frames"]}
                       for r in population if r["score"] is None],
        "calibration_descriptive": discrimination(scores(cal_pos), scores(cal_gen)),
        "evaluation_base": {
            **discrimination(scores(eva_pos), scores(eva_gen)),
            "tpr_at_threshold": flagged(eva_pos, threshold),
            "fpr_at_threshold": flagged(eva_gen, threshold),
            "by_positive_family": {
                family: {"auroc_vs_all_evaluation_genuine": auroc(scores(rs), scores(eva_gen)),
                         "tpr_at_threshold": flagged(rs, threshold),
                         "label": rs[0]["label"]}
                for family, rs in group_by(eva_pos, "family").items()
            },
            "by_genuine_stratum": {stratum: flagged(rs, threshold)
                                   for stratum, rs in group_by(eva_gen, "stratum").items()},
            "by_positive_label": {
                label: {**discrimination(scores(rs), scores(eva_gen)),
                        "tpr_at_threshold": flagged(rs, threshold)}
                for label, rs in group_by(eva_pos, "label").items()
            },
        },
        "evaluation_derived_by_stratum": {
            stratum: {"positive": rs[0]["positive"], "flagged": flagged(rs, threshold)}
            for stratum, rs in group_by(derived, "stratum").items()
        },
        "r11_vcd1_external_genuine": {
            "flagged": flagged(vcd1, threshold),
            "by_subset": {f: flagged(rs, threshold) for f, rs in group_by(vcd1, "family").items()},
        },
        "r15_oos_genuine_stratum": {
            "flagged": flagged(r15, threshold),
            "videos": [{
                "id": r["id"], "file_hash": r["file_hash"], "score": r["score"],
                "flagged": r["score"] > threshold, "frames": r["frames"],
                "calibration_genuine_percentile":
                    sum(1 for s in cal_sorted if s < r["score"]) / len(cal_sorted),
            } for r in sorted(r15, key=lambda r: r["id"])],
        },
        "score_summary": {
            name: {"n": len(rs), "min": min(scores(rs)), "median": float(np.median(scores(rs))),
                   "max": max(scores(rs))}
            for name, rs in [("calibration_genuine", cal_gen), ("calibration_positive", cal_pos),
                             ("evaluation_genuine", eva_gen), ("evaluation_positive", eva_pos),
                             ("vcd1_genuine", vcd1), ("r15_genuine", r15)] if rs
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    freeze = check_freeze(args.freeze)
    population = load_population()
    if len(population) != freeze["expected_population"]:
        sys.exit(f"refusing to run: population {len(population)} != frozen "
                 f"{freeze['expected_population']}")
    for record in population:
        if "file_hash" in record and sha256_file(Path(record["path"])) != record["file_hash"]:
            sys.exit(f"refusing to run: R15 file hash mismatch for {record['id']}")

    started = datetime.now(timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record, result in zip(population, pool.map(score_clip, [r["path"] for r in population],
                                                         chunksize=8)):
            record.update(result)

    output = {
        "task": "R15-T7",
        "method_id": METHOD_ID,
        "freeze_record": str(args.freeze),
        "freeze_record_sha256": sha256_file(args.freeze),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                 text=True).stdout.splitlines()[0],
        "numpy": np.__version__,
        "analysis": analyse(population),
        "clips": population,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
