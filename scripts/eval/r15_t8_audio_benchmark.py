"""R15-T8: offline benchmark of the deployed AASIST checkpoint on R15 out-of-sample audio.

The checkpoint, the preprocessing, the file-level reading, the score direction, the corpus and
the threshold rule are fixed in `docs/ai/r15_t8_freeze.md` and pinned by
`docs/ai/r15_t8_freeze.json` before any corpus file is scored. This runner refuses to start if
its own bytes, the detector module, the audio-preparation module, the checkpoint or the R15-T5
manifest differ from the hashes recorded there, so the configuration that produced the numbers is
the one that was declared, and only one configuration is ever run.

Detector: the production `app/audio_detector.py`, loaded by file path (it imports nothing from
`app`, and loading it this way skips the package's credential gate), over the checkpoint the
worker image carries. Nothing about it is reimplemented here.

Preprocessing: the production `speaker_diarization._extract_audio` FFmpeg arguments, verbatim
(first audio stream -> mono, 16 kHz, pcm_s16le WAV), then `analyze_audio_authenticity`.

File-level reading: `windows[0].bona_fide_logit` - the first 64600 samples, tiled if the file is
shorter. This is upstream's evaluation input (`clovaai/aasist` `data_utils.pad()`: truncate to
64600, else repeat), so no DeepGuard aggregate is introduced. Later windows are kept in the raw
output and no metric is computed from them.

Direction is declared, not learned: higher `bona_fide_logit` = more bona fide
(`clovaai/aasist/main.py:307`). A file is called spoof when its logit is below the threshold. An
AUROC below 0.5 is reported as measured and is never flipped.

Threshold: the equal-error point of the CALIBRATION split of `primary_oos.audio` (all genuine and
all spoof files), computed once and applied unchanged to HOLDOUT. HOLDOUT is the primary result
and is threshold-free (AUROC with a speaker-cluster bootstrap, average precision) plus the rates
at the frozen threshold.

Nothing here touches the database, the API or the risk engine. Output is one JSON file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval.stats import rate

METHOD_ID = "AASIST-2019LA-W0"
DETECTOR_MODULE = Path("/app/app/audio_detector.py")
PREPARATION_MODULE = Path("/app/app/speaker_diarization.py")
MODEL_PATH = Path("/models/aasist.onnx")
R15_MANIFEST = Path("/home/adentechio/deepguard-corpus/r15_oos_manifest.json")
AASIST_TARGETED = ("A18", "A20", "A23", "A30")
# Pre-declared within-chain comparisons: the targeted attack against the attack it is built on,
# where that base attack is itself in primary_oos (A20's A12 and A23's A09 are not).
TARGETED_BASE = {"A18": "A17", "A30": "A18"}
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261001
GENUINE, SPOOF = "GENUINE", "AUDIO_MANIPULATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detector():
    spec = importlib.util.spec_from_file_location("frozen_audio_detector", DETECTOR_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_freeze(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    if freeze["method_id"] != METHOD_ID:
        sys.exit("refusing to run: method id differs from the freeze record")
    pinned = {
        Path(__file__).resolve(): freeze["runner_sha256"],
        DETECTOR_MODULE: freeze["detector_module_sha256"],
        PREPARATION_MODULE: freeze["preparation_module_sha256"],
        MODEL_PATH: freeze["checkpoint"]["sha256"],
        R15_MANIFEST: freeze["manifest_sha256"],
    }
    drift = [str(p) for p, h in pinned.items() if sha256_file(p) != h]
    if drift:
        sys.exit(f"refusing to run: differs from the freeze record: {drift}")
    if load_detector().MODEL_SHA256 != freeze["checkpoint"]["sha256"]:
        sys.exit("refusing to run: audio_detector.MODEL_SHA256 differs from the frozen checkpoint")
    return freeze


def load_population(expected: dict) -> tuple[list[dict], dict]:
    """primary_oos.audio and nothing else, with the corpus checks the packet asks for."""
    manifest = json.loads(R15_MANIFEST.read_text())
    records = manifest["primary_oos"]["audio"]
    auxiliary_groups = manifest["auxiliary_known_family"]
    auxiliary = {r["file_hash"] for group in auxiliary_groups.values() for r in group}
    video = {r["file_hash"] for r in manifest["primary_oos"]["video"]}
    video |= {r["file_hash"] for r in auxiliary_groups["video_auxiliary_genuine_reference"]}
    excluded = {h for hashes in manifest["excluded_must_not_enter"].values() for h in hashes}

    population = [{
        "id": r["recording_identity"], "path": r["file_path"], "file_hash": r["file_hash"],
        "split": r["dataset_split"], "label": r["label"], "speaker": r["speaker_id"],
        "attack_id": r["attack_id"] or "bonafide", "attack_family": r["attack_family"],
        "stratum": r["stratum_primary"], "media_type": r["media_type"],
    } for r in records]

    hashes = {r["file_hash"] for r in population}
    speakers = {split: {r["speaker"] for r in population if r["split"] == split}
                for split in ("CALIBRATION", "HOLDOUT")}
    counts = Counter(f"{r['split']}/{r['label']}" for r in population)
    checks = {
        "only_media_type_audio": all(r["media_type"] == "audio" for r in population),
        "auxiliary_known_family_hashes_loaded": len(hashes & auxiliary),
        "auxiliary_known_family_attack_ids_loaded":
            sorted({r["attack_id"] for r in population} & {"A19", "A29"}),
        "video_hashes_loaded": len(hashes & video),
        "excluded_must_not_enter_hashes_loaded": len(hashes & excluded),
        "speaker_overlap_calibration_holdout": len(speakers["CALIBRATION"] & speakers["HOLDOUT"]),
        "speakers_by_split": {s: len(v) for s, v in speakers.items()},
        "files_by_split_and_label": dict(sorted(counts.items())),
        "duplicate_hashes": len(population) - len(hashes),
    }
    failures = [name for name, bad in [
        ("non-audio record", not checks["only_media_type_audio"]),
        ("auxiliary hash", checks["auxiliary_known_family_hashes_loaded"]),
        ("auxiliary attack id", checks["auxiliary_known_family_attack_ids_loaded"]),
        ("video hash", checks["video_hashes_loaded"]),
        ("excluded hash", checks["excluded_must_not_enter_hashes_loaded"]),
        ("speaker overlap", checks["speaker_overlap_calibration_holdout"]),
        ("duplicate hash", checks["duplicate_hashes"]),
        ("split/label counts", checks["files_by_split_and_label"] != expected),
    ] if bad]
    if failures:
        sys.exit(f"refusing to run: corpus checks failed: {failures} {checks}")
    return population, checks


def extract_wav(source: str, destination: str) -> None:
    # Verbatim from speaker_diarization._extract_audio (pinned by hash in the freeze record).
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", source, "-vn", "-map", "0:a:0", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", destination],
        capture_output=True, check=True,
    )


def score_file(record: dict) -> dict:
    if sha256_file(Path(record["path"])) != record["file_hash"]:
        return {"score": None, "windows": [], "unscorable": "file hash differs from manifest"}
    detector = load_detector()
    with tempfile.TemporaryDirectory() as scratch:
        wav = str(Path(scratch) / "prepared.wav")
        try:
            extract_wav(record["path"], wav)
            evidence = detector.analyze_audio_authenticity(Path(wav), model_path=MODEL_PATH)
        except subprocess.CalledProcessError as error:
            return {"score": None, "windows": [],
                    "unscorable": f"ffmpeg failed: {error.stderr.decode(errors='replace')[-200:]}"}
        except detector.AudioDetectorError as error:
            return {"score": None, "windows": [], "unscorable": f"{type(error).__name__}: {error}"}
    windows = [{"index": w.window_index, "start": w.start_sample, "end": w.end_sample,
                "padded": w.padded_samples, "logits": list(w.logits),
                "bona_fide_logit": w.bona_fide_logit} for w in evidence.windows]
    return {"score": windows[0]["bona_fide_logit"], "total_samples": evidence.total_samples,
            "windows": windows, "unscorable": None}


def auroc(genuine: list[float], spoof: list[float]) -> float | None:
    """P(genuine logit > spoof logit), ties = 1/2 (Mann-Whitney). Direction never flipped."""
    if not genuine or not spoof:
        return None
    ordered = sorted([(s, 1) for s in genuine] + [(s, 0) for s in spoof])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2.0 * sum(flag for _, flag in ordered[i:j])
        i = j
    n_gen, n_spoof = len(genuine), len(spoof)
    return (rank_sum - n_gen * (n_gen + 1) / 2.0) / (n_gen * n_spoof)


def average_precision(genuine: list[float], spoof: list[float]) -> float | None:
    """AP with spoof as the positive class, ranked by ascending bona fide logit; ties as one block."""
    if not spoof:
        return None
    scored = sorted([(s, 1) for s in spoof] + [(s, 0) for s in genuine])
    ap, tp, fp, recall_prev, i = 0.0, 0, 0, 0.0, 0
    while i < len(scored):
        j = i
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        tp += sum(f for _, f in scored[i:j])
        fp += sum(1 - f for _, f in scored[i:j])
        recall = tp / len(spoof)
        ap += (recall - recall_prev) * (tp / (tp + fp))
        recall_prev = recall
        i = j
    return ap


def bootstrap_auroc(genuine: list[dict], spoof: list[dict]) -> list[float] | None:
    """Percentile CI, resampling speakers (not files) so a speaker's files move together."""
    if not genuine or not spoof:
        return None
    by_speaker: dict[str, list[dict]] = {}
    for record in genuine + spoof:
        by_speaker.setdefault(record["speaker"], []).append(record)
    speakers = sorted(by_speaker)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    while len(draws) < BOOTSTRAP_RESAMPLES:
        sample = [r for _ in speakers for r in by_speaker[rng.choice(speakers)]]
        value = auroc([r["score"] for r in sample if r["label"] == GENUINE],
                      [r["score"] for r in sample if r["label"] == SPOOF])
        if value is not None:
            draws.append(value)
    draws.sort()
    return [draws[int(0.025 * BOOTSTRAP_RESAMPLES)], draws[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]]


def equal_error_threshold(genuine: list[float], spoof: list[float]) -> dict:
    """Spoof iff score < t. Candidates are the observed scores; minimise |FPR - FNR|, then lowest t."""
    best = None
    for t in sorted(set(genuine + spoof)):
        fpr = sum(1 for s in genuine if s < t) / len(genuine)
        fnr = sum(1 for s in spoof if s >= t) / len(spoof)
        key = (abs(fpr - fnr), t)
        if best is None or key < best[0]:
            best = (key, t, fpr, fnr)
    _, t, fpr, fnr = best
    return {"value": t, "fpr": fpr, "fnr": fnr, "eer": (fpr + fnr) / 2.0}


def called_spoof(records: list[dict], threshold: float) -> dict:
    return rate(sum(1 for r in records if r["score"] < threshold), len(records)).as_dict()


def summary(records: list[dict]) -> dict | None:
    values = [r["score"] for r in records]
    if not values:
        return None
    return {"n": len(values), "min": min(values), "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)), "p75": float(np.percentile(values, 75)),
            "max": max(values)}


def analyse(population: list[dict]) -> dict:
    scored = [r for r in population if r["score"] is not None]
    pick = lambda split, label: [r for r in scored if r["split"] == split and r["label"] == label]  # noqa: E731
    scores = lambda records: [r["score"] for r in records]  # noqa: E731
    cal_gen, cal_spoof = pick("CALIBRATION", GENUINE), pick("CALIBRATION", SPOOF)
    hold_gen, hold_spoof = pick("HOLDOUT", GENUINE), pick("HOLDOUT", SPOOF)
    threshold = equal_error_threshold(scores(cal_gen), scores(cal_spoof))
    t = threshold["value"]

    def discrimination(spoof: list[dict], genuine: list[dict]) -> dict:
        return {"n_spoof": len(spoof), "n_genuine": len(genuine),
                "auroc": auroc(scores(genuine), scores(spoof)),
                "auroc_speaker_bootstrap_95": bootstrap_auroc(genuine, spoof),
                "average_precision_spoof_positive": average_precision(scores(genuine), scores(spoof)),
                "spoof_prevalence": len(spoof) / (len(spoof) + len(genuine))}

    def by_attack(spoof: list[dict], genuine: list[dict]) -> dict:
        groups: dict[str, list[dict]] = {}
        for record in spoof:
            groups.setdefault(record["attack_id"], []).append(record)
        return {attack: {"attack_family": rs[0]["attack_family"], "stratum": rs[0]["stratum"],
                         "auroc_vs_split_genuine": auroc(scores(genuine), scores(rs)),
                         "called_spoof_at_threshold": called_spoof(rs, t),
                         "logit": summary(rs)}
                for attack, rs in sorted(groups.items())}

    targeted = lambda records: [r for r in records if r["attack_id"] in AASIST_TARGETED]  # noqa: E731
    untargeted = lambda records: [r for r in records if r["attack_id"] not in AASIST_TARGETED]  # noqa: E731
    strata = sorted({r["stratum"] for r in hold_spoof})

    return {
        "threshold": {**threshold, "rule": "calibration-split equal-error point over all "
                      "calibration genuine and spoof files; spoof iff bona_fide_logit < value",
                      "derived_from": {"genuine": len(cal_gen), "spoof": len(cal_spoof)}},
        "unscorable": [{"id": r["id"], "split": r["split"], "label": r["label"],
                        "attack_id": r["attack_id"], "reason": r["unscorable"]}
                       for r in population if r["score"] is None],
        "calibration_descriptive": discrimination(cal_spoof, cal_gen),
        "holdout": {
            "all_attacks": {**discrimination(hold_spoof, hold_gen),
                            "genuine_called_spoof_at_threshold": called_spoof(hold_gen, t),
                            "spoof_called_spoof_at_threshold": called_spoof(hold_spoof, t),
                            "holdout_eer_descriptive": equal_error_threshold(
                                scores(hold_gen), scores(hold_spoof))["eer"]},
            "excluding_aasist_targeted": {
                **discrimination(untargeted(hold_spoof), hold_gen),
                "spoof_called_spoof_at_threshold": called_spoof(untargeted(hold_spoof), t)},
            "by_stratum": {s: {**discrimination([r for r in hold_spoof if r["stratum"] == s],
                                                hold_gen),
                               "called_spoof_at_threshold":
                                   called_spoof([r for r in hold_spoof if r["stratum"] == s], t)}
                           for s in strata},
            "by_attack": by_attack(hold_spoof, hold_gen),
        },
        "aasist_targeted_stress": {
            "attacks": list(AASIST_TARGETED),
            "holdout_pooled": {**discrimination(targeted(hold_spoof), hold_gen),
                               "called_spoof_at_threshold": called_spoof(targeted(hold_spoof), t)},
            "holdout_by_attack": {a: v for a, v in by_attack(hold_spoof, hold_gen).items()
                                  if a in AASIST_TARGETED},
            "calibration_by_attack_descriptive": {
                a: v for a, v in by_attack(cal_spoof, cal_gen).items() if a in AASIST_TARGETED},
            "holdout_vs_base_attack": {
                attack: {"base": base,
                         "auroc_targeted_vs_genuine": auroc(
                             scores(hold_gen), scores([r for r in hold_spoof if r["attack_id"] == attack])),
                         "auroc_base_vs_genuine": auroc(
                             scores(hold_gen), scores([r for r in hold_spoof if r["attack_id"] == base])),
                         "median_logit_targeted": float(np.median(scores(
                             [r for r in hold_spoof if r["attack_id"] == attack]))),
                         "median_logit_base": float(np.median(scores(
                             [r for r in hold_spoof if r["attack_id"] == base])))}
                for attack, base in TARGETED_BASE.items()},
        },
        "logit_summary": {name: summary(rs) for name, rs in [
            ("calibration_genuine", cal_gen), ("calibration_spoof", cal_spoof),
            ("holdout_genuine", hold_gen), ("holdout_spoof", hold_spoof)]},
        "windows_per_file": dict(sorted(Counter(len(r["windows"]) for r in scored).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    freeze = check_freeze(args.freeze)
    population, corpus_checks = load_population(freeze["expected_files_by_split_and_label"])

    started = datetime.now(timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record, result in zip(population, pool.map(score_file, population, chunksize=8)):
            record.update(result)

    import onnxruntime
    output = {
        "task": "R15-T8",
        "method_id": METHOD_ID,
        "freeze_record": str(args.freeze),
        "freeze_record_sha256": sha256_file(args.freeze),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True,
                                 text=True).stdout.splitlines()[0],
        "numpy": np.__version__,
        "onnxruntime": onnxruntime.__version__,
        "corpus_checks": corpus_checks,
        "analysis": analyse(population),
        "files": population,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
