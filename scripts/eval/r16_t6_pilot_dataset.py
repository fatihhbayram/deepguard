"""R16-T6: build, seal and verify the blind Real-World Pilot dataset.

    # 1. synthetic speech for the audio-substitution cases (Kokoro venv; CPU; offline)
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= <kokoro-venv>/bin/python scripts/eval/r16_t6_pilot_dataset.py \
        tts --work ~/deepguard-corpus/r16_pilot_raw/build
    # 2. media, both manifests, freshness (api venv, so the GT vocabulary is imported)
    cd apps/api && PYTHONPATH=. .venv/bin/python ../../scripts/eval/r16_t6_pilot_dataset.py build \
        --work ~/deepguard-corpus/r16_pilot_raw/build --out ~/deepguard-corpus/r16_pilot \
        --root ~/deepguard-corpus --root ~/deepguard-p4-media --root ~/deepguard-corpus-r7t5-private \
        --root ~/deepguard --root ~/.cache/huggingface --db-container deepguard-postgres-1
    # 3. seal (pure function of the two manifests and the media bytes), 4. leakage + integrity check
    ... r16_t6_pilot_dataset.py seal   --out ~/deepguard-corpus/r16_pilot
    ... r16_t6_pilot_dataset.py verify --out ~/deepguard-corpus/r16_pilot

Nothing is uploaded, nothing is written to the database, no detector runs. The database is only
read, inside a READ ONLY transaction, to show no pilot file is already held there.

**Two artifacts, one direction of knowledge.** `pilot_manifest.json` is what the T7 runner
reads: neutral case id, neutral path, SHA-256, modality, nothing else. Case ids are assigned in
SHA-256 order, so neither the id nor the position carries the build order or the stratum.
Media are byte copies (no symlink can point back at a labelled source path) with one fixed
mtime. `sealed/pilot_ground_truth.json` holds labels, strata, method, provenance, licence and
freshness, and stays unread until T7 scoring is done.

**Lineage, not files, is the unit.** 34 of the cases descend from 10 camera originals of one
owner, one device, one session; T7 must score by `lineage_cluster`, not as 34 independent
observations. A splice or an audio swap joins its two roots into one cluster.

**Encoding parity.** The originals are never touched. Every project-made case, genuine
derivative or manipulation alike, goes through one of three fixed export paths (P0 stream copy,
P1 messaging-like, P2 social-like), so container and codec do not separate the labels.

**Deterministic seal.** The seal is computed from the bytes on disk, with sorted keys, fixed
separators and no timestamps: a rerun over the same files is byte-identical.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXEC_SCHEMA = "r16-t6-pilot-exec-1"
GT_SCHEMA = "r16-t6-pilot-gt-1"
SEAL_SCHEMA = "r16-t6-pilot-seal-1"
CASE_PREFIX = "R16P-"
FIXED_MTIME = 1791331200  # 2026-10-07T00:00:00Z for every pilot file
MODALITY = "video"  # production accepts video/mp4 and video/quicktime only (analyses.py)

SOURCE_DIR = Path.home() / "deepguard-p4-media/R15-T2-Real-Video"
OWNER_LINEAGE = "r16-owner-session-2026-10-07"
OWNER_CONSENT_REF = "r16_pilot_raw/genuine_source/SOURCES.txt#OWNER-INPUT-2026-10-07"
# Frozen at intake (r16_pilot_raw/genuine_source/metadata/SHA256SUMS.txt) with the owner's notes.
ROOTS = {
    "IMG_1018": ("9a3a65ecd7eac046a124730b3331d61c8dbe229a7200fc31080b61cdbf5891ca", "indoor · frontal · medium · handheld"),
    "IMG_1019": ("034e1464728a0c3641f12fc37d9f7c073f2ddcab14bfea369c9b96b65cccaab1", "indoor · slight-left · medium · handheld"),
    "IMG_1020": ("0ed52c14dc2a11ad8167465f66176574ff0772e614201b56996bf67d531274d4", "indoor · slight-right · medium · handheld"),
    "IMG_1021": ("0c15c45f89944afadabeef19d221d24b9777e8272da8c9f406cdc6ef1b024816", "indoor · frontal · close · steady"),
    "IMG_1022": ("c6430537559929bebe7c5c7b1332e4098d6b02a7375d722867e75fed614fb691", "indoor · moving · medium · handheld"),
    "IMG_1023": ("78aec7c91510b09316be4b7eeada7c7af32046f453a6e37020b9af8d02b2df70", "indoor · frontal · medium · steady"),
    "IMG_1024": ("afc56c5679063a9b50530d8f6c1bf136563ed57547c9ba214bb22270c37e084f", "indoor · slight-left · close · handheld"),
    "IMG_1025": ("31699de660447d0559c677039d8b37dba09d6a23485fa686acfdb358126f6eb6", "indoor · frontal · medium · handheld"),
    "IMG_1026": ("c786884b2e1095de5df0feb3fc6f7ab9d440af137fb008e4973cb61288d61375", "indoor · slight-right · medium · handheld"),
    "IMG_1028": ("897cc0b3fcc750ef99c7c72344549c7cdceb82081dd9c775781c2a358619d2db", "indoor · moving · medium · handheld"),
}

TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv")
PATHS = {
    "P1": {"desc": "messaging-like: HLG->SDR, 480 px wide, H.264 Main CRF 28 <=1 Mbit/s, AAC 64k mono, metadata stripped",
           "scale": "480:-2", "v": ["-profile:v", "main", "-crf", "28", "-maxrate", "1M", "-bufsize", "2M"],
           "a": ["-b:a", "64k", "-ac", "1"]},
    "P2": {"desc": "social re-upload-like: HLG->SDR, 1080 px wide, H.264 High CRF 23 <=4 Mbit/s, AAC 128k, metadata stripped",
           "scale": "1080:-2", "v": ["-profile:v", "high", "-crf", "23", "-maxrate", "4M", "-bufsize", "8M"],
           "a": ["-b:a", "128k", "-ac", "2"]},
    "P0": {"desc": "stream copy into QuickTime, camera metadata kept (Apple tags; timed-metadata tracks dropped)"},
}

# Kokoro-82M (Apache-2.0), pinned revision; voices from the same revision.
KOKORO_REPO, KOKORO_REV = "hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987"
TTS_LINES = {
    "t1": ("af_heart", "a", "I never agreed to that contract, and I want everyone to know it."),
    "t2": ("am_michael", "a", "The meeting is cancelled. Please transfer the deposit to the new account today."),
    "t3": ("bf_emma", "b", "We tested the product ourselves, and the results were completely fine."),
    "t4": ("am_michael", "a", "I can confirm the payment was sent this morning from my personal account."),
    "t5": ("af_heart", "a", "Please ignore my earlier message. The delivery address has changed."),
    "t6": ("bf_emma", "b", "This is my official statement, and I will not be commenting any further."),
}

# key -> recipe. kind: original | transcode | edit | tts. roots: owner clips the case descends from.
RECIPES = {
    **{f"orig_{r}": {"kind": "original", "roots": [r]} for r in ROOTS},
    **{f"deriv_P1_{r}": {"kind": "transcode", "roots": [r], "path": "P1"}
       for r in ("IMG_1018", "IMG_1020", "IMG_1022", "IMG_1024", "IMG_1026")},
    **{f"deriv_P2_{r}": {"kind": "transcode", "roots": [r], "path": "P2"}
       for r in ("IMG_1019", "IMG_1021", "IMG_1023", "IMG_1025", "IMG_1028")},
    "edit_segment_deletion": {"kind": "edit", "roots": ["IMG_1028"], "path": "P0",
                              "method": "segment deletion: [9.500 s, 18.035 s) removed at keyframes, rest joined by stream copy; the open-GOP junction drops a few undecodable leading frames, as lossless-cut tools do"},
    "edit_splice": {"kind": "edit", "roots": ["IMG_1021", "IMG_1023"], "path": "P0",
                    "method": "splice: IMG_1021 [0, 4.267 s) + IMG_1023 [4.267 s, end), joined at keyframes by stream copy; the open-GOP junction drops a few undecodable leading frames, as lossless-cut tools do"},
    "edit_av_swap": {"kind": "edit", "roots": ["IMG_1019", "IMG_1022"], "path": "P0",
                     "method": "audio swap: IMG_1019 video with IMG_1022 audio, cut to the shorter stream"},
    "edit_slowdown": {"kind": "edit", "roots": ["IMG_1024"], "path": "P1",
                      "method": "slow-down to 0.75x speed, audio time-stretched (pitch kept)"},
    "edit_speedup": {"kind": "edit", "roots": ["IMG_1026"], "path": "P2",
                     "method": "speed-up to 1.25x, audio time-stretched (pitch kept)"},
    "edit_jumpcut": {"kind": "edit", "roots": ["IMG_1018"], "path": "P1",
                     "method": "jump cut: [3.0 s, 4.5 s) removed from video and audio"},
    "edit_reorder": {"kind": "edit", "roots": ["IMG_1020"], "path": "P2",
                     "method": "reorder: second half [3.7 s, end) placed before first half"},
    "edit_mute": {"kind": "edit", "roots": ["IMG_1022"], "path": "P1",
                  "method": "speech removal: audio silenced over [2.5 s, 5.0 s)"},
    "tts_t1": {"kind": "tts", "roots": ["IMG_1023"], "path": "P0", "tts": "t1"},
    "tts_t2": {"kind": "tts", "roots": ["IMG_1025"], "path": "P0", "tts": "t2"},
    "tts_t3": {"kind": "tts", "roots": ["IMG_1021"], "path": "P1", "tts": "t3"},
    "tts_t4": {"kind": "tts", "roots": ["IMG_1019"], "path": "P1", "tts": "t4"},
    "tts_t5": {"kind": "tts", "roots": ["IMG_1024"], "path": "P2", "tts": "t5"},
    "tts_t6": {"kind": "tts", "roots": ["IMG_1026"], "path": "P2", "tts": "t6"},
}

# Anchors: R7-R15 benchmark media, evaluation split only, base clips only (no derivation), never a
# file that also sits in a calibration set. 7 AI-visual (the class no fresh case covers) and 3
# genuine, so "not the owner's footage" does not by itself mean "manipulated". talkinghead_sonic is not
# offered: all 10 of its R7-T5 evaluation clips are byte-identical to R4-T1 calibration clips.
# A candidate enters the pilot only if its licence explicitly allows internal commercial evaluation
# (LICENCE_CLEAR). Every R7 clip is under research-use terms, so the AI-visual picks go to the
# research_only_reference sidecar, which is not sealed with the pilot and not scored in T7.
# The licence-clear genuine picks (VCD1) go there too (Architect 2026-10-07): with no manipulated
# anchor left, "not the owner" would mean "genuine" to anyone who knows the owner. The T7 set is
# therefore anchor-free; the selection is kept so the sidecar records what was considered.
ANCHOR_FAMILIES = [("r7t5", "faceswap_inswapper"), ("r7t5", "faceswap_roop"), ("r7t9", "inthewild_social_deepfake"),
                   ("r7t9", "faceswap_e4s"), ("r7t9", "reenact_hyperreenact"), ("r7t9", "t2v_gen3"),
                   ("r7t9", "t2v_pyramidflow")]
ANCHOR_GENUINE_VCD1 = 3
LICENCE_CLEAR = {"Community Data License Agreement - Permissive - Version 2.0 (CDLA-Permissive-2.0)"}
ANCHOR_LABEL = {"face_swap": "FACE_SWAP", "synthetic": "AI_GENERATED"}

STRATA = ["genuine_original", "genuine_derivative", "manipulated_edit", "manipulated_audio_tts",
          "anchor_manipulated_ai_visual", "anchor_genuine", "unknown_truth"]
LIMITATIONS = [
    "Claim scope: controlled blind product pilot on a single-subject, single-session-derived population; not a population-generalization benchmark.",
    "All 34 cases descend from one owner, one device (iPhone 15), one 3.5-minute indoor session; 10/10 indoor, no low-light take.",
    "The person on camera is the same person as in the R15-T2 genuine set (owner statement): subject lineage overlaps R15.",
    "Fresh AI-visual manipulation (face-swap, lip-sync, image-to-video): 0 cases. No licence-clean local tool (R15-T3; 5.75 GB VRAM); cloud excluded by owner consent and Architect. No AI-visual case is in the T7 set at all: the 7 R7 AI-visual anchors were moved to the research_only_reference sidecar (research-use licence, Architect 2026-10-07).",
    "unknown_truth: 0 cases (Architect ruling: no source sought to fill it).",
    "Speech of the 6 audio-substitution cases is English Kokoro TTS over the owner's video; voice and lips do not match by construction.",
    "Kokoro was used before in the R15-T9 spike (stimuli only, not a benchmark corpus or detector training set).",
    "Edits that keep camera metadata (P0) still lose the Apple timed-metadata tracks and gain an encoder tag; that container change is evidence a forensic tool may legitimately read, not label leakage.",
    "No anchors: R7 AI-visual anchors (licence unclear) and VCD1 genuine anchors (blind leakage by content) are in the research_only_reference sidecar, outside T7 scoring.",
    "R7-T5 evaluation clips overlap the production R4-T1 calibration set byte-for-byte (carry-forward defect); no such clip is in this set.",
]
EXPECTED = {
    "genuine_original": "Unaltered camera original. Expected: no manipulation finding. Camera metadata is intact.",
    "genuine_derivative": "Genuine content re-exported through a common channel. Loss of camera metadata and recompression must not be read as manipulation.",
    "manipulated_edit": "Edited without AI (cut/splice/speed/reorder/audio swap/mute). Current detectors are not built for edit forensics; a miss is a capability gap to record, not a ruleset change.",
    "manipulated_audio_tts": "Original video with its speech replaced by synthetic TTS. Expected evidence, if any, is on the audio or audio-visual side.",
    "anchor_manipulated_ai_visual": "Known R7 benchmark manipulation, used as a reference point. Earlier results exist; they are not fresh evidence.",
    "anchor_genuine": "Known genuine benchmark clip (VCD1), used as a reference point.",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1, separators=(",", ": ")) + "\n").encode()


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], check=True, capture_output=True, text=True).stdout)


# ---------------------------------------------------------------------------------------- tts

def cmd_tts(args):
    from kokoro import KModel, KPipeline  # noqa: PLC0415 - only the Kokoro venv has it
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    import kokoro  # noqa: PLC0415

    out = Path(args.work) / "tts"
    out.mkdir(parents=True, exist_ok=True)
    # The pinned snapshot, by path: offline mode cannot resolve a branch ref.
    snap = Path.home() / ".cache/huggingface/hub" / ("models--" + KOKORO_REPO.replace("/", "--")) / "snapshots" / KOKORO_REV
    model = KModel(repo_id=KOKORO_REPO, config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).eval()
    pipes, index = {}, {}
    for key, (voice, lang, text) in sorted(TTS_LINES.items()):
        p = pipes.setdefault(lang, KPipeline(lang_code=lang, repo_id=KOKORO_REPO, model=model))
        audio = np.concatenate([r.audio.numpy() for r in p(text, voice=str(snap / "voices" / f"{voice}.pt"))])
        path = out / f"{key}.wav"
        sf.write(path, audio, 24000, subtype="PCM_16")
        index[key] = {"voice": voice, "lang_code": lang, "text": text, "sha256": sha256(path),
                      "seconds": round(len(audio) / 24000, 3)}
    meta = {"model": KOKORO_REPO, "revision": KOKORO_REV, "license": "Apache-2.0",
            "kokoro_version": kokoro.__version__, "device": "cpu",
            "note": "not bit-reproducible across runs; frozen by content hash", "lines": index}
    (out / "tts_index.json").write_bytes(canonical(meta))
    print(json.dumps(meta, indent=1))


# -------------------------------------------------------------------------------------- build

def encode(inputs, graph, path_key, out, audio_label="[a]"):
    """Re-encode through P1/P2. `graph` must produce [v0] (edited video) and `audio_label`."""
    p = PATHS[path_key]
    full = f"{graph};[v0]{TONEMAP},scale={p['scale']},format=yuv420p[v]"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for i in inputs:
        cmd += ["-i", str(i)]
    cmd += ["-filter_complex", full, "-map", "[v]", "-map", audio_label, "-r", "30",
            "-c:v", "libx264", "-preset", "medium", *p["v"], "-c:a", "aac", "-ar", "48000", *p["a"],
            "-map_metadata", "-1", "-movflags", "+faststart", str(out)]
    run(cmd)


def p0_copy(cmd_inputs, maps, out, meta_from=0, extra=()):
    run(["ffmpeg", "-v", "error", "-y", *cmd_inputs, *maps, "-c", "copy", *extra, "-tag:v", "hvc1",
         "-map_metadata", str(meta_from), "-movflags", "use_metadata_tags", str(out)])


def concat_copy(parts, out, meta_src, tmp):
    lst = tmp / (out.stem + "_concat.txt")
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(meta_src),
         "-map", "0:v", "-map", "0:a", "-c", "copy", "-tag:v", "hvc1", "-map_metadata", "1",
         "-movflags", "use_metadata_tags", str(out)])


def build_one(key, rec, src, work, tts_dir):
    tmp = work / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    r0 = src[rec["roots"][0]]
    kind, pk = rec["kind"], rec.get("path")
    ext = ".mov" if kind == "original" or pk == "P0" else ".mp4"
    out = work / "cases" / (key + ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind == "original":
        shutil.copyfile(r0, out)
    elif kind == "transcode":
        encode([r0], "[0:v]null[v0];[0:a]anull[a]", pk, out)
    elif key == "edit_segment_deletion":
        a, b = tmp / "segdel_a.mov", tmp / "segdel_b.mov"
        p0_copy(["-i", str(r0)], ["-map", "0:v", "-map", "0:a"], a, extra=("-t", "9.5"))
        p0_copy(["-ss", "18.035", "-i", str(r0)], ["-map", "0:v", "-map", "0:a"], b)
        concat_copy([a, b], out, r0, tmp)
    elif key == "edit_splice":
        r1 = src[rec["roots"][1]]
        a, b = tmp / "splice_a.mov", tmp / "splice_b.mov"
        p0_copy(["-i", str(r0)], ["-map", "0:v", "-map", "0:a"], a, extra=("-t", "4.266667"))
        p0_copy(["-ss", "4.266667", "-i", str(r1)], ["-map", "0:v", "-map", "0:a"], b)
        concat_copy([a, b], out, r0, tmp)
    elif key == "edit_av_swap":
        r1 = src[rec["roots"][1]]
        p0_copy(["-i", str(r0), "-i", str(r1)], ["-map", "0:v", "-map", "1:a"], out, extra=("-shortest",))
    elif key == "edit_slowdown":
        encode([r0], "[0:v]setpts=PTS/0.75[v0];[0:a]atempo=0.75[a]", pk, out)
    elif key == "edit_speedup":
        encode([r0], "[0:v]setpts=PTS/1.25[v0];[0:a]atempo=1.25[a]", pk, out)
    elif key == "edit_jumpcut":
        encode([r0], "[0:v]select='not(between(t,3,4.5))',setpts=N/FRAME_RATE/TB[v0];"
                     "[0:a]aselect='not(between(t,3,4.5))',asetpts=N/SR/TB[a]", pk, out)
    elif key == "edit_reorder":
        encode([r0], "[0:v]split[x][y];[x]trim=start=3.7,setpts=PTS-STARTPTS[v2];[y]trim=end=3.7,setpts=PTS-STARTPTS[v1];"
                     "[0:a]asplit[p][q];[p]atrim=start=3.7,asetpts=PTS-STARTPTS[a2];[q]atrim=end=3.7,asetpts=PTS-STARTPTS[a1];"
                     "[v2][a2][v1][a1]concat=n=2:v=1:a=1[v0][a]", pk, out)
    elif key == "edit_mute":
        encode([r0], "[0:v]null[v0];[0:a]volume=enable='between(t,2.5,5.0)':volume=0[a]", pk, out)
    elif kind == "tts":
        wav = tts_dir / f"{rec['tts']}.wav"
        dur = duration(r0)
        if pk == "P0":
            run(["ffmpeg", "-v", "error", "-y", "-i", str(r0), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-tag:v", "hvc1", "-af", f"apad,atrim=0:{dur:.3f}", "-c:a", "aac",
                 "-ar", "48000", "-ac", "2", "-b:a", "160k", "-t", f"{dur:.3f}", "-map_metadata", "0",
                 "-movflags", "use_metadata_tags", str(out)])
        else:
            encode([r0, wav], f"[0:v]null[v0];[1:a]aresample=48000,apad,atrim=0:{dur:.3f}[a]", pk, out)
    else:
        raise KeyError(key)
    return out


def walk(root, skip):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if any(os.path.realpath(dirpath).startswith(s) for s in skip):
            dirnames[:] = []
            continue
        for f in filenames:
            p = os.path.join(dirpath, f)
            if os.path.isfile(p) and not os.path.islink(p):
                yield p


def exact_hash_index(roots, wanted_sizes, skip):
    """Every on-disk file whose bytes equal a pilot file: exact, size-matched first."""
    hits = collections.defaultdict(list)
    for root in roots:
        for p in walk(root, skip):
            try:
                if os.path.getsize(p) not in wanted_sizes:
                    continue
            except OSError:
                continue
            hits[sha256(p)].append(p)
    return hits


def db_hashes(container):
    q = ("BEGIN TRANSACTION READ ONLY; SELECT original_sha256 FROM media_files UNION ALL "
         "SELECT derivative_sha256 FROM media_files WHERE derivative_sha256 IS NOT NULL UNION ALL "
         "SELECT media_sha256 FROM ground_truth UNION ALL SELECT media_sha256 FROM media_governance; COMMIT;")
    out = subprocess.run(["docker", "exec", container, "psql", "-U", "deepguard", "-d", "deepguard", "-At",
                          "-c", q], check=True, capture_output=True, text=True).stdout
    return {l.strip() for l in out.splitlines() if len(l.strip()) == 64}


def corpus_of(path):
    home = str(Path.home())
    rel = os.path.relpath(path, home)
    if rel.startswith("deepguard-p4-media/R15-T2-Real-Video/"):
        return "owner_source_folder"
    return rel.split(os.sep)[0] + ("/" + rel.split(os.sep)[1] if rel.startswith("deepguard-corpus/") else "")


# Calibration sets of the production detectors and of the R7 benchmarks. An anchor whose bytes sit
# in any of them would score a threshold on the data it was fitted to (R7-T5 inswapper_x_000 is
# byte-identical to R4-T1 calibration inswapper_en_00).
CALIBRATION_DIRS = ("r4t1_calibration", "ff++_c23_r3t1")


def calibration_hashes(corpus_root):
    hashes = set()
    for d in CALIBRATION_DIRS:
        hashes |= {sha256(p) for p in walk(corpus_root / d, []) if os.path.splitext(p)[1].lower() in (".mp4", ".mov")}
    for corpus in ("r7t5", "r7t9"):
        hashes |= {i["sha256"] for i in json.load(open(corpus_root / corpus / "corpus.json"))["items"]
                   if i.get("split") == "calibration" and i.get("sha256")}
    return hashes


def pick_anchors(corpus_root):
    calib, picks, skipped = calibration_hashes(corpus_root), [], []
    for corpus, family in ANCHOR_FAMILIES:
        items = json.load(open(corpus_root / corpus / "corpus.json"))["items"]
        cands = sorted((i for i in items if i.get("family") == family and i.get("split") == "evaluation"
                        and i.get("derivation") == "none"), key=lambda i: i["clip_id"])
        for it in cands:
            p = corpus_root / corpus / it["path"]
            if not p.is_file():
                continue
            if sha256(p) in calib:
                skipped.append(f"{corpus}/{it['clip_id']}")
                continue
            picks.append(("anchor_manipulated_ai_visual", corpus, it, p))
            break
        else:
            sys.exit(f"no calibration-free evaluation clip for {corpus}/{family}")
    print("anchors skipped (bytes in a calibration set):", skipped or "none", flush=True)
    vcd = json.load(open(corpus_root / "external-genuine-vcd1/vcd1_intake_manifest.json"))
    recs = sorted((r for r in vcd["file_records"] if r["subset"] == "th-m"), key=lambda r: r["subject_id"])
    for r in recs[:ANCHOR_GENUINE_VCD1]:
        picks.append(("anchor_genuine", "external-genuine-vcd1", r, corpus_root / "external-genuine-vcd1" / r["relative_path"]))
    return picks, vcd


def cmd_build(args):
    from app.ground_truth import derive_manipulation_family  # noqa: PLC0415

    work, out = Path(args.work).expanduser(), Path(args.out).expanduser()
    corpus_root = Path.home() / "deepguard-corpus"
    if (out / "pilot_manifest.json").exists() and not args.force:
        sys.exit(f"{out} already holds a pilot; refusing to overwrite (--force)")
    src = {}
    for r, (h, _) in ROOTS.items():
        p = SOURCE_DIR / f"{r}.MOV"
        if sha256(p) != h:
            sys.exit(f"source {p} does not match its intake hash")
        src[r] = p
    tts_dir = work / "tts"
    tts_meta = json.load(open(tts_dir / "tts_index.json"))
    for k, v in tts_meta["lines"].items():
        if sha256(tts_dir / f"{k}.wav") != v["sha256"]:
            sys.exit(f"tts {k} changed since synthesis")

    cases = []
    for key, rec in sorted(RECIPES.items()):
        built = work / "cases"
        existing = [p for p in built.glob(key + ".*")] if built.exists() else []
        path = existing[0] if existing and not args.rebuild else build_one(key, rec, src, work, tts_dir)
        cases.append({"key": key, "rec": rec, "file": path})
        print("built", key, flush=True)

    picks, vcd = pick_anchors(corpus_root)
    research_only = []
    for stratum, corpus, it, p in picks:  # every anchor goes to the sidecar; the T7 set has none
        lic = vcd["license"]["data_license_name"] if corpus == "external-genuine-vcd1" else it.get("license")
        clear = lic in LICENCE_CLEAR
        research_only.append({
            "benchmark": corpus, "benchmark_id": it.get("clip_id") or it.get("subject_id"),
            "family": it.get("family", "vcd1_genuine"), "label": "GENUINE" if clear else ANCHOR_LABEL[it["label"]],
            "path": str(p), "sha256": sha256(p), "license_type": lic,
            "commercial_internal_eval_allowed": True if clear else None,
            "source": it.get("source") or it.get("upstream_url"),
            "source_lineage_id": it.get("source_lineage_id") or f"vcd1:{it.get('subject_id')}",
            "exclusion_reason": ("blind leakage: the only non-owner cases would all be genuine" if clear
                                 else "research-use dataset terms; internal commercial evaluation not explicitly allowed")})

    for c in cases:
        c["sha256"] = sha256(c["file"])
        c["bytes"] = os.path.getsize(c["file"])
    shas = [c["sha256"] for c in cases]
    if len(set(shas)) != len(shas):
        sys.exit("duplicate bytes among pilot cases")

    skip = [os.path.realpath(out), os.path.realpath(work.parent)]
    disk = exact_hash_index([Path(r).expanduser() for r in args.root], {c["bytes"] for c in cases}, skip)
    db = db_hashes(args.db_container) if args.db_container else set()

    # Neutral ids in SHA-256 order; byte copies with one fixed mtime.
    cases.sort(key=lambda c: c["sha256"])
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    (out / "sealed").mkdir(exist_ok=True)
    exec_cases, gt_cases = [], []
    for n, c in enumerate(cases, 1):
        cid = f"{CASE_PREFIX}{n:03d}"
        ext = c["file"].suffix.lower()
        rel = f"media/{cid}{ext}"
        dst = out / rel
        shutil.copyfile(c["file"], dst)
        os.utime(dst, (FIXED_MTIME, FIXED_MTIME))
        if sha256(dst) != c["sha256"]:
            sys.exit(f"copy of {c['key']} differs")
        exec_cases.append({"case_id": cid, "path": rel, "sha256": c["sha256"], "modality": MODALITY})

        hits = [corpus_of(p) for p in disk.get(c["sha256"], [])]
        rec, kind = c["rec"], c["rec"]["kind"]
        stratum = {"original": "genuine_original", "transcode": "genuine_derivative",
                   "edit": "manipulated_edit", "tts": "manipulated_audio_tts"}[kind]
        label = {"original": "GENUINE", "transcode": "GENUINE", "edit": "OTHER_MANIPULATION",
                 "tts": "AUDIO_MANIPULATION"}[kind]
        if kind == "original":
            method, family, tool = "none", "none", None
        elif kind == "transcode":
            method, family, tool = PATHS[rec["path"]]["desc"], f"transcode_{rec['path']}", "ffmpeg 4.4.2 (libx264, zscale)"
        elif kind == "edit":
            method, family, tool = rec["method"], "non_ai_edit_" + c["key"].removeprefix("edit_"), "ffmpeg 4.4.2"
        else:
            line = tts_meta["lines"][rec["tts"]]
            method = (f"speech replaced by Kokoro TTS voice {line['voice']}: \"{line['text']}\", "
                      "padded with silence to the video length")
            family, tool = "tts_kokoro_82m", f"Kokoro-82M {KOKORO_REV[:8]} (Apache-2.0), kokoro {tts_meta['kokoro_version']}"
        roots = rec["roots"]
        g = {
            "truth": "genuine" if label == "GENUINE" else "manipulated", "label": label, "stratum": stratum,
            "anchor": False, "source_class": "OWNER_KNOWN" if kind == "original" else "CONTROLLED_TEST",
            "manipulation": {"family": family, "gt_family": derive_manipulation_family(label), "method": method,
                             "tool": tool, "encode_path": "camera original" if kind == "original" else rec["path"],
                             "encode_path_desc": None if kind == "original" else PATHS[rec["path"]]["desc"],
                             "tts_sha256": tts_meta["lines"][rec["tts"]]["sha256"] if kind == "tts" else None},
            "provenance": {"judgement": ("camera original, owner-attested (first person, 2026-10-07)" if kind == "original"
                                         else "made by the project from owner camera originals under owner consent"),
                           "root_notes": {r: ROOTS[r][1] for r in roots},
                           "root_sha256": {r: ROOTS[r][0] for r in roots}},
            "license": {"license_type": "owner consent R16 (internal forensic evaluation; derivatives; local synthetic manipulation)",
                        "commercial_internal_eval_allowed": True, "redistribution_allowed": False,
                        "consent_ref": OWNER_CONSENT_REF,
                        "tool_licenses": None if kind in ("original",) else
                        ({"ffmpeg": "LGPL/GPL (generation side only)"} |
                         ({"kokoro_82m": "Apache-2.0"} if kind == "tts" else {}))},
            "lineage": {"root_sources": [f"owner:{r}" for r in roots], "owner_lineage": OWNER_LINEAGE,
                        "subject_overlap_r15": True},
            "freshness": {"exact_hash_overlap": sorted(set(hits)), "db_overlap": c["sha256"] in db,
                          "source_url_overlap": False, "lineage_overlap_r7_r15": False,
                          "family_overlap_r7_r15": kind == "transcode",
                          "family_overlap_note": {"original": "n/a (genuine)",
                                                  "transcode": "re-encode/resize derivations exist in R7 derive.py and M3",
                                                  "edit": "non-AI edit families are absent from R7-R15 corpora",
                                                  "tts": "Kokoro absent from R7-R15 benchmark corpora and detector training; used as R15-T9 spike stimuli only"}[kind]},
        }
        g["case_id"], g["path"], g["sha256"], g["bytes"] = cid, rel, c["sha256"], c["bytes"]
        g["build_key"] = c["key"]
        g["scoring_included"] = g["stratum"] != "unknown_truth"
        g["expected_interpretation"] = EXPECTED[g["stratum"]]
        gt_cases.append(g)

    # Lineage clusters: connected components over owner roots (a splice joins two roots).
    parent = {r: r for r in ROOTS}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for g in gt_cases:
        rs = [s.removeprefix("owner:") for s in g["lineage"]["root_sources"] if s.startswith("owner:")]
        for a in rs[1:]:
            parent[find(a)] = find(rs[0])
    for g in gt_cases:
        rs = [s.removeprefix("owner:") for s in g["lineage"]["root_sources"] if s.startswith("owner:")]
        if rs:
            g["lineage"]["lineage_cluster"] = "owner:" + "+".join(sorted(r for r in ROOTS if find(r) == find(rs[0])))

    sidecar = {"schema_version": "r16-t6-research-only-reference-1",
               "status": "NOT part of the R16-T7 pilot: not sealed with it, not scored, not uploaded in T7",
               "reason": "per-reference exclusion_reason (Architect 2026-10-07)",
               "references": sorted(research_only, key=lambda r: r["sha256"])}
    (work.parent / "research_only_reference.json").write_bytes(canonical(sidecar))
    exec_manifest = {"schema_version": EXEC_SCHEMA, "cases": exec_cases}
    gt = {
        "schema_version": GT_SCHEMA,
        "sealed_until": "R16-T7 scoring is complete; the T7 runner must not read this file",
        "claim_scope": "Controlled blind product pilot on a single-subject, single-session-derived population; not a population-generalization benchmark.",
        "scoring_rules": {
            "excluded_from_tp_tn_fp_fn": ["unknown_truth"],
            "positive_class": "manipulated", "negative_class": "genuine",
            "unit_of_independence": "lineage_cluster: report per-cluster as well as per-case; never treat cases of one cluster as independent samples",
            "anchors": "none in this set (all moved to research_only_reference)",
        },
        "strata_definitions": {s: EXPECTED.get(s, "Truth genuinely unknowable; excluded from TP/TN/FP/FN. Architect ruling: 0 cases.")
                               for s in STRATA},
        "limitations": LIMITATIONS,
        "encode_paths": {k: v["desc"] for k, v in PATHS.items()},
        "tts": tts_meta,
        "freshness_method": {
            "exact_hash": "every file under " + ", ".join(args.root) + " (pilot dirs excluded), size-matched then SHA-256",
            "database": "media_files original+derivative, ground_truth, media_governance (READ ONLY)" if args.db_container else "not checked",
            "source_url": "owner-supplied files; no URL exists for any case",
            "lineage": "owner roots: R16 intake dHash lineage check (0 foreign; IMG_1025~t2v_allegro_000 = heuristic false positive, Architect 2026-10-07)",
            "family": "per-case family vs R7-R15 corpus families",
        },
        "cases": sorted(gt_cases, key=lambda g: g["case_id"]),
    }
    (out / "pilot_manifest.json").write_bytes(canonical(exec_manifest))
    gt_path = out / "sealed" / "pilot_ground_truth.json"
    if gt_path.exists():
        gt_path.chmod(0o644)
    gt_path.write_bytes(canonical(gt))
    gt_path.chmod(0o444)
    os.utime(out / "pilot_manifest.json", (FIXED_MTIME, FIXED_MTIME))
    print(f"cases {len(exec_cases)}; db hashes compared {len(db)}")


# --------------------------------------------------------------------------------------- seal

def freshness_summary(gt):
    fresh = [c for c in gt["cases"] if not c["anchor"]]
    anchors = [c for c in gt["cases"] if c["anchor"]]
    foreign = lambda c: [h for h in c["freshness"]["exact_hash_overlap"] if h != "owner_source_folder"]  # noqa: E731
    return {
        "fresh_cases": len(fresh), "anchor_cases": len(anchors),
        "exact_hash_overlap_r7_r15": {"fresh": sum(1 for c in fresh if foreign(c)), "anchors": sum(1 for c in anchors if foreign(c))},
        "db_overlap": {"fresh": sum(c["freshness"]["db_overlap"] for c in fresh), "anchors": sum(c["freshness"]["db_overlap"] for c in anchors)},
        "source_url_overlap": {"fresh": sum(c["freshness"]["source_url_overlap"] for c in fresh),
                               "anchors": sum(c["freshness"]["source_url_overlap"] for c in anchors)},
        "lineage_overlap_r7_r15": {"fresh_source_recording": sum(c["freshness"]["lineage_overlap_r7_r15"] for c in fresh),
                                   "fresh_subject_overlap_r15_t2": sum(c["lineage"]["subject_overlap_r15"] for c in fresh),
                                   "anchors": sum(c["freshness"]["lineage_overlap_r7_r15"] for c in anchors)},
        "family_overlap_r7_r15": {"fresh": sum(c["freshness"]["family_overlap_r7_r15"] for c in fresh),
                                  "anchors": sum(c["freshness"]["family_overlap_r7_r15"] for c in anchors)},
        "byte_copies_of_owner_originals": sum(1 for c in fresh if "owner_source_folder" in c["freshness"]["exact_hash_overlap"]),
        "fresh_ai_visual_cases": 0,
    }


def compute_seal(out):
    exec_b = (out / "pilot_manifest.json").read_bytes()
    gt_b = (out / "sealed" / "pilot_ground_truth.json").read_bytes()
    ex, gt = json.loads(exec_b), json.loads(gt_b)
    per_file = {}
    for c in ex["cases"]:
        h = sha256(out / c["path"])
        if h != c["sha256"]:
            sys.exit(f"{c['case_id']}: file hash differs from manifest")
        per_file[c["case_id"]] = h
    strata = collections.Counter(c["stratum"] for c in gt["cases"])
    return {
        "schema_version": SEAL_SCHEMA,
        "execution_manifest_sha256": hashlib.sha256(exec_b).hexdigest(),
        "gt_manifest_sha256": hashlib.sha256(gt_b).hexdigest(),
        "ordered_case_ids": [c["case_id"] for c in ex["cases"]],
        "per_file_sha256": per_file,
        "case_count": len(ex["cases"]),
        "modality_counts": dict(sorted(collections.Counter(c["modality"] for c in ex["cases"]).items())),
        "stratum_counts": {s: strata.get(s, 0) for s in STRATA},
        "anchor_count": sum(1 for c in gt["cases"] if c["anchor"]),
        "freshness_summary": freshness_summary(gt),
    }


def cmd_seal(args):
    out = Path(args.out).expanduser()
    seal = compute_seal(out)
    (out / "SEAL.json").write_bytes(canonical(seal))
    print(canonical(seal).decode())


# ------------------------------------------------------------------------------------- verify

LEAK_TOKENS = ("genuine", "real", "fake", "manip", "swap", "tts", "edit", "anchor", "orig", "deriv", "img_",
               "splice", "synthetic", "owner", "label", "truth", "r7t", "vcd", "sonic", "gen3")


def cmd_verify(args):
    out = Path(args.out).expanduser()
    lines, fails = [], []
    ex = json.loads((out / "pilot_manifest.json").read_bytes())
    gt = json.loads((out / "sealed" / "pilot_ground_truth.json").read_bytes())
    seal = json.loads((out / "SEAL.json").read_bytes())
    if canonical(compute_seal(out)) != canonical(seal):
        fails.append("seal does not reproduce from the files on disk")
    lines.append("seal reproduces byte-identically: " + ("no" if fails else "yes"))

    keys = {k for c in ex["cases"] for k in c}
    if keys != {"case_id", "path", "sha256", "modality"} or set(ex) != {"schema_version", "cases"}:
        fails.append(f"execution manifest carries extra fields: {sorted(keys)} {sorted(ex)}")
    blob = json.dumps(ex).lower()
    found = [t for t in LEAK_TOKENS if t in blob.replace("schema_version", "").replace(EXEC_SCHEMA, "")]
    if found:
        fails.append(f"label tokens in execution manifest: {found}")
    lines.append(f"execution manifest fields: {sorted(keys)}; label tokens found: {found or 'none'}")

    names = sorted(os.listdir(out / "media"))
    if names != sorted(os.path.basename(c["path"]) for c in ex["cases"]):
        fails.append("media dir does not equal the manifest list")
    links = [n for n in names if os.path.islink(out / "media" / n)]
    mtimes = {int(os.path.getmtime(out / "media" / n)) for n in names}
    if links or mtimes != {FIXED_MTIME}:
        fails.append(f"symlinks {links} / mtimes {mtimes}")
    lines.append(f"media files {len(names)}; symlinks {len(links)}; distinct mtimes {len(mtimes)}")
    order = [c["case_id"] for c in ex["cases"]]
    if [c["sha256"] for c in ex["cases"]] != sorted(c["sha256"] for c in ex["cases"]) or order != sorted(order):
        fails.append("case ids are not in SHA-256 order")
    lines.append("case ids assigned in SHA-256 order: yes")

    truth = {c["case_id"]: c["truth"] for c in gt["cases"]}
    by_ext = collections.Counter((os.path.splitext(c["path"])[1], truth[c["case_id"]]) for c in ex["cases"])
    lines.append("extension x truth (the only label-adjacent field a runner can see): "
                 + ", ".join(f"{e}/{t}={n}" for (e, t), n in sorted(by_ext.items())))
    n = len(ex["cases"])
    majority = max(collections.Counter(truth.values()).values())
    best_ext = sum(max(by_ext.get((e, t), 0) for t in ("genuine", "manipulated")) for e in {e for e, _ in by_ext})
    lines.append(f"single-field guess accuracy from extension {best_ext}/{n} vs majority baseline {majority}/{n}")
    half = n // 2
    first = collections.Counter(truth[c] for c in order[:half])
    lines.append(f"truth in first half of id order: {dict(first)}; second half: "
                 f"{dict(collections.Counter(truth[c] for c in order[half:]))}")
    if any(c["stratum"] == "unknown_truth" and c["scoring_included"] for c in gt["cases"]):
        fails.append("unknown_truth case included in scoring")
    lines.append("unknown_truth cases: " + str(sum(c["stratum"] == "unknown_truth" for c in gt["cases"])))
    unclear = [c["case_id"] for c in gt["cases"] if c["license"]["commercial_internal_eval_allowed"] is not True]
    if unclear:
        fails.append(f"cases without explicit internal-evaluation rights: {unclear}")
    lines.append(f"cases without explicit internal-evaluation rights: {len(unclear)}")
    lines.append("result: " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
    report = "\n".join(lines) + "\n"
    (out.parent / "r16_pilot_raw" / "verify.txt").write_text(report)
    print(report)
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tts")
    t.add_argument("--work", required=True)
    b = sub.add_parser("build")
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--root", action="append", default=[])
    b.add_argument("--db-container")
    b.add_argument("--force", action="store_true")
    b.add_argument("--rebuild", action="store_true", help="re-encode cases already built in --work")
    for name in ("seal", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"tts": cmd_tts, "build": cmd_build, "seal": cmd_seal, "verify": cmd_verify}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
