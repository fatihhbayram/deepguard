"""R18-T4: characterize the forensic detectors on the frozen DEV partition, before any calibration.

    # 1. score the 155 DEV clips (long: LipForensics is ~2 min per clip on CPU; resumable)
    PYTHONPATH=scripts setsid nohup python3 scripts/eval/r18_t4_characterize.py score \
        --work ~/deepguard-r18-t4-work > ~/deepguard-r18-t4-work/score.log 2>&1 &
    # 2. metrics (pure function of the scores file)
    PYTHONPATH=scripts python3 scripts/eval/r18_t4_characterize.py analyze
    # 3. data sections of the report
    PYTHONPATH=scripts python3 scripts/eval/r18_t4_characterize.py report --out <path>

Applies `r18_t4_implementation_plan.md` (rev 2) and decides nothing beyond it.

**DEV only.** Cases come from `load_dev_partition`, which refuses any split but DEV before the
freeze is opened, and then reads DEV through R18-T3's `load_partition` (which verifies the whole
freeze digest and returns only DEV records). VAL and LOCKED_TEST records are never returned,
scored or summarised here.

**Production is read, never run or changed.** The anchors (ruleset version, thresholds, provider
versions) are parsed out of `git show HEAD:apps/api/app/risk_engine.py` with `ast`; nothing is
imported from `apps/*` on the host and no service, container or database is touched for them.
The detectors are the committed `app.*` functions, unchanged, inside ephemeral
`docker compose run --rm --no-deps api-worker` containers with the corpus mounted read-only: no
running service is started, restarted or modified, and the containers are removed on exit. Each
container first reports the digests of the modules it runs, and a mismatch with HEAD stops the
run. A clip is prepared exactly as the worker prepares it (probe, then the P1 transcode when
`needs_normalization` says so) into the container's own temp space.

**Availability is not discrimination.** A provider timeout, RESOURCE_EXHAUSTED or transport
failure is a TIMEOUT (coverage), an abstention (no face, no tracked mouth) is ABSTAINED, and
neither is ever a score. Every metric is computed on SUCCESS scores only. Successful NVIDIA
replies are cached add-only by (clip sha256, provider version) in `r18_t4_svd_cache.json`.

**No calibration.** Operating points are read off the empirical step function (comparator
`score >= threshold`); the interpolated ROC TPR is a straight line between empirical ROC points.
No isotonic, Platt or any other fitted map exists in this file.
"""

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wasserstein_distance

from eval import r18_t3_freeze_split as t3

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "scripts/eval/specs/r18_t3_freeze.json"
BASELINES = REPO / "scripts/eval/baselines"
SCORES = BASELINES / "r18_t4_detector_scores.json"
METRICS = BASELINES / "r18_t4_characterization_metrics.json"
SVD_CACHE = BASELINES / "r18_t4_svd_cache.json"
RISK_ENGINE_REL = "apps/api/app/risk_engine.py"
CORPUS = Path.home() / "deepguard-corpus"
AUTHORITY = "r18_t4_implementation_plan.md (rev 2)"
PURPOSE = "R18-T4"

SCHEMA_SCORES = "r18-t4-detector-scores-1"
SCHEMA_METRICS = "r18-t4-characterization-metrics-1"
SCHEMA_CACHE = "r18-t4-svd-cache-1"

SEED = 20261021
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
TARGET_FPRS = (0.0, 0.01, 0.05)

SVD, FACE, LIP, EFFORT = "synthetic_video", "face_manipulation", "lip_forensics", "effort"
DETECTORS = (SVD, FACE, LIP, EFFORT)
COMPARED = (SVD, FACE, LIP)  # task: paired comparison across SVD, B7 and LipForensics
# Which production constants anchor each detector; Effort has none (informational only).
THRESHOLD_ANCHOR = {SVD: "SVD_T_HIGH", FACE: "FACE_T_HIGH", LIP: "LIP_T_HIGH", EFFORT: None}
VERSION_ANCHOR = {SVD: "SVD_PROVIDER_VERSION", FACE: "FACE_PROVIDER_VERSION",
                  LIP: "LIP_PROVIDER_VERSION", EFFORT: None}
ANCHOR_NAMES = ("RULES_VERSION_V5", "SVD_T_HIGH", "FACE_T_HIGH", "LIP_T_HIGH",
                "SVD_PROVIDER_VERSION", "FACE_PROVIDER_VERSION", "LIP_PROVIDER_VERSION")

SUCCESS, ABSTAINED, FAILED, TIMEOUT = "SUCCESS", "ABSTAINED", "FAILED", "TIMEOUT"
UNAVAILABLE, NOT_ATTEMPTED = "UNAVAILABLE", "NOT_ATTEMPTED"
STATUSES = (SUCCESS, ABSTAINED, FAILED, TIMEOUT, UNAVAILABLE, NOT_ATTEMPTED)

AI, REAL = t3.VERIFIED_AI, t3.VERIFIED_REAL

# Plan §5.5. Descriptive, per case, never causal.
STATE_LOW = "SCORED_LOW_RELATIVE_TO_REAL"
STATE_ABOVE = "SCORED_ABOVE_REAL_BASELINE_BUT_BELOW_PRODUCTION_THRESHOLD"
STATE_UNSCORED = "UNSCORED_COVERAGE_FAILURE"
# Plan §5.5 leaves (median(REAL), p90(REAL)] unnamed; it is reported as its own band rather than
# folded into either neighbour, and flagged in the report for the Architect.
STATE_MID = "SCORED_WITHIN_REAL_RANGE_ABOVE_MEDIAN"
# Not a failure state: the AI case already reaches the production threshold.
STATE_FLAGGED = "FLAGGED_AT_PRODUCTION_THRESHOLD"
CASE_STATES = (STATE_FLAGGED, STATE_ABOVE, STATE_MID, STATE_LOW, STATE_UNSCORED)
REAL_BASELINE = "p90(REAL)"

# Scoring execution (plan §7.1).
SERVICE = "api-worker"
CONTAINER_CORPUS = "/r18t4_corpus"
SVD_PACE_S = 2.0
SVD_ATTEMPTS = 3
SVD_BACKOFF_S = 10.0
CONTAINER_MODULES = ("nvidia_video", "face_detector", "lip_forensics", "effort",
                     "normalization", "media", "limits", "risk_engine", "config")


class Refusal(SystemExit):
    """A stop condition: nothing is written."""


class PartitionViolationError(Exception):
    """T4 tooling asked for a split other than DEV."""


# --- partition and anchors --------------------------------------------------------------------

def load_dev_partition(freeze_path=FREEZE, split=t3.DEV):
    """DEV records of an intact freeze. Anything else is refused before the file is opened."""
    if split != t3.DEV:
        raise PartitionViolationError(
            f"R18-T4 may load only DEV; {split!r} is reserved (VAL -> R18-T5, "
            "LOCKED_TEST -> R18-T6)")
    records = t3.load_partition(freeze_path, t3.DEV, purpose=PURPOSE)
    if any(r["split"] != t3.DEV for r in records):
        raise PartitionViolationError("load_partition returned a non-DEV record")
    return records


def freeze_identity(freeze_path=FREEZE):
    freeze = json.loads(Path(freeze_path).read_text())
    return {"dataset_name": freeze["dataset_name"], "manifest_sha256": freeze["manifest_sha256"],
            "freeze_path": "scripts/eval/specs/r18_t3_freeze.json"}


def git(*args, repo=REPO):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def module_constants(source):
    """Literal module-level assignments of a Python source, by name. Nothing is executed."""
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return out


def introspect_anchors(run=git):
    """Production anchors read from HEAD's risk_engine source; no import, service or DB."""
    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        raise Refusal("git rev-parse HEAD failed")
    shown = run("show", f"HEAD:{RISK_ENGINE_REL}")
    if shown.returncode != 0:
        raise Refusal(f"cannot read HEAD:{RISK_ENGINE_REL}")
    consts = module_constants(shown.stdout)
    missing = [n for n in ANCHOR_NAMES if n not in consts]
    if missing:
        raise Refusal(f"risk_engine at HEAD lacks {missing} (S3: no fallback constants)")
    for name in ("SVD_T_HIGH", "FACE_T_HIGH", "LIP_T_HIGH"):
        if not isinstance(consts[name], float):
            raise Refusal(f"{name} is not a float literal at HEAD")
    worktree = REPO / RISK_ENGINE_REL
    return {
        "commit_sha": head.stdout.strip(),
        "ruleset_version": consts["RULES_VERSION_V5"],
        "svd_t_high": consts["SVD_T_HIGH"],
        "face_t_high": consts["FACE_T_HIGH"],
        "lip_t_high": consts["LIP_T_HIGH"],
        "svd_provider_version": consts["SVD_PROVIDER_VERSION"],
        "face_provider_version": consts["FACE_PROVIDER_VERSION"],
        "lip_provider_version": consts["LIP_PROVIDER_VERSION"],
        "source": f"git show HEAD:{RISK_ENGINE_REL} (ast literal parse, not imported)",
        "risk_engine_head_sha256": hashlib.sha256(shown.stdout.encode()).hexdigest(),
        "risk_engine_worktree_matches_head": (worktree.is_file()
                                              and worktree.read_text() == shown.stdout),
    }


def threshold_of(anchors, detector):
    name = THRESHOLD_ANCHOR[detector]
    return None if name is None else anchors[name.lower()]


def version_of(anchors, detector):
    name = VERSION_ANCHOR[detector]
    return None if name is None else anchors[name.lower()]


# --- io helpers -------------------------------------------------------------------------------

def dumps(payload):
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dumps(payload))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- svd cache --------------------------------------------------------------------------------

def cache_key(sha256, provider_version):
    return f"{sha256}|{provider_version}"


def load_cache(path=SVD_CACHE):
    if not Path(path).exists():
        return {"schema": SCHEMA_CACHE,
                "key_rule": "<clip sha256>|<NVIDIA function id that answered>",
                "policy": "add-only: an existing key is never overwritten; a differing reply "
                          "for an existing key is a refusal",
                "entries": {}}
    cache = json.loads(Path(path).read_text())
    if cache.get("schema") != SCHEMA_CACHE:
        raise Refusal(f"{path} is not an {SCHEMA_CACHE} file")
    return cache


def cache_add(cache, sha256, record):
    """Add one successful reply; immutable once written."""
    if record.get("status") != SUCCESS:
        raise ValueError("only SUCCESS replies are cached")
    key = cache_key(sha256, record["provider_version"])
    entry = {"score": record["score"], "provider_version": record["provider_version"],
             "evidence": record.get("evidence"), "artifact": record.get("artifact"),
             "cached_at": record.get("recorded_at")}
    held = cache["entries"].get(key)
    if held is not None:
        if (held["score"], held["evidence"]) != (entry["score"], entry["evidence"]):
            raise Refusal(f"cache key {key} already holds a different reply (immutable)")
        return False
    cache["entries"][key] = entry
    return True


# --- scoring: the container runner ------------------------------------------------------------

RUNNER = r'''
import asyncio, hashlib, json, os, sys, time
sys.path.insert(0, "/app")
detector, pace, attempts, backoff = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
jobs = json.loads(sys.argv[5])
modules = json.loads(sys.argv[6])

def emit(o):
    print(json.dumps(o, sort_keys=True), flush=True)

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

pre = {"kind": "preflight", "detector": detector,
       "module_sha256": {m: digest(f"/app/app/{m}.py") for m in modules},
       "svd_function_id": os.environ.get("NVIDIA_SVD_FUNCTION_ID")}
try:
    import torch
    pre["torch_version"], pre["torch_threads"] = torch.__version__, torch.get_num_threads()
except Exception as e:
    pre["torch_error"] = type(e).__name__
emit(pre)

from app.media import probe_media
from app.normalization import needs_normalization, normalize_to_mp4

def err(e):
    return {"error_class": type(e).__name__, "error_mro": [c.__name__ for c in type(e).__mro__],
            "error_detail": str(e)[:400]}

def prepare(path):
    meta = asyncio.run(probe_media(path))
    art = {"source_codec": meta.codec_name, "source_frame_rate": meta.frame_rate,
           "source_duration": meta.duration, "source_constant_frame_rate": meta.constant_frame_rate,
           "normalized": needs_normalization(meta)}
    if not art["normalized"]:
        return path, art, None
    d = asyncio.run(normalize_to_mp4(path, meta.frame_rate))
    art["derivative_sha256"], art["derivative_width"], art["derivative_height"] = d.sha256, d.width, d.height
    return d.path, art, d.path

def run_svd(p):
    from app import nvidia_video as nv
    tried = []
    for a in range(attempts):
        if a:
            time.sleep(backoff * 2 ** (a - 1))
        try:
            r = asyncio.run(nv.analyze_video(p))
            return {"status": "SUCCESS", "score": r.probability, "provider_version": r.function_id,
                    "evidence": {"logit": r.logit, "probability": r.probability,
                                 "total_clips": r.total_clips,
                                 "clip_logits": [c.logit for c in r.clips]}, "attempts": tried + ["SUCCESS"]}
        except (nv.NvidiaProviderTimeout, nv.NvidiaProviderUnavailable) as e:
            tried.append(type(e).__name__)
            last = {"status": "TIMEOUT", "availability_event": True, **err(e)}
        except nv.NvidiaVideoError as e:
            return {"status": "FAILED", "availability_event": False, "attempts": tried + [type(e).__name__], **err(e)}
    return {**last, "attempts": tried}

def run_face(p):
    from app import face_detector as fd
    try:
        ev = fd.analyze_face_manipulation(p)
    except fd.FaceDetectorNoFaceFound as e:
        return {"status": "ABSTAINED", **err(e)}
    except fd.FaceDetectorError as e:
        return {"status": "FAILED", **err(e)}
    return {"status": "SUCCESS", "score": ev.score,
            "provider_version": f"{ev.classifier_repository}@{ev.classifier_revision}",
            "evidence": {"frames_requested": ev.frames_requested, "frames_decoded": ev.frames_decoded,
                         "frames_scored": ev.frames_scored,
                         "frame_probabilities": [f.probability for f in ev.frame_scores]}}

def run_lip(p):
    from app import lip_forensics as lf
    try:
        ev = lf.analyze_lip_forensics(p)
    except lf.LipForensicsNoTrackedFace as e:
        return {"status": "ABSTAINED", **err(e)}
    except lf.LipForensicsError as e:
        return {"status": "FAILED", **err(e)}
    return {"status": "SUCCESS", "score": ev.score,
            "provider_version": f"{ev.upstream_repository}@{ev.upstream_revision}+{ev.weights_sha256}",
            "evidence": {"windows_requested": ev.windows_requested, "windows_read": ev.windows_read,
                         "windows_scored": ev.windows_scored, "device": ev.device,
                         "window_logits": [w.logit for w in ev.window_logits]}}

def run_effort(p):
    try:
        from app import effort as ef
    except Exception as e:
        return {"status": "UNAVAILABLE", **err(e)}
    try:
        ev = ef.analyze_effort(p)
    except ef.EffortNoFaceDetected as e:
        return {"status": "ABSTAINED", **err(e)}
    except ef.EffortModelUnavailable as e:
        return {"status": "UNAVAILABLE", **err(e)}
    except ef.EffortError as e:
        return {"status": "FAILED", **err(e)}
    return {"status": "SUCCESS", "score": ev.score,
            "provider_version": f"{ev.upstream_repository}@{ev.upstream_revision}+{ev.checkpoint_sha256}",
            "evidence": {"frames_requested": ev.frames_requested, "frames_decoded": ev.frames_decoded,
                         "frames_with_face": ev.frames_with_face,
                         "frame_scores": list(ev.frame_scores)}}

RUN = {"synthetic_video": run_svd, "face_manipulation": run_face, "lip_forensics": run_lip,
       "effort": run_effort}[detector]

for i, job in enumerate(jobs):
    if detector == "synthetic_video" and i:
        time.sleep(pace)
    t0 = time.monotonic()
    tmp = None
    try:
        path, art, tmp = prepare(job["path"])
    except Exception as e:
        emit({"kind": "result", "sha256": job["sha256"], "status": "FAILED",
              "stage": "prepare", "elapsed_s": round(time.monotonic() - t0, 3), **err(e)})
        continue
    try:
        out = RUN(path)
    except Exception as e:
        out = {"status": "FAILED", "stage": "detector", **err(e)}
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    emit({"kind": "result", "sha256": job["sha256"], "artifact": art,
          "elapsed_s": round(time.monotonic() - t0, 3), **out})
'''


def container_cmd(detector, jobs, corpus, pace=SVD_PACE_S, backoff=SVD_BACKOFF_S, name=None):
    named = ["--name", name] if name else []
    return ["docker", "compose", "run", "--rm", "--no-deps", "-T", *named,
            "-v", f"{Path(corpus).resolve()}:{CONTAINER_CORPUS}:ro",
            "--entrypoint", "python", SERVICE, "-I", "-c", RUNNER, detector, str(pace),
            str(SVD_ATTEMPTS), str(backoff), json.dumps(jobs),
            json.dumps(list(CONTAINER_MODULES))]


def host_module_digests():
    return {m: sha256_file(REPO / "apps/api/app" / f"{m}.py") for m in CONTAINER_MODULES}


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def latest_results(work, detector):
    """Latest result per sha256 across every shard log of one detector."""
    out = {}
    for log in sorted(Path(work).glob(f"{detector}*.jsonl")):
        if log.name.endswith(".preflight.jsonl"):
            continue
        for rec in read_jsonl(log):
            out[rec["sha256"]] = rec
    return out


def pending_jobs(cases, detector, work, cache, anchors, corpus_mount=CONTAINER_CORPUS):
    done = latest_results(work, detector)
    jobs = []
    for c in cases:
        if detector == SVD and cache_key(c["sha256"], anchors["svd_provider_version"]) in \
                cache["entries"]:
            continue
        prev = done.get(c["sha256"])
        if prev is not None and prev["status"] not in (TIMEOUT, UNAVAILABLE):
            continue  # deterministic outcomes are final; availability events are retried
        jobs.append({"sha256": c["sha256"], "path": f"{corpus_mount}/{c['canonical_path']}"})
    return jobs


def run_container(detector, jobs, work, corpus, tag, expected_digests, lock,
                  pace=SVD_PACE_S, backoff=SVD_BACKOFF_S, breaker=0):
    """Stream one container's results into <work>/<tag>.jsonl; stop on a digest mismatch.

    `breaker` > 0 stops the container after that many consecutive availability events (a
    provider quota that outlasts the backoff); the remaining clips stay pending for a later
    round. It changes when a clip is asked, never what its outcome is recorded as.
    """
    log = Path(work) / f"{tag}.jsonl"
    pre_log = Path(work) / f"{tag}.preflight.jsonl"
    with lock, open(Path(work) / "runs.jsonl", "a") as f:
        f.write(json.dumps({"tag": tag, "detector": detector, "jobs": len(jobs),
                            "svd_pace_s": pace, "svd_attempts": SVD_ATTEMPTS,
                            "svd_backoff_s": backoff, "svd_breaker": breaker,
                            "started_at": now()},
                           sort_keys=True) + "\n")
    # Named, so it can be stopped: killing the compose client alone leaves the container
    # running headless, still calling the provider, with its output going nowhere.
    name = f"r18t4-{tag.replace('.', '-')}-{int(time.time())}"
    proc = subprocess.Popen(container_cmd(detector, jobs, corpus, pace, backoff, name), cwd=REPO,
                            text=True,
                            stdout=subprocess.PIPE, stderr=open(Path(work) / f"{tag}.stderr",
                                                                "a"))
    n = streak = 0
    for line in proc.stdout:
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        rec["recorded_at"] = now()
        if rec.get("kind") == "preflight":
            with open(pre_log, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
            if rec["module_sha256"] != expected_digests:
                stop_container(name)
                proc.kill()
                raise Refusal(f"{tag}: container modules differ from HEAD; refusing")
            continue
        with lock:
            with open(log, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        n += 1
        print(f"[{now()}] {tag} {n}/{len(jobs)} {rec['sha256'][:12]} {rec['status']} "
              f"{rec.get('score')} {rec.get('error_class', '')}", flush=True)
        streak = streak + 1 if rec["status"] == TIMEOUT else 0
        if breaker and streak >= breaker:
            print(f"[{now()}] {tag} circuit breaker: {streak} consecutive availability "
                  "events, stopping this round", flush=True)
            stop_container(name)
            proc.kill()
            break
    proc.wait()
    print(f"[{now()}] {tag} container exit {proc.returncode}", flush=True)


def stop_container(name):
    subprocess.run(["docker", "stop", name], capture_output=True)


def verify_clips(cases, corpus):
    bad = []
    for c in cases:
        p = Path(corpus) / c["canonical_path"]
        if not p.is_file() or sha256_file(p) != c["sha256"]:
            bad.append(c["canonical_path"])
    if bad:
        raise Refusal(f"{len(bad)} DEV clip(s) missing or not their frozen bytes: {bad[:5]}")


def shard(items, n):
    return [items[i::n] for i in range(n)] if n > 1 else [items]


def case_record(c):
    keep = ("sha256", "canonical_path", "ground_truth", "generator_family", "generator_model",
            "capture_domain", "face_present", "speech_talking_head", "source_cluster_id",
            "source_collection_id")
    return {k: c[k] for k in keep}


def detector_entry(detector, rec, cached, anchors):
    """The scores-file entry for one case and detector, from the cache or the latest log line."""
    if cached is not None:
        return {"status": SUCCESS, "score": cached["score"],
                "provider_version": cached["provider_version"], "evidence": cached["evidence"],
                "artifact": cached["artifact"], "source": "svd_cache"}
    if rec is None:
        return {"status": NOT_ATTEMPTED, "score": None}
    entry = {k: rec.get(k) for k in ("status", "score", "provider_version", "evidence",
                                     "artifact", "error_class", "error_detail", "stage",
                                     "attempts", "availability_event", "elapsed_s",
                                     "recorded_at") if rec.get(k) is not None}
    entry.setdefault("score", None)
    expected = version_of(anchors, detector)
    if entry["status"] == SUCCESS and expected is not None and \
            entry.get("provider_version") != expected:
        entry.update(status=FAILED, score=None, error_class="ProviderVersionMismatch",
                     error_detail=f"answered by {entry.get('provider_version')}, "
                                  f"production binds {expected}")
    if entry["status"] != SUCCESS:
        entry["score"] = None  # never a score for anything but SUCCESS
    return entry


def assemble(cases, work, cache, anchors, preflight):
    svd_v = anchors["svd_provider_version"]
    logs = {d: latest_results(work, d) for d in DETECTORS}
    out = []
    for c in sorted(cases, key=lambda c: c["sha256"]):
        dets = {}
        for d in DETECTORS:
            cached = cache["entries"].get(cache_key(c["sha256"], svd_v)) if d == SVD else None
            dets[d] = detector_entry(d, logs[d].get(c["sha256"]), cached, anchors)
        out.append({**case_record(c), "detectors": dets})
    return {"schema": SCHEMA_SCORES, "authority": AUTHORITY, "partition": t3.DEV,
            "freeze": freeze_identity(), "anchors": anchors,
            "execution": {
                "environment": f"docker compose run --rm --no-deps {SERVICE} (ephemeral, "
                               f"corpus mounted read-only at {CONTAINER_CORPUS})",
                "preparation": "app.media.probe_media, then app.normalization.normalize_to_mp4 "
                               "when needs_normalization (worker P1), container temp only",
                "runs": read_jsonl(Path(work) / "runs.jsonl"), "preflight": preflight},
            "status_vocabulary": list(STATUSES),
            "cases": out}


def cmd_score(args):
    anchors = introspect_anchors()
    cases = load_dev_partition(args.freeze)
    if args.limit:
        cases = sorted(cases, key=lambda c: c["sha256"])[:args.limit]
    work = Path(args.work).resolve()
    if REPO in work.parents or work == REPO or Path(args.corpus).resolve() in work.parents:
        raise Refusal("--work must lie outside the repository and the corpus")
    work.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] anchors {anchors['ruleset_version']} @ {anchors['commit_sha']}", flush=True)
    verify_clips(cases, args.corpus)
    cache = load_cache(args.cache)
    expected = host_module_digests()
    if not 1.5 <= args.svd_pace <= 2.0:
        raise Refusal("--svd-pace must lie in the plan's 1.5-2.0 s band")
    lock = threading.Lock()
    threads, errors = [], []
    detectors = [] if args.assemble_only else (args.detectors or list(DETECTORS))
    for d in detectors:
        jobs = pending_jobs(cases, d, work, cache, anchors)
        n = args.lip_shards if d == LIP else args.local_shards if d != SVD else 1
        for i, part in enumerate(shard(jobs, n)):
            if not part:
                continue
            tag = d if n == 1 else f"{d}.shard{i}"

            def target(d=d, part=part, tag=tag):
                try:
                    run_container(d, part, work, args.corpus, tag, expected, lock,
                                  args.svd_pace, args.svd_backoff,
                                  args.svd_breaker if d == SVD else 0)
                except BaseException as exc:  # noqa: BLE001 - reported, then the run stops
                    errors.append(f"{tag}: {exc}")
            print(f"[{now()}] {tag}: {len(part)} job(s)", flush=True)
            threads.append(threading.Thread(target=target, daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    added = 0
    for rec in latest_results(work, SVD).values():
        if rec["status"] == SUCCESS:
            added += cache_add(cache, rec["sha256"], rec)
    if added or not Path(args.cache).exists():
        write_json(args.cache, cache)
    preflight = {}
    for log in sorted(work.glob("*.preflight.jsonl")):
        for rec in read_jsonl(log):
            preflight[log.name.split(".")[0]] = {k: rec.get(k) for k in
                                                 ("module_sha256", "torch_version",
                                                  "torch_threads", "torch_error")}
    if not args.limit:
        write_json(args.out, assemble(cases, work, cache, anchors, preflight))
        print(f"[{now()}] wrote {args.out}", flush=True)
    if errors:
        raise Refusal("; ".join(errors))


# --- metrics ----------------------------------------------------------------------------------

def _f(x):
    return None if x is None else float(x)


def auroc(ai, real):
    """Mann-Whitney AUROC (ties count one half) = trapezoid area of the >= ROC; None if undefined."""
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    if not len(ai) or not len(real):
        return None
    pooled = np.concatenate([ai, real])
    if np.all(pooled == pooled[0]):
        return None
    ranks = rankdata(pooled)
    u = ranks[:len(ai)].sum() - len(ai) * (len(ai) + 1) / 2
    return float(u / (len(ai) * len(real)))


def roc_points(ai, real):
    """Empirical ROC under `score >= t`, from (0, 0) at t = nextafter(max, inf) to (1, 1)."""
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    pooled = np.concatenate([ai, real])
    top = float(np.nextafter(pooled.max(), np.inf))
    pts = [{"threshold": top, "fpr": 0.0, "tpr": 0.0}]
    for t in np.unique(pooled)[::-1]:
        pts.append({"threshold": float(t), "fpr": float((real >= t).mean()),
                    "tpr": float((ai >= t).mean())})
    return pts


def pr_points(ai, real):
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    pts = []
    for t in np.unique(np.concatenate([ai, real]))[::-1]:
        tp, fp = int((ai >= t).sum()), int((real >= t).sum())
        pts.append({"threshold": float(t), "recall": tp / len(ai), "precision": tp / (tp + fp)})
    return pts


def auprc(ai, real):
    """Average precision: sum over distinct thresholds of (R_k - R_k-1) * P_k; None if undefined."""
    if not len(ai) or not len(real):
        return None
    ap, prev = 0.0, 0.0
    for p in pr_points(ai, real):
        ap += (p["recall"] - prev) * p["precision"]
        prev = p["recall"]
    return float(ap)


def roc_upper_envelope(points):
    """(fpr, tpr) with one point per distinct FPR, the highest TPR reached there."""
    best = {}
    for p in points:
        best[p["fpr"]] = max(best.get(p["fpr"], 0.0), p["tpr"])
    xs = sorted(best)
    return np.array(xs), np.array([best[x] for x in xs])


def interpolated_roc_tpr(ai, real, target):
    """Linear interpolation between empirical ROC points at FPR = target. No fitted model."""
    xs, ys = roc_upper_envelope(roc_points(ai, real))
    return float(np.interp(target, xs, ys))


def allowed_false_positives(target, n_real):
    return int(math.floor(target * n_real + 1e-9))


def fixed_fpr_point(ai, real, target):
    """Plan §5.4: the lowest threshold whose empirical FPR is the largest achievable <= target.

    With k = floor(target * N) false alarms allowed, the threshold sits strictly above the
    (k+1)-th highest REAL score: `nextafter(sorted_desc[k], inf)`. At k = 0 that is
    `nextafter(max(REAL), inf)`, so a REAL clip scoring exactly max(REAL) is not flagged.
    """
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    n = len(real)
    k = allowed_false_positives(target, n)
    desc = np.sort(real)[::-1]
    if k >= n:
        threshold = float(min(desc[-1], ai.min() if len(ai) else desc[-1]))
        rule = "min(score): every clip flagged"
    else:
        threshold = float(np.nextafter(desc[k], np.inf))
        rule = (f"nextafter(sorted_desc(REAL)[{k}], inf)" if k else
                "nextafter(max(REAL), inf)")
    fp, tp = int((real >= threshold).sum()), int((ai >= threshold).sum())
    return {"target_fpr": target, "usable_real_denominator": n,
            "allowed_false_positives": k, "threshold": threshold, "threshold_rule": rule,
            "false_positives": fp, "achievable_empirical_fpr": fp / n,
            "true_positives": tp, "usable_ai_denominator": len(ai),
            "tpr_at_achievable_fpr": tp / len(ai) if len(ai) else None,
            "interpolated_roc_tpr": interpolated_roc_tpr(ai, real, target),
            "interpolation": "linear between empirical ROC points (upper envelope per FPR)"}


def fixed_threshold_point(ai, real, threshold):
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    tp, fp = int((ai >= threshold).sum()), int((real >= threshold).sum())
    return {"threshold": threshold, "comparator": "score >= threshold",
            "true_positives": tp, "usable_ai_denominator": len(ai),
            "empirical_tpr": tp / len(ai) if len(ai) else None,
            "false_positives": fp, "usable_real_denominator": len(real),
            "empirical_fpr": fp / len(real) if len(real) else None}


def describe(values):
    v = np.asarray(values, float)
    if not len(v):
        return {"count": 0}
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"count": int(len(v)), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else None,
            "min": float(v.min()), "p10": float(q[0]), "q1": float(q[1]),
            "median": float(q[2]), "q3": float(q[3]), "p90": float(q[4]),
            "max": float(v.max()), "iqr": float(q[3] - q[1])}


def cohens_d(ai, real):
    ai, real = np.asarray(ai, float), np.asarray(real, float)
    if len(ai) < 2 or len(real) < 2:
        return None
    pooled = ((len(ai) - 1) * ai.var(ddof=1) + (len(real) - 1) * real.var(ddof=1)) / \
        (len(ai) + len(real) - 2)
    if pooled <= 0:
        return None
    return float((ai.mean() - real.mean()) / math.sqrt(pooled))


def separation(ai, real):
    return {"ai": describe(ai), "real": describe(real), "cohens_d": cohens_d(ai, real),
            "wasserstein_distance": (float(wasserstein_distance(ai, real))
                                     if len(ai) and len(real) else None)}


# --- cluster bootstrap ------------------------------------------------------------------------

def cluster_index(labels, clusters):
    """{class: [(cluster, [row indices])...]} sorted, so resampling is order-independent."""
    groups = {1: defaultdict(list), 0: defaultdict(list)}
    for i, (y, c) in enumerate(zip(labels, clusters)):
        groups[int(y)][c].append(i)
    return {y: sorted(g.items()) for y, g in groups.items()}


def bootstrap(labels, clusters, statistic, n=N_BOOTSTRAP, seed=SEED):
    """Class-stratified source-cluster bootstrap (plan §5.3).

    Each replicate draws |C_AI| AI clusters from the AI clusters and |C_REAL| REAL clusters from
    the REAL clusters, with replacement, and keeps every row of each drawn cluster. `statistic`
    maps an index array to {name: value or None}; a replicate in which any value is None is
    invalid, counted, and left out of every interval.
    """
    groups = cluster_index(labels, clusters)
    rng = np.random.default_rng(seed)
    kept, invalid = defaultdict(list), 0
    for _ in range(n):
        idx = []
        for y in (1, 0):
            members = groups[y]
            if not members:
                continue
            for j in rng.integers(0, len(members), size=len(members)):
                idx.extend(members[j][1])
        values = statistic(np.asarray(idx, int))
        if any(v is None or not math.isfinite(v) for v in values.values()):
            invalid += 1
            continue
        for k, v in values.items():
            kept[k].append(v)
    lo, hi = 100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2
    ci = {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in
          sorted(kept.items())}
    return {"replicates": n, "seed": seed, "valid_replicate_count": n - invalid,
            "invalid_replicate_count": invalid, "ci_level": CI_LEVEL,
            "ci_method": "percentile [2.5, 97.5] over valid replicates",
            "resampling": "class-stratified source-cluster (whole clusters, with replacement)",
            "intervals": ci}


def discrimination(rows, detector):
    """AUROC/AUPRC with cluster-bootstrap CIs on one detector's usable rows."""
    ai = [r["score"] for r in rows if r["y"] == 1]
    real = [r["score"] for r in rows if r["y"] == 0]
    out = {"usable_ai": len(ai), "usable_real": len(real), "auroc": auroc(ai, real),
           "auprc": auprc(ai, real),
           "auprc_prevalence_baseline": len(ai) / (len(ai) + len(real)) if rows else None}
    if out["auroc"] is None:
        out["bootstrap"] = None
        return out
    scores = np.array([r["score"] for r in rows], float)
    labels = np.array([r["y"] for r in rows], int)

    def stat(idx):
        s, y = scores[idx], labels[idx]
        return {"auroc": auroc(s[y == 1], s[y == 0]), "auprc": auprc(s[y == 1], s[y == 0])}
    out["bootstrap"] = bootstrap(labels, [r["cluster"] for r in rows], stat)
    return out


# --- case states ------------------------------------------------------------------------------

def case_state(status, score, real_scores, production_threshold):
    """Descriptive state of one AI case (plan §5.5); never a causal label."""
    if status != SUCCESS or score is None:
        return STATE_UNSCORED
    if production_threshold is not None and score >= production_threshold:
        return STATE_FLAGGED
    real = np.asarray(real_scores, float)
    if score > np.percentile(real, 90):
        return STATE_ABOVE
    if score <= np.median(real):
        return STATE_LOW
    return STATE_MID


# --- analysis ---------------------------------------------------------------------------------

def usable_rows(cases, detector, keep=lambda c: True):
    rows = []
    for c in cases:
        e = c["detectors"][detector]
        if keep(c) and e["status"] == SUCCESS and e["score"] is not None:
            rows.append({"sha256": c["sha256"], "y": int(c["ground_truth"] == AI),
                         "score": float(e["score"]), "cluster": c["source_cluster_id"]})
    return rows


def census(cases, detector):
    counts = Counter(c["detectors"][detector]["status"] for c in cases)
    total_ai = sum(c["ground_truth"] == AI for c in cases)
    total_real = len(cases) - total_ai
    usable = usable_rows(cases, detector)
    reasons = Counter((c["detectors"][detector]["status"],
                       c["detectors"][detector].get("error_class"))
                      for c in cases if c["detectors"][detector]["status"] != SUCCESS)
    by_face = {}
    for fv in (True, False):
        sub = [c for c in cases if c["face_present"] is fv]
        u = usable_rows(sub, detector)
        by_face[str(fv).lower()] = {
            "total_ai": sum(c["ground_truth"] == AI for c in sub),
            "total_real": sum(c["ground_truth"] == REAL for c in sub),
            "usable_ai": sum(r["y"] == 1 for r in u), "usable_real": sum(r["y"] == 0 for r in u)}
    return {"total_cases": len(cases), "attempted": len(cases) - counts.get(NOT_ATTEMPTED, 0),
            **{s: counts.get(s, 0) for s in STATUSES},
            "usable_ai": sum(r["y"] == 1 for r in usable), "total_ai": total_ai,
            "usable_real": sum(r["y"] == 0 for r in usable), "total_real": total_real,
            "coverage_ai": sum(r["y"] == 1 for r in usable) / total_ai if total_ai else None,
            "coverage_real": sum(r["y"] == 0 for r in usable) / total_real if total_real else None,
            "non_success_reasons": [{"status": s, "error_class": e, "count": n}
                                    for (s, e), n in sorted(reasons.items(),
                                                            key=lambda kv: (kv[0][0],
                                                                            str(kv[0][1])))],
            "by_face_present": by_face}


def subgroup_ai(cases, detector, key, real_scores, prod_t, zero_fp_t):
    groups = defaultdict(list)
    for c in cases:
        if c["ground_truth"] == AI:
            groups[c[key]].append(c)
    out = {}
    for g, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scores = [c["detectors"][detector]["score"] for c in members
                  if c["detectors"][detector]["status"] == SUCCESS]
        states = Counter(case_state(c["detectors"][detector]["status"],
                                    c["detectors"][detector]["score"], real_scores, prod_t)
                         for c in members) if len(real_scores) else Counter()
        out[str(g)] = {
            "total": len(members), "usable": len(scores),
            "clusters": len({c["source_cluster_id"] for c in members}),
            "distribution": describe(scores),
            "auroc_vs_all_usable_real": auroc(scores, real_scores),
            "tpr_at_production_threshold": (sum(s >= prod_t for s in scores) / len(scores)
                                            if scores and prod_t is not None else None),
            "tpr_at_dev_zero_fp_threshold": (sum(s >= zero_fp_t for s in scores) / len(scores)
                                             if scores and zero_fp_t is not None else None),
            "case_states": {s: states.get(s, 0) for s in CASE_STATES}}
    return out


def subgroup_real(cases, detector, key, ai_scores, thresholds):
    groups = defaultdict(list)
    for c in cases:
        if c["ground_truth"] == REAL:
            groups[c[key]].append(c)
    out = {}
    for g, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scores = [c["detectors"][detector]["score"] for c in members
                  if c["detectors"][detector]["status"] == SUCCESS]
        out[str(g)] = {
            "total": len(members), "usable": len(scores),
            "clusters": len({c["source_cluster_id"] for c in members}),
            "distribution": describe(scores),
            "auroc_all_usable_ai_vs_this_domain": auroc(ai_scores, scores),
            "false_positives_at": {name: (sum(s >= t for s in scores) if t is not None else None)
                                   for name, t in thresholds.items()}}
    return out


def detector_metrics(cases, detector, anchors):
    rows = usable_rows(cases, detector)
    ai = [r["score"] for r in rows if r["y"] == 1]
    real = [r["score"] for r in rows if r["y"] == 0]
    prod_t = threshold_of(anchors, detector)
    out = {"census": census(cases, detector),
           "production_threshold_anchor": THRESHOLD_ANCHOR[detector],
           "production_threshold": prod_t}
    if not ai or not real:
        out["status"] = "INSUFFICIENT_USABLE_SCORES"
        return out
    out["distribution"] = separation(ai, real)
    out["discrimination"] = discrimination(rows, detector)
    out["roc_curve"] = roc_points(ai, real)
    out["pr_curve"] = pr_points(ai, real)
    ops = [fixed_fpr_point(ai, real, a) for a in TARGET_FPRS]
    out["fixed_fpr_operating_points"] = ops
    if prod_t is not None:
        p = fixed_threshold_point(ai, real, prod_t)
        scores = np.array([r["score"] for r in rows], float)
        labels = np.array([r["y"] for r in rows], int)

        def stat(idx):
            s, y = scores[idx], labels[idx]
            return {"tpr": float((s[y == 1] >= prod_t).mean()),
                    "fpr": float((s[y == 0] >= prod_t).mean())}
        p["bootstrap"] = bootstrap(labels, [r["cluster"] for r in rows], stat)
        out["production_operating_point"] = p
    zero_fp_t = ops[0]["threshold"]
    thresholds = {"production": prod_t, "dev_fpr_1pct": ops[1]["threshold"],
                  "dev_fpr_5pct": ops[2]["threshold"]}
    states = Counter(case_state(c["detectors"][detector]["status"],
                                c["detectors"][detector]["score"], real, prod_t)
                     for c in cases if c["ground_truth"] == AI)
    out["case_states_ai"] = {
        "baseline": REAL_BASELINE, "real_median": float(np.median(real)),
        "real_p90": float(np.percentile(real, 90)), "real_max": float(max(real)),
        "production_threshold": prod_t,
        "counts": {s: states.get(s, 0) for s in CASE_STATES},
        "ai_scored_above_real_max": sum(s > max(real) for s in ai)}
    out["subgroups"] = {
        "generator_family": subgroup_ai(cases, detector, "generator_family", real, prod_t,
                                        zero_fp_t),
        "ai_source_collection": subgroup_ai(cases, detector, "source_collection_id", real,
                                            prod_t, zero_fp_t),
        "real_capture_domain": subgroup_real(cases, detector, "capture_domain", ai, thresholds),
        "real_source_collection": subgroup_real(cases, detector, "source_collection_id", ai,
                                                thresholds),
        "face_present": {}}
    for fv in (True, False):
        sub = usable_rows(cases, detector, keep=lambda c, fv=fv: c["face_present"] is fv)
        s_ai = [r["score"] for r in sub if r["y"] == 1]
        s_real = [r["score"] for r in sub if r["y"] == 0]
        entry = {"usable_ai": len(s_ai), "usable_real": len(s_real),
                 "distribution": separation(s_ai, s_real) if s_ai and s_real else None}
        if s_ai and s_real:
            entry["discrimination"] = discrimination(sub, detector)
            entry["production_operating_point"] = (fixed_threshold_point(s_ai, s_real, prod_t)
                                                   if prod_t is not None else None)
        out["subgroups"]["face_present"][str(fv).lower()] = entry
    out["status"] = "CHARACTERIZED"
    return out


def paired_metrics(cases, detectors):
    """AUROC/AUPRC of several detectors on the cases all of them scored, with shared replicates."""
    common = [c for c in cases
              if all(c["detectors"][d]["status"] == SUCCESS for d in detectors)]
    n_ai = sum(c["ground_truth"] == AI for c in common)
    out = {"detectors": list(detectors), "cases": len(common), "usable_ai": n_ai,
           "usable_real": len(common) - n_ai,
           "clusters": len({c["source_cluster_id"] for c in common}),
           "face_present": sum(bool(c["face_present"]) for c in common)}
    if not n_ai or n_ai == len(common):
        out["status"] = "INSUFFICIENT_COMMON_SUCCESS"
        return out
    labels = np.array([int(c["ground_truth"] == AI) for c in common], int)
    S = {d: np.array([c["detectors"][d]["score"] for c in common], float) for d in detectors}
    out["auroc"] = {d: auroc(S[d][labels == 1], S[d][labels == 0]) for d in detectors}
    out["auprc"] = {d: auprc(S[d][labels == 1], S[d][labels == 0]) for d in detectors}

    def stat(idx):
        y = labels[idx]
        vals = {}
        for d in detectors:
            s = S[d][idx]
            vals[f"auroc.{d}"] = auroc(s[y == 1], s[y == 0])
            vals[f"auprc.{d}"] = auprc(s[y == 1], s[y == 0])
        for i, a in enumerate(detectors):
            for b in detectors[i + 1:]:
                if vals[f"auroc.{a}"] is None or vals[f"auroc.{b}"] is None:
                    vals[f"auroc_diff.{a}-{b}"] = None
                else:
                    vals[f"auroc_diff.{a}-{b}"] = vals[f"auroc.{a}"] - vals[f"auroc.{b}"]
        return vals
    out["bootstrap"] = bootstrap(labels, [c["source_cluster_id"] for c in common], stat)
    out["status"] = "CHARACTERIZED"
    return out


def analyze(scores, dev_records):
    """Metrics as a pure function of the scores file (and the DEV identity it must match)."""
    if scores.get("schema") != SCHEMA_SCORES or scores.get("partition") != t3.DEV:
        raise Refusal("not an R18-T4 DEV scores file")
    cases = sorted(scores["cases"], key=lambda c: c["sha256"])
    if {c["sha256"] for c in cases} != {r["sha256"] for r in dev_records}:
        raise Refusal("scores do not cover exactly the frozen DEV partition (S1)")
    frozen = {r["sha256"]: r for r in dev_records}
    for c in cases:
        r = frozen[c["sha256"]]
        if any(c[k] != r[k] for k in ("ground_truth", "source_cluster_id", "face_present",
                                      "generator_family", "capture_domain")):
            raise Refusal(f"case {c['sha256'][:12]} disagrees with the frozen DEV record")
    for c in cases:
        for d, e in c["detectors"].items():
            if (e["status"] == SUCCESS) != (e.get("score") is not None):
                raise Refusal(f"{c['sha256'][:12]}/{d}: a score without SUCCESS (or vice versa)")
    anchors = scores["anchors"]
    per = {d: detector_metrics(cases, d, anchors) for d in DETECTORS}
    paired = {}
    for i, a in enumerate(COMPARED):
        for b in COMPARED[i + 1:]:
            paired[f"{a}+{b}"] = paired_metrics(cases, (a, b))
    paired["+".join(COMPARED)] = paired_metrics(cases, COMPARED)
    ai = [c for c in cases if c["ground_truth"] == AI]
    return {
        "schema": SCHEMA_METRICS, "authority": AUTHORITY, "partition": t3.DEV,
        "freeze": scores["freeze"], "anchors": anchors,
        "inputs": {"scores_sha256": hashlib.sha256(dumps(scores).encode()).hexdigest(),
                   "scores_path": "scripts/eval/baselines/r18_t4_detector_scores.json"},
        "cohort": {"cases": len(cases), "ai": len(ai), "real": len(cases) - len(ai),
                   "clusters": len({c["source_cluster_id"] for c in cases}),
                   "ai_clusters": len({c["source_cluster_id"] for c in ai}),
                   "real_clusters": len({c["source_cluster_id"] for c in cases
                                         if c["ground_truth"] == REAL}),
                   "generator_family": dict(sorted(Counter(str(c["generator_family"])
                                                           for c in ai).items())),
                   "real_capture_domain": dict(sorted(Counter(
                       str(c["capture_domain"]) for c in cases
                       if c["ground_truth"] == REAL).items())),
                   "face_present": {f"{t}|{str(f).lower()}": n for (t, f), n in sorted(
                       Counter((c["ground_truth"], c["face_present"]) for c in cases).items())}},
        "method": {"seed": SEED, "bootstrap_replicates": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                   "target_fprs": list(TARGET_FPRS), "comparator": "score >= threshold",
                   "zero_fp_threshold": "np.nextafter(max(REAL), inf)",
                   "interpolation": "linear between empirical ROC points; no fitted model",
                   "metric_population": "status == SUCCESS only; abstentions, failures and "
                                        "timeouts are census entries, never scores",
                   "case_state_baseline": REAL_BASELINE},
        "detectors": per, "paired_common_success": paired}


def cmd_analyze(args):
    scores = json.loads(Path(args.scores).read_text())
    current = introspect_anchors()
    for k in ("ruleset_version", "svd_t_high", "face_t_high", "lip_t_high",
              "svd_provider_version", "face_provider_version", "lip_provider_version"):
        if current[k] != scores["anchors"][k]:
            raise Refusal(f"production anchor {k} changed since scoring "
                          f"({scores['anchors'][k]!r} -> {current[k]!r})")
    write_json(args.out, analyze(scores, load_dev_partition(args.freeze)))
    print(f"wrote {args.out}")


# --- report -----------------------------------------------------------------------------------

LABEL = {SVD: "SVD (NVIDIA)", FACE: "EfficientNet-B7", LIP: "LipForensics", EFFORT: "Effort"}


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def ci(boot, key, nd=3):
    if not boot or key not in boot["intervals"]:
        return "—"
    lo, hi = boot["intervals"][key]
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def render_report(m):
    a = m["anchors"]
    D = m["detectors"]
    s = []
    s.append("## Production anchors (introspected from HEAD)\n")
    s.append(table(["Anchor", "Value"], [
        ["commit_sha", f"`{a['commit_sha']}`"], ["ruleset_version", f"`{a['ruleset_version']}`"],
        ["SVD_T_HIGH", repr(a["svd_t_high"])], ["FACE_T_HIGH", repr(a["face_t_high"])],
        ["LIP_T_HIGH", repr(a["lip_t_high"]) + " (inert in r9-v5.0.0: no rule compares)"],
        ["SVD_PROVIDER_VERSION", f"`{a['svd_provider_version']}`"],
        ["source", a["source"]],
        ["risk_engine worktree == HEAD", str(a["risk_engine_worktree_matches_head"])]]))
    s.append("\n## Execution census\n")
    rows = []
    for d in DETECTORS:
        c = D[d]["census"]
        rows.append([LABEL[d], c["total_cases"], c["attempted"], c[SUCCESS], c[ABSTAINED],
                     c[FAILED], c[TIMEOUT], c[UNAVAILABLE],
                     f"{c['usable_ai']}/{c['total_ai']}", f"{c['usable_real']}/{c['total_real']}"])
    s.append(table(["Detector", "total", "attempted", "SUCCESS", "ABSTAINED", "FAILED",
                    "TIMEOUT", "UNAVAILABLE", "usable AI", "usable REAL"], rows))
    s.append("\nNon-success reasons:\n")
    rows = [[LABEL[d], r["status"], f"`{r['error_class']}`", r["count"]]
            for d in DETECTORS for r in D[d]["census"]["non_success_reasons"]]
    s.append(table(["Detector", "status", "error class", "count"], rows) if rows else "none")
    s.append("\nCoverage by face presence (usable / total):\n")
    rows = []
    for d in DETECTORS:
        f = D[d]["census"]["by_face_present"]
        rows.append([LABEL[d], f"{f['true']['usable_ai']}/{f['true']['total_ai']}",
                     f"{f['true']['usable_real']}/{f['true']['total_real']}",
                     f"{f['false']['usable_ai']}/{f['false']['total_ai']}",
                     f"{f['false']['usable_real']}/{f['false']['total_real']}"])
    s.append(table(["Detector", "face AI", "face REAL", "faceless AI", "faceless REAL"], rows))
    done = [d for d in DETECTORS if D[d].get("status") == "CHARACTERIZED"]
    s.append("\n## Score distributions (usable subset)\n")
    rows = []
    for d in done:
        dist = D[d]["distribution"]
        for cls in ("ai", "real"):
            x = dist[cls]
            rows.append([LABEL[d], cls.upper(), x["count"], fmt(x["mean"]), fmt(x["median"]),
                         f"{fmt(x['q1'])}–{fmt(x['q3'])}", fmt(x["iqr"]), fmt(x["p10"]),
                         fmt(x["p90"]), fmt(x["min"]), fmt(x["max"])])
    s.append(table(["Detector", "class", "n", "mean", "median", "Q1–Q3", "IQR", "p10", "p90",
                    "min", "max"], rows))
    s.append("")
    s.append(table(["Detector", "Cohen's d (AI−REAL)", "Wasserstein"],
                   [[LABEL[d], fmt(D[d]["distribution"]["cohens_d"], 2),
                     fmt(D[d]["distribution"]["wasserstein_distance"])] for d in done]))
    s.append("\n## Discrimination (usable subset, 95% class-stratified cluster bootstrap)\n")
    rows = []
    for d in done:
        x = D[d]["discrimination"]
        b = x["bootstrap"]
        rows.append([LABEL[d], f"{x['usable_ai']}/{x['usable_real']}", fmt(x["auroc"]),
                     ci(b, "auroc"), fmt(x["auprc"]), ci(b, "auprc"),
                     fmt(x["auprc_prevalence_baseline"]),
                     f"{b['valid_replicate_count']}/{b['invalid_replicate_count']}"])
    s.append(table(["Detector", "usable AI/REAL", "AUROC", "95% CI", "AUPRC", "95% CI",
                    "AUPRC baseline (prevalence)", "valid/invalid replicates"], rows))
    s.append("\n## Fixed-FPR operating points (comparator `score >= threshold`)\n")
    rows = []
    for d in done:
        for p in D[d]["fixed_fpr_operating_points"]:
            rows.append([LABEL[d], pct(p["target_fpr"]), p["usable_real_denominator"],
                         p["allowed_false_positives"], f"{p['threshold']!r}",
                         f"{p['false_positives']} ({pct(p['achievable_empirical_fpr'])})",
                         f"{p['true_positives']}/{p['usable_ai_denominator']} "
                         f"({pct(p['tpr_at_achievable_fpr'])})",
                         pct(p["interpolated_roc_tpr"])])
    s.append(table(["Detector", "target FPR", "usable REAL N", "allowed FP", "threshold",
                    "achievable FP (FPR)", "TPR at achievable FPR", "interpolated ROC TPR"],
                   rows))
    s.append("\n## Production operating points on DEV\n")
    rows = []
    for d in done:
        p = D[d].get("production_operating_point")
        if not p:
            rows.append([LABEL[d], "— (no production threshold)", "", "", "", ""])
            continue
        b = p["bootstrap"]
        rows.append([LABEL[d], repr(p["threshold"]),
                     f"{p['true_positives']}/{p['usable_ai_denominator']} "
                     f"({pct(p['empirical_tpr'])})", ci(b, "tpr"),
                     f"{p['false_positives']}/{p['usable_real_denominator']} "
                     f"({pct(p['empirical_fpr'])})", ci(b, "fpr")])
    s.append(table(["Detector", "threshold", "TP (TPR)", "TPR 95% CI", "FP (FPR)",
                    "FPR 95% CI"], rows))
    s.append("\n## Paired common-success subsets\n")
    for key, p in m["paired_common_success"].items():
        names = " ∩ ".join(LABEL[d] for d in p["detectors"])
        s.append(f"**{names}** — {p['cases']} cases ({p['usable_ai']} AI / "
                 f"{p['usable_real']} REAL, {p['clusters']} clusters, "
                 f"{p['face_present']} face-present)\n")
        if p["status"] != "CHARACTERIZED":
            s.append(f"{p['status']}\n")
            continue
        b = p["bootstrap"]
        s.append(table(["Detector", "AUROC", "95% CI", "AUPRC", "95% CI"],
                       [[LABEL[d], fmt(p["auroc"][d]), ci(b, f"auroc.{d}"),
                         fmt(p["auprc"][d]), ci(b, f"auprc.{d}")] for d in p["detectors"]]))
        diffs = [k for k in b["intervals"] if k.startswith("auroc_diff.")]
        if diffs:
            s.append("")
            s.append(table(["Paired AUROC difference", "point", "95% CI"], [
                [k.split(".", 1)[1].replace(SVD, "SVD").replace(FACE, "B7").replace(LIP, "Lip"),
                 fmt(p["auroc"][k.split(".", 1)[1].split("-")[0]]
                     - p["auroc"][k.split(".", 1)[1].split("-")[1]]), ci(b, k)] for k in diffs]))
        s.append(f"\nvalid/invalid replicates: {b['valid_replicate_count']}/"
                 f"{b['invalid_replicate_count']}\n")
    s.append("\n## Subgroups: AI generator family (each family vs all usable DEV REAL)\n")
    for d in done:
        s.append(f"\n**{LABEL[d]}**\n")
        rows = []
        for g, x in D[d]["subgroups"]["generator_family"].items():
            st = x["case_states"]
            rows.append([g, f"{x['usable']}/{x['total']}", x["clusters"],
                         fmt(x["distribution"].get("median")), fmt(x["auroc_vs_all_usable_real"]),
                         pct(x["tpr_at_production_threshold"]),
                         pct(x["tpr_at_dev_zero_fp_threshold"]),
                         st[STATE_FLAGGED], st[STATE_ABOVE], st[STATE_MID], st[STATE_LOW],
                         st[STATE_UNSCORED]])
        s.append(table(["family", "usable/total", "clusters", "median", "AUROC vs REAL",
                        "TPR@prod", "TPR@DEV 0%FPR", "FLAGGED", "ABOVE_BASELINE", "MID", "LOW",
                        "UNSCORED"], rows))
        rows = []
        for g, x in D[d]["subgroups"]["ai_source_collection"].items():
            rows.append([g, f"{x['usable']}/{x['total']}", fmt(x["distribution"].get("median")),
                         fmt(x["auroc_vs_all_usable_real"]),
                         pct(x["tpr_at_production_threshold"])])
        s.append("")
        s.append(table(["AI collection (T2V vs I2V)", "usable/total", "median",
                        "AUROC vs REAL", "TPR@prod"], rows))
    s.append("\n## Subgroups: REAL capture domain (all usable DEV AI vs each domain)\n")
    for d in done:
        s.append(f"\n**{LABEL[d]}**\n")
        rows = []
        for g, x in D[d]["subgroups"]["real_capture_domain"].items():
            fp = x["false_positives_at"]
            rows.append([g, f"{x['usable']}/{x['total']}", x["clusters"],
                         fmt(x["distribution"].get("median")), fmt(x["distribution"].get("p90")),
                         fmt(x["distribution"].get("max")),
                         fmt(x["auroc_all_usable_ai_vs_this_domain"]), fmt(fp["production"]),
                         fp["dev_fpr_1pct"], fp["dev_fpr_5pct"]])
        s.append(table(["domain", "usable/total", "clusters", "median", "p90", "max",
                        "AUROC (AI vs domain)", "FP@prod", "FP@DEV1%", "FP@DEV5%"], rows))
    s.append("\n## Subgroups: face presence\n")
    rows = []
    for d in done:
        for fv in ("true", "false"):
            x = D[d]["subgroups"]["face_present"][fv]
            disc = x.get("discrimination")
            op = x.get("production_operating_point")
            rows.append([LABEL[d], fv, f"{x['usable_ai']}/{x['usable_real']}",
                         fmt(disc["auroc"]) if disc else "—",
                         ci(disc["bootstrap"], "auroc") if disc and disc["bootstrap"] else "—",
                         pct(op["empirical_tpr"]) if op else "—",
                         pct(op["empirical_fpr"]) if op else "—"])
    s.append(table(["Detector", "face_present", "usable AI/REAL", "AUROC", "95% CI",
                    "TPR@prod", "FPR@prod"], rows))
    s.append("\n## Descriptive case states (AI cases; baseline = p90 of usable DEV REAL)\n")
    rows = []
    for d in done:
        cs = D[d]["case_states_ai"]
        n = cs["counts"]
        rows.append([LABEL[d], fmt(cs["real_median"]), fmt(cs["real_p90"]), fmt(cs["real_max"]),
                     n[STATE_FLAGGED], n[STATE_ABOVE], n[STATE_MID], n[STATE_LOW],
                     n[STATE_UNSCORED], cs["ai_scored_above_real_max"]])
    s.append(table(["Detector", "REAL median", "REAL p90", "REAL max", STATE_FLAGGED,
                    STATE_ABOVE, STATE_MID, STATE_LOW, STATE_UNSCORED, "AI > max(REAL)"], rows))
    return "\n".join(s) + "\n"


def cmd_report(args):
    m = json.loads(Path(args.metrics).read_text())
    Path(args.out).write_text(render_report(m))
    print(f"wrote {args.out}")


# --- cli --------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--freeze", default=str(FREEZE))
    s.add_argument("--corpus", default=str(CORPUS))
    s.add_argument("--work", required=True)
    s.add_argument("--cache", default=str(SVD_CACHE))
    s.add_argument("--out", default=str(SCORES))
    s.add_argument("--detectors", nargs="*", choices=DETECTORS)
    s.add_argument("--lip-shards", type=int, default=2)
    s.add_argument("--local-shards", type=int, default=1)
    s.add_argument("--svd-pace", type=float, default=SVD_PACE_S,
                   help="seconds between NVIDIA calls (plan §7.1: 1.5-2.0)")
    s.add_argument("--svd-backoff", type=float, default=SVD_BACKOFF_S,
                   help="first retry delay after an availability event; doubles per attempt")
    s.add_argument("--svd-breaker", type=int, default=0,
                   help="stop the SVD round after this many consecutive TIMEOUTs (0 = never)")
    s.add_argument("--assemble-only", action="store_true",
                   help="run no container; build the scores file from the work logs and cache")
    s.add_argument("--limit", type=int, default=0, help="smoke test: first N DEV clips by "
                                                        "sha256; writes no scores file")
    s.set_defaults(func=cmd_score)
    a = sub.add_parser("analyze")
    a.add_argument("--freeze", default=str(FREEZE))
    a.add_argument("--scores", default=str(SCORES))
    a.add_argument("--out", default=str(METRICS))
    a.set_defaults(func=cmd_analyze)
    r = sub.add_parser("report")
    r.add_argument("--metrics", default=str(METRICS))
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
