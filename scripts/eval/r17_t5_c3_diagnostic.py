"""R17-T5 C3: re-query NVIDIA synthetic_video for the 40 sealed T3 failures, as an isolated diagnostic.

    PYTHONPATH=scripts setsid nohup python3 scripts/eval/r17_t5_c3_diagnostic.py \
        --spec-sha256 <recorded before any experiment> \
        --phase-a ~/deepguard-corpus/r17_t3_phase_a --results ~/deepguard-corpus/r17_t3_phase_b/results.json \
        --dataset ~/deepguard-corpus/r17-ai-visual-v1 --clips ~/deepguard-corpus/r17_t3_work/phase_a_input \
        --out-dir ~/deepguard-corpus/r17_t5_diagnostic > ~/deepguard-corpus/r17_t5_diagnostic.log 2>&1 &

Applied exactly as `scripts/eval/specs/r17_t5_challenger_spec.md` §3 C3 froze it. This is not an
offline evaluator run: it calls the external provider. Every record it writes carries
`provenance: "diagnostic_revalidation"` and lands only under `--out-dir`, which may not lie inside
the repository or any sealed T3 directory. Nothing sealed is read for writing.

Each call is the committed `app.nvidia_video.analyze_video`, unchanged, inside an ephemeral
`docker compose run --rm --no-deps` container of the `api` image with the clips mounted
read-only. No running service is started, restarted or modified. One attempt per case, no
retries: a case with an attempt marker but no result is recorded as interrupted, never re-sent.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from eval.r17_t2_run import now, sha256_file, write_json
from eval.r17_t4_analyze import REPO, RISK_ENGINE, Refusal, load_risk_engine
from eval.r17_t5_challenger import (C3_POPULATION, C3_PROVENANCE, T4_RECORD, check_constants,
                                    load_cases, svd_usable, verify_all)

NVIDIA_VIDEO = REPO / "apps/api/app/nvidia_video.py"
PAUSE_S = 30
CALL_DEADLINE_S = 600 + 180  # the production deadline (app.limits) plus container start-up
FORMAT_CODES = frozenset({"INVALID_ARGUMENT", "FAILED_PRECONDITION", "OUT_OF_RANGE",
                          "UNIMPLEMENTED"})
CONTAINER_CLIPS = "/r17t5_clips"

# Spec §3 C3 outcome table, error branch in table order; the first matching class name wins.
ERROR_OUTCOMES = (("NvidiaProviderUnavailable", "provider_unavailable"),
                  ("NvidiaProviderTimeout", "provider_timeout"),
                  ("NvidiaAuthenticationError", "authentication_error"),
                  ("NvidiaProviderInvalidResponse", "invalid_response"),
                  ("NvidiaLocalFileError", "local_file_error"))

CONTAINER_SNIPPET = r"""
import asyncio, hashlib, json, os, sys, time
from datetime import datetime, timezone
sys.path.insert(0, "/app")
from pathlib import Path

def digest(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

out = {"nvidia_video_sha256": digest("/app/app/nvidia_video.py"),
       "risk_engine_sha256": digest("/app/app/risk_engine.py")}
if sys.argv[1] == "--preflight":
    out["function_id_matches"] = os.environ.get("NVIDIA_SVD_FUNCTION_ID") == sys.argv[2]
    print(json.dumps(out))
    sys.exit(0)
from app import nvidia_video as nv
out["started_at"] = datetime.now(timezone.utc).isoformat()
t0 = time.monotonic()
try:
    r = asyncio.run(nv.analyze_video(Path(sys.argv[1])))
    out["result"] = {"probability": r.probability, "logit": r.logit, "total_clips": r.total_clips,
                     "function_id": r.function_id, "clips_returned": len(r.clips)}
except Exception as e:
    cause = e.__cause__
    code = cause.code().name if cause is not None and hasattr(cause, "code") else None
    out["error"] = {"class": type(e).__name__, "mro": [c.__name__ for c in type(e).__mro__],
                    "grpc_code": code, "message": str(e)[:500]}
out["elapsed_s"] = round(time.monotonic() - t0, 2)
out["finished_at"] = datetime.now(timezone.utc).isoformat()
print(json.dumps(out))
"""


def container(clips_dir, *argv, timeout):
    cmd = ["docker", "compose", "run", "--rm", "--no-deps", "-T",
           "-v", f"{Path(clips_dir).resolve()}:{CONTAINER_CLIPS}:ro",
           "--entrypoint", "python", "api", "-I", "-c", CONTAINER_SNIPPET, *argv]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "host deadline exceeded"
    lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        return None, f"container exit {proc.returncode}: {proc.stderr[-800:]}"
    return json.loads(lines[-1]), None


def classify(re_mod, reply):
    """Spec §3 C3: exactly one outcome per case; returns (outcome, svd_evidence or None)."""
    if reply is None:
        return "other_provider_error", None
    if "result" in reply:
        r = reply["result"]
        ev = {"provider": re_mod.SVD_PROVIDER, "signal_type": re_mod.SVD_SIGNAL_TYPE,
              "status": re_mod.SIGNAL_STATUS_SUCCESS, "provider_version": r["function_id"],
              "score": r["probability"], "total_clips": r["total_clips"]}
        usable = svd_usable(re_mod, re_mod.SvdEvidence(**ev))
        return ("usable_response" if usable else "ineligible_response"), ev
    err = reply["error"]
    if "NvidiaProviderError" in err["mro"] and err.get("grpc_code") in FORMAT_CODES:
        return "input_format_incompatibility", None
    for cls, outcome in ERROR_OUTCOMES:
        if cls in err["mro"]:
            return outcome, None
    return "other_provider_error", None


def check_out_dir(out_dir, sealed):
    out = Path(out_dir).expanduser().resolve()
    for forbidden in (REPO, *sealed):
        f = Path(forbidden).expanduser().resolve()
        if out == f or f in out.parents:
            raise Refusal(f"--out-dir {out} lies inside {f}; C3 output must stay isolated (S4)")
    return out


def preflight(clips_dir, re_mod):
    reply, err = container(clips_dir, "--preflight", re_mod.SVD_PROVIDER_VERSION, timeout=300)
    if reply is None:
        raise Refusal(f"preflight container failed: {err}")
    expected = {"nvidia_video_sha256": sha256_file(NVIDIA_VIDEO),
                "risk_engine_sha256": sha256_file(RISK_ENGINE)}
    for key, value in expected.items():
        if reply[key] != value:
            raise Refusal(f"image {key} {reply[key]} differs from the committed file {value}")
    if not reply["function_id_matches"]:
        raise Refusal("configured NVIDIA_SVD_FUNCTION_ID is not the calibrated deployment")
    return reply


def run(args):
    verification = verify_all(args.phase_a, args.results, args.dataset, args.spec_sha256)
    out = check_out_dir(args.out_dir, (args.phase_a, Path(args.results).parent, args.dataset,
                                                  args.clips))
    re_mod = load_risk_engine()
    check_constants(re_mod)
    cases = load_cases(args.phase_a, args.results, json.loads(T4_RECORD.read_text()))
    manifest = {m["case_id"]: m for m in json.loads((Path(args.dataset) / "manifest.json").read_text())}
    population = [c for c in cases if (c["svd"] or {}).get("status") in ("FAILED", "TIMEOUT")]
    if len(population) != C3_POPULATION:
        raise Refusal(f"{len(population)} sealed SVD failures, spec freezes {C3_POPULATION}")

    (out / "cases").mkdir(parents=True, exist_ok=True)
    pre = preflight(args.clips, re_mod)
    started = now()
    print(f"[{started}] preflight ok; {len(population)} cases", flush=True)

    for i, c in enumerate(population):
        cid = c["case_id"]
        result_path, marker = out / "cases" / f"{cid}.json", out / "cases" / f"{cid}.attempt.json"
        if result_path.exists():
            continue
        base = {"provenance": C3_PROVENANCE, "case_id": cid, "blind_id": c["blind_id"],
                "truth": c["truth"], "t3_analysis_id": c["analysis_id"]}
        if marker.exists():
            write_json(result_path, {**base, "outcome": "other_provider_error",
                                     "note": "interrupted attempt, not retried (one attempt per case)",
                                     "recorded_at": now()})
            continue
        clip = Path(args.clips) / f"{c['blind_id']}.mp4"
        sealed_media = json.loads((Path(args.phase_a) / "cases" / c["blind_id"] /
                                   "analysis.json").read_text())["media_sha256s"]
        live = sha256_file(clip) if clip.exists() else None
        if live is None or live != manifest[cid]["clip_sha256"] or [live] != sealed_media:
            write_json(result_path, {**base, "outcome": "hash_blocked", "clip_sha256": live,
                                     "recorded_at": now()})
            continue
        write_json(marker, {**base, "attempt_started_at": now()})
        reply, err = container(args.clips, f"{CONTAINER_CLIPS}/{clip.name}", timeout=CALL_DEADLINE_S)
        outcome, ev = classify(re_mod, reply)
        write_json(result_path, {**base, "outcome": outcome, "clip_sha256": live,
                                 "svd_evidence": ev,
                                 "grpc_code": ((reply or {}).get("error") or {}).get("grpc_code"),
                                 "reply": reply, "container_error": err, "recorded_at": now()})
        print(f"[{now()}] {i + 1}/{len(population)} {cid} {outcome}", flush=True)
        if i + 1 < len(population):
            time.sleep(PAUSE_S)

    records = [json.loads((out / "cases" / f"{c['case_id']}.json").read_text()) for c in population]
    write_json(out / "c3_records.json", {
        "provenance": C3_PROVENANCE, "task": "R17-T5 C3 NVIDIA availability diagnostic",
        "not_merged_into": "sealed R17-T3 Phase A/B, T3 anchor, T4 record",
        "spec_sha256": verification["spec_sha256"], "verification": verification,
        "nvidia_video_sha256": pre["nvidia_video_sha256"],
        "risk_engine_sha256": pre["risk_engine_sha256"],
        "pause_s": PAUSE_S, "attempts_per_case": 1,
        "run_started_at": started, "run_finished_at": now(), "records": records})
    print(f"[{now()}] done", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--spec-sha256", "--phase-a", "--results", "--dataset", "--clips", "--out-dir"):
        p.add_argument(name, required=True)
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
