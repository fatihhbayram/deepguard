"""R16-T7: run the blind Real-World Pilot through the product, then seal what it produced.

    # 0. the pilot account and a session for it (inside the api container; token to a 0600 file)
    python3 scripts/eval/r16_t7_blind_pilot.py session --token-file $SCRATCH/pilot.token
    # 1. Phase A: blind execution (sealed/ is chmod 000 for the whole run)
    python3 scripts/eval/r16_t7_blind_pilot.py run --pilot ~/deepguard-corpus/r16_pilot \
        --out ~/deepguard-corpus/r16_t7_phase_a --token-file $SCRATCH/pilot.token
    # 2. seal the Phase A output (read-only afterwards); only then may sealed/ be reopened
    python3 scripts/eval/r16_t7_blind_pilot.py seal --out ~/deepguard-corpus/r16_t7_phase_a

**What this file can reach is the blindness argument.** It reads `pilot_manifest.json` and the
files the manifest names under `media/`, and nothing else from the pilot directory: there is no
argument, constant or import that names `sealed/`, the ground-truth file or `r16_pilot_raw/`, and
a manifest path that escapes `media/` is refused. `run` also makes `sealed/` unreadable (mode 000)
before the first upload and leaves it that way; `seal` restores it only after the snapshot hash is
on disk. Scoring (`r16_t7_score.py`) is a separate file written after the seal.

**Nothing is decided here.** The verdict, rule, ruleset and calibration are read off the analysis
the production worker committed (`docs/ai/r16_t7_scoring_freeze.md` §2). The runner records what
came back, how long it took and whether the report rendered; it maps nothing to Fake/Real.

Product path: `POST /api/v1/analyses` with a session cookie, `Origin: http://localhost:3005` and
`mode=deep_analysis` — the dashboard upload, unchanged. One case at a time, in manifest order.
"""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

API = "http://127.0.0.1:8001"
WEB = "http://localhost:3005"
ORIGIN = "http://localhost:3005"
API_CONTAINER = "deepguard-api-1"
PILOT_EMAIL = "r16t7-pilot@example.com"
MODE = "deep_analysis"
EXPECTED_RULESET = "r9-v5.0.0"
FREEZE_DOC = Path(__file__).resolve().parents[2] / "docs/ai/r16_t7_scoring_freeze.md"

DECISION_TIMEOUT_S = 45 * 60
ENRICHMENT_TIMEOUT_S = 120 * 60
POLL_S = 10
CONTAINERS = ("deepguard-api-1", "deepguard-api-worker-3", "deepguard-api-worker-enrichment-2",
              "deepguard-web-1")
TERMINAL_ENRICHMENT = {"ENRICHMENT_COMPLETE", "ENRICHMENT_PARTIAL", "ENRICHMENT_FAILED"}
SIGNALS = ("synthetic_video", "face_manipulation", "lip_forensics", "audio_authenticity",
           "active_speaker", "provenance")

RENDER_SCRIPT = Path(__file__).with_name("r16_t7_render_check.mjs")
PW_CORE = os.environ.get("PW_CORE", "/home/adentechio/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core")
CHROME = os.environ.get("CHROME", "/home/adentechio/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")

# Runs inside the api container: create the pilot account if it is not there, open a session.
# The password is random and dropped; the account is only ever used through the minted session.
SESSION_SNIPPET = r"""
import secrets, sys
from sqlalchemy import select
from app.db.session import SessionLocal
from app.db.models import User, USER_ROLE_USER
from app.web_auth import hash_password, normalize_email, start_session
email = normalize_email(sys.argv[1])
with SessionLocal() as s:
    user = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        user = User(email=email, password_hash=hash_password(secrets.token_urlsafe(32)),
                    role=USER_ROLE_USER, is_active=True)
        s.add(user); s.commit(); s.refresh(user)
        print("created", user.id, file=sys.stderr)
    print(start_session(s, user))
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def cmd_session(args):
    token = subprocess.run(
        ["docker", "exec", "-i", "-w", "/app", API_CONTAINER, "python", "-", PILOT_EMAIL],
        input=SESSION_SNIPPET, check=True, capture_output=True, text=True).stdout.strip()
    fd = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    print(f"session for {PILOT_EMAIL} written to {args.token_file}")


def request(method, path, token, body=None, headers=None, timeout=600):
    req = urllib.request.Request(API + path, data=body, method=method)
    req.add_header("Cookie", f"deepguard_session={token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        raw = err.read()
        try:
            return err.code, json.loads(raw)
        except ValueError:
            return err.code, raw.decode("utf-8", "replace")[:500]


def upload(token, path, filename):
    boundary = uuid.uuid4().hex
    ctype = "video/quicktime" if filename.lower().endswith(".mov") else "video/mp4"
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{MODE}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode()
    body = head + Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return request("POST", "/api/v1/analyses", token, body, {
        "Content-Type": f"multipart/form-data; boundary={boundary}", "Origin": ORIGIN})


def render_check(token, analysis_id, sha, out_dir):
    proc = subprocess.run(
        ["node", str(RENDER_SCRIPT), WEB, token, analysis_id, sha, str(out_dir)],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "PW_CORE": PW_CORE, "CHROME": CHROME})
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": "render script produced no result",
                "returncode": proc.returncode, "stderr": proc.stderr[-1000:]}


def run_case(case, media_root, token, case_dir, adopted=None):
    rec = {"case_id": case["case_id"], "manifest_sha256": case["sha256"],
           "manifest_path": case["path"], "events": []}
    path = (media_root.parent / case["path"]).resolve()
    if media_root not in path.parents:
        rec.update(outcome_operational="manifest path escapes media/", analysis_id=None)
        return rec
    observed = sha256_file(path)
    rec["observed_sha256"] = observed
    if observed != case["sha256"]:
        rec.update(outcome_operational="sha256 mismatch; not uploaded", analysis_id=None)
        return rec

    t0 = time.monotonic()
    if adopted:
        # Freeze addendum A1: uploaded once in attempt 1 and never again. Polled, not re-sent;
        # its upload→decision wall clock includes the worker outage and is excluded from latency.
        rec.update(adopted_from_attempt_1=True, wall_clock_latency_excluded=True,
                   analysis_id=adopted, adopted_at=now())
        return poll_case(rec, case, token, case_dir, t0)
    rec["submitted_at"] = now()
    try:
        code, body = upload(token, path, path.name)
    except (urllib.error.URLError, OSError) as err:
        rec.update(outcome_operational=f"upload transport error: {type(err).__name__}",
                   analysis_id=None, upload_reached_server=False)
        return rec
    rec["upload_http_status"] = code
    rec["upload_seconds"] = round(time.monotonic() - t0, 3)
    write_json(case_dir / "upload_response.json", body)
    if code != 202 or not isinstance(body, dict) or "id" not in body:
        rec.update(outcome_operational=f"upload refused: HTTP {code}", analysis_id=None)
        return rec
    rec["analysis_id"] = str(body["id"])
    return poll_case(rec, case, token, case_dir, t0)


def poll_case(rec, case, token, case_dir, t0):
    aid = rec["analysis_id"]
    decided_at = enriched_at = None
    detail = None
    while True:
        elapsed = time.monotonic() - t0
        try:
            code, detail = request("GET", f"/api/v1/analyses/{aid}", token, timeout=60)
        except (urllib.error.URLError, OSError) as err:
            rec["events"].append([round(elapsed, 1), f"poll error {type(err).__name__}"])
            time.sleep(POLL_S)
            continue
        if code != 200 or not isinstance(detail, dict):
            rec["events"].append([round(elapsed, 1), f"poll HTTP {code}"])
        else:
            if decided_at is None and detail.get("status") in ("completed", "failed"):
                decided_at = elapsed
                rec["events"].append([round(elapsed, 1), f"status {detail['status']}"])
            if detail.get("status") == "failed":
                break
            if decided_at is not None and detail.get("aggregate_enrichment_state") in TERMINAL_ENRICHMENT:
                enriched_at = elapsed
                break
            if decided_at is None and elapsed > DECISION_TIMEOUT_S:
                rec["timeout"] = "decision"
                break
            if elapsed > ENRICHMENT_TIMEOUT_S:
                rec["timeout"] = "enrichment"
                break
        time.sleep(POLL_S)

    write_json(case_dir / "analysis.json", detail)
    rec["seconds_to_decision"] = None if decided_at is None else round(decided_at, 1)
    rec["seconds_to_enrichment_terminal"] = None if enriched_at is None else round(enriched_at, 1)
    if isinstance(detail, dict):
        rec["status"] = detail.get("status")
        for k in ("risk_level", "risk_rules_version", "risk_rule_id", "risk_calibration_id",
                  "decision_state", "aggregate_enrichment_state"):
            rec[k] = detail.get(k)
        trace = detail.get("risk_trace")
        rec["risk_trace_present"] = trace is not None
        rec["risk_trace_rule_id"] = trace.get("rule_id") if isinstance(trace, dict) else None
        rec["signals"] = {s: (None if detail.get(s) is None else detail[s].get("status"))
                          for s in SIGNALS}
        rec["media_sha256_reported"] = detail.get("original_sha256")
    if rec.get("status") == "completed":
        rec["render"] = render_check(token, aid, case["sha256"], case_dir)
    return rec


def cmd_run(args):
    pilot = Path(args.pilot).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not args.resume:
        sys.exit(f"{out} exists; Phase A is run once")
    if args.resume and not out.exists():
        sys.exit(f"{out} does not exist; nothing to resume")
    adopt = dict(a.split("=", 1) for a in args.adopt)
    manifest_path = pilot / "pilot_manifest.json"
    media_root = (pilot / "media").resolve()
    token = Path(args.token_file).read_text().strip()

    # The technical half of the blindness guard: nothing in this process, or anything it starts,
    # can open the ground truth while Phase A runs.
    sealed = pilot / "sealed"
    if sealed.exists():
        os.chmod(sealed, 0)

    if args.resume:
        return resume(args, pilot, out, media_root, token, adopt)
    out.mkdir(parents=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    started = now()
    images = {c: subprocess.run(["docker", "inspect", "-f", "{{.Image}}", c], capture_output=True,
                                text=True).stdout.strip() for c in CONTAINERS}
    write_json(out / "run_meta.json", {
        "task": "R16-T7 Phase A", "attempt": args.attempt, "started_at": started,
        "adopted_cases": adopt, "container_images": images, "api": API, "web": WEB, "mode": MODE,
        "expected_ruleset": EXPECTED_RULESET, "account": PILOT_EMAIL,
        "execution_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "freeze_doc_sha256": sha256_file(FREEZE_DOC),
        "sealed_dir_mode_during_run": "000",
        "runner_sha256": sha256_file(__file__),
        "render_script_sha256": sha256_file(RENDER_SCRIPT),
    })

    results = []
    for case in manifest["cases"]:
        case_dir = out / "cases" / case["case_id"]
        case_dir.mkdir(parents=True)
        print(f"[{now()}] {case['case_id']} …", flush=True)
        rec = run_case(case, media_root, token, case_dir, adopt.get(case["case_id"]))
        write_json(case_dir / "record.json", rec)
        results.append(rec)
        print(f"   → {rec.get('status')} {rec.get('risk_level')} {rec.get('risk_rule_id')} "
              f"enrich={rec.get('aggregate_enrichment_state')} "
              f"render={(rec.get('render') or {}).get('ok')} {rec.get('outcome_operational', '')}",
              flush=True)

    for name in ("deepguard-api-worker-3", "deepguard-api-worker-enrichment-2"):
        log = subprocess.run(["docker", "logs", "--since", started, name], capture_output=True,
                             text=True)
        (out / f"{name}.log").write_text(log.stdout + log.stderr)
    worker_times(out, results)
    write_json(out / "results.json", {"cases": results, "finished_at": now()})
    print(f"Phase A done: {len(results)} cases → {out}. Seal it next.")


def resume(args, pilot, out, media_root, token, adopt):
    """Freeze addendum A2: continue an interrupted attempt in place. A case with a `record.json`
    is finished and never touched again; a case with only `upload_response.json` reached the
    server and is adopted (polled, never re-sent, wall clock excluded); the rest run as usual."""
    started_first = json.loads((out / "run_meta.json").read_text())["started_at"]
    manifest = json.loads((pilot / "pilot_manifest.json").read_text())
    resumes = sorted(out.glob("resume_meta*.json"))
    meta_path = out / f"resume_meta_{len(resumes) + 1}.json"
    interrupted = {}
    for case in manifest["cases"]:
        d = out / "cases" / case["case_id"]
        if (d / "upload_response.json").exists() and not (d / "record.json").exists():
            interrupted[case["case_id"]] = json.loads((d / "upload_response.json").read_text())["id"]
    adopt = {**interrupted, **adopt}
    write_json(meta_path, {
        "resumed_at": now(), "interrupted_cases_adopted": interrupted, "adopted_cases": adopt,
        "runner_sha256": sha256_file(__file__), "freeze_doc_sha256": sha256_file(FREEZE_DOC),
        "render_script_sha256": sha256_file(RENDER_SCRIPT),
        "container_images": {c: subprocess.run(["docker", "inspect", "-f", "{{.Image}}", c],
                             capture_output=True, text=True).stdout.strip() for c in CONTAINERS},
    })
    results = []
    for case in manifest["cases"]:
        case_dir = out / "cases" / case["case_id"]
        if (case_dir / "record.json").exists():
            results.append(json.loads((case_dir / "record.json").read_text()))
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{now()}] {case['case_id']} …", flush=True)
        rec = run_case(case, media_root, token, case_dir, adopt.get(case["case_id"]))
        if case["case_id"] in interrupted:
            rec["interrupted_runner_adopted"] = True
            rec.pop("adopted_from_attempt_1", None)
        write_json(case_dir / "record.json", rec)
        results.append(rec)
        print(f"   → {rec.get('status')} {rec.get('risk_level')} {rec.get('risk_rule_id')} "
              f"enrich={rec.get('aggregate_enrichment_state')} "
              f"render={(rec.get('render') or {}).get('ok')} {rec.get('outcome_operational', '')}",
              flush=True)
    for name in ("deepguard-api-worker-3", "deepguard-api-worker-enrichment-2"):
        log = subprocess.run(["docker", "logs", "--since", started_first, name],
                             capture_output=True, text=True)
        (out / f"{name}.log").write_text(log.stdout + log.stderr)
    worker_times(out, results)
    write_json(out / "results.json", {"cases": results, "finished_at": now()})
    print(f"Phase A done: {len(results)} cases → {out}. Seal it next.")


def worker_times(out, results):
    """Worker processing time per analysis, from the decision worker's own log lines: the
    `Claimed job … for analysis <id>` line to the `Completed job` line of the same request."""
    claimed, completed = {}, {}
    for line in (out / "deepguard-api-worker-3.log").read_text().splitlines():
        ts = line[:23]
        if "Claimed job" in line and "for analysis" in line:
            aid = line.split("for analysis ", 1)[1].split(".", 1)[0].strip()
            rid = line.rsplit("request_id=", 1)[-1].strip()
            claimed[rid] = (aid, ts)
        elif ("Completed job" in line or ("Job " in line and " failed." in line)) and "request_id=" in line:
            completed[line.rsplit("request_id=", 1)[-1].strip()] = ts
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    by_aid = {}
    for rid, (aid, ts) in claimed.items():
        if rid in completed:
            by_aid[aid] = round((datetime.strptime(completed[rid], fmt)
                                 - datetime.strptime(ts, fmt)).total_seconds(), 3)
    for rec in results:
        rec["worker_decision_processing_seconds"] = by_aid.get(rec.get("analysis_id"))


def cmd_seal(args):
    out = Path(args.out).expanduser().resolve()
    seal_path = out / "PHASE_A_SEAL.json"
    if seal_path.exists():
        sys.exit("already sealed")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    per_file = {str(p.relative_to(out)): sha256_file(p) for p in files}
    root = hashlib.sha256(json.dumps(per_file, sort_keys=True).encode()).hexdigest()
    write_json(seal_path, {"schema": "r16-t7-phase-a-seal-1", "sealed_at": now(),
                           "file_count": len(per_file), "per_file_sha256": per_file,
                           "snapshot_sha256": root})
    for p in [*out.rglob("*"), out]:
        mode = os.stat(p).st_mode
        os.chmod(p, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"sealed {len(per_file)} files, snapshot_sha256={root}")
    pilot = Path(args.pilot).expanduser().resolve()
    if (pilot / "sealed").exists():
        os.chmod(pilot / "sealed", 0o775)
        print("sealed/ readable again (Phase B may start)")


def cmd_verify(args):
    out = Path(args.out).expanduser().resolve()
    seal = json.loads((out / "PHASE_A_SEAL.json").read_text())
    bad = [k for k, v in seal["per_file_sha256"].items() if sha256_file(out / k) != v]
    extra = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} \
        - set(seal["per_file_sha256"]) - {"PHASE_A_SEAL.json"}
    root = hashlib.sha256(json.dumps(seal["per_file_sha256"], sort_keys=True).encode()).hexdigest()
    ok = not bad and not extra and root == seal["snapshot_sha256"]
    print(json.dumps({"ok": ok, "changed": bad, "extra": sorted(extra), "snapshot_sha256": root}))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session"); s.add_argument("--token-file", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--token-file", required=True)
    r.add_argument("--attempt", type=int, required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--adopt", action="append", default=[], metavar="CASE_ID=ANALYSIS_ID")
    se = sub.add_parser("seal"); se.add_argument("--out", required=True); se.add_argument("--pilot", required=True)
    v = sub.add_parser("verify"); v.add_argument("--out", required=True)
    args = ap.parse_args()
    {"session": cmd_session, "run": cmd_run, "seal": cmd_seal, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
