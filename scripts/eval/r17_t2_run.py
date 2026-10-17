"""R17-T2 Phase A: submit a blind execution manifest through the product, then seal what came back.

    # 0. a session for the harness account (inside the api container; token to a 0600 file)
    python3 scripts/eval/r17_t2_run.py session --token-file $SCRATCH/harness.token
    # 1. Phase A, one case at a time (rerun with --resume after an interruption)
    python3 scripts/eval/r17_t2_run.py run --input $SCRATCH/phase_a_input \
        --out $SCRATCH/phase_a --token-file $SCRATCH/harness.token
    # 2. seal (plan §4.5); copy the printed root into the run log and the report
    python3 scripts/eval/r17_t2_run.py seal --out $SCRATCH/phase_a
    python3 scripts/eval/r17_t2_run.py verify --out $SCRATCH/phase_a --expected-seal-root <root>

**What this file can reach is the blindness argument.** It reads the input directory, and
nothing else of the run: `execution_manifest.json` (per case `blind_id`, `clip_path`,
`clip_sha256`, nothing more) and `<blind_id>.mp4` files beside it. A directory holding any
other file is refused, so the blind map cannot sit next to the clips, and there is no argument,
constant or import that names it. Of the dataset it only hashes the three anchored files
(through `check_anchors`); it parses none of them.

**The hash gate holds here too.** The runner refuses to start if `HASH_MISMATCH.json` exists or
any clip differs from its manifest hash, and it re-hashes each clip just before the upload; a
mismatch then records `hash_blocked`, submits nothing, and stops the run.

**Nothing is decided here.** Verdict, rule, ruleset and calibration are what the production
worker committed. The runner maps nothing to Fake/Real. A completed analysis on a ruleset other
than `r9-v5.0.0` is recorded and stops the run (stop S5).

Product path: `POST /api/v1/analyses` with a session cookie, the web origin and
`mode=deep_analysis`, as the dashboard does. Ports, origin and container names come from the
current Compose config (read only), not from constants.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.r17_t2_extract import check_anchors  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / "scripts/eval/specs"
FREEZE_DOC = SPECS / "r17_t2_scoring_freeze.md"
SCHEMA = SPECS / "r17_t2_result_schema.json"
DIGESTS = SPECS / "r17_t2_expected_digests.json"
COMPOSE = REPO / "docker-compose.yml"
ACCOUNT = "r17t2-harness@example.com"
MODE = "deep_analysis"
EXPECTED_RULESET = "r9-v5.0.0"
MANIFEST_KEYS = {"blind_id", "clip_path", "clip_sha256"}
BLIND_ID = re.compile(r"^B\d{3}$")
SEAL = "PHASE_A_SEAL.json"
SEAL_ROOT = "PHASE_A_SEAL.root.sha256"
SEAL_SCHEMA = "r17-t2-seal-1"

DECISION_TIMEOUT_S = 45 * 60
ENRICHMENT_TIMEOUT_S = 120 * 60
POLL_S = 10
TERMINAL_ENRICHMENT = {"ENRICHMENT_COMPLETE", "ENRICHMENT_PARTIAL", "ENRICHMENT_FAILED"}
SIGNALS = ("synthetic_video", "face_manipulation", "lip_forensics", "audio_authenticity",
           "active_speaker", "provenance")
SERVICES = ("api", "api-worker", "api-worker-enrichment", "web")

RENDER_SCRIPT = REPO / "scripts/eval/r16_t7_render_check.mjs"  # used read-only, only with --render-check
PW_CORE = os.environ.get("PW_CORE", "/home/adentechio/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core")
CHROME = os.environ.get("CHROME", "/home/adentechio/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome")

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
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- seal (plan §4.5)

def seal_files(out):
    """[[posix relative path, sha256]] of every regular file except the seal and its root file,
    ordered by the UTF-8 bytes of the path. Anything that is neither a file nor a directory is refused."""
    out = Path(out)
    entries = []
    for p in out.rglob("*"):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):
            raise SystemExit(f"not a regular file: {p}")
        if p.is_dir():
            continue
        rel = p.relative_to(out).as_posix()
        if rel in (SEAL, SEAL_ROOT):
            continue
        entries.append([rel, sha256_file(p)])
    entries.sort(key=lambda e: e[0].encode("utf-8"))
    return entries


def canonical(entries):
    return json.dumps({"schema": SEAL_SCHEMA, "files": entries}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def seal_root(entries):
    return hashlib.sha256(canonical(entries)).hexdigest()


def verify_seal(out, expected_root):
    """Recompute from disk. Missing, extra or changed files, or a root other than the expected
    one passed in by the operator, are a refusal. The expected value never comes from `out`."""
    out = Path(out)
    entries = seal_files(out)
    root = seal_root(entries)
    detail = {"root": root, "expected": expected_root}
    if (out / SEAL).is_file():
        listed = dict(map(tuple, json.loads((out / SEAL).read_text())["files"]))
        disk = dict(map(tuple, entries))
        detail.update(missing=sorted(set(listed) - set(disk)), extra=sorted(set(disk) - set(listed)),
                      changed=sorted(k for k in set(listed) & set(disk) if listed[k] != disk[k]))
    else:
        detail["seal_file"] = "missing"
    ok = root == expected_root and not detail.get("missing") and not detail.get("extra") \
        and not detail.get("changed") and "seal_file" not in detail
    return ok, detail


def cmd_seal(args):
    out = Path(args.out).expanduser().resolve()
    if (out / SEAL).exists() or (out / SEAL_ROOT).exists():
        sys.exit("already sealed")
    entries = seal_files(out)
    root = seal_root(entries)
    write_json(out / SEAL, {"schema": SEAL_SCHEMA, "files": entries, "root_sha256": root,
                            "sealed_at": now()})
    (out / SEAL_ROOT).write_text(root + "\n")
    # Convenience only: read-only is never counted as integrity evidence.
    for p in [*out.rglob("*"), out]:
        os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    print(f"sealed {len(entries)} files")
    print(f"PHASE_A_SEAL_ROOT={root}")


def cmd_verify(args):
    ok, detail = verify_seal(Path(args.out).expanduser().resolve(), args.expected_seal_root)
    print(json.dumps({"ok": ok, **detail}))
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- input (blind)

def load_input(input_dir):
    """The execution manifest and its clips, refused unless the directory is exactly that."""
    input_dir = Path(input_dir).resolve()
    if (input_dir / "HASH_MISMATCH.json").exists():
        raise SystemExit("HASH_MISMATCH.json present: the hash gate is closed, refusing to start")
    if not (input_dir / "execution_manifest.json").is_file():
        raise SystemExit(f"no execution_manifest.json in {input_dir}")
    manifest = json.loads((input_dir / "execution_manifest.json").read_text())
    cases = manifest["cases"]
    allowed = {"execution_manifest.json"}
    for c in cases:
        if set(c) != MANIFEST_KEYS:
            raise SystemExit(f"execution manifest entry has keys {sorted(c)}")
        if not BLIND_ID.match(c["blind_id"]) or c["clip_path"] != f"{c['blind_id']}.mp4":
            raise SystemExit(f"unexpected blind id or clip path: {c['blind_id']} {c['clip_path']}")
        allowed.add(c["clip_path"])
    present = {p.name for p in input_dir.iterdir()}
    if present != allowed:
        raise SystemExit(f"input directory holds unexpected entries: {sorted(present ^ allowed)}")
    bad = [c["blind_id"] for c in cases if sha256_file(input_dir / c["clip_path"]) != c["clip_sha256"]]
    if bad:
        raise SystemExit(f"clip hash mismatch for {bad}: refusing to start, nothing submitted")
    return input_dir, cases


# ---------------------------------------------------------------- compose (read only)

def compose_endpoints():
    cfg = json.loads(subprocess.run(["docker", "compose", "-f", str(COMPOSE), "config", "--format",
                                     "json"], capture_output=True, text=True, check=True).stdout)
    svc = cfg["services"]

    def published(name):
        p = svc[name]["ports"][0]
        host = p.get("host_ip") or "127.0.0.1"
        return ("127.0.0.1" if host == "0.0.0.0" else host), p["published"]

    api_host, api_port = published("api")
    _, web_port = published("web")
    origins = (svc["api"].get("environment") or {}).get("DEEPGUARD_WEB_ORIGIN", "").split(",")
    origin = next((o for o in origins if o.strip() == f"http://localhost:{web_port}"), None)
    if origin is None:
        raise SystemExit(f"no allowed origin http://localhost:{web_port} in DEEPGUARD_WEB_ORIGIN")
    ps = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "ps", "--format", "json"],
                        capture_output=True, text=True, check=True).stdout
    rows = json.loads(ps) if ps.lstrip().startswith("[") else [json.loads(l) for l in ps.splitlines() if l]
    containers = {r["Service"]: r["Name"] for r in rows}
    return {"api": f"http://{api_host}:{api_port}", "web": f"http://localhost:{web_port}",
            "origin": origin, "containers": {s: containers.get(s) for s in SERVICES}}


# ---------------------------------------------------------------- product path

class Client:
    def __init__(self, ep, token):
        self.ep, self.token = ep, token

    def request(self, method, path, body=None, headers=None, timeout=600):
        req = urllib.request.Request(self.ep["api"] + path, data=body, method=method)
        req.add_header("Cookie", f"deepguard_session={self.token}")
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

    def upload(self, path):
        boundary = uuid.uuid4().hex
        head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\n{MODE}\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
                f"Content-Type: video/mp4\r\n\r\n").encode()
        body = head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self.request("POST", "/api/v1/analyses", body, {
            "Content-Type": f"multipart/form-data; boundary={boundary}", "Origin": self.ep["origin"]})

    def get(self, aid):
        return self.request("GET", f"/api/v1/analyses/{aid}", timeout=60)


def record_detail(rec, detail):
    """Only the fields production committed. Nothing is mapped or decided."""
    rec["status"] = detail.get("status")
    for k in ("risk_level", "risk_rule_id", "risk_rules_version", "risk_calibration_id",
              "decision_state", "aggregate_enrichment_state"):
        rec[k] = detail.get(k)
    rec["signals"] = {s: None if not isinstance(detail.get(s), dict) else
                      {"status": detail[s].get("status"), "score": detail[s].get("score")}
                      for s in SIGNALS}


def poll(client, rec, case_dir, t0, sleep=time.sleep):
    decided_at = enriched_at = None
    detail = None
    while True:
        elapsed = time.monotonic() - t0
        try:
            code, detail = client.get(rec["analysis_id"])
        except (urllib.error.URLError, OSError) as err:
            rec["events"].append([round(elapsed, 1), f"poll error {type(err).__name__}"])
            sleep(POLL_S)
            continue
        if code == 200 and isinstance(detail, dict):
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
        else:
            rec["events"].append([round(elapsed, 1), f"poll HTTP {code}"])
        sleep(POLL_S)
    write_json(case_dir / "analysis.json", detail)
    rec["seconds_to_decision"] = None if decided_at is None else round(decided_at, 1)
    rec["seconds_to_enrichment_terminal"] = None if enriched_at is None else round(enriched_at, 1)
    if isinstance(detail, dict):
        record_detail(rec, detail)
    return rec


def run_case(client, case, input_dir, case_dir, adopted=None):
    rec = {"blind_id": case["blind_id"], "clip_sha256_expected": case["clip_sha256"], "events": [],
           "analysis_id": None, "error": None}
    path = input_dir / case["clip_path"]
    observed = sha256_file(path)
    rec["clip_sha256_observed"] = observed
    if observed != case["clip_sha256"]:
        rec.update(hash_blocked=True, error="clip sha256 mismatch; not submitted")
        return rec
    t0 = time.monotonic()
    if adopted:
        rec.update(analysis_id=adopted, adopted_after_interruption=True, adopted_at=now())
        return poll(client, rec, case_dir, t0)
    rec["submitted_at"] = now()
    try:
        code, body = client.upload(path)
    except (urllib.error.URLError, OSError) as err:
        rec["error"] = f"upload transport error: {type(err).__name__}"
        return rec
    rec["upload_http_status"] = code
    rec["upload_seconds"] = round(time.monotonic() - t0, 3)
    write_json(case_dir / "upload_response.json", body)
    if code != 202 or not isinstance(body, dict) or "id" not in body:
        rec["error"] = f"upload refused: HTTP {code}"
        return rec
    rec["analysis_id"] = str(body["id"])
    return poll(client, rec, case_dir, t0)


def render_check(ep, token, rec, case_dir):
    proc = subprocess.run(["node", str(RENDER_SCRIPT), ep["web"], token, rec["analysis_id"],
                           rec["clip_sha256_observed"], str(case_dir)],
                          capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PW_CORE": PW_CORE, "CHROME": CHROME})
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "error": "render script produced no result", "returncode": proc.returncode}


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def environment(ep, input_dir, anchors):
    images = {s: subprocess.run(["docker", "inspect", "-f", "{{.Image}}", c], capture_output=True,
                                text=True).stdout.strip() if c else None
              for s, c in ep["containers"].items()}
    return {
        "task": "R17-T2 Phase A", "started_at": now(), "mode": MODE, "account": ACCOUNT,
        "expected_ruleset": EXPECTED_RULESET, "endpoints": ep, "container_images": images,
        "repo_head": git("rev-parse", "HEAD"),
        "specs_dirty": bool(git("status", "--porcelain", "--", "scripts/eval/specs/")),
        "spec_sha256": {"scoring_freeze": sha256_file(FREEZE_DOC), "result_schema": sha256_file(SCHEMA),
                        "expected_digests": sha256_file(DIGESTS)},
        "dataset_anchors_verified": anchors,
        "execution_manifest_sha256": sha256_file(input_dir / "execution_manifest.json"),
        "runner_sha256": sha256_file(__file__),
    }


def execute(client, cases, input_dir, out, adopt, render=None):
    """One case at a time. A finished case (`record.json`) is never touched again. Returns an exit code."""
    results = []
    for case in cases:
        case_dir = out / "cases" / case["blind_id"]
        if (case_dir / "record.json").exists():
            results.append(json.loads((case_dir / "record.json").read_text()))
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{now()}] {case['blind_id']} …", flush=True)
        rec = run_case(client, case, input_dir, case_dir, adopt.get(case["blind_id"]))
        if render and rec.get("status") == "completed":
            rec["render"] = render(rec, case_dir)
        write_json(case_dir / "record.json", rec)
        results.append(rec)
        print(f"   → {rec.get('status')} {rec.get('risk_level')} {rec.get('risk_rule_id')} "
              f"enrich={rec.get('aggregate_enrichment_state')} {rec.get('error') or ''}", flush=True)
        if rec.get("hash_blocked"):
            print("HARD STOP: clip hash mismatch at submission time", file=sys.stderr)
            return 2
        if rec.get("status") == "completed" and rec.get("risk_rules_version") != EXPECTED_RULESET:
            print(f"STOP S5: ruleset {rec.get('risk_rules_version')!r}", file=sys.stderr)
            return 3
    write_json(out / "results.json", {"cases": results, "finished_at": now()})
    print(f"Phase A done: {len(results)} cases → {out}. Seal it next.")
    return 0


def cmd_run(args):
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not args.resume:
        sys.exit(f"{out} exists; Phase A is run once (use --resume)")
    if args.resume and not out.exists():
        sys.exit(f"{out} does not exist; nothing to resume")
    if args.resume and (out / SEAL).exists():
        sys.exit("already sealed")
    _, anchors = check_anchors()
    input_dir, cases = load_input(Path(args.input).expanduser())
    if input_dir == out or input_dir in out.parents or out in input_dir.parents:
        sys.exit("output and input directories must not overlap")
    ep = compose_endpoints()
    token = Path(args.token_file).read_text().strip()
    client = Client(ep, token)
    adopt = {}
    if args.resume:
        n = len(list(out.glob("resume_meta_*.json"))) + 1
        for case in cases:
            d = out / "cases" / case["blind_id"]
            if (d / "upload_response.json").exists() and not (d / "record.json").exists():
                adopt[case["blind_id"]] = str(json.loads((d / "upload_response.json").read_text())["id"])
        write_json(out / f"resume_meta_{n}.json", {**environment(ep, input_dir, anchors),
                                                   "adopted_after_interruption": adopt})
    else:
        out.mkdir(parents=True)
        write_json(out / "environment.json", environment(ep, input_dir, anchors))
        (out / "execution_manifest.json").write_bytes((input_dir / "execution_manifest.json").read_bytes())
    render = (lambda rec, d: render_check(ep, token, rec, d)) if args.render_check else None
    sys.exit(execute(client, cases, input_dir, out, adopt, render))


def cmd_session(args):
    ep = compose_endpoints()
    token = subprocess.run(
        ["docker", "exec", "-i", "-w", "/app", ep["containers"]["api"], "python", "-", ACCOUNT],
        input=SESSION_SNIPPET, check=True, capture_output=True, text=True).stdout.strip()
    fd = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    print(f"session for {ACCOUNT} written to {args.token_file}")


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("session")
    s.add_argument("--token-file", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--token-file", required=True)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--render-check", action="store_true", help="EN/TR report render check (off by default)")
    se = sub.add_parser("seal")
    se.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    v.add_argument("--expected-seal-root", required=True)
    return ap


def main():
    args = build_parser().parse_args()
    {"session": cmd_session, "run": cmd_run, "seal": cmd_seal, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
