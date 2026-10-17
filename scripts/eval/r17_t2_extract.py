"""R17-T2: cut the frozen R17 clips into a blind Phase A input, or make a non-dataset smoke input.

    # dataset: the 45 frozen clips, cut with T1's pinned image and exact recipe (no analysis)
    python3 scripts/eval/r17_t2_extract.py dataset \
        --source https://www.youtube.com/watch?v=HK6y8DAPN_0=$SRC/HK6y8DAPN_0.mp4 ... \
        --clips-out $SCRATCH/phase_a_input --blind-map-out $SCRATCH/blind_map
    # smoke: 1-2 synthetic lavfi clips, never an R17 case
    python3 scripts/eval/r17_t2_extract.py smoke --count 2 \
        --clips-out $SCRATCH/smoke_input --blind-map-out $SCRATCH/smoke_map

**Trust anchors first.** Before anything is read, the live `FREEZE.json`, `manifest.csv` and
`manifest.json` must hash to the values in the tracked `specs/r17_t2_expected_digests.json`
(plan §4.4). The runner and the scorer import `check_anchors` from here and run it too.

**The clip hash is a hard gate** (plan §4.1). Any clip whose sha256 differs from its frozen
`clip_sha256` is deleted, `HASH_MISMATCH.json` is written to the clip directory, no execution
manifest is written, and the process exits 2. There is no tolerance and no count threshold.

**Blinding** (plan §4.2). The clip directory holds only `execution_manifest.json` (per case:
`blind_id`, `clip_path`, `clip_sha256`) and `<blind_id>.mp4`. `blind_id`s are `B001…` in
`sha256(case_id)` order, which does not follow truth. The case and cluster ids go to
`blind_map.json` in a separate directory that the runner has no way to name. Every clip's
container and stream tags are read back and refused if they carry a truth, case, cluster,
generator or publisher string (exit 3, stop S3: clip bytes cannot change without breaking the gate).
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIGESTS = REPO / "scripts/eval/specs/r17_t2_expected_digests.json"
ANCHORED = ("FREEZE.json", "manifest.csv", "manifest.json")
CLIP_SECONDS = "8.000"
GENERATOR_WORDS = ("sora", "veo", "runway", "nasa", "artemis", "sls", "openai", "deepmind",
                   "google", "orion", "gen-3", "gen-4", "gen3", "gen4")
_TRUTH_WORD = re.compile(r"(?<![a-z0-9])(ai|real)(?![a-z0-9])", re.I)
_CASE_ID = re.compile(r"(AI|REAL)-\d{3}", re.I)
SMOKE_PATTERNS = ("testsrc2", "smptehdbars")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


class AnchorError(SystemExit):
    pass


def check_anchors(dataset=None, digests_path=DIGESTS):
    """Live dataset files vs the tracked digests. Never against a value a run recorded."""
    expected = json.loads(Path(digests_path).read_text())
    dataset = Path(dataset or expected["dataset_path"])
    bad = {}
    for name in ANCHORED:
        live = sha256_file(dataset / name) if (dataset / name).is_file() else None
        if live != expected["sha256"][name]:
            bad[name] = {"expected": expected["sha256"][name], "live": live}
    if bad:
        raise AnchorError(f"trust anchor mismatch, refusing: {json.dumps(bad)}")
    return dataset, dict(expected["sha256"])


def parse_ts(t):
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def blind_order(case_ids):
    """Truth-independent submission order: ascending sha256(case_id) hex."""
    return sorted(case_ids, key=lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest())


def leak_tokens(case_ids=(), cluster_ids=()):
    return [*(c.lower() for c in case_ids), *(c.lower() for c in cluster_ids), *GENERATOR_WORDS]


def leaks(text, tokens):
    """The forbidden strings found in `text`: a truth word, a case-id shape, or a listed token."""
    low = str(text).lower()
    found = [t for t in tokens if t and t in low]
    found += [m.group(0) for m in _TRUTH_WORD.finditer(low)]
    found += [m.group(0) for m in _CASE_ID.finditer(low)]
    return sorted(set(found))


def check_dirs(clips_out, map_out, forbidden_roots):
    """Separate, non-nested, new directories outside the repo and the dataset."""
    clips_out, map_out = Path(clips_out).resolve(), Path(map_out).resolve()
    if clips_out == map_out or clips_out in map_out.parents or map_out in clips_out.parents:
        raise SystemExit("clip and blind-map directories must be separate and not nested")
    for d in (clips_out, map_out):
        for root in forbidden_roots:
            root = Path(root).resolve()
            if d == root or root in d.parents:
                raise SystemExit(f"{d} lies inside {root}")
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"{d} exists and is not empty")
    return clips_out, map_out


def load_frozen(dataset):
    """45 cases from the anchored FREEZE.json, cross-checked against the anchored manifest."""
    freeze = json.loads((dataset / "FREEZE.json").read_text())
    rows = {r["case_id"]: r for r in json.loads((dataset / "manifest.json").read_text())}
    cases = []
    for c in freeze["clips"]:
        r = rows[c["case_id"]]
        for k in ("source_cluster_id", "clip_start", "clip_end", "clip_sha256"):
            if r[k] != c[k]:
                raise SystemExit(f"{c['case_id']}: FREEZE.json and manifest.json differ on {k}")
        if round(parse_ts(c["clip_end"]) - parse_ts(c["clip_start"]), 3) != float(CLIP_SECONDS):
            raise SystemExit(f"{c['case_id']}: window is not {CLIP_SECONDS} s")
        cases.append({**c, "source_url": r["source_url"]})
    if len(cases) != freeze["counts"]["cases"] or set(rows) != {c["case_id"] for c in cases}:
        raise SystemExit("FREEZE.json clips and manifest.json cases differ")
    return freeze, cases


def ffmpeg_cmd(image, mounts, args):
    uid = f"{os.getuid()}:{os.getgid()}"
    vol = [x for host, ctr, mode in mounts for x in ("-v", f"{host}:{ctr}:{mode}")]
    return ["docker", "run", "--rm", "--network", "none", "--user", uid, *vol,
            "--entrypoint", "ffmpeg", image, "-nostdin", "-loglevel", "error", "-y", *args]


# T1's exact recipe (VALIDATION_REPORT §5), argument for argument.
def encode(dst):
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-threads", "1", "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-map_metadata", "-1",
            "-fflags", "+bitexact", "-flags", "+bitexact", "-movflags", "+faststart", dst]


def recipe(start_s, src, dst):
    return ["-ss", f"{start_s:.3f}", "-i", src, "-t", CLIP_SECONDS, "-map", "0:v:0", "-map", "0:a:0?",
            *encode(dst)]


def probe_tags(image, clip):
    """Every container and stream tag, read with the same pinned image."""
    uid = f"{os.getuid()}:{os.getgid()}"
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--user", uid, "-v",
         f"{clip.parent}:/c:ro", "--entrypoint", "ffprobe", image, "-v", "error",
         "-show_entries", "format_tags:stream_tags", "-of", "json", f"/c/{clip.name}"],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    tags = dict((d.get("format") or {}).get("tags") or {})
    for i, s in enumerate(d.get("streams") or []):
        tags.update({f"stream{i}.{k}": v for k, v in (s.get("tags") or {}).items()})
    return tags


def run_extraction(cases, clips_out, map_out, cut, tags_of, tokens, population="r17-dataset",
                   extra_log=None):
    """Cut every case, gate on the hash, check tags, then write the blind input. Returns an exit code.

    `cut(case, dst)` writes the clip; `tags_of(path)` returns its tags. Both are injected so the
    gate and the blinding are tested without docker.
    """
    clips_out.mkdir(parents=True, exist_ok=True)
    map_out.mkdir(parents=True, exist_ok=True)
    order = blind_order([c["case_id"] for c in cases])
    by_id = {c["case_id"]: c for c in cases}
    entries, mismatches, tag_leaks = [], [], []
    path_leak = leaks(str(clips_out), tokens)
    if path_leak:
        raise SystemExit(f"clip directory path leaks {path_leak}")
    for i, case_id in enumerate(order, 1):
        c = by_id[case_id]
        blind_id = f"B{i:03d}"
        dst = clips_out / f"{blind_id}.mp4"
        cut(c, dst)
        observed = sha256_file(dst)
        entry = {"blind_id": blind_id, "case_id": case_id, "source_cluster_id": c["source_cluster_id"],
                 "clip_sha256": c["clip_sha256"], "observed_sha256": observed}
        if population == "smoke":
            entry["truth"] = c["truth"]
        print(f"{blind_id} {'OK' if observed == c['clip_sha256'] else 'MISMATCH'}", flush=True)
        if observed != c["clip_sha256"]:
            mismatches.append(entry)
            dst.unlink()
            continue
        found = sorted({x for k, v in tags_of(dst).items() for x in leaks(f"{k}={v}", tokens)})
        if found:
            tag_leaks.append({"blind_id": blind_id, "case_id": case_id, "leaks": found})
        entries.append(entry)

    log = {"population": population, "clips": len(order), "mismatches": len(mismatches),
           "tag_leaks": tag_leaks, **(extra_log or {})}
    if mismatches:
        write_json(clips_out / "HASH_MISMATCH.json", {
            "schema": "r17-t2-hash-mismatch-1", "final_state": "hash_blocked",
            "clips": [{k: m[k] for k in ("blind_id", "clip_sha256", "observed_sha256")}
                      for m in mismatches]})
        write_json(map_out / "HASH_MISMATCH_detail.json", {"mismatches": mismatches})
        write_json(map_out / "extraction_log.json", log)
        print(f"HARD STOP: {len(mismatches)} clip hash mismatch(es); nothing may be submitted",
              file=sys.stderr)
        return 2
    if tag_leaks:
        write_json(map_out / "extraction_log.json", log)
        print(f"STOP S3: clip tags leak {tag_leaks}", file=sys.stderr)
        return 3
    write_json(clips_out / "execution_manifest.json", {
        "schema": "r17-t2-execution-manifest-1",
        "cases": [{"blind_id": e["blind_id"], "clip_path": f"{e['blind_id']}.mp4",
                   "clip_sha256": e["observed_sha256"]} for e in entries]})
    write_json(map_out / "blind_map.json", {
        "schema": "r17-t2-blind-map-1", "population": population,
        "order_rule": "ascending sha256(case_id) hex",
        "entries": [{k: v for k, v in e.items() if k != "observed_sha256"} for e in entries]})
    write_json(map_out / "extraction_log.json", log)
    print(f"{len(entries)} clips, 0 mismatches → {clips_out}; blind map → {map_out}")
    return 0


def cmd_dataset(args):
    dataset, anchors = check_anchors(args.dataset)
    freeze, cases = load_frozen(dataset)
    image = freeze["tools"]["tool_image"]
    subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True)
    sources = {}
    for s in args.source:
        url, path = s.rsplit("=", 1)
        sources[url] = Path(path).expanduser().resolve()
    missing = {c["source_url"] for c in cases} - set(sources)
    if missing:
        raise SystemExit(f"no --source for {sorted(missing)}")
    clips_out, map_out = check_dirs(args.clips_out, args.blind_map_out, (REPO, dataset))

    def cut(c, dst):
        src = sources[c["source_url"]]
        mounts = [(src.parent, "/src", "ro"), (dst.parent, "/out", "rw")]
        subprocess.run(ffmpeg_cmd(image, mounts, recipe(parse_ts(c["clip_start"]), f"/src/{src.name}",
                                                         f"/out/{dst.name}")), check=True)

    tokens = leak_tokens([c["case_id"] for c in cases], [c["source_cluster_id"] for c in cases])
    log = {"tool_image": image, "anchors": anchors,
           "sources": {u: {"path": str(p), "sha256": sha256_file(p)} for u, p in sources.items()},
           "extractor_sha256": sha256_file(__file__)}
    sys.exit(run_extraction(cases, clips_out, map_out, cut, lambda p: probe_tags(image, p), tokens,
                            extra_log=log))


def cmd_smoke(args):
    """Synthetic lavfi clips through the same image and encoder settings. Never an R17 case."""
    dataset, anchors = check_anchors(args.dataset)
    freeze = json.loads((dataset / "FREEZE.json").read_text())
    image = freeze["tools"]["tool_image"]
    if not 1 <= args.count <= len(SMOKE_PATTERNS):
        raise SystemExit("smoke uses 1-2 clips")
    clips_out, map_out = check_dirs(args.clips_out, args.blind_map_out, (REPO, dataset))
    tmp = map_out / "smoke_src"
    tmp.mkdir(parents=True)
    cases = []
    pairs = list(zip(SMOKE_PATTERNS, ("REAL", "AI")))[:args.count]  # placeholder labels
    for i, (pattern, truth) in enumerate(pairs, 1):
        dst = tmp / f"smoke{i}.mp4"
        lavfi = ["-f", "lavfi", "-i", f"{pattern}=size=1280x720:rate=30:duration=8",
                 "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
                 "-t", CLIP_SECONDS, "-map", "0:v:0", "-map", "1:a:0"]
        subprocess.run(ffmpeg_cmd(image, [(tmp, "/out", "rw")], [*lavfi, *encode(f"/out/{dst.name}")]),
                       check=True)
        cases.append({"case_id": f"SMOKE-{i:03d}", "source_cluster_id": f"SMOKE-CL-{i}",
                      "truth": truth, "clip_sha256": sha256_file(dst), "_src": dst})
    real_ids = {c["case_id"] for c in freeze["clips"]}
    if real_ids & {c["case_id"] for c in cases}:
        raise SystemExit("a smoke case id collides with an R17 case id")

    def cut(c, dst):
        dst.write_bytes(c["_src"].read_bytes())

    tokens = leak_tokens([c["case_id"] for c in cases], [c["source_cluster_id"] for c in cases])
    code = run_extraction(cases, clips_out, map_out, cut, lambda p: probe_tags(image, p), tokens,
                          population="smoke", extra_log={"tool_image": image, "anchors": anchors,
                                                         "patterns": list(SMOKE_PATTERNS[:args.count])})
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--dataset", default=None, help="defaults to dataset_path in the tracked digests")
    d.add_argument("--source", action="append", default=[], metavar="SOURCE_URL=PATH")
    d.add_argument("--clips-out", required=True)
    d.add_argument("--blind-map-out", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--dataset", default=None)
    s.add_argument("--count", type=int, default=2)
    s.add_argument("--clips-out", required=True)
    s.add_argument("--blind-map-out", required=True)
    args = ap.parse_args()
    {"dataset": cmd_dataset, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
