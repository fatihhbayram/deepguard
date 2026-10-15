"""R17-T1: validate the AI-visual evaluation manifest and freeze it.

    # 1. every check, nothing written; exit status 1 if anything fails
    PYTHONPATH=scripts python3 scripts/eval/r17_validate_manifest.py validate \
        --dataset ~/deepguard-corpus/r17-ai-visual-v1
    # 2. the same checks, then the frozen manifest and FREEZE.json (all 45 cases or nothing)
    PYTHONPATH=scripts python3 scripts/eval/r17_validate_manifest.py freeze \
        --dataset ~/deepguard-corpus/r17-ai-visual-v1 \
        --yt-dlp 2026.08.19 --ffmpeg 7.1.5 --tool-image sha256:...

Pure checks over files already on disk. Nothing here downloads, decodes or scores anything:
source metadata, truth evidence, visual checks and clip hashes are columns of the manifest,
filled in from the official sources before this runs, and this module refuses to freeze unless
every one of them is present and every case is confirmed.

**Ground truth is not the channel.** A case is frozen only with a verbatim official assertion
(AI) or capture statement (REAL) from an official host, a visual check that passed, and
`gt_status = CONFIRMED`. One case short of that and nothing is frozen: there is no partial
freeze.

**The original columns change only where the CHANGELOG says so.** `manifest.provisional.*` is
the byte-for-byte archive of the assembled manifest; the frozen manifest keeps its ten columns
and may differ from it only in cells with a CHANGELOG row (and in `validation_status`, which
freezing sets). Corrections themselves are chosen by `shift_window` and `first_free_shot`,
fixed rules declared before any clip was looked at by a detector.

**Timestamps are `MM:SS` or `MM:SS.fff`.** Showcase scenes do not start on whole seconds, and
an 8-second window has to fit inside one shot. All arithmetic is in integer milliseconds.

**Cases are not independent.** Cases sharing a `source_url` share a `source_cluster_id`; any
rate computed later is reported per case and per cluster, and a cluster-level result over this
few clusters is not generalisable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

FREEZE_SCHEMA = "r17-t1-freeze-1"
CLIP_MS = 8000
DEFAULT_MARGIN_MS = 500

ORIGINAL_COLUMNS = (
    "case_id", "truth", "generator_or_source", "publisher", "source_url",
    "clip_start", "clip_end", "notes", "provenance", "validation_status",
)
ADDED_COLUMNS = (
    "source_title", "source_channel", "source_duration_s", "source_upload_date",
    "accessed_at_utc", "url_status", "clip_duration_s", "clip_sha256", "clip_format_id",
    "clip_resolution", "clip_fps", "clip_has_audio", "source_cluster_id", "timestamp_status",
    "truth_evidence_type", "truth_evidence_url", "truth_evidence_quote",
    "truth_evidence_accessed_at_utc", "truth_assertion_scope", "visual_gt_check",
    "visual_gt_note", "gt_status",
)
COLUMNS = ORIGINAL_COLUMNS + ADDED_COLUMNS

# The dataset as agreed: 45 cases, and how they divide. Changing any of this is a redesign.
# The two NASA sources are the Architect-approved replacements (R17-T1 ruling) for the 33 s SLS
# package and the graphics-laden Artemis I broadcast; the 10/5 split is unchanged.
EXPECTED_TRUTH = {"AI": 30, "REAL": 15}
EXPECTED_GENERATORS = {
    "Sora": 17, "Veo": 11, "Runway Gen-3": 1, "Runway Gen-4": 1,
    "NASA Artemis II Orion SAW1": 10, "NASA Artemis II Orion ascent": 5,
}
OFFICIAL_PUBLISHERS = {"OpenAI", "Google DeepMind", "Runway", "NASA"}

EVIDENCE_TYPES = {
    "official_statement_description", "official_page", "official_model_documentation",
    "official_capture_statement",
}
# Hosts an assertion may come from. A YouTube description counts only as the publisher's own
# statement on its own verified channel (`source_channel` must be an official publisher).
OFFICIAL_EVIDENCE_HOSTS = ("openai.com", "deepmind.google", "runway.com", "runwayml.com", "nasa.gov")
YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "youtu.be")
SCOPES = {"whole_video", "segment_verified"}
TIMESTAMP_STATUSES = {"ORIGINAL", "CORRECTED", "ASSIGNED"}
STATUSES_BEFORE_FREEZE = {"PROVISIONAL", "NEEDS_TIMESTAMP", "VALIDATED"}

_TS = re.compile(r"^(\d{2,3}):([0-5]\d)(?:\.(\d{1,3}))?$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """A manifest that cannot be read as one, as opposed to one that fails a check."""


# --- timestamps -----------------------------------------------------------------------------

def parse_ts(text):
    """`MM:SS` or `MM:SS.f`–`MM:SS.fff` to integer milliseconds."""
    m = _TS.match(text or "")
    if not m:
        raise ManifestError(f"bad timestamp {text!r}: expected MM:SS or MM:SS.fff")
    frac = (m.group(3) or "").ljust(3, "0")
    return (int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(frac or 0)


def format_ts(ms):
    """Integer milliseconds to `MM:SS.fff` — the form every corrected cell is written in."""
    minutes, rest = divmod(int(ms), 60000)
    return f"{minutes:02d}:{rest // 1000:02d}.{rest % 1000:03d}"


def shift_window(orig_ms, shot, occupied=(), length=CLIP_MS, margin=DEFAULT_MARGIN_MS):
    """Rule 1: the window inside `shot` closest to `orig_ms`, overlapping nothing in `occupied`.

    `shot` is the (start, end) of one clean shot, in ms. The window keeps a margin from both
    cuts — `margin`, or less when the shot is too short to afford it (an exactly 8.000 s shot
    gets none). Returns (start, end) or None when no window fits: that is rule 2's case.
    Ties go to the earlier start, so the result is a function of the inputs alone.
    """
    a, b = shot
    m = max(0, min(margin, (b - a - length) // 2))
    lo, hi = a + m, b - m - length
    if lo > hi:
        return None

    def free(s):
        return lo <= s <= hi and all(s >= e or s + length <= st for st, e in occupied)

    candidates = {min(max(orig_ms, lo), hi), lo, hi}
    for st, e in occupied:
        candidates.update((e, st - length))
    feasible = sorted((abs(s - orig_ms), s) for s in candidates if free(s))
    if not feasible:
        return None
    s = feasible[0][1]
    return s, s + length


def first_free_shot(shots, used, length=CLIP_MS, margin=DEFAULT_MARGIN_MS):
    """Rule 2: the first clean shot, in source order, that holds no assigned clip.

    `shots` is a list of {"start_ms", "end_ms", "clean"}; `used` the (start, end) windows already
    assigned in the same source. Returns (start, end) or None — rule 3, which is a STOP.
    """
    for shot in shots:
        a, b = shot["start_ms"], shot["end_ms"]
        if not shot["clean"] or b - a < length:
            continue
        if any(st < b and e > a for st, e in used):
            continue
        return shift_window(a, (a, b), (), length, margin)
    return None


# --- reading and writing ------------------------------------------------------------------

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        header = tuple(reader.fieldnames or ())
    return header, rows


def read_json(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ManifestError(f"{path}: expected a JSON list of objects")
    return rows


def render_csv(rows, columns=COLUMNS):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in columns})
    return buf.getvalue()


def render_json(rows, columns=COLUMNS):
    return json.dumps([{c: r.get(c, "") for c in columns} for r in rows],
                      indent=2, ensure_ascii=False) + "\n"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


# --- checks ---------------------------------------------------------------------------------

def structure_findings(header, rows):
    """V1: columns, row count, unique ids, vocabulary and the agreed composition."""
    out = []
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        out.append(f"missing columns: {missing}")
    ids = [r.get("case_id", "") for r in rows]
    dup = sorted(k for k, v in Counter(ids).items() if v > 1)
    if dup:
        out.append(f"duplicate case_id: {dup}")
    if len(rows) != sum(EXPECTED_TRUTH.values()):
        out.append(f"{len(rows)} rows, expected {sum(EXPECTED_TRUTH.values())}")
    truth = Counter(r.get("truth") for r in rows)
    if dict(truth) != EXPECTED_TRUTH:
        out.append(f"truth counts {dict(truth)} != {EXPECTED_TRUTH}")
    gens = Counter(r.get("generator_or_source") for r in rows)
    if dict(gens) != EXPECTED_GENERATORS:
        out.append(f"generator counts {dict(gens)} != {EXPECTED_GENERATORS}")
    for r in rows:
        if r.get("publisher") not in OFFICIAL_PUBLISHERS:
            out.append(f"{r.get('case_id')}: publisher {r.get('publisher')!r} is not official")
        if r.get("validation_status") not in STATUSES_BEFORE_FREEZE:
            out.append(f"{r.get('case_id')}: validation_status {r.get('validation_status')!r}")
    return out


def equivalence_findings(csv_rows, json_rows):
    """CSV and JSON must hold the same rows, in the same order, cell for cell."""
    if len(csv_rows) != len(json_rows):
        return [f"csv has {len(csv_rows)} rows, json {len(json_rows)}"]
    out = []
    for i, (c, j) in enumerate(zip(csv_rows, json_rows)):
        if {k: str(v) for k, v in j.items()} != c:
            out.append(f"row {i} ({c.get('case_id')}): csv and json differ")
    return out


def windows(rows):
    """{source_url: [(start_ms, end_ms, case_id)]}, sorted. Unparseable rows are skipped."""
    by_url = {}
    for r in rows:
        try:
            s, e = parse_ts(r["clip_start"]), parse_ts(r["clip_end"])
        except (ManifestError, KeyError):
            continue
        by_url.setdefault(r["source_url"], []).append((s, e, r["case_id"]))
    return {u: sorted(v) for u, v in by_url.items()}


def overlaps(rows):
    """Every pair of clips from one source that share any time: (case_a, case_b, overlap_ms)."""
    out = []
    for clips in windows(rows).values():
        for i, (s1, e1, c1) in enumerate(clips):
            for s2, e2, c2 in clips[i + 1:]:
                if s2 < e1:
                    out.append((c1, c2, min(e1, e2) - s2))
    return out


def zero_gaps(rows):
    """Adjacent clips where one ends exactly where the next starts — legal, but worth a look."""
    out = []
    for clips in windows(rows).values():
        for (_, e1, c1), (s2, _, c2) in zip(clips, clips[1:]):
            if e1 == s2:
                out.append((c1, c2))
    return out


def timestamp_findings(rows):
    """V3: parseable, exactly 8.000 s, inside the source, no overlap within a source."""
    out = []
    for r in rows:
        cid = r.get("case_id")
        try:
            s, e = parse_ts(r.get("clip_start", "")), parse_ts(r.get("clip_end", ""))
        except ManifestError as exc:
            out.append(f"{cid}: {exc}")
            continue
        if e - s != CLIP_MS:
            out.append(f"{cid}: clip is {(e - s) / 1000:.3f}s, expected 8.000s")
        dur = r.get("source_duration_s", "")
        try:
            if e > round(float(dur) * 1000):
                out.append(f"{cid}: clip_end {r['clip_end']} is past source end {dur}s")
        except ValueError:
            out.append(f"{cid}: source_duration_s {dur!r} is not a number")
        if r.get("clip_duration_s") != "8.000":
            out.append(f"{cid}: clip_duration_s {r.get('clip_duration_s')!r} != '8.000'")
        if r.get("timestamp_status") not in TIMESTAMP_STATUSES:
            out.append(f"{cid}: timestamp_status {r.get('timestamp_status')!r}")
    for c1, c2, ms in overlaps(rows):
        out.append(f"{c1}/{c2}: overlap {ms / 1000:.3f}s")
    return out


def cluster_findings(rows):
    """One cluster per source URL and one source URL per cluster."""
    out = []
    url_to = {}
    cl_to = {}
    for r in rows:
        url, cl = r.get("source_url"), r.get("source_cluster_id")
        if not cl:
            out.append(f"{r.get('case_id')}: empty source_cluster_id")
            continue
        url_to.setdefault(url, set()).add(cl)
        cl_to.setdefault(cl, set()).add(url)
    out += [f"source {u} has clusters {sorted(c)}" for u, c in url_to.items() if len(c) > 1]
    out += [f"cluster {c} spans sources {sorted(u)}" for c, u in cl_to.items() if len(u) > 1]
    return out


def evidence_findings(r):
    """V2b: the truth claim is an official, verbatim, dated, scoped assertion — or nothing."""
    cid = r.get("case_id")
    out = []
    if r.get("truth_evidence_type") not in EVIDENCE_TYPES:
        out.append(f"{cid}: truth_evidence_type {r.get('truth_evidence_type')!r}")
    if not (r.get("truth_evidence_quote") or "").strip():
        out.append(f"{cid}: empty truth_evidence_quote")
    if not (r.get("truth_evidence_accessed_at_utc") or "").strip():
        out.append(f"{cid}: empty truth_evidence_accessed_at_utc")
    if r.get("truth_assertion_scope") not in SCOPES:
        out.append(f"{cid}: truth_assertion_scope {r.get('truth_assertion_scope')!r}")
    host = (urlparse(r.get("truth_evidence_url") or "").hostname or "").lower()
    if host in YOUTUBE_HOSTS:
        if r.get("truth_evidence_type") not in ("official_statement_description",
                                                 "official_capture_statement"):
            out.append(f"{cid}: a YouTube page can only carry a description statement")
        if r.get("source_channel") not in OFFICIAL_PUBLISHERS:
            out.append(f"{cid}: YouTube evidence from non-official channel "
                       f"{r.get('source_channel')!r}")
    elif not any(host == h or host.endswith("." + h) for h in OFFICIAL_EVIDENCE_HOSTS):
        out.append(f"{cid}: truth_evidence_url host {host!r} is not an official host")
    return out


def case_findings(r):
    """Everything one case needs before it may be frozen."""
    cid = r.get("case_id")
    out = evidence_findings(r)
    if r.get("url_status") != "OK":
        out.append(f"{cid}: url_status {r.get('url_status')!r}")
    if r.get("source_channel") not in OFFICIAL_PUBLISHERS:
        out.append(f"{cid}: source_channel {r.get('source_channel')!r} is not official")
    if not _SHA.match(r.get("clip_sha256") or ""):
        out.append(f"{cid}: clip_sha256 missing or malformed")
    for col in ("clip_format_id", "clip_resolution", "clip_fps", "clip_has_audio",
                "source_title", "source_upload_date", "accessed_at_utc"):
        if not (r.get(col) or "").strip():
            out.append(f"{cid}: empty {col}")
    if r.get("visual_gt_check") != "PASS":
        out.append(f"{cid}: visual_gt_check {r.get('visual_gt_check')!r}")
    if not (r.get("visual_gt_note") or "").strip():
        out.append(f"{cid}: empty visual_gt_note")
    if r.get("gt_status") != "CONFIRMED":
        out.append(f"{cid}: gt_status {r.get('gt_status')!r}")
    return out


# --- CHANGELOG and the provisional diff -----------------------------------------------------

_CHANGE_ROW = re.compile(r"^\|\s*((?:AI|REAL)-\d{3})\s*\|\s*([a-z_]+)\s*\|\s*`([^`]*)`\s*\|\s*`([^`]*)`\s*\|")


def parse_changelog(text):
    """{(case_id, field): (old, new)} from the CHANGELOG's change table rows.

    A change row is `| CASE | field | `old` | `new` | ...`; anything else in the file is prose.
    """
    out = {}
    for line in text.splitlines():
        m = _CHANGE_ROW.match(line.strip())
        if m:
            key = (m.group(1), m.group(2))
            if key in out:
                raise ManifestError(f"CHANGELOG lists {key} twice")
            out[key] = (m.group(3), m.group(4))
    return out


def diff_findings(provisional, frozen, changes):
    """The original columns differ from the archive only where the CHANGELOG says, as it says."""
    out = []
    prov = {r["case_id"]: r for r in provisional}
    froz = {r["case_id"]: r for r in frozen}
    if list(prov) != list(froz):
        return ["case ids or order differ from the provisional archive"]
    for cid, p in prov.items():
        for col in ORIGINAL_COLUMNS:
            if col == "validation_status":
                continue
            old, new = p.get(col, ""), froz[cid].get(col, "")
            logged = changes.get((cid, col))
            if old != new and logged is None:
                out.append(f"{cid}.{col}: changed without a CHANGELOG row")
            elif logged is not None and logged != (old, new):
                out.append(f"{cid}.{col}: CHANGELOG says {logged}, files say {(old, new)}")
    for cid, col in changes:
        if col not in ORIGINAL_COLUMNS or cid not in prov:
            out.append(f"CHANGELOG row {cid}.{col} names no original cell")
    return out


# --- validate and freeze --------------------------------------------------------------------

def validate(dataset):
    """Every check over the dataset directory. Returns (rows, findings)."""
    dataset = Path(dataset)
    header, rows = read_csv(dataset / "manifest.csv")
    findings = structure_findings(header, rows)
    findings += equivalence_findings(rows, read_json(dataset / "manifest.json"))
    findings += timestamp_findings(rows)
    findings += cluster_findings(rows)
    for r in rows:
        findings += case_findings(r)
    _, provisional = read_csv(dataset / "manifest.provisional.csv")
    findings += equivalence_findings(provisional, read_json(dataset / "manifest.provisional.json"))
    changes = parse_changelog((dataset / "CHANGELOG.md").read_text(encoding="utf-8"))
    findings += diff_findings(provisional, rows, changes)
    return rows, findings


def freeze_record(dataset, rows, tools, frozen_at):
    """FREEZE.json's content. `frozen_at` is passed in so a rerun can keep the first one."""
    dataset = Path(dataset)
    clusters = {}
    for r in rows:
        c = clusters.setdefault(r["source_cluster_id"], {
            "source_cluster_id": r["source_cluster_id"], "source_url": r["source_url"],
            "truth": r["truth"], "generator_or_source": r["generator_or_source"],
            "source_title": r["source_title"], "cases": [],
            "truth_evidence": {k: r[k] for k in (
                "truth_evidence_type", "truth_evidence_url", "truth_evidence_quote",
                "truth_evidence_accessed_at_utc", "truth_assertion_scope")},
        })
        c["cases"].append(r["case_id"])
    return {
        "schema": FREEZE_SCHEMA,
        "frozen_at_utc": frozen_at,
        "dataset_path": str(dataset),
        "sha256": {name: sha256_file(dataset / name) for name in (
            "manifest.csv", "manifest.json", "manifest.provisional.csv",
            "manifest.provisional.json", "CHANGELOG.md")},
        "counts": {
            "cases": len(rows),
            "truth": dict(sorted(Counter(r["truth"] for r in rows).items())),
            "generator_or_source": dict(sorted(Counter(r["generator_or_source"] for r in rows).items())),
            "source_clusters": len(clusters),
            "validation_status": dict(Counter(r["validation_status"] for r in rows)),
        },
        "tools": tools,
        "clips": [{k: r[k] for k in ("case_id", "source_cluster_id", "clip_start", "clip_end",
                                    "clip_sha256", "clip_format_id", "clip_resolution",
                                    "clip_fps", "clip_has_audio")} for r in rows],
        "source_clusters": list(clusters.values()),
        "limitations": [
            f"{len(rows)} cases come from {len(clusters)} source clusters; cases within a "
            "cluster are not independent observations.",
            f"Cluster-level results are not generalisable: there are only {len(clusters)} "
            "source clusters.",
            "The REAL controls are a single domain (NASA rocket/space footage).",
            "All material is a derivative of a public web upload (YouTube or a vendor CDN), "
            "not a camera or generator original.",
            "clip_sha256 identifies the clip that was validated; it is not a reproducibility "
            "guarantee, since a platform may re-encode or change formats.",
        ],
    }


def freeze(dataset, tools, now=None):
    """Validate, then write the frozen manifest and FREEZE.json — every case, or nothing.

    Idempotent: a rerun over an already frozen dataset rewrites identical bytes and keeps the
    original `frozen_at_utc`.
    """
    dataset = Path(dataset)
    rows, findings = validate(dataset)
    if findings:
        return findings
    for r in rows:
        r["validation_status"] = "VALIDATED"
    (dataset / "manifest.csv").write_text(render_csv(rows), encoding="utf-8")
    (dataset / "manifest.json").write_text(render_json(rows), encoding="utf-8")
    freeze_path = dataset / "FREEZE.json"
    frozen_at = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        candidate = freeze_record(dataset, rows, tools, previous.get("frozen_at_utc"))
        if candidate == previous:
            return []
    record = freeze_record(dataset, rows, tools, frozen_at)
    freeze_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return []


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--dataset", required=True, type=Path)
    f = sub.add_parser("freeze")
    f.add_argument("--dataset", required=True, type=Path)
    f.add_argument("--yt-dlp", required=True)
    f.add_argument("--ffmpeg", required=True)
    f.add_argument("--tool-image", required=True)
    args = p.parse_args(argv)
    if args.cmd == "validate":
        rows, findings = validate(args.dataset)
        for z in zero_gaps(rows):
            print(f"note: zero gap {z[0]} -> {z[1]}")
    else:
        findings = freeze(args.dataset, {"yt_dlp": args.yt_dlp, "ffmpeg": args.ffmpeg,
                                         "tool_image": args.tool_image})
    for line in findings:
        print(f"FAIL {line}")
    print(f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
