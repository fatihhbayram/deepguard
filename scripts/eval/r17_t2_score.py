"""R17-T2 Phase B: verify the trust anchors and the seal, then apply the frozen mapping.

    PYTHONPATH=scripts python3 scripts/eval/r17_t2_score.py \
        --phase-a $SCRATCH/phase_a --expected-seal-root <root from the Phase A log> \
        --blind-map $SCRATCH/blind_map/blind_map.json --out $SCRATCH/phase_b

Applied exactly as `scripts/eval/specs/r17_t2_scoring_freeze.md` froze it; nothing is decided
here. The order of operations is the freeze's §6. Any failure aborts before one verdict is read:

1. the live `FREEZE.json` / `manifest.csv` / `manifest.json` vs the tracked digests;
2. the Phase A seal vs `--expected-seal-root` (never a value read from the sealed directory);
3. the spec hashes Phase A recorded in `environment.json` vs the spec files in the repository;
4. only then the records, the blind-map join and truth (from the anchored `manifest.json`).

`<K>` comes from `counts.source_clusters` in the anchored `FREEZE.json`, and is checked against
the distinct cluster ids. A `smoke` blind map (freeze §7) carries its own placeholder truth and
is labelled as a harness check, not a measurement.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from eval.r17_t2_extract import DIGESTS, check_anchors, sha256_file
from eval.r17_t2_run import FREEZE_DOC, SCHEMA, verify_seal
from eval.stats import clopper_pearson_lower, clopper_pearson_upper

EXPECTED_RULESET = "r9-v5.0.0"
LEVELS = {"MANIPULATION_DETECTED": "positive", "NO_CALIBRATED_MANIPULATION_SIGNAL": "negative",
          "INCONCLUSIVE": "abstention"}
STATES = ("hash_blocked", "operational_failure", "protocol_violation", "abstention", "positive",
          "negative")
FROZEN_COUNTS = {"cases": 45, "AI": 30, "REAL": 15}
SMOKE_LABEL = "SMOKE — harness verification, not a measurement"


class Refusal(SystemExit):
    pass


# ---------------------------------------------------------------- frozen mapping (freeze §2-§4)

def final_state(rec):
    """Exactly one state; the first matching rule wins (freeze §2)."""
    if rec is None:
        return "operational_failure"
    if rec.get("hash_blocked"):
        return "hash_blocked"
    if rec.get("status") != "completed" or rec.get("timeout") == "decision":
        return "operational_failure"
    if rec.get("risk_rules_version") != EXPECTED_RULESET or rec.get("risk_level") not in LEVELS:
        return "protocol_violation"
    return LEVELS[rec["risk_level"]]


def confusion(truth, state):
    if state not in ("positive", "negative"):
        return None
    if truth == "AI":
        return "TP" if state == "positive" else "FN"
    if truth == "REAL":
        return "FP" if state == "positive" else "TN"
    raise Refusal(f"unknown truth {truth!r}")


def interval(k, n):
    """Exact two-sided 95% Clopper-Pearson (each tail 0.975). A zero denominator is N/A (None)."""
    if n == 0:
        return {"k": k, "n": 0, "rate": None, "ci95": None}
    return {"k": k, "n": n, "rate": round(k / n, 4),
            "ci95": [round(clopper_pearson_lower(k, n, 0.975), 4),
                     round(clopper_pearson_upper(k, n, 0.975), 4)]}


def metrics(rows):
    st = Counter(r["final_state"] for r in rows)
    cf = Counter(r["confusion"] for r in rows if r["confusion"])
    n = len(rows)
    n_ai = sum(r["truth"] == "AI" for r in rows)
    n_real = sum(r["truth"] == "REAL" for r in rows)
    abst_ai = sum(r["truth"] == "AI" and r["final_state"] == "abstention" for r in rows)
    abst_real = sum(r["truth"] == "REAL" and r["final_state"] == "abstention" for r in rows)
    tp, fn, fp, tn = cf["TP"], cf["FN"], cf["FP"], cf["TN"]
    return {
        "cases": n, "AI": n_ai, "REAL": n_real,
        "states": {s: st[s] for s in STATES},
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "TPR": interval(tp, tp + fn), "FNR": interval(fn, tp + fn),
        "FPR": interval(fp, fp + tn), "TNR": interval(tn, fp + tn),
        "coverage": interval(tp + fn + fp + tn, n),
        "coverage_AI": interval(tp + fn, n_ai), "coverage_REAL": interval(fp + tn, n_real),
        "abstention_rate": interval(st["abstention"], n),
        "abstention_rate_AI": interval(abst_ai, n_ai), "abstention_rate_REAL": interval(abst_real, n_real),
        "operational_failure_rate": interval(st["operational_failure"], n),
        "protocol_violation_rate": interval(st["protocol_violation"], n),
        "hash_blocked_rate": interval(st["hash_blocked"], n),
        "secondary_recall_abstention_as_miss": {
            "label": "secondary, abstention-as-miss",
            **interval(tp, tp + fn + abst_ai)},
    }


def check_invariant(rows, counts):
    st = Counter(r["final_state"] for r in rows)
    total = sum(st[s] for s in STATES)
    per = {t: sum(1 for r in rows if r["truth"] == t and r["final_state"] in STATES) for t in ("AI", "REAL")}
    if total != counts["cases"] or per["AI"] != counts["AI"] or per["REAL"] != counts["REAL"]:
        raise Refusal(f"invariant broken: total {total}, AI {per['AI']}, REAL {per['REAL']} "
                      f"vs {counts}")


# ---------------------------------------------------------------- verification (freeze §6)

def verify_inputs(phase_a, expected_seal_root, dataset=None, digests_path=DIGESTS):
    dataset, anchors = check_anchors(dataset, digests_path)                       # 1
    ok, detail = verify_seal(phase_a, expected_seal_root)                         # 2
    if not ok:
        raise Refusal(f"Phase A seal refused: {json.dumps(detail)}")
    env = json.loads((phase_a / "environment.json").read_text())                  # 3
    repo = {"scoring_freeze": sha256_file(FREEZE_DOC), "result_schema": sha256_file(SCHEMA)}
    recorded = {k: env["spec_sha256"].get(k) for k in repo}
    if recorded != repo:
        raise Refusal(f"spec hashes differ: Phase A {recorded} vs repo {repo}")
    return dataset, anchors, detail, env


def load_records(phase_a, manifest_cases):
    recs = {}
    for c in manifest_cases:
        p = phase_a / "cases" / c["blind_id"] / "record.json"
        recs[c["blind_id"]] = json.loads(p.read_text()) if p.is_file() else None
    return recs


def join(phase_a, blind_map, dataset):
    """Blind ids → cases, truth and clusters, with every cross-check of freeze §1 and §6.4."""
    exe = json.loads((phase_a / "execution_manifest.json").read_text())["cases"]
    bm = json.loads(Path(blind_map).read_text())
    entries = {e["blind_id"]: e for e in bm["entries"]}
    if set(entries) != {c["blind_id"] for c in exe}:
        raise Refusal("blind map and execution manifest name different blind ids")
    freeze = json.loads((dataset / "FREEZE.json").read_text())
    if bm["population"] == "r17-dataset":
        rows = {r["case_id"]: r for r in json.loads((dataset / "manifest.json").read_text())}
        frozen = {c["case_id"]: c for c in freeze["clips"]}
        if {e["case_id"] for e in entries.values()} != set(rows):
            raise Refusal("blind map does not cover exactly the frozen cases")
        if any("truth" in e for e in entries.values()):
            raise Refusal("a dataset blind map must not carry truth; truth comes from manifest.json")
        order = sorted(entries.values(), key=lambda e: e["blind_id"])
        expect = sorted(rows, key=lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest())
        if [e["case_id"] for e in order] != expect:
            raise Refusal("blind ids do not follow the frozen sha256(case_id) order")
        cases = {}
        for e in entries.values():
            r = rows[e["case_id"]]
            if e["source_cluster_id"] != r["source_cluster_id"]:
                raise Refusal(f"{e['blind_id']}: cluster differs from manifest.json")
            cases[e["blind_id"]] = {"case_id": e["case_id"], "truth": r["truth"],
                                    "source_cluster_id": r["source_cluster_id"],
                                    "generator_or_source": r["generator_or_source"],
                                    "frozen_sha256": frozen[e["case_id"]]["clip_sha256"]}
        k = freeze["counts"]["source_clusters"]
        counts = {"cases": freeze["counts"]["cases"], **freeze["counts"]["truth"]}
        if counts != FROZEN_COUNTS:
            raise Refusal(f"FREEZE.json counts {counts} differ from the frozen 45/30/15")
        k_source = "FREEZE.json counts.source_clusters"
    elif bm["population"] == "smoke":
        if {e["case_id"] for e in entries.values()} & {c["case_id"] for c in freeze["clips"]}:
            raise Refusal("a smoke case id is an R17 case id")
        cases = {b: {"case_id": e["case_id"], "truth": e["truth"],
                     "source_cluster_id": e["source_cluster_id"], "generator_or_source": None,
                     "frozen_sha256": e["clip_sha256"]} for b, e in entries.items()}
        k = len({e["source_cluster_id"] for e in entries.values()})
        counts = {"cases": len(cases), "AI": sum(c["truth"] == "AI" for c in cases.values()),
                  "REAL": sum(c["truth"] == "REAL" for c in cases.values())}
        k_source = "distinct smoke clusters (freeze §7)"
    else:
        raise Refusal(f"unknown population {bm['population']!r}")
    for c in exe:
        if c["clip_sha256"] != cases[c["blind_id"]]["frozen_sha256"]:
            raise Refusal(f"{c['blind_id']}: execution manifest hash differs from the frozen clip hash")
    if len({c["source_cluster_id"] for c in cases.values()}) != k:
        raise Refusal(f"K={k} ({k_source}) differs from the distinct cluster ids")
    return bm["population"], exe, cases, counts, k, k_source


def row(case, blind_id, rec, signals):
    state = final_state(rec)
    rec = rec or {}
    out = {"case_id": case["case_id"], "source_cluster_id": case["source_cluster_id"],
           "truth": case["truth"], "blind_id": blind_id}
    for k in ("analysis_id", "status", "risk_level", "risk_rule_id", "risk_rules_version",
              "risk_calibration_id"):
        out[k] = rec.get(k)
    out["final_state"] = state
    out["confusion"] = confusion(case["truth"], state)
    sig = rec.get("signals") or {}
    for s in signals:
        v = sig.get(s) or {}
        out[f"signals_{s}_status"] = v.get("status")
        out[f"signals_{s}_score"] = v.get("score")
    out["enrichment_state"] = rec.get("aggregate_enrichment_state")
    out["decision_latency_s"] = rec.get("seconds_to_decision")
    out["error"] = rec.get("error") if rec else "no record"
    return out


def check_schema(rows, schema):
    names = [c["name"] for c in schema["columns"]]
    types = {"string": str, "number": (int, float)}
    for r in rows:
        if list(r) != names:
            raise Refusal(f"row columns differ from the frozen schema: {list(r)}")
        for c in schema["columns"]:
            v = r[c["name"]]
            if v is None:
                if not c["nullable"]:
                    raise Refusal(f"{c['name']} is null")
                continue
            if not isinstance(v, types[c["type"]]) or isinstance(v, bool):
                raise Refusal(f"{c['name']}={v!r} is not {c['type']}")
            if "enum" in c and v not in c["enum"]:
                raise Refusal(f"{c['name']}={v!r} not in {c['enum']}")
    return names


def score(phase_a, expected_seal_root, blind_map, dataset=None, digests_path=DIGESTS):
    phase_a = Path(phase_a)
    dataset, anchors, seal, env = verify_inputs(phase_a, expected_seal_root, dataset, digests_path)
    population, exe, cases, counts, k, k_source = join(phase_a, blind_map, dataset)
    schema = json.loads(SCHEMA.read_text())
    recs = load_records(phase_a, exe)
    rows = [row(cases[c["blind_id"]], c["blind_id"], recs[c["blind_id"]], schema["signals"])
            for c in exe]
    rows.sort(key=lambda r: r["case_id"])
    check_invariant(rows, counts)
    names = check_schema(rows, schema)
    clusters = sorted({r["source_cluster_id"] for r in rows})
    result = {
        "schema": "r17-t2-results-1", "population": population,
        "label": SMOKE_LABEL if population == "smoke" else None,
        "verification": {"anchors": anchors, "seal_root": seal["root"],
                         "expected_seal_root": expected_seal_root,
                         "spec_sha256": env["spec_sha256"], "repo_head_at_phase_a": env["repo_head"],
                         "specs_dirty_at_phase_a": env["specs_dirty"]},
        "case_level": metrics(rows),
        "cluster_level": {
            "label": f"{k} source clusters, not independent, not generalisable",
            "K": k, "K_source": k_source,
            "clusters": {cl: metrics([r for r in rows if r["source_cluster_id"] == cl]) for cl in clusters},
        },
        "descriptive_by_generator_or_source": None if population == "smoke" else {
            g: metrics([r for r in rows if cases[r["blind_id"]]["generator_or_source"] == g])
            for g in sorted({c["generator_or_source"] for c in cases.values()})},
        "cases": rows,
    }
    return result, names


def write_outputs(result, names, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for r in result["cases"]:
            w.writerow(["N/A" if r[n] is None else r[n] for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", required=True)
    ap.add_argument("--expected-seal-root", required=True)
    ap.add_argument("--blind-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    phase_a, out = Path(args.phase_a).expanduser().resolve(), Path(args.out).expanduser().resolve()
    if out == phase_a or phase_a in out.parents:
        raise SystemExit("--out must lie outside the sealed Phase A directory")
    result, names = score(phase_a, args.expected_seal_root, Path(args.blind_map).expanduser(),
                          args.dataset)
    write_outputs(result, names, out)
    print(json.dumps({"population": result["population"], "label": result["label"],
                      "verification": result["verification"],
                      "case_level_states": result["case_level"]["states"],
                      "cluster_label": result["cluster_level"]["label"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
