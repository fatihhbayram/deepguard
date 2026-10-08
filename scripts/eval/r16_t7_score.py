"""R16-T7 Phase B: score the sealed Phase A snapshot against the unsealed ground truth.

    PYTHONPATH=scripts python3 scripts/eval/r16_t7_score.py \
        --phase-a ~/deepguard-corpus/r16_t7_phase_a2 \
        --gt ~/deepguard-corpus/r16_pilot/sealed/pilot_ground_truth.json \
        --t6-seal ~/deepguard-corpus/r16_pilot/SEAL.json \
        --out ~/deepguard-corpus/r16_t7_phase_b/scores.json

Written after the seal and applied exactly as `docs/ai/r16_t7_scoring_freeze.md` froze it. The
canonical mapping (§2), the confusion counts (§3), the lineage-group worst-case rule (§4), the
strata roll-up (§5), report completeness (§6) and the latency rules (§7, A1, A2) are restated
here as code, not re-decided. Before reading a single verdict, the script refuses to run unless
the Phase A seal reproduces and the ground truth hashes to the value T6 sealed.

Everything not named in the freeze is descriptive and labelled as such in the output: a
breakdown by encode path, and the per-signal status counts. No rule is derived from it.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from eval.stats import clopper_pearson_lower, clopper_pearson_upper

EXPECTED_RULESET = "r9-v5.0.0"
POSITIVE = "MANIPULATION_DETECTED"
NEGATIVE = "NO_CALIBRATED_MANIPULATION_SIGNAL"
ABSTAIN = "INCONCLUSIVE"
SIGNALS = ("synthetic_video", "face_manipulation", "lip_forensics", "audio_authenticity",
           "active_speaker", "provenance")
TERMINAL_ENRICHMENT = {"ENRICHMENT_COMPLETE", "ENRICHMENT_PARTIAL", "ENRICHMENT_FAILED"}
ROLLUP = {"genuine_original": "genuine original", "genuine_derivative": "genuine derivative",
          "manipulated_edit": "non-AI manipulation", "manipulated_audio_tts": "TTS manipulation"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_seal(phase_a):
    seal = json.loads((phase_a / "PHASE_A_SEAL.json").read_text())
    bad = [k for k, v in seal["per_file_sha256"].items() if sha(phase_a / k) != v]
    root = hashlib.sha256(json.dumps(seal["per_file_sha256"], sort_keys=True).encode()).hexdigest()
    if bad or root != seal["snapshot_sha256"]:
        raise SystemExit(f"Phase A seal does not reproduce: {bad[:5]}")
    return seal["snapshot_sha256"]


def interval(k, n):
    """Exact two-sided 95% Clopper-Pearson (each tail at 0.975). Never a bare 0%."""
    if n == 0:
        return {"k": k, "n": 0, "rate": None, "ci95": None}
    return {"k": k, "n": n, "rate": round(k / n, 4),
            "ci95": [round(clopper_pearson_lower(k, n, 0.975) or 0.0, 4),
                     round(clopper_pearson_upper(k, n, 0.975) or 1.0, 4)]}


def classify(rec):
    """§2: operational failure, protocol violation, or the canonical bucket."""
    if rec.get("outcome_operational") or rec.get("status") != "completed" or rec.get("timeout") == "decision":
        return "operational_failure"
    if rec.get("risk_rules_version") != EXPECTED_RULESET:
        return "protocol_violation"
    return {POSITIVE: "positive", NEGATIVE: "negative", ABSTAIN: "abstention"}.get(
        rec.get("risk_level"), "protocol_violation")


def confusion(truth, cls):
    if truth == "unknown_truth" or cls not in ("positive", "negative"):
        return None
    if truth == "manipulated":
        return "TP" if cls == "positive" else "FN"
    return "FP" if cls == "positive" else "TN"


def complete(rec):
    """§6 report completeness, five conditions, each reported separately."""
    render = rec.get("render") or {}
    conds = {
        "1_completed": rec.get("status") == "completed",
        "2_verdict_traced": all(rec.get(k) for k in ("risk_level", "risk_rule_id",
                                                      "risk_rules_version", "risk_calibration_id")),
        "3_signals_explicit": all((rec.get("signals") or {}).get(s) is not None for s in SIGNALS),
        "4_enrichment_terminal": rec.get("aggregate_enrichment_state") in TERMINAL_ENRICHMENT,
        "5_render_en_tr": bool(render.get("ok")) and all(
            (render.get("locales") or {}).get(l, {}).get("ok") for l in ("en", "tr")),
    }
    traceable = conds["2_verdict_traced"] and rec.get("risk_trace_present") and \
        rec.get("risk_trace_rule_id") == rec.get("risk_rule_id")
    return conds, all(conds.values()), bool(traceable)


def summarise(rows):
    c = Counter(r["confusion"] for r in rows if r["confusion"])
    cls = Counter(r["class"] for r in rows)
    n = len(rows)
    decided = c["TP"] + c["FN"] + c["FP"] + c["TN"]
    return {
        "cases": n, "TP": c["TP"], "FN": c["FN"], "FP": c["FP"], "TN": c["TN"],
        "abstention": cls["abstention"], "operational_failure": cls["operational_failure"],
        "protocol_violation": cls["protocol_violation"],
        "TPR": interval(c["TP"], c["TP"] + c["FN"]), "FNR": interval(c["FN"], c["TP"] + c["FN"]),
        "FPR": interval(c["FP"], c["FP"] + c["TN"]), "TNR": interval(c["TN"], c["FP"] + c["TN"]),
        "abstention_rate": interval(cls["abstention"], n),
        "operational_failure_rate": interval(cls["operational_failure"], n),
        "coverage": interval(decided, n),
    }


def group_outcome(rows, side):
    """§4 conservative worst case for one side of one lineage group."""
    scored = [r["confusion"] for r in rows]
    bad, good = ("FP", "TN") if side == "genuine" else ("FN", "TP")
    if bad in scored:
        return bad
    if good in scored:
        return good
    classes = {r["class"] for r in rows}
    return "operational_failure" if classes == {"operational_failure"} else "abstention"


def latency(rows):
    def stats(xs):
        xs = [x for x in xs if x is not None]
        return None if not xs else {"n": len(xs), "median": round(statistics.median(xs), 1),
                                     "max": round(max(xs), 1), "min": round(min(xs), 1)}
    wall = [r for r in rows if not r["wall_clock_excluded"]]
    return {
        "upload_seconds": stats(r["upload_seconds"] for r in wall),
        "seconds_to_decision": stats(r["seconds_to_decision"] for r in wall),
        "seconds_to_enrichment_terminal": stats(r["seconds_to_enrichment_terminal"] for r in wall),
        "worker_decision_processing_seconds_all_cases": stats(r["worker_seconds"] for r in rows),
        "wall_clock_excluded_cases": [r["case_id"] for r in rows if r["wall_clock_excluded"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-a", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--t6-seal", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    phase_a = Path(args.phase_a).expanduser()

    snapshot = verify_seal(phase_a)
    gt_sha = sha(Path(args.gt).expanduser())
    if gt_sha != json.loads(Path(args.t6_seal).expanduser().read_text())["gt_manifest_sha256"]:
        raise SystemExit("ground truth does not match the T6 seal")

    gt = {c["case_id"]: c for c in json.loads(Path(args.gt).expanduser().read_text())["cases"]}
    results = json.loads((phase_a / "results.json").read_text())["cases"]
    supplement = json.loads((phase_a / "supplement_R16P-001_worker_time.json").read_text())
    if set(gt) != {r["case_id"] for r in results}:
        raise SystemExit("case sets differ")

    rows = []
    for rec in results:
        g = gt[rec["case_id"]]
        cls = classify(rec)
        conds, is_complete, traceable = complete(rec)
        worker = rec.get("worker_decision_processing_seconds")
        if worker is None and rec["case_id"] == supplement["case_id"]:
            worker = supplement["worker_decision_processing_seconds"]
        rows.append({
            "case_id": rec["case_id"], "analysis_id": rec.get("analysis_id"),
            "truth": g["truth"], "stratum": g["stratum"], "rollup": ROLLUP[g["stratum"]],
            "family": (g.get("manipulation") or {}).get("family"),
            "encode_path": (g.get("manipulation") or {}).get("encode_path") or "original",
            "roots": g["lineage"]["root_sources"],
            "lineage_cluster": g["lineage"]["lineage_cluster"],
            "verdict": rec.get("risk_level"), "rule": rec.get("risk_rule_id"),
            "class": cls, "confusion": confusion(g["truth"], cls),
            "signals": rec.get("signals"), "enrichment": rec.get("aggregate_enrichment_state"),
            "completeness": conds, "complete": is_complete, "traceable": traceable,
            "wall_clock_excluded": bool(rec.get("wall_clock_latency_excluded")),
            "adopted": "attempt_1" if rec.get("adopted_from_attempt_1") else
                       "interrupted_runner" if rec.get("interrupted_runner_adopted") else None,
            "upload_seconds": rec.get("upload_seconds"),
            "seconds_to_decision": rec.get("seconds_to_decision"),
            "seconds_to_enrichment_terminal": rec.get("seconds_to_enrichment_terminal"),
            "worker_seconds": worker,
        })

    by = lambda key: {k: summarise([r for r in rows if r[key] == k])
                      for k in sorted({r[key] for r in rows})}

    groups = defaultdict(list)
    for r in rows:
        for root in r["roots"]:
            groups[root].append(r)
    group_rows = {}
    for root, rs in sorted(groups.items()):
        gen = [r for r in rs if r["truth"] == "genuine"]
        man = [r for r in rs if r["truth"] == "manipulated"]
        group_rows[root] = {
            "cases": [r["case_id"] for r in rs],
            "genuine_side": group_outcome(gen, "genuine") if gen else None,
            "manipulated_side": group_outcome(man, "manipulated") if man else None,
        }
    gc = Counter(v["genuine_side"] for v in group_rows.values() if v["genuine_side"])
    mc = Counter(v["manipulated_side"] for v in group_rows.values() if v["manipulated_side"])
    group_summary = {
        "groups": len(group_rows),
        "genuine_side": dict(gc), "manipulated_side": dict(mc),
        "group_FPR": interval(gc["FP"], gc["FP"] + gc["TN"]),
        "group_TPR": interval(mc["TP"], mc["TP"] + mc["FN"]),
        "group_genuine_abstained": interval(gc["abstention"] + gc["operational_failure"],
                                            sum(gc.values())),
        "group_manipulated_abstained": interval(mc["abstention"] + mc["operational_failure"],
                                                sum(mc.values())),
        "multi_root_cases": {r["case_id"]: r["roots"] for r in rows if len(r["roots"]) > 1},
        "owner_level_n": 1,
    }

    # The same §4 worst-case outcome rule, applied to T6's own sealed `lineage_cluster` field
    # (8 disjoint clusters: the multi-root A/V-swap and splice join their two roots). The frozen
    # primary grouping above is per root with overlap (§4); this is the T6 unit of independence
    # (`scoring_rules.unit_of_independence`), reported beside it, never instead of it.
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["lineage_cluster"]].append(r)
    cluster_rows = {}
    for cl, rs in sorted(clusters.items()):
        gen = [r for r in rs if r["truth"] == "genuine"]
        man = [r for r in rs if r["truth"] == "manipulated"]
        cluster_rows[cl] = {
            "cases": [r["case_id"] for r in rs],
            "genuine_side": group_outcome(gen, "genuine") if gen else None,
            "manipulated_side": group_outcome(man, "manipulated") if man else None,
        }
    kc = Counter(v["genuine_side"] for v in cluster_rows.values() if v["genuine_side"])
    km = Counter(v["manipulated_side"] for v in cluster_rows.values() if v["manipulated_side"])
    cluster_summary = {
        "clusters": len(cluster_rows),
        "clusters_with_genuine_side": sum(1 for v in cluster_rows.values() if v["genuine_side"]),
        "clusters_with_manipulated_side": sum(1 for v in cluster_rows.values() if v["manipulated_side"]),
        "genuine_side": dict(kc), "manipulated_side": dict(km),
        "cluster_FPR": interval(kc["FP"], kc["FP"] + kc["TN"]),
        "cluster_TPR": interval(km["TP"], km["TP"] + km["FN"]),
        "cluster_manipulated_abstained": interval(km["abstention"] + km["operational_failure"],
                                                  sum(km.values())),
    }

    sig = {s: dict(Counter((r["signals"] or {}).get(s) for r in rows)) for s in SIGNALS}
    out = {
        "schema": "r16-t7-scores-1",
        "phase_a_snapshot_sha256": snapshot, "ground_truth_sha256": gt_sha,
        "case_level": summarise(rows),
        "by_rollup_stratum": by("rollup"), "by_gt_stratum": by("stratum"),
        "lineage_groups": group_summary, "lineage_group_detail": group_rows,
        "t6_lineage_clusters": cluster_summary, "t6_lineage_cluster_detail": cluster_rows,
        "provider_failure": {
            s: interval(sum(1 for r in rows if (r["signals"] or {}).get(s) not in ("SUCCESS", None)),
                        len(rows)) for s in SIGNALS},
        "completeness": {
            "complete": interval(sum(r["complete"] for r in rows), len(rows)),
            "traceable": interval(sum(r["traceable"] for r in rows), len(rows)),
            "per_condition": {k: sum(r["completeness"][k] for r in rows)
                              for k in rows[0]["completeness"]},
        },
        "latency": latency(rows),
        "descriptive_not_frozen": {
            "by_encode_path": by("encode_path"),
            "signal_status": sig,
            "verdict_by_rule": dict(Counter(f"{r['verdict']} / {r['rule']}" for r in rows)),
        },
        "cases": rows,
    }
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).expanduser().write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("case_level", "lineage_groups", "t6_lineage_clusters",
                                          "provider_failure")},
                     indent=1))


if __name__ == "__main__":
    main()
