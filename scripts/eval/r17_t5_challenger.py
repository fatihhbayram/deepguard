"""R17-T5: measure the preregistered challengers C1/C2 offline on the sealed R17-T3 evidence.

    PYTHONPATH=scripts python3 scripts/eval/r17_t5_challenger.py \
        --spec-sha256 <recorded before any experiment> \
        --phase-a ~/deepguard-corpus/r17_t3_phase_a --results ~/deepguard-corpus/r17_t3_phase_b/results.json \
        --dataset ~/deepguard-corpus/r17-ai-visual-v1 \
        [--c3 ~/deepguard-corpus/r17_t5_diagnostic/c3_records.json] \
        --out scripts/eval/baselines/r17_t5_challenger_results.json

Applied exactly as `scripts/eval/specs/r17_t5_challenger_spec.md` froze it; nothing is decided
here. Every integrity check of spec §1 runs before a single case is read, and the spec file must
hash to the value recorded before execution. Any mismatch is a refusal.

Nothing here runs a detector or submits a clip, and nothing here is imported by `apps/*`: the
challengers exist only in this file. Thresholds and predicates are the committed constants of
`apps/api/app/risk_engine.py`, loaded by path and never changed. `--c3` only reads the isolated
diagnostic output `r17_t5_c3_diagnostic.py` wrote; it is kept in its own section, labelled
`diagnostic_revalidation`, and never merged with the sealed T3 evidence.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.r17_t2_run import sha256_file, verify_seal, write_json
from eval.r17_t2_score import LEVELS, confusion, metrics
from eval.r17_t4_analyze import (ANCHOR, ANCHOR_REL, CLUSTER_LABEL, EXPECTED_RULESET, REPO,
                                 Refusal, git, load_risk_engine, verify_t3_commit)

SPEC = REPO / "scripts/eval/specs/r17_t5_challenger_spec.md"
T4_RECORD = REPO / "scripts/eval/baselines/r17_t4_case_causes.json"
T4_RECORD_REL = "scripts/eval/baselines/r17_t4_case_causes.json"
T4_CLOSURE_COMMIT = "0c4f2cf9f75fa4ef33f5cebd6616eda71d98b47b"
SCHEMA = "r17-t5-challenger-results-1"

# Spec §2: the production constants, read from the module and refused if they ever differ.
EXPECTED_SVD_T_HIGH = 0.9550971388816833
EXPECTED_LIP_T_HIGH = 0.22962537594139576
FACELESS_REASON = "FaceDetectorNoFaceFound"
FACE_ERROR_FIELD = "face_manipulation.metadata.error"

DETECTED, NEGATIVE, INCONCLUSIVE = ("MANIPULATION_DETECTED", "NO_CALIBRATED_MANIPULATION_SIGNAL",
                                    "INCONCLUSIVE")
AUXILIARY = "MANIPULATION_DETECTED_AUXILIARY"
C3_PROVENANCE = "diagnostic_revalidation"
C3_POPULATION = 40
COUNTERFACTUAL_LABEL = "counterfactual — diagnostic_revalidation, not a measurement of production"

C1_PASS_MIN_GAIN = 4
C3_PASS_MIN_USABLE = 35
C3_FAIL_MIN_FORMAT = 10

LIMITATION = ("The 15 REAL controls are a single narrow technical domain (NASA Artemis II "
              "spacecraft cameras, 2 source clusters) with zero human faces. Zero false positives "
              "on this set is not general-population FPR or safety evidence. C2's lip-based FPR "
              "cannot be tested here at all. Cluster-level results come from 6 source clusters, "
              "which are not independent and not generalisable.")

SVD_FIELDS = ("provider", "signal_type", "status", "provider_version", "score", "total_clips")
FACE_FIELDS = ("provider", "signal_type", "status", "provider_version", "score", "frames_scored")
LIP_FIELDS = ("provider", "signal_type", "status", "provider_version", "score", "windows_scored")


# ---------------------------------------------------------------- integrity (spec §1)

def verify_t4_record(commit=T4_CLOSURE_COMMIT, record=T4_RECORD, record_rel=T4_RECORD_REL, run=git):
    """The T4 closure commit exists and the tracked record is byte-identical to the one it holds."""
    if run("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise Refusal(f"T4 closure commit {commit} is missing, refusing (S2)")
    committed = run("show", f"{commit}:{record_rel}")
    if committed.returncode != 0 or committed.stdout != Path(record).read_bytes():
        raise Refusal(f"tracked T4 record differs from {commit}:{record_rel}, refusing (S2)")
    return commit


def verify_inputs(anchor, phase_a, results, dataset, spec, spec_sha256):
    """Spec §1 steps 3-6. Returns the record of what was checked."""
    live_spec = sha256_file(spec)
    if live_spec != spec_sha256:
        raise Refusal(f"spec sha256 {live_spec} != recorded {spec_sha256}, refusing (S3)")
    ok, detail = verify_seal(phase_a, anchor["phase_a_seal_root"])
    if not ok:
        raise Refusal(f"Phase A seal does not verify against the tracked anchor: {detail} (S2)")
    live_results = sha256_file(results)
    if live_results != anchor["results_json_sha256"]:
        raise Refusal(f"results.json sha256 {live_results} != anchor, refusing (S2)")
    bad = {name: sha256_file(Path(dataset) / name) for name in ("manifest.json", "FREEZE.json")
           if sha256_file(Path(dataset) / name) != anchor["dataset_anchors"][name]}
    if bad:
        raise Refusal(f"dataset anchor mismatch {bad}, refusing (S2)")
    if anchor.get("ruleset") != EXPECTED_RULESET:
        raise Refusal(f"anchor ruleset {anchor.get('ruleset')} is not {EXPECTED_RULESET}")
    return {"spec_sha256": live_spec, "phase_a_seal_root": detail["root"],
            "results_json_sha256": live_results,
            "dataset_anchors": {n: anchor["dataset_anchors"][n] for n in ("manifest.json",
                                                                         "FREEZE.json")}}


def verify_all(phase_a, results, dataset, spec_sha256, spec=SPEC):
    anchor = json.loads(ANCHOR.read_text())
    record = {"t4_closure_commit": verify_t4_record(),
              "t3_closure_commit": verify_t3_commit(anchor=ANCHOR, anchor_rel=ANCHOR_REL)}
    record.update(verify_inputs(anchor, Path(phase_a), Path(results), Path(dataset), spec,
                                spec_sha256))
    record["t4_record_sha256"] = sha256_file(T4_RECORD)
    return record


def check_constants(re_mod):
    if re_mod.SVD_T_HIGH != EXPECTED_SVD_T_HIGH or re_mod.LIP_T_HIGH != EXPECTED_LIP_T_HIGH:
        raise Refusal(f"committed thresholds differ from the spec: SVD_T_HIGH {re_mod.SVD_T_HIGH}, "
                      f"LIP_T_HIGH {re_mod.LIP_T_HIGH} (S1)")


# ---------------------------------------------------------------- evidence (spec §2)

def evidence(cls, sig, fields):
    """The committed evidence type from a sealed signal; a missing field is None, as in the DB."""
    sig = sig or {}
    return cls(*(sig.get(k) for k in fields))


def svd_usable(re_mod, ev):
    return re_mod.is_eligible_svd(ev) and re_mod.is_usable_svd(ev)


def lip_hit(re_mod, ev):
    """No committed comparator exists for lip (v5 reads it nowhere), so `>=` is stated here."""
    return (re_mod.is_eligible_lip(ev) and re_mod.is_usable_lip(ev)
            and ev.score >= re_mod.LIP_T_HIGH)


def face_reasons(t4_record):
    """`face_manipulation.metadata.error` per case, from the T4 record (production_committed_row)."""
    out = {}
    for case in t4_record["cases"]:
        ptrs = list(case.get("primary_evidence", []))
        ptrs += [p for c in case.get("contributing_causes", []) for p in c["evidence"]]
        values = {p["value"] for p in ptrs if p["field"] == FACE_ERROR_FIELD}
        if len(values) > 1:
            raise Refusal(f"{case['case_id']}: conflicting face failure reasons {values}")
        out[case["case_id"]] = values.pop() if values else None
    return out


# ---------------------------------------------------------------- candidates (spec §3)

def c0(re_mod, svd, face):
    d = re_mod.evaluate_v5(svd=svd, face=face)
    return d.risk_level, d.rule_id


def c1(re_mod, c0_level, faceless, svd):
    if c0_level != INCONCLUSIVE or not (faceless and svd_usable(re_mod, svd)):
        return c0_level
    return DETECTED if re_mod.svd_threshold_reached(svd) else NEGATIVE


def c2(re_mod, x_level, lip):
    if x_level == INCONCLUSIVE and lip_hit(re_mod, lip):
        return AUXILIARY
    return x_level


def state(level):
    return "positive" if level == AUXILIARY else LEVELS[level]


def decide_case(re_mod, case, svd):
    """Every variant's verdict for one case, given the SVD evidence to use."""
    face = evidence(re_mod.FaceEvidence, case["face"], FACE_FIELDS)
    lip = evidence(re_mod.LipEvidence, case["lip"], LIP_FIELDS)
    level0, rule0 = c0(re_mod, svd, face)
    level1 = c1(re_mod, level0, case["faceless"], svd)
    return {"svd_usable": svd_usable(re_mod, svd), "lip_hit": lip_hit(re_mod, lip),
            "C0": level0, "C0_rule": rule0, "C1": level1,
            "C2@C0": c2(re_mod, level0, lip), "C2@C1": c2(re_mod, level1, lip)}


VARIANTS = ("C0", "C1", "C2@C0", "C2@C1")


def check_c1_structure(decisions):
    """Spec §3: C1 only ever turns an abstention into a negative. Refuse if the data disagrees."""
    for cid, d in decisions.items():
        if d["C1"] != d["C0"] and (d["C0"], d["C1"]) != (INCONCLUSIVE, NEGATIVE):
            raise Refusal(f"{cid}: C1 changed {d['C0']} to {d['C1']}, contradicting spec §3")


# ---------------------------------------------------------------- metrics (spec §4)

def variant_rows(cases, decisions, variant):
    rows = []
    for c in cases:
        level = decisions[c["case_id"]][variant]
        st = state(level)
        rows.append({"case_id": c["case_id"], "truth": c["truth"],
                     "source_cluster_id": c["source_cluster_id"], "final_state": st,
                     "confusion": confusion(c["truth"], st), "auxiliary": level == AUXILIARY})
    return rows


def level_metrics(rows, k):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["source_cluster_id"]].append(r)
    if len(clusters) != k:
        raise Refusal(f"{len(clusters)} distinct clusters != FREEZE.json K={k}")
    return {"case_level": metrics(rows),
            "cluster_level": {"label": CLUSTER_LABEL.format(k=k),
                              "clusters": {n: metrics(v) for n, v in sorted(clusters.items())}}}


def figures(cases, decisions):
    by_id = {c["case_id"]: c for c in cases}
    faceless = [c for c in cases if c["faceless"]]

    def decided(variant):
        return sum(state(decisions[c["case_id"]][variant]) in ("positive", "negative")
                   for c in faceless)

    def positives_on_real(variant, subset=None):
        return [cid for cid, d in sorted(decisions.items())
                if by_id[cid]["truth"] == "REAL" and state(d[variant]) == "positive"
                and (subset is None or cid in subset)]

    def tp(variant):
        return sum(by_id[cid]["truth"] == "AI" and state(d[variant]) == "positive"
                   for cid, d in decisions.items())

    usable_real = {c["case_id"] for c in faceless
                   if c["truth"] == "REAL" and decisions[c["case_id"]]["svd_usable"]}
    out = {
        "faceless_cases": len(faceless),
        "faceless_by_truth": {t: sum(c["truth"] == t for c in faceless) for t in ("AI", "REAL")},
        "faceless_decided": {v: decided(v) for v in ("C0", "C1")},
        "faceless_coverage_gain": decided("C1") - decided("C0"),
        "FP_C1_all_real": len(positives_on_real("C1")),
        "FP_C1_faceless_usable_real": len(positives_on_real("C1", usable_real)),
        "usable_real_denominator": len(usable_real),
        "usable_real_cases": sorted(usable_real),
        "c1_changed_cases": sorted(cid for cid, d in decisions.items() if d["C1"] != d["C0"]),
    }
    for x in ("C0", "C1"):
        aux = [cid for cid, d in decisions.items() if d[f"C2@{x}"] == AUXILIARY]
        out[f"c2_delta_tp@{x}"] = tp(f"C2@{x}") - tp(x)
        out[f"c2_aux_cases@{x}"] = sorted(aux)
        out[f"c2_aux_fp@{x}"] = sorted(cid for cid in aux if by_id[cid]["truth"] == "REAL")
    return out


# ---------------------------------------------------------------- decision rules (spec §5)

def c1_verdict(f):
    if f["usable_real_denominator"] == 0:
        return "INCONCLUSIVE"
    if f["FP_C1_all_real"] > 0:
        return "FAIL-SAFETY"
    if f["faceless_coverage_gain"] >= C1_PASS_MIN_GAIN:
        return "PASS"
    return "FAIL-BENEFIT"


def c3_verdict(usable, format_incompatible):
    if usable >= C3_PASS_MIN_USABLE:
        return "DIAGNOSTIC PASS"
    if format_incompatible >= C3_FAIL_MIN_FORMAT:
        return "DIAGNOSTIC FAIL"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------- inputs

def load_cases(phase_a, results_path, t4_record):
    rows = json.loads(Path(results_path).read_text())["cases"]
    reasons = face_reasons(t4_record)
    cases = []
    for r in rows:
        a = json.loads((Path(phase_a) / "cases" / r["blind_id"] / "analysis.json").read_text())
        if a.get("id") != r["analysis_id"]:
            raise Refusal(f"{r['blind_id']}: sealed analysis id differs from results.json")
        face = a.get("face_manipulation") or {}
        cases.append({
            "case_id": r["case_id"], "blind_id": r["blind_id"], "truth": r["truth"],
            "source_cluster_id": r["source_cluster_id"], "analysis_id": r["analysis_id"],
            "stored": (a.get("risk_level"), a.get("risk_rule_id")),
            "svd": a.get("synthetic_video"), "face": face, "lip": a.get("lip_forensics"),
            "face_reason": reasons.get(r["case_id"]),
            "faceless": face.get("status") == "FAILED" and reasons.get(r["case_id"]) == FACELESS_REASON,
        })
    return sorted(cases, key=lambda c: c["case_id"])


def evaluate(re_mod, cases, svd_for):
    decisions = {c["case_id"]: decide_case(re_mod, c, svd_for(c)) for c in cases}
    check_c1_structure(decisions)
    return decisions


def measure(cases, decisions, k):
    return {v: level_metrics(variant_rows(cases, decisions, v), k) for v in VARIANTS}


# ---------------------------------------------------------------- C3 (spec §3 C3, counterfactual)

def c3_section(re_mod, cases, c3_path, k, factual_metrics):
    data = json.loads(Path(c3_path).read_text())
    if data.get("provenance") != C3_PROVENANCE:
        raise Refusal(f"C3 input provenance {data.get('provenance')!r} is not {C3_PROVENANCE}")
    records = {r["case_id"]: r for r in data["records"]}
    population = [c["case_id"] for c in cases
                  if (c["svd"] or {}).get("status") in ("FAILED", "TIMEOUT")]
    if len(population) != C3_POPULATION or set(records) != set(population):
        raise Refusal(f"C3 population {len(population)} / records {len(records)} do not match "
                      f"the frozen {C3_POPULATION} sealed SVD failures")

    by_id = {c["case_id"]: c for c in cases}

    def svd_for(c):
        rec = records.get(c["case_id"])
        if rec is not None and rec["outcome"] == "usable_response":
            return re_mod.SvdEvidence(**rec["svd_evidence"])
        return evidence(re_mod.SvdEvidence, c["svd"], SVD_FIELDS)

    outcomes = defaultdict(int)
    for rec in records.values():
        outcomes[rec["outcome"]] += 1
    usable = outcomes["usable_response"]
    fmt = outcomes["input_format_incompatibility"]

    decisions = evaluate(re_mod, cases, svd_for)
    cf = measure(cases, decisions, k)
    deltas = {}
    for v in VARIANTS:
        a, b = factual_metrics[v]["case_level"], cf[v]["case_level"]
        deltas[v] = {"coverage_decided": b["coverage"]["k"] - a["coverage"]["k"],
                     "TP": b["TP"] - a["TP"], "FN": b["FN"] - a["FN"],
                     "FP": b["FP"] - a["FP"], "TN": b["TN"] - a["TN"],
                     "TPR_factual": a["TPR"], "TPR_counterfactual": b["TPR"]}
    return {
        "provenance": C3_PROVENANCE, "label": COUNTERFACTUAL_LABEL,
        "source": {"path": str(c3_path), "sha256": sha256_file(c3_path),
                   "run_started_at": data.get("run_started_at"),
                   "run_finished_at": data.get("run_finished_at"),
                   "nvidia_video_sha256": data.get("nvidia_video_sha256")},
        "population": len(population),
        "outcomes": dict(sorted(outcomes.items())),
        "usable": usable, "input_format_incompatibility": fmt,
        "verdict": c3_verdict(usable, fmt),
        "per_case": [{"case_id": cid, "truth": by_id[cid]["truth"],
                      "outcome": records[cid]["outcome"],
                      "grpc_code": records[cid].get("grpc_code"),
                      "score": (records[cid].get("svd_evidence") or {}).get("score"),
                      "counterfactual": {v: decisions[cid][v] for v in VARIANTS}}
                     for cid in sorted(population)],
        "counterfactual_figures": figures(cases, decisions),
        "counterfactual_metrics": cf,
        "delta_vs_factual_case_level": deltas,
        "promotion_eligible": False,
    }


# ---------------------------------------------------------------- driver

def run(args):
    verification = verify_all(args.phase_a, args.results, args.dataset, args.spec_sha256)
    re_mod = load_risk_engine()
    check_constants(re_mod)
    t4_record = json.loads(T4_RECORD.read_text())
    cases = load_cases(args.phase_a, args.results, t4_record)
    k = json.loads((Path(args.dataset) / "FREEZE.json").read_text())["counts"]["source_clusters"]

    def sealed_svd(c):
        return evidence(re_mod.SvdEvidence, c["svd"], SVD_FIELDS)

    decisions = evaluate(re_mod, cases, sealed_svd)
    mismatches = [c["case_id"] for c in cases
                  if (decisions[c["case_id"]]["C0"], decisions[c["case_id"]]["C0_rule"]) != c["stored"]]
    if mismatches:
        raise Refusal(f"C0 reproduction gate failed for {mismatches}")

    measured = measure(cases, decisions, k)
    f = figures(cases, decisions)
    out = {
        "schema": SCHEMA, "task": "R17-T5 Challenger Experiments", "ruleset": EXPECTED_RULESET,
        "dataset_limitation": LIMITATION,
        "verification": {**verification,
                         "c0_reproduction": {"n": len(cases), "matches": len(cases)}},
        "thresholds": {"SVD_T_HIGH": re_mod.SVD_T_HIGH, "LIP_T_HIGH": re_mod.LIP_T_HIGH,
                       "source": "apps/api/app/risk_engine.py (committed, unchanged)"},
        "figures": f,
        "verdicts": {
            "C1": {"verdict": c1_verdict(f),
                   "promotion": "eligible only for R17-T6 further promotion review; the 15 "
                                "faceless NASA REAL controls are not sufficient by themselves "
                                "for production promotion",
                   "note": "C1 adds no positive by construction (spec §3); every change is "
                           "abstention -> negative"},
            "C2": {"verdict": "DESCRIPTIVE ONLY", "promotion_eligible": False,
                   "delta_tp": {x: f[f"c2_delta_tp@{x}"] for x in ("C0", "C1")},
                   "aux_fp": {x: f[f"c2_aux_fp@{x}"] for x in ("C0", "C1")}},
        },
        "metrics": measured,
        "cases": [{"case_id": c["case_id"], "truth": c["truth"], "blind_id": c["blind_id"],
                   "source_cluster_id": c["source_cluster_id"],
                   "svd_status": (c["svd"] or {}).get("status"),
                   "face_status": c["face"].get("status"), "face_reason": c["face_reason"],
                   "face_reason_provenance": "production_committed_row (via T4 record)",
                   "lip_status": (c["lip"] or {}).get("status"), "faceless": c["faceless"],
                   **decisions[c["case_id"]]} for c in cases],
        "c3_diagnostic": None,
    }
    if args.c3:
        out["c3_diagnostic"] = c3_section(re_mod, cases, args.c3, k, measured)
        out["verdicts"]["C3"] = {"verdict": out["c3_diagnostic"]["verdict"],
                                 "promotion_eligible": False}
    write_json(args.out, out)
    print(json.dumps({"figures": f, "verdicts": out["verdicts"]}, indent=1, sort_keys=True))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spec-sha256", required=True)
    p.add_argument("--phase-a", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--c3")
    p.add_argument("--out", required=True)
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
