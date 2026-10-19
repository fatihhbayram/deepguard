"""R17-T4: attribute every R17-T3 baseline outcome to one frozen cause, offline and read-only.

    PYTHONPATH=scripts python3 scripts/eval/r17_t4_analyze.py classify \
        --taxonomy-sha256 <recorded before classification> \
        --phase-a ~/deepguard-corpus/r17_t3_phase_a --results ~/deepguard-corpus/r17_t3_phase_b/results.json \
        --dataset ~/deepguard-corpus/r17-ai-visual-v1 \
        --production-rows $WORK/production_rows.json --worker-log-coverage $WORK/worker_log_coverage.json \
        [--reproduction $WORK/reproduction.json] --out scripts/eval/baselines/r17_t4_case_causes.json

    PYTHONPATH=scripts python3 scripts/eval/r17_t4_analyze.py reproduce \
        --phase-a ~/deepguard-corpus/r17_t3_phase_a --results ... --out $WORK/reproduction.json

Applied exactly as `scripts/eval/specs/r17_t4_cause_taxonomy.md` froze it; nothing is decided
here. Before one case is read, every integrity check in taxonomy §1 must pass and the taxonomy
file must hash to the value recorded before classification. Any mismatch is a refusal.

Nothing here runs a detector, submits a clip or computes an alternative verdict. `reproduce`
(taxonomy §8, plan D2) runs the committed `evaluate_v5`, unchanged, on the sealed fields only.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval.r17_t2_run import sha256_file, verify_seal, write_json

REPO = Path(__file__).resolve().parents[2]
TAXONOMY = REPO / "scripts/eval/specs/r17_t4_cause_taxonomy.md"
ANCHOR = REPO / "scripts/eval/baselines/r17_t3_baseline_anchor.json"
ANCHOR_REL = "scripts/eval/baselines/r17_t3_baseline_anchor.json"
RISK_ENGINE = REPO / "apps/api/app/risk_engine.py"
T3_CLOSURE_COMMIT = "74e91e76f567d280947af882f5724e77a8a1cfbf"
EXPECTED_RULESET = "r9-v5.0.0"
SCHEMA = "r17-t4-case-causes-1"

SEALED = "sealed_t3_record"
PROD = "production_committed_row"
LOG = "worker_log_supplemental"
PROVENANCE = (SEALED, PROD, LOG)

# Taxonomy §5, in order. Classes 5 and 6 are kept although r9-v5.0.0 cannot reach them.
PRIMARY_ORDER = ("decision_contradicts_truth", "detector_input_ineligible",
                 "detector_runtime_failure", "rule_requires_missing_signal",
                 "rule_uncertain_band", "rule_conflicting_signals",
                 "unclassifiable_from_stored_evidence")
AGREES = "none_decision_agrees_with_truth"
INPUT_CAUSE, RUNTIME_CAUSE, UNCLASSIFIABLE = PRIMARY_ORDER[1], PRIMARY_ORDER[2], PRIMARY_ORDER[6]
ATTRIBUTIONS = ("attributable_to_detector_evidence", "attributable_to_aggregation_rule",
                "indeterminate")

# Taxonomy §3, read from `evaluate_v5`. All three rules read the same decision-eligible pair.
DECISION_PATH = {rule: ("synthetic_video", "face_manipulation")
                 for rule in ("R9-100", "R9-101", "R9-102", "R9-200", "R9-300", "R9-301")}
RULE_CONDITIONS = {
    "R9-100": "D_hits >= 2",
    "R9-101": "synthetic_video usable and score >= SVD_T_HIGH",
    "R9-102": "face_manipulation usable and score >= FACE_T_HIGH",
    "R9-200": "D_usable == D_TOTAL_V5 (2) and D_hits == 0",
    "R9-300": "0 < D_usable < D_TOTAL_V5 (2) and D_hits == 0",
    "R9-301": "D_usable == 0",
}
ABSTENTION_RULES = ("R9-300", "R9-301")
CANONICAL = {"R9-100": "AI", "R9-101": "AI", "R9-102": "AI", "R9-200": "REAL"}

# The six signals T3 recorded, plus the enrichment-only component production also wrote.
SIGNALS = {"synthetic_video": "nvidia", "face_manipulation": "efficientnet-b7",
           "lip_forensics": "lipforensics", "audio_authenticity": "aasist",
           "active_speaker": "nvidia", "provenance": "c2pa"}
ENRICHMENT_ONLY = {"face_forgery": "effort"}
FAILED_STATUSES = ("FAILED", "TIMEOUT")

# Taxonomy §4: the exception classes the committed detectors write into `metadata.error`.
INPUT_REASONS = frozenset({
    "FaceDetectorNoFaceFound", "FaceDetectorMediaError", "LipForensicsNoTrackedFace",
    "LipForensicsMediaError", "EffortNoFaceDetected", "EffortMediaError",
    "NvidiaActiveSpeakerInputError", "SpeakerDiarizationAudioError", "AudioDetectorAudioError"})
RUNTIME_REASONS = frozenset({
    "NvidiaProviderError", "NvidiaProviderUnavailable", "NvidiaAuthenticationError",
    "NvidiaProviderTimeout", "NvidiaProviderInvalidResponse", "NvidiaLocalFileError",
    "NvidiaActiveSpeakerProviderError", "NvidiaActiveSpeakerUnavailable",
    "NvidiaActiveSpeakerAuthenticationError", "NvidiaActiveSpeakerTimeout",
    "NvidiaActiveSpeakerInvalidResponse", "NvidiaActiveSpeakerLocalFileError",
    "FaceDetectorModelUnavailable", "FaceDetectorInferenceError", "LipForensicsModelUnavailable",
    "LipForensicsInferenceError", "EffortModelUnavailable", "EffortInferenceError",
    "SpeakerDiarizationUnavailable", "SpeakerDiarizationModelError",
    "AudioDetectorModelUnavailable", "AudioDetectorInferenceError", "AudioDetectorInputError"})

CLUSTER_LABEL = "{k} source clusters, not independent, not generalisable"


class Refusal(SystemExit):
    pass


def load_risk_engine(path=RISK_ENGINE):
    """The committed module, loaded by path so `app/__init__`'s configuration gate never runs.

    Its only imports are `math` and `dataclasses`; registering it in `sys.modules` first is what
    `dataclass` needs to resolve the module's annotations.
    """
    spec = importlib.util.spec_from_file_location("r17_t4_risk_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- integrity (taxonomy §1)

def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)


def verify_t3_commit(commit=T3_CLOSURE_COMMIT, anchor=ANCHOR, anchor_rel=ANCHOR_REL, run=git):
    """The closure commit exists and the tracked anchor is byte-identical to the one it holds."""
    if run("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise Refusal(f"T3 closure commit {commit} is missing, refusing (S2)")
    committed = run("show", f"{commit}:{anchor_rel}")
    if committed.returncode != 0 or committed.stdout != Path(anchor).read_bytes():
        raise Refusal(f"tracked T3 anchor differs from {commit}:{anchor_rel}, refusing (S2)")
    return commit


def verify_inputs(anchor, phase_a, results, dataset, taxonomy, taxonomy_sha256):
    """Everything taxonomy §1 requires, before a single case is read. Returns the record of it."""
    live_tax = sha256_file(taxonomy)
    if live_tax != taxonomy_sha256:
        raise Refusal(f"taxonomy sha256 {live_tax} != recorded {taxonomy_sha256}, refusing (S4)")
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
    return {"taxonomy_sha256": live_tax, "phase_a_seal_root": detail["root"],
            "results_json_sha256": live_results,
            "dataset_anchors": {n: anchor["dataset_anchors"][n] for n in ("manifest.json",
                                                                         "FREEZE.json")}}


# ---------------------------------------------------------------- evidence

def pointer(analysis_id, provenance, source, field, value):
    if provenance not in PROVENANCE:
        raise ValueError(f"unknown provenance {provenance!r}")
    return {"analysis_id": analysis_id, "provenance": provenance, "source": source,
            "field": field, "value": value}


def reason_category(status, error):
    """Taxonomy §4. A TIMEOUT is runtime whatever it carries; an unmapped class is unknown."""
    if status == "TIMEOUT":
        return "runtime"
    if error in INPUT_REASONS:
        return "input"
    if error in RUNTIME_REASONS:
        return "runtime"
    return "unknown"


def usability(re_mod, name, sig):
    """Taxonomy §3, through the committed `is_eligible_*`/`is_usable_*` predicates.

    None when the sealed record is missing a field the predicate needs: that is `indeterminate`,
    not unusable.
    """
    count_field = {"synthetic_video": "total_clips", "face_manipulation": "frames_scored"}[name]
    needed = ("provider", "signal_type", "status", "provider_version", "score", count_field)
    if sig is None or "status" not in sig:
        return None
    # A status other than SUCCESS fails `is_eligible_*` whatever the other fields hold.
    if sig["status"] != re_mod.SIGNAL_STATUS_SUCCESS:
        return False
    if any(k not in sig for k in needed):
        return None
    if name == "synthetic_video":
        ev = re_mod.SvdEvidence(*(sig[k] for k in needed))
        return re_mod.is_eligible_svd(ev) and re_mod.is_usable_svd(ev)
    ev = re_mod.FaceEvidence(*(sig[k] for k in needed))
    return re_mod.is_eligible_face(ev) and re_mod.is_usable_face(ev)


def build_case(row, analysis, prod_rows, log_coverage, manifest_row):
    """One case's normalised inputs, every value tagged with where it came from."""
    aid = row["analysis_id"]
    sealed_src = f"r17_t3_phase_a/cases/{row['blind_id']}/analysis.json"
    signals = {}
    for name in SIGNALS:
        sig = analysis.get(name)
        signals[name] = None if sig is None else {
            k: sig.get(k) for k in ("provider", "signal_type", "status", "provider_version",
                                    "score", "total_clips", "frames_scored") if k in sig}
    reasons = {}
    for name, provider in {**SIGNALS, **ENRICHMENT_ONLY}.items():
        match = [r for r in prod_rows if r["signal_type"] == name and r["provider"] == provider]
        reasons[name] = None if len(match) != 1 else {"status": match[0]["status"],
                                                       "error": match[0]["error"]}
    media = analysis.get("media") or {}
    return {
        "case_id": row["case_id"], "truth": row["truth"], "blind_id": row["blind_id"],
        "analysis_id": aid, "source_cluster_id": row["source_cluster_id"],
        "generator_or_source": manifest_row["generator_or_source"],
        "clip_has_audio": manifest_row.get("clip_has_audio"),
        "risk_level": analysis.get("risk_level"), "rule_id": analysis.get("risk_rule_id"),
        "rules_version": analysis.get("risk_rules_version"),
        "created_at": analysis.get("created_at"),
        "media": {k: media.get(k) for k in ("width", "height", "frame_rate", "duration")},
        "signals": signals, "reasons": reasons, "sealed_source": sealed_src,
        "worker_log": {"available": bool(log_coverage and log_coverage.get("covers_t3_window"))},
    }


# ---------------------------------------------------------------- classification (taxonomy §5-§7)

def classify(case, re_mod):
    """Exactly one primary cause, its contributing causes and, for an abstention, an attribution."""
    aid, rule, src = case["analysis_id"], case["rule_id"], case["sealed_source"]
    rule_ptr = pointer(aid, SEALED, src, "risk_rule_id", rule)
    path = DECISION_PATH.get(rule)
    out = {"primary_cause": None, "primary_evidence": [], "rests_on_non_sealed_evidence": False,
           "contributing_causes": [], "decision_path": list(path or ()), "fired_rule": rule,
           "rule_condition": RULE_CONDITIONS.get(rule), "attribution": None,
           "decision_path_signals": {}}
    if case["rules_version"] != EXPECTED_RULESET or path is None:
        out["primary_cause"] = UNCLASSIFIABLE
        out["primary_evidence"] = [rule_ptr, pointer(aid, SEALED, src, "risk_rules_version",
                                                     case["rules_version"])]
        return out

    failed = {}
    for name in (*SIGNALS, *ENRICHMENT_ONLY):
        sealed = case["signals"].get(name)
        reason = case["reasons"].get(name)
        status = sealed["status"] if sealed else (reason or {}).get("status")
        if status not in FAILED_STATUSES:
            continue
        ptrs = []
        if sealed:
            ptrs.append(pointer(aid, SEALED, src, f"{name}.status", status))
        if reason is None:
            category = "unknown"
        else:
            category = reason_category(status, reason["error"])
            ptrs.append(pointer(aid, PROD, "analysis_signals", f"{name}.metadata.error",
                                reason["error"]))
            if not sealed:
                ptrs.append(pointer(aid, PROD, "analysis_signals", f"{name}.status",
                                    reason["status"]))
        failed[name] = {"category": category, "evidence": ptrs, "on_decision_path": name in path}

    for name in path:
        u = usability(re_mod, name, case["signals"].get(name))
        out["decision_path_signals"][name] = {
            "status": (case["signals"].get(name) or {}).get("status"), "usable": u,
            "evidence": [pointer(aid, SEALED, src, f"{name}.{k}", (case["signals"][name] or {}).get(k))
                         for k in ("status", "provider_version", "score")]
            if case["signals"].get(name) else []}

    if rule in CANONICAL:
        truth_ptr = pointer(aid, SEALED, "r17_t3_phase_b/results.json", "truth", case["truth"])
        level_ptr = pointer(aid, SEALED, src, "risk_level", case["risk_level"])
        out["primary_cause"] = ("decision_contradicts_truth" if CANONICAL[rule] != case["truth"]
                                else AGREES)
        out["primary_evidence"] = [truth_ptr, level_ptr, rule_ptr]
        used = set()
    else:
        on_path = {n: f for n, f in failed.items() if f["on_decision_path"]}
        usable = [d["usable"] for d in out["decision_path_signals"].values()]
        used = set()
        for cause, category in ((INPUT_CAUSE, "input"), (RUNTIME_CAUSE, "runtime")):
            hits = [n for n, f in on_path.items() if f["category"] == category]
            if hits:
                out["primary_cause"] = cause
                out["primary_evidence"] = [rule_ptr] + [p for n in hits
                                                        for p in on_path[n]["evidence"]]
                used = set(hits)
                break
        else:
            if any(u is True for u in usable) and any(u is False for u in usable):
                out["primary_cause"] = "rule_requires_missing_signal"
                out["primary_evidence"] = [rule_ptr] + [
                    p for d in out["decision_path_signals"].values() for p in d["evidence"]]
            else:
                # Every needed signal usable would be class 5 or 6, which r9-v5.0.0 has no band
                # or conflict rule for; anything else lacks the evidence a class needs.
                out["primary_cause"] = UNCLASSIFIABLE
                out["primary_evidence"] = [rule_ptr] + [
                    p for d in out["decision_path_signals"].values() for p in d["evidence"]]

        if None in usable:
            out["attribution"] = "indeterminate"
        elif False in usable:
            out["attribution"] = "attributable_to_detector_evidence"
        else:
            out["attribution"] = "attributable_to_aggregation_rule"

    out["rests_on_non_sealed_evidence"] = any(p["provenance"] != SEALED
                                              for p in out["primary_evidence"])
    for name, f in failed.items():
        if name in used:
            continue
        cause = {"input": INPUT_CAUSE, "runtime": RUNTIME_CAUSE}.get(f["category"], UNCLASSIFIABLE)
        out["contributing_causes"].append({"cause": cause, "signal": name,
                                           "on_decision_path": f["on_decision_path"],
                                           "evidence": f["evidence"]})
    out["supplemental_sources"] = {
        PROD: "present" if any(r is not None for r in case["reasons"].values()) else "absent",
        LOG: "present" if case["worker_log"]["available"] else "absent_or_rotated"}
    return out


# ---------------------------------------------------------------- inventory and breakdowns

def failure_inventory(cases, results):
    """Plan §6 step 3: every FAILED/TIMEOUT status of every signal, its reason and provenance."""
    inv = {}
    for name, provider in {**SIGNALS, **ENRICHMENT_ONLY}.items():
        rows = []
        for c, r in zip(cases, results):
            sealed = c["signals"].get(name)
            reason = c["reasons"].get(name) or {}
            status = sealed["status"] if sealed else reason.get("status")
            if status not in FAILED_STATUSES:
                continue
            rows.append({"case_id": c["case_id"], "truth": c["truth"],
                         "reason": reason.get("error"),
                         "category": reason_category(status, reason.get("error"))
                         if reason else "unknown",
                         "on_fired_rule_decision_path": name in r["decision_path"],
                         "status_provenance": SEALED if sealed else PROD,
                         "reason_provenance": PROD if reason else None})
        statuses = Counter(((c["signals"].get(name) or {}).get("status")
                            or (c["reasons"].get(name) or {}).get("status")) for c in cases)
        inv[name] = {
            "provider": provider, "sealed_in_t3_record": name in SIGNALS,
            "status_counts": dict(sorted(statuses.items())),
            "failed": len(rows),
            "by_reason": dict(sorted(Counter(f"{x['category']}:{x['reason']}" for x in rows).items())),
            "by_truth": dict(sorted(Counter(x["truth"] for x in rows).items())),
            "on_fired_rule_decision_path": sum(x["on_fired_rule_decision_path"] for x in rows),
            "cases": [x["case_id"] for x in rows],
        }
    return inv


def tally(items, key):
    return dict(sorted(Counter(key(i) for i in items).items()))


def breakdowns(cases, results, k):
    pairs = list(zip(cases, results))

    def group(field):
        g = defaultdict(list)
        for c, r in pairs:
            g[c[field]].append((c, r))
        return {name: {"cases": len(v), "AI": sum(c["truth"] == "AI" for c, _ in v),
                       "REAL": sum(c["truth"] == "REAL" for c, _ in v),
                       "rules": tally(v, lambda x: x[0]["rule_id"]),
                       "primary": tally(v, lambda x: x[1]["primary_cause"]),
                       "attribution": tally([x for x in v if x[1]["attribution"]],
                                            lambda x: x[1]["attribution"]),
                       "synthetic_video_status": tally(
                           v, lambda x: (x[0]["signals"]["synthetic_video"] or {}).get("status")),
                       "face_manipulation_status": tally(
                           v, lambda x: (x[0]["signals"]["face_manipulation"] or {}).get("status"))}
                for name, v in sorted(g.items())}

    by_provider = defaultdict(Counter)
    for c in cases:
        for name, provider in {**SIGNALS, **ENRICHMENT_ONLY}.items():
            status = ((c["signals"].get(name) or {}).get("status")
                      or (c["reasons"].get(name) or {}).get("status"))
            by_provider[provider][f"{name}:{status}"] += 1

    order = sorted(pairs, key=lambda x: x[0]["created_at"])
    svd_order = [{"order": i + 1, "blind_id": c["blind_id"], "case_id": c["case_id"],
                  "truth": c["truth"], "created_at": c["created_at"],
                  "synthetic_video_status": (c["signals"]["synthetic_video"] or {}).get("status"),
                  "reason": (c["reasons"]["synthetic_video"] or {}).get("error")}
                 for i, (c, _) in enumerate(order)]
    return {
        "by_truth": group("truth"),
        "by_generator_or_source": group("generator_or_source"),
        "by_cluster": {"label": CLUSTER_LABEL.format(k=k), "clusters": group("source_cluster_id")},
        "by_provider": {p: dict(sorted(v.items())) for p, v in sorted(by_provider.items())},
        "synthetic_video_by_submission_order": svd_order,
    }


# ---------------------------------------------------------------- reproduction (taxonomy §8)

CONTAINER_SNIPPET = r"""
import hashlib, importlib.util, json, sys
path = "/app/app/risk_engine.py"
spec = importlib.util.spec_from_file_location("risk_engine_r17t4", path)
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
out = {"risk_engine_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
       "python": sys.version.split()[0], "cases": {}}
for c in json.load(sys.stdin):
    d = m.evaluate_v5(svd=m.SvdEvidence(**c["svd"]), face=m.FaceEvidence(**c["face"]))
    out["cases"][c["analysis_id"]] = {"risk_level": d.risk_level, "rule_id": d.rule_id,
                                      "rules_version": d.rules_version}
print(json.dumps(out))
"""


def reproduction_payload(rows, analyses):
    payload = []
    for row in rows:
        a = analyses[row["blind_id"]]
        svd, face = a["synthetic_video"], a["face_manipulation"]
        payload.append({"analysis_id": row["analysis_id"],
                        "svd": {k: svd.get(k) for k in ("provider", "signal_type", "status",
                                                        "provider_version", "score", "total_clips")},
                        "face": {k: face.get(k) for k in ("provider", "signal_type", "status",
                                                          "provider_version", "score",
                                                          "frames_scored")}})
    return payload


def compare_reproduction(rows, analyses, reproduced, committed_sha):
    if reproduced["risk_engine_sha256"] != committed_sha:
        return {"exercised": False, "reason": "image risk_engine.py differs from the committed file",
                "image_sha256": reproduced["risk_engine_sha256"], "committed_sha256": committed_sha}
    mismatches = []
    for row in rows:
        a, got = analyses[row["blind_id"]], reproduced["cases"].get(row["analysis_id"])
        if got is None or (got["risk_level"], got["rule_id"]) != (a["risk_level"], a["risk_rule_id"]):
            mismatches.append({"case_id": row["case_id"], "stored": [a["risk_level"], a["risk_rule_id"]],
                               "reproduced": got and [got["risk_level"], got["rule_id"]]})
    return {"exercised": True, "risk_engine_sha256": committed_sha,
            "python": reproduced.get("python"), "n": len(rows),
            "matches": len(rows) - len(mismatches), "mismatches": mismatches}


# ---------------------------------------------------------------- drivers

def load_sealed(phase_a, results_path):
    results = json.loads(Path(results_path).read_text())
    rows = results["cases"]
    analyses = {r["blind_id"]: json.loads((Path(phase_a) / "cases" / r["blind_id"] /
                                           "analysis.json").read_text()) for r in rows}
    for r in rows:
        if analyses[r["blind_id"]].get("id") != r["analysis_id"]:
            raise Refusal(f"{r['blind_id']}: sealed analysis id differs from results.json")
    return rows, analyses


def summarise(cases, results):
    def count(items, key):
        return tally(items, key)

    contrib = [x for r in results for x in r["contributing_causes"]]
    abst = [r for r in results if r["attribution"]]
    return {
        "cases": len(results),
        "truth": count(cases, lambda c: c["truth"]),
        "primary": {p: sum(r["primary_cause"] == p for r in results)
                    for p in (*PRIMARY_ORDER, AGREES)},
        "primary_by_truth": {t: count([r for c, r in zip(cases, results) if c["truth"] == t],
                                      lambda r: r["primary_cause"]) for t in ("AI", "REAL")},
        "primary_resting_on_non_sealed_evidence": sum(r["rests_on_non_sealed_evidence"]
                                                      for r in results),
        "contributing": count(contrib, lambda x: f"{x['cause']}:{x['signal']}"),
        "contributing_total": len(contrib),
        "contributing_on_decision_path": sum(x["on_decision_path"] for x in contrib),
        "attribution": {a: sum(r["attribution"] == a for r in abst) for a in ATTRIBUTIONS},
        "abstentions": len(abst),
    }


def run_classify(args):
    anchor = json.loads(ANCHOR.read_text())
    t3_commit = verify_t3_commit()
    verification = verify_inputs(anchor, Path(args.phase_a), Path(args.results), Path(args.dataset),
                                 TAXONOMY, args.taxonomy_sha256)
    verification["t3_closure_commit"] = t3_commit
    re_mod = load_risk_engine()
    rows, analyses = load_sealed(args.phase_a, args.results)
    manifest = {m["case_id"]: m for m in json.loads((Path(args.dataset) / "manifest.json").read_text())}
    k = json.loads((Path(args.dataset) / "FREEZE.json").read_text())["counts"]["source_clusters"]

    prod = json.loads(Path(args.production_rows).read_text()) if args.production_rows else None
    by_aid = defaultdict(list)
    for r in (prod or {}).get("rows") or []:
        by_aid[r["analysis_id"]].append(r)
    ids = {r["analysis_id"] for r in rows}
    if set(by_aid) - ids:
        raise Refusal("production export carries analyses outside the T3 population")
    logs = json.loads(Path(args.worker_log_coverage).read_text()) if args.worker_log_coverage else None

    cases = [build_case(r, analyses[r["blind_id"]], by_aid.get(r["analysis_id"], []), logs,
                        manifest[r["case_id"]]) for r in rows]
    results = [classify(c, re_mod) for c in cases]
    if sum(r["primary_cause"] is not None for r in results) != len(rows):
        raise Refusal("a case left without a primary cause")

    reproduction = {"exercised": False, "reason": "no reproduction output supplied"}
    if args.reproduction:
        reproduction = compare_reproduction(rows, analyses,
                                            json.loads(Path(args.reproduction).read_text()),
                                            sha256_file(RISK_ENGINE))
        reproduction["output_sha256"] = sha256_file(args.reproduction)

    out = {
        "schema": SCHEMA, "task": "R17-T4 Failure & Aggregation Analysis",
        "ruleset": EXPECTED_RULESET, "verification": verification,
        "inputs": {
            "production_rows": None if prod is None else {
                "provenance": PROD, "sha256": sha256_file(args.production_rows),
                "query_time": prod.get("query_time"),
                "transaction_read_only": prod.get("transaction_read_only"),
                "rows": len(prod.get("rows") or [])},
            "worker_log_coverage": None if logs is None else {
                "provenance": LOG, "sha256": sha256_file(args.worker_log_coverage),
                "covers_t3_window": logs.get("covers_t3_window"),
                "checked_at": logs.get("checked_at")},
        },
        "decision_paths": {r: {"condition": RULE_CONDITIONS[r], "signals": list(DECISION_PATH[r])}
                           for r in ("R9-200", "R9-300", "R9-301")},
        "summary": summarise(cases, results),
        "failure_inventory": failure_inventory(cases, results),
        "breakdowns": breakdowns(cases, results, k),
        "reproduction": reproduction,
        "cases": [{**{key: c[key] for key in ("case_id", "truth", "blind_id", "analysis_id",
                                              "source_cluster_id", "generator_or_source",
                                              "risk_level", "rule_id")}, **r}
                  for c, r in sorted(zip(cases, results), key=lambda x: x[0]["case_id"])],
    }
    write_json(args.out, out)
    print(json.dumps(out["summary"], indent=1, sort_keys=True))


def run_reproduce(args):
    anchor = json.loads(ANCHOR.read_text())
    ok, detail = verify_seal(Path(args.phase_a), anchor["phase_a_seal_root"])
    if not ok or sha256_file(args.results) != anchor["results_json_sha256"]:
        raise Refusal(f"sealed inputs do not verify, refusing: {detail}")
    rows, analyses = load_sealed(args.phase_a, args.results)
    payload = json.dumps(reproduction_payload(rows, analyses))
    # Ephemeral, no dependencies started, no network use by the snippet: only the committed
    # function over the sealed fields. The running stack is not touched.
    proc = subprocess.run(["docker", "compose", "run", "--rm", "--no-deps", "-T",
                           "--entrypoint", "python", "api", "-I", "-c", CONTAINER_SNIPPET],
                          input=payload, capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        raise Refusal(f"reproduction container failed: {proc.stderr[-2000:]}")
    reproduced = json.loads(proc.stdout.strip().splitlines()[-1])
    write_json(args.out, reproduced)
    print(json.dumps(compare_reproduction(rows, analyses, reproduced, sha256_file(RISK_ENGINE)),
                     indent=1))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--taxonomy-sha256", required=True)
    c.add_argument("--phase-a", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--production-rows")
    c.add_argument("--worker-log-coverage")
    c.add_argument("--reproduction")
    c.add_argument("--out", required=True)
    r = sub.add_parser("reproduce")
    r.add_argument("--phase-a", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--out", required=True)
    return p


def main():
    args = build_parser().parse_args()
    {"classify": run_classify, "reproduce": run_reproduce}[args.cmd](args)


if __name__ == "__main__":
    main()
