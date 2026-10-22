"""R18-T4 characterization: DEV-only guard, HEAD anchor introspection, zero-FP comparator, fixed-FPR semantics, cluster bootstrap, coverage accounting, case states, determinism (synthetic inputs only)."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval import r18_t3_freeze_split as t3
from eval import r18_t4_characterize as ch

MODULE_PATH = Path(ch.__file__)
ANCHORS = {
    "commit_sha": "0" * 40, "ruleset_version": "r9-v5.0.0", "svd_t_high": 0.9,
    "face_t_high": 0.95, "lip_t_high": 0.3, "svd_provider_version": "svd-v",
    "face_provider_version": "face-v", "lip_provider_version": "lip-v",
    "source": "test", "risk_engine_head_sha256": "x", "risk_engine_worktree_matches_head": True,
}


def sha(tag):
    return hashlib.sha256(str(tag).encode()).hexdigest()


# --- partition isolation ----------------------------------------------------------------------

@pytest.mark.parametrize("split", ["VAL", "LOCKED_TEST", "TEST", "dev"])
def test_load_dev_partition_rejects_every_non_dev_split(split, tmp_path):
    missing = tmp_path / "never_opened.json"
    with pytest.raises(ch.PartitionViolationError):
        ch.load_dev_partition(missing, split)


def test_load_dev_partition_refuses_before_opening_the_freeze(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(t3, "load_partition", lambda *a, **k: called.append(a) or [])
    with pytest.raises(ch.PartitionViolationError):
        ch.load_dev_partition(tmp_path / "f.json", "LOCKED_TEST")
    assert called == []


def test_load_dev_partition_asks_t3_for_dev_only(monkeypatch):
    seen = []
    monkeypatch.setattr(t3, "load_partition",
                        lambda path, split, purpose=None: seen.append((split, purpose))
                        or [{"split": "DEV", "sha256": "a"}])
    assert ch.load_dev_partition("x") == [{"split": "DEV", "sha256": "a"}]
    assert seen == [("DEV", "R18-T4")]


def test_load_dev_partition_rejects_a_leaked_non_dev_record(monkeypatch):
    monkeypatch.setattr(t3, "load_partition",
                        lambda *a, **k: [{"split": "DEV"}, {"split": "VAL"}])
    with pytest.raises(ch.PartitionViolationError):
        ch.load_dev_partition("x")


def test_frozen_dev_partition_matches_the_t3_freeze():
    dev = ch.load_dev_partition()
    assert len(dev) == 155
    assert sum(r["ground_truth"] == t3.VERIFIED_AI for r in dev) == 72
    assert {r["split"] for r in dev} == {"DEV"}


def test_module_never_names_a_non_dev_split_for_loading():
    tree = ast.parse(MODULE_PATH.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "load_partition"]
    assert calls, "load_partition must be the only route to cases"
    for call in calls:
        assert ast.unparse(call.args[1]) == "t3.DEV"


# --- anchor introspection ---------------------------------------------------------------------

RISK_SRC = '''
import math
RULES_VERSION_V5 = "r9-v5.0.0"
SVD_PROVIDER_VERSION = "847b"
SVD_T_HIGH = 0.9550971388816833
FACE_PROVIDER_VERSION = (
    "repo@rev"
)
FACE_T_HIGH = 0.9867589175701141
LIP_PROVIDER_VERSION = ("https://x" "@abc" "+def")
LIP_T_HIGH = 0.22962537594139576
def f():
    SVD_T_HIGH = 1.0
'''


def fake_git(source, head="a" * 40, fail=None):
    def run(*args):
        if args[0] == fail:
            return SimpleNamespace(returncode=1, stdout="")
        if args[0] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=head + "\n")
        if args[0] == "show":
            return SimpleNamespace(returncode=0, stdout=source)
        raise AssertionError(args)
    return run


def test_introspection_reads_literals_from_head_source():
    a = ch.introspect_anchors(run=fake_git(RISK_SRC))
    assert a["ruleset_version"] == "r9-v5.0.0"
    assert a["svd_t_high"] == 0.9550971388816833
    assert a["face_t_high"] == 0.9867589175701141
    assert a["lip_t_high"] == 0.22962537594139576
    assert a["lip_provider_version"] == "https://x@abc+def"
    assert a["commit_sha"] == "a" * 40


def test_introspection_ignores_function_scoped_assignments():
    consts = ch.module_constants(RISK_SRC)
    assert consts["SVD_T_HIGH"] == 0.9550971388816833


def test_introspection_refuses_when_an_anchor_is_missing():
    with pytest.raises(ch.Refusal):
        ch.introspect_anchors(run=fake_git(RISK_SRC.replace("LIP_T_HIGH = 0.2296", "X = 0.2")))


def test_introspection_refuses_without_git_head():
    with pytest.raises(ch.Refusal):
        ch.introspect_anchors(run=fake_git(RISK_SRC, fail="rev-parse"))


def test_introspection_executes_nothing():
    src = RISK_SRC + "\nraise SystemExit('executed')\nSIDE = open('/nonexistent')\n"
    a = ch.introspect_anchors(run=fake_git(src))
    assert a["svd_t_high"] == 0.9550971388816833


def test_real_head_anchors_match_the_committed_risk_engine():
    a = ch.introspect_anchors()
    assert a["ruleset_version"] == "r9-v5.0.0"
    assert a["risk_engine_worktree_matches_head"] is True
    for name in ("svd_t_high", "face_t_high", "lip_t_high"):
        assert isinstance(a[name], float)


def test_module_holds_no_threshold_literal():
    text = MODULE_PATH.read_text()
    for literal in ("0.9550971388816833", "0.9867589175701141", "0.22962537594139576"):
        assert literal not in text


# --- zero-FP comparator and fixed-FPR semantics -----------------------------------------------

def test_zero_fpr_threshold_is_strictly_above_max_real():
    real = [0.1, 0.5, 0.8]
    ai = [0.8, 0.9, 0.2]
    p = ch.fixed_fpr_point(ai, real, 0.0)
    assert p["threshold"] > 0.8
    assert p["threshold"] == float(np.nextafter(0.8, np.inf))
    assert p["false_positives"] == 0
    assert p["true_positives"] == 1  # the AI clip tied with max(REAL) is not counted


def test_threshold_at_max_real_would_have_produced_a_false_alarm():
    real = np.array([0.1, 0.5, 0.8])
    assert (real >= real.max()).sum() == 1
    assert (real >= ch.fixed_fpr_point([0.9], real, 0.0)["threshold"]).sum() == 0


def test_fixed_fpr_reports_achievable_step_fpr_not_target():
    real = list(np.linspace(0, 1, 83))
    ai = [0.99] * 10
    p1 = ch.fixed_fpr_point(ai, real, 0.01)
    assert p1["allowed_false_positives"] == 0
    assert p1["achievable_empirical_fpr"] == 0.0
    assert p1["usable_real_denominator"] == 83
    p5 = ch.fixed_fpr_point(ai, real, 0.05)
    assert p5["allowed_false_positives"] == 4
    assert p5["achievable_empirical_fpr"] == pytest.approx(4 / 83)
    assert p5["achievable_empirical_fpr"] <= 0.05


def test_fixed_fpr_with_tied_real_scores_never_exceeds_target():
    real = [0.9, 0.9, 0.9, 0.1] + [0.0] * 16
    p = ch.fixed_fpr_point([0.95, 0.9], real, 0.10)  # two FPs allowed, but three tie at 0.9
    assert p["false_positives"] == 0
    assert p["achievable_empirical_fpr"] == 0.0
    assert p["true_positives"] == 1


def test_interpolated_tpr_is_a_separate_linear_field():
    ai = [0.9, 0.7, 0.4]
    real = [0.8, 0.3]
    p = ch.fixed_fpr_point(ai, real, 0.25)
    # ROC upper envelope: FPR 0 -> TPR 1/3, FPR 0.5 -> TPR 1; halfway between is 2/3
    assert p["interpolated_roc_tpr"] == pytest.approx(2 / 3)
    assert p["tpr_at_achievable_fpr"] == pytest.approx(1 / 3)
    assert "linear" in p["interpolation"]


def test_interpolated_tpr_at_zero_equals_zero_fp_step_tpr():
    rng = np.random.default_rng(1)
    ai, real = rng.random(30), rng.random(40) * 0.9
    p = ch.fixed_fpr_point(ai, real, 0.0)
    assert p["interpolated_roc_tpr"] == pytest.approx(p["tpr_at_achievable_fpr"])


def test_no_fitted_calibration_model_in_module():
    text = MODULE_PATH.read_text().lower()
    for banned in ("isotonicregression", "sklearn", "logisticregression", "curve_fit"):
        assert banned not in text


def test_production_point_uses_greater_equal_comparator():
    p = ch.fixed_threshold_point([0.5, 0.4], [0.5, 0.1], 0.5)
    assert p["true_positives"] == 1 and p["false_positives"] == 1


# --- discrimination ---------------------------------------------------------------------------

def test_auroc_matches_pairwise_definition_with_ties():
    ai, real = [0.9, 0.5, 0.5], [0.5, 0.1]
    pairs = [(a > r) + 0.5 * (a == r) for a in ai for r in real]
    assert ch.auroc(ai, real) == pytest.approx(sum(pairs) / len(pairs))


def test_auroc_equals_trapezoid_area_of_roc():
    rng = np.random.default_rng(3)
    ai, real = np.round(rng.random(25), 1), np.round(rng.random(30) * 0.8, 1)
    pts = ch.roc_points(ai, real)
    x = [p["fpr"] for p in pts]
    y = [p["tpr"] for p in pts]
    assert ch.auroc(ai, real) == pytest.approx(np.trapezoid(y, x))


def test_auroc_undefined_for_constant_or_missing_class():
    assert ch.auroc([0.5, 0.5], [0.5]) is None
    assert ch.auroc([], [0.1]) is None


def test_auprc_perfect_and_prevalence_behaviour():
    assert ch.auprc([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)
    assert ch.auprc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_distribution_effect_sizes():
    sep = ch.separation([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    assert sep["cohens_d"] == pytest.approx(1.0)
    assert sep["wasserstein_distance"] == pytest.approx(1.0)
    assert sep["ai"]["median"] == 2.0 and sep["ai"]["iqr"] == pytest.approx(1.0)


# --- cluster bootstrap ------------------------------------------------------------------------

def test_bootstrap_is_seed_deterministic():
    labels = [1, 1, 1, 0, 0, 0, 0]
    clusters = ["a", "a", "b", "r1", "r1", "r2", "r3"]
    scores = np.array([0.9, 0.8, 0.4, 0.5, 0.3, 0.2, 0.6])
    y = np.array(labels)

    def stat(idx):
        return {"auroc": ch.auroc(scores[idx][y[idx] == 1], scores[idx][y[idx] == 0])}
    one = ch.bootstrap(labels, clusters, stat, n=200)
    two = ch.bootstrap(labels, clusters, stat, n=200)
    assert one == two
    assert one["seed"] == 20261021


def test_bootstrap_preserves_clusters_and_class_strata():
    labels = [1, 1, 1, 0, 0, 0]
    clusters = ["a", "a", "b", "r", "r", "s"]
    seen = []

    def stat(idx):
        seen.append(idx.copy())
        return {"n": float(len(idx))}
    ch.bootstrap(labels, clusters, stat, n=300)
    for idx in seen:
        rows = list(idx)
        # cluster "a" (rows 0,1) and "r" (rows 3,4) always travel whole
        assert rows.count(0) == rows.count(1)
        assert rows.count(3) == rows.count(4)
        # two AI clusters and two REAL clusters drawn every time
        ai_draws = rows.count(0) + rows.count(2)
        real_draws = rows.count(3) + rows.count(5)
        assert ai_draws == 2 and real_draws == 2


def test_bootstrap_counts_invalid_replicates():
    labels = [1, 1, 0, 0]
    clusters = ["a", "b", "r", "s"]
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    y = np.array(labels)

    def stat(idx):
        return {"auroc": ch.auroc(scores[idx][y[idx] == 1], scores[idx][y[idx] == 0])}
    out = ch.bootstrap(labels, clusters, stat, n=400)
    # both clusters "a" and "r" drawn twice -> all scores 0.5 -> undefined AUROC
    assert out["invalid_replicate_count"] > 0
    assert out["valid_replicate_count"] + out["invalid_replicate_count"] == 400


def test_bootstrap_default_budget_is_1000():
    out = ch.bootstrap([1, 0], ["a", "b"], lambda idx: {"n": float(len(idx))})
    assert out["replicates"] == 1000


# --- coverage vs discrimination ---------------------------------------------------------------

def make_case(tag, truth, status, score=None, cluster=None, face=True, gen=None, domain=None):
    ai = truth == "AI"
    det = {"status": status, "score": score}
    return {"sha256": sha(tag), "canonical_path": f"x/{tag}.mp4",
            "ground_truth": t3.VERIFIED_AI if ai else t3.VERIFIED_REAL,
            "generator_family": gen if ai else None, "generator_model": None,
            "capture_domain": None if ai else (domain or "webcam"), "face_present": face,
            "speech_talking_head": "NOT_APPLICABLE",
            "source_cluster_id": cluster or f"c{tag}", "source_collection_id": "pub",
            "detectors": {d: dict(det) for d in ch.DETECTORS}}


def toy_cases():
    cases = [make_case(i, "AI", "SUCCESS", 0.5 + i / 40, gen="Luma" if i % 2 else "Sora",
                       cluster=f"ai{i // 2}") for i in range(10)]
    cases += [make_case(100 + i, "REAL", "SUCCESS", i / 20, cluster=f"re{i}",
                        domain="webcam" if i % 2 else "nasa") for i in range(12)]
    cases.append(make_case(200, "AI", "ABSTAINED", face=False))
    cases.append(make_case(201, "REAL", "TIMEOUT"))
    cases.append(make_case(202, "AI", "FAILED"))
    return cases


def test_census_counts_every_status_and_usable_ratios():
    c = ch.census(toy_cases(), ch.SVD)
    assert c["total_cases"] == 25 and c["attempted"] == 25
    assert (c["SUCCESS"], c["ABSTAINED"], c["TIMEOUT"], c["FAILED"]) == (22, 1, 1, 1)
    assert (c["usable_ai"], c["total_ai"]) == (10, 12)
    assert (c["usable_real"], c["total_real"]) == (12, 13)


def test_abstentions_and_timeouts_never_enter_auroc():
    cases = toy_cases()
    rows = ch.usable_rows(cases, ch.FACE)
    assert len(rows) == 22
    with_zero = rows + [{"y": 1, "score": 0.0, "cluster": "z", "sha256": "z"}]
    ai = [r["score"] for r in rows if r["y"] == 1]
    real = [r["score"] for r in rows if r["y"] == 0]
    assert ch.detector_metrics(cases, ch.FACE, ANCHORS)["discrimination"]["auroc"] == \
        pytest.approx(ch.auroc(ai, real))
    ai0 = [r["score"] for r in with_zero if r["y"] == 1]
    assert ch.auroc(ai0, real) != ch.auroc(ai, real)


def test_analyze_refuses_score_without_success():
    cases = toy_cases()
    cases[-1]["detectors"][ch.SVD]["score"] = 0.0
    scores = {"schema": ch.SCHEMA_SCORES, "partition": "DEV", "anchors": ANCHORS,
              "freeze": {}, "cases": cases}
    dev = [{k: c[k] for k in ("sha256", "ground_truth", "source_cluster_id", "face_present",
                              "generator_family", "capture_domain")} for c in cases]
    with pytest.raises(ch.Refusal):
        ch.analyze(scores, dev)


def test_analyze_refuses_cases_outside_dev():
    cases = toy_cases()
    scores = {"schema": ch.SCHEMA_SCORES, "partition": "DEV", "anchors": ANCHORS,
              "freeze": {}, "cases": cases}
    dev = [{k: c[k] for k in ("sha256", "ground_truth", "source_cluster_id", "face_present",
                              "generator_family", "capture_domain")} for c in cases[:-1]]
    with pytest.raises(ch.Refusal):
        ch.analyze(scores, dev)


def test_detector_entry_maps_version_mismatch_to_failed_and_never_scores_non_success():
    e = ch.detector_entry(ch.SVD, {"status": "SUCCESS", "score": 0.99,
                                   "provider_version": "other"}, None, ANCHORS)
    assert e["status"] == "FAILED" and e["score"] is None
    e = ch.detector_entry(ch.SVD, {"status": "TIMEOUT", "score": 0.0}, None, ANCHORS)
    assert e["score"] is None
    assert ch.detector_entry(ch.LIP, None, None, ANCHORS)["status"] == "NOT_ATTEMPTED"


def test_effort_unavailable_is_non_blocking():
    cases = toy_cases()
    for c in cases:
        c["detectors"][ch.EFFORT] = {"status": "UNAVAILABLE", "score": None}
    m = ch.detector_metrics(cases, ch.EFFORT, ANCHORS)
    assert m["status"] == "INSUFFICIENT_USABLE_SCORES"
    assert m["census"]["UNAVAILABLE"] == len(cases)


def test_paired_subset_uses_only_common_success():
    cases = toy_cases()
    cases[0]["detectors"][ch.LIP] = {"status": "ABSTAINED", "score": None}
    p = ch.paired_metrics(cases, (ch.SVD, ch.LIP))
    assert p["cases"] == 21
    assert "auroc_diff.synthetic_video-lip_forensics" in p["bootstrap"]["intervals"]


# --- svd cache --------------------------------------------------------------------------------

def test_svd_cache_is_add_only():
    cache = ch.load_cache(Path("/nonexistent/cache.json"))
    rec = {"status": "SUCCESS", "score": 0.4, "provider_version": "v", "evidence": {"x": 1}}
    assert ch.cache_add(cache, "s", rec) is True
    assert ch.cache_add(cache, "s", rec) is False
    with pytest.raises(ch.Refusal):
        ch.cache_add(cache, "s", {**rec, "score": 0.5})
    with pytest.raises(ValueError):
        ch.cache_add(cache, "t", {"status": "TIMEOUT", "provider_version": "v"})


def test_cached_svd_jobs_are_not_resubmitted(tmp_path):
    cases = [{"sha256": "s1", "canonical_path": "a.mp4"}, {"sha256": "s2",
                                                           "canonical_path": "b.mp4"}]
    cache = ch.load_cache(tmp_path / "c.json")
    ch.cache_add(cache, "s1", {"status": "SUCCESS", "score": 0.1, "provider_version": "svd-v"})
    (tmp_path / "synthetic_video.jsonl").write_text(
        json.dumps({"sha256": "s2", "status": "TIMEOUT"}) + "\n")
    jobs = ch.pending_jobs(cases, ch.SVD, tmp_path, cache, ANCHORS)
    assert [j["sha256"] for j in jobs] == ["s2"]  # timeouts are retried, cache hits are not


# --- case states ------------------------------------------------------------------------------

def test_case_state_mapping():
    real = list(np.linspace(0, 1, 11))  # median 0.5, p90 0.9
    assert ch.case_state("ABSTAINED", None, real, 0.95) == ch.STATE_UNSCORED
    assert ch.case_state("TIMEOUT", None, real, 0.95) == ch.STATE_UNSCORED
    assert ch.case_state("SUCCESS", 0.97, real, 0.95) == ch.STATE_FLAGGED
    assert ch.case_state("SUCCESS", 0.92, real, 0.95) == ch.STATE_ABOVE
    assert ch.case_state("SUCCESS", 0.5, real, 0.95) == ch.STATE_LOW
    assert ch.case_state("SUCCESS", 0.7, real, 0.95) == ch.STATE_MID


def test_case_state_without_production_threshold_never_flags():
    real = [0.1, 0.2, 0.3]
    assert ch.case_state("SUCCESS", 0.99, real, None) == ch.STATE_ABOVE


# --- determinism ------------------------------------------------------------------------------

def test_metrics_are_byte_identical_for_identical_scores():
    cases = toy_cases()
    scores = {"schema": ch.SCHEMA_SCORES, "partition": "DEV", "anchors": ANCHORS,
              "freeze": {"manifest_sha256": "m"}, "cases": cases}
    dev = [{k: c[k] for k in ("sha256", "ground_truth", "source_cluster_id", "face_present",
                              "generator_family", "capture_domain")} for c in cases]
    a = ch.dumps(ch.analyze(json.loads(json.dumps(scores)), dev))
    b = ch.dumps(ch.analyze(json.loads(json.dumps(scores)), list(reversed(dev))))
    assert a == b
    assert "NaN" not in a


# --- svd circuit breaker ----------------------------------------------------------------------

class FakeProc:
    def __init__(self, lines):
        self.stdout, self.killed, self.returncode = iter(lines), False, 0

    def kill(self):
        self.killed = True

    def wait(self):
        return 0


def test_circuit_breaker_stops_after_consecutive_timeouts_and_records_them(monkeypatch, tmp_path):
    import threading
    digests = {"m": "d"}
    lines = [json.dumps({"kind": "preflight", "module_sha256": digests}) + "\n"]
    lines += [json.dumps({"kind": "result", "sha256": f"s{i}", "status": st}) + "\n"
              for i, st in enumerate(["SUCCESS", "TIMEOUT", "SUCCESS", "TIMEOUT", "TIMEOUT",
                                      "SUCCESS"])]
    proc = FakeProc(lines)
    stopped = []
    monkeypatch.setattr(ch.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(ch, "stop_container", stopped.append)
    ch.run_container(ch.SVD, [{}] * 6, tmp_path, tmp_path, "synthetic_video", digests,
                     threading.Lock(), breaker=2)
    assert proc.killed
    assert len(stopped) == 1 and stopped[0].startswith("r18t4-synthetic_video-")
    recorded = [r["status"] for r in ch.read_jsonl(tmp_path / "synthetic_video.jsonl")]
    assert recorded == ["SUCCESS", "TIMEOUT", "SUCCESS", "TIMEOUT", "TIMEOUT"]


def test_container_digest_mismatch_is_a_refusal(monkeypatch, tmp_path):
    import threading
    proc = FakeProc([json.dumps({"kind": "preflight", "module_sha256": {"m": "x"}}) + "\n"])
    monkeypatch.setattr(ch.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(ch, "stop_container", lambda name: None)
    with pytest.raises(ch.Refusal):
        ch.run_container(ch.FACE, [{}], tmp_path, tmp_path, "face_manipulation", {"m": "d"},
                         threading.Lock())


def test_container_command_is_named_ephemeral_and_read_only():
    cmd = ch.container_cmd(ch.SVD, [], "/tmp", name="r18t4-x")
    assert cmd[:6] == ["docker", "compose", "run", "--rm", "--no-deps", "-T"]
    assert cmd[cmd.index("--name") + 1] == "r18t4-x"
    assert any(a.endswith(":/r18t4_corpus:ro") for a in cmd)
