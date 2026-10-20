"""R17-T5 challengers: C1/C2 rules, coverage and FP figures, exhaustive verdicts, C3 outcome mapping (synthetic data only)."""

import subprocess
from pathlib import Path

import pytest

from eval import r17_t5_c3_diagnostic as c3
from eval import r17_t5_challenger as ch

RE = ch.load_risk_engine()
SVD_LOW = {"provider": "nvidia", "signal_type": "synthetic_video", "status": "SUCCESS",
           "provider_version": RE.SVD_PROVIDER_VERSION, "score": 0.5, "total_clips": 10}
SVD_HIGH = {**SVD_LOW, "score": 0.99}
SVD_FAILED = {"provider": "nvidia", "signal_type": "synthetic_video", "status": "FAILED",
              "provider_version": None, "score": None, "total_clips": None}
FACE_OK = {"provider": "efficientnet-b7", "signal_type": "face_manipulation", "status": "SUCCESS",
           "provider_version": RE.FACE_PROVIDER_VERSION, "score": 0.5, "frames_scored": 4}
FACE_FAILED = {"provider": "efficientnet-b7", "signal_type": "face_manipulation",
               "status": "FAILED", "provider_version": None, "score": None, "frames_scored": None}
LIP_HIT = {"provider": "lipforensics", "signal_type": "lip_forensics", "status": "SUCCESS",
           "provider_version": RE.LIP_PROVIDER_VERSION, "score": 0.5, "windows_scored": 3}
LIP_LOW = {**LIP_HIT, "score": 0.1}
LIP_FAILED = {"provider": "lipforensics", "signal_type": "lip_forensics", "status": "FAILED",
              "provider_version": None, "score": None, "windows_scored": None}


def case(cid="AI-001", truth="AI", svd=SVD_LOW, face=FACE_FAILED, lip=LIP_FAILED, faceless=True,
         cluster="CL-X"):
    return {"case_id": cid, "truth": truth, "source_cluster_id": cluster, "svd": svd,
            "face": face, "lip": lip, "faceless": faceless}


def decide(c):
    return ch.decide_case(RE, c, ch.evidence(RE.SvdEvidence, c["svd"], ch.SVD_FIELDS))


# ---------------------------------------------------------------- constants

def test_thresholds_are_the_production_constants():
    ch.check_constants(RE)
    assert round(RE.SVD_T_HIGH, 4) == 0.9551


def test_changed_threshold_is_refused():
    class Fake:
        SVD_T_HIGH, LIP_T_HIGH = 0.9, RE.LIP_T_HIGH
    with pytest.raises(ch.Refusal):
        ch.check_constants(Fake)


# ---------------------------------------------------------------- C1

def test_c1_faceless_usable_below_threshold_becomes_negative():
    d = decide(case())
    assert (d["C0"], d["C0_rule"]) == ("INCONCLUSIVE", "R9-300")
    assert d["C1"] == "NO_CALIBRATED_MANIPULATION_SIGNAL"


def test_c1_svd_hit_is_already_positive_under_c0():
    d = decide(case(svd=SVD_HIGH))
    assert d["C0"] == d["C1"] == "MANIPULATION_DETECTED"


def test_c1_threshold_equality_reaches():
    d = ch.c1(RE, "INCONCLUSIVE", True,
              RE.SvdEvidence(*({**SVD_LOW, "score": RE.SVD_T_HIGH}[k] for k in ch.SVD_FIELDS)))
    assert d == "MANIPULATION_DETECTED"


def test_c1_needs_usable_svd():
    assert decide(case(svd=SVD_FAILED))["C1"] == "INCONCLUSIVE"
    foreign = {**SVD_LOW, "provider_version": "other-function"}
    assert decide(case(svd=foreign))["C1"] == "INCONCLUSIVE"
    assert decide(case(svd={**SVD_LOW, "total_clips": 0}))["C1"] == "INCONCLUSIVE"


def test_c1_needs_faceless_reason():
    # Face failed for another reason: C1 does not apply.
    assert decide(case(faceless=False))["C1"] == "INCONCLUSIVE"


def test_c1_leaves_decided_cases_alone():
    d = decide(case(face=FACE_OK, faceless=False))
    assert d["C0"] == d["C1"] == "NO_CALIBRATED_MANIPULATION_SIGNAL"


def test_c1_structure_check_refuses_other_changes():
    ch.check_c1_structure({"A": {"C0": "INCONCLUSIVE", "C1": "NO_CALIBRATED_MANIPULATION_SIGNAL"}})
    with pytest.raises(ch.Refusal):
        ch.check_c1_structure({"A": {"C0": "INCONCLUSIVE", "C1": "MANIPULATION_DETECTED"}})


# ---------------------------------------------------------------- C2

def test_c2_aux_only_on_inconclusive_with_lip_hit():
    d = decide(case(svd=SVD_FAILED, lip=LIP_HIT))
    assert d["C0"] == "INCONCLUSIVE"
    assert d["C2@C0"] == ch.AUXILIARY and d["C2@C1"] == ch.AUXILIARY
    assert decide(case(svd=SVD_FAILED, lip=LIP_LOW))["C2@C0"] == "INCONCLUSIVE"


def test_c2_does_not_override_decided_cases():
    d = decide(case(svd=SVD_LOW, lip=LIP_HIT))
    # C0 abstains, so C2@C0 fires; C1 already decided negative, so C2@C1 does not.
    assert d["C2@C0"] == ch.AUXILIARY
    assert d["C2@C1"] == "NO_CALIBRATED_MANIPULATION_SIGNAL"


def test_c2_needs_calibrated_lip():
    foreign = {**LIP_HIT, "provider_version": "other"}
    assert decide(case(svd=SVD_FAILED, lip=foreign))["C2@C0"] == "INCONCLUSIVE"
    assert decide(case(svd=SVD_FAILED, lip={**LIP_HIT, "windows_scored": 0}))["C2@C0"] == "INCONCLUSIVE"


def test_auxiliary_scores_as_positive():
    assert ch.state(ch.AUXILIARY) == "positive"
    rows = ch.variant_rows([case(svd=SVD_FAILED, lip=LIP_HIT)],
                           {"AI-001": decide(case(svd=SVD_FAILED, lip=LIP_HIT))}, "C2@C0")
    assert rows[0]["confusion"] == "TP" and rows[0]["auxiliary"]


# ---------------------------------------------------------------- figures and metrics

def population():
    return [case("AI-001"), case("AI-002", svd=SVD_FAILED, lip=LIP_HIT),
            case("AI-003", face=FACE_OK, faceless=False),
            case("REAL-001", "REAL"), case("REAL-002", "REAL", svd=SVD_FAILED, cluster="CL-Y")]


def test_figures_coverage_gain_and_fp():
    cases = population()
    decisions = ch.evaluate(RE, cases, lambda c: ch.evidence(RE.SvdEvidence, c["svd"], ch.SVD_FIELDS))
    f = ch.figures(cases, decisions)
    assert f["faceless_cases"] == 4
    assert f["faceless_decided"] == {"C0": 0, "C1": 2}
    assert f["faceless_coverage_gain"] == 2
    assert f["usable_real_denominator"] == 1 and f["usable_real_cases"] == ["REAL-001"]
    assert f["FP_C1_all_real"] == 0
    assert f["c1_changed_cases"] == ["AI-001", "REAL-001"]
    assert f["c2_delta_tp@C0"] == 1 and f["c2_aux_cases@C0"] == ["AI-002"]
    assert f["c2_aux_fp@C0"] == []


def test_metrics_case_and_cluster_level():
    cases = population()
    decisions = ch.evaluate(RE, cases, lambda c: ch.evidence(RE.SvdEvidence, c["svd"], ch.SVD_FIELDS))
    m = ch.measure(cases, decisions, 2)
    c1 = m["C1"]["case_level"]
    assert (c1["TP"], c1["FN"], c1["FP"], c1["TN"]) == (0, 2, 0, 1)
    assert c1["FPR"]["rate"] == 0.0 and c1["FPR"]["ci95"][1] > 0.9  # n=1: wide interval, not bare 0%
    assert m["C0"]["case_level"]["FPR"]["rate"] is None  # zero denominator is N/A
    assert m["C1"]["cluster_level"]["label"].startswith("2 source clusters")
    with pytest.raises(ch.Refusal):
        ch.measure(cases, decisions, 3)


# ---------------------------------------------------------------- exhaustive verdicts

@pytest.mark.parametrize("denominator, fp, gain, expected", [
    (0, 0, 9, "INCONCLUSIVE"),
    (3, 1, 9, "FAIL-SAFETY"),
    (3, 0, 4, "PASS"),
    (3, 0, 3, "FAIL-BENEFIT"),
    (3, 2, 0, "FAIL-SAFETY"),
])
def test_c1_verdict(denominator, fp, gain, expected):
    f = {"usable_real_denominator": denominator, "FP_C1_all_real": fp,
         "faceless_coverage_gain": gain}
    assert ch.c1_verdict(f) == expected


@pytest.mark.parametrize("usable, fmt, expected", [
    (35, 0, "DIAGNOSTIC PASS"), (40, 0, "DIAGNOSTIC PASS"),
    (34, 10, "DIAGNOSTIC FAIL"), (0, 40, "DIAGNOSTIC FAIL"),
    (34, 9, "INCONCLUSIVE"), (0, 0, "INCONCLUSIVE"),
])
def test_c3_verdict(usable, fmt, expected):
    assert ch.c3_verdict(usable, fmt) == expected


# ---------------------------------------------------------------- C3 outcome mapping

def reply_error(cls, mro, code=None):
    return {"error": {"class": cls, "mro": [cls, *mro, "Exception"], "grpc_code": code}}


PROVIDER = ["NvidiaProviderError", "NvidiaVideoError"]


@pytest.mark.parametrize("reply, expected", [
    (None, "other_provider_error"),
    ({"result": {"function_id": RE.SVD_PROVIDER_VERSION, "probability": 0.4, "total_clips": 3}},
     "usable_response"),
    ({"result": {"function_id": "other", "probability": 0.4, "total_clips": 3}},
     "ineligible_response"),
    ({"result": {"function_id": RE.SVD_PROVIDER_VERSION, "probability": 0.4, "total_clips": 0}},
     "ineligible_response"),
    (reply_error("NvidiaProviderError", PROVIDER[1:], "INVALID_ARGUMENT"),
     "input_format_incompatibility"),
    (reply_error("NvidiaProviderError", PROVIDER[1:], "INTERNAL"), "other_provider_error"),
    (reply_error("NvidiaProviderUnavailable", PROVIDER, "UNAVAILABLE"), "provider_unavailable"),
    (reply_error("NvidiaProviderUnavailable", PROVIDER, "RESOURCE_EXHAUSTED"), "provider_unavailable"),
    (reply_error("NvidiaProviderTimeout", PROVIDER, "DEADLINE_EXCEEDED"), "provider_timeout"),
    (reply_error("NvidiaAuthenticationError", PROVIDER, "UNAUTHENTICATED"), "authentication_error"),
    (reply_error("NvidiaProviderInvalidResponse", PROVIDER), "invalid_response"),
    (reply_error("NvidiaLocalFileError", ["NvidiaVideoError"]), "local_file_error"),
    (reply_error("RuntimeError", []), "other_provider_error"),
])
def test_c3_classify(reply, expected):
    outcome, ev = c3.classify(RE, reply)
    assert outcome == expected
    assert (ev is not None) == ("result" in (reply or {}))


def test_c3_out_dir_must_be_isolated(tmp_path):
    sealed = tmp_path / "r17_t3_phase_a"
    with pytest.raises(ch.Refusal):
        c3.check_out_dir(sealed / "sub", (sealed,))
    with pytest.raises(ch.Refusal):
        c3.check_out_dir(ch.REPO / "scripts/eval/baselines", ())
    assert c3.check_out_dir(tmp_path / "r17_t5_diagnostic", (sealed,)).name == "r17_t5_diagnostic"


def test_c3_section_refuses_wrong_provenance(tmp_path):
    p = tmp_path / "c3.json"
    p.write_text('{"provenance": "sealed_t3_record", "records": []}')
    with pytest.raises(ch.Refusal):
        ch.c3_section(RE, population(), p, 2, {})


# ---------------------------------------------------------------- integrity

def test_spec_hash_mismatch_is_refused(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("frozen")
    with pytest.raises(ch.Refusal, match="spec sha256"):
        ch.verify_inputs({}, tmp_path, tmp_path, tmp_path, spec, "0" * 64)


def test_t4_record_must_match_commit(tmp_path):
    record = tmp_path / "rec.json"
    record.write_bytes(b"live")

    def run(*args):
        if args[0] == "cat-file":
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, 0, stdout=b"committed")
    with pytest.raises(ch.Refusal, match="differs"):
        ch.verify_t4_record(record=record, run=run)

    def missing(*args):
        return subprocess.CompletedProcess(args, 1)
    with pytest.raises(ch.Refusal, match="missing"):
        ch.verify_t4_record(record=record, run=missing)


def test_face_reasons_reads_both_evidence_lists():
    def ptr(v):
        return {"field": ch.FACE_ERROR_FIELD, "value": v}
    rec = {"cases": [
        {"case_id": "A", "primary_evidence": [ptr("FaceDetectorNoFaceFound")], "contributing_causes": []},
        {"case_id": "B", "primary_evidence": [], "contributing_causes": [{"evidence": [ptr("X")]}]},
        {"case_id": "C", "primary_evidence": [], "contributing_causes": []},
    ]}
    assert ch.face_reasons(rec) == {"A": "FaceDetectorNoFaceFound", "B": "X", "C": None}
    rec["cases"][0]["contributing_causes"] = [{"evidence": [ptr("Other")]}]
    with pytest.raises(ch.Refusal):
        ch.face_reasons(rec)
