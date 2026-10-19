"""R17-T4 analyser: frozen cause order, decision-path restriction, provenance and refusals (synthetic data only)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from eval import r17_t2_run as run
from eval import r17_t4_analyze as an

RE = an.load_risk_engine()
SVD_OK = {"provider": "nvidia", "signal_type": "synthetic_video", "status": "SUCCESS",
          "provider_version": RE.SVD_PROVIDER_VERSION, "score": 0.1, "total_clips": 480}
FACE_OK = {"provider": "efficientnet-b7", "signal_type": "face_manipulation", "status": "SUCCESS",
           "provider_version": RE.FACE_PROVIDER_VERSION, "score": 0.5, "frames_scored": 4}


def failed(provider, signal_type):
    return {"provider": provider, "signal_type": signal_type, "status": "FAILED",
            "provider_version": None, "score": None}


SVD_FAILED = {**failed("nvidia", "synthetic_video"), "total_clips": None}
FACE_FAILED = {**failed("efficientnet-b7", "face_manipulation"), "frames_scored": None}
OFF_PATH_OK = {"lip_forensics": {"status": "SUCCESS"}, "audio_authenticity": {"status": "SUCCESS"},
               "active_speaker": {"status": "SUCCESS"}, "provenance": {"status": "SUCCESS"}}


def case(rule, truth="AI", svd=SVD_OK, face=FACE_OK, reasons=None, off_path=None,
         level=None, version="r9-v5.0.0", logs=False):
    signals = {**OFF_PATH_OK, **(off_path or {}), "synthetic_video": svd, "face_manipulation": face}
    return {"case_id": f"{truth}-001", "truth": truth, "blind_id": "B001", "analysis_id": "a-1",
            "source_cluster_id": "CL-X", "generator_or_source": "gen", "clip_has_audio": "true",
            "risk_level": level or {"R9-200": "NO_CALIBRATED_MANIPULATION_SIGNAL"}.get(
                rule, "MANIPULATION_DETECTED" if rule.startswith("R9-1") else "INCONCLUSIVE"),
            "rule_id": rule, "rules_version": version, "created_at": "2026-10-17T20:00:00Z",
            "media": {}, "signals": signals, "reasons": {**{n: None for n in (*an.SIGNALS,
                                                                            *an.ENRICHMENT_ONLY)},
                                                         **(reasons or {})},
            "sealed_source": "r17_t3_phase_a/cases/B001/analysis.json",
            "worker_log": {"available": logs}}


def reason(error, status="FAILED"):
    return {"status": status, "error": error}


def all_pointers(result):
    yield from result["primary_evidence"]
    for c in result["contributing_causes"]:
        yield from c["evidence"]
    for d in result["decision_path_signals"].values():
        yield from d["evidence"]


# ---------------------------------------------------------------- frozen order (taxonomy §5)

def test_primary_order_is_the_frozen_one():
    assert an.PRIMARY_ORDER == ("decision_contradicts_truth", "detector_input_ineligible",
                                "detector_runtime_failure", "rule_requires_missing_signal",
                                "rule_uncertain_band", "rule_conflicting_signals",
                                "unclassifiable_from_stored_evidence")


def test_no_calibrated_signal_on_ai_contradicts_truth():
    r = an.classify(case("R9-200", "AI"), RE)
    assert r["primary_cause"] == "decision_contradicts_truth"
    assert r["attribution"] is None
    assert {p["field"] for p in r["primary_evidence"]} == {"truth", "risk_level", "risk_rule_id"}
    assert not r["rests_on_non_sealed_evidence"]


def test_manipulation_detected_on_real_contradicts_truth():
    assert an.classify(case("R9-101", "REAL"), RE)["primary_cause"] == "decision_contradicts_truth"


def test_agreeing_decision_has_no_failure_cause():
    assert an.classify(case("R9-200", "REAL"), RE)["primary_cause"] == an.AGREES


def test_input_beats_runtime_when_both_path_signals_failed():
    c = case("R9-301", svd=SVD_FAILED, face=FACE_FAILED,
             reasons={"synthetic_video": reason("NvidiaProviderUnavailable"),
                      "face_manipulation": reason("FaceDetectorNoFaceFound")})
    r = an.classify(c, RE)
    assert r["primary_cause"] == "detector_input_ineligible"
    assert [(x["cause"], x["signal"], x["on_decision_path"]) for x in r["contributing_causes"]] == \
        [("detector_runtime_failure", "synthetic_video", True)]
    assert r["attribution"] == "attributable_to_detector_evidence"


def test_runtime_on_path_is_primary_when_no_input_failure():
    c = case("R9-300", face=FACE_OK, svd=SVD_FAILED,
             reasons={"synthetic_video": reason("NvidiaProviderUnavailable")})
    r = an.classify(c, RE)
    assert r["primary_cause"] == "detector_runtime_failure"
    assert r["rests_on_non_sealed_evidence"]


def test_timeout_status_is_runtime_whatever_it_carries():
    c = case("R9-300", svd={**SVD_FAILED, "status": "TIMEOUT"},
             reasons={"synthetic_video": reason("Anything", "TIMEOUT")})
    assert an.classify(c, RE)["primary_cause"] == "detector_runtime_failure"


# ---------------------------------------------------------------- decision-path restriction

def test_off_path_failure_is_never_primary():
    """Lip, active speaker and audio fail with known reasons; the face failure has none stored."""
    c = case("R9-300", face=FACE_FAILED,
             off_path={"lip_forensics": {"status": "FAILED"},
                       "active_speaker": {"status": "FAILED"},
                       "audio_authenticity": {"status": "FAILED"}},
             reasons={"lip_forensics": reason("LipForensicsInferenceError"),
                      "active_speaker": reason("NvidiaActiveSpeakerInputError"),
                      "audio_authenticity": reason("SpeakerDiarizationAudioError")})
    r = an.classify(c, RE)
    assert r["primary_cause"] == "rule_requires_missing_signal"
    off = {x["signal"]: (x["cause"], x["on_decision_path"]) for x in r["contributing_causes"]}
    assert off["lip_forensics"] == ("detector_runtime_failure", False)
    assert off["active_speaker"] == ("detector_input_ineligible", False)
    assert off["audio_authenticity"] == ("detector_input_ineligible", False)
    assert off["face_manipulation"] == ("unclassifiable_from_stored_evidence", True)


def test_off_path_failure_beside_a_decision_is_contributing_only():
    c = case("R9-200", "AI", off_path={"lip_forensics": {"status": "FAILED"}},
             reasons={"lip_forensics": reason("LipForensicsNoTrackedFace")})
    r = an.classify(c, RE)
    assert r["primary_cause"] == "decision_contradicts_truth"
    assert [x["on_decision_path"] for x in r["contributing_causes"]] == [False]


def test_enrichment_only_component_is_contributing_only():
    c = case("R9-300", face=FACE_FAILED,
             reasons={"face_forgery": reason("EffortModelUnavailable")})
    r = an.classify(c, RE)
    assert r["primary_cause"] == "rule_requires_missing_signal"
    assert r["contributing_causes"][-1]["signal"] == "face_forgery"
    assert r["contributing_causes"][-1]["on_decision_path"] is False


def test_every_rule_reads_only_the_decision_eligible_pair():
    assert set(an.DECISION_PATH.values()) == {("synthetic_video", "face_manipulation")}
    src = an.RISK_ENGINE.read_text()
    body = src[src.index("def evaluate_v5("):]
    assert "svd_usable = " in body and "face_usable = " in body
    assert not re.search(r"\blip\.\w", body), "evaluate_v5 must not read lip"


# ---------------------------------------------------------------- missing evidence, provenance

def test_missing_supplemental_evidence_is_unclassifiable_not_forced():
    c = case("R9-301", svd=SVD_FAILED, face=FACE_FAILED)
    r = an.classify(c, RE)
    assert r["primary_cause"] == "unclassifiable_from_stored_evidence"
    assert r["supplemental_sources"] == {an.PROD: "absent", an.LOG: "absent_or_rotated"}
    assert r["attribution"] == "attributable_to_detector_evidence"


def test_unmapped_reason_is_not_forced_into_a_class():
    c = case("R9-301", svd=SVD_FAILED, face=FACE_FAILED,
             reasons={"synthetic_video": reason("SomethingNew"),
                      "face_manipulation": reason(None)})
    assert an.classify(c, RE)["primary_cause"] == "unclassifiable_from_stored_evidence"


def test_every_pointer_carries_exactly_one_known_provenance():
    c = case("R9-301", svd=SVD_FAILED, face=FACE_FAILED,
             off_path={"lip_forensics": {"status": "FAILED"}},
             reasons={"synthetic_video": reason("NvidiaProviderUnavailable"),
                      "face_manipulation": reason("FaceDetectorNoFaceFound"),
                      "lip_forensics": reason("LipForensicsNoTrackedFace"),
                      "face_forgery": reason("EffortNoFaceDetected")})
    ptrs = list(all_pointers(an.classify(c, RE)))
    assert ptrs and all(p["provenance"] in an.PROVENANCE for p in ptrs)
    with pytest.raises(ValueError):
        an.pointer("a", "live_state", "x", "y", 1)


def test_reasons_come_only_from_production_rows_and_are_flagged_non_sealed():
    c = case("R9-300", face=FACE_FAILED,
             reasons={"face_manipulation": reason("FaceDetectorNoFaceFound")})
    r = an.classify(c, RE)
    reason_ptrs = [p for p in r["primary_evidence"] if p["field"].endswith("metadata.error")]
    assert [p["provenance"] for p in reason_ptrs] == [an.PROD]
    assert r["rests_on_non_sealed_evidence"]


def test_every_committed_detector_failure_class_has_a_category():
    """Taxonomy §4 maps classes by definition, so no class the code can write is left unknown."""
    app = an.REPO / "apps/api/app"
    bases = {"FaceDetectorError", "LipForensicsError", "EffortError", "AudioDetectorError",
             "SpeakerDiarizationError", "NvidiaActiveSpeakerError", "NvidiaVideoError"}
    found = set()
    for module in ("face_detector", "lip_forensics", "effort", "audio_detector",
                   "speaker_diarization", "nvidia_active_speaker", "nvidia_video"):
        for name, parent in re.findall(r"^class (\w+)\((\w+)\):", (app / f"{module}.py").read_text(),
                                       re.M):
            if parent in bases or parent.endswith("ProviderError"):
                found.add(name)
    assert found
    assert found - an.INPUT_REASONS - an.RUNTIME_REASONS == set()
    assert not an.INPUT_REASONS & an.RUNTIME_REASONS


# ---------------------------------------------------------------- attribution (taxonomy §7)

def test_attribution_aggregation_rule_when_every_path_signal_is_usable():
    r = an.classify(case("R9-300"), RE)
    assert r["attribution"] == "attributable_to_aggregation_rule"
    # r9-v5.0.0 has no band or conflict rule, so classes 5 and 6 are never assigned.
    assert r["primary_cause"] == "unclassifiable_from_stored_evidence"


def test_attribution_indeterminate_when_a_sealed_field_is_missing():
    svd = {k: v for k, v in SVD_OK.items() if k != "total_clips"}
    c = case("R9-300", svd=svd, face=FACE_FAILED)
    assert an.classify(c, RE)["attribution"] == "indeterminate"


def test_uncalibrated_deployment_is_unusable():
    c = case("R9-300", svd={**SVD_OK, "provider_version": SVD_OK["provider_version"] + "-x"})
    r = an.classify(c, RE)
    assert r["decision_path_signals"]["synthetic_video"]["usable"] is False
    assert r["primary_cause"] == "rule_requires_missing_signal"


def test_unknown_ruleset_is_unclassifiable():
    r = an.classify(case("R9-300", version="r7-v4.0.0"), RE)
    assert r["primary_cause"] == "unclassifiable_from_stored_evidence"


# ---------------------------------------------------------------- population sums

def test_one_primary_cause_per_case_and_sum_45():
    cases = []
    for i in range(45):
        truth = "AI" if i < 30 else "REAL"
        c = case("R9-301", truth, svd=SVD_FAILED, face=FACE_FAILED,
                 reasons={"synthetic_video": reason("NvidiaProviderUnavailable"),
                          "face_manipulation": reason("FaceDetectorNoFaceFound")})
        c["case_id"] = f"{truth}-{i:03d}"
        cases.append(c)
    cases[0] = case("R9-200", "AI")
    results = [an.classify(c, RE) for c in cases]
    s = an.summarise(cases, results)
    assert sum(s["primary"].values()) == 45
    assert s["truth"] == {"AI": 30, "REAL": 15}
    assert all(sum(r["primary_cause"] == p for p in (*an.PRIMARY_ORDER, an.AGREES)) == 1
               for r in results)
    assert s["abstentions"] == 44 and sum(s["attribution"].values()) == 44


# ---------------------------------------------------------------- integrity refusals (taxonomy §1)

def sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def sealed(tmp_path):
    pa = tmp_path / "phase_a"
    (pa / "cases/B001").mkdir(parents=True)
    (pa / "cases/B001/analysis.json").write_text(json.dumps({"id": "a-1"}))
    entries = run.seal_files(pa)
    root = run.seal_root(entries)
    (pa / run.SEAL).write_text(json.dumps({"files": entries}))
    (pa / run.SEAL_ROOT).write_text(root + "\n")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"cases": []}))
    ds = tmp_path / "dataset"
    ds.mkdir()
    (ds / "manifest.json").write_text("[]")
    (ds / "FREEZE.json").write_text("{}")
    tax = tmp_path / "taxonomy.md"
    tax.write_text("frozen")
    anchor = {"phase_a_seal_root": root, "results_json_sha256": sha(results.read_bytes()),
              "ruleset": "r9-v5.0.0",
              "dataset_anchors": {"manifest.json": sha(b"[]"), "FREEZE.json": sha(b"{}")}}
    return {"anchor": anchor, "phase_a": pa, "results": results, "dataset": ds, "taxonomy": tax,
            "tax_sha": sha(b"frozen")}


def verify(s, **over):
    args = {"anchor": s["anchor"], "phase_a": s["phase_a"], "results": s["results"],
            "dataset": s["dataset"], "taxonomy": s["taxonomy"], "taxonomy_sha256": s["tax_sha"]}
    args.update(over)
    return an.verify_inputs(**args)


def test_verified_inputs_pass(sealed):
    assert verify(sealed)["phase_a_seal_root"] == sealed["anchor"]["phase_a_seal_root"]


def test_refuses_taxonomy_changed_after_its_hash_was_recorded(sealed):
    sealed["taxonomy"].write_text("edited")
    with pytest.raises(an.Refusal, match="S4"):
        verify(sealed)


def test_refuses_a_changed_sealed_file(sealed):
    (sealed["phase_a"] / "cases/B001/analysis.json").write_text("{}")
    with pytest.raises(an.Refusal, match="seal"):
        verify(sealed)


def test_refuses_an_extra_file_in_the_sealed_directory(sealed):
    (sealed["phase_a"] / "cases/B001/new.json").write_text("{}")
    with pytest.raises(an.Refusal, match="seal"):
        verify(sealed)


def test_refuses_a_seal_root_other_than_the_anchor(sealed):
    with pytest.raises(an.Refusal, match="seal"):
        verify(sealed, anchor={**sealed["anchor"], "phase_a_seal_root": "0" * 64})


def test_refuses_a_changed_results_json(sealed):
    sealed["results"].write_text(json.dumps({"cases": [1]}))
    with pytest.raises(an.Refusal, match="results.json"):
        verify(sealed)


def test_refuses_a_changed_manifest(sealed):
    (sealed["dataset"] / "manifest.json").write_text("[1]")
    with pytest.raises(an.Refusal, match="dataset anchor"):
        verify(sealed)


class FakeGit:
    def __init__(self, exists=True, committed=b"anchor"):
        self.exists, self.committed = exists, committed

    def __call__(self, *args):
        class P:
            pass
        p = P()
        if args[0] == "cat-file":
            p.returncode, p.stdout = (0 if self.exists else 1), b""
        else:
            p.returncode, p.stdout = 0, self.committed
        return p


def test_refuses_a_missing_t3_closure_commit(tmp_path):
    a = tmp_path / "anchor.json"
    a.write_bytes(b"anchor")
    with pytest.raises(an.Refusal, match="missing"):
        an.verify_t3_commit(anchor=a, run=FakeGit(exists=False))


def test_refuses_an_anchor_that_differs_from_the_closure_commit(tmp_path):
    a = tmp_path / "anchor.json"
    a.write_bytes(b"edited")
    with pytest.raises(an.Refusal, match="differs"):
        an.verify_t3_commit(anchor=a, run=FakeGit())
    a.write_bytes(b"anchor")
    assert an.verify_t3_commit(anchor=a, run=FakeGit()) == an.T3_CLOSURE_COMMIT


# ---------------------------------------------------------------- reproduction comparison (D2)

def test_reproduction_skipped_when_image_code_differs():
    out = an.compare_reproduction([], {}, {"risk_engine_sha256": "x", "cases": {}}, "y")
    assert out["exercised"] is False


def test_reproduction_reports_a_mismatch():
    rows = [{"case_id": "AI-001", "blind_id": "B001", "analysis_id": "a-1"}]
    analyses = {"B001": {"risk_level": "INCONCLUSIVE", "risk_rule_id": "R9-300"}}
    got = {"risk_engine_sha256": "s", "cases": {"a-1": {"risk_level": "INCONCLUSIVE",
                                                        "rule_id": "R9-301"}}}
    out = an.compare_reproduction(rows, analyses, got, "s")
    assert out["exercised"] and out["matches"] == 0 and len(out["mismatches"]) == 1


def test_container_snippet_reproduces_on_the_committed_module():
    """The snippet's logic, run here against the committed file instead of the image's copy."""
    for svd, face, rule in ((SVD_OK, FACE_OK, "R9-200"), (SVD_FAILED, FACE_OK, "R9-300"),
                            (SVD_FAILED, FACE_FAILED, "R9-301")):
        d = RE.evaluate_v5(svd=RE.SvdEvidence(**svd), face=RE.FaceEvidence(**face))
        assert d.rule_id == rule
