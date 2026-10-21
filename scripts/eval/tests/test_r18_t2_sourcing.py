"""R18-T2 sourcing: seven-field provenance gate, frozen face guard, talking-head tri-state, GT/annotation segregation, cluster diagnostics, isolation (synthetic inputs only)."""

import ast
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from eval import r18_t2_source as src

MODULE_PATH = Path(src.__file__)


def entry(**over):
    e = {
        "item_id": "x:1", "source_uri": "https://example.org/a.mp4",
        "publisher_or_owner": "Publisher", "provenance_evidence": "per-file record",
        "license_or_usage_basis": "CC-BY-4.0 per-clip licence", "download_method":
        "direct_http_urllib", "source_cluster": "c1", "expected_truth": src.VERIFIED_REAL,
        "license_id": "CC-BY-4.0", "provenance_status": "ESTABLISHED",
        "provenance_tier": "published_dataset_with_per_clip_attribution",
        "download_url": "https://example.org/a.mp4",
        "expected_digest": {"algorithm": "sha256", "value": "0" * 64},
        "source_collection_id": "g", "generator_family": None, "generator_model": None,
        "generation_mode": None, "prompt": None, "capture_domain": "ugc_vlog",
        "subject_id": "s1", "speech_primary_annotation": None, "edit_screen_required": True,
    }
    e.update(over)
    return e


# --- seven-field schema and usage rights ----------------------------------------------------

def test_complete_entry_is_eligible():
    assert src.validate_entry(entry()) == (src.ELIGIBLE, None)


@pytest.mark.parametrize("field", src.MANDATORY_FIELDS)
def test_each_missing_mandatory_field_leaves_a_candidate(field):
    status, reason = src.validate_entry(entry(**{field: ""}))
    assert status == src.CANDIDATE_ONLY and field in reason


@pytest.mark.parametrize("basis", ["official promotional media", "Fair use benchmark",
                                   "unknown", "licence unclear"])
def test_promotional_or_fair_use_is_not_a_licence(basis):
    assert src.validate_entry(entry(license_or_usage_basis=basis))[0] == src.CANDIDATE_ONLY


@pytest.mark.parametrize("lic", [None, "CC-BY-NC-4.0", "proprietary", "other"])
def test_licence_outside_the_accepted_set_is_a_candidate(lic):
    assert src.validate_entry(entry(license_id=lic))[0] == src.CANDIDATE_ONLY


def test_unclear_provenance_is_a_candidate_even_with_a_licence():
    assert src.validate_entry(entry(provenance_status="UNCLEAR"))[0] == src.CANDIDATE_ONLY


def test_non_verified_expected_truth_is_a_candidate():
    assert src.validate_entry(entry(expected_truth="AMBIGUOUS"))[0] == src.CANDIDATE_ONLY


def test_entry_without_publisher_digest_is_a_candidate():
    assert src.validate_entry(entry(expected_digest=None))[0] == src.CANDIDATE_ONLY


def test_every_recorded_refusal_stays_a_candidate():
    for e in src.CANDIDATE_ONLY_SOURCES:
        assert all(f in e for f in src.MANDATORY_FIELDS)
        assert src.validate_entry(e)[0] == src.CANDIDATE_ONLY


# --- download and publisher digests ---------------------------------------------------------

def test_download_verifies_sha256_and_md5(tmp_path):
    payload = b"video-bytes"
    source = tmp_path / "up.bin"
    source.write_bytes(payload)
    url = source.as_uri()
    got = src.download(url, tmp_path / "out/a.mp4",
                       {"algorithm": "sha256", "value": hashlib.sha256(payload).hexdigest()})
    assert got["status"] == "OK" and (tmp_path / "out/a.mp4").read_bytes() == payload
    md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
    assert src.download(url, tmp_path / "out/b.mp4",
                        {"algorithm": "md5_base64", "value": md5})["status"] == "OK"


def test_digest_mismatch_keeps_nothing(tmp_path):
    source = tmp_path / "up.bin"
    source.write_bytes(b"tampered")
    got = src.download(source.as_uri(), tmp_path / "out/a.mp4",
                       {"algorithm": "sha256", "value": "f" * 64})
    assert got["status"] == "DIGEST_MISMATCH"
    assert not any((tmp_path / "out").iterdir())


# --- frozen face-annotation tooling ---------------------------------------------------------

class FakeCv2:
    __version__ = "5.0.0"
    INTER_AREA = "inter-area"
    resized = []

    @staticmethod
    def resize(img, size, interpolation=None):
        FakeCv2.resized.append((size, interpolation))
        return np.zeros((size[1], size[0], 3), np.uint8)

    class dnn:
        DNN_BACKEND_OPENCV = "backend-opencv"
        DNN_TARGET_CPU = "target-cpu"

    created = []

    class FaceDetectorYN:
        scores = []

        @classmethod
        def create(cls, *args):
            FakeCv2.created.append(args)
            return cls()

        def setInputSize(self, size):
            pass

        def detect(self, frame):
            s = FakeCv2.FaceDetectorYN.scores.pop(0)
            return 1, (None if s is None else np.array([[0] * 14 + [s]], dtype=np.float32))


def test_frozen_values_are_exactly_the_plan_pins():
    assert src.FACE_SPEC["model_artifact"] == "face_detection_yunet_2023mar.onnx"
    assert src.FACE_SPEC["model_sha256"] == \
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert src.FACE_SPEC["opencv_version"] == "5.0.0"
    assert src.FACE_SPEC["score_threshold"] == 0.80
    assert (src.FACE_SPEC["backend"], src.FACE_SPEC["target"]) == ("DNN_BACKEND_OPENCV",
                                                                   "DNN_TARGET_CPU")


def test_guard_refuses_another_opencv_version(tmp_path):
    fake = type("cv", (), {"__version__": "4.14.0"})
    assert "4.14.0" in src.face_guard(tmp_path / src.FACE_SPEC["model_artifact"], fake)


def test_guard_refuses_a_model_with_another_digest(tmp_path):
    model = tmp_path / src.FACE_SPEC["model_artifact"]
    model.write_bytes(b"not the pinned bytes")
    assert "sha256" in src.face_guard(model, FakeCv2)


def test_guard_refuses_another_artifact_name_and_a_missing_file(tmp_path):
    other = tmp_path / "face_detection_yunet_2023mar_int8.onnx"
    other.write_bytes(b"x")
    assert "artifact" in src.face_guard(other, FakeCv2)
    assert "not found" in src.face_guard(tmp_path / src.FACE_SPEC["model_artifact"], FakeCv2)


def test_guard_passes_only_on_exact_pins(tmp_path, monkeypatch):
    model = tmp_path / src.FACE_SPEC["model_artifact"]
    model.write_bytes(b"pinned")
    monkeypatch.setitem(src.FACE_SPEC, "model_sha256", hashlib.sha256(b"pinned").hexdigest())
    assert src.face_guard(model, FakeCv2) is None


def test_halted_guard_never_runs_the_detector(tmp_path):
    calls = []
    out = src.face_annotation(tmp_path / "v.mp4", 5.0, tmp_path / "missing.onnx", FakeCv2,
                              grab=lambda p, t: calls.append(t))
    assert out["status"] == "ERROR" and out["value"] is None and "halted" in out["error"]
    assert calls == []


@pytest.mark.parametrize("duration,expected", [(3.5, [0.0, 1.0, 2.0, 3.0]),
                                               (3.0, [0.0, 1.0, 2.0, 3.0]),
                                               (0.4, [0.0]), (0, []), (None, [])])
def test_one_fps_timestamp_sampling(duration, expected):
    assert src.sample_timestamps(duration) == expected


def run_face(monkeypatch, scores, duration):
    monkeypatch.setattr(src, "face_guard", lambda *a: None)
    FakeCv2.FaceDetectorYN.scores = list(scores)
    FakeCv2.created = []
    frame = np.zeros((4, 6, 3), np.uint8)
    return src.face_annotation("v.mp4", duration, "m.onnx", FakeCv2, grab=lambda p, t: frame)


def test_face_present_at_thirty_percent_of_frames(monkeypatch):
    out = run_face(monkeypatch, [0.95, 0.9, 0.9, None, None, None, None, None, None, None],
                   9.0)
    assert out["status"] == "DETERMINED" and out["value"] is True
    assert out["frames_decoded"] == 10 and out["frames_with_face"] == 3
    assert FakeCv2.created[0][3] == 0.80
    assert FakeCv2.created[0][-2:] == ("backend-opencv", "target-cpu")


def test_face_absent_below_thirty_percent_and_threshold_is_inclusive_of_080(monkeypatch):
    out = run_face(monkeypatch, [0.80, 0.79, 0.79, 0.5], 3.0)
    assert out["value"] is False and out["frames_with_face"] == 1


def test_undecodable_media_is_unknown_not_false(monkeypatch):
    monkeypatch.setattr(src, "face_guard", lambda *a: None)
    out = src.face_annotation("v.mp4", 4.0, "m.onnx", FakeCv2, grab=lambda p, t: None)
    assert out["status"] == "UNKNOWN" and out["value"] is None


def test_detector_failure_is_an_error_annotation(monkeypatch):
    monkeypatch.setattr(src, "face_guard", lambda *a: None)
    FakeCv2.FaceDetectorYN.scores = []  # detect() raises on pop
    out = src.face_annotation("v.mp4", 2.0, "m.onnx", FakeCv2,
                              grab=lambda p, t: np.zeros((2, 2, 3), np.uint8))
    assert out["status"] == "ERROR" and out["value"] is None


# --- talking-head semantics -----------------------------------------------------------------

DET_FALSE = {"status": "DETERMINED", "value": False}
DET_TRUE = {"status": "DETERMINED", "value": True}


def test_trusted_primary_annotation_is_determined():
    out = src.speech_annotation(DET_TRUE, {"value": True, "evidence": "protocol §3"})
    assert (out["status"], out["value"]) == ("DETERMINED", True)


def test_primary_annotation_comes_first_in_the_hierarchy():
    out = src.speech_annotation(DET_FALSE, {"value": False, "evidence": "protocol"})
    assert (out["status"], out["value"]) == ("DETERMINED", False)


def test_faceless_is_not_applicable_never_false():
    out = src.speech_annotation(DET_FALSE)
    assert out["status"] == "NOT_APPLICABLE" and out["value"] is None


@pytest.mark.parametrize("face", [DET_TRUE, {"status": "ERROR", "value": None},
                                  {"status": "UNKNOWN", "value": None}])
def test_otherwise_unknown(face):
    out = src.speech_annotation(face)
    assert out["status"] == "UNKNOWN" and out["value"] is None and out["tool"] == "none"


# --- GT / annotation segregation and failure decoupling -------------------------------------

def test_annotation_failure_never_touches_ground_truth():
    provenance = {"ground_truth": src.VERIFIED_REAL, "source_cluster_id": "c"}
    face = {**src._face_base(), "status": "ERROR", "error": "annotation halted"}
    rec = src.anchor_record("ab" * 32, "d/a.mp4", True, dict(provenance), face, None, "o")
    assert rec["ground_truth_provenance"] == provenance
    assert set(rec["derived_annotations"]) == {"face_present", "speech_talking_head"}
    assert not set(rec["ground_truth_provenance"]) & {"face_present", "speech_talking_head"}
    assert rec["derived_annotations"]["speech_talking_head"]["status"] == "UNKNOWN"


LED_OK = {"ingest_status": "OK", "technical_status": "INSPECTABLE", "scene_cuts": 0}


def test_external_real_needs_both_screens():
    e = entry()
    assert src.external_decision(e, LED_OK, None)[0] is None
    assert src.external_decision(e, LED_OK, {"verdict": "PASS", "reason": ""})[0] == \
        src.VERIFIED_REAL
    assert src.external_decision(e, LED_OK, {"verdict": "QUARANTINE",
                                             "reason": "screencast"})[0] == src.AMBIGUOUS


def test_scene_cuts_quarantine_real_as_edited():
    truth, reason = src.external_decision(entry(), {**LED_OK, "scene_cuts": 2},
                                          {"verdict": "PASS", "reason": ""})
    assert truth == src.AMBIGUOUS and "edited" in reason


def test_ai_is_not_edit_screened_and_failures_are_not_admitted():
    ai = entry(expected_truth=src.VERIFIED_AI, edit_screen_required=False)
    assert src.external_decision(ai, {**LED_OK, "scene_cuts": None}, None)[0] == src.VERIFIED_AI
    assert src.external_decision(ai, {"ingest_status": "DIGEST_MISMATCH", "error": "x"},
                                 None)[0] is None
    assert src.external_decision(ai, {**LED_OK, "technical_status": "UNREADABLE",
                                      "technical_reason": "no video"}, None)[0] is None
    assert src.external_decision(ai, None, None)[0] is None


# --- cluster diversity and concentration ----------------------------------------------------

def test_concentration_diagnostics():
    c = src.concentration(["a"] * 4 + ["b"] * 2 + list("cdefg"))
    assert c["clusters"] == 7 and c["cases"] == 11
    assert c["max_cluster"] == "a" and c["max_cluster_share"] == round(4 / 11, 4)
    assert c["top5_share"] == round(9 / 11, 4)


def test_empty_concentration_is_defined():
    assert src.concentration([])["max_cluster_share"] is None


# --- selection rules ------------------------------------------------------------------------

def tree_file(path, oid):
    return {"type": "file", "path": path, "lfs": {"oid": oid, "size": 10}}


def test_rapidata_requires_row_and_filename_to_agree_and_spreads_prompts():
    src_def = {"repo": "R/x", "revision": "r" * 40, "mode": "text_to_video"}
    rows = {"alpha_0000_1.mp4": ("alpha", "p0"), "alpha_0001_1.mp4": ("alpha", "p1"),
            "ray2_0000_0.mp4": ("ray2", "p0"), "ray2_0001_0.mp4": ("ray2", "p1"),
            "pika_0000_0.mp4": ("alpha", "p0"),  # row and name disagree
            "alpha_0002_1_ui.mp4": ("alpha", "p2")}
    tree = [tree_file("Videos/" + n, str(i) * 64) for i, n in enumerate(rows)]
    usage = Counter()
    out = src.rapidata_entries(src_def, rows, tree, {"alpha": 1, "ray2": 1}, usage)
    assert {e["item_id"].rsplit(":", 1)[1] for e in out} <= {
        "alpha_0000_1.mp4", "alpha_0001_1.mp4", "ray2_0000_0.mp4", "ray2_0001_0.mp4"}
    assert len({e["prompt"] for e in out}) == 2  # the second family took the unused prompt
    assert all(src.validate_entry(e)[0] == src.ELIGIBLE for e in out)
    assert all(e["expected_digest"]["value"] for e in out)


def test_ugc_one_clip_per_creator_and_only_camera_categories():
    att = "\n".join([
        "Vlog_720P-aaaa is an audio-removed excerpt from 'A' by 'Ann' licensed under CC BY 4.0",
        "Vlog_1080P-bbbb is an audio-removed excerpt from 'B' by 'Ann' licensed under CC BY 4.0",
        "Lecture_720P-cccc is an audio-removed excerpt from 'C' by 'Cy' licensed under CC BY 4.0",
        "Gaming_720P-dddd is an audio-removed excerpt from 'D' by 'Di' licensed under CC BY 4.0",
        "Vlog_360P-eeee is an audio-removed excerpt from 'E' by 'Ed' licensed under CC BY 4.0"])
    h264 = [{"name": f"original_videos_h264/{s}_crf_10_ss_00_t_20.0.mp4", "generation": "1",
             "md5Hash": "m", "size": "5"} for s in
            ("Vlog_720P-aaaa", "Vlog_1080P-bbbb", "Lecture_720P-cccc", "Gaming_720P-dddd",
             "Vlog_360P-eeee")]
    raw = [{"name": "original_videos/Vlog/720P/Vlog_720P-aaaa.mkv",
            "timeCreated": "2019-02-01T00:00:00Z"},
           {"name": "original_videos/Vlog/1080P/Vlog_1080P-bbbb.mkv",
            "timeCreated": "2019-02-01T00:00:00Z"}]
    out = src.ugc_entries(att, {"generation": "9"}, h264, raw)
    ids = [e["item_id"] for e in out]
    assert len([i for i in ids if "Vlog" in i]) == 1  # Ann once; 360P never
    assert not any("Gaming" in i for i in ids)
    lecture = next(e for e in out if "Lecture" in e["item_id"])
    assert lecture["provenance_status"] == "UNCLEAR"  # no dated raw original
    assert all(e["edit_screen_required"] and e["expected_truth"] == src.VERIFIED_REAL
               for e in out)


def test_mirror_pattern_spans_a_line_break():
    text = 'RemoteSource("k", "34data/gen-videos-kling",\n        "' + "a" * 40 + '", "d.zip")'
    assert src.MIRROR_RE.findall(text) == [("34data/gen-videos-kling", "a" * 40)]


# --- anchor generation end to end -----------------------------------------------------------

def build_inputs(tmp_path):
    intake = tmp_path / "intake"
    intake.mkdir()
    face_err = {**src._face_base(), "status": "ERROR", "error": "annotation halted: x"}
    e_ai = entry(item_id="ai:1", expected_truth=src.VERIFIED_AI, edit_screen_required=False,
                 generator_family="Kling", generator_model="Kling v2.1", source_cluster="p1",
                 capture_domain=None, provenance_tier="third_party_benchmark_generation_record")
    e_real = entry(item_id="real:1", source_cluster="u1")
    e_real_cut = entry(item_id="real:2", source_cluster="u2")
    e_cand = entry(item_id="cand:1", license_id=None,
                   license_or_usage_basis="official promotional media")
    spec = {"entries": [e_ai, e_real, e_real_cut], "candidate_only_sources": [e_cand],
            "visual_review": {"real:1": {"verdict": "PASS", "reason": ""},
                              "real:2": {"verdict": "PASS", "reason": ""}},
            "edit_screen": {"threshold": 0.4}}
    ledger = {"items": {
        "ai:1": {**LED_OK, "item_id": "ai:1", "sha256": "a" * 64, "path": "g/p1_aaa.mp4",
                 "face_present": face_err},
        "real:1": {**LED_OK, "item_id": "real:1", "sha256": "b" * 64, "path": "g/u1_bbb.mp4",
                   "face_present": face_err},
        "real:2": {**LED_OK, "item_id": "real:2", "sha256": "c" * 64, "path": "g/u2_ccc.mp4",
                   "scene_cuts": 3, "face_present": face_err}}}
    vcd_prov = {"ground_truth": src.VERIFIED_REAL, "source_cluster_id": "vcd1_subject_1",
                "source_collection_id": "microsoft/VCD", "capture_domain": "webcam_talking_head_scenario",
                "generator_family": None}
    audit = {"face_guard": "halted", "benchmark_mirrors": [{}], "rapidata": {},
             "t1_carry_forward_face": {"d" * 64: face_err},
             "vcd1": {"clips": [
                 {"item_id": "vcd1:a", "sha256": "e" * 64, "canonical_path": "v/a.mp4",
                  "decision": src.VERIFIED_REAL, "decision_reason": None,
                  "technical_status": "INSPECTABLE", "technical_reason": None,
                  "ground_truth_provenance": vcd_prov, "face_present": face_err,
                  "speech_primary_annotation": None},
                 {"item_id": "vcd1:b", "sha256": "f" * 64, "canonical_path": "v/b.mp4",
                  "decision": src.CANDIDATE_ONLY, "decision_reason": "failed: upstream",
                  "technical_status": "INSPECTABLE", "technical_reason": None,
                  "ground_truth_provenance": vcd_prov, "face_present": face_err,
                  "speech_primary_annotation": None}]}}
    t1_inv = {"items": [
        {"sha256": "d" * 64, "canonical_path": "r17/AI-1.mp4", "subcorpus": "r17-ai-visual-v1",
         "confidence_classification": "VERIFIED_AI", "technical_status": "INSPECTABLE",
         "classification_rationale": "frozen manifest", "format": {"duration_s": 8},
         "associated_metadata_records": [{"trusted": True, "generator_or_source": "Runway Gen-4",
                                          "source_cluster_id": "CL-RUNWAY-GEN4"}]},
        {"sha256": "9" * 64, "canonical_path": "x/y.mp4", "subcorpus": "r7t9",
         "confidence_classification": "LABEL_PRESENT_BUT_GT_INSUFFICIENT",
         "technical_status": "INSPECTABLE", "associated_metadata_records": []}]}
    for name, payload in (("spec.json", spec), ("t1.json", t1_inv)):
        (tmp_path / name).write_text(json.dumps(payload))
    (intake / src.LEDGER_NAME).write_text(json.dumps(ledger))
    (intake / src.AUDIT_NAME).write_text(json.dumps(audit))
    return intake


def run_anchor(tmp_path, intake, out="anchor.json"):
    src.main(["generate-anchor", "--spec", str(tmp_path / "spec.json"), "--intake", str(intake),
              "--t1-inventory", str(tmp_path / "t1.json"), "--out", str(tmp_path / out)])
    return json.loads((tmp_path / out).read_text())


def test_anchor_admits_only_verified_and_records_why_others_are_not(tmp_path):
    a = run_anchor(tmp_path, build_inputs(tmp_path))
    shas = {r["sha256"] for r in a["records"]}
    assert shas == {"a" * 64, "b" * 64, "e" * 64}
    excluded = {x["item_id"]: x["class"] for x in a["not_admitted"]}
    assert excluded == {"cand:1": src.CANDIDATE_ONLY, "real:2": src.AMBIGUOUS,
                        "vcd1:b": src.CANDIDATE_ONLY,
                        "t1:r17/AI-1.mp4": "EXCLUDED_USAGE_NOT_ESTABLISHED"}
    assert "9" * 64 not in shas  # T1 candidate stock is never self-promoted


def test_anchor_keeps_gt_when_every_annotation_failed(tmp_path):
    a = run_anchor(tmp_path, build_inputs(tmp_path))
    assert all(r["derived_annotations"]["face_present"]["status"] == "ERROR" for r in a["records"])
    assert Counter(r["ground_truth_provenance"]["ground_truth"] for r in a["records"]) == \
        {src.VERIFIED_AI: 1, src.VERIFIED_REAL: 2}


def test_carried_forward_without_usage_basis_is_not_admitted(tmp_path):
    a = run_anchor(tmp_path, build_inputs(tmp_path))
    assert not any(r["origin"] == "r18_t1_carry_forward" for r in a["records"])
    assert "d" * 64 not in {r["sha256"] for r in a["records"]}
    s = a["summary"]
    assert s["cluster_diversity"]["all"]["clusters"] == 3


def test_t1_real_with_established_usage_is_carried_with_a_collection(tmp_path):
    inv = {"items": [{"sha256": "7" * 64, "canonical_path": "r16/a.mov", "subcorpus": "r16_pilot",
                      "confidence_classification": "VERIFIED_REAL",
                      "technical_status": "INSPECTABLE", "classification_rationale": "owner",
                      "associated_metadata_records": [{"trusted": True,
                                                       "source_cluster_id": "r16-s1"}]}]}
    (tmp_path / "t1.json").write_text(json.dumps(inv))
    out, excluded = src.t1_carry_forward(tmp_path / "t1.json", {})
    assert excluded == []
    p = out[0]["ground_truth_provenance"]
    assert p["usage_rights_status"] == "ESTABLISHED"
    assert p["source_collection_id"] == "r16-owner-pilot-2026-10-07"


def test_every_admitted_record_carries_cluster_and_collection(tmp_path):
    a = run_anchor(tmp_path, build_inputs(tmp_path))
    for r in a["records"]:
        assert r["ground_truth_provenance"]["source_cluster_id"]
        assert r["ground_truth_provenance"]["source_collection_id"]
    assert a["summary"]["collection_diversity"]["all"]["clusters"] == 2


def test_anchor_refuses_a_record_without_collection(tmp_path):
    intake = build_inputs(tmp_path)
    audit = json.loads((intake / src.AUDIT_NAME).read_text())
    audit["vcd1"]["clips"][0]["ground_truth_provenance"]["source_collection_id"] = None
    (intake / src.AUDIT_NAME).write_text(json.dumps(audit))
    with pytest.raises(src.Refusal):
        run_anchor(tmp_path, intake)


# --- D-B: Rapidata acceptance guard ---------------------------------------------------------

RAPI = "Rapidata/x"


def rapi_entry(name="kling-v2.1-master-25-7-25_3_0.mp4", model="Kling v2.1 Master (Kuaishou)"):
    return entry(item_id=f"rapidata:x:{name}", source_collection_id=RAPI,
                 expected_truth=src.VERIFIED_AI, generator_model=model)


def rapi_audit(accepted=True, rows=None):
    checks = {"main_is_pinned_revision": True, "card_license_field_apache_2_0": accepted,
              "repository_holds_data_only": True, "no_other_licence_file": True}
    return {"rapidata": {RAPI: {"accepted": accepted, "checks": checks, "file_model_rows":
                                rows if rows is not None else
                                {"kling-v2.1-master-25-7-25_3_0.mp4":
                                 "kling-v2.1-master-25-7-25"}}}}


def test_rapidata_guard_passes_with_collection_licence_and_per_file_record():
    assert src.rapidata_decision(rapi_entry(), rapi_audit()) is None


def test_rapidata_guard_rejects_a_failed_collection_check():
    assert "card_license_field_apache_2_0" in src.rapidata_decision(rapi_entry(),
                                                                    rapi_audit(accepted=False))


def test_rapidata_guard_rejects_missing_or_contradicting_per_file_record():
    assert "per-file" in src.rapidata_decision(rapi_entry(), rapi_audit(rows={}))
    wrong = rapi_audit(rows={"kling-v2.1-master-25-7-25_3_0.mp4": "ray2"})
    assert "per-file" in src.rapidata_decision(rapi_entry(), wrong)
    assert "per-file" in src.rapidata_decision(rapi_entry(model="Luma Ray 2"), rapi_audit())
    assert src.rapidata_decision(rapi_entry(), {}) is not None


def test_rapidata_entry_failing_the_guard_is_a_candidate_in_the_anchor(tmp_path):
    intake = build_inputs(tmp_path)
    spec = json.loads((tmp_path / "spec.json").read_text())
    spec["entries"][0]["source_collection_id"] = RAPI  # ai:1, no Rapidata evidence recorded
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    a = run_anchor(tmp_path, intake)
    assert "a" * 64 not in {r["sha256"] for r in a["records"]}
    assert {x["item_id"]: x["class"] for x in a["not_admitted"]}["ai:1"] == src.CANDIDATE_ONLY


def test_anchor_is_byte_identical_on_rerun(tmp_path):
    intake = build_inputs(tmp_path)
    run_anchor(tmp_path, intake, "a1.json")
    run_anchor(tmp_path, intake, "a2.json")
    assert (tmp_path / "a1.json").read_bytes() == (tmp_path / "a2.json").read_bytes()


def test_anchor_refuses_to_anchor_the_same_bytes_twice(tmp_path):
    intake = build_inputs(tmp_path)
    ledger = json.loads((intake / src.LEDGER_NAME).read_text())
    ledger["items"]["real:1"]["sha256"] = "a" * 64
    (intake / src.LEDGER_NAME).write_text(json.dumps(ledger))
    with pytest.raises(src.Refusal):
        run_anchor(tmp_path, intake)


# --- isolation ------------------------------------------------------------------------------

def test_candidates_are_never_downloaded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(src, "download", lambda *a, **k: calls.append(a) or
                        {"status": "DOWNLOAD_FAILED", "error": "test"})
    spec = {"entries": [entry(item_id="cand", license_id=None)], "candidate_only_sources": []}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    src.main(["ingest-external", "--spec", str(tmp_path / "spec.json"), "--intake",
              str(tmp_path / "intake"), "--face-model", str(tmp_path / "none.onnx")])
    assert calls == []


def test_no_production_detector_api_or_audio_model_is_referenced():
    text = MODULE_PATH.read_text().lower()
    for banned in ("/api/v1/analyses", "lipforensics", "efficientnet", "facetorch", "nvcf",
                   "webrtcvad", "silero", "whisper", "pyannote", "localhost", "from apps",
                   "import apps"):
        assert banned not in text, banned


def test_only_ffmpeg_and_ffprobe_are_spawned():
    tree = ast.parse(MODULE_PATH.read_text())
    programs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and node.elts and isinstance(node.elts[0], ast.Constant) \
                and node.elts[0].value in ("ffmpeg", "ffprobe", "docker", "curl", "python3"):
            programs.add(node.elts[0].value)
    assert programs == {"ffmpeg"}  # ffprobe is reached through r18_t1_inventory.run_ffprobe
    assert "subprocess.Popen" not in MODULE_PATH.read_text()


def test_reingest_refreshes_only_a_stale_annotation_and_reuses_media(tmp_path, monkeypatch):
    intake = tmp_path / "intake"
    media = intake / "g/c1_abc.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    old = {"item_id": "x:1", "ingest_status": "OK", "path": "g/c1_abc.mp4",
           "sha256": hashlib.sha256(b"media").hexdigest(), "technical_status": "INSPECTABLE",
           "scene_cuts": 0, "format": {"duration_s": 2.0},
           "face_present": {**src._face_base(), "model_sha256": "0b6c" + "0" * 60,
                            "status": "ERROR"}}
    monkeypatch.setattr(src, "download", lambda *a, **k: pytest.fail("media re-downloaded"))
    monkeypatch.setattr(src, "face_annotation",
                        lambda *a: {**src._face_base(), "status": "DETERMINED", "value": False})
    new = src.ingest_one(entry(), intake, "m.onnx", {"x:1": old})
    assert new["face_present"]["status"] == "DETERMINED"
    assert {k: v for k, v in new.items() if k != "face_present"} == \
        {k: v for k, v in old.items() if k != "face_present"}


# --- D-H: native + long-side-640 preprocessing ----------------------------------------------

@pytest.mark.parametrize("w,h,expected", [(1080, 1920, (360, 640)), (1920, 1080, (640, 360)),
                                          (640, 480, None), (600, 300, None),
                                          (1281, 721, (640, 360)), (1000, 1000, (640, 640))])
def test_secondary_size_preserves_aspect_and_skips_small_frames(w, h, expected):
    assert src.secondary_size(w, h) == expected


def run_multiscale(monkeypatch, scores, w, h, duration):
    monkeypatch.setattr(src, "face_guard", lambda *a: None)
    FakeCv2.FaceDetectorYN.scores = list(scores)
    FakeCv2.resized = []
    frame = np.zeros((h, w, 3), np.uint8)
    return src.face_annotation("v.mp4", duration, "m.onnx", FakeCv2, grab=lambda p, t: frame)


def test_frame_is_positive_when_only_the_640_copy_detects(monkeypatch):
    # 1080x1920, 2 frames: native misses both, secondary hits both
    out = run_multiscale(monkeypatch, [None, 0.93, None, 0.91], 1080, 1920, 1.0)
    assert out["value"] is True and out["frames_with_face"] == 2
    assert (out["frames_with_face_native"], out["frames_with_face_secondary"]) == (0, 2)
    assert FakeCv2.resized == [((360, 640), "inter-area")] * 2


def test_either_scale_counts_once_and_threshold_is_unchanged(monkeypatch):
    # frames: (native 0.9, sec 0.9) -> 1 ; (0.79, 0.79) -> 0 ; (None, 0.5) -> 0 ; (0.85, None) -> 1
    out = run_multiscale(monkeypatch, [0.9, 0.9, 0.79, 0.79, None, 0.5, 0.85, None],
                         1920, 1080, 3.0)
    assert out["frames_decoded"] == 4 and out["frames_with_face"] == 2
    assert out["value"] is True and out["confidence"] == 0.5


def test_small_frames_are_scored_at_native_scale_only(monkeypatch):
    out = run_multiscale(monkeypatch, [0.95, None], 640, 360, 1.0)
    assert FakeCv2.resized == [] and out["frames_with_face"] == 1
    assert out["preprocessing"] == src.FACE_SPEC["preprocessing"]


def test_reingest_refreshes_an_annotation_made_under_older_preprocessing(tmp_path, monkeypatch):
    intake = tmp_path / "intake"
    media = intake / "g/c1_abc.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    stale = {**src._face_base(), "status": "DETERMINED", "value": False}
    stale.pop("preprocessing")
    old = {"item_id": "x:1", "ingest_status": "OK", "path": "g/c1_abc.mp4",
           "sha256": hashlib.sha256(b"media").hexdigest(), "format": {"duration_s": 2.0},
           "face_present": stale}
    monkeypatch.setattr(src, "face_annotation",
                        lambda *a: {**src._face_base(), "status": "DETERMINED", "value": True})
    assert src.ingest_one(entry(), intake, "m.onnx", {"x:1": old})["face_present"]["value"]


def test_annotate_rewrites_only_face_annotations(tmp_path, monkeypatch):
    intake = build_inputs(tmp_path)
    for rel in ("g/p1_aaa.mp4", "g/u1_bbb.mp4", "g/u2_ccc.mp4"):
        (intake / rel).parent.mkdir(parents=True, exist_ok=True)
        (intake / rel).write_bytes(b"x")
    before_l = json.loads((intake / src.LEDGER_NAME).read_text())
    before_a = json.loads((intake / src.AUDIT_NAME).read_text())
    monkeypatch.setattr(src, "face_annotation",
                        lambda *a: {**src._face_base(), "status": "DETERMINED", "value": True})
    src.main(["annotate", "--intake", str(intake), "--corpus", str(tmp_path), "--t1-inventory",
              str(tmp_path / "t1.json"), "--face-model", str(tmp_path / "none.onnx")])
    after_l = json.loads((intake / src.LEDGER_NAME).read_text())
    after_a = json.loads((intake / src.AUDIT_NAME).read_text())
    for k, r in after_l["items"].items():
        assert r["face_present"]["value"] is True
        assert {x: v for x, v in r.items() if x != "face_present"} == \
            {x: v for x, v in before_l["items"][k].items() if x != "face_present"}
    for c, b in zip(after_a["vcd1"]["clips"], before_a["vcd1"]["clips"]):
        assert c["ground_truth_provenance"] == b["ground_truth_provenance"]
        assert c["decision"] == b["decision"] and c["face_present"]["value"] is True
