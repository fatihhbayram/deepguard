"""R18-T1 inventory: dedup, conflict quarantine, trust anchors, no detectors, technical status (synthetic trees only)."""

import hashlib
import json
import shutil
import subprocess
import sys

import pytest

from eval import r18_t1_inventory as inv

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(root, rel, data=b"x"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def rec(claim, trusted=False, source="m.csv", **extra):
    return inv.record("c", source, claim, claim, trusted, "basis-" + source, **extra)


def ok_probe(duration="8.0", streams=None):
    return {"format": {"format_name": "mov,mp4", "duration": duration},
            "streams": streams if streams is not None else [
                {"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720,
                 "avg_frame_rate": "30/1", "nb_frames": "240"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "44100",
                 "channels": 2}]}


def run_on(tmp_path, corpus, monkeypatch, probe=None):
    if probe is not None:
        monkeypatch.setattr(inv, "run_ffprobe", probe)
    out_inv, out_gap = tmp_path / "out/inv.json", tmp_path / "out/gap.json"
    inv.main(["--corpus", str(corpus), "--inventory", str(out_inv), "--gap", str(out_gap),
              "--workers", "2"])
    items = json.loads(out_inv.read_text())["items"]
    return {i["canonical_path"]: i for i in items}, json.loads(out_gap.read_text())


def fake_probe(path):
    return ok_probe(), None


# --- deduplication and canonical path -------------------------------------------------------

def test_canonical_path_is_lexicographically_smallest():
    assert inv.canonical_path(["r7t9/clips/a.mp4", "r7t8/clips/z.mp4", "r7t8-v1/clips/a.mp4"]) \
        == "r7t8-v1/clips/a.mp4"


def test_identical_bytes_collapse_to_one_canonical_entry(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "zeta/clips/one.mp4", b"same")
    put(corpus, "alpha/clips/two.mp4", b"same")
    put(corpus, "alpha/clips/other.mp4", b"different")
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert set(items) == {"alpha/clips/two.mp4", "alpha/clips/other.mp4"}
    dup = items["alpha/clips/two.mp4"]
    assert dup["duplicate_instances"] == ["alpha/clips/two.mp4", "zeta/clips/one.mp4"]
    assert dup["sha256"] == sha(b"same")
    s = gap["stock_discovery"]
    assert (s["media_paths"], s["unique_sha256"], s["duplicate_paths"], s["duplicate_bytes"]) \
        == (3, 2, 1, 4)


def test_records_from_every_instance_are_preserved(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "b/clips/x.mp4", b"same")
    put(corpus, "a/clips/y.mp4", b"same")
    (corpus / "a/manifest.csv").write_text("clip_id,path,label\ny,clips/y.mp4,face_swap\n")
    (corpus / "b/manifest.csv").write_text("clip_id,path,label\nx,clips/x.mp4,synthetic\n")
    items, _ = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    item = items["a/clips/y.mp4"]
    assert sorted(r["source_manifest"] for r in item["associated_metadata_records"]) \
        == ["a/manifest.csv", "b/manifest.csv"]
    # AI_GENERATED vs FACE_SWAP is a vocabulary difference, not a truth contradiction.
    assert item["has_metadata_conflict"] is False
    assert item["confidence_classification"] == inv.INSUFFICIENT


# --- truth conflict quarantine --------------------------------------------------------------

def test_contradictory_claims_are_quarantined_as_ambiguous():
    cls, why, conflict = inv.classify([rec(inv.REAL, source="a.csv"), rec(inv.AI, source="b.csv")])
    assert (cls, conflict) == (inv.AMBIGUOUS, True)
    assert "a.csv=REAL" in why and "b.csv=AI_GENERATED" in why


def test_a_trusted_claim_does_not_win_a_conflict():
    # No precedence: a verified REAL record against an unreviewed AI label is still a conflict.
    cls, _, conflict = inv.classify([rec(inv.REAL, trusted=True), rec(inv.AI, source="x.csv")])
    assert (cls, conflict) == (inv.AMBIGUOUS, True)


def test_conflicts_are_listed_in_the_registry(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "a/clips/x.mp4", b"same")
    put(corpus, "b/clips/x.mp4", b"same")
    (corpus / "a/manifest.csv").write_text("clip_id,path,label\nx,clips/x.mp4,real\n")
    (corpus / "b/manifest.csv").write_text("clip_id,path,label\nx,clips/x.mp4,synthetic\n")
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items["a/clips/x.mp4"]["confidence_classification"] == inv.AMBIGUOUS
    reg = gap["metadata_conflict_registry"]
    assert [r["sha256"] for r in reg] == [sha(b"same")]
    assert reg[0]["claims"] == ["a/manifest.csv: real", "b/manifest.csv: synthetic"]
    assert gap["ground_truth_confidence"]["unique"][inv.AMBIGUOUS] == 1


# --- trust allowlist and anchors ------------------------------------------------------------

def r17_tree(corpus, tmp_path, clip=b"sora-clip", real_clip=b"nasa-clip"):
    base = corpus / "r17-ai-visual-v1"
    base.mkdir(parents=True)
    cases = [
        {"case_id": "AI-001", "truth": "AI", "generator_or_source": "Sora",
         "validation_status": "VALIDATED", "clip_sha256": sha(clip), "source_cluster_id": "CL-S"},
        {"case_id": "REAL-001", "truth": "REAL", "generator_or_source": "NASA",
         "validation_status": "VALIDATED", "clip_sha256": sha(real_clip),
         "source_cluster_id": "CL-N"}]
    (base / "manifest.json").write_text(json.dumps(cases))
    (base / "manifest.csv").write_text("case_id\nAI-001\n")
    (base / "FREEZE.json").write_text(json.dumps({"source_clusters": []}))
    digests = {f: inv.sha256_file(base / f) for f in ("FREEZE.json", "manifest.json",
                                                      "manifest.csv")}
    spec = tmp_path / "digests.json"
    spec.write_text(json.dumps({"sha256": digests}))
    put(corpus, "r17_t3_work/phase_a_input/B001.mp4", clip)
    put(corpus, "r17_t3_work/phase_a_input/B002.mp4", real_clip)
    return spec


def test_allowlisted_corpus_with_matching_anchor_yields_verified(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    monkeypatch.setattr(inv, "R17_DIGESTS", r17_tree(corpus, tmp_path))
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items["r17_t3_work/phase_a_input/B001.mp4"]["confidence_classification"] \
        == inv.VERIFIED_AI
    assert items["r17_t3_work/phase_a_input/B002.mp4"]["confidence_classification"] \
        == inv.VERIFIED_REAL
    assert gap["verified_ai_by_generator"]["inspectable"] == {"Sora": 1}
    assert next(a for a in gap["trust_anchors"] if a["corpus"] == "r17-ai-visual-v1")["verified"]


def test_anchor_mismatch_withholds_verified_status(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    monkeypatch.setattr(inv, "R17_DIGESTS", r17_tree(corpus, tmp_path))
    (corpus / "r17-ai-visual-v1/manifest.csv").write_text("case_id\nAI-001\ntampered\n")
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items["r17_t3_work/phase_a_input/B001.mp4"]["confidence_classification"] \
        == inv.INSUFFICIENT
    assert not next(a for a in gap["trust_anchors"]
                    if a["corpus"] == "r17-ai-visual-v1")["verified"]
    assert gap["gap_vs_r18_target"]["verified_and_inspectable"]["total"] == 0


def test_folder_names_are_never_ground_truth(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    for rel in ("real/a.mp4", "generated/b.mp4", "deepfake/c.mp4", "genuine/d.mp4"):
        put(corpus, rel, rel.encode())
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert {i["confidence_classification"] for i in items.values()} == {inv.UNLABELED}
    assert gap["ground_truth_confidence"]["unique"][inv.UNLABELED] == 4


def test_unreviewed_manifest_label_is_insufficient_not_verified(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "r7t9/clips/x.mp4", b"x")
    (corpus / "r7t9/manifest.csv").write_text("clip_id,path,label,family\nx,clips/x.mp4,real,"
                                              "phone\n")
    items, _ = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items["r7t9/clips/x.mp4"]["confidence_classification"] == inv.INSUFFICIENT


def test_manifest_rows_outside_the_root_are_counted_not_followed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(tmp_path, "private/y.mp4", b"y")
    put(corpus, "r7t5/clips/x.mp4", b"x")
    (corpus / "r7t5/manifest.csv").write_text(
        f"clip_id,path,label\nx,clips/x.mp4,real\ny,{tmp_path}/private/y.mp4,real\n"
        "z,clips/missing.mp4,real\n")
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert list(items) == ["r7t5/clips/x.mp4"]
    assert gap["stock_discovery"]["manifest_rows_with_unresolved_paths"] == {
        "missing_inside_root": {"r7t5/manifest.csv": 1},
        "pointing_outside_root": {"r7t5/manifest.csv": 1}}


def test_anchored_but_not_allowlisted_corpus_grants_nothing(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "external-genuine-vcd1/clips/th/a.mp4", b"vcd")
    manifest = {"file_records": [{"subject_id": "a", "subset": "th",
                                  "sha256_local_freeze": sha(b"vcd")}]}
    put(corpus, "external-genuine-vcd1/vcd1_intake_manifest.json", json.dumps(manifest).encode())
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items["external-genuine-vcd1/clips/th/a.mp4"]["confidence_classification"] \
        == inv.INSUFFICIENT
    assert gap["allowlist_candidates_not_granted"]["external-genuine-vcd1"]["unique"] == 1


@pytest.mark.parametrize("trusted", [True, False])
def test_rendered_and_non_binary_truth_are_ambiguous_at_any_trust(trusted):
    cls, why, conflict = inv.classify([rec(inv.RENDERED, trusted=trusted)])
    assert (cls, conflict) == (inv.AMBIGUOUS, False) and "rendered" in why
    cls, _, conflict = inv.classify([rec(inv.MANIPULATED, trusted=trusted)])
    assert (cls, conflict) == (inv.AMBIGUOUS, False)


def test_matching_but_untracked_anchor_is_reported_and_grants_nothing():
    a = inv.anchor_check("x", {"m.json": "d"}, {"m.json": "d"}, tracked=False)
    assert (a["verified"], a["tracked"], a["grants_verified"]) == (True, False, False)
    assert inv.anchor_check("x", {"m.json": "d"}, {"m.json": "d"}, tracked=True)[
        "grants_verified"]
    assert not inv.anchor_check("x", {"m.json": "d"}, {"m.json": "e"}, tracked=True)[
        "grants_verified"]


def roots_script(tmp_path, roots):
    script = tmp_path / "r16_t6_pilot_dataset.py"
    body = ",\n".join(f'    "{k}": ("{v}", "notes")' for k, v in roots.items())
    script.write_text(f"import this_module_does_not_exist\nROOTS = {{\n{body},\n}}\n")
    return script


def test_r16_roots_are_read_without_executing_the_script(tmp_path):
    # The import at the top would raise if the script were executed.
    script = roots_script(tmp_path, {"IMG_1": "aa", "IMG_2": "bb"})
    assert inv.tracked_r16_roots(script) == {"aa": "IMG_1", "bb": "IMG_2"}


def test_the_committed_r16_build_pins_ten_owner_originals():
    assert len(inv.tracked_r16_roots()) == 10


def test_r16_only_tracked_originals_are_verified(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    blobs = {"orig": b"camera", "unpinned": b"camera2", "deriv": b"p1", "edit": b"splice"}
    for name, data in blobs.items():
        put(corpus, f"r16_pilot/media/{name}.mov", data)
    case = lambda cid, data, truth, label, sc: {  # noqa: E731
        "case_id": cid, "sha256": sha(data), "truth": truth, "label": label, "source_class": sc}
    gt = {"cases": [case("P1", blobs["orig"], "genuine", "GENUINE", "OWNER_KNOWN"),
                    case("P2", blobs["unpinned"], "genuine", "GENUINE", "OWNER_KNOWN"),
                    case("P3", blobs["deriv"], "genuine", "GENUINE", "CONTROLLED_TEST"),
                    case("P4", blobs["edit"], "manipulated", "OTHER_MANIPULATION",
                         "CONTROLLED_TEST")]}
    put(corpus, "r16_pilot/sealed/pilot_ground_truth.json", json.dumps(gt).encode())
    monkeypatch.setattr(inv, "R16_ROOTS_SCRIPT", roots_script(tmp_path, {"IMG_1":
                                                                         sha(blobs["orig"])}))
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    got = {p.split("/")[-1]: i["confidence_classification"] for p, i in items.items()}
    assert got == {"orig.mov": inv.VERIFIED_REAL, "unpinned.mov": inv.INSUFFICIENT,
                   "deriv.mov": inv.INSUFFICIENT, "edit.mov": inv.AMBIGUOUS}
    r16 = next(a for a in gap["trust_anchors"] if a["corpus"] == "r16_pilot")
    assert r16["tracked"] is False and r16["tracked_sub_anchor"]["matched_in_corpus"] == 1


def test_turnturk_animation_is_ambiguous_and_its_audio_not_verified(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    base = "external-genuine-real-turnturk"
    put(corpus, f"{base}/dataset/rec01/rec01__animation.mp4", b"anim")
    put(corpus, f"{base}/dataset/rec01/rec01__L_p06.webm", b"voice")
    files = [{"path": "rec01/rec01__animation.mp4", "modality": "video+audio",
              "file_role": "derived face-vector animation",
              "carries_original_human_video_pixels": False, "sha256": sha(b"anim")},
             {"path": "rec01/rec01__L_p06.webm", "modality": "audio",
              "file_role": "per-speaker isolated audio",
              "carries_original_human_video_pixels": False, "sha256": sha(b"voice")}]
    put(corpus, f"{base}/manifest.json", json.dumps({"files": files}).encode())
    monkeypatch.setattr(inv, "TURNTURK_MANIFEST_SHA256", inv.sha256_file(corpus / base /
                                                                          "manifest.json"))
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items[f"{base}/dataset/rec01/rec01__animation.mp4"]["confidence_classification"] \
        == inv.AMBIGUOUS
    # The digest matches, but it is recorded only in git-excluded docs/ai/: no VERIFIED.
    assert items[f"{base}/dataset/rec01/rec01__L_p06.webm"]["confidence_classification"] \
        == inv.INSUFFICIENT
    a = next(a for a in gap["trust_anchors"] if a["corpus"] == base)
    assert (a["verified"], a["grants_verified"]) == (True, False)


def test_unlabeled_when_records_carry_no_claim():
    assert inv.classify([rec(None)])[0] == inv.UNLABELED
    assert inv.classify([])[0] == inv.UNLABELED


# --- no detector, no model, face fields stay null -------------------------------------------

def test_undocumented_face_fields_remain_null():
    assert inv.trusted_attributes([rec(inv.REAL, trusted=True)]) \
        == {"face_present": None, "talking_head": None}
    # An untrusted record cannot fill them either.
    assert inv.trusted_attributes([rec(inv.REAL, face_present=True)])["face_present"] is None
    assert inv.trusted_attributes([rec(inv.REAL, trusted=True, talking_head=True)])[
        "talking_head"] is True


def test_only_ffprobe_is_ever_executed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "a/clips/x.mp4", b"not really a video")
    calls, real_run = [], subprocess.run

    def spy(cmd, *a, **k):
        calls.append(cmd[0])
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(inv.subprocess, "run", spy)
    items, gap = run_on(tmp_path, corpus, monkeypatch)
    assert calls and set(calls) == {"ffprobe"}
    assert gap["facial_semantic_metadata"]["detector_or_vision_model_run"] is False
    assert all(v is None for v in items["a/clips/x.mp4"]["trusted_attributes"].values())


def test_module_imports_no_model_or_vision_library():
    banned = ("torch", "cv2", "onnxruntime", "tensorflow", "insightface", "mediapipe",
              "facenet_pytorch", "retinaface", "dlib", "requests", "urllib3", "httpx")
    src = open(inv.__file__).read()
    assert not any(f"import {b}" in src or f"from {b}" in src for b in banned)


def test_corpus_is_not_modified(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "a/clips/x.mp4", b"same")
    put(corpus, "b/clips/x.mp4", b"same")
    (corpus / "a/manifest.csv").write_text("clip_id,path,label\nx,clips/x.mp4,real\n")

    def snapshot():
        return sorted((p.relative_to(corpus).as_posix(), p.stat().st_size, p.stat().st_mtime_ns,
                       p.read_bytes()) for p in corpus.rglob("*") if p.is_file())
    before = snapshot()
    run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert snapshot() == before


def test_symlinks_and_excluded_trees_are_not_followed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    outside = put(tmp_path, "elsewhere/x.mov", b"outside")
    (corpus / "a").mkdir(parents=True)
    (corpus / "a/link.MOV").symlink_to(outside)
    put(corpus, "models/m.mp4", b"model-tree")
    put(corpus, "a/b.flv", b"flv")
    items, gap = run_on(tmp_path, corpus, monkeypatch, fake_probe)
    assert items == {}
    s = gap["stock_discovery"]
    assert [x["path"] for x in s["symlinks_not_followed"]] == ["a/link.MOV"]
    assert s["out_of_extension_media_not_classified"] == ["a/b.flv"]


# --- technical status ---------------------------------------------------------------------

def test_parsed_video_with_duration_is_inspectable():
    status, reason, profile = inv.technical_profile(ok_probe(), size_bytes=10)
    assert (status, reason) == (inv.INSPECTABLE, None)
    assert profile["video"] == {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0,
                                "frames": 240}
    assert profile["has_audio"] and profile["audio"]["sample_rate"] == 44100


@pytest.mark.parametrize("probe,size,reason", [
    (None, 10, "container did not parse"),
    (ok_probe(), 0, "zero-byte file"),
    (ok_probe(streams=[{"codec_type": "audio", "codec_name": "opus"}]), 10, "no video stream"),
    (ok_probe(duration="0"), 10, "no positive duration"),
    (ok_probe(streams=[{"codec_type": "video", "codec_name": "h264", "width": 0,
                        "height": 0}]), 10, "video stream has no resolution"),
    (ok_probe(streams=[{"codec_type": "video", "codec_name": "mjpeg", "width": 10, "height": 10,
                        "disposition": {"attached_pic": 1}}]), 10, "no video stream"),
])
def test_unreadable_cases(probe, size, reason):
    status, why, _ = inv.technical_profile(probe, size_bytes=size)
    assert (status, why) == (inv.UNREADABLE, reason)


def test_technical_status_is_independent_of_confidence():
    # An audio-only file of verified genuine speech: verified GT, yet not inspectable video.
    status, _, _ = inv.technical_profile(ok_probe(streams=[{"codec_type": "audio"}]), size_bytes=5)
    assert status == inv.UNREADABLE
    assert inv.classify([rec(inv.REAL, trusted=True)])[0] == inv.VERIFIED_REAL


def test_conflict_still_wins_over_content_ambiguity():
    cls, _, conflict = inv.classify([rec(inv.MANIPULATED, source="a"), rec(inv.REAL, source="b")])
    assert (cls, conflict) == (inv.AMBIGUOUS, True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_real_ffprobe_separates_video_from_garbage(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    put(corpus, "a/garbage.mp4", b"\x00" * 64)
    clip = corpus / "a/ok.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=duration=1:size=64x48:rate=10", "-pix_fmt", "yuv420p", str(clip)],
                   check=True)
    items, gap = run_on(tmp_path, corpus, monkeypatch)
    assert items["a/ok.mp4"]["technical_status"] == inv.INSPECTABLE
    assert items["a/ok.mp4"]["format"]["video"]["width"] == 64
    assert items["a/garbage.mp4"]["technical_status"] == inv.UNREADABLE
    assert gap["technical_inspectability"]["unique"] == {inv.INSPECTABLE: 1, inv.UNREADABLE: 1}
