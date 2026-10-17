"""R17-T2 harness: frozen mapping, blinding, hash gate, trust anchors and seal (synthetic data only)."""

import hashlib
import json
from pathlib import Path

import pytest

from eval import r17_t2_extract as ex
from eval import r17_t2_run as run
from eval import r17_t2_score as sc

RULES = "r9-v5.0.0"


# ---------------------------------------------------------------- synthetic dataset

def h(b):
    return hashlib.sha256(b).hexdigest()


def clip_bytes(case_id):
    return f"synthetic clip {case_id}".encode()


LAYOUT = [("AI", "CL-A1", 17), ("AI", "CL-A2", 11), ("AI", "CL-A3", 1), ("AI", "CL-A4", 1),
          ("REAL", "CL-R1", 10), ("REAL", "CL-R2", 5)]


def make_dataset(tmp_path):
    """A 45-case dataset shaped like R17-T1 (6 clusters, 30/15), plus tracked-style digests."""
    ds = tmp_path / "dataset"
    ds.mkdir()
    rows, clips, n = [], [], {"AI": 0, "REAL": 0}
    for truth, cl, count in LAYOUT:
        for _ in range(count):
            n[truth] += 1
            cid = f"{truth}-{n[truth]:03d}"
            sha = h(clip_bytes(cid))
            rows.append({"case_id": cid, "truth": truth, "generator_or_source": f"gen-{cl}",
                         "source_url": f"https://example.org/{cl}", "source_cluster_id": cl,
                         "clip_start": "00:00.000", "clip_end": "00:08.000", "clip_sha256": sha})
            clips.append({"case_id": cid, "source_cluster_id": cl, "clip_start": "00:00.000",
                          "clip_end": "00:08.000", "clip_sha256": sha})
    freeze = {"counts": {"cases": 45, "truth": {"AI": 30, "REAL": 15}, "source_clusters": 6},
              "tools": {"tool_image": "sha256:x"}, "clips": clips}
    (ds / "manifest.json").write_text(json.dumps(rows))
    (ds / "manifest.csv").write_text("case_id\n" + "\n".join(r["case_id"] for r in rows))
    (ds / "FREEZE.json").write_text(json.dumps(freeze))
    digests = tmp_path / "expected_digests.json"
    digests.write_text(json.dumps({"dataset_path": str(ds), "sha256": {
        k: ex.sha256_file(ds / k) for k in ex.ANCHORED}}))
    return ds, digests, rows


def extract(tmp_path, ds, rows, cut=None, tags=None):
    cases = [{**r} for r in rows]
    tokens = ex.leak_tokens([r["case_id"] for r in rows], [r["source_cluster_id"] for r in rows])
    cut = cut or (lambda c, dst: dst.write_bytes(clip_bytes(c["case_id"])))
    clips, bmap = tmp_path / "input", tmp_path / "map"
    code = ex.run_extraction(cases, clips, bmap, cut, tags or (lambda p: {"encoder": "Lavf"}), tokens)
    return code, clips, bmap


def detail(level, rules=RULES, status="completed"):
    return {"status": status, "risk_level": level, "risk_rule_id": "R", "risk_rules_version": rules,
            "risk_calibration_id": "C", "decision_state": "x", "aggregate_enrichment_state": "ENRICHMENT_COMPLETE",
            **{s: {"status": "SUCCESS", "score": 0.5} for s in run.SIGNALS}}


def phase_a(tmp_path, clips, outcome, monkeypatch):
    """Run the real runner loop against a fake product. `outcome(blind_id)` gives the detail."""
    _, cases = run.load_input(clips)
    out = tmp_path / "phase_a"
    out.mkdir()
    env = {"spec_sha256": {"scoring_freeze": ex.sha256_file(run.FREEZE_DOC),
                           "result_schema": ex.sha256_file(run.SCHEMA)},
           "repo_head": "0" * 40, "specs_dirty": True}
    run.write_json(out / "environment.json", env)
    (out / "execution_manifest.json").write_bytes((clips / "execution_manifest.json").read_bytes())

    class Fake:
        def upload(self, path):
            return 202, {"id": path.stem}

        def get(self, aid):
            return 200, outcome(aid)

    monkeypatch.setattr(run, "POLL_S", 0)
    code = run.execute(Fake(), cases, clips, out, {})
    return code, out


def seal(out):
    entries = run.seal_files(out)
    root = run.seal_root(entries)
    run.write_json(out / run.SEAL, {"schema": run.SEAL_SCHEMA, "files": entries})
    (out / run.SEAL_ROOT).write_text(root + "\n")
    return root


@pytest.fixture
def world(tmp_path):
    ds, digests, rows = make_dataset(tmp_path)
    code, clips, bmap = extract(tmp_path, ds, rows)
    assert code == 0
    return ds, digests, rows, clips, bmap


# ---------------------------------------------------------------- final state, precedence, invariant

@pytest.mark.parametrize("rec,state", [
    (None, "operational_failure"),
    ({"hash_blocked": True}, "hash_blocked"),
    ({"hash_blocked": True, **detail("MANIPULATION_DETECTED")}, "hash_blocked"),
    ({"status": "failed", "risk_rules_version": "other"}, "operational_failure"),
    ({"error": "upload refused: HTTP 500"}, "operational_failure"),
    ({"status": "processing", "timeout": "decision"}, "operational_failure"),
    ({**detail("MANIPULATION_DETECTED"), "timeout": "decision"}, "operational_failure"),
    (detail("INCONCLUSIVE", rules="r9-v4.0.0"), "protocol_violation"),
    (detail(None), "protocol_violation"),
    (detail("SOMETHING_ELSE"), "protocol_violation"),
    (detail("INCONCLUSIVE"), "abstention"),
    (detail("MANIPULATION_DETECTED"), "positive"),
    (detail("NO_CALIBRATED_MANIPULATION_SIGNAL"), "negative"),
    ({**detail("NO_CALIBRATED_MANIPULATION_SIGNAL"), "timeout": "enrichment"}, "negative"),
])
def test_final_state_rows_and_precedence(rec, state):
    assert sc.final_state(rec) == state


def test_invariant_holds_and_breaks():
    rows = [{"truth": "AI", "final_state": s} for s in sc.STATES] * 5
    rows = rows[:30] + [{"truth": "REAL", "final_state": "negative"}] * 15
    sc.check_invariant(rows, sc.FROZEN_COUNTS)
    with pytest.raises(sc.Refusal):
        sc.check_invariant(rows[:-1], sc.FROZEN_COUNTS)
    with pytest.raises(sc.Refusal):
        sc.check_invariant(rows[1:] + [{"truth": "REAL", "final_state": "negative"}], sc.FROZEN_COUNTS)


def test_every_case_has_exactly_one_state(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    levels = ["MANIPULATION_DETECTED", "NO_CALIBRATED_MANIPULATION_SIGNAL", "INCONCLUSIVE", None]
    code, out = phase_a(tmp_path, clips, lambda b: detail(levels[int(b[1:]) % 4]), monkeypatch)
    assert code == 0
    root = seal(out)
    result, _ = sc.score(out, root, bmap / "blind_map.json", ds, digests)
    st = result["case_level"]["states"]
    assert sum(st.values()) == 45 and len(result["cases"]) == 45
    assert sum(1 for r in result["cases"] if r["truth"] == "AI") == 30


# ---------------------------------------------------------------- confusion and intervals

def test_confusion_cells():
    assert sc.confusion("AI", "positive") == "TP"
    assert sc.confusion("AI", "negative") == "FN"
    assert sc.confusion("REAL", "positive") == "FP"
    assert sc.confusion("REAL", "negative") == "TN"
    for s in ("abstention", "operational_failure", "protocol_violation", "hash_blocked"):
        assert sc.confusion("AI", s) is None and sc.confusion("REAL", s) is None


def test_zero_denominator_is_na_and_k0_has_interval():
    assert sc.interval(0, 0) == {"k": 0, "n": 0, "rate": None, "ci95": None}
    k0 = sc.interval(0, 15)
    assert k0["rate"] == 0.0 and k0["ci95"][0] == 0.0 and 0.2 < k0["ci95"][1] < 0.25
    m = sc.metrics([{"truth": "AI", "final_state": "abstention", "confusion": None}] * 3)
    assert m["TPR"]["rate"] is None and m["TPR"]["ci95"] is None
    assert m["FPR"]["rate"] is None and m["coverage_REAL"]["rate"] is None


def test_csv_writes_na_for_null(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    code, out = phase_a(tmp_path, clips, lambda b: detail("INCONCLUSIVE"), monkeypatch)
    root = seal(out)
    result, names = sc.score(out, root, bmap / "blind_map.json", ds, digests)
    assert result["case_level"]["TPR"]["rate"] is None
    sc.write_outputs(result, names, tmp_path / "b")
    lines = (tmp_path / "b/results.csv").read_text().splitlines()
    assert lines[0].split(",") == names and "N/A" in lines[1]


# ---------------------------------------------------------------- secondary recall

def test_secondary_recall_excludes_failures():
    rows = ([{"truth": "AI", "final_state": "positive", "confusion": "TP"}] * 4
            + [{"truth": "AI", "final_state": "negative", "confusion": "FN"}] * 2
            + [{"truth": "AI", "final_state": "abstention", "confusion": None}] * 3
            + [{"truth": "AI", "final_state": s, "confusion": None}
               for s in ("operational_failure", "protocol_violation", "hash_blocked")] * 5)
    m = sc.metrics(rows)
    sec = m["secondary_recall_abstention_as_miss"]
    assert (sec["k"], sec["n"]) == (4, 9) and sec["label"] == "secondary, abstention-as-miss"
    assert (m["TPR"]["k"], m["TPR"]["n"]) == (4, 6)
    assert m["states"]["operational_failure"] == 5 and m["states"]["hash_blocked"] == 5


# ---------------------------------------------------------------- cluster level

def test_cluster_level_na_and_k_from_freeze(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    _, out = phase_a(tmp_path, clips, lambda b: detail("NO_CALIBRATED_MANIPULATION_SIGNAL"), monkeypatch)
    root = seal(out)
    result, _ = sc.score(out, root, bmap / "blind_map.json", ds, digests)
    cl = result["cluster_level"]
    assert cl["K"] == 6 and cl["label"] == "6 source clusters, not independent, not generalisable"
    assert cl["K_source"] == "FREEZE.json counts.source_clusters"
    real_only, ai_only = cl["clusters"]["CL-R1"], cl["clusters"]["CL-A1"]
    assert real_only["TPR"]["rate"] is None and real_only["secondary_recall_abstention_as_miss"]["rate"] is None
    assert real_only["FPR"]["rate"] == 0.0
    assert ai_only["FPR"]["rate"] is None and ai_only["TNR"]["rate"] is None
    assert ai_only["TPR"]["rate"] == 0.0


def test_k_cross_check_refuses_mismatch(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    freeze = json.loads((ds / "FREEZE.json").read_text())
    freeze["counts"]["source_clusters"] = 7
    (ds / "FREEZE.json").write_text(json.dumps(freeze))
    digests.write_text(json.dumps({"dataset_path": str(ds), "sha256": {
        k: ex.sha256_file(ds / k) for k in ex.ANCHORED}}))
    _, out = phase_a(tmp_path, clips, lambda b: detail("INCONCLUSIVE"), monkeypatch)
    root = seal(out)
    with pytest.raises(sc.Refusal, match="K=7"):
        sc.score(out, root, bmap / "blind_map.json", ds, digests)


# ---------------------------------------------------------------- blinding

def test_execution_manifest_keys_and_names(world):
    ds, digests, rows, clips, bmap = world
    exe = json.loads((clips / "execution_manifest.json").read_text())
    assert set(exe) == {"schema", "cases"}
    tokens = ex.leak_tokens([r["case_id"] for r in rows], [r["source_cluster_id"] for r in rows])
    for c in exe["cases"]:
        assert set(c) == {"blind_id", "clip_path", "clip_sha256"}
        assert ex.leaks(c["blind_id"], tokens) == [] and ex.leaks(c["clip_path"], tokens) == []
    raw = (clips / "execution_manifest.json").read_text()
    assert ex.leaks(raw, tokens) == []
    assert sorted(p.name for p in clips.iterdir()) == sorted(
        ["execution_manifest.json"] + [f"B{i:03d}.mp4" for i in range(1, 46)])
    assert not (clips / "blind_map.json").exists() and (bmap / "blind_map.json").exists()


def test_leak_detector_catches_truth_and_generators():
    tokens = ex.leak_tokens(["AI-001", "REAL-002"], ["CL-SORA"])
    for bad in ("AI-001.mp4", "real-002", "clip_AI.mp4", "cl-sora", "Sora", "veo_x", "RUNWAY",
                "nasa", "artemis2", "SLS", "/tmp/ai/x", "REAL", "AI-123"):
        assert ex.leaks(bad, tokens), bad
    for ok in ("B001.mp4", "/tmp/scratch/input", "VideoHandler", "Lavf",
               "isomiso2avc1mp41", "SoundHandler", "und"):
        assert ex.leaks(ok, tokens) == [], ok


def test_leaking_tag_stops(tmp_path):
    ds, digests, rows = make_dataset(tmp_path)
    code, clips, bmap = extract(tmp_path, ds, rows, tags=lambda p: {"title": "Sora demo"})
    assert code == 3 and not (clips / "execution_manifest.json").exists()
    assert not (bmap / "blind_map.json").exists()


def test_leaking_path_stops(tmp_path):
    ds, digests, rows = make_dataset(tmp_path)
    tokens = ex.leak_tokens([r["case_id"] for r in rows], [r["source_cluster_id"] for r in rows])
    with pytest.raises(SystemExit, match="leaks"):
        ex.run_extraction(rows, tmp_path / "real_input", tmp_path / "m", lambda c, d: None,
                          lambda p: {}, tokens)


def test_order_is_truth_independent(world):
    ds, digests, rows, clips, bmap = world
    entries = json.loads((bmap / "blind_map.json").read_text())["entries"]
    by_blind = [e["case_id"] for e in sorted(entries, key=lambda e: e["blind_id"])]
    assert by_blind == sorted(by_blind, key=lambda c: h(c.encode()))
    truths = [c.split("-")[0] for c in by_blind]
    assert truths != sorted(truths) and truths != sorted(truths, reverse=True)
    assert "REAL" in truths[:15] and "AI" in truths[-15:]


def test_runner_cannot_reach_blind_map(world, tmp_path):
    ds, digests, rows, clips, bmap = world
    source = Path(run.__file__).read_text()
    assert "blind_map" not in source
    actions = {a.dest for sp in run.build_parser()._subparsers._group_actions
               for p in sp.choices.values() for a in p._actions}
    assert actions == {"help", "token_file", "input", "out", "resume", "render_check",
                       "expected_seal_root"}
    (clips / "blind_map.json").write_bytes((bmap / "blind_map.json").read_bytes())
    with pytest.raises(SystemExit, match="unexpected entries"):
        run.load_input(clips)
    with pytest.raises(SystemExit):
        run.load_input(bmap)


def test_runner_refuses_extra_manifest_keys(world):
    ds, digests, rows, clips, bmap = world
    exe = json.loads((clips / "execution_manifest.json").read_text())
    exe["cases"][0]["case_id"] = "AI-001"
    (clips / "execution_manifest.json").write_text(json.dumps(exe))
    with pytest.raises(SystemExit, match="keys"):
        run.load_input(clips)


# ---------------------------------------------------------------- hash gate

def test_forced_mismatch_blocks_everything(tmp_path):
    ds, digests, rows = make_dataset(tmp_path)

    def cut(c, dst):
        dst.write_bytes(clip_bytes(c["case_id"]) + (b"!" if c["case_id"] == "REAL-007" else b""))

    code, clips, bmap = extract(tmp_path, ds, rows, cut=cut)
    assert code == 2
    mm = json.loads((clips / "HASH_MISMATCH.json").read_text())
    assert mm["final_state"] == "hash_blocked" and len(mm["clips"]) == 1
    assert ex.leaks(json.dumps(mm), ex.leak_tokens([r["case_id"] for r in rows])) == []
    assert not (clips / "execution_manifest.json").exists()
    assert not (clips / f"{mm['clips'][0]['blind_id']}.mp4").exists()
    with pytest.raises(SystemExit, match="HASH_MISMATCH"):
        run.load_input(clips)


def test_runner_refuses_tampered_clip_before_any_submission(world):
    ds, digests, rows, clips, bmap = world
    (clips / "B031.mp4").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="refusing to start"):
        run.load_input(clips)


def test_mismatch_at_submission_is_hash_blocked_and_not_sent(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    _, cases = run.load_input(clips)
    (clips / "B002.mp4").write_bytes(b"swapped after the start check")
    sent = []

    class Fake:
        def upload(self, path):
            sent.append(path.name)
            return 202, {"id": path.stem}

        def get(self, aid):
            return 200, detail("INCONCLUSIVE")

    out = tmp_path / "pa"
    out.mkdir()
    monkeypatch.setattr(run, "POLL_S", 0)
    assert run.execute(Fake(), cases, clips, out, {}) == 2
    assert sent == ["B001.mp4"]
    rec = json.loads((out / "cases/B002/record.json").read_text())
    assert rec["hash_blocked"] and sc.final_state(rec) == "hash_blocked"


def test_wrong_ruleset_stops_the_run(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    code, out = phase_a(tmp_path, clips, lambda b: detail("INCONCLUSIVE", rules="r9-v4.9.9"), monkeypatch)
    assert code == 3 and len(list((out / "cases").iterdir())) == 1


# ---------------------------------------------------------------- trust anchors

def test_changed_freeze_refused_even_if_phase_a_recorded_it(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    _, out = phase_a(tmp_path, clips, lambda b: detail("INCONCLUSIVE"), monkeypatch)
    (ds / "FREEZE.json").write_text((ds / "FREEZE.json").read_text() + " ")
    env = json.loads((out / "environment.json").read_text())
    env["dataset_anchors_verified"] = {"FREEZE.json": ex.sha256_file(ds / "FREEZE.json")}
    run.write_json(out / "environment.json", env)
    root = seal(out)
    with pytest.raises(SystemExit, match="trust anchor mismatch"):
        sc.score(out, root, bmap / "blind_map.json", ds, digests)
    with pytest.raises(SystemExit, match="trust anchor mismatch"):
        ex.check_anchors(ds, digests)


@pytest.mark.parametrize("name", ["manifest.csv", "manifest.json"])
def test_changed_manifest_refused(world, name):
    ds, digests, *_ = world
    (ds / name).write_bytes((ds / name).read_bytes() + b"\n")
    with pytest.raises(SystemExit, match=name):
        ex.check_anchors(ds, digests)


def test_spec_hash_mismatch_refused(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    _, out = phase_a(tmp_path, clips, lambda b: detail("INCONCLUSIVE"), monkeypatch)
    env = json.loads((out / "environment.json").read_text())
    env["spec_sha256"]["scoring_freeze"] = "0" * 64
    run.write_json(out / "environment.json", env)
    root = seal(out)
    with pytest.raises(sc.Refusal, match="spec hashes"):
        sc.score(out, root, bmap / "blind_map.json", ds, digests)


# ---------------------------------------------------------------- seal

@pytest.fixture
def sealed(world, tmp_path, monkeypatch):
    ds, digests, rows, clips, bmap = world
    _, out = phase_a(tmp_path, clips, lambda b: detail("INCONCLUSIVE"), monkeypatch)
    return out, seal(out), (ds, digests, bmap)


def test_seal_verifies_and_excludes_itself(sealed):
    out, root, _ = sealed
    ok, d = run.verify_seal(out, root)
    assert ok
    listed = [p for p, _ in json.loads((out / run.SEAL).read_text())["files"]]
    assert run.SEAL not in listed and run.SEAL_ROOT not in listed
    assert listed == sorted(listed, key=lambda p: p.encode("utf-8"))
    assert all("\\" not in p and not p.startswith("./") for p in listed)
    (out / run.SEAL_ROOT).write_text("anything")
    assert run.verify_seal(out, root)[0]


def test_seal_canonical_form():
    doc = run.canonical([["a/b", "1" * 64]])
    assert doc == b'{"files":[["a/b","' + b"1" * 64 + b'"]],"schema":"r17-t2-seal-1"}'
    assert run.seal_root([["a/b", "1" * 64]]) == h(doc)


@pytest.mark.parametrize("tamper", ["wrong_root", "extra", "missing", "byte"])
def test_seal_refusals(sealed, tamper):
    out, root, (ds, digests, bmap) = sealed
    expected = root
    if tamper == "wrong_root":
        expected = "f" * 64
    elif tamper == "extra":
        (out / "cases/B001/extra.json").write_text("{}")
    elif tamper == "missing":
        (out / "cases/B001/record.json").unlink()
    else:
        p = out / "cases/B001/record.json"
        p.write_bytes(p.read_bytes().replace(b"INCONCLUSIVE", b"INCONCLUSIVF"))
    assert not run.verify_seal(out, expected)[0]
    with pytest.raises(sc.Refusal, match="seal refused"):
        sc.score(out, expected, bmap / "blind_map.json", ds, digests)


def test_rewritten_seal_file_does_not_help(sealed):
    out, root, _ = sealed
    p = out / "cases/B001/record.json"
    p.write_bytes(p.read_bytes().replace(b"INCONCLUSIVE", b"INCONCLUSIVF"))
    entries = run.seal_files(out)
    run.write_json(out / run.SEAL, {"schema": run.SEAL_SCHEMA, "files": entries})
    (out / run.SEAL_ROOT).write_text(run.seal_root(entries))
    assert not run.verify_seal(out, root)[0]


# ---------------------------------------------------------------- schema and smoke population

def test_result_rows_match_frozen_schema(sealed):
    out, root, (ds, digests, bmap) = sealed
    result, names = sc.score(out, root, bmap / "blind_map.json", ds, digests)
    schema = json.loads(run.SCHEMA.read_text())
    assert names == [c["name"] for c in schema["columns"]]
    assert "clip_sha_drift" not in names and all(list(r) == names for r in result["cases"])


def test_dataset_blind_map_with_truth_refused(sealed):
    out, root, (ds, digests, bmap) = sealed
    bm = json.loads((bmap / "blind_map.json").read_text())
    bm["entries"][0]["truth"] = "AI"
    (bmap / "blind_map.json").write_text(json.dumps(bm))
    with pytest.raises(sc.Refusal, match="must not carry truth"):
        sc.score(out, root, bmap / "blind_map.json", ds, digests)


def test_smoke_population_scores_and_is_labelled(tmp_path, monkeypatch):
    ds, digests, rows = make_dataset(tmp_path)
    cases = [{"case_id": f"SMOKE-{i:03d}", "source_cluster_id": f"SMOKE-CL-{i}", "truth": t,
              "clip_sha256": h(f"smoke {i}".encode())} for i, t in ((1, "REAL"), (2, "AI"))]
    tokens = ex.leak_tokens([c["case_id"] for c in cases], [c["source_cluster_id"] for c in cases])
    clips, bmap = tmp_path / "s_in", tmp_path / "s_map"
    cut = lambda c, dst: dst.write_bytes(f"smoke {c['case_id'][-1]}".encode())
    assert ex.run_extraction(cases, clips, bmap, cut, lambda p: {}, tokens, population="smoke") == 0
    _, out = phase_a(tmp_path, clips, lambda b: detail("NO_CALIBRATED_MANIPULATION_SIGNAL"), monkeypatch)
    root = seal(out)
    result, _ = sc.score(out, root, bmap / "blind_map.json", ds, digests)
    assert result["label"] == sc.SMOKE_LABEL and result["cluster_level"]["K"] == 2
    assert result["case_level"]["states"]["negative"] == 2
    bm = json.loads((bmap / "blind_map.json").read_text())
    bm["entries"][0]["case_id"] = "AI-001"
    (bmap / "blind_map.json").write_text(json.dumps(bm))
    with pytest.raises(sc.Refusal, match="R17 case id"):
        sc.score(out, root, bmap / "blind_map.json", ds, digests)
