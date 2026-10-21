"""R18-T3 freeze & split: atomic units, known-subject isolation, generator invariant, dHash registry, determinism, tamper detection, LOCKED_TEST guard (synthetic inputs only)."""

import ast
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from eval import r18_t3_freeze_split as fs

MODULE_PATH = Path(fs.__file__)
TS = "2026-10-21T16:00:00Z"
DOMAINS = ("webcam_talking_head_scenario", "ugc_vlog", "spacecraft_onboard_camera")


@pytest.fixture(autouse=True)
def _fast_solver(monkeypatch):
    monkeypatch.setattr(fs, "RESTARTS", 4)


def sha(tag):
    return hashlib.sha256(str(tag).encode()).hexdigest()


def record(tag, truth, cluster, gen=None, domain=None, subject=None, face=False,
           collection=None):
    return {
        "sha256": sha(tag), "canonical_path": f"x/{tag}.mp4", "inspectable": True,
        "origin": "test",
        "ground_truth_provenance": {
            "ground_truth": truth, "generator_family": gen,
            "generator_model": f"{gen} v1" if gen else None, "capture_domain": domain,
            "source_cluster_id": cluster, "subject_id": subject,
            "source_collection_id": collection or ("pub/ai" if gen else "pub/real")},
        "derived_annotations": {
            "face_present": {"status": "DETERMINED", "value": face},
            "speech_talking_head": {"status": "NOT_APPLICABLE"}},
    }


def dataset(extra=()):
    """~90 clips: 6 single-clip clusters per generator, REAL clusters incl. a 10-clip monolith."""
    recs = []
    for g in fs.GENERATORS:
        for k in range(6):
            recs.append(record(f"{g}-{k}", fs.VERIFIED_AI, f"ai-{g}-{k}", gen=g, face=k % 2 == 0))
    for k in range(10):
        recs.append(record(f"mono-{k}", fs.VERIFIED_REAL, "CL-MONO", domain=DOMAINS[2]))
    for k in range(32):
        recs.append(record(f"r-{k}", fs.VERIFIED_REAL, f"real-{k}", domain=DOMAINS[k % 2],
                           subject=f"subj-{k}" if k % 2 == 0 else None, face=k % 2 == 0))
    recs.extend(extra)
    return recs


def write_anchor(tmp_path, recs, name="anchors.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"schema": fs.SCHEMA_ANCHOR, "records": recs}))
    return p


def cases_of(recs):
    return sorted((fs.case_view(r) for r in recs), key=lambda c: c["sha256"])


def far_hashes(cases, seed=7):
    rng = random.Random(seed)
    return {c["sha256"]: {"duration_s": 4.0,
                          "frames": [f"{rng.getrandbits(64):016x}" for _ in range(8)]}
            for c in cases}


def write_registry(tmp_path, recs, frame_hashes, states=None, seed=fs.DEFAULT_SEED):
    """A registry as audit-leakage would write it, with optional decisions by pair_id."""
    cases = cases_of(recs)
    prev = {"candidates": [{"pair_id": pid, "adjudication_state": st,
                            "adjudication_notes": "reviewed"}
                           for pid, st in (states or {}).items()]}
    reg, _ = fs.build_registry(cases, frame_hashes, prev,
                               fs.solve(cases, fs.build_units(cases), seed), seed)
    if any(st == fs.SHARED_SOURCE for st in (states or {}).values()):
        split_of = fs.solve(cases, fs.build_units(cases, fs.shared_pairs_of(reg)), seed)
        reg, _ = fs.build_registry(cases, frame_hashes, reg, split_of, seed)
    p = tmp_path / "adj.json"
    p.write_text(json.dumps(reg))
    return p, reg


# --- subject normalisation and partition units ------------------------------------------------

@pytest.mark.parametrize("raw,expected", [(" Alice ", "alice"), ("ALICE", "alice"),
                                          ("ytugc_creator_ab", "ytugc_creator_ab")])
def test_known_subjects_are_normalised(raw, expected):
    assert fs.normalise_subject(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "  ", "UNKNOWN", "unknown", "null", "None", "N/A"])
def test_null_like_subjects_are_no_subject(raw):
    assert fs.normalise_subject(raw) is None


def test_same_cluster_is_one_unit():
    cases = cases_of([record("a", fs.VERIFIED_REAL, "c1"), record("b", fs.VERIFIED_REAL, "c1")])
    assert len(fs.build_units(cases)) == 1


def test_known_subject_couples_clusters_after_normalisation():
    cases = cases_of([record("a", fs.VERIFIED_REAL, "c1", subject=" Bob "),
                      record("b", fs.VERIFIED_REAL, "c2", subject="bob")])
    units = fs.build_units(cases)
    assert [u["unit_id"] for u in units] == ["c1+c2"]


def test_null_and_unknown_subjects_do_not_form_a_group():
    cases = cases_of([record("a", fs.VERIFIED_REAL, "c1", subject=None),
                      record("b", fs.VERIFIED_REAL, "c2", subject=None),
                      record("c", fs.VERIFIED_REAL, "c3", subject="UNKNOWN"),
                      record("d", fs.VERIFIED_REAL, "c4", subject="")])
    assert len(fs.build_units(cases)) == 4


def test_duplicate_sha_couples_clusters():
    a = record("same", fs.VERIFIED_REAL, "c1")
    b = record("same", fs.VERIFIED_REAL, "c2")
    assert len(fs.build_units(cases_of([a, b]))) == 1


def test_shared_source_pair_merges_units():
    cases = cases_of([record("a", fs.VERIFIED_REAL, "c1"), record("b", fs.VERIFIED_REAL, "c2")])
    units = fs.build_units(cases, [(sha("a"), sha("b"))])
    assert [u["unit_id"] for u in units] == ["c1+c2"]


def test_shared_source_pair_outside_the_anchor_refuses():
    cases = cases_of([record("a", fs.VERIFIED_REAL, "c1")])
    with pytest.raises(fs.Refusal):
        fs.build_units(cases, [(sha("a"), sha("ghost"))])


def test_non_verified_ground_truth_refuses():
    with pytest.raises(fs.Refusal):
        fs.case_view(record("a", "AMBIGUOUS", "c1"))


# --- the partitioner --------------------------------------------------------------------------

def solved(recs, seed=fs.DEFAULT_SEED):
    cases = cases_of(recs)
    return cases, fs.solve(cases, fs.build_units(cases), seed)


def test_zero_cluster_overlap_and_monolith_kept_whole():
    cases, split_of = solved(dataset())
    audit = fs.identity_audit(cases, split_of)
    assert audit["cluster_overlap_count"] == 0
    assert len({split_of[c["sha256"]] for c in cases if c["source_cluster_id"] == "CL-MONO"}) == 1


def test_known_subject_spanning_clusters_never_crosses_splits():
    extra = [record(f"s-{k}", fs.VERIFIED_REAL, f"subj-cl-{k}", domain=DOMAINS[0],
                    subject="Shared Person" if k % 2 else "shared person ") for k in range(6)]
    cases, split_of = solved(dataset(extra))
    assert fs.identity_audit(cases, split_of)["subject_overlap_count"] == 0
    assert len({split_of[sha(f"s-{k}")] for k in range(6)}) == 1


def test_null_subjects_are_free_to_spread():
    cases, split_of = solved(dataset())
    nulls = {split_of[c["sha256"]] for c in cases
             if c["ground_truth"] == fs.VERIFIED_REAL and c["subject_id"] is None
             and c["source_cluster_id"] != "CL-MONO"}
    assert nulls == set(fs.SPLITS)


def test_zero_sha_overlap():
    dup = [record("r-0", fs.VERIFIED_REAL, "dup-cluster", domain=DOMAINS[0])]
    cases, split_of = solved(dataset(dup))
    assert fs.identity_audit(cases, split_of)["hash_overlap_count"] == 0


def test_every_generator_in_every_split():
    cases, split_of = solved(dataset())
    assert fs.identity_audit(cases, split_of)["missing_generator_families"] == []


def test_generator_that_cannot_cover_three_splits_is_a_constraint_conflict():
    recs = [r for r in dataset() if r["ground_truth_provenance"]["generator_family"] != "Sora"]
    recs += [record(f"sora-{k}", fs.VERIFIED_AI, "sora-one", gen="Sora") for k in range(3)]
    with pytest.raises(fs.Refusal, match="constraint conflict"):
        solved(recs)


def test_case_shares_approach_the_targets():
    cases, split_of = solved(dataset())
    summary = fs.split_summary(cases, split_of)
    for s in fs.SPLITS:
        assert abs(summary[s]["case_share"] - fs.FRACTIONS[s]) < 0.06


def test_split_is_deterministic_for_a_seed():
    assert solved(dataset())[1] == solved(dataset())[1]


def test_no_cluster_is_pinned_and_no_clock_is_read():
    text = MODULE_PATH.read_text()
    for banned in ("CL-NASA", "r16-owner", "SAW1", "ASCENT", ".now(", "utcnow", "time.time",
                   "date.today", "import time"):
        assert banned not in text, banned


# --- timestamp --------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["2026-10-21T16:00:00", "2026-10-21T16:00:00+00:00",
                                 "2026-10-21 16:00:00Z", "2026-13-01T00:00:00Z", None, ""])
def test_freeze_timestamp_must_be_explicit_utc_iso(bad):
    with pytest.raises(fs.Refusal):
        fs.validate_timestamp(bad)


def test_freeze_requires_the_timestamp_argument(tmp_path):
    with pytest.raises(SystemExit):
        fs.main(["freeze", "--anchors", "a", "--adjudications", "b", "--freeze-out", "c",
                 "--manifest-out", "d"])


# --- dHash ------------------------------------------------------------------------------------

def test_dhash_of_gradient_and_flat_frames():
    ramp = np.tile(np.arange(90, dtype=np.uint8) * 2, (80, 1))
    assert fs.dhash_frame(ramp) == "ffffffffffffffff"
    assert fs.dhash_frame(ramp[:, ::-1].copy()) == "0000000000000000"
    assert fs.dhash_frame(np.full((80, 90), 7, np.uint8)) == "0000000000000000"


def test_frame_times_are_evenly_spaced_inside_the_clip():
    assert fs.frame_times(16.0) == [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]


def test_pair_distance_is_the_closest_frame_pair():
    assert fs.hamming("ff", "0f") == 4
    assert fs.min_distance(["ff", "00"], ["0f", None, "01"]) == 1


def test_pair_id_is_order_independent():
    assert fs.pair_id(sha(1), sha(2)) == fs.pair_id(sha(2), sha(1))


def _hashes_with_distance(cases, a, b, bits):
    h = far_hashes(cases)
    base = int(h[sha(a)]["frames"][3], 16)
    h[sha(b)]["frames"][5] = f"{base ^ ((1 << bits) - 1):016x}"
    return h


@pytest.mark.parametrize("bits,flagged", [(10, True), (11, False)])
def test_candidate_threshold_is_inclusive_ten_bits(bits, flagged):
    cases = cases_of(dataset())
    found = fs.find_candidates(cases, _hashes_with_distance(cases, "r-1", "r-3", bits))
    pairs = {tuple(sorted(p[:2])) for p in found}
    assert (tuple(sorted((sha("r-1"), sha("r-3")))) in pairs) is flagged


def test_same_cluster_pairs_are_not_candidates():
    cases = cases_of(dataset())
    found = fs.find_candidates(cases, _hashes_with_distance(cases, "mono-0", "mono-1", 0))
    assert found == []


# --- adjudication registry and freeze gate ----------------------------------------------------

def pick_pair(recs, cross):
    """Two single-clip REAL clips that the solved split puts in different (or the same) splits."""
    _, split_of = solved(recs)
    tags = [f"r-{k}" for k in range(32)]
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            if (split_of[sha(a)] != split_of[sha(b)]) == cross:
                return a, b
    raise AssertionError("no such pair")


def test_cross_split_candidates_start_unresolved_and_decisions_survive(tmp_path):
    recs = dataset()
    a, b = pick_pair(recs, cross=True)
    h = _hashes_with_distance(cases_of(recs), a, b, 2)
    _, reg = write_registry(tmp_path, recs, h)
    (entry,) = reg["candidates"]
    assert entry["scope"] == fs.CROSS_SPLIT and entry["adjudication_state"] == fs.UNRESOLVED
    _, reg2 = write_registry(tmp_path, recs, h, {entry["pair_id"]: fs.INDEPENDENT})
    assert reg2["candidates"][0]["adjudication_state"] == fs.INDEPENDENT


def test_cross_split_unresolved_candidate_halts_the_freeze(tmp_path):
    recs = dataset()
    a, b = pick_pair(recs, cross=True)
    adj, _ = write_registry(tmp_path, recs, _hashes_with_distance(cases_of(recs), a, b, 2))
    with pytest.raises(fs.Refusal, match="UNRESOLVED"):
        fs.build_freeze(write_anchor(tmp_path, recs), adj, fs.DEFAULT_SEED, TS)


def test_same_split_candidate_is_diagnostic_and_does_not_block(tmp_path):
    recs = dataset()
    a, b = pick_pair(recs, cross=False)
    adj, reg = write_registry(tmp_path, recs, _hashes_with_distance(cases_of(recs), a, b, 2))
    (entry,) = reg["candidates"]
    assert entry["scope"] == fs.SAME_SPLIT
    assert entry["adjudication_state"] == fs.NOT_REQUIRED_SAME_SPLIT
    freeze, _ = fs.build_freeze(write_anchor(tmp_path, recs), adj, fs.DEFAULT_SEED, TS)
    lv = freeze["leakage_verification"]
    assert lv["verification_status"] == "CLEAN" and lv["same_split_candidate_pair_count"] == 1
    assert lv["cross_split_unresolved_count"] == 0


def test_not_required_pair_moved_across_splits_must_be_adjudicated():
    cases = cases_of(dataset())
    a, b = sorted((sha("r-1"), sha("r-3")))
    h = _hashes_with_distance(cases, "r-1", "r-3", 2)
    prev = {"candidates": [{"pair_id": fs.pair_id(a, b),
                            "adjudication_state": fs.NOT_REQUIRED_SAME_SPLIT}]}
    split_of = {c["sha256"]: fs.DEV for c in cases}
    split_of[b] = fs.VAL
    reg, _ = fs.build_registry(cases, h, prev, split_of, fs.DEFAULT_SEED)
    assert reg["candidates"][0]["adjudication_state"] == fs.UNRESOLVED


def test_independent_cross_split_candidate_allows_clean(tmp_path):
    recs = dataset()
    a, b = pick_pair(recs, cross=True)
    h = _hashes_with_distance(cases_of(recs), a, b, 2)
    adj, _ = write_registry(tmp_path, recs, h, {fs.pair_id(sha(a), sha(b)): fs.INDEPENDENT})
    freeze, _ = fs.build_freeze(write_anchor(tmp_path, recs), adj, fs.DEFAULT_SEED, TS)
    lv = freeze["leakage_verification"]
    assert lv["verification_status"] == "CLEAN" and lv["cross_split_candidate_pair_count"] == 1
    assert lv["unresolved_perceptual_candidate_count"] == 0


def test_shared_source_merges_clusters_and_re_splits(tmp_path):
    recs = dataset()
    a, b = pick_pair(recs, cross=True)
    h = _hashes_with_distance(cases_of(recs), a, b, 2)
    adj, reg = write_registry(tmp_path, recs, h, {fs.pair_id(sha(a), sha(b)): fs.SHARED_SOURCE})
    assert reg["candidates"][0]["scope"] == fs.SAME_SPLIT
    freeze, _ = fs.build_freeze(write_anchor(tmp_path, recs), adj, fs.DEFAULT_SEED, TS)
    split = {r["sha256"]: r["split"] for r in freeze["records"]}
    assert split[sha(a)] == split[sha(b)]
    merged = "+".join(sorted((f"real-{a[2:]}", f"real-{b[2:]}")))
    assert freeze["partition_units"]["merged_by_shared_source"] == [merged]
    lv = freeze["leakage_verification"]
    assert lv["confirmed_cross_split_shared_source_count"] == 0
    assert lv["confirmed_shared_source_count"] == 1


def entry_at(distance, scope, state):
    return {"min_hamming_distance": distance, "scope": scope, "adjudication_state": state}


def test_summary_counts_by_scope_and_distance_bucket():
    summary = fs.perceptual_summary([
        entry_at(1, fs.CROSS_SPLIT, fs.SHARED_SOURCE), entry_at(4, fs.CROSS_SPLIT, fs.UNRESOLVED),
        entry_at(10, fs.CROSS_SPLIT, fs.INDEPENDENT),
        entry_at(9, fs.SAME_SPLIT, fs.NOT_REQUIRED_SAME_SPLIT)])
    assert summary["confirmed_cross_split_shared_source_count"] == 1
    assert summary["cross_split_unresolved_count"] == 1
    assert summary["same_split_candidate_pair_count"] == 1
    b = summary["distance_buckets"]
    assert b["0-2"]["confirmed_shared_source"] == 1 and b["3-5"]["cross_split_unresolved"] == 1
    assert b["9-10"] == {"candidates": 2, "cross_split": 1, "same_split": 1,
                         "cross_split_independent": 1, "confirmed_shared_source": 0,
                         "cross_split_unresolved": 0}


@pytest.mark.parametrize("scope,state,notes", [
    (fs.CROSS_SPLIT, fs.INDEPENDENT, ""), (fs.CROSS_SPLIT, "MAYBE", "x"),
    (fs.CROSS_SPLIT, fs.NOT_REQUIRED_SAME_SPLIT, ""), (fs.SAME_SPLIT, fs.UNRESOLVED, ""),
    ("ANY", fs.INDEPENDENT, "x")])
def test_registry_rejects_bad_state_scope_or_missing_rationale(tmp_path, scope, state, notes):
    p = tmp_path / "adj.json"
    p.write_text(json.dumps({"schema": fs.SCHEMA_ADJUDICATIONS, "candidates": [
        {"pair_id": "p", "scope": scope, "adjudication_state": state,
         "adjudication_notes": notes}]}))
    with pytest.raises(fs.Refusal):
        fs.load_registry(p)


def test_freeze_refuses_stale_registry_candidates(tmp_path):
    recs = dataset()
    cases = cases_of(recs)
    adj, reg = write_registry(tmp_path, recs, far_hashes(cases))
    reg["frame_hashes"][sha("r-3")]["frames"][0] = reg["frame_hashes"][sha("r-1")]["frames"][0]
    adj.write_text(json.dumps(reg))
    with pytest.raises(fs.Refusal, match="stale"):
        fs.build_freeze(write_anchor(tmp_path, recs), adj, fs.DEFAULT_SEED, TS)


def test_freeze_refuses_registry_split_for_another_seed(tmp_path):
    recs = dataset()
    adj, _ = write_registry(tmp_path, recs, far_hashes(cases_of(recs)), seed=1)
    with pytest.raises(fs.Refusal, match="stale"):
        fs.build_freeze(write_anchor(tmp_path, recs), adj, fs.DEFAULT_SEED, TS)


def test_freeze_refuses_frame_hashes_not_covering_the_anchor(tmp_path):
    recs = dataset()
    adj, _ = write_registry(tmp_path, recs, far_hashes(cases_of(recs)))
    more = recs + [record("late", fs.VERIFIED_REAL, "late-cl", domain=DOMAINS[0])]
    with pytest.raises(fs.Refusal, match="cover"):
        fs.build_freeze(write_anchor(tmp_path, more), adj, fs.DEFAULT_SEED, TS)


# --- outputs: determinism, tamper detection, LOCKED_TEST guard --------------------------------

def run_freeze(tmp_path, tag):
    recs = dataset()
    anchors = write_anchor(tmp_path, recs)
    adj, _ = write_registry(tmp_path, recs, far_hashes(cases_of(recs)))
    out, man = tmp_path / f"freeze_{tag}.json", tmp_path / f"manifest_{tag}.csv"
    fs.main(["freeze", "--anchors", str(anchors), "--adjudications", str(adj),
             "--freeze-timestamp", TS, "--freeze-out", str(out), "--manifest-out", str(man)])
    return out, man


def test_freeze_is_byte_identical_on_rerun(tmp_path):
    a = run_freeze(tmp_path, "a")
    b = run_freeze(tmp_path, "b")
    assert a[0].read_bytes() == b[0].read_bytes() and a[1].read_bytes() == b[1].read_bytes()
    freeze = json.loads(a[0].read_text())
    assert freeze["freeze_timestamp"] == TS and freeze["total_records"] == len(dataset())
    assert fs.verify_freeze(freeze, a[1].read_text()) == []


def test_manifest_csv_has_the_plan_columns(tmp_path):
    _, man = run_freeze(tmp_path, "a")
    lines = man.read_text().splitlines()
    assert lines[0] == ",".join(fs.MANIFEST_COLUMNS) and len(lines) == len(dataset()) + 1


def test_tampering_with_a_split_breaks_the_manifest_digest(tmp_path):
    out, man = run_freeze(tmp_path, "a")
    freeze = json.loads(out.read_text())
    r = freeze["records"][0]
    r["split"] = fs.DEV if r["split"] != fs.DEV else fs.VAL
    assert "manifest_sha256 does not match the records" in fs.verify_freeze(freeze)


def test_tampering_with_a_label_or_the_csv_is_detected(tmp_path):
    out, man = run_freeze(tmp_path, "a")
    freeze = json.loads(out.read_text())
    freeze["records"][1]["ground_truth"] = fs.VERIFIED_REAL
    assert fs.verify_freeze(freeze)
    intact = json.loads(out.read_text())
    assert fs.verify_freeze(intact, man.read_text().replace("VERIFIED_AI", "VERIFIED_REAL", 1))


def test_locked_test_is_refused_outside_t6(tmp_path):
    out, _ = run_freeze(tmp_path, "a")
    with pytest.raises(PermissionError):
        fs.load_partition(out, fs.LOCKED_TEST)
    with pytest.raises(PermissionError):
        fs.load_partition(out, fs.LOCKED_TEST, purpose="R18-T5")
    assert all(r["split"] == fs.DEV for r in fs.load_partition(out, fs.DEV))
    assert fs.load_partition(out, fs.LOCKED_TEST, purpose=fs.LOCKED_TEST_PURPOSE)


def test_governance_notice_carries_the_mandatory_statement():
    assert ("LOCKED_TEST is source-cluster- and subject-isolated, but not collection- or "
            "publisher-independent.") in fs.GOVERNANCE_NOTICE


# --- media integrity and isolation ------------------------------------------------------------

def test_audit_refuses_media_whose_digest_differs(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "x").mkdir(parents=True)
    rec = record("a", fs.VERIFIED_REAL, "c1")
    (corpus / "x/a.mp4").write_bytes(b"not the anchored bytes")
    with pytest.raises(fs.Refusal, match="sha256"):
        fs.main(["audit-leakage", "--anchors", str(write_anchor(tmp_path, [rec])),
                 "--adjudications", str(tmp_path / "adj.json"), "--corpus", str(corpus)])
    assert not (tmp_path / "adj.json").exists()


def test_no_production_detector_api_network_or_subprocess_is_used():
    text = MODULE_PATH.read_text().lower()
    for banned in ("/api/v1/analyses", "lipforensics", "efficientnet", "facetorch", "nvcf",
                   "urllib", "requests", "localhost", "from apps", "import apps", "docker",
                   "subprocess", "unlink", "rmtree", "os.remove"):
        assert banned not in text, banned
    imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(MODULE_PATH.read_text()))
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "subprocess" not in imported
