"""R12-T6: what the operational export includes, what it refuses, and that replay reads it.

Everything here is pure: fixture rows stand in for the database, and are test data. The same
pipeline is run against real PostgreSQL rows in
`apps/api/tests/test_export_operational_dataset.py`.
"""

import ast
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from eval import export_operational_dataset as ex
from eval import replay as replay_module
from eval.corpus import CorpusItem, leakage_findings, split_by_lineage

API_APP = Path(__file__).resolve().parents[3] / "apps" / "api" / "app"
engine = replay_module.load_risk_engine()

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

METADATA = {
    "license": "test fixture licence",
    "permission_status": "test fixture permission",
    "redistributable": False,
    "private": False,
    "stratum_primary": "test_stratum",
    "source": "test fixture source",
    "acquisition_type": "test_acquisition",
    "benchmark_family": "test_family",
}


def truth(sha=SHA_A, source_class="CONTROLLED_TEST", label="GENUINE"):
    return ex.GroundTruthRow(sha, source_class, label)


def governance(sha=SHA_A, *, lineage="lineage-1", split="TEST", parent=None, steps=None, **kw):
    return ex.GovernanceRow(
        media_sha256=sha,
        source_lineage_id=lineage,
        dataset_split=split,
        derived_from_sha256=parent,
        transformations=steps,
        **{**METADATA, **kw},
    )


def media(sha=SHA_A, *, media_id="m-1", created="2026-09-24T10:00:00+00:00", key=None):
    return ex.MediaRow(
        media_file_id=media_id,
        analysis_id=f"analysis-{media_id}",
        analysis_created_at=created,
        media_sha256=sha,
        size_bytes=4096,
        storage_key=key if key is not None else f"originals/{sha}",
    )


def signal(
    sha=SHA_A,
    *,
    signal_id="s-1",
    provider="nvidia",
    signal_type="synthetic_video",
    version=engine.SVD_PROVIDER_VERSION,
    status="SUCCESS",
    score=0.5,
    metadata=None,
    created="2026-09-24T10:00:00+00:00",
):
    if metadata is None:
        key = ex.REPLAY_SLOTS.get((provider, signal_type))
        metadata = {key.metadata_count_key: 3} if key and status == "SUCCESS" else {}
    return ex.SignalRow(
        signal_id=signal_id,
        analysis_id=f"analysis-{signal_id}",
        created_at=created,
        media_sha256=sha,
        provider=provider,
        signal_type=signal_type,
        provider_version=version,
        status=status,
        score=score,
        metadata=metadata,
    )


def corpus_of(truths, gov, med, split=None):
    return ex.build_corpus(truths, gov, med, split)


# --- Ground Truth eligibility and label mapping ----------------------------------------------


def test_an_unknown_source_class_is_ineligible_whatever_the_label():
    result = corpus_of([truth(source_class="UNKNOWN", label="UNKNOWN")], [governance()], [media()])
    assert result["items"] == []
    assert result["exclusion_counts"]["ground_truth_ineligible"] == 1
    assert result["exclusion_counts"]["unknown_label"] == 0


def test_an_unknown_label_is_excluded():
    result = corpus_of([truth(label="UNKNOWN")], [governance()], [media()])
    assert result["items"] == []
    assert result["exclusion_counts"]["unknown_label"] == 1


@pytest.mark.parametrize("label", ["AUDIO_MANIPULATION", "OTHER_MANIPULATION"])
def test_a_label_without_a_benchmark_counterpart_is_excluded(label):
    result = corpus_of([truth(label=label)], [governance()], [media()])
    assert result["items"] == []
    assert result["exclusion_counts"]["unsupported_label"] == 1


@pytest.mark.parametrize(
    ("label", "expected"),
    [("GENUINE", "real"), ("AI_GENERATED", "synthetic"), ("FACE_SWAP", "face_swap")],
)
def test_labels_map_to_the_benchmark_vocabulary(label, expected):
    result = corpus_of([truth(label=label)], [governance()], [media()])
    assert [item.label for item in result["items"]] == [expected]


def test_a_value_outside_the_ground_truth_vocabulary_raises():
    with pytest.raises(ex.ExportError, match="label"):
        corpus_of([truth(label="DEEPFAKE")], [governance()], [media()])


# --- corpus metadata -------------------------------------------------------------------------


@pytest.mark.parametrize("field", ex.CORPUS_METADATA_FIELDS)
def test_a_null_corpus_field_excludes_the_record(field):
    result = corpus_of([truth()], [governance(**{field: None})], [media()])
    assert result["items"] == []
    assert result["exclusion_counts"]["insufficient_corpus_metadata"] == 1


def test_no_governance_record_is_insufficient_metadata():
    result = corpus_of([truth()], [], [media()])
    assert result["exclusion_counts"]["insufficient_corpus_metadata"] == 1


def test_false_booleans_are_statements_and_are_exported():
    result = corpus_of(
        [truth()], [governance(redistributable=False, private=False)], [media()]
    )
    [item] = result["items"]
    assert item.redistributable is False
    assert item.private is False


def test_fields_map_across_unchanged_and_family_is_the_benchmark_family():
    result = corpus_of([truth(label="FACE_SWAP", source_class="OWNER_KNOWN")], [governance()], [media()])
    [item] = result["items"]
    record = asdict(item)
    assert "benchmark_family" not in record
    assert item.family == METADATA["benchmark_family"]
    assert item.label_provenance == "OWNER_KNOWN"
    for name in ("license", "permission_status", "stratum_primary", "source", "acquisition_type"):
        assert record[name] == METADATA[name]
    assert item.clip_id == item.sha256 == SHA_A
    assert item.split == "TEST"
    assert item.path == f"originals/{SHA_A}"
    assert item.bytes == 4096
    assert item.derivation == "none"
    assert item.strata == []


def test_a_derivative_without_recorded_steps_is_insufficient_metadata():
    result = corpus_of(
        [truth(SHA_A), truth(SHA_B)],
        [governance(SHA_A), governance(SHA_B, parent=SHA_A)],
        [media(SHA_A), media(SHA_B, media_id="m-2")],
    )
    assert [item.clip_id for item in result["items"]] == [SHA_A]
    assert result["exclusion_counts"]["insufficient_corpus_metadata"] == 1


def test_a_derivative_carries_its_steps_its_parent_and_its_root():
    result = corpus_of(
        [truth(SHA_A), truth(SHA_B), truth(SHA_C)],
        [
            governance(SHA_A),
            governance(SHA_B, parent=SHA_A, steps=("reencode",)),
            governance(SHA_C, parent=SHA_B, steps=("reencode", "resize")),
        ],
        [media(SHA_A), media(SHA_B, media_id="m-2"), media(SHA_C, media_id="m-3")],
    )
    by_id = {item.clip_id: item for item in result["items"]}
    assert by_id[SHA_C].derivative_of == SHA_B
    assert by_id[SHA_C].base_media_id == SHA_A
    assert by_id[SHA_C].derivation == "reencode+resize"
    assert result["leakage"]["leaked"] is False


# --- media representative and path --------------------------------------------------------


def test_one_item_per_hash_from_the_earliest_upload_with_the_id_breaking_ties():
    rows = [
        media(media_id="m-3", created="2026-09-24T11:00:00+00:00", key=f"originals/{SHA_A}"),
        media(media_id="m-2", created="2026-09-24T10:00:00+00:00"),
        media(media_id="m-1", created="2026-09-24T10:00:00+00:00"),
    ]
    chosen, duplicates = ex.representative_media(rows)
    assert chosen[SHA_A].media_file_id == "m-1"
    assert duplicates == 2

    result = corpus_of([truth()], [governance()], rows)
    assert len(result["items"]) == 1
    assert result["media_file_duplicates"] == 2


def test_no_media_row_is_excluded():
    result = corpus_of([truth()], [governance()], [])
    assert result["exclusion_counts"]["no_media_file"] == 1


def test_a_storage_key_that_does_not_name_the_bytes_is_excluded_not_converted():
    result = corpus_of([truth()], [governance()], [media(key="originals/r5t4-seed-0")])
    assert result["items"] == []
    assert result["exclusion_counts"]["unusable_storage_path"] == 1


# --- splits and leakage -------------------------------------------------------------------


def test_the_split_filter_keeps_one_split_and_counts_the_rest_apart():
    result = corpus_of(
        [truth(SHA_A), truth(SHA_B)],
        [governance(SHA_A, split="TEST"), governance(SHA_B, lineage="lineage-2", split="HOLDOUT")],
        [media(SHA_A), media(SHA_B, media_id="m-2")],
        split="HOLDOUT",
    )
    assert [item.clip_id for item in result["items"]] == [SHA_B]
    assert result["outside_split_filter"] == [SHA_A]
    assert sum(result["exclusion_counts"].values()) == 0


def test_an_unknown_split_filter_is_refused():
    with pytest.raises(ex.ExportError):
        corpus_of([truth()], [governance()], [media()], split="evaluation")


def test_a_derivative_whose_parent_was_not_exported_is_reported_not_failed():
    result = corpus_of(
        [truth(SHA_A, label="UNKNOWN"), truth(SHA_B)],
        [governance(SHA_A), governance(SHA_B, parent=SHA_A, steps=("reencode",))],
        [media(SHA_A), media(SHA_B, media_id="m-2")],
    )
    assert [item.clip_id for item in result["items"]] == [SHA_B]
    assert result["leakage"]["missing_exported_parent"] == [SHA_B]
    assert result["leakage"]["derivatives_without_base"] == []
    assert result["leakage"]["leaked"] is False


def _item(clip_id, lineage, split, derivative_of=None):
    return CorpusItem(
        clip_id=clip_id, path=clip_id, label="real", family="f", stratum_primary="s",
        source_lineage_id=lineage, base_media_id=derivative_of or clip_id,
        derivative_of=derivative_of, derivation="none", source="s", acquisition_type="a",
        label_provenance="p", license="l", permission_status="p", redistributable=False,
        private=False, sha256=clip_id, bytes=1, split=split,
    )


@pytest.mark.parametrize(("first", "second"), [("CALIBRATION", "HOLDOUT"), ("VALIDATION", "TEST")])
def test_a_lineage_under_two_operational_splits_leaks(first, second):
    findings = leakage_findings([_item(SHA_A, "l", first), _item(SHA_B, "l", second)])
    assert findings["shared_lineage_ids"] == ["l"]
    assert findings["leaked"] is True


def test_an_absent_parent_nobody_accounted_for_still_leaks():
    findings = leakage_findings([_item(SHA_B, "l", "TEST", derivative_of=SHA_A)])
    assert findings["derivatives_without_base"] == [SHA_B]
    assert findings["missing_exported_parent"] == []
    assert findings["leaked"] is True


def test_split_by_lineage_accepts_the_operational_splits():
    [item] = split_by_lineage([_item(SHA_A, "l", None)], {"l": "HOLDOUT"})
    assert item.split == "HOLDOUT"


def test_a_leaking_export_writes_nothing(tmp_path):
    corpus = corpus_of([truth()], [governance()], [media()])
    corpus["leakage"] = {**corpus["leakage"], "leaked": True, "shared_lineage_ids": ["x"]}
    out = tmp_path / "export"
    with pytest.raises(ex.ExportError, match="leakage"):
        ex.write_export(out, corpus, ex.build_results([], set()), split_filter=None,
                        exporter_identity={}, source_db={})
    assert not out.exists()


# --- signals ---------------------------------------------------------------------------------


def test_repeated_runs_collapse_to_the_latest_with_the_id_breaking_ties():
    rows = [
        signal(signal_id="s-1", score=0.1, created="2026-09-24T10:00:00+00:00"),
        signal(signal_id="s-3", score=0.3, created="2026-09-24T12:00:00+00:00"),
        signal(signal_id="s-2", score=0.2, created="2026-09-24T12:00:00+00:00"),
    ]
    results = ex.build_results(rows, {SHA_A})
    [payload] = results["runs"].values()
    assert [clip["signal_id"] for clip in payload["clips"]] == ["s-3"]
    assert results["signal_duplicates"] == 2


def test_each_version_and_signal_type_gets_its_own_file_with_its_identity_intact():
    rows = [
        signal(SHA_A, signal_id="s-1"),
        signal(SHA_B, signal_id="s-2", version="some-other-function-id"),
        signal(SHA_A, signal_id="s-3", provider="efficientnet-b7",
               signal_type="face_manipulation", version=engine.FACE_PROVIDER_VERSION),
    ]
    results = ex.build_results(rows, {SHA_A, SHA_B})
    names = {ex.results_filename(identity) for identity in results["runs"]}
    assert len(results["runs"]) == 3
    assert len(names) == 3
    for identity, payload in results["runs"].items():
        assert (payload["provider"], payload["provider_version"], payload["signal_type"]) == identity
        assert {c["clip_id"] for c in payload["clips"]} <= {SHA_A, SHA_B}
    svd_versions = {
        payload["provider_version"]
        for payload in results["runs"].values()
        if payload["provider"] == "nvidia"
    }
    assert svd_versions == {engine.SVD_PROVIDER_VERSION, "some-other-function-id"}


def test_filenames_are_path_safe_and_never_collide_after_sanitising():
    risky = ("nvi dia", "../../etc/passwd", "a/b")
    near = ("nvi dia", "../../etc/passwd", "a_b")
    name = ex.results_filename(risky)
    assert "/" not in name and ".." not in name and " " not in name
    assert name.startswith("results_") and name.endswith(".json")
    assert ex.results_filename(near) != name
    assert ex.results_filename(risky) == name


@pytest.mark.parametrize(
    ("provider", "signal_type"),
    [("effort", "face_forgery"), ("aasist", "audio_authenticity"), ("lipforensics", "lip_sync"),
     ("nvidia", "active_speaker"), ("c2pa", "provenance")],
)
def test_a_signal_replay_does_not_read_is_unsupported(provider, signal_type):
    results = ex.build_results(
        [signal(provider=provider, signal_type=signal_type, metadata={})], {SHA_A}
    )
    assert results["runs"] == {}
    assert results["exclusion_counts"]["unsupported_signal_schema"] == 1


def test_a_status_other_than_success_or_failed_is_unsupported():
    results = ex.build_results([signal(status="TIMEOUT", score=None)], {SHA_A})
    assert results["runs"] == {}
    assert results["exclusion_counts"]["unsupported_signal_schema"] == 1


@pytest.mark.parametrize(
    "broken",
    [{"score": None}, {"metadata": {}}, {"metadata": {"total_clips": True}},
     {"metadata": {"total_clips": "3"}}],
)
def test_a_success_without_its_score_or_count_is_not_exported(broken):
    results = ex.build_results([signal(**broken)], {SHA_A})
    assert results["runs"] == {}
    assert results["exclusion_counts"]["invalid_success_payload"] == 1


def test_a_failed_signal_is_exported_as_an_error_under_its_stored_version():
    results = ex.build_results([signal(status="FAILED", score=None)], {SHA_A})
    [(identity, payload)] = results["runs"].items()
    assert identity == ("nvidia", engine.SVD_PROVIDER_VERSION, "synthetic_video")
    assert payload["clips"][0]["status"] == "error"
    assert payload["clips"][0]["score"] is None
    assert payload["evidence_counts"] == {"total_clips": {}}


@pytest.mark.parametrize("status", ["FAILED", "SUCCESS"])
def test_a_signal_without_a_provider_version_goes_into_no_results_file(status, tmp_path):
    """Regression (Architect, T6 PASS WITH FIXES): no "unversioned" detector identity."""
    rows = [
        signal(SHA_A, signal_id="s-1"),
        signal(SHA_B, signal_id="s-2", status=status, score=None if status == "FAILED" else 0.5,
               version=None),
    ]
    results = ex.build_results(rows, {SHA_A, SHA_B})
    assert list(results["runs"]) == [("nvidia", engine.SVD_PROVIDER_VERSION, "synthetic_video")]
    assert results["exclusion_counts"]["missing_provider_version"] == 1
    for payload in results["runs"].values():
        assert payload["provider_version"] is not None
        assert SHA_B not in {clip["clip_id"] for clip in payload["clips"]}

    corpus = ex.build_corpus(
        [truth(SHA_A), truth(SHA_B)],
        [governance(SHA_A), governance(SHA_B, lineage="lineage-2")],
        [media(SHA_A), media(SHA_B, media_id="m-2")],
    )
    out = tmp_path / "export"
    metadata = ex.write_export(out, corpus, results, split_filter=None,
                               exporter_identity={}, source_db={})
    assert metadata["exclusion_counts"]["missing_provider_version"] == 1
    for path in out.glob("results_*.json"):
        payload = json.loads(path.read_text())
        assert payload["provider_version"] is not None
        assert SHA_B not in {clip["clip_id"] for clip in payload["clips"]}
        assert "unversioned" not in path.name


def test_signals_of_media_that_was_not_exported_are_left_out():
    results = ex.build_results([signal(SHA_B)], {SHA_A})
    assert results["runs"] == {}
    assert results["signals_for_unexported_media"] == 1


def test_counts_are_explicit_evidence_counts_and_no_provenance_is_written(tmp_path):
    """Regression (Architect, T6 PASS WITH FIXES): no fabricated `model_provenance`."""
    results = ex.build_results([signal(metadata={"total_clips": 7})], {SHA_A})
    [payload] = results["runs"].values()
    assert payload["evidence_counts"] == {"total_clips": {SHA_A: 7}}
    assert "model_provenance" not in payload["run"]
    assert "model_provenance" not in json.dumps(payload)

    corpus = ex.build_corpus([truth()], [governance()], [media()])
    out = tmp_path / "export"
    ex.write_export(out, corpus, results, split_filter=None, exporter_identity={}, source_db={})
    [path] = out.glob("results_*.json")
    assert "model_provenance" not in path.read_text()

    run = replay_module.read_run(path)
    assert run["provenance"] == {}
    assert replay_module.clip_count(run, "total_clips", SHA_A) == 7
    svd, _, _ = replay_module.build_evidence(
        SHA_A, engine, run, replay_module.read_run(None), replay_module.read_run(None),
        {"svd": engine.SVD_PROVIDER_VERSION, "face": None, "lip": None},
    )
    assert svd.total_clips == 7


def test_a_benchmark_run_still_reads_its_counts_from_provenance():
    run = {"clips": {}, "provenance": {"frames_scored_by_clip_id": {"c": 4}}, "ran": True}
    assert replay_module.clip_count(run, "frames_scored", "c") == 4


# --- writing, and replay reading it ----------------------------------------------------------


def _full_export(tmp_path):
    truths = [truth(SHA_A, label="GENUINE"), truth(SHA_B, label="FACE_SWAP")]
    gov = [governance(SHA_A), governance(SHA_B, lineage="lineage-2", split="HOLDOUT")]
    med = [media(SHA_A), media(SHA_A, media_id="m-0", created="2026-09-25T00:00:00+00:00"),
           media(SHA_B, media_id="m-2")]
    signals = [
        signal(SHA_A, signal_id="s-1", score=0.2),
        signal(SHA_B, signal_id="s-2", score=0.99),
        signal(SHA_B, signal_id="s-3", version="some-other-function-id", score=0.4),
        signal(SHA_A, signal_id="s-4", provider="efficientnet-b7", signal_type="face_manipulation",
               version=engine.FACE_PROVIDER_VERSION, score=0.1),
        signal(SHA_B, signal_id="s-5", provider="efficientnet-b7", signal_type="face_manipulation",
               version=engine.FACE_PROVIDER_VERSION, score=0.999),
        signal(SHA_B, signal_id="s-6", provider="effort", signal_type="face_forgery", metadata={}),
    ]
    corpus = ex.build_corpus(truths, gov, med)
    results = ex.build_results(signals, {i.clip_id for i in corpus["items"]})
    out = tmp_path / "export"
    metadata = ex.write_export(
        out, corpus, results, split_filter=None,
        exporter_identity={"script": "test"}, source_db={"database": "fixture"},
    )
    return out, metadata


def test_the_export_writes_every_artifact_and_hashes_them(tmp_path):
    out, metadata = _full_export(tmp_path)
    for key in ("export_timestamp", "exporter_identity", "source_db", "exported_sha_count",
                "exclusion_counts", "duplicate_counts", "split_filter", "artifact_hashes"):
        assert key in metadata
    assert metadata["exported_sha_count"] == 2
    assert metadata["duplicate_counts"] == {"media_files": 1, "signals": 0}
    assert metadata["exclusion_counts"]["unsupported_signal_schema"] == 1
    for reason in ("insufficient_corpus_metadata", "ground_truth_ineligible", "unknown_label",
                   "unsupported_label", "unsupported_signal_schema", "invalid_success_payload"):
        assert reason in metadata["exclusion_counts"]
    results = sorted(p.name for p in out.glob("results_*.json"))
    assert len(results) == 3
    assert set(metadata["artifact_hashes"]) == {"corpus.json", "manifest.csv", *results}
    for name, digest in metadata["artifact_hashes"].items():
        assert ex._sha256_file(out / name) == digest
    assert json.loads((out / "export_metadata.json").read_text()) == metadata


def test_an_export_is_never_written_over_another(tmp_path):
    out, _ = _full_export(tmp_path)
    corpus = ex.build_corpus([truth()], [governance()], [media()])
    with pytest.raises(ex.ExportError, match="not empty"):
        ex.write_export(out, corpus, ex.build_results([], set()), split_filter=None,
                        exporter_identity={}, source_db={})


def _results_file(out, provider, version):
    [path] = [
        p for p in out.glob("results_*.json")
        if (lambda d: (d["provider"], d["provider_version"]) == (provider, version))(
            json.loads(p.read_text())
        )
    ]
    return path


def test_replay_consumes_the_export_under_the_explicit_provider_version(tmp_path):
    out, _ = _full_export(tmp_path)
    svd = _results_file(out, "nvidia", engine.SVD_PROVIDER_VERSION)
    face = _results_file(out, "efficientnet-b7", engine.FACE_PROVIDER_VERSION)
    report_path = tmp_path / "replay.json"

    assert replay_module.main([
        "--corpus", str(out / "corpus.json"), "--svd-run", str(svd), "--face-run", str(face),
        "--output", str(report_path),
    ]) == 0

    report = json.loads(report_path.read_text())
    assert report["eligibility"]["svd"]["observed_provider_version"] == engine.SVD_PROVIDER_VERSION
    assert report["eligibility"]["svd"]["matches_calibrated_deployment"] is True
    assert report["eligibility"]["face"]["matches_calibrated_deployment"] is True
    assert report["eligibility"]["lip"]["observed_provider_version"] is None
    decisions = {d["clip_id"]: d for d in report["decisions"]}
    assert set(decisions) == {SHA_A, SHA_B}
    assert decisions[SHA_B]["scores"] == {"svd": 0.99, "face": 0.999, "lip": None}
    assert decisions[SHA_B]["counts"]["svd_total_clips"] == 3
    assert decisions[SHA_B]["split"] == "HOLDOUT"


def test_replay_uses_the_other_version_exactly_and_finds_it_uncalibrated(tmp_path):
    out, _ = _full_export(tmp_path)
    other = _results_file(out, "nvidia", "some-other-function-id")
    report_path = tmp_path / "replay.json"
    replay_module.main(["--corpus", str(out / "corpus.json"), "--svd-run", str(other),
                        "--output", str(report_path)])
    report = json.loads(report_path.read_text())
    assert report["eligibility"]["svd"]["observed_provider_version"] == "some-other-function-id"
    assert report["eligibility"]["svd"]["matches_calibrated_deployment"] is False


# --- replay: the explicit version is optional and exact -------------------------------------


def _write(tmp_path, payload):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    return path


def test_a_run_without_an_explicit_version_is_reconstructed_as_before(tmp_path):
    run = replay_module.read_run(_write(tmp_path, {
        "run": {"model_provenance": {"observed_function_ids": ["fn-1"]}}, "clips": []}))
    assert "explicit_provider_version" not in run
    assert replay_module.run_provider_version(run, replay_module.svd_provider_version) == "fn-1"


def test_an_explicit_version_wins_over_provenance_even_when_null(tmp_path):
    run = replay_module.read_run(_write(tmp_path, {
        "provider_version": None,
        "run": {"model_provenance": {"observed_function_ids": ["fn-1"]}}, "clips": []}))
    assert replay_module.run_provider_version(run, replay_module.svd_provider_version) is None


def test_an_explicit_version_that_is_not_a_string_is_refused(tmp_path):
    with pytest.raises(ValueError, match="provider_version"):
        replay_module.read_run(_write(tmp_path, {"provider_version": 5, "run": {}, "clips": []}))


# --- the restated vocabularies against their sources ---------------------------------------


def _literal_values(module_path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(elt) for elt in node.value.slice.elts)
    raise AssertionError(f"{name} not found in {module_path}")


def test_ground_truth_and_split_vocabularies_match_their_contracts():
    assert _literal_values(API_APP / "ground_truth.py", "SourceClass") == ex.SOURCE_CLASSES
    assert _literal_values(API_APP / "ground_truth.py", "GroundTruthLabel") == ex.GROUND_TRUTH_LABELS
    assert _literal_values(API_APP / "dataset_governance.py", "DatasetSplit") == ex.OPERATIONAL_SPLITS


def test_replay_slots_are_the_risk_engine_detectors():
    assert set(ex.REPLAY_SLOTS) == {
        (engine.SVD_PROVIDER, engine.SVD_SIGNAL_TYPE),
        (engine.FACE_PROVIDER, engine.FACE_SIGNAL_TYPE),
        (engine.LIP_PROVIDER, engine.LIP_SIGNAL_TYPE),
    }
    source = Path(replay_module.__file__).read_text()
    for slot in ex.REPLAY_SLOTS.values():
        assert f'"{slot.metadata_count_key}"' in source


def test_the_eight_corpus_fields_are_the_governance_columns():
    assert set(ex.CORPUS_METADATA_FIELDS) == {
        name for name in ex.GovernanceRow.__dataclass_fields__
        if name not in {"media_sha256", "source_lineage_id", "dataset_split",
                        "derived_from_sha256", "transformations"}
    }
