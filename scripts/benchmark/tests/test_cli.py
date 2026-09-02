"""The run loop and the artifacts it writes, including a model that fails mid-corpus."""

import hashlib
import json
from datetime import datetime, timezone

import pytest

from benchmark.cli import (
    build_results,
    main,
    mock_model,
    per_label_breakdown,
    resolve_model,
    run_benchmark,
)
from benchmark.dataset import Clip, load_manifest


def make_clip(clip_id, label, tmp_path):
    path = tmp_path / f"{clip_id}.mp4"
    path.write_bytes(b"dummy")
    return Clip(clip_id=clip_id, path=path, label=label)


def build_corpus(tmp_path):
    """Two genuine clips and two manipulated ones, plus the manifest describing them."""
    clips = [
        make_clip("clip_a", "real", tmp_path),
        make_clip("clip_b", "real", tmp_path),
        make_clip("clip_c", "synthetic", tmp_path),
        make_clip("clip_d", "face_swap", tmp_path),
    ]
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip_id,path,label\n"
        + "".join(f"{c.clip_id},{c.path.name},{c.label}\n" for c in clips),
        encoding="utf-8",
    )
    return clips, manifest


def perfect_model(clip):
    return 0.9 if clip.is_manipulated else 0.1


def test_mock_model_is_deterministic_and_in_range(tmp_path):
    clip = make_clip("clip_a", "real", tmp_path)
    assert mock_model(clip) == mock_model(clip)
    assert 0.0 <= mock_model(clip) < 1.0
    # Label-blind: the score follows the id and nothing else.
    relabelled = Clip(clip_id="clip_a", path=clip.path, label="face_swap")
    assert mock_model(relabelled) == mock_model(clip)


def test_resolve_model_accepts_mock_and_a_dotted_reference():
    assert resolve_model("mock") is mock_model
    assert resolve_model("benchmark.cli:mock_model") is mock_model


@pytest.mark.parametrize(
    "reference,message",
    [
        ("nope", "unknown model"),
        ("no_such_module_here:detect", "cannot import module"),
        ("benchmark.cli:not_a_thing", "has no attribute"),
        ("benchmark.cli:SCHEMA_VERSION", "not callable"),
    ],
)
def test_resolve_model_rejects_what_it_cannot_run(reference, message):
    with pytest.raises(ValueError, match=message):
        resolve_model(reference)


def test_run_benchmark_records_a_score_and_a_latency_per_clip(tmp_path):
    clips, _ = build_corpus(tmp_path)

    records, latencies = run_benchmark(clips, perfect_model, threshold=0.5)

    assert [r["status"] for r in records] == ["ok"] * 4
    assert all(r["correct"] for r in records)
    assert len(latencies) == 4
    assert all(latency >= 0 for latency in latencies)


def test_a_clip_the_model_crashes_on_is_recorded_and_excluded(tmp_path):
    _, manifest = build_corpus(tmp_path)
    dataset = load_manifest(manifest)

    def flaky(clip):
        if clip.clip_id == "clip_d":
            raise RuntimeError("decoder blew up")
        return perfect_model(clip)

    records, latencies = run_benchmark(dataset.clips, flaky, threshold=0.5)
    results = build_results(
        dataset=dataset,
        model_reference="flaky",
        threshold=0.5,
        records=records,
        latencies_ms=latencies,
        baseline_rss_mb=10.0,
        peak_mb=12.0,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )

    failed = [r for r in records if r["status"] == "error"]
    assert [r["clip_id"] for r in failed] == ["clip_d"]
    assert failed[0]["error"] == "RuntimeError: decoder blew up"
    assert failed[0]["score"] is None

    # The crash is not scored as a wrong answer: three clips reached the matrix.
    assert results["counts"] == {"scored": 3, "errors": 1}
    assert results["confusion_matrix"]["total"] == 3
    assert results["confusion_matrix"]["false_negatives"] == 0
    assert results["performance"]["latency"]["count"] == 3
    assert results["per_label"]["face_swap"] == {
        "total": 1,
        "scored": 0,
        "errors": 1,
        "flagged": 0,
        "flagged_rate": None,
    }


def test_a_non_finite_score_is_a_clip_error_not_a_crash(tmp_path):
    clips, _ = build_corpus(tmp_path)
    records, latencies = run_benchmark(clips, lambda clip: float("nan"), threshold=0.5)
    assert [r["status"] for r in records] == ["error"] * 4
    assert "non-finite score" in records[0]["error"]
    assert latencies == []


def test_per_label_breakdown_separates_the_families():
    records = [
        {"label": "real", "status": "ok", "predicted_manipulated": False},
        {"label": "real", "status": "ok", "predicted_manipulated": True},
        {"label": "face_swap", "status": "ok", "predicted_manipulated": True},
        {"label": "face_swap", "status": "error", "predicted_manipulated": None},
    ]
    assert per_label_breakdown(records) == {
        "face_swap": {
            "total": 2,
            "scored": 1,
            "errors": 1,
            "flagged": 1,
            "flagged_rate": 1.0,
        },
        "real": {
            "total": 2,
            "scored": 2,
            "errors": 0,
            "flagged": 1,
            "flagged_rate": 0.5,
        },
    }


def test_main_writes_the_artifacts(tmp_path, capsys):
    _, manifest = build_corpus(tmp_path)
    output = tmp_path / "run"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--model",
            "benchmark.tests.test_cli:perfect_model",
            "--output-dir",
            str(output),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert results["schema_version"] == "r2-benchmark-1"
    assert results["metrics"] == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
    }
    assert results["dataset"]["clip_count"] == 4
    assert len(results["dataset"]["manifest_sha256"]) == 64
    assert results["performance"]["memory"]["peak_rss_mb"] > 0
    assert [c["clip_id"] for c in results["clips"]] == [
        "clip_a",
        "clip_b",
        "clip_c",
        "clip_d",
    ]

    report = (output / "report.md").read_text(encoding="utf-8")
    assert "# Detector benchmark run" in report
    assert "False positive rate | 0.0000" in report


def test_main_refuses_to_overwrite_a_previous_result(tmp_path, capsys):
    _, manifest = build_corpus(tmp_path)
    output = tmp_path / "run"
    argv = [
        "--manifest",
        str(manifest),
        "--model",
        "mock",
        "--output-dir",
        str(output),
    ]

    assert main(argv) == 0
    original = (output / "results.json").read_text(encoding="utf-8")

    assert main(argv) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert (output / "results.json").read_text(encoding="utf-8") == original

    assert main(argv + ["--overwrite"]) == 0


def test_main_reports_a_broken_manifest_and_writes_nothing(tmp_path, capsys):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("clip_id,path,label\nx,gone.mp4,real\n", encoding="utf-8")
    output = tmp_path / "run"

    exit_code = main(
        ["--manifest", str(manifest), "--model", "mock", "--output-dir", str(output)]
    )

    assert exit_code == 1
    assert "media file not found" in capsys.readouterr().err
    assert not output.exists()


def test_fingerprint_is_the_manifest_that_produced_the_records(tmp_path):
    """A manifest edited mid-run must not relabel the artifact with the new bytes.

    The digest exists so a result set names the ground truth it was measured against.
    If it were taken after scoring, a manifest rewritten while the model was running
    would stamp records parsed from the *old* corpus with the digest of the *new* one —
    an artifact pointing at a file that never produced it, which is worse than no
    fingerprint at all because it looks authoritative.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (corpus / name).write_bytes(b"not really media")
    manifest = corpus / "manifest.csv"
    original_bytes = b"clip_id,path,label\na,a.mp4,real\n"
    manifest.write_bytes(original_bytes)

    ingested_sha = hashlib.sha256(original_bytes).hexdigest()

    def rewrite_manifest_then_score(clip):
        # The edit lands after ingestion and before the artifact is assembled, which is
        # the window a post-run hash would read from.
        manifest.write_bytes(b"clip_id,path,label\na,a.mp4,real\nb,b.mp4,synthetic\n")
        return 0.9

    dataset = load_manifest(manifest)
    records, latencies = run_benchmark(
        dataset.clips, rewrite_manifest_then_score, 0.5
    )
    results = build_results(
        dataset=dataset,
        model_reference="test",
        threshold=0.5,
        records=records,
        latencies_ms=latencies,
        baseline_rss_mb=1.0,
        peak_mb=2.0,
        started_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )

    assert manifest.read_bytes() != original_bytes  # the file really did change
    assert results["dataset"]["manifest_sha256"] == ingested_sha
    assert results["dataset"]["clip_count"] == 1
    assert [c["clip_id"] for c in results["clips"]] == ["a"]
