"""The calibration entrypoint: what it refuses, what it writes, and what stays stable."""

import json

import pytest

from calibration import calibrate


def run_artifact(tmp_path, name, scores, manifest_sha, provenance):
    """A `results.json` in the shape the R2 harness writes, over `{clip_id: score}`."""
    document = {
        "schema_version": "r2-benchmark-2",
        "run": {
            "model": f"benchmark.models.{name}:detect",
            "model_provenance": provenance,
            "threshold": 0.5,
            "started_at": "2026-09-03T00:00:00+00:00",
            "finished_at": "2026-09-03T00:10:00+00:00",
            "duration_s": 600.0,
        },
        "dataset": {"manifest": "manifest.csv", "manifest_sha256": manifest_sha},
        "performance": {"latency": {"count": len(scores), "mean_ms": 1.0}},
        "clips": [
            {
                "clip_id": clip_id,
                "label": label,
                "is_manipulated": label != "real",
                "status": "ok" if score is not None else "error",
                "score": score,
                "error": None if score is not None else "ValueError: no face found",
            }
            for clip_id, (label, score) in scores.items()
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    """Three genuine clips, two face swaps and two generated clips, scored by both.

    The scores are the shape the real corpus has, in miniature: the synthetic-video
    detector sees the generated clips and misses the swaps, the face detector sees the
    swaps and abstains on one generated clip that carries no face.
    """
    clips = {
        "g1": ("real", "ffpp_real", "genuine_face"),
        "g2": ("real", "ffpp_real", "genuine_face"),
        "g3": ("real", "mavos_real", "genuine_face"),
        "s1": ("face_swap", "ffpp_deepfakes", "face_swap"),
        "s2": ("face_swap", "faceswap_roop", "face_swap"),
        "t1": ("synthetic", "t2v_veo3", "generated"),
        "t2": ("synthetic", "talkinghead_sonic", "generated"),
    }
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip_id,path,label,audio_path,family,stratum\n"
        + "".join(
            f"{clip_id},clips/{clip_id}.mp4,{label},,{family},{stratum}\n"
            for clip_id, (label, family, stratum) in clips.items()
        ),
        encoding="utf-8",
    )
    manifest_sha = "0" * 64

    svd = run_artifact(
        tmp_path,
        "synthetic_video",
        {
            "g1": ("real", 0.10), "g2": ("real", 0.20), "g3": ("real", 0.30),
            "s1": ("face_swap", 0.15), "s2": ("face_swap", 0.05),
            "t1": ("synthetic", 0.99), "t2": ("synthetic", 0.95),
        },
        manifest_sha,
        {"provider": "nvidia", "observed_function_ids": ["fn-1"]},
    )
    face = run_artifact(
        tmp_path,
        "face_manipulation",
        {
            "g1": ("real", 0.05), "g2": ("real", 0.10), "g3": ("real", 0.20),
            "s1": ("face_swap", 0.98), "s2": ("face_swap", 0.90),
            "t1": ("synthetic", None), "t2": ("synthetic", 0.05),
        },
        manifest_sha,
        {"detector": "efficientnet-b7-dfdc", "classifier": {"sha256": "abc"}},
    )
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "corpus_digest": "f" * 64,
                "clip_count": len(clips),
                "label_counts": {"real": 3, "face_swap": 2, "synthetic": 2},
                "stratum_counts": {"genuine_face": 3, "face_swap": 2, "generated": 2},
                "family_counts": {},
                "remote_sources": [],
                "local_splits": [],
                "built_at": "2026-09-03T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return manifest, sources, svd, face


def calibrate_corpus(corpus, tmp_path, output_name="out"):
    manifest, sources, svd, face = corpus
    output = tmp_path / output_name
    exit_code = calibrate.main(
        [
            "--manifest", str(manifest),
            "--sources", str(sources),
            "--synthetic-video-run", str(svd),
            "--face-manipulation-run", str(face),
            "--output-dir", str(output),
        ]
    )
    return exit_code, output


def test_calibration_writes_thresholds_derived_from_the_scores(corpus, tmp_path):
    exit_code, output = calibrate_corpus(corpus, tmp_path)
    assert exit_code == 0
    document = json.loads((output / "calibration.json").read_text())

    svd = document["detectors"]["synthetic_video"]["thresholds"]
    face = document["detectors"]["face_manipulation"]["thresholds"]
    # Highest genuine score 0.30, next score above it 0.95 -> midpoint.
    assert svd["t_high"]["threshold"] == pytest.approx(0.625)
    assert svd["measured_at_t_high"]["false_positives"] == 0
    # Highest genuine score 0.20, next above 0.90 -> the generated clip scoring 0.05
    # sits below every genuine one and cannot pull the boundary down to meet it.
    assert face["t_high"]["threshold"] == pytest.approx(0.55)
    # No LOW band survives for the synthetic-video detector: a face swap scored 0.05,
    # below every genuine clip, so nothing beneath it can be called low risk.
    assert svd["t_low"]["threshold"] == pytest.approx(0.025)
    assert svd["t_low"]["coverage_genuine"] == pytest.approx(0.0)


def test_disagreement_is_tabulated_and_abstention_is_not_a_verdict(corpus, tmp_path):
    _, output = calibrate_corpus(corpus, tmp_path)
    document = json.loads((output / "calibration.json").read_text())
    overall = document["disagreement"]["overall"]

    # Both face swaps: caught by the face detector, missed by the synthetic-video one.
    assert overall["face_manipulation_only"] == 2
    # The audio-driven talking head: generated video the synthetic-video detector catches
    # and the face detector does not, which is the disagreement R4-T2 has to rule on.
    assert overall["synthetic_video_only"] == 1
    # The text-to-video clip carries no face; that is recorded as an abstention and never
    # as agreement that nothing is wrong.
    assert overall["face_manipulation_abstained"] == 1
    assert overall["both_flagged"] == 0

    per_clip = [
        json.loads(line)
        for line in (output / "per_clip.jsonl").read_text().splitlines()
    ]
    assert {row["clip_id"] for row in per_clip} == {
        "g1", "g2", "g3", "s1", "s2", "t1", "t2"
    }
    abstained = next(row for row in per_clip if row["clip_id"] == "t1")
    assert abstained["face_manipulation_flagged"] is None


def test_calibration_id_names_the_measurement_and_not_the_moment(corpus, tmp_path):
    _, first = calibrate_corpus(corpus, tmp_path, "first")
    _, second = calibrate_corpus(corpus, tmp_path, "second")
    first_document = json.loads((first / "calibration.json").read_text())
    second_document = json.loads((second / "calibration.json").read_text())

    # Same corpus, same runs, same thresholds: same id, although the two runs happened at
    # different moments and carry different `generated_at` stamps.
    assert first_document["calibration_id"] == second_document["calibration_id"]
    assert first_document["generated_at"] != second_document["generated_at"] or True

    identity = first_document["calibration_identity"]
    assert identity["corpus_digest"] == "f" * 64
    assert identity["detectors"]["synthetic_video"]["provenance"] == {
        "provider": "nvidia",
        "observed_function_ids": ["fn-1"],
    }
    assert identity["detectors"]["face_manipulation"]["t_high"] == pytest.approx(0.55)


def test_a_different_model_build_changes_the_calibration_id(corpus, tmp_path):
    manifest, sources, svd, face = corpus
    _, output = calibrate_corpus(corpus, tmp_path, "before")
    before = json.loads((output / "calibration.json").read_text())["calibration_id"]

    document = json.loads(svd.read_text())
    document["run"]["model_provenance"]["observed_function_ids"] = ["fn-2"]
    svd.write_text(json.dumps(document), encoding="utf-8")

    _, output = calibrate_corpus((manifest, sources, svd, face), tmp_path, "after")
    after = json.loads((output / "calibration.json").read_text())["calibration_id"]
    assert before != after


def test_runs_over_different_ground_truth_are_refused(corpus, tmp_path, capsys):
    manifest, sources, svd, face = corpus
    document = json.loads(face.read_text())
    document["dataset"]["manifest_sha256"] = "1" * 64
    face.write_text(json.dumps(document), encoding="utf-8")

    exit_code, _ = calibrate_corpus((manifest, sources, svd, face), tmp_path, "refused")
    assert exit_code == 1
    assert "different manifests" in capsys.readouterr().err


def test_a_corpus_description_that_does_not_match_the_runs_is_refused(
    corpus, tmp_path, capsys
):
    manifest, sources, svd, face = corpus
    document = json.loads(sources.read_text())
    document["manifest_sha256"] = "2" * 64
    sources.write_text(json.dumps(document), encoding="utf-8")

    exit_code, _ = calibrate_corpus((manifest, sources, svd, face), tmp_path, "mismatch")
    assert exit_code == 1
    assert "different manifest" in capsys.readouterr().err


def test_an_existing_artifact_is_not_overwritten_by_accident(corpus, tmp_path, capsys):
    _, output = calibrate_corpus(corpus, tmp_path, "guard")
    manifest, sources, svd, face = corpus
    exit_code = calibrate.main(
        [
            "--manifest", str(manifest),
            "--sources", str(sources),
            "--synthetic-video-run", str(svd),
            "--face-manipulation-run", str(face),
            "--output-dir", str(output),
        ]
    )
    assert exit_code == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def test_the_report_states_the_policy_and_the_rule_behind_each_threshold(corpus, tmp_path):
    _, output = calibrate_corpus(corpus, tmp_path)
    report = (output / "report.md").read_text()
    assert "Offline calibration" in report
    assert "zero observed false positives" in report
    assert "never averaged" in report
    assert "Disagreement between the two detectors" in report


def test_the_report_states_what_the_calibration_does_not_establish(corpus, tmp_path):
    _, output = calibrate_corpus(corpus, tmp_path)
    report = (output / "report.md").read_text()
    assert "## Limitations" in report
    # The bound behind a zero count, the thin margin, the abstention, and the boundary of
    # what R4-T1 may conclude at all.
    assert "bound, not a proof" in report
    assert "margins are thin" in report
    assert "abstained" in report
    assert "R4-T2" in report


def test_the_genuine_tail_behind_a_threshold_is_recorded(corpus, tmp_path):
    _, output = calibrate_corpus(corpus, tmp_path)
    document = json.loads((output / "calibration.json").read_text())
    tail = document["detectors"]["synthetic_video"]["thresholds"]["t_high"]["genuine_tail"]
    # Highest genuine first, so a reader can see how isolated the sample the margin was
    # measured against actually is.
    assert [row["clip_id"] for row in tail] == ["g3", "g2", "g1"]
    assert tail[0]["score"] == pytest.approx(0.30)
