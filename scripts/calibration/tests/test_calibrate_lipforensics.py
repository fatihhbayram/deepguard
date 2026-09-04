"""The LipForensics calibration: what it refuses, what it derives, and what stays stable."""

import hashlib
import json

import pytest

from calibration import calibrate_lipforensics as calibrate


def run_artifact(tmp_path, name, scores, manifest_sha, device="cpu"):
    """A `results.json` in the shape the R2 harness writes, over `{clip_id: (label, score)}`."""
    document = {
        "schema_version": "r2-benchmark-2",
        "run": {
            "model": calibrate.EXPECTED_MODEL,
            "model_provenance": {
                "detector": "lipforensics",
                "classifier": {"artifact": "lipforensics_ff.pth", "sha256": "a" * 64},
                "upstream": {"repository": "https://example.invalid", "revision": "b" * 40},
                "landmarks": {"library": "face-alignment 1.5.0"},
                "device": device,
                "windows": 4,
                "frames_per_window": 25,
                "crop_size": 96,
                "input_size": 88,
                "score_semantics": "sigmoid(mean logit over sampled 25-frame mouth runs)",
            },
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
                "error": None if score is not None else "ValueError: no mouth tracked",
            }
            for clip_id, (label, score) in scores.items()
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# Three genuine clips and three face swaps, separated by a wide gap so the selection rules
# have an unambiguous answer that can be checked by hand: the highest genuine score is
# 0.02, the lowest manipulated one 0.40, and the midpoint between them is 0.21.
CLIPS = {
    "real_a": ("real", "FaceForensics++_C23/real/001.mp4", 0.001),
    "real_b": ("real", "FaceForensics++_C23/real/002.mp4", 0.02),
    "real_c": ("real", "FaceForensics++_C23/real/003.mp4", 0.0004),
    "Deepfakes_x": ("face_swap", "FaceForensics++_C23/fake/Deepfakes/001_002.mp4", 0.40),
    "Deepfakes_y": ("face_swap", "FaceForensics++_C23/fake/Deepfakes/003_004.mp4", 0.99),
    "FaceSwap_z": ("face_swap", "FaceForensics++_C23/fake/FaceSwap/005_006.mp4", 1.0),
}


@pytest.fixture
def corpus(tmp_path):
    """A manifest, the clip files it points at, and the run that scored them."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for clip_id in CLIPS:
        (clips_dir / f"{clip_id}.mp4").write_bytes(clip_id.encode("utf-8"))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip_id,path,label,audio_path,source\n"
        + "".join(
            f"{clip_id},clips/{clip_id}.mp4,{label},,{source}\n"
            for clip_id, (label, source, _) in CLIPS.items()
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    run = run_artifact(
        tmp_path,
        "run",
        {clip_id: (label, score) for clip_id, (label, _, score) in CLIPS.items()},
        manifest_sha,
    )
    return {"dir": tmp_path, "manifest": manifest, "sha": manifest_sha, "run": run}


def calibrate_corpus(corpus, output, *extra):
    return calibrate.main(
        [
            "--manifest", str(corpus["manifest"]),
            "--run", str(corpus["run"]),
            "--output-dir", str(output),
            *extra,
        ]
    )


def read_artifact(output):
    return json.loads((output / "calibration.json").read_text(encoding="utf-8"))


class TestProvenanceMapping:
    def test_family_and_stratum_come_from_the_source_path(self):
        assert calibrate.clip_family_and_stratum(
            "FaceForensics++_C23/fake/Deepfakes/048_029.mp4", "face_swap"
        ) == ("ffpp_deepfakes", calibrate.FACE_SWAP_STRATUM)
        assert calibrate.clip_family_and_stratum(
            "FaceForensics++_C23/real/012.mp4", "real"
        ) == (calibrate.GENUINE_FAMILY, calibrate.GENUINE_STRATUM)

    def test_an_unrecognised_source_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="unrecognised source provenance"):
            calibrate.clip_family_and_stratum("some/other/corpus.mp4", "real")


class TestRefusals:
    def test_a_run_over_a_different_manifest_is_refused(self, corpus, tmp_path, capsys):
        mismatched = run_artifact(
            tmp_path,
            "mismatched",
            {clip_id: (label, score) for clip_id, (label, _, score) in CLIPS.items()},
            "0" * 64,
        )
        assert calibrate.main(
            ["--manifest", str(corpus["manifest"]), "--run", str(mismatched),
             "--output-dir", str(tmp_path / "out")]
        ) == 1
        assert "different manifest" in capsys.readouterr().err

    def test_a_run_from_another_device_is_not_the_run_calibrated_from(
        self, corpus, tmp_path, capsys
    ):
        cuda = run_artifact(
            tmp_path,
            "cuda",
            {clip_id: (label, score) for clip_id, (label, _, score) in CLIPS.items()},
            corpus["sha"],
            device="cuda",
        )
        assert calibrate.main(
            ["--manifest", str(corpus["manifest"]), "--run", str(cuda),
             "--output-dir", str(tmp_path / "out")]
        ) == 1
        assert calibrate.PRODUCTION_DEVICE in capsys.readouterr().err

    def test_another_detectors_run_is_refused(self, corpus, tmp_path):
        document = json.loads(corpus["run"].read_text(encoding="utf-8"))
        document["run"]["model"] = "benchmark.models.face_manipulation:detect"
        other = tmp_path / "other.json"
        other.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=calibrate.EXPECTED_MODEL):
            calibrate.load_run(other, {})

    def test_a_missing_clip_file_is_refused(self, corpus, tmp_path):
        (corpus["dir"] / "clips" / "real_a.mp4").unlink()
        with pytest.raises(FileNotFoundError, match="corpus digest"):
            calibrate_corpus(corpus, tmp_path / "out")

    def test_an_existing_artifact_is_not_overwritten_without_the_flag(
        self, corpus, tmp_path, capsys
    ):
        output = tmp_path / "out"
        assert calibrate_corpus(corpus, output) == 0
        assert calibrate_corpus(corpus, output) == 1
        assert "refusing to overwrite" in capsys.readouterr().err
        assert calibrate_corpus(corpus, output, "--overwrite") == 0


class TestSelection:
    def test_the_threshold_is_the_midpoint_of_the_measured_gap(self, corpus, tmp_path):
        output = tmp_path / "out"
        assert calibrate_corpus(corpus, output) == 0
        high = read_artifact(output)["detectors"]["lip_forensics"]["thresholds"]["t_high"]
        # Highest genuine 0.02, lowest score above it 0.40, so the midpoint is 0.21.
        assert high["genuine_max"] == pytest.approx(0.02)
        assert high["next_score_above_genuine_max"] == pytest.approx(0.40)
        assert high["threshold"] == pytest.approx(0.21)

    def test_the_benchmark_reporting_threshold_is_not_the_selected_one(
        self, corpus, tmp_path
    ):
        output = tmp_path / "out"
        assert calibrate_corpus(corpus, output) == 0
        detector = read_artifact(output)["detectors"]["lip_forensics"]
        selected = detector["thresholds"]["t_high"]["threshold"]
        assert selected != calibrate.BENCHMARK_REPORTING_THRESHOLD

        comparison = detector["benchmark_threshold_comparison"]
        assert comparison["reporting_threshold"] == 0.5
        # 0.5 sits above one manipulated clip here, and the artifact names it rather than
        # asserting that the benchmark threshold was worse.
        assert [
            record["clip_id"]
            for record in comparison["manipulated_below_reporting_threshold"]
        ] == ["Deepfakes_x"]
        assert comparison["at_reporting_threshold"]["true_positive_rate"] == pytest.approx(2 / 3)
        assert comparison["at_derived_threshold"]["true_positive_rate"] == pytest.approx(1.0)

    def test_a_clean_gap_is_recorded_as_one_rather_than_read_as_no_ambiguity(
        self, corpus, tmp_path
    ):
        output = tmp_path / "out"
        assert calibrate_corpus(corpus, output) == 0
        thresholds = read_artifact(output)["detectors"]["lip_forensics"]["thresholds"]
        band = thresholds["band_between_them"]
        assert thresholds["t_low"]["threshold"] == thresholds["t_high"]["threshold"]
        assert band["kind"] == "clean_gap"
        assert "not a finding that the detector has no ambiguous region" in band["reading"]

    def test_an_overlapping_corpus_yields_an_ambiguous_band(self, tmp_path):
        rows = [
            {"clip_id": "g1", "label": "real", "score": 0.1, "family": "ffpp_real",
             "stratum": calibrate.GENUINE_STRATUM},
            {"clip_id": "g2", "label": "real", "score": 0.6, "family": "ffpp_real",
             "stratum": calibrate.GENUINE_STRATUM},
            {"clip_id": "m1", "label": "face_swap", "score": 0.3,
             "family": "ffpp_deepfakes", "stratum": calibrate.FACE_SWAP_STRATUM},
            {"clip_id": "m2", "label": "face_swap", "score": 0.9,
             "family": "ffpp_deepfakes", "stratum": calibrate.FACE_SWAP_STRATUM},
        ]
        thresholds = calibrate.select({"rows": rows})
        band = thresholds["band_between_them"]
        assert band["kind"] == "ambiguous_band"
        assert band["width"] > 0
        assert (
            thresholds["t_low"]["threshold"] < thresholds["t_high"]["threshold"]
        )


class TestCrossDevice:
    def test_a_second_device_run_is_compared_but_never_selected_from(
        self, corpus, tmp_path
    ):
        moved = dict(CLIPS)
        moved["Deepfakes_x"] = ("face_swap", moved["Deepfakes_x"][1], 0.33)
        cuda = run_artifact(
            corpus["dir"],
            "cuda",
            {clip_id: (label, score) for clip_id, (label, _, score) in moved.items()},
            corpus["sha"],
            device="cuda",
        )
        output = tmp_path / "out"
        assert calibrate_corpus(corpus, output, "--cross-device-run", str(cuda)) == 0
        detector = read_artifact(output)["detectors"]["lip_forensics"]

        # The selection still comes from the CPU run alone.
        assert detector["thresholds"]["t_high"]["threshold"] == pytest.approx(0.21)
        stability = detector["cross_device_stability"]
        assert stability["max_abs_delta"] == pytest.approx(0.07)
        assert stability["decision_flips_at_selected_threshold"] == []
        assert stability["t_high_reselected_on_other_device"] == pytest.approx(0.175)
        assert stability["selection_rule_is_device_stable"] is True

    def test_a_decision_that_flips_across_devices_is_reported(self, corpus, tmp_path):
        flipped = dict(CLIPS)
        # Below the 0.21 threshold the CPU run selects: the clip changes side.
        flipped["Deepfakes_x"] = ("face_swap", flipped["Deepfakes_x"][1], 0.05)
        cuda = run_artifact(
            corpus["dir"],
            "cuda",
            {clip_id: (label, score) for clip_id, (label, _, score) in flipped.items()},
            corpus["sha"],
            device="cuda",
        )
        output = tmp_path / "out"
        assert calibrate_corpus(corpus, output, "--cross-device-run", str(cuda)) == 0
        stability = read_artifact(output)["detectors"]["lip_forensics"][
            "cross_device_stability"
        ]
        assert stability["decision_flips_at_selected_threshold"] == ["Deepfakes_x"]
        assert stability["selection_rule_is_device_stable"] is False


class TestArtifact:
    def test_the_same_inputs_produce_the_same_calibration_id(self, corpus, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        assert calibrate_corpus(corpus, first) == 0
        assert calibrate_corpus(corpus, second) == 0
        left, right = read_artifact(first), read_artifact(second)
        assert left["calibration_id"] == right["calibration_id"]
        assert left["calibration_identity"] == right["calibration_identity"]
        # The id identifies the measurement, and nothing that merely describes the run
        # may enter it — the two artifacts carry different timestamps and the same id.
        assert "generated_at" not in left["calibration_identity"]

    def test_the_id_changes_when_the_measured_scores_change(self, corpus, tmp_path):
        output = tmp_path / "a"
        assert calibrate_corpus(corpus, output) == 0
        before = read_artifact(output)["calibration_id"]

        moved = dict(CLIPS)
        moved["real_b"] = ("real", moved["real_b"][1], 0.05)
        corpus["run"] = run_artifact(
            corpus["dir"],
            "run",
            {clip_id: (label, score) for clip_id, (label, _, score) in moved.items()},
            corpus["sha"],
        )
        assert calibrate_corpus(corpus, tmp_path / "b") == 0
        assert read_artifact(tmp_path / "b")["calibration_id"] != before

    def test_the_id_changes_when_the_model_provenance_changes(self, corpus, tmp_path):
        output = tmp_path / "a"
        assert calibrate_corpus(corpus, output) == 0
        before = read_artifact(output)["calibration_id"]

        document = json.loads(corpus["run"].read_text(encoding="utf-8"))
        document["run"]["model_provenance"]["classifier"]["sha256"] = "c" * 64
        corpus["run"].write_text(json.dumps(document), encoding="utf-8")
        assert calibrate_corpus(corpus, tmp_path / "b") == 0
        assert read_artifact(tmp_path / "b")["calibration_id"] != before

    def test_dataset_identity_names_the_media_and_not_only_the_manifest(
        self, corpus, tmp_path
    ):
        output = tmp_path / "a"
        assert calibrate_corpus(corpus, output) == 0
        artifact = read_artifact(output)
        assert artifact["corpus"]["manifest_sha256"] == corpus["sha"]
        assert artifact["corpus"]["clip_count"] == len(CLIPS)
        assert artifact["corpus"]["source_datasets"] == {"FaceForensics++_C23": len(CLIPS)}
        assert {record["clip_id"] for record in artifact["corpus"]["clips"]} == set(CLIPS)

        # Repoint the manifest at different bytes under the same clip ids: the manifest
        # digest moves, but so must the corpus digest, which is the point of hashing media.
        before = artifact["corpus"]["corpus_digest"]
        (corpus["dir"] / "clips" / "real_a.mp4").write_bytes(b"different media")
        assert calibrate_corpus(corpus, tmp_path / "b") == 0
        assert read_artifact(tmp_path / "b")["corpus"]["corpus_digest"] != before

    def test_score_semantics_and_provenance_survive_into_the_artifact(
        self, corpus, tmp_path
    ):
        output = tmp_path / "a"
        assert calibrate_corpus(corpus, output) == 0
        detector = read_artifact(output)["detectors"]["lip_forensics"]
        assert detector["score_semantics"] == (
            "sigmoid(mean logit over sampled 25-frame mouth runs)"
        )
        provenance = detector["run"]["model_provenance"]
        assert provenance["classifier"]["sha256"] == "a" * 64
        assert provenance["upstream"]["revision"] == "b" * 40
        assert detector["run"]["device"] == calibrate.PRODUCTION_DEVICE

    def test_abstentions_are_carried_through_rather_than_dropped(self, corpus, tmp_path):
        withheld = dict(CLIPS)
        withheld["real_c"] = ("real", withheld["real_c"][1], None)
        corpus["run"] = run_artifact(
            corpus["dir"],
            "run",
            {clip_id: (label, score) for clip_id, (label, _, score) in withheld.items()},
            corpus["sha"],
        )
        output = tmp_path / "a"
        assert calibrate_corpus(corpus, output) == 0
        detector = read_artifact(output)["detectors"]["lip_forensics"]
        assert [record["clip_id"] for record in detector["abstentions"]] == ["real_c"]
        assert detector["measurements"]["abstentions"] == 1
        assert detector["measurements"]["scored"] == len(CLIPS) - 1

        rows = [
            json.loads(line)
            for line in (output / "per_clip.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == len(CLIPS)
        # An abstention is not a clip that was found clean.
        abstained = next(row for row in rows if row["clip_id"] == "real_c")
        assert abstained["flagged_at_t_high"] is None

    def test_the_report_states_the_rule_and_names_the_rejected_threshold(
        self, corpus, tmp_path
    ):
        output = tmp_path / "a"
        assert calibrate_corpus(corpus, output) == 0
        report = (output / "report.md").read_text(encoding="utf-8")
        assert "lowest threshold with zero observed false positives" in report
        assert "**not selected**" in report
        assert "Why not `0.5`" in report
        assert "reported, not selected" in report
