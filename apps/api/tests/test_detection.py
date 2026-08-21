"""Detection, exercised directly against a local artifact.

`app.detection` is what the worker runs on one video. It opens no transaction and writes
no row, so nothing here needs a database: the subject is what NVIDIA's answer — or its
refusal to give one — is turned into.
"""

import asyncio
from pathlib import Path

import pytest

from app import detection, nvidia_video
from app.c2pa_extractor import C2paEvidence
from app.detection import MAX_PERSISTED_SEGMENTS, detect_synthetic_video, extract_provenance

NVIDIA_PROBABILITY = 0.8734567165374756
NVIDIA_LOGIT = 1.9142135381698608
NVIDIA_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """Stand in for the detector, so no test in this module can reach NVIDIA.

    Autouse deliberately: a test that forgot to request it would mean real credentials,
    real network traffic and a real bill from the suite.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            # The per-clip evidence the provider reports back, overridable per test.
            self.clips = (nvidia_video.NvidiaClipResult(index=0, logit=-2.25),)
            self.paths = []

        async def analyze(self, file_path, **kwargs):
            self.paths.append(Path(file_path))
            if self.error:
                raise self.error
            return nvidia_video.NvidiaVideoResult(
                logit=NVIDIA_LOGIT,
                probability=NVIDIA_PROBABILITY,
                total_clips=7,
                csv_data="0,-2.25\n8,3.5\n",
                function_id=NVIDIA_FUNCTION_ID,
                clips=self.clips,
            )

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_video", recorder.analyze)
    return recorder


def run_detection(tmp_path):
    artifact = tmp_path / "canonical.mp4"
    artifact.write_bytes(b"canonical-mp4-bytes")

    return asyncio.run(detect_synthetic_video(artifact))


def test_a_successful_detection_becomes_a_nvidia_signal(tmp_path, fake_nvidia):
    signal, _ = run_detection(tmp_path)

    assert signal.provider == "nvidia"
    assert signal.signal_type == "synthetic_video"
    assert signal.status == "SUCCESS"
    assert signal.provider_version == NVIDIA_FUNCTION_ID


def test_the_signal_score_is_nvidias_probability_untouched(tmp_path, fake_nvidia):
    signal, _ = run_detection(tmp_path)

    # Not rounded, not rescaled to a percentage, not turned into a DeepGuard number.
    assert signal.score == NVIDIA_PROBABILITY


def test_the_signal_carries_no_risk_level(tmp_path, fake_nvidia):
    signal, _ = run_detection(tmp_path)

    # A provider score is evidence, not a classification. Nothing here may invent one.
    assert signal.risk_level is None


def test_signal_metadata_keeps_the_aggregate_figures_only(tmp_path, fake_nvidia):
    signal, _ = run_detection(tmp_path)

    # The clip table and NVIDIA's CSV rendering of it are timeline evidence and grow
    # with the video's length; neither is dumped into this row.
    assert signal.signal_metadata == {"logit": NVIDIA_LOGIT, "total_clips": 7}


def test_clip_evidence_becomes_segments(tmp_path, fake_nvidia):
    fake_nvidia.clips = (
        nvidia_video.NvidiaClipResult(index=0, logit=-2.25),
        nvidia_video.NvidiaClipResult(index=8, logit=3.5),
    )

    _, segments = run_detection(tmp_path)

    # Strongest first, exactly the figures NVIDIA reported for those clips.
    assert [(s.clip_index, s.logit) for s in segments] == [(8, 3.5), (0, -2.25)]


def test_segments_are_capped(tmp_path, fake_nvidia):
    # A long video scores far more clips than may be kept.
    fake_nvidia.clips = tuple(
        nvidia_video.NvidiaClipResult(index=index, logit=float(index))
        for index in range(MAX_PERSISTED_SEGMENTS + 50)
    )

    _, segments = run_detection(tmp_path)

    assert len(segments) == MAX_PERSISTED_SEGMENTS
    # The cap keeps the strongest evidence and drops the weakest.
    assert segments[0].logit == float(MAX_PERSISTED_SEGMENTS + 49)
    assert min(s.logit for s in segments) == float(50)


def test_the_full_clip_count_is_still_reported_when_segments_were_capped(tmp_path, fake_nvidia):
    fake_nvidia.clips = tuple(
        nvidia_video.NvidiaClipResult(index=index, logit=float(index))
        for index in range(MAX_PERSISTED_SEGMENTS + 50)
    )

    signal, _ = run_detection(tmp_path)

    # `total_clips` is the provider's own count, so a truncated set of rows can never be
    # mistaken for the whole of what NVIDIA scored.
    assert signal.signal_metadata["total_clips"] == 7


def test_segment_logits_are_untransformed(tmp_path, fake_nvidia):
    fake_nvidia.clips = (nvidia_video.NvidiaClipResult(index=3, logit=-1.4362945556640625),)

    _, segments = run_detection(tmp_path)

    # A raw model logit, not rescaled into a probability and not rounded.
    assert segments[0].logit == -1.4362945556640625
    assert segments[0].clip_index == 3


def test_ties_are_broken_by_clip_index_so_the_selection_is_deterministic(tmp_path, fake_nvidia):
    fake_nvidia.clips = (
        nvidia_video.NvidiaClipResult(index=9, logit=1.5),
        nvidia_video.NvidiaClipResult(index=2, logit=1.5),
        nvidia_video.NvidiaClipResult(index=5, logit=1.5),
    )

    _, segments = run_detection(tmp_path)

    assert [s.clip_index for s in segments] == [2, 5, 9]


def test_a_detection_reporting_no_clips_yields_no_segments(tmp_path, fake_nvidia):
    fake_nvidia.clips = ()

    signal, segments = run_detection(tmp_path)

    # The signal still stands on its own aggregate figures.
    assert signal.status == "SUCCESS"
    assert segments == []


def test_a_detector_timeout_becomes_a_signal_of_its_own(tmp_path, fake_nvidia):
    fake_nvidia.error = nvidia_video.NvidiaProviderTimeout("deadline exceeded")

    signal, segments = run_detection(tmp_path)

    assert signal.status == "TIMEOUT"
    # NVIDIA may still have been working; recording 0.0 would be inventing an answer.
    assert signal.score is None
    assert signal.risk_level is None
    assert segments == []


@pytest.mark.parametrize(
    "error",
    [
        nvidia_video.NvidiaAuthenticationError("rejected"),
        nvidia_video.NvidiaProviderUnavailable("NVIDIA is unavailable"),
        nvidia_video.NvidiaProviderInvalidResponse("no final result"),
    ],
)
def test_every_known_provider_failure_is_recorded_rather_than_raised(
    tmp_path, fake_nvidia, error
):
    fake_nvidia.error = error

    signal, segments = run_detection(tmp_path)

    assert signal.status == "FAILED"
    assert signal.score is None
    assert signal.provider_version is None
    # A detector that did not answer produced no clip evidence, and none is invented.
    assert segments == []


def test_a_failed_signal_records_the_failure_kind_and_nothing_more(tmp_path, fake_nvidia):
    fake_nvidia.error = nvidia_video.NvidiaAuthenticationError(
        "NVIDIA rejected the request (UNAUTHENTICATED): bad key abc123"
    )

    signal, _ = run_detection(tmp_path)

    # The provider's own message can quote request detail, so only the class name of the
    # failure is persisted. The detail stays in the server log.
    assert signal.signal_metadata == {"error": "NvidiaAuthenticationError"}


def test_an_unreadable_artifact_is_not_recorded_as_a_provider_failure(tmp_path, fake_nvidia):
    fake_nvidia.error = nvidia_video.NvidiaLocalFileError("cannot read the staged file")

    # Losing an artifact this server stored is a fault here, not at NVIDIA. Writing a
    # FAILED signal would blame the provider for our defect and bury a broken machine
    # among routine evidence, so it propagates instead.
    with pytest.raises(nvidia_video.NvidiaLocalFileError):
        run_detection(tmp_path)


# Provenance. A different evidence source, a different artifact, and — unlike a detector
# — one that can legitimately answer "there is nothing here".


UNSIGNED_MEDIA = Path(__file__).parent / "fixtures" / "still.mp4"


def evidence(**overrides) -> C2paEvidence:
    fields = {
        "manifest_exists": True,
        "sdk_version": "0.90.14",
        "validation_state": "Trusted",
        "validation_failures": (),
        "is_embedded": True,
        "active_manifest_label": "urn:c2pa:6f0e1a2b",
        "claim_generator": "test-camera",
        "signature_issuer": "Test Signing Cert",
        "signature_time": "2026-08-21T09:15:00Z",
        "assertion_labels": ("c2pa.actions.v2",),
        "manifest_json": '{"active_manifest": "urn:c2pa:6f0e1a2b"}',
    }
    fields.update(overrides)

    return C2paEvidence(**fields)


def fake_extraction(monkeypatch, result):
    """Answer the next extraction with `result`, or raise it if it is an exception."""

    def extract(_file_path):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(detection, "extract_c2pa_evidence", extract)


def test_provenance_becomes_a_c2pa_signal_of_its_own(monkeypatch, tmp_path):
    fake_extraction(monkeypatch, evidence())

    signal = extract_provenance(tmp_path / "original.mp4")

    assert signal.provider == "c2pa"
    assert signal.signal_type == "provenance"
    assert signal.status == "SUCCESS"
    assert signal.provider_version == "0.90.14"


def test_a_provenance_signal_carries_no_score_and_no_risk_level(monkeypatch, tmp_path):
    fake_extraction(monkeypatch, evidence())

    signal = extract_provenance(tmp_path / "original.mp4")

    # There is no number here. One would end up beside NVIDIA's probability as though a
    # signature and a model output were figures of the same kind.
    assert signal.score is None
    assert signal.risk_level is None


def test_the_signal_metadata_reports_the_extractor_facts(monkeypatch, tmp_path):
    fake_extraction(monkeypatch, evidence(validation_failures=("timeStamp.untrusted",)))

    signal = extract_provenance(tmp_path / "original.mp4")

    assert signal.signal_metadata["manifest_exists"] is True
    assert signal.signal_metadata["validation_state"] == "Trusted"
    assert signal.signal_metadata["validation_failures"] == ["timeStamp.untrusted"]
    assert signal.signal_metadata["claim_generator"] == "test-camera"
    assert signal.signal_metadata["signature_issuer"] == "Test Signing Cert"
    assert signal.signal_metadata["signature_time"] == "2026-08-21T09:15:00Z"
    assert signal.signal_metadata["assertion_labels"] == ["c2pa.actions.v2"]


def test_the_verbatim_manifest_is_not_dumped_into_the_signal(monkeypatch, tmp_path):
    fake_extraction(monkeypatch, evidence())

    signal = extract_provenance(tmp_path / "original.mp4")

    # Unbounded provider output does not belong in a column that holds a small supporting
    # document — the same reason the clip table travels as segment rows.
    assert "manifest_json" not in signal.signal_metadata


def test_media_with_no_credentials_reads_successfully():
    """Against the real reader and real media, with nothing stood in for."""
    signal = extract_provenance(UNSIGNED_MEDIA)

    assert signal.status == "SUCCESS"
    assert signal.signal_metadata["manifest_exists"] is False
    assert signal.signal_metadata["validation_state"] is None
    # Absent credentials say nothing about the media. This is a missing signal, not a
    # finding, and every key is still present so a reader never has to guess which.
    assert signal.signal_metadata["validation_failures"] == []
    assert signal.signal_metadata["assertion_labels"] == []


def test_a_broken_reading_becomes_a_failed_signal_rather_than_an_exception(
    monkeypatch, tmp_path
):
    fake_extraction(monkeypatch, RuntimeError("native library gave up on /tmp/deepguard-job-x"))

    signal = extract_provenance(tmp_path / "original.mp4")

    # One evidence source breaking must not cost the analysis the others, so this never
    # propagates. The failure kind is persisted; the message quotes a local path.
    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "RuntimeError"}


def test_an_unreadable_artifact_is_recorded_rather_than_raised(tmp_path):
    """Deliberately unlike the detector, which propagates the same condition.

    Losing the artifact is still a fault on this side either way. The difference is what
    it costs: NVIDIA is the work the job exists to do, while provenance is one source
    among several, and taking the whole analysis down over it would trade a known gap for
    no evidence at all. The job's own failure path still catches a fetch that never
    produced a file.
    """
    signal = extract_provenance(tmp_path / "absent.mp4")

    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "C2paLocalFileError"}
