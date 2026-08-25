"""The local audio-authenticity evidence, and the WAV the audio sources share.

Two things are under test and they are deliberately in one file, because the second is the
reason the first looks the way it does.

`detect_audio_authenticity` turns what the AASIST checkpoint emitted into a signal and its
window rows. `analyse_audio` prepares one WAV, hands it to the active-speaker chain and to
that checkpoint, and keeps their outcomes apart. The checkpoint itself is stubbed here — it
has its own tests in `test_audio_detector.py`, and loading a real ONNX graph to check how a
dataclass is copied into rows would test onnxruntime rather than this module.

Nothing here opens a transaction or writes a row.
"""

import asyncio
from pathlib import Path

import pytest

from app import detection, speaker_diarization
from app.audio_detector import (
    AudioAuthenticityEvidence,
    AudioDetectorAudioError,
    AudioDetectorInferenceError,
    AudioDetectorModelUnavailable,
    WindowEvidence,
)
from app.detection import (
    AUDIO_AUTHENTICITY_SIGNAL,
    MAX_PERSISTED_AUDIO_WINDOWS,
    analyse_audio,
    detect_audio_authenticity,
)
from app.nvidia_active_speaker import (
    NvidiaActiveSpeakerFrame,
    NvidiaActiveSpeakerResult,
    NvidiaBoundingBox,
    NvidiaSpeakerObservation,
)
from app.speaker_diarization import SpeakerTurn

VIDEO = Path("/tmp/deepguard-normalized-abc.mp4")
AUDIO = Path("/tmp/deepguard-diarization-abc.wav")
FRAME_RATE = 25.0

# The pinned artifact, restated here rather than imported, so a checkpoint swapped without
# anyone noticing shows up as a failing test instead of a silently different measurement.
MODEL_REPOSITORY = "SpeechAntiSpoofingBenchmarks/AASIST"
MODEL_REVISION = "16774d458d86d2a021ae31646c1bf66a5331b53e"
MODEL_SHA256 = "130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1"

# 64600 samples at 16 kHz. Both are the model's, not this codebase's choice.
WINDOW_SAMPLES = 64600
SAMPLE_RATE = 16000

ASD_FUNCTION_ID = "f286f937-05c4-454b-8312-fba67a2a6fa7"


def window(index, logits, *, samples=WINDOW_SAMPLES):
    """One window of raw output, over `samples` of real audio at the usual stride."""
    start = index * WINDOW_SAMPLES

    return WindowEvidence(
        window_index=index,
        start_sample=start,
        end_sample=start + samples,
        padded_samples=WINDOW_SAMPLES - samples,
        logits=tuple(logits),
        bona_fide_logit=logits[1],
    )


def evidence(*windows, total_samples=None):
    """What the checkpoint returns for one recording."""
    return AudioAuthenticityEvidence(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        model_sha256=MODEL_SHA256,
        sample_rate=SAMPLE_RATE,
        channels=1,
        window_samples=WINDOW_SAMPLES,
        window_padding_scheme="repeat-tile",
        total_samples=(
            total_samples if total_samples is not None else len(windows) * WINDOW_SAMPLES
        ),
        windows=tuple(windows),
    )


@pytest.fixture(autouse=True)
def fake_aasist(monkeypatch):
    """Stand in for the checkpoint, so no test here loads onnxruntime or a model file."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.audio_paths = []
            self.evidence = evidence(
                window(0, (-1.5, 4.25)),
                window(1, (2.0, -0.5)),
            )

        def analyze(self, audio_path, **kwargs):
            self.audio_paths.append(Path(audio_path))
            if self.error:
                raise self.error
            return self.evidence

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_audio_authenticity", recorder.analyze)
    return recorder


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch, tmp_path):
    """Stand in for ffmpeg, handing back a real path without decoding anything."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.sources = []

        async def extract(self, source, destination):
            self.sources.append(Path(source))
            if self.error:
                raise self.error
            Path(destination).write_bytes(b"RIFF....WAVE")

    recorder = Recorder()
    monkeypatch.setattr(speaker_diarization, "_extract_audio", recorder.extract)
    return recorder


@pytest.fixture(autouse=True)
def fake_diarization(monkeypatch):
    """Stand in for pyannote, so no test here loads torch or reaches Hugging Face."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.audio_paths = []

        async def diarize(self, audio_path, **kwargs):
            self.audio_paths.append(Path(audio_path))
            if self.error:
                raise self.error
            return (SpeakerTurn(start_time=0.0, end_time=2.0, speaker_id="SPEAKER_00"),)

    recorder = Recorder()
    monkeypatch.setattr(detection, "diarize_speakers", recorder.diarize)
    return recorder


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """Stand in for the ASD NIM, so no test in this module can reach NVIDIA."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.calls = []
            self.result = NvidiaActiveSpeakerResult(
                frames=tuple(
                    NvidiaActiveSpeakerFrame(
                        frame_id=frame_id,
                        speakers=(
                            NvidiaSpeakerObservation(
                                face_id=0,
                                diarized_speaker_id=0,
                                is_speaking=True,
                                face_detection_confidence=0.99,
                                bounding_box=NvidiaBoundingBox(
                                    x=1.0, y=2.0, width=32.0, height=32.0
                                ),
                            ),
                        ),
                    )
                    for frame_id in range(25)
                ),
                function_id=ASD_FUNCTION_ID,
                speaker_detection_threshold=0.5,
            )

        async def analyze(self, video_path, diarization, *, audio_path=None, **kwargs):
            self.calls.append((Path(video_path), list(diarization), audio_path))
            if self.error:
                raise self.error
            return self.result

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_active_speaker", recorder.analyze)
    return recorder


def run_audio(video=VIDEO, frame_rate=FRAME_RATE):
    """The audio-authenticity half of what the orchestrator returns."""
    _speaker, audio = asyncio.run(analyse_audio(video, frame_rate))
    return audio


# --- the signal itself ---------------------------------------------------------------------


def test_a_successful_reading_is_an_independent_local_signal():
    signal, segments = detect_audio_authenticity(AUDIO)

    # Named for the checkpoint that produced it, not a company that was called: nothing
    # left this machine to obtain it.
    assert signal.provider == "aasist"
    assert signal.signal_type == AUDIO_AUTHENTICITY_SIGNAL
    assert signal.status == "SUCCESS"
    assert len(segments) == 2


def test_a_successful_reading_still_has_no_file_level_score():
    """The model publishes no softmax, threshold or class, so there is no figure to store."""
    signal, _ = detect_audio_authenticity(AUDIO)

    # A number here would be this codebase inventing one and parking it in the same column
    # as NVIDIA's probability, as though the two could be compared.
    assert signal.score is None
    assert signal.risk_level is None


def test_the_signal_names_the_exact_artifact_that_produced_it():
    signal, _ = detect_audio_authenticity(AUDIO)

    assert signal.provider_version == f"{MODEL_REPOSITORY}@{MODEL_REVISION}"
    assert signal.signal_metadata["model_repository"] == MODEL_REPOSITORY
    assert signal.signal_metadata["model_revision"] == MODEL_REVISION
    # The digest the image verified at build time. A different checkpoint is a different
    # measurement, and this is what lets a stored signal be checked back against one.
    assert signal.signal_metadata["model_sha256"] == MODEL_SHA256


def test_the_signal_records_the_input_contract_the_windows_were_produced_under():
    signal, _ = detect_audio_authenticity(AUDIO)
    metadata = signal.signal_metadata

    # The same audio cut at a different rate or window length is a different number, so the
    # preparation is recorded beside the output rather than assumed from the code.
    assert metadata["sample_rate"] == SAMPLE_RATE
    assert metadata["window_samples"] == WINDOW_SAMPLES
    assert metadata["channels"] == 1
    assert metadata["window_padding_scheme"] == "repeat-tile"
    assert metadata["total_samples"] == 2 * WINDOW_SAMPLES


def test_the_signal_says_which_logit_carries_the_documented_meaning():
    signal, _ = detect_audio_authenticity(AUDIO)

    # Upstream reads output column 1 as the bona fide score. That mapping is the
    # checkpoint's own fact about its model, so it is recorded rather than left implied by
    # a column name.
    assert signal.signal_metadata["bona_fide_logit_index"] == 1


def test_the_signal_says_the_window_times_are_preprocessing_bounds():
    signal, _ = detect_audio_authenticity(AUDIO)

    # Stated in the evidence itself, because a reader of the database would otherwise have
    # to assume: AASIST publishes no chunk-to-time mapping and reports no segments, so
    # these are the bounds of what was fed to the graph and never a temporal detection.
    assert signal.signal_metadata["window_bounds"] == "deepguard_preprocessing"


# --- window evidence -----------------------------------------------------------------------


def test_both_raw_logits_are_preserved(fake_aasist):
    fake_aasist.evidence = evidence(window(0, (-1.5, 4.25)))

    _, segments = detect_audio_authenticity(AUDIO)

    # In graph order, untouched. Keeping only one of the two would throw away half of what
    # the model actually said, and neither can be derived from the other.
    assert (segments[0].logit, segments[0].bona_fide_logit) == (-1.5, 4.25)


def test_window_bounds_are_the_samples_that_were_fed_to_the_model(fake_aasist):
    fake_aasist.evidence = evidence(window(0, (0.5, 0.5)), window(1, (0.5, 0.5)))

    _, segments = detect_audio_authenticity(AUDIO)

    # 64600 / 16000 is exactly 4.0375 s, which is arithmetic on numbers this codebase chose
    # — not a timestamp the provider published.
    assert [(s.start_time, s.end_time) for s in segments] == [
        (0.0, 4.0375),
        (4.0375, 8.075),
    ]


def test_a_final_short_window_ends_where_the_audio_does(fake_aasist):
    fake_aasist.evidence = evidence(window(0, (1.0, 1.0)), window(1, (1.0, 1.0), samples=8000))

    _, segments = detect_audio_authenticity(AUDIO)

    # The last window was tiled up to the model's fixed length, but the bound records the
    # real audio it saw: half a second, not another four.
    assert segments[1].end_time == pytest.approx(4.0375 + 0.5)


def test_each_window_keeps_its_place_in_the_sequence(fake_aasist):
    fake_aasist.evidence = evidence(*(window(index, (0.1, 0.2)) for index in range(4)))

    _, segments = detect_audio_authenticity(AUDIO)

    assert [s.clip_index for s in segments] == [0, 1, 2, 3]


def test_windows_are_persisted_in_the_order_they_were_cut(fake_aasist):
    fake_aasist.evidence = evidence(
        window(0, (0.0, -9.0)),
        window(1, (0.0, 9.0)),
        window(2, (0.0, 0.0)),
    )

    _, segments = detect_audio_authenticity(AUDIO)

    # Chronological, never sorted by logit: P6-T1 §7 measured consecutive windows of one
    # genuine recording crossing zero in both directions, and reordering them by magnitude
    # would present that disagreement as a ranking.
    assert [s.start_time for s in segments] == [0.0, 4.0375, 8.075]
    assert [s.bona_fide_logit for s in segments] == [-9.0, 9.0, 0.0]


def test_window_rows_invent_no_speaker_identity(fake_aasist):
    fake_aasist.evidence = evidence(window(0, (1.0, 2.0)))

    _, segments = detect_audio_authenticity(AUDIO)

    # There is no face and no voice identity in an anti-spoofing result.
    assert (segments[0].face_id, segments[0].speaker_label) == (None, None)


# --- the persistence cap ---------------------------------------------------------------------


def test_evidence_is_capped_and_the_total_is_reported(fake_aasist):
    total = MAX_PERSISTED_AUDIO_WINDOWS + 17
    fake_aasist.evidence = evidence(*(window(index, (0.1, 0.2)) for index in range(total)))

    signal, segments = detect_audio_authenticity(AUDIO)

    assert len(segments) == MAX_PERSISTED_AUDIO_WINDOWS
    assert signal.signal_metadata["total_audio_windows"] == total
    assert signal.signal_metadata["persisted_audio_windows"] == MAX_PERSISTED_AUDIO_WINDOWS
    assert signal.signal_metadata["windows_truncated"] is True


def test_evidence_under_the_cap_is_kept_whole(fake_aasist):
    fake_aasist.evidence = evidence(*(window(index, (0.1, 0.2)) for index in range(3)))

    signal, segments = detect_audio_authenticity(AUDIO)

    assert len(segments) == 3
    assert signal.signal_metadata["total_audio_windows"] == 3
    assert signal.signal_metadata["windows_truncated"] is False


def test_the_cap_keeps_the_chronological_prefix_not_the_largest_logits(fake_aasist):
    """The one selection rule the checkpoint gives no basis for."""
    windows = [window(index, (0.0, 0.0)) for index in range(MAX_PERSISTED_AUDIO_WINDOWS + 5)]
    # The largest logits by far, and they are all past the cap.
    windows[-1] = window(len(windows) - 1, (99.0, 99.0))
    windows[-2] = window(len(windows) - 2, (-99.0, -99.0))
    fake_aasist.evidence = evidence(*windows)

    _, segments = detect_audio_authenticity(AUDIO)

    # Ranking by logit would be this codebase inventing the operating point the model
    # deliberately does not ship, so the extremes are dropped like any other late window.
    assert [s.clip_index for s in segments] == list(range(MAX_PERSISTED_AUDIO_WINDOWS))
    assert 99.0 not in [s.logit for s in segments]


# --- failure -------------------------------------------------------------------------------


def test_a_missing_checkpoint_is_a_failed_signal(fake_aasist):
    fake_aasist.error = AudioDetectorModelUnavailable("checkpoint is missing")

    signal, segments = detect_audio_authenticity(AUDIO)

    assert signal.provider == "aasist"
    assert signal.signal_type == AUDIO_AUTHENTICITY_SIGNAL
    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "AudioDetectorModelUnavailable"}
    assert segments == []


def test_a_failed_reading_carries_no_figures(fake_aasist):
    fake_aasist.error = AudioDetectorInferenceError("onnxruntime broke")

    signal, _ = detect_audio_authenticity(AUDIO)

    assert signal.score is None
    assert signal.risk_level is None
    assert signal.provider_version is None


def test_a_failure_never_leaks_the_local_path(fake_aasist):
    fake_aasist.error = AudioDetectorAudioError(
        "Audio could not be decoded: /tmp/deepguard-diarization-secret.wav"
    )

    signal, _ = detect_audio_authenticity(AUDIO)

    assert signal.signal_metadata == {"error": "AudioDetectorAudioError"}


# --- one WAV, two independent readings -------------------------------------------------------


def test_the_audio_is_extracted_once_and_shared_by_every_reader(
    fake_audio, fake_diarization, fake_nvidia, fake_aasist
):
    """One decode, three consumers. Extracting per consumer would decode the same media
    three times to produce three identical files."""
    run_audio()

    assert len(fake_audio.sources) == 1
    # From the artifact NVIDIA is given, so every reading describes the same recording.
    assert fake_audio.sources == [VIDEO]

    _, _, nvidia_audio = fake_nvidia.calls[0]
    assert fake_diarization.audio_paths == [nvidia_audio]
    assert fake_aasist.audio_paths == [nvidia_audio]


def test_the_temporary_wav_is_removed_afterwards(fake_aasist):
    run_audio()

    assert not fake_aasist.audio_paths[0].exists()


def test_a_broken_checkpoint_does_not_cost_the_speaker_timeline(fake_aasist):
    fake_aasist.error = AudioDetectorModelUnavailable("checkpoint is missing")

    speaker, audio = asyncio.run(analyse_audio(VIDEO, FRAME_RATE))

    assert audio[0].status == "FAILED"
    # The two share a WAV and nothing else. A model this container never received says
    # nothing about who was speaking in the video.
    assert speaker[0].status == "SUCCESS"
    assert len(speaker[1]) == 1


def test_a_broken_diarizer_does_not_cost_the_audio_windows(fake_diarization):
    fake_diarization.error = speaker_diarization.SpeakerDiarizationUnavailable(
        "HUGGINGFACE_TOKEN is not configured"
    )

    speaker, audio = asyncio.run(analyse_audio(VIDEO, FRAME_RATE))

    # The isolation runs both ways: the local checkpoint needs no token and no network, and
    # is run regardless of what the other chain came back with.
    assert speaker[0].status == "FAILED"
    assert audio[0].status == "SUCCESS"
    assert len(audio[1]) == 2


def test_media_with_no_audio_fails_both_readings_and_nothing_else(fake_audio, fake_aasist):
    fake_audio.error = speaker_diarization.SpeakerDiarizationAudioError("no audio stream")

    speaker, audio = asyncio.run(analyse_audio(VIDEO, FRAME_RATE))

    # The one failure the two genuinely share, because both were waiting on the same file.
    # Each records it in its own right rather than one standing in for the other.
    assert speaker[0].signal_type == "active_speaker"
    assert speaker[0].status == "FAILED"
    assert speaker[0].signal_metadata == {"error": "SpeakerDiarizationAudioError"}
    assert audio[0].provider == "aasist"
    assert audio[0].signal_type == AUDIO_AUTHENTICITY_SIGNAL
    assert audio[0].status == "FAILED"
    assert audio[0].signal_metadata == {"error": "SpeakerDiarizationAudioError"}
    assert (speaker[1], audio[1]) == ([], [])
    # Nothing was inferred, so nothing was read either.
    assert fake_aasist.audio_paths == []
