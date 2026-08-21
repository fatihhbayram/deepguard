"""Tests for the local speaker diarization module.

None of these load pyannote or run a real model, and none need a real ffmpeg binary. The
subprocess boundary is faked the way `test_normalization.py` fakes it, and the model boundary
is faked either at `_load_pipeline` or — where loading itself is what is under test — by
putting a stand-in `pyannote.audio` in `sys.modules` and letting the real import find it.

What is actually exercised here is this module's own behaviour: the argv it builds, the turns
it copies out, the errors it maps and the temp file it is responsible for removing.
"""

import asyncio
import builtins
import os
import sys
import types
from pathlib import Path

import pytest

from app import speaker_diarization as diarization
from app.speaker_diarization import (
    SpeakerDiarizationAudioError,
    SpeakerDiarizationModelError,
    SpeakerDiarizationTimeout,
    SpeakerDiarizationUnavailable,
    SpeakerTurn,
)

TOKEN = "hf_test_secret_token_value"
MEDIA = Path("/media/interview.mp4")


class FakeSegment:
    """Stands in for `pyannote.core.Segment`: only `start` and `end` are read."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeAnnotation:
    """Stands in for `pyannote.core.Annotation` at the one method this module calls."""

    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label, "The module must ask for labels; unlabelled turns say nothing."
        return iter(self._tracks)


class FakeDiarizeOutput:
    """Stands in for community-1's `DiarizeOutput` wrapper around the annotation."""

    def __init__(self, annotation):
        self.speaker_diarization = annotation
        self.speaker_embeddings = object()


class FakePipeline:
    """Records the path it was called with and replays a scripted result."""

    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.called_with = None

    def __call__(self, audio_path):
        self.called_with = audio_path
        if self._raises is not None:
            raise self._raises
        return self._result


class FakeProcess:
    """A stand-in child process: never exits on its own unless `hang` is False."""

    def __init__(self, *, stderr=b"", returncode=0, hang=False):
        self._stderr = stderr
        self._hang = hang
        self.returncode = None if hang else returncode
        self.killed = False
        self.waited = False

    async def communicate(self):
        if self._hang:
            await asyncio.Event().wait()
        return b"", self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        self.returncode = -9
        return self.returncode


def annotation(*tracks):
    """Build an annotation out of `(start, end, label)` triples.

    The middle element of each track is pyannote's track name, which this module ignores.
    """
    return FakeAnnotation([
        (FakeSegment(start, end), f"_{index}", label)
        for index, (start, end, label) in enumerate(tracks)
    ])


def diarize_output(*tracks):
    """Build a scripted community-1 result out of `(start, end, label)` triples."""
    return FakeDiarizeOutput(annotation(*tracks))


@pytest.fixture(autouse=True)
def configured_token(monkeypatch):
    """Every test but the credential ones assumes a configured token."""
    monkeypatch.setenv("HUGGINGFACE_TOKEN", TOKEN)


@pytest.fixture
def spawned(monkeypatch):
    """Capture the argv `_extract_audio` would execute and hand back a fake process."""
    calls = []

    def spawn(process):
        async def create_subprocess_exec(program, *args, **kwargs):
            calls.append((program, args))
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
        return calls

    return spawn


@pytest.fixture
def temp_files(monkeypatch):
    """Record every temp path this module creates, so cleanup can be asserted on."""
    created = []
    real_mkstemp = diarization.tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name))
        return descriptor, name

    monkeypatch.setattr(diarization.tempfile, "mkstemp", mkstemp)
    return created


@pytest.fixture
def loaded_pipeline(monkeypatch):
    """Replace the model boundary, capturing the model name and token it was asked for."""
    calls = []

    def load(pipeline):
        def _load_pipeline(model, token):
            calls.append((model, token))
            return pipeline

        monkeypatch.setattr(diarization, "_load_pipeline", _load_pipeline)
        return calls

    return load


@pytest.fixture
def fake_pyannote(monkeypatch):
    """Install a stand-in `pyannote.audio` so the real `_load_pipeline` can be exercised."""

    def install(pipeline_class):
        audio_module = types.ModuleType("pyannote.audio")
        audio_module.Pipeline = pipeline_class
        package = types.ModuleType("pyannote")
        package.audio = audio_module
        monkeypatch.setitem(sys.modules, "pyannote", package)
        monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)

    return install


# --- audio preparation -------------------------------------------------------------------


def test_audio_is_extracted_as_mono_16khz_pcm_wav(spawned, temp_files, loaded_pipeline):
    calls = spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output((0.0, 1.0, "SPEAKER_00"))))

    asyncio.run(diarization.diarize_speakers(MEDIA))

    program, args = calls[0]
    assert program == diarization.FFMPEG_BINARY
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-ar") + 1] == "16000"
    assert args[args.index("-c:a") + 1] == "pcm_s16le"
    assert args[args.index("-f") + 1] == "wav"
    # The video is never decoded, and the audio stream is required rather than optional:
    # `0:a:0?` would silently produce an empty WAV for a video with no audio in it.
    assert "-vn" in args
    assert args[args.index("-map") + 1] == "0:a:0"


def test_source_and_destination_are_separate_argv_entries(
    spawned, temp_files, loaded_pipeline
):
    """A filename can never be read as shell syntax, however it was named."""
    calls = spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output()))

    asyncio.run(diarization.diarize_speakers(Path("/media/two words; rm -rf.mp4")))

    _, args = calls[0]
    assert args[args.index("-i") + 1] == "/media/two words; rm -rf.mp4"
    assert args[-1] == str(temp_files[0])
    assert args[-1].endswith(".wav")


def test_extraction_writes_to_a_temp_file_not_beside_the_media(
    spawned, temp_files, loaded_pipeline
):
    """The destination is a temp file, never anything next to the caller's media."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output()))

    asyncio.run(diarization.diarize_speakers(MEDIA))

    assert temp_files[0].parent != MEDIA.parent
    assert temp_files[0].name.startswith(diarization.AUDIO_TEMP_PREFIX)


def test_the_prepared_wav_is_what_reaches_the_model(spawned, temp_files, loaded_pipeline):
    """The model sees the extracted WAV, never the original container."""
    spawned(FakeProcess())
    pipeline = FakePipeline(diarize_output())
    loaded_pipeline(pipeline)

    asyncio.run(diarization.diarize_speakers(MEDIA))

    assert pipeline.called_with == str(temp_files[0])


def test_real_media_is_extracted_to_a_mono_16khz_wav(tmp_path):
    """The one test that runs ffmpeg for real, on a generated tone.

    The faked-subprocess tests above assert the argv; this asserts that argv actually
    produces the format the model requires, which no amount of argv checking proves.
    """
    if not _ffmpeg_available():
        pytest.skip("ffmpeg is not installed")

    source = tmp_path / "tone.mp4"
    _generate_media(source, with_audio=True)
    destination = tmp_path / "prepared.wav"

    asyncio.run(diarization._extract_audio(source, destination))

    assert destination.exists()
    # A canonical PCM WAV header: 'RIFF' ... 'WAVE', mono at 16 kHz in the fmt chunk.
    header = destination.read_bytes()[:44]
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert int.from_bytes(header[22:24], "little") == diarization.AUDIO_CHANNELS
    assert int.from_bytes(header[24:28], "little") == diarization.AUDIO_SAMPLE_RATE


def test_real_media_without_an_audio_track_is_rejected(tmp_path):
    """The no-audio case, against the real ffmpeg rather than a scripted return code."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg is not installed")

    source = tmp_path / "silent.mp4"
    _generate_media(source, with_audio=False)

    with pytest.raises(SpeakerDiarizationAudioError):
        asyncio.run(diarization._extract_audio(source, tmp_path / "prepared.wav"))


def _ffmpeg_available() -> bool:
    import shutil

    return shutil.which(diarization.FFMPEG_BINARY) is not None


def _generate_media(destination: Path, *, with_audio: bool) -> None:
    """Synthesize a one-second MP4, with or without an AAC audio track.

    AAC on purpose: it is what `normalization.py` writes and what NVIDIA cannot take, so it
    is the case this module exists to absorb.
    """
    import subprocess

    args = [
        diarization.FFMPEG_BINARY, "-nostdin", "-y",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
    ]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(destination)]

    subprocess.run(args, check=True, capture_output=True)


# --- audio preparation failures ----------------------------------------------------------


def test_media_without_audio_is_a_media_failure(spawned, temp_files, loaded_pipeline):
    """ffmpeg refusing the input is a fact about the upload, not a server fault."""
    spawned(FakeProcess(returncode=1, stderr=b"Stream map '0:a:0' matches no streams."))
    loaded_pipeline(FakePipeline(diarize_output()))

    with pytest.raises(SpeakerDiarizationAudioError):
        asyncio.run(diarization.diarize_speakers(MEDIA))


def test_a_missing_ffmpeg_binary_is_unavailable(monkeypatch, temp_files):
    """No ffmpeg is a server problem: nothing is known about the media either way."""

    async def create_subprocess_exec(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)

    with pytest.raises(SpeakerDiarizationUnavailable):
        asyncio.run(diarization.diarize_speakers(MEDIA))


def test_slow_extraction_is_killed_reaped_and_reported(monkeypatch, spawned, temp_files):
    process = FakeProcess(hang=True)
    spawned(process)
    monkeypatch.setattr(diarization, "FFMPEG_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(SpeakerDiarizationTimeout):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert process.killed
    assert process.waited


def test_the_model_is_never_loaded_when_extraction_failed(
    spawned, temp_files, loaded_pipeline
):
    """Loading torch costs seconds; a file with no audio must not pay for it."""
    spawned(FakeProcess(returncode=1))
    calls = loaded_pipeline(FakePipeline(diarize_output()))

    with pytest.raises(SpeakerDiarizationAudioError):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert calls == []


# --- credentials and model loading -------------------------------------------------------


def test_a_missing_token_is_reported_before_anything_runs(monkeypatch, spawned, temp_files):
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    calls = spawned(FakeProcess())

    with pytest.raises(SpeakerDiarizationUnavailable):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert calls == []
    assert temp_files == []


def test_the_token_comes_from_configuration(spawned, temp_files, loaded_pipeline):
    spawned(FakeProcess())
    calls = loaded_pipeline(FakePipeline(diarize_output()))

    asyncio.run(diarization.diarize_speakers(MEDIA))

    assert calls == [(diarization.DIARIZATION_MODEL, TOKEN)]


def test_an_explicit_token_wins_over_the_environment(spawned, temp_files, loaded_pipeline):
    spawned(FakeProcess())
    calls = loaded_pipeline(FakePipeline(diarize_output()))

    asyncio.run(diarization.diarize_speakers(MEDIA, auth_token="hf_explicit"))

    assert calls == [(diarization.DIARIZATION_MODEL, "hf_explicit")]


def test_the_pinned_model_is_community_1():
    """The model identity is part of the evidence; it is not free to drift."""
    assert diarization.DIARIZATION_MODEL == "pyannote/speaker-diarization-community-1"


def test_the_pinned_model_is_what_is_fetched(spawned, temp_files, fake_pyannote):
    requested = []

    class Pipeline:
        @staticmethod
        def from_pretrained(model, token=None):
            requested.append((model, token))
            return FakePipeline(diarize_output())

    fake_pyannote(Pipeline)
    spawned(FakeProcess())

    asyncio.run(diarization.diarize_speakers(MEDIA))

    assert requested == [("pyannote/speaker-diarization-community-1", TOKEN)]


def test_an_inaccessible_model_is_unavailable(spawned, temp_files, fake_pyannote):
    """`from_pretrained` answers with None — not an exception — when gating blocks the token."""

    class Pipeline:
        @staticmethod
        def from_pretrained(model, token=None):
            return None

    fake_pyannote(Pipeline)
    spawned(FakeProcess())

    with pytest.raises(SpeakerDiarizationUnavailable, match="not accessible"):
        asyncio.run(diarization.diarize_speakers(MEDIA))


def test_a_failing_model_load_is_unavailable_without_leaking_the_token(
    spawned, temp_files, fake_pyannote
):
    class Pipeline:
        @staticmethod
        def from_pretrained(model, token=None):
            raise RuntimeError(f"401 Unauthorized for {model} with token {token}")

    fake_pyannote(Pipeline)
    spawned(FakeProcess())

    with pytest.raises(SpeakerDiarizationUnavailable) as raised:
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert TOKEN not in str(raised.value)
    assert "RuntimeError" in str(raised.value)


def test_a_missing_pyannote_install_is_unavailable(monkeypatch, spawned, temp_files):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise ImportError("No module named 'pyannote'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spawned(FakeProcess())

    with pytest.raises(SpeakerDiarizationUnavailable, match="not installed"):
        asyncio.run(diarization.diarize_speakers(MEDIA))


# --- parsing the model's output ----------------------------------------------------------


def test_turns_are_returned_exactly_as_reported(spawned, temp_files, loaded_pipeline):
    """Timestamps and labels are copied, not rounded, renumbered or renamed."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output(
        (0.4978125, 3.1246875, "SPEAKER_00"),
        (3.6084375, 7.9059375, "SPEAKER_01"),
        (8.2209375, 11.4215625, "SPEAKER_00"),
    )))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert result == (
        SpeakerTurn(start_time=0.4978125, end_time=3.1246875, speaker_id="SPEAKER_00"),
        SpeakerTurn(start_time=3.6084375, end_time=7.9059375, speaker_id="SPEAKER_01"),
        SpeakerTurn(start_time=8.2209375, end_time=11.4215625, speaker_id="SPEAKER_00"),
    )


def test_overlapping_turns_are_preserved(spawned, temp_files, loaded_pipeline):
    """Two voices at once is what a diarizer is for; the overlap is not resolved away."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output(
        (1.0, 4.0, "SPEAKER_00"),
        (3.5, 6.0, "SPEAKER_01"),
    )))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert len(result) == 2
    assert result[0].end_time > result[1].start_time


def test_adjacent_turns_of_one_speaker_are_not_merged(spawned, temp_files, loaded_pipeline):
    """Merging would be an interpretation; the model reported two turns, so two come back."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output(
        (1.0, 2.0, "SPEAKER_00"),
        (2.0, 3.0, "SPEAKER_00"),
    )))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert len(result) == 2


def test_very_short_turns_are_not_dropped(spawned, temp_files, loaded_pipeline):
    """A 40 ms turn is what the model reported, so it survives untouched."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output((1.0, 1.04, "SPEAKER_01"))))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert len(result) == 1


def test_non_default_speaker_labels_survive(spawned, temp_files, loaded_pipeline):
    """Labels are pyannote's strings, not indices parsed out of them."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output((0.0, 1.0, "Interviewer"))))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert result[0].speaker_id == "Interviewer"


def test_a_bare_annotation_is_accepted_too(spawned, temp_files, loaded_pipeline):
    """Older pyannote pipelines answer with the annotation rather than a wrapper."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(annotation((0.0, 1.5, "SPEAKER_00"))))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert result == (SpeakerTurn(start_time=0.0, end_time=1.5, speaker_id="SPEAKER_00"),)


def test_speech_free_audio_is_an_empty_result_not_a_failure(
    spawned, temp_files, loaded_pipeline
):
    """Silence is a real answer about the media, so it is not raised as an error."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output()))

    assert asyncio.run(diarization.diarize_speakers(MEDIA)) == ()


def test_turns_are_frozen(spawned, temp_files, loaded_pipeline):
    """Evidence does not get edited after the fact."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output((0.0, 1.0, "SPEAKER_00"))))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    with pytest.raises(Exception):
        result[0].start_time = 99.0


def test_inference_failure_is_a_model_error(spawned, temp_files, loaded_pipeline):
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(raises=RuntimeError("out of memory")))

    with pytest.raises(SpeakerDiarizationModelError):
        asyncio.run(diarization.diarize_speakers(MEDIA))


def test_turns_carry_what_the_nvidia_client_requires(spawned, temp_files, loaded_pipeline):
    """A turn holds everything `DiarizationSegment` needs, in the units it needs converting
    from: seconds to milliseconds, and label to index. Performing that conversion is P5-T3's
    job, not this module's — this asserts only that nothing is missing for it.
    """
    from app.nvidia_active_speaker import DiarizationSegment

    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output(
        (0.5, 3.125, "SPEAKER_00"),
        (3.6, 7.9, "SPEAKER_01"),
    )))

    result = asyncio.run(diarization.diarize_speakers(MEDIA))
    labels = sorted({turn.speaker_id for turn in result})
    segments = [
        DiarizationSegment(
            start_time_ms=int(turn.start_time * 1000),
            end_time_ms=int(turn.end_time * 1000),
            speaker_id=labels.index(turn.speaker_id),
        )
        for turn in result
    ]

    assert segments == [
        DiarizationSegment(start_time_ms=500, end_time_ms=3125, speaker_id=0),
        DiarizationSegment(start_time_ms=3600, end_time_ms=7900, speaker_id=1),
    ]


# --- temp file cleanup -------------------------------------------------------------------


def test_the_temp_wav_is_removed_after_a_successful_run(
    spawned, temp_files, loaded_pipeline
):
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output((0.0, 1.0, "SPEAKER_00"))))

    asyncio.run(diarization.diarize_speakers(MEDIA))

    assert len(temp_files) == 1
    assert not temp_files[0].exists()


def test_the_temp_wav_is_removed_after_an_extraction_failure(
    spawned, temp_files, loaded_pipeline
):
    spawned(FakeProcess(returncode=1))
    loaded_pipeline(FakePipeline(diarize_output()))

    with pytest.raises(SpeakerDiarizationAudioError):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert not temp_files[0].exists()


def test_the_temp_wav_is_removed_after_a_model_failure(spawned, temp_files, loaded_pipeline):
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(raises=RuntimeError("boom")))

    with pytest.raises(SpeakerDiarizationModelError):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert not temp_files[0].exists()


def test_the_temp_wav_is_removed_after_a_failed_model_load(
    spawned, temp_files, fake_pyannote
):
    class Pipeline:
        @staticmethod
        def from_pretrained(model, token=None):
            return None

    fake_pyannote(Pipeline)
    spawned(FakeProcess())

    with pytest.raises(SpeakerDiarizationUnavailable):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert not temp_files[0].exists()


def test_the_temp_wav_is_removed_after_a_timeout(monkeypatch, spawned, temp_files):
    spawned(FakeProcess(hang=True))
    monkeypatch.setattr(diarization, "FFMPEG_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(SpeakerDiarizationTimeout):
        asyncio.run(diarization.diarize_speakers(MEDIA))

    assert not temp_files[0].exists()


def test_the_temp_wav_is_removed_when_the_caller_cancels(monkeypatch, temp_files):
    """A cancelled job must not leak a WAV per cancelled analysis."""
    started = asyncio.Event()

    async def create_subprocess_exec(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)

    async def scenario():
        task = asyncio.ensure_future(diarization.diarize_speakers(MEDIA))
        await started.wait()
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    assert not temp_files[0].exists()


def test_a_failing_cleanup_does_not_mask_a_good_result(
    monkeypatch, spawned, temp_files, loaded_pipeline
):
    """Cleanup is best effort: a temp file that vanished must not fail the analysis."""
    spawned(FakeProcess())
    loaded_pipeline(FakePipeline(diarize_output((0.0, 1.0, "SPEAKER_00"))))

    real_unlink = os.unlink

    def unlink(path):
        real_unlink(path)
        raise OSError("already gone")

    monkeypatch.setattr(diarization.os, "unlink", unlink)

    result = asyncio.run(diarization.diarize_speakers(MEDIA))

    assert result == (SpeakerTurn(start_time=0.0, end_time=1.0, speaker_id="SPEAKER_00"),)
    assert not temp_files[0].exists()
