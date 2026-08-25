"""Tests for the local AASIST audio evidence extractor.

None of these load the real 1.6 MB checkpoint or run onnxruntime. The model boundary is faked
by putting a stand-in `onnxruntime` in `sys.modules` and letting the real import inside
`_load_session` find it, the way `test_speaker_diarization.py` fakes `pyannote.audio`. The
audio boundary is real: every fixture here is a WAV written with the standard library, because
what the header says is exactly what the module is supposed to check.

The checkpoint contract these fakes reproduce, verified against the pinned artifact
(`SpeechAntiSpoofingBenchmarks/AASIST` @ 16774d45, sha256 130e5362…):

    input   `wav`     ['batch', 64600]  float32
    output  `logits`  ['batch', 2]      float32

Two raw logits, no class, no threshold, no calibration. Upstream reads column 1 as the bona
fide score (`clovaai/aasist/main.py:307`), which is the only reason `bona_fide_logit` exists
and is asserted here as `logits[1]` — the mapping comes from the checkpoint's repository, not
from column order.

Real inference over the real checkpoint is verified out of band (P6-T2 live run), not here.
"""

import struct
import sys
import types
import wave
from pathlib import Path

import pytest

from app import audio_detector
from app.audio_detector import (
    AudioAuthenticityEvidence,
    AudioDetectorAudioError,
    AudioDetectorInferenceError,
    AudioDetectorInputError,
    AudioDetectorModelUnavailable,
    analyze_audio_authenticity,
)

WINDOW = audio_detector.EXPECTED_WINDOW_SAMPLES
MODEL_PATH = "/models/aasist.onnx"


def _write_wav(
    path: Path,
    *,
    frames: int,
    channels: int = 1,
    sample_rate: int = 16000,
    sample_width: int = 2,
) -> Path:
    """Write a real WAV with the given header, filled with a deterministic ramp."""
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        payload = b"".join(
            struct.pack("<h", (index % 1000) - 500) * channels for index in range(frames)
        )
        target.writeframes(payload)

    return path


class _FakeInput:
    def __init__(self, name="wav", shape=("batch", WINDOW)):
        self.name = name
        self.shape = list(shape)


class _FakeSession:
    """A stand-in for the real graph: records what it was fed, answers with fixed logits."""

    def __init__(
        self,
        *,
        logits_sequence=None,
        inputs=None,
        run_error=None,
        output=None,
    ):
        self._logits_sequence = logits_sequence
        self._inputs = inputs if inputs is not None else [_FakeInput()]
        self._run_error = run_error
        self._output = output
        self.calls = []

    def get_inputs(self):
        return self._inputs

    def run(self, output_names, feeds):
        import numpy as np

        self.calls.append((tuple(output_names), feeds["wav"].copy()))

        if self._run_error is not None:
            raise self._run_error

        if self._output is not None:
            return [self._output]

        index = len(self.calls) - 1
        if self._logits_sequence is None:
            values = [-3.0082068, 1.4435526]
        else:
            values = self._logits_sequence[index % len(self._logits_sequence)]

        return [np.asarray([values], dtype=np.float32)]


def _install_onnxruntime(monkeypatch, session=None, *, session_error=None):
    """Put a fake `onnxruntime` on `sys.modules` and report the session it will hand out."""
    session = session if session is not None else _FakeSession()
    module = types.ModuleType("onnxruntime")

    def InferenceSession(path, *args, **kwargs):  # noqa: N802 — mirrors the real name
        if session_error is not None:
            raise session_error
        return session

    module.InferenceSession = InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    return session


def _analyze(audio_path: Path, **kwargs) -> AudioAuthenticityEvidence:
    return analyze_audio_authenticity(audio_path, model_path=MODEL_PATH, **kwargs)


@pytest.fixture(autouse=True)
def _model_file_exists(monkeypatch):
    """`_load_session` checks the artifact is on disk; most tests want that check to pass."""
    monkeypatch.setattr(Path, "is_file", lambda self: True)


# --- Valid inference ------------------------------------------------------------------------


def test_single_window_reports_the_raw_logits(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(monkeypatch, _FakeSession(logits_sequence=[[-3.5, 2.25]]))

    evidence = _analyze(audio)

    assert len(evidence.windows) == 1
    window = evidence.windows[0]
    assert window.window_index == 0
    assert (window.start_sample, window.end_sample) == (0, WINDOW)
    assert window.padded_samples == 0
    assert window.logits == pytest.approx((-3.5, 2.25))
    # Column 1 is bona fide by the checkpoint's own repository, not by position.
    assert window.bona_fide_logit == pytest.approx(2.25)
    assert window.bona_fide_logit == pytest.approx(window.logits[1])


def test_evidence_carries_the_pinned_model_provenance(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    assert evidence.model_repository == "SpeechAntiSpoofingBenchmarks/AASIST"
    assert evidence.model_revision == "16774d458d86d2a021ae31646c1bf66a5331b53e"
    assert evidence.model_sha256 == (
        "130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1"
    )
    assert evidence.sample_rate == 16000
    assert evidence.channels == 1
    assert evidence.window_samples == WINDOW
    assert evidence.total_samples == WINDOW


def test_no_file_level_verdict_is_invented(tmp_path, monkeypatch):
    """The result must expose evidence only — no score, probability, class or risk."""
    audio = _write_wav(tmp_path / "clip.wav", frames=3 * WINDOW)
    _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    forbidden = {
        "score",
        "probability",
        "confidence",
        "verdict",
        "label",
        "classification",
        "is_synthetic",
        "is_fake",
        "risk",
        "risk_level",
        "segments",
    }
    assert forbidden.isdisjoint(vars(evidence))
    assert forbidden.isdisjoint(vars(evidence.windows[0]))


def test_the_result_is_frozen(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    assert isinstance(evidence.windows, tuple)
    with pytest.raises(Exception):
        evidence.windows[0].bona_fide_logit = 0.0


def test_samples_are_scaled_to_the_libsndfile_convention(tmp_path, monkeypatch):
    """PCM16 is divided by 32768, which is what upstream's reader does."""
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    session = _install_onnxruntime(monkeypatch)

    _analyze(audio)

    fed = session.calls[0][1]
    assert fed.shape == (1, WINDOW)
    assert fed.dtype.name == "float32"
    # The ramp's first sample is -500 as PCM16.
    assert fed[0][0] == pytest.approx(-500 / 32768.0)
    assert abs(fed).max() < 1.0


# --- Multiple windows -----------------------------------------------------------------------


def test_long_audio_produces_independent_chronological_windows(tmp_path, monkeypatch):
    """Three whole windows: three records, in order, none a proxy for the others."""
    audio = _write_wav(tmp_path / "long.wav", frames=3 * WINDOW)
    session = _install_onnxruntime(
        monkeypatch,
        _FakeSession(logits_sequence=[[-1.0, 1.0], [2.0, -2.0], [-0.5, 0.25]]),
    )

    evidence = _analyze(audio)

    assert len(evidence.windows) == 3
    assert [window.window_index for window in evidence.windows] == [0, 1, 2]
    assert [
        (window.start_sample, window.end_sample) for window in evidence.windows
    ] == [(0, WINDOW), (WINDOW, 2 * WINDOW), (2 * WINDOW, 3 * WINDOW)]
    assert [window.bona_fide_logit for window in evidence.windows] == pytest.approx(
        [1.0, -2.0, 0.25]
    )
    # Windows disagree here on purpose (P6-T1 §7). Nothing collapses them.
    assert all(window.padded_samples == 0 for window in evidence.windows)
    assert len(session.calls) == 3


def test_each_window_is_fed_its_own_slice_in_order(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "long.wav", frames=2 * WINDOW)
    session = _install_onnxruntime(monkeypatch)

    _analyze(audio)

    first, second = session.calls[0][1][0], session.calls[1][1][0]
    # The ramp repeats every 1000 samples, and 64600 is not a multiple of it, so consecutive
    # windows genuinely differ.
    assert first[0] != pytest.approx(second[0])
    assert second[0] == pytest.approx(((WINDOW % 1000) - 500) / 32768.0)


# --- Final-window padding -------------------------------------------------------------------


def test_final_short_window_is_padded_and_recorded(tmp_path, monkeypatch):
    remainder = 1000
    audio = _write_wav(tmp_path / "ragged.wav", frames=WINDOW + remainder)
    session = _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    assert len(evidence.windows) == 2
    last = evidence.windows[1]
    assert (last.start_sample, last.end_sample) == (WINDOW, WINDOW + remainder)
    assert last.padded_samples == WINDOW - remainder
    assert evidence.window_padding_scheme == "repeat-tile"
    # The graph still gets a full window; the padding is DeepGuard's, and it is declared.
    assert session.calls[1][1].shape == (1, WINDOW)


def test_padding_tiles_the_remaining_audio_deterministically(tmp_path, monkeypatch):
    remainder = 100
    audio = _write_wav(tmp_path / "short.wav", frames=remainder)
    session = _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    assert len(evidence.windows) == 1
    assert evidence.windows[0].padded_samples == WINDOW - remainder

    fed = session.calls[0][1][0]
    # Tiled, not zero-filled: the window repeats the 100 real samples end to end.
    assert fed[:remainder] == pytest.approx(fed[remainder : 2 * remainder])
    assert fed[-1] != pytest.approx(0.0)


def test_audio_shorter_than_a_window_yields_exactly_one_record(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "tiny.wav", frames=WINDOW - 1)
    _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    assert len(evidence.windows) == 1
    assert evidence.windows[0].padded_samples == 1


def test_exact_multiple_leaves_no_padding(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "exact.wav", frames=2 * WINDOW)
    _install_onnxruntime(monkeypatch)

    evidence = _analyze(audio)

    assert len(evidence.windows) == 2
    assert [window.padded_samples for window in evidence.windows] == [0, 0]


# --- Invalid audio input --------------------------------------------------------------------


def test_wrong_sample_rate_is_refused_before_inference(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "44k.wav", frames=WINDOW, sample_rate=44100)
    session = _install_onnxruntime(monkeypatch)

    with pytest.raises(AudioDetectorInputError, match="16000 Hz"):
        _analyze(audio)

    # The graph validates nothing and would have scored it, so refusing early is the point.
    assert session.calls == []


def test_wrong_channel_count_is_refused(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "stereo.wav", frames=WINDOW, channels=2)
    session = _install_onnxruntime(monkeypatch)

    with pytest.raises(AudioDetectorInputError, match="1-channel"):
        _analyze(audio)

    assert session.calls == []


def test_wrong_sample_width_is_refused(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "8bit.wav", frames=WINDOW, sample_width=1)
    _install_onnxruntime(monkeypatch)

    with pytest.raises(AudioDetectorInputError, match="16-bit"):
        _analyze(audio)


def test_empty_audio_is_refused(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "silent.wav", frames=0)
    _install_onnxruntime(monkeypatch)

    with pytest.raises(AudioDetectorInputError, match="no frames"):
        _analyze(audio)


# --- Unreadable audio -----------------------------------------------------------------------


def test_non_wav_bytes_are_a_decode_failure(tmp_path, monkeypatch):
    audio = tmp_path / "not-audio.wav"
    audio.write_bytes(b"\x00\x01\x02 this is not a RIFF header at all")
    _install_onnxruntime(monkeypatch)

    with pytest.raises(AudioDetectorAudioError, match="could not be decoded"):
        _analyze(audio)


def test_missing_audio_file_is_a_decode_failure(tmp_path, monkeypatch):
    _install_onnxruntime(monkeypatch)

    # `is_file` is patched true for the checkpoint; opening the audio still fails.
    with pytest.raises(AudioDetectorAudioError):
        _analyze(tmp_path / "absent.wav")


def test_truncated_data_chunk_is_a_decode_failure(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "truncated.wav", frames=WINDOW)
    intact = audio.read_bytes()
    # Keep the 44-byte canonical header, which still promises WINDOW frames, and drop the data.
    audio.write_bytes(intact[:44])
    _install_onnxruntime(monkeypatch)

    with pytest.raises(AudioDetectorAudioError):
        _analyze(audio)


# --- Missing / wrong model ------------------------------------------------------------------


def test_missing_checkpoint_is_unavailable(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(monkeypatch)
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    with pytest.raises(AudioDetectorModelUnavailable, match="missing"):
        _analyze(audio)


def test_unreadable_checkpoint_is_unavailable(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(monkeypatch, session_error=RuntimeError("invalid protobuf"))

    with pytest.raises(AudioDetectorModelUnavailable, match="could not be loaded"):
        _analyze(audio)


def test_missing_onnxruntime_is_unavailable(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(AudioDetectorModelUnavailable, match="not installed"):
        _analyze(audio)


def test_graph_with_a_different_window_length_is_refused(tmp_path, monkeypatch):
    """A swapped checkpoint must fail loudly rather than be fed the wrong window."""
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    session = _install_onnxruntime(
        monkeypatch, _FakeSession(inputs=[_FakeInput(shape=("batch", 32000))])
    )

    with pytest.raises(AudioDetectorModelUnavailable, match="32000 samples"):
        _analyze(audio)

    assert session.calls == []


def test_graph_with_a_symbolic_window_length_is_refused(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(
        monkeypatch, _FakeSession(inputs=[_FakeInput(shape=("batch", "samples"))])
    )

    with pytest.raises(AudioDetectorModelUnavailable, match="fixed window length"):
        _analyze(audio)


def test_graph_with_an_unexpected_input_is_refused(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(
        monkeypatch, _FakeSession(inputs=[_FakeInput(name="audio_input")])
    )

    with pytest.raises(AudioDetectorModelUnavailable, match="audio_input"):
        _analyze(audio)


# --- Malformed model output -----------------------------------------------------------------


def test_output_with_the_wrong_width_is_an_inference_error(tmp_path, monkeypatch):
    import numpy as np

    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(
        monkeypatch, _FakeSession(output=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32))
    )

    with pytest.raises(AudioDetectorInferenceError, match=r"shape \(1, 2\)"):
        _analyze(audio)


def test_output_with_the_wrong_rank_is_an_inference_error(tmp_path, monkeypatch):
    import numpy as np

    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(
        monkeypatch, _FakeSession(output=np.asarray([1.0, 2.0], dtype=np.float32))
    )

    with pytest.raises(AudioDetectorInferenceError, match="shape"):
        _analyze(audio)


def test_non_finite_logits_are_an_inference_error(tmp_path, monkeypatch):
    import numpy as np

    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(
        monkeypatch, _FakeSession(output=np.asarray([[float("nan"), 1.0]], dtype=np.float32))
    )

    with pytest.raises(AudioDetectorInferenceError, match="non-finite"):
        _analyze(audio)


def test_empty_output_list_is_an_inference_error(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)

    class _EmptySession(_FakeSession):
        def run(self, output_names, feeds):
            self.calls.append((tuple(output_names), feeds["wav"]))
            return []

    _install_onnxruntime(monkeypatch, _EmptySession())

    with pytest.raises(AudioDetectorInferenceError, match="no output"):
        _analyze(audio)


# --- ONNX Runtime inference failure ---------------------------------------------------------


def test_runtime_failure_is_an_inference_error(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    _install_onnxruntime(
        monkeypatch, _FakeSession(run_error=RuntimeError("Non-zero status code returned"))
    )

    with pytest.raises(AudioDetectorInferenceError, match="Non-zero status code"):
        _analyze(audio)


def test_runtime_failure_on_a_later_window_fails_the_call(tmp_path, monkeypatch):
    """No partial evidence: a failure mid-file is a failed signal, not a shorter timeline."""
    audio = _write_wav(tmp_path / "long.wav", frames=3 * WINDOW)

    class _FailsOnSecond(_FakeSession):
        def run(self, output_names, feeds):
            import numpy as np

            self.calls.append((tuple(output_names), feeds["wav"]))
            if len(self.calls) == 2:
                raise RuntimeError("inference aborted")
            return [np.asarray([[-1.0, 1.0]], dtype=np.float32)]

    session = _install_onnxruntime(monkeypatch, _FailsOnSecond())

    with pytest.raises(AudioDetectorInferenceError, match="inference aborted"):
        _analyze(audio)

    assert len(session.calls) == 2


# --- Statelessness --------------------------------------------------------------------------


def test_the_module_holds_no_session_between_calls(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    builds = []

    module = types.ModuleType("onnxruntime")

    def InferenceSession(path, *args, **kwargs):  # noqa: N802
        builds.append(path)
        return _FakeSession()

    module.InferenceSession = InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", module)

    _analyze(audio)
    _analyze(audio)

    assert builds == [MODEL_PATH, MODEL_PATH]


def test_the_model_path_comes_from_the_environment_by_default(tmp_path, monkeypatch):
    audio = _write_wav(tmp_path / "clip.wav", frames=WINDOW)
    builds = []

    module = types.ModuleType("onnxruntime")

    def InferenceSession(path, *args, **kwargs):  # noqa: N802
        builds.append(path)
        return _FakeSession()

    module.InferenceSession = InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    monkeypatch.setenv("AASIST_MODEL_PATH", "/elsewhere/aasist.onnx")

    analyze_audio_authenticity(audio)

    assert builds == ["/elsewhere/aasist.onnx"]
