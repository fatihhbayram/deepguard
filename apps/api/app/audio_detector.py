"""Local audio anti-spoofing evidence — AASIST, run offline on CPU.

P6-T1 benchmarked two local countermeasures and selected `SpeechAntiSpoofingBenchmarks/AASIST`
(`aasist.onnx`, MIT, 1.6 MB), the official `clovaai/aasist` ASVspoof2019-LA checkpoint exported
to ONNX. See D028 and `docs/ai/reviews/P6_T1_AUDIO_DETECTOR_BENCHMARK.md`.

What this module produces is the model's raw output and nothing else. It is deliberately not a
detector in the product sense:

- The graph emits **two raw logits** per window. It publishes no `id2label`, no softmax, no
  calibration and no decision threshold, so there is no probability, no confidence and no
  class to report. Whoever reads this evidence downstream is reading numbers the model
  emitted, not a judgement it made (AGENTS.md rule 11).
- The model consumes **exactly 64600 samples** (4.0375 s at 16 kHz) and publishes no
  chunk-to-time mapping. Anything longer therefore has to be cut up, and the cutting is
  DeepGuard's construct. `WindowEvidence.start_sample`/`end_sample` are a record of what this
  module fed the graph — preprocessing metadata — and are *not* temporal detections of
  synthesis. P6-T1 §7 measured successive windows of the same genuine recording crossing zero;
  the per-window numbers disagree with each other and are reported as they are.
- Windows are not averaged, ranked, voted on, thresholded or reduced to a file-level number.
  A ten-minute recording produces 149 independent records in chronological order, and that is
  the whole answer.

Input preparation is entirely the caller's responsibility as far as the graph is concerned —
it validates nothing and will score 16 kHz speech, 44.1 kHz music and silence alike, returning
a plausible-looking number for each. So this module validates the contract itself, before
inference, rather than trusting ONNX Runtime to refuse anything.

Stateless by design: no cached session, no module-level model, no shared state between calls.
The session is built and dropped per call (~0.33 s on the benchmark host). Inference is
blocking CPU work; an async caller should push `analyze_audio_authenticity` through
`asyncio.to_thread`, the way `speaker_diarization` does with pyannote.

Deliberately shares no abstraction with `nvidia_video.py`, `nvidia_active_speaker.py`,
`c2pa_extractor.py` or `speaker_diarization.py`, and must not grow one (AGENTS.md, abstraction
rule). This evidence is also never fused with theirs.
"""

import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- Model artifact -------------------------------------------------------------------------
#
# Pinned to one Hugging Face repository revision, not to `main`. The Dockerfile downloads this
# exact URL at build time and refuses the image if the bytes do not hash to MODEL_SHA256, so
# the checkpoint is fixed at build time and nothing is fetched at runtime.
MODEL_REPOSITORY = "SpeechAntiSpoofingBenchmarks/AASIST"
MODEL_REVISION = "16774d458d86d2a021ae31646c1bf66a5331b53e"
MODEL_FILENAME = "aasist.onnx"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{MODEL_FILENAME}"
)
MODEL_SHA256 = "130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1"
MODEL_SIZE_BYTES = 1_615_195

# Where the image puts it. Overridable so a developer can point at a checkout without editing
# code; it is configuration, never anything a request can influence.
DEFAULT_MODEL_PATH = Path("/models/aasist.onnx")
MODEL_PATH_ENV = "AASIST_MODEL_PATH"

# --- Graph contract -------------------------------------------------------------------------
#
# Verified against the pinned artifact: input `wav` of shape ['batch', 64600] float32, output
# `logits` of shape ['batch', 2] float32. The batch dimension is dynamic; the window length is
# static in the graph and is re-read from the loaded session rather than assumed — see
# `_window_samples`.
MODEL_INPUT_NAME = "wav"
MODEL_OUTPUT_NAME = "logits"
MODEL_OUTPUT_WIDTH = 2

# 64600 samples at 16 kHz is 4.0375 s. Stated here as the value this codebase expects; the
# artifact is checked against it at load time so a swapped checkpoint fails loudly rather than
# being silently fed the wrong window.
EXPECTED_WINDOW_SAMPLES = 64600
MODEL_SAMPLE_RATE = 16000
MODEL_CHANNELS = 1
# Only uncompressed 16-bit PCM is decoded here. That is exactly what
# `speaker_diarization.prepared_audio` produces, and refusing everything else keeps the
# float conversion below to one documented rule instead of a codec table.
MODEL_SAMPLE_WIDTH_BYTES = 2

# libsndfile — which upstream AASIST reads its audio through — maps PCM16 to floats by
# dividing by 32768. Matching that divisor is what makes numbers from this module comparable
# with the P6-T1 benchmark's.
PCM16_FULL_SCALE = 32768.0

# Upstream reads column 1 as the bona fide score (`batch_out[:, 1]` in
# `clovaai/aasist/main.py:307`) and column 0 as the spoof score. Higher column 1 = more bona
# fide. This mapping is established by the checkpoint's own repository, which is the only
# reason the value is surfaced by name at all — it is not inferred from column order, and it
# remains a raw logit, not a score with an operating point.
BONA_FIDE_LOGIT_INDEX = 1

# Windows are consecutive and non-overlapping, and the last one is filled by tiling the
# remaining audio, which is upstream's `data_utils.pad()` behaviour for short input. Tiling
# repeats real audio rather than inserting silence the model has never been trained on; either
# choice changes the number, so the choice is named here and the amount is recorded per window.
WINDOW_PADDING_SCHEME = "repeat-tile"


class AudioDetectorError(Exception):
    """Base class for every failure raised by this module."""


class AudioDetectorModelUnavailable(AudioDetectorError):
    """The checkpoint is missing, unreadable, or not the graph this module expects.

    A server/deployment fault, not a fact about the media: nothing was inferred, so nothing is
    known about the audio either way.
    """


class AudioDetectorInputError(AudioDetectorError):
    """The audio is readable but not in the shape the model requires.

    Wrong sample rate, wrong channel count, a compressed or non-16-bit WAV, or no frames at
    all. Kept separate from a decode failure because this is a caller/pipeline mistake with an
    exact remedy, and because the graph would have accepted the wrong audio silently.
    """


class AudioDetectorAudioError(AudioDetectorError):
    """The audio could not be opened or decoded at all."""


class AudioDetectorInferenceError(AudioDetectorError):
    """ONNX Runtime failed while inferring, or returned something this module cannot read.

    Covers both a raised runtime error and an output whose shape or dtype does not match the
    graph contract above — the second matters because a malformed output read positionally
    would turn into confident-looking evidence.
    """


@dataclass(frozen=True)
class WindowEvidence:
    """The model's raw output for one window, with the exact input that produced it.

    `window_index`, `start_sample` and `end_sample` describe **DeepGuard's** slicing of the
    recording, not a finding by the model. `end_sample` is exclusive and bounded by the real
    audio, so `end_sample - start_sample` is how much genuine material this window saw;
    `padded_samples` is how many further samples were tiled in to reach the model's fixed
    window length, and is zero for every window but possibly the last.

    `logits` is the pair the graph emitted, in graph order, unmodified. `bona_fide_logit` is
    `logits[1]` restated under the name the checkpoint's own repository gives it. It is a raw
    logit: not a probability, not a confidence, and not a threshold away from a verdict.
    """

    window_index: int
    start_sample: int
    end_sample: int
    padded_samples: int
    logits: tuple[float, ...]
    bona_fide_logit: float


@dataclass(frozen=True)
class AudioAuthenticityEvidence:
    """Every window of one recording, chronologically, plus what produced them.

    The provenance fields are here so a stored signal can be reproduced exactly: a different
    checkpoint revision or a different window length is a different measurement, and the same
    audio sliced at a different offset is a different number (P6-T1 §7).

    There is deliberately no file-level field of any kind — no score, no mean, no verdict, no
    risk level. `windows` is the evidence.
    """

    model_repository: str
    model_revision: str
    model_sha256: str
    sample_rate: int
    channels: int
    window_samples: int
    window_padding_scheme: str
    total_samples: int
    windows: tuple[WindowEvidence, ...]


def _resolve_model_path(model_path: Path | str | None) -> Path:
    if model_path is not None:
        return Path(model_path)

    configured = os.getenv(MODEL_PATH_ENV)
    return Path(configured) if configured else DEFAULT_MODEL_PATH


def _read_pcm16_mono(audio_path: Path) -> np.ndarray:
    """Decode a mono 16 kHz PCM16 WAV into floats in [-1, 1), validating the contract first.

    The standard library's `wave` module is enough here and deliberately so: the only audio
    this detector is ever handed is the WAV `speaker_diarization.prepared_audio` writes, and
    reading it with a parser that exposes the header fields directly is what makes the
    validation below explicit rather than a guess about what some decoder did.
    """
    try:
        with wave.open(str(audio_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()

            # Header checks come before reading the frames: a 44.1 kHz file is refused without
            # decoding it, and the graph never sees audio it would have scored anyway.
            if channels != MODEL_CHANNELS:
                raise AudioDetectorInputError(
                    f"AASIST requires {MODEL_CHANNELS}-channel audio, got {channels}"
                )
            if sample_rate != MODEL_SAMPLE_RATE:
                raise AudioDetectorInputError(
                    f"AASIST requires {MODEL_SAMPLE_RATE} Hz audio, got {sample_rate} Hz"
                )
            if sample_width != MODEL_SAMPLE_WIDTH_BYTES:
                raise AudioDetectorInputError(
                    f"AASIST input must be {MODEL_SAMPLE_WIDTH_BYTES * 8}-bit PCM, got "
                    f"{sample_width * 8}-bit"
                )
            if frame_count <= 0:
                raise AudioDetectorInputError("Audio contains no frames")

            frames = source.readframes(frame_count)
    except AudioDetectorInputError:
        raise
    except (wave.Error, EOFError, OSError, ValueError) as error:
        raise AudioDetectorAudioError(
            f"Audio could not be decoded ({type(error).__name__}: {error})"
        ) from error

    samples = np.frombuffer(frames, dtype="<i2")
    if samples.size == 0:
        # A header that promises frames over a truncated data chunk lands here.
        raise AudioDetectorAudioError("Audio decoded to no samples")

    return (samples.astype(np.float32) / PCM16_FULL_SCALE).astype(np.float32)


def _load_session(model_path: Path):
    """Build a fresh inference session over the pinned checkpoint.

    onnxruntime is imported here rather than at module scope so that importing this module's
    error types — which the API process does — costs nothing.
    """
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise AudioDetectorModelUnavailable("onnxruntime is not installed") from error

    if not model_path.is_file():
        raise AudioDetectorModelUnavailable(
            f"AASIST checkpoint is missing at '{model_path}'"
        )

    try:
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise AudioDetectorModelUnavailable(
            f"AASIST checkpoint at '{model_path}' could not be loaded "
            f"({type(error).__name__}: {error})"
        ) from error


def _window_samples(session) -> int:
    """Read the graph's fixed window length instead of trusting the constant.

    The batch dimension is symbolic and the sample dimension is not, so the second element of
    the declared input shape is an integer for this checkpoint. Anything else — a symbolic
    length, a missing input, a length that is not 64600 — means the artifact on disk is not the
    graph the rest of this module was written against, and feeding it a 64600-sample window
    would produce a number that looks fine and means nothing.
    """
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != MODEL_INPUT_NAME:
        raise AudioDetectorModelUnavailable(
            f"AASIST graph should take one input named '{MODEL_INPUT_NAME}', got "
            f"{[candidate.name for candidate in inputs]}"
        )

    shape = inputs[0].shape
    if len(shape) != 2 or not isinstance(shape[1], int):
        raise AudioDetectorModelUnavailable(
            f"AASIST graph should declare a fixed window length, got shape {shape}"
        )

    if shape[1] != EXPECTED_WINDOW_SAMPLES:
        raise AudioDetectorModelUnavailable(
            f"AASIST graph expects {shape[1]} samples per window, this build is written "
            f"against {EXPECTED_WINDOW_SAMPLES}"
        )

    return shape[1]


def _fill_window(samples: np.ndarray, start: int, window_samples: int) -> np.ndarray:
    """Cut one window at `start`, tiling the tail if the recording ends inside it."""
    window = samples[start : start + window_samples]
    if window.size == window_samples:
        return window

    repeats = window_samples // window.size + 1
    return np.tile(window, repeats)[:window_samples]


def _infer_window(session, window: np.ndarray) -> tuple[float, ...]:
    """Run one window and copy the logits out, refusing an output we cannot read."""
    batch = window.reshape(1, -1)

    try:
        outputs = session.run([MODEL_OUTPUT_NAME], {MODEL_INPUT_NAME: batch})
    except Exception as error:
        raise AudioDetectorInferenceError(
            f"AASIST inference failed ({type(error).__name__}: {error})"
        ) from error

    if not outputs:
        raise AudioDetectorInferenceError("AASIST returned no output tensor")

    logits = np.asarray(outputs[0])
    if logits.shape != (1, MODEL_OUTPUT_WIDTH):
        raise AudioDetectorInferenceError(
            f"AASIST output should have shape (1, {MODEL_OUTPUT_WIDTH}), got {logits.shape}"
        )

    values = tuple(float(value) for value in logits[0])
    if not all(np.isfinite(values)):
        raise AudioDetectorInferenceError(f"AASIST returned non-finite logits: {values}")

    return values


def analyze_audio_authenticity(
    audio_path: Path,
    *,
    model_path: Path | str | None = None,
) -> AudioAuthenticityEvidence:
    """Score every consecutive window of a prepared WAV and report the raw outputs.

    `audio_path` must be a mono 16 kHz 16-bit PCM WAV — the shape
    `speaker_diarization.prepared_audio` writes. It is only read; nothing is written beside it.

    The recording is cut into consecutive, non-overlapping windows of the model's exact length
    in chronological order, the last one tiled up to length if the audio ends inside it, and
    each is inferred independently. Nothing is aggregated: a file shorter than one window
    yields one record, a long file yields many, and no window is treated as a proxy for the
    others.

    Raises `AudioDetectorModelUnavailable` when the checkpoint is missing or is not the pinned
    graph, `AudioDetectorInputError` when the audio is not in the required format,
    `AudioDetectorAudioError` when it cannot be decoded, and `AudioDetectorInferenceError`
    when ONNX Runtime fails or answers with something unreadable.

    Blocking and CPU-bound — roughly 137 ms per window on the P6-T1 host. An async caller
    should run it in a thread.
    """
    resolved_model = _resolve_model_path(model_path)

    # Audio first: a malformed input is the caller's mistake and is worth reporting without
    # paying for a session build. The checkpoint is the more expensive of the two to load.
    samples = _read_pcm16_mono(Path(audio_path))

    session = _load_session(resolved_model)
    window_samples = _window_samples(session)

    total_samples = int(samples.size)
    windows: list[WindowEvidence] = []

    for window_index, start in enumerate(range(0, total_samples, window_samples)):
        end = min(start + window_samples, total_samples)
        window = _fill_window(samples, start, window_samples)
        logits = _infer_window(session, window)

        windows.append(
            WindowEvidence(
                window_index=window_index,
                start_sample=start,
                end_sample=end,
                padded_samples=window_samples - (end - start),
                logits=logits,
                bona_fide_logit=logits[BONA_FIDE_LOGIT_INDEX],
            )
        )

    logger.info(
        "AASIST scored %d window(s) over %d samples from %s",
        len(windows),
        total_samples,
        audio_path,
    )

    return AudioAuthenticityEvidence(
        model_repository=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        model_sha256=MODEL_SHA256,
        sample_rate=MODEL_SAMPLE_RATE,
        channels=MODEL_CHANNELS,
        window_samples=window_samples,
        window_padding_scheme=WINDOW_PADDING_SCHEME,
        total_samples=total_samples,
        windows=tuple(windows),
    )
