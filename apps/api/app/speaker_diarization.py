"""Local speaker diarization — who spoke when, answered on our own hardware.

NVIDIA's Active Speaker Detection NIM does not diarize. It takes diarization as an *input*
and matches the voices it is handed onto the faces it sees; handed nothing, it reports every
face with `diarized_speaker_id == -1`. So the speaker turns have to come from somewhere, and
this module is that somewhere: `pyannote/speaker-diarization-community-1`, running locally.

Two steps, and since P5-T3 they are two separate calls:

1. `prepared_audio` has FFmpeg extract a temporary mono 16 kHz PCM WAV. The model wants that
   shape, and going through FFmpeg means the codec inside the uploaded container stops
   mattering — AAC in an MP4, Opus in a WebM and a bare MP3 all arrive at the model as the
   same thing.
2. `diarize_speakers` runs pyannote over that WAV and returns its own speech turns.

They used to be one call. They were split because NVIDIA's ASD NIM wants the same audio as a
separate stream alongside the video, and extracting it twice would mean decoding the same
media twice to produce two identical files — so the caller prepares the WAV once and hands it
to both. The extraction still lives here rather than somewhere neutral because this is where
the shape it produces is specified, and moving it would be a larger change than the sharing
requires.

One consequence of the split is worth stating: the Hugging Face token is checked when
`diarize_speakers` is called, which is after the WAV exists. An unconfigured token therefore
costs one wasted extraction per job rather than being caught before any work. That is the
honest cost of the audio being shared — the WAV is owed to NVIDIA whether or not pyannote can
run — and it is wasted work, never a wrong answer.

What comes back is pyannote's own output, unchanged: its timestamps and its speaker labels
(rule 11). This module invents no confidence score — the pipeline reports none — merges no
adjacent turns, drops no short ones, and renames no speaker. `SPEAKER_00` is what pyannote
called that voice, and it stays that string; deciding what integer NVIDIA should see for it
is the caller's business, not a fact about the audio.

Deliberately separate from `nvidia_active_speaker.py` and sharing no abstraction with it.
That module is a gRPC client for a hosted provider; this one is a local model behind a local
transcode. They sit next to each other in one pipeline, which is not the same as having a
shape worth extracting.
"""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FFMPEG_BINARY = "ffmpeg"

# The pinned diarization model. It is gated on Hugging Face, so reaching it needs an account
# that has accepted its conditions plus the token below.
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# The shape the model expects: single channel, 16 kHz, uncompressed. Anything else is
# resampled inside the pipeline anyway, so doing it once here in FFmpeg keeps the input to the
# model exact and identical for every source container.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CODEC = "pcm_s16le"

# Extraction only demuxes and re-encodes audio, which runs far faster than real time — this
# bound exists so a pathological input cannot stall the caller forever, not to pace normal
# work. Inference itself is not bounded here; it is CPU-bound local work with no deadline to
# enforce, and killing it halfway would produce nothing useful.
FFMPEG_TIMEOUT_SECONDS = 300

AUDIO_TEMP_PREFIX = "deepguard-diarization-"
AUDIO_TEMP_SUFFIX = ".wav"


class SpeakerDiarizationError(Exception):
    """Base class for every failure raised by this module."""


class SpeakerDiarizationAudioError(SpeakerDiarizationError):
    """The media carries no audio this module can diarize.

    Raised when FFmpeg refuses the file — most often because there is no audio stream in it
    at all, which is a fact about the upload rather than a server fault.
    """


class SpeakerDiarizationTimeout(SpeakerDiarizationError):
    """Audio extraction outlived its limit and was stopped."""


class SpeakerDiarizationUnavailable(SpeakerDiarizationError):
    """The diarizer could not be run at all — a server problem, not a media problem.

    Covers a missing FFmpeg binary, missing credentials, and a model that cannot be fetched
    or loaded. Nothing was inferred, so nothing is known about the media either way.
    """


class SpeakerDiarizationModelError(SpeakerDiarizationError):
    """The model loaded and then failed while inferring over the prepared audio."""


@dataclass(frozen=True)
class SpeakerTurn:
    """One stretch of speech pyannote attributed to one voice.

    `start_time` and `end_time` are seconds from the start of the audio, as floats, which is
    what pyannote reports — they are not rounded or quantized here.

    `speaker_id` is pyannote's own label for the voice, typically `SPEAKER_00`, `SPEAKER_01`
    and so on. It is a label, not an index: it identifies the same voice across the turns of
    one file and means nothing across files. It is kept as the string pyannote produced
    rather than parsed into a number, because the numeric form NVIDIA's proto wants is a
    representation choice for whoever calls NVIDIA, not something this model asserted.

    Turns may overlap — two people talking at once is exactly what a diarizer is for — and
    overlapping turns are preserved rather than resolved.
    """

    start_time: float
    end_time: float
    speaker_id: str


def _load_token(auth_token: str | None) -> str:
    """Resolve the Hugging Face token, preferring an explicit argument.

    Configuration only, never anything a request can influence, and never logged or placed in
    an exception message.
    """
    resolved = auth_token or os.getenv("HUGGINGFACE_TOKEN")
    if not resolved:
        raise SpeakerDiarizationUnavailable("HUGGINGFACE_TOKEN is not configured")

    return resolved


def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        process.kill()
    except ProcessLookupError:
        # The child raced us to exit; the following wait() still reaps it.
        pass


async def _extract_audio(source: Path, destination: Path) -> None:
    """Demux the first audio stream into a mono 16 kHz PCM WAV, without a shell.

    Both paths are separate argv entries, so a filename can never be read as shell syntax.
    """
    args = (
        "-nostdin",
        "-y",
        "-i", str(source),
        # Video is irrelevant to diarization and decoding it would be the expensive part.
        "-vn",
        # The first audio stream, and it must exist: an input with no audio cannot be
        # diarized, and FFmpeg failing here is how that is discovered.
        "-map", "0:a:0",
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", AUDIO_CODEC,
        # Stated explicitly rather than inferred from the suffix, so the container is the
        # model's format regardless of what the temp file ended up being called.
        "-f", "wav",
        str(destination),
    )

    try:
        process = await asyncio.create_subprocess_exec(
            FFMPEG_BINARY,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise SpeakerDiarizationUnavailable("ffmpeg could not be executed") from error

    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        _terminate(process)
        await process.wait()
        raise SpeakerDiarizationTimeout(
            f"Audio extraction timed out after {FFMPEG_TIMEOUT_SECONDS}s"
        ) from None

    if process.returncode != 0:
        # stderr is diagnostic only: it can quote container internals and the temp path.
        logger.warning(
            "ffmpeg audio extraction exited with %s: %s",
            process.returncode,
            stderr.decode("utf-8", "replace").strip(),
        )
        raise SpeakerDiarizationAudioError(
            f"No diarizable audio could be extracted (ffmpeg exited with {process.returncode})"
        )


def _load_pipeline(model: str, token: str):
    """Fetch and instantiate the diarization pipeline.

    pyannote is imported here rather than at module scope on purpose. It drags in torch and
    the whole Lightning stack, which costs seconds and a large amount of memory, and the API
    process imports this module's error types without ever diarizing anything.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise SpeakerDiarizationUnavailable("pyannote.audio is not installed") from error

    try:
        pipeline = Pipeline.from_pretrained(model, token=token)
    except Exception as error:
        # The token can appear in the messages huggingface_hub raises, so only the model name
        # and the exception type are reported.
        raise SpeakerDiarizationUnavailable(
            f"Diarization model '{model}' could not be loaded ({type(error).__name__})"
        ) from error

    # `from_pretrained` returns None instead of raising when the checkpoint exists but the
    # token cannot see it — an un-accepted gating agreement lands here.
    if pipeline is None:
        raise SpeakerDiarizationUnavailable(
            f"Diarization model '{model}' is not accessible with the configured token"
        )

    return pipeline


def _run_pipeline(pipeline, audio_path: Path) -> tuple[SpeakerTurn, ...]:
    """Infer over the prepared WAV and copy the turns out of pyannote's own objects.

    Blocking and CPU-bound; the caller keeps it off the event loop.
    """
    try:
        output = pipeline(str(audio_path))
    except Exception as error:
        raise SpeakerDiarizationModelError(
            f"Diarization failed during inference ({type(error).__name__}: {error})"
        ) from error

    # community-1 answers with a `DiarizeOutput` carrying the annotation alongside speaker
    # embeddings; pyannote's older pipelines answer with the bare annotation.
    annotation = getattr(output, "speaker_diarization", output)

    # Turns are yielded in pyannote's own order, which is chronological, and that order is
    # kept. An annotation with no tracks means the model heard no speech, which is a real
    # answer about the audio and comes back as an empty result rather than an error.
    return tuple(
        SpeakerTurn(
            start_time=segment.start,
            end_time=segment.end,
            speaker_id=str(label),
        )
        for segment, _track, label in annotation.itertracks(yield_label=True)
    )


@asynccontextmanager
async def prepared_audio(media_path: Path) -> AsyncIterator[Path]:
    """Extract the analysable audio once, for the block, and remove it afterwards.

    `media_path` may be any container FFmpeg can demux — the uploaded original or a
    normalized derivative, video or audio-only. Its audio codec does not matter: what comes
    out is always a mono 16 kHz PCM WAV, which is both the shape pyannote wants and one of
    the three containers NVIDIA's ASD NIM accepts as a separate audio stream. One extraction
    therefore serves both, which is the whole reason this is a context manager the caller
    holds rather than a step hidden inside `diarize_speakers`.

    Nothing is written beside the input and the input is only read. The temp file is removed
    on every path out of the block, including failure and cancellation, so neither a job that
    broke halfway nor one that was cancelled leaves a WAV behind.

    Raises `SpeakerDiarizationAudioError` when the media yields no usable audio,
    `SpeakerDiarizationTimeout` when extraction outlives its limit, and
    `SpeakerDiarizationUnavailable` when FFmpeg itself cannot be run.
    """
    descriptor, name = tempfile.mkstemp(prefix=AUDIO_TEMP_PREFIX, suffix=AUDIO_TEMP_SUFFIX)
    os.close(descriptor)
    audio_path = Path(name)

    try:
        await _extract_audio(media_path, audio_path)
        yield audio_path
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            logger.warning("Removing the diarization temp audio file failed.")


async def diarize_speakers(
    audio_path: Path,
    *,
    auth_token: str | None = None,
    model: str = DIARIZATION_MODEL,
) -> tuple[SpeakerTurn, ...]:
    """Return who spoke when in prepared audio, as pyannote reported it.

    `audio_path` is a WAV in the shape `prepared_audio` produces. This function extracts
    nothing and owns no temp file: the audio is the caller's, because the caller shares it
    with NVIDIA, and it is only read here.

    Returns pyannote's turns in chronological order, empty when the model heard no speech.
    Raises `SpeakerDiarizationUnavailable` when the diarizer cannot be run at all — no token,
    no pyannote, or a model that cannot be fetched — and `SpeakerDiarizationModelError` when
    inference itself fails.
    """
    token = _load_token(auth_token)

    # Both loading and inference are blocking, so both are pushed off the event loop.
    pipeline = await asyncio.to_thread(_load_pipeline, model, token)
    return await asyncio.to_thread(_run_pipeline, pipeline, audio_path)
