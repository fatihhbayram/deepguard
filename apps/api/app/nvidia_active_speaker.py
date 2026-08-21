"""Client for the NVIDIA Maxine Active Speaker Detection NIM.

The detector is reached over NVIDIA's gRPC API. Inference is a single bidirectional
streaming call: the client sends one configuration message, then streams the video bytes,
the audio bytes and the diarization segments up in interleaved chunks, and NVIDIA streams
per-frame speaker evidence back.

The hosted Preview ("Try API") deployment runs behind NVIDIA Cloud Functions, so the same
RPC is addressed by a function ID carried in call metadata rather than by a per-service
hostname.

This module talks to exactly one provider and deliberately shares no base class with
`nvidia_video.py`. The two NIMs have different inputs, a different stream protocol and a
different result shape; the only thing they have in common is that NVIDIA hosts both, and
that is not a stable shared pattern to abstract over.

What comes back is NVIDIA's own per-frame observation — a face box, the diarized speaker it
was matched to, whether that face is speaking in that frame, and NVIDIA's face-detection
confidence. Those numbers are returned exactly as reported (rule 11). This module invents no
speaker segments, merges no frames, and derives no confidence of its own; turning frame
evidence into DeepGuard signals is a later concern.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Sequence

import grpc
from dotenv import load_dotenv

from app.nvidia_active_speaker_proto import activespeakerdetection_pb2 as asd_pb2
from app.nvidia_active_speaker_proto import activespeakerdetection_pb2_grpc as asd_pb2_grpc
from app.nvidia_active_speaker_proto import audio_pb2, video_pb2

load_dotenv()

logger = logging.getLogger(__name__)

# NVIDIA's hosted Preview endpoint. TLS is mandatory here and is never relaxed.
PREVIEW_TARGET = "grpc.nvcf.nvidia.com:443"

# Matches the chunk size NVIDIA's own Active Speaker sample client streams with. It is
# deliberately smaller than the Synthetic Video client's: this stream interleaves three
# inputs, so smaller chunks keep them advancing together.
DATA_CHUNK_SIZE = 64 * 1024

# Diarization segments are sent in batches rather than one message per segment, again
# matching NVIDIA's sample client.
DIARIZATION_BATCH_SIZE = 100

# Detection is a full per-frame video pass on NVIDIA's GPU, so the deadline is generous. It
# exists to guarantee the call always ends, not to bound normal inference.
DEFAULT_TIMEOUT_SECONDS = 600.0

# NVIDIA accepts H.264 in MP4 and nothing else, which is what P1's derivative already is.
VIDEO_CODEC = video_pb2.VIDEO_CODEC_H264

# The audio codec is read from the file's extension, because these are the only three
# containers NVIDIA's AudioCodec enum can describe.
_AUDIO_CODECS_BY_SUFFIX = {
    ".wav": audio_pb2.AUDIO_CODEC_WAV,
    ".mp3": audio_pb2.AUDIO_CODEC_MP3,
    ".opus": audio_pb2.AUDIO_CODEC_OPUS,
}

# When no separate audio file is given, NVIDIA demuxes the audio out of the video container
# itself. It still wants a codec on the config; this is the value NVIDIA's own sample client
# sends for that case.
EMBEDDED_AUDIO_CODEC = audio_pb2.AUDIO_CODEC_OPUS


class NvidiaActiveSpeakerError(Exception):
    """Base class for every failure raised by this module."""


class NvidiaActiveSpeakerInputError(NvidiaActiveSpeakerError):
    """The inputs this module was handed are not usable. Nothing was sent to NVIDIA."""


class NvidiaActiveSpeakerLocalFileError(NvidiaActiveSpeakerError):
    """A local input file could not be opened or read. Nothing usable reached NVIDIA."""


class NvidiaActiveSpeakerProviderError(NvidiaActiveSpeakerError):
    """NVIDIA was reached but the call did not produce a usable result."""


class NvidiaActiveSpeakerAuthenticationError(NvidiaActiveSpeakerProviderError):
    """NVIDIA rejected the credentials or the function is not accessible to them."""


class NvidiaActiveSpeakerUnavailable(NvidiaActiveSpeakerProviderError):
    """NVIDIA could not be reached, or the service reported itself as unavailable."""


class NvidiaActiveSpeakerTimeout(NvidiaActiveSpeakerProviderError):
    """The call exceeded its deadline before NVIDIA finished streaming results."""


class NvidiaActiveSpeakerInvalidResponse(NvidiaActiveSpeakerProviderError):
    """The stream completed without carrying a single per-frame result."""


@dataclass(frozen=True)
class DiarizationSegment:
    """One diarized speech segment, as this module requires it on input.

    Times are milliseconds from the start of the audio, matching NVIDIA's own
    `AudioSegmentInfo`. `speaker_id` is whatever distinct integer the diarizer assigned to a
    voice; NVIDIA echoes it back on the frames it matched to that voice.
    """

    start_time_ms: int
    end_time_ms: int
    speaker_id: int


@dataclass(frozen=True)
class NvidiaBoundingBox:
    """A face box in pixels, with `x`/`y` at its top-left corner.

    Reported by NVIDIA as float32 and kept unrounded and unscaled.
    """

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class NvidiaSpeakerObservation:
    """NVIDIA's observation of one face in one frame, exactly as reported.

    `face_id` identifies the tracked face across frames and starts at 0.
    `diarized_speaker_id` is the diarized voice NVIDIA matched that face to, and is -1 when
    it matched none. `is_speaking` is NVIDIA's own verdict for this face in this frame,
    already decided against its speaker-detection threshold.

    `face_detection_confidence` is confidence in the *face detection*, from 0.0 to 1.0. It
    is not a confidence in `is_speaking`, and must not be read as one.
    """

    face_id: int
    diarized_speaker_id: int
    is_speaking: bool
    face_detection_confidence: float
    bounding_box: NvidiaBoundingBox


@dataclass(frozen=True)
class NvidiaActiveSpeakerFrame:
    """Every face NVIDIA observed in one frame.

    `speakers` is empty when NVIDIA reported the frame with no faces in it, which is itself
    evidence and is preserved rather than dropped.
    """

    frame_id: int
    speakers: tuple[NvidiaSpeakerObservation, ...] = field(default=())


@dataclass(frozen=True)
class NvidiaActiveSpeakerResult:
    """NVIDIA's per-frame active-speaker evidence for one video.

    `frames` holds one entry per frame NVIDIA reported, in arrival order. It is bounded by
    the length of the video, which P1 already bounds at upload; no raw response payload is
    retained behind it.

    `speaker_detection_threshold` is the threshold NVIDIA echoed back on its configuration
    message — the value `is_speaking` was actually decided against. It is `None` when NVIDIA
    echoed no configuration.

    `function_id` is the NVCF function that answered. NVIDIA reports no model version of its
    own, and the function ID is what pins the deployed detector behind this result, so it
    travels with the evidence it produced. It is a public catalog identifier, never a
    credential.
    """

    frames: tuple[NvidiaActiveSpeakerFrame, ...]
    function_id: str
    speaker_detection_threshold: float | None = None


def _load_credentials(api_key: str | None, function_id: str | None) -> tuple[str, str]:
    """Resolve the API key and Preview function ID, preferring explicit arguments.

    Both are configuration, never arguments a request may influence, and neither is ever
    logged or placed in an exception message.
    """
    resolved_key = api_key or os.getenv("NVIDIA_API_KEY")
    if not resolved_key:
        raise NvidiaActiveSpeakerAuthenticationError("NVIDIA_API_KEY is not configured")

    resolved_function_id = function_id or os.getenv("NVIDIA_ASD_FUNCTION_ID")
    if not resolved_function_id:
        raise NvidiaActiveSpeakerAuthenticationError("NVIDIA_ASD_FUNCTION_ID is not configured")

    return resolved_key, resolved_function_id


def _validate_diarization(diarization: Sequence[DiarizationSegment]) -> None:
    """Reject diarization NVIDIA's proto cannot carry, before any RPC is started.

    Segment times are `uint32` on the wire, so a negative offset would fail mid-stream as an
    opaque serialization error. Checking here turns that into a stated input failure.

    Empty diarization is refused too. NVIDIA tolerates a session without it, but then every
    face is reported with no speaker assigned, so an empty sequence reaching this required
    argument is a caller defect rather than a request for that mode.
    """
    if not diarization:
        raise NvidiaActiveSpeakerInputError("Diarization is required and carries no segments")

    for position, segment in enumerate(diarization):
        if segment.start_time_ms < 0 or segment.end_time_ms < 0:
            raise NvidiaActiveSpeakerInputError(
                f"Diarization segment {position} has a negative time offset "
                f"({segment.start_time_ms}, {segment.end_time_ms})"
            )
        if segment.end_time_ms < segment.start_time_ms:
            raise NvidiaActiveSpeakerInputError(
                f"Diarization segment {position} ends before it starts "
                f"({segment.start_time_ms} > {segment.end_time_ms})"
            )


def _audio_codec(audio_path: Path) -> int:
    """Map the audio file's extension onto NVIDIA's codec enum."""
    codec = _AUDIO_CODECS_BY_SUFFIX.get(audio_path.suffix.lower())
    if codec is None:
        supported = ", ".join(sorted(_AUDIO_CODECS_BY_SUFFIX))
        raise NvidiaActiveSpeakerInputError(
            f"Unsupported audio format '{audio_path.suffix}' for '{audio_path.name}'. "
            f"NVIDIA accepts: {supported}"
        )
    return codec


def _build_config(
    audio_path: Path | None,
    speaker_detection_threshold: float | None,
) -> asd_pb2.ActiveSpeakerDetectionConfig:
    """Describe the inputs to NVIDIA.

    With a separate audio file NVIDIA is told to expect its own audio stream; without one it
    demuxes the audio out of the video container instead.
    """
    if audio_path is None:
        audio_source = asd_pb2.AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO
        encoding = EMBEDDED_AUDIO_CODEC
    else:
        audio_source = asd_pb2.AUDIO_SOURCE_CONFIG_SEPARATE_STREAM
        encoding = _audio_codec(audio_path)

    config = asd_pb2.ActiveSpeakerDetectionConfig(
        input_video_config=video_pb2.VideoConfig(codec=VIDEO_CODEC),
        input_audio_config=audio_pb2.AudioConfig(encoding=encoding),
        audio_source_config=audio_source,
    )

    # Left unset when not given, so NVIDIA applies its own documented default rather than
    # one this module made up.
    if speaker_detection_threshold is not None:
        config.speaker_detection_threshold = speaker_detection_threshold

    return config


def _open_input(file_path: Path, kind: str):
    """Open a local input, failing before any RPC is started."""
    try:
        return open(file_path, "rb")
    except OSError as error:
        raise NvidiaActiveSpeakerLocalFileError(
            f"Cannot read {kind} file '{file_path}': {error}"
        ) from error


class _RequestStream:
    """Streams configuration and inputs to NVIDIA, and remembers a local read failure.

    gRPC consumes the request iterator in its own task, so an OSError raised mid-stream
    surfaces to us only as an aborted RPC. Recording it here lets the caller be told the
    truth — that our disk failed — instead of a misleading provider error.

    Video, audio and diarization are interleaved rather than sent one after another. NVIDIA
    switches a streamable MP4 into streaming mode and begins inferring before the upload
    finishes, and diarization that arrives after that first inference is ignored for it.
    Advancing all three together keeps the speaker assignments correct in that mode.
    """

    def __init__(
        self,
        *,
        video_path: Path,
        audio_path: Path | None,
        diarization: Sequence[DiarizationSegment],
        config: asd_pb2.ActiveSpeakerDetectionConfig,
    ) -> None:
        self._video_path = video_path
        self._audio_path = audio_path
        self._diarization = diarization
        self._config = config
        self.local_error: NvidiaActiveSpeakerLocalFileError | None = None

    def _diarization_batch(self, start: int) -> asd_pb2.DetectActiveSpeakerRequest:
        batch = self._diarization[start : start + DIARIZATION_BATCH_SIZE]
        return asd_pb2.DetectActiveSpeakerRequest(
            data=asd_pb2.ActiveSpeakerDetectionData(
                diarization_info=asd_pb2.AudioDiarizationInfo(
                    segments=[
                        asd_pb2.AudioSegmentInfo(
                            start_time=segment.start_time_ms,
                            end_time=segment.end_time_ms,
                            speaker_id=segment.speaker_id,
                        )
                        for segment in batch
                    ]
                )
            )
        )

    async def requests(self) -> AsyncIterator[asd_pb2.DetectActiveSpeakerRequest]:
        video_file = _open_input(self._video_path, "video")
        audio_file = _open_input(self._audio_path, "audio") if self._audio_path else None

        try:
            # The configuration message must lead the stream; NVIDIA rejects data before it.
            yield asd_pb2.DetectActiveSpeakerRequest(config=self._config)

            video_done = False
            audio_done = audio_file is None
            segments_sent = 0

            while not (video_done and audio_done and segments_sent >= len(self._diarization)):
                if not video_done:
                    # Reads are blocking; keep them off the event loop.
                    chunk = await asyncio.to_thread(video_file.read, DATA_CHUNK_SIZE)
                    if chunk:
                        yield asd_pb2.DetectActiveSpeakerRequest(
                            data=asd_pb2.ActiveSpeakerDetectionData(video_data=chunk)
                        )
                    else:
                        video_done = True

                if not audio_done:
                    chunk = await asyncio.to_thread(audio_file.read, DATA_CHUNK_SIZE)
                    if chunk:
                        yield asd_pb2.DetectActiveSpeakerRequest(
                            data=asd_pb2.ActiveSpeakerDetectionData(audio_data=chunk)
                        )
                    else:
                        audio_done = True

                if segments_sent < len(self._diarization):
                    request = self._diarization_batch(segments_sent)
                    segments_sent += len(request.data.diarization_info.segments)
                    yield request

        except NvidiaActiveSpeakerLocalFileError as error:
            self.local_error = error
            raise
        except OSError as error:
            self.local_error = NvidiaActiveSpeakerLocalFileError(
                f"Failed reading local input for '{self._video_path}': {error}"
            )
            raise self.local_error from error
        finally:
            video_file.close()
            if audio_file is not None:
                audio_file.close()


def _provider_error(error: grpc.RpcError) -> NvidiaActiveSpeakerProviderError:
    """Map a gRPC status onto the failure modes the caller has to tell apart.

    Only the status code and NVIDIA's own details text are used, so no credential can reach
    the message.

    One gRPC limitation is worth knowing: a status the server sends while a large upload is
    still in flight can reach the client as INTERNAL instead of the real code, so a rejection
    during upload may land in the generic bucket rather than a specific one.
    """
    code = error.code()
    details = error.details() or ""

    if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
        return NvidiaActiveSpeakerAuthenticationError(
            f"NVIDIA rejected the request ({code.name}): {details}"
        )
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return NvidiaActiveSpeakerTimeout(
            f"NVIDIA did not finish before the deadline: {details}"
        )
    if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.RESOURCE_EXHAUSTED):
        return NvidiaActiveSpeakerUnavailable(f"NVIDIA is unavailable ({code.name}): {details}")
    return NvidiaActiveSpeakerProviderError(f"NVIDIA call failed ({code.name}): {details}")


def _parse_frame(result: asd_pb2.ActiveSpeakerDetectionResult) -> NvidiaActiveSpeakerFrame:
    """Copy one frame's evidence out of the protobuf, dropping the payload behind it."""
    return NvidiaActiveSpeakerFrame(
        frame_id=result.frame_id,
        speakers=tuple(
            NvidiaSpeakerObservation(
                face_id=speaker.face_id,
                diarized_speaker_id=speaker.diarized_speaker_id,
                is_speaking=speaker.is_speaking,
                face_detection_confidence=speaker.face_detection_confidence,
                bounding_box=NvidiaBoundingBox(
                    x=speaker.speaker_bbox.x,
                    y=speaker.speaker_bbox.y,
                    width=speaker.speaker_bbox.width,
                    height=speaker.speaker_bbox.height,
                ),
            )
            for speaker in result.speaker_data
        ),
    )


async def _collect(call, function_id: str) -> NvidiaActiveSpeakerResult:
    """Drain the response stream into the provider's typed result.

    Each response is parsed and released as it arrives, so the raw protobuf messages are
    never accumulated. Keepalive messages exist only to hold the stream open during a long
    inference and carry no evidence.
    """
    frames: list[NvidiaActiveSpeakerFrame] = []
    threshold: float | None = None

    async for response in call:
        if response.HasField("active_speaker_detection_result"):
            frames.append(_parse_frame(response.active_speaker_detection_result))
        elif response.HasField("config"):
            # The echoed configuration reports the threshold `is_speaking` was decided
            # against. Only that one number is kept — never the message itself.
            if response.config.HasField("speaker_detection_threshold"):
                threshold = response.config.speaker_detection_threshold
        elif response.HasField("keepalive"):
            continue

    if not frames:
        raise NvidiaActiveSpeakerInvalidResponse(
            "NVIDIA closed the stream without reporting a single frame"
        )

    return NvidiaActiveSpeakerResult(
        frames=tuple(frames),
        function_id=function_id,
        speaker_detection_threshold=threshold,
    )


async def analyze_active_speaker(
    video_path: Path,
    diarization: Sequence[DiarizationSegment],
    *,
    audio_path: Path | None = None,
    speaker_detection_threshold: float | None = None,
    api_key: str | None = None,
    function_id: str | None = None,
    target: str = PREVIEW_TARGET,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> NvidiaActiveSpeakerResult:
    """Run NVIDIA active-speaker detection over a video and its diarization.

    `video_path` must be an MP4/H.264 file, ideally streamable (`moov` atom first, which
    `ffmpeg -movflags +faststart` produces); NVIDIA falls back to transactional processing
    for the rest. This function never inspects, transcodes or rewrites the media.

    `diarization` must carry at least one segment. Pass `audio_path` to send audio as its own
    WAV/MP3/Opus stream; leave it `None` to have NVIDIA demux the audio out of the video
    container.

    Leave `speaker_detection_threshold` unset to use NVIDIA's own default.

    Returns the provider's per-frame evidence untransformed. Raises
    `NvidiaActiveSpeakerInputError` for unusable arguments,
    `NvidiaActiveSpeakerLocalFileError` if our own file cannot be read, and a
    `NvidiaActiveSpeakerProviderError` subclass for every provider-side outcome.
    """
    resolved_key, resolved_function_id = _load_credentials(api_key, function_id)
    _validate_diarization(diarization)

    # Builds the config before opening anything, so an unsupported audio format is reported
    # as the input error it is rather than after a connection has been paid for.
    config = _build_config(audio_path, speaker_detection_threshold)

    metadata = (
        ("authorization", f"Bearer {resolved_key}"),
        ("function-id", resolved_function_id),
    )

    # Fail on an unreadable input before opening a channel, so the common local error never
    # costs a connection.
    _open_input(video_path, "video").close()
    if audio_path is not None:
        _open_input(audio_path, "audio").close()

    stream = _RequestStream(
        video_path=video_path,
        audio_path=audio_path,
        diarization=diarization,
        config=config,
    )
    channel = grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
    call = None

    try:
        stub = asd_pb2_grpc.ActiveSpeakerDetectionServiceStub(channel)
        call = stub.DetectActiveSpeaker(
            stream.requests(),
            metadata=metadata,
            timeout=timeout_seconds,
        )
        return await _collect(call, resolved_function_id)
    except grpc.RpcError as error:
        # A local read failure aborts the RPC, so the gRPC status describes a symptom rather
        # than the cause. The recorded cause wins.
        if stream.local_error is not None:
            raise stream.local_error from error
        raise _provider_error(error) from error
    except asyncio.CancelledError as error:
        # When the request iterator raises, grpc.aio cancels the call and the response
        # iterator reports that cancellation rather than a status — so a failure on our own
        # disk arrives here, not above. Only our own recorded cause is converted; a
        # cancellation from anywhere else, including the caller, still propagates untouched.
        if stream.local_error is not None:
            raise stream.local_error from error
        raise
    finally:
        # Cancelling first stops NVIDIA-side work and unblocks the request iterator; closing
        # the channel then releases the connection. Both run on every path, including caller
        # cancellation.
        if call is not None:
            call.cancel()
        await channel.close()
