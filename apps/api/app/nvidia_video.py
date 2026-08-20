"""Client for the NVIDIA Synthetic Video Detector (D009).

The detector is reached over NVIDIA's gRPC API. Inference is a single bidirectional
streaming call: the client streams the MP4 bytes up in chunks and NVIDIA streams
per-clip results back, followed by one aggregate result for the video.

The hosted Preview ("Try API") deployment runs behind NVIDIA Cloud Functions, so the
same RPC is addressed by a function ID carried in call metadata rather than by a
per-service hostname.

This module talks to exactly one provider and deliberately has no provider abstraction:
NVIDIA's numbers are returned as NVIDIA reports them, untransformed and unaggregated
(rule 11), and turning them into a DeepGuard risk level is a later concern.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import grpc
from dotenv import load_dotenv

from app.nvidia_svd import syntheticvideodetector_pb2 as svd_pb2
from app.nvidia_svd import syntheticvideodetector_pb2_grpc as svd_pb2_grpc

load_dotenv()

logger = logging.getLogger(__name__)

# NVIDIA's hosted Preview endpoint. TLS is mandatory here and is never relaxed.
PREVIEW_TARGET = "grpc.nvcf.nvidia.com:443"

# Matches the chunk size NVIDIA's own sample client streams with.
DATA_CHUNK_SIZE = 1024 * 1024

# Detection is a full video pass on NVIDIA's GPU, so the deadline is generous. It exists
# to guarantee the call always ends, not to bound normal inference.
DEFAULT_TIMEOUT_SECONDS = 600.0


class NvidiaVideoError(Exception):
    """Base class for every failure raised by this module."""


class NvidiaLocalFileError(NvidiaVideoError):
    """The local video file could not be opened or read. Nothing was sent to NVIDIA."""


class NvidiaProviderError(NvidiaVideoError):
    """NVIDIA was reached but the call did not produce a usable result."""


class NvidiaAuthenticationError(NvidiaProviderError):
    """NVIDIA rejected the credentials or the function is not accessible to them."""


class NvidiaProviderUnavailable(NvidiaProviderError):
    """NVIDIA could not be reached, or the service reported itself as unavailable."""


class NvidiaProviderTimeout(NvidiaProviderError):
    """The call exceeded its deadline before NVIDIA returned a final result."""


class NvidiaProviderInvalidResponse(NvidiaProviderError):
    """The stream completed without the final aggregate result the protocol defines."""


@dataclass(frozen=True)
class NvidiaClipResult:
    """One clip-level detection result, exactly as NVIDIA reported it.

    NVIDIA emits a raw model logit per clip and no probability; `index` is the frame
    index of the clip's middle frame. Neither value is rescaled here.
    """

    index: int
    logit: float


@dataclass(frozen=True)
class NvidiaVideoResult:
    """NVIDIA's aggregate result for one video, plus the clip evidence behind it.

    `logit` and `probability` are the provider's own aggregate figures. `csv_data` is
    NVIDIA's `index,logit` rendering of the clip table, kept verbatim because it is the
    provider's own evidence artifact.
    """

    logit: float
    probability: float
    total_clips: int
    csv_data: str
    clips: tuple[NvidiaClipResult, ...] = field(default=())


def _load_credentials(api_key: str | None, function_id: str | None) -> tuple[str, str]:
    """Resolve the API key and Preview function ID, preferring explicit arguments.

    Both are configuration, never arguments a request may influence, and neither is ever
    logged or placed in an exception message.
    """
    resolved_key = api_key or os.getenv("NVIDIA_API_KEY")
    if not resolved_key:
        raise NvidiaAuthenticationError("NVIDIA_API_KEY is not configured")

    resolved_function_id = function_id or os.getenv("NVIDIA_SVD_FUNCTION_ID")
    if not resolved_function_id:
        raise NvidiaAuthenticationError("NVIDIA_SVD_FUNCTION_ID is not configured")

    return resolved_key, resolved_function_id


def _open_video(file_path: Path):
    """Open the provider-compatible file, failing before any RPC is started.

    P1 already normalized and validated this media (D013); the only question left is
    whether the local file can still be read.
    """
    try:
        return open(file_path, "rb")
    except OSError as error:
        raise NvidiaLocalFileError(f"Cannot read video file '{file_path}': {error}") from error


class _VideoStream:
    """Streams the file to NVIDIA and remembers a local read failure.

    gRPC consumes the request iterator in its own task, so an OSError raised mid-stream
    surfaces to us only as an aborted RPC. Recording it here lets the caller be told the
    truth — that our disk failed — instead of a misleading provider error.
    """

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path
        self.local_error: NvidiaLocalFileError | None = None

    async def requests(self) -> AsyncIterator[svd_pb2.DetectSyntheticVideoRequest]:
        video_file = _open_video(self._file_path)
        try:
            while True:
                # Reads are blocking; keep them off the event loop.
                chunk = await asyncio.to_thread(video_file.read, DATA_CHUNK_SIZE)
                if not chunk:
                    break
                yield svd_pb2.DetectSyntheticVideoRequest(video_file_data=chunk)
        except NvidiaLocalFileError as error:
            self.local_error = error
            raise
        except OSError as error:
            self.local_error = NvidiaLocalFileError(
                f"Failed reading video file '{self._file_path}': {error}"
            )
            raise self.local_error from error
        finally:
            video_file.close()


def _provider_error(error: grpc.RpcError) -> NvidiaProviderError:
    """Map a gRPC status onto the failure modes the caller has to tell apart.

    Only the status code and NVIDIA's own details text are used, so no credential can
    reach the message.

    One gRPC limitation is worth knowing: a status the server sends while a large upload
    is still in flight can reach the client as INTERNAL instead of the real code, so a
    rejection during upload may land in the generic bucket rather than a specific one.
    """
    code = error.code()
    details = error.details() or ""

    if code in (grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED):
        return NvidiaAuthenticationError(f"NVIDIA rejected the request ({code.name}): {details}")
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return NvidiaProviderTimeout(f"NVIDIA did not respond before the deadline: {details}")
    if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.RESOURCE_EXHAUSTED):
        return NvidiaProviderUnavailable(f"NVIDIA is unavailable ({code.name}): {details}")
    return NvidiaProviderError(f"NVIDIA call failed ({code.name}): {details}")


async def _collect(call) -> NvidiaVideoResult:
    """Drain the response stream into the provider's typed result.

    Keepalive messages exist only to hold the stream open during a long inference and
    carry no evidence.
    """
    clips: list[NvidiaClipResult] = []
    final: svd_pb2.VideoResult | None = None

    async for response in call:
        if response.HasField("clip_result"):
            clips.append(
                NvidiaClipResult(
                    index=response.clip_result.index,
                    logit=response.clip_result.logit,
                )
            )
        elif response.HasField("final_result"):
            final = response.final_result
        elif response.HasField("keepalive"):
            continue

    if final is None:
        raise NvidiaProviderInvalidResponse(
            "NVIDIA closed the stream without a final result "
            f"(received {len(clips)} clip results)"
        )

    return NvidiaVideoResult(
        logit=final.logit,
        probability=final.probability,
        total_clips=final.total_clips,
        csv_data=final.csv_data,
        clips=tuple(clips),
    )


async def analyze_video(
    file_path: Path,
    *,
    api_key: str | None = None,
    function_id: str | None = None,
    target: str = PREVIEW_TARGET,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> NvidiaVideoResult:
    """Run NVIDIA synthetic-video detection on an already provider-compatible MP4.

    `file_path` must be the MP4/H.264 constant-frame-rate derivative P1 produced; this
    function never inspects, transcodes or rewrites the media.

    Raises `NvidiaLocalFileError` if our own file cannot be read, and a
    `NvidiaProviderError` subclass for every provider-side outcome.
    """
    resolved_key, resolved_function_id = _load_credentials(api_key, function_id)

    metadata = (
        ("authorization", f"Bearer {resolved_key}"),
        ("function-id", resolved_function_id),
    )

    # Fail on an unreadable file before opening a channel, so the common local error
    # never costs a connection.
    _open_video(file_path).close()

    stream = _VideoStream(file_path)
    channel = grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
    call = None

    try:
        stub = svd_pb2_grpc.SyntheticVideoDetectorServiceStub(channel)
        call = stub.DetectSyntheticVideo(
            stream.requests(),
            metadata=metadata,
            timeout=timeout_seconds,
        )
        return await _collect(call)
    except grpc.RpcError as error:
        # A local read failure aborts the RPC, so the gRPC status describes a symptom
        # rather than the cause. The recorded cause wins.
        if stream.local_error is not None:
            raise stream.local_error from error
        raise _provider_error(error) from error
    finally:
        # Cancelling first stops NVIDIA-side work and unblocks the request iterator;
        # closing the channel then releases the connection. Both run on every path,
        # including caller cancellation.
        if call is not None:
            call.cancel()
        await channel.close()
