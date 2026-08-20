"""Tests for the NVIDIA Synthetic Video Detector client.

These never reach NVIDIA. A fake servicer implementing NVIDIA's own generated service
class is served on loopback, and only the channel constructor is redirected at it, so
the real protobuf serialization, the real bidirectional streaming and the real gRPC
status codes are all exercised against a stand-in provider.
"""

import asyncio
import socket
from pathlib import Path

import grpc
import pytest
from google.protobuf import empty_pb2

from app import nvidia_video
from app.nvidia_svd import syntheticvideodetector_pb2 as svd_pb2
from app.nvidia_svd import syntheticvideodetector_pb2_grpc as svd_pb2_grpc

API_KEY = "nvapi-test-secret-key-value"
FUNCTION_ID = "11111111-2222-3333-4444-555555555555"

# Exactly representable as float32, so the protobuf round trip must preserve them bit
# for bit and the client must not be rescaling anything.
FINAL_LOGIT = 1.5
FINAL_PROBABILITY = 0.87890625


class FakeDetector(svd_pb2_grpc.SyntheticVideoDetectorServiceServicer):
    """A stand-in for NVIDIA that replays a scripted stream and records what it got."""

    def __init__(self, *, responses=None, abort=None, hang=False):
        self._responses = responses or []
        self._abort = abort
        self._hang = hang
        self.received_bytes = b""
        self.received_metadata = {}

    async def DetectSyntheticVideo(self, request_iterator, context):
        self.received_metadata = dict(context.invocation_metadata())

        # Always drain before aborting. gRPC only delivers the server's status to the
        # client reliably once the request stream is finished; a status raised while a
        # large upload is still in flight reaches the client as INTERNAL instead, which
        # would make these tests race rather than assert the client's status mapping.
        async for request in request_iterator:
            self.received_bytes += request.video_file_data

        if self._abort is not None:
            code, details = self._abort
            await context.abort(code, details)

        if self._hang:
            await asyncio.Event().wait()

        for response in self._responses:
            yield response


class RecordingChannel:
    """Delegates to a real insecure channel and records that it was closed.

    The stub only ever asks a channel for `stream_stream`, so this is enough to stand in
    for one while still proving the client releases it.
    """

    def __init__(self, inner):
        self._inner = inner
        self.closed = False

    def stream_stream(self, *args, **kwargs):
        return self._inner.stream_stream(*args, **kwargs)

    async def close(self, grace=None):
        self.closed = True
        await self._inner.close(grace)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def redirect_to(monkeypatch, address: str, channels=None) -> list[RecordingChannel]:
    """Point the client's channel constructor at `address` and record the channels."""
    channels = [] if channels is None else channels

    def fake_secure_channel(target, credentials):
        channel = RecordingChannel(grpc.aio.insecure_channel(address))
        channels.append(channel)
        return channel

    monkeypatch.setattr(nvidia_video.grpc.aio, "secure_channel", fake_secure_channel)
    return channels


async def serve(detector: FakeDetector, address: str):
    server = grpc.aio.server()
    svd_pb2_grpc.add_SyntheticVideoDetectorServiceServicer_to_server(detector, server)
    server.add_insecure_port(address)
    await server.start()
    return server


def analyze_against(detector, monkeypatch, video: Path, channels_out: list, **kwargs):
    """Serve `detector` on loopback and run one analysis against it.

    Created channels are appended to `channels_out` so cleanup stays observable on the
    failure paths too, where no result is ever returned.
    """
    address = f"127.0.0.1:{free_port()}"
    redirect_to(monkeypatch, address, channels_out)

    async def scenario():
        server = await serve(detector, address)
        try:
            return await nvidia_video.analyze_video(
                video,
                api_key=API_KEY,
                function_id=FUNCTION_ID,
                **kwargs,
            )
        finally:
            await server.stop(None)

    return asyncio.run(scenario())


def run_analysis(detector, monkeypatch, video: Path, **kwargs):
    channels: list[RecordingChannel] = []
    return analyze_against(detector, monkeypatch, video, channels, **kwargs), channels


def clip(index: int, logit: float):
    return svd_pb2.DetectSyntheticVideoResponse(
        clip_result=svd_pb2.ClipResult(index=index, logit=logit)
    )


def final(csv_data="0,-2.25\n8,3.5\n", total_clips=2):
    return svd_pb2.DetectSyntheticVideoResponse(
        final_result=svd_pb2.VideoResult(
            logit=FINAL_LOGIT,
            probability=FINAL_PROBABILITY,
            csv_data=csv_data,
            total_clips=total_clips,
        )
    )


@pytest.fixture
def video(tmp_path) -> Path:
    """A file spanning several stream chunks, so reassembly is actually tested."""
    path = tmp_path / "derivative.mp4"
    path.write_bytes(b"\x00\x01\x02\x03" * (nvidia_video.DATA_CHUNK_SIZE // 2) + b"tail")
    return path


def test_final_result_is_parsed_and_preserved_exactly(monkeypatch, video):
    detector = FakeDetector(responses=[clip(0, -2.25), clip(8, 3.5), final()])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.logit == FINAL_LOGIT
    assert result.probability == FINAL_PROBABILITY
    assert result.total_clips == 2
    assert result.csv_data == "0,-2.25\n8,3.5\n"


def test_clip_evidence_is_parsed_when_present(monkeypatch, video):
    detector = FakeDetector(responses=[clip(0, -2.25), clip(8, 3.5), final()])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.clips == (
        nvidia_video.NvidiaClipResult(index=0, logit=-2.25),
        nvidia_video.NvidiaClipResult(index=8, logit=3.5),
    )


def test_keepalive_messages_carry_no_evidence(monkeypatch, video):
    keepalive = svd_pb2.DetectSyntheticVideoResponse(keepalive=empty_pb2.Empty())
    detector = FakeDetector(responses=[keepalive, clip(0, -2.25), keepalive, final()])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.clips == (nvidia_video.NvidiaClipResult(index=0, logit=-2.25),)
    assert result.probability == FINAL_PROBABILITY


def test_whole_video_is_streamed_to_the_provider(monkeypatch, video):
    detector = FakeDetector(responses=[final()])

    run_analysis(detector, monkeypatch, video)

    assert detector.received_bytes == video.read_bytes()


def test_credentials_are_sent_as_nvcf_call_metadata(monkeypatch, video):
    detector = FakeDetector(responses=[final()])

    run_analysis(detector, monkeypatch, video)

    assert detector.received_metadata["authorization"] == f"Bearer {API_KEY}"
    assert detector.received_metadata["function-id"] == FUNCTION_ID


def test_stream_without_a_final_result_is_rejected(monkeypatch, video):
    detector = FakeDetector(responses=[clip(0, -2.25)])

    with pytest.raises(nvidia_video.NvidiaProviderInvalidResponse):
        run_analysis(detector, monkeypatch, video)


@pytest.mark.parametrize(
    "code",
    [grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED],
)
def test_rejected_credentials_raise_an_authentication_error(monkeypatch, video, code):
    detector = FakeDetector(abort=(code, "invalid credentials"))

    with pytest.raises(nvidia_video.NvidiaAuthenticationError):
        run_analysis(detector, monkeypatch, video)


def test_unreachable_provider_is_reported_as_unavailable(monkeypatch, video):
    # Nothing is listening on this port, so the connection attempt fails outright.
    redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(nvidia_video.NvidiaProviderUnavailable):
        asyncio.run(
            nvidia_video.analyze_video(
                video, api_key=API_KEY, function_id=FUNCTION_ID, timeout_seconds=5.0
            )
        )


def test_exceeding_the_deadline_raises_a_timeout(monkeypatch, video):
    detector = FakeDetector(responses=[final()], hang=True)

    with pytest.raises(nvidia_video.NvidiaProviderTimeout):
        run_analysis(detector, monkeypatch, video, timeout_seconds=0.5)


def test_other_provider_failures_stay_generic(monkeypatch, video):
    detector = FakeDetector(abort=(grpc.StatusCode.INTERNAL, "backend exploded"))

    with pytest.raises(nvidia_video.NvidiaProviderError) as raised:
        run_analysis(detector, monkeypatch, video)

    assert not isinstance(raised.value, nvidia_video.NvidiaAuthenticationError)
    assert "backend exploded" in str(raised.value)


def test_missing_local_file_fails_before_any_connection(monkeypatch, tmp_path):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(nvidia_video.NvidiaLocalFileError):
        asyncio.run(
            nvidia_video.analyze_video(
                tmp_path / "absent.mp4", api_key=API_KEY, function_id=FUNCTION_ID
            )
        )

    assert channels == []


def test_channel_is_released_after_a_provider_failure(monkeypatch, video):
    detector = FakeDetector(abort=(grpc.StatusCode.UNAUTHENTICATED, "nope"))
    channels: list[RecordingChannel] = []

    with pytest.raises(nvidia_video.NvidiaAuthenticationError):
        analyze_against(detector, monkeypatch, video, channels)

    assert [channel.closed for channel in channels] == [True]


def test_channel_is_released_after_success(monkeypatch, video):
    detector = FakeDetector(responses=[final()])

    _, channels = run_analysis(detector, monkeypatch, video)

    assert [channel.closed for channel in channels] == [True]


def test_api_key_never_appears_in_provider_errors(monkeypatch, video):
    detector = FakeDetector(abort=(grpc.StatusCode.UNAUTHENTICATED, "invalid credentials"))

    with pytest.raises(nvidia_video.NvidiaAuthenticationError) as raised:
        run_analysis(detector, monkeypatch, video)

    assert API_KEY not in str(raised.value)
    assert API_KEY not in repr(raised.value)


def test_api_key_never_appears_in_the_result(monkeypatch, video):
    detector = FakeDetector(responses=[clip(0, -2.25), final()])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert API_KEY not in repr(result)


def test_missing_api_key_is_reported_without_contacting_the_provider(monkeypatch, video):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(nvidia_video.NvidiaAuthenticationError):
        asyncio.run(nvidia_video.analyze_video(video, function_id=FUNCTION_ID))

    assert channels == []


def test_missing_function_id_is_reported_without_contacting_the_provider(monkeypatch, video):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")
    monkeypatch.delenv("NVIDIA_SVD_FUNCTION_ID", raising=False)

    with pytest.raises(nvidia_video.NvidiaAuthenticationError):
        asyncio.run(nvidia_video.analyze_video(video, api_key=API_KEY))

    assert channels == []


def test_credentials_fall_back_to_the_environment(monkeypatch, video):
    monkeypatch.setenv("NVIDIA_API_KEY", API_KEY)
    monkeypatch.setenv("NVIDIA_SVD_FUNCTION_ID", FUNCTION_ID)

    detector = FakeDetector(responses=[final()])
    address = f"127.0.0.1:{free_port()}"
    redirect_to(monkeypatch, address)

    async def scenario():
        server = await serve(detector, address)
        try:
            return await nvidia_video.analyze_video(video)
        finally:
            await server.stop(None)

    result = asyncio.run(scenario())

    assert result.probability == FINAL_PROBABILITY
    assert detector.received_metadata["function-id"] == FUNCTION_ID
