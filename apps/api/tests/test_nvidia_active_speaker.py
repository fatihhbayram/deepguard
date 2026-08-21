"""Tests for the NVIDIA Active Speaker Detection client.

These never reach NVIDIA. A fake servicer implementing NVIDIA's own generated service class
is served on loopback, and only the channel constructor is redirected at it, so the real
protobuf serialization, the real bidirectional streaming and the real gRPC status codes are
all exercised against a stand-in provider.
"""

import asyncio
import builtins
import socket
from pathlib import Path

import grpc
import pytest
from google.protobuf import empty_pb2

from app import nvidia_active_speaker as asd
from app.nvidia_active_speaker_proto import activespeakerdetection_pb2 as asd_pb2
from app.nvidia_active_speaker_proto import activespeakerdetection_pb2_grpc as asd_pb2_grpc
from app.nvidia_active_speaker_proto import audio_pb2, common_pb2, video_pb2

API_KEY = "nvapi-test-secret-key-value"
FUNCTION_ID = "11111111-2222-3333-4444-555555555555"

# Exactly representable as float32, so the protobuf round trip must preserve them bit for bit
# and the client must not be rescaling anything.
CONFIDENCE = 0.828125
THRESHOLD = 0.59375
BBOX = (10.5, 20.25, 64.0, 128.5)

DIARIZATION = (
    asd.DiarizationSegment(start_time_ms=0, end_time_ms=1200, speaker_id=0),
    asd.DiarizationSegment(start_time_ms=1200, end_time_ms=2400, speaker_id=1),
)


class FakeDetector(asd_pb2_grpc.ActiveSpeakerDetectionServiceServicer):
    """A stand-in for NVIDIA that replays a scripted stream and records what it got."""

    def __init__(self, *, responses=None, abort=None, hang=False):
        self._responses = responses or []
        self._abort = abort
        self._hang = hang
        self.received_video = b""
        self.received_audio = b""
        self.received_segments = []
        self.received_config = None
        self.received_metadata = {}
        self.request_kinds = []

    async def DetectActiveSpeaker(self, request_iterator, context):
        self.received_metadata = dict(context.invocation_metadata())

        # Always drain before aborting. gRPC only delivers the server's status to the client
        # reliably once the request stream is finished; a status raised while a large upload
        # is still in flight reaches the client as INTERNAL instead, which would make these
        # tests race rather than assert the client's status mapping.
        async for request in request_iterator:
            if request.HasField("config"):
                self.received_config = request.config
                self.request_kinds.append("config")
                continue

            data = request.data
            if data.HasField("video_data"):
                self.received_video += data.video_data
                self.request_kinds.append("video")
            elif data.HasField("audio_data"):
                self.received_audio += data.audio_data
                self.request_kinds.append("audio")
            elif data.HasField("diarization_info"):
                self.received_segments.extend(data.diarization_info.segments)
                self.request_kinds.append("diarization")

        if self._abort is not None:
            code, details = self._abort
            await context.abort(code, details)

        if self._hang:
            await asyncio.Event().wait()

        for response in self._responses:
            yield response


class RecordingChannel:
    """Delegates to a real insecure channel and records that it was closed.

    The stub only ever asks a channel for `stream_stream`, so this is enough to stand in for
    one while still proving the client releases it.
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

    monkeypatch.setattr(asd.grpc.aio, "secure_channel", fake_secure_channel)
    return channels


async def serve(detector: FakeDetector, address: str):
    server = grpc.aio.server()
    asd_pb2_grpc.add_ActiveSpeakerDetectionServiceServicer_to_server(detector, server)
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
    kwargs.setdefault("diarization", DIARIZATION)

    async def scenario():
        server = await serve(detector, address)
        try:
            return await asd.analyze_active_speaker(
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


def speaker(
    *,
    face_id=0,
    diarized_speaker_id=0,
    is_speaking=True,
    confidence=CONFIDENCE,
    bbox=BBOX,
):
    x, y, width, height = bbox
    return asd_pb2.SpeakerInfo(
        speaker_bbox=common_pb2.BoundingBox(x=x, y=y, width=width, height=height),
        diarized_speaker_id=diarized_speaker_id,
        face_id=face_id,
        is_speaking=is_speaking,
        face_detection_confidence=confidence,
    )


def frame(frame_id: int, *speakers):
    return asd_pb2.DetectActiveSpeakerResponse(
        active_speaker_detection_result=asd_pb2.ActiveSpeakerDetectionResult(
            frame_id=frame_id, speaker_data=list(speakers)
        )
    )


def echoed_config(threshold=THRESHOLD):
    config = asd_pb2.ActiveSpeakerDetectionConfig()
    if threshold is not None:
        config.speaker_detection_threshold = threshold
    return asd_pb2.DetectActiveSpeakerResponse(config=config)


def keepalive():
    return asd_pb2.DetectActiveSpeakerResponse(keepalive=empty_pb2.Empty())


@pytest.fixture
def video(tmp_path) -> Path:
    """A file spanning several stream chunks, so reassembly is actually tested."""
    path = tmp_path / "derivative.mp4"
    path.write_bytes(b"\x00\x01\x02\x03" * asd.DATA_CHUNK_SIZE + b"tail")
    return path


@pytest.fixture
def audio(tmp_path) -> Path:
    path = tmp_path / "derivative.wav"
    path.write_bytes(b"RIFF" + b"\x07\x06\x05\x04" * asd.DATA_CHUNK_SIZE)
    return path


# --- per-frame result parsing ------------------------------------------------------------


def test_per_frame_evidence_is_parsed_exactly_as_reported(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker()), frame(1, speaker(is_speaking=False))])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.frames == (
        asd.NvidiaActiveSpeakerFrame(
            frame_id=0,
            speakers=(
                asd.NvidiaSpeakerObservation(
                    face_id=0,
                    diarized_speaker_id=0,
                    is_speaking=True,
                    face_detection_confidence=CONFIDENCE,
                    bounding_box=asd.NvidiaBoundingBox(x=10.5, y=20.25, width=64.0, height=128.5),
                ),
            ),
        ),
        asd.NvidiaActiveSpeakerFrame(
            frame_id=1,
            speakers=(
                asd.NvidiaSpeakerObservation(
                    face_id=0,
                    diarized_speaker_id=0,
                    is_speaking=False,
                    face_detection_confidence=CONFIDENCE,
                    bounding_box=asd.NvidiaBoundingBox(x=10.5, y=20.25, width=64.0, height=128.5),
                ),
            ),
        ),
    )


def test_multiple_faces_in_one_frame_are_all_preserved(monkeypatch, video):
    detector = FakeDetector(
        responses=[
            frame(
                7,
                speaker(face_id=0, diarized_speaker_id=1, is_speaking=True),
                speaker(face_id=1, diarized_speaker_id=-1, is_speaking=False, confidence=0.5),
            )
        ]
    )

    result, _ = run_analysis(detector, monkeypatch, video)

    (only_frame,) = result.frames
    assert only_frame.frame_id == 7
    assert [s.face_id for s in only_frame.speakers] == [0, 1]
    # -1 is NVIDIA's "no diarized voice matched to this face"; it must survive untouched.
    assert [s.diarized_speaker_id for s in only_frame.speakers] == [1, -1]
    assert [s.is_speaking for s in only_frame.speakers] == [True, False]
    assert only_frame.speakers[1].face_detection_confidence == 0.5


def test_a_frame_with_no_faces_is_kept_as_evidence(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0), frame(1, speaker())])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert [(f.frame_id, len(f.speakers)) for f in result.frames] == [(0, 0), (1, 1)]


def test_frame_order_is_preserved(monkeypatch, video):
    detector = FakeDetector(responses=[frame(i, speaker(face_id=i)) for i in range(25)])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert [f.frame_id for f in result.frames] == list(range(25))


def test_echoed_threshold_is_recorded(monkeypatch, video):
    detector = FakeDetector(responses=[echoed_config(), frame(0, speaker())])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.speaker_detection_threshold == THRESHOLD


def test_threshold_is_none_when_the_provider_echoes_none(monkeypatch, video):
    detector = FakeDetector(responses=[echoed_config(threshold=None), frame(0, speaker())])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.speaker_detection_threshold is None


def test_keepalive_messages_carry_no_evidence(monkeypatch, video):
    detector = FakeDetector(responses=[keepalive(), frame(0, speaker()), keepalive()])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert len(result.frames) == 1


def test_the_answering_function_travels_with_the_evidence(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert result.function_id == FUNCTION_ID


def test_results_are_frozen(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    result, _ = run_analysis(detector, monkeypatch, video)

    with pytest.raises(Exception):
        result.frames[0].speakers[0].is_speaking = False


# --- what is sent to the provider ---------------------------------------------------------


def test_configuration_leads_the_stream(monkeypatch, video, audio):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, audio_path=audio)

    assert detector.request_kinds[0] == "config"
    assert detector.request_kinds.count("config") == 1


def test_video_and_audio_arrive_intact(monkeypatch, video, audio):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, audio_path=audio)

    assert detector.received_video == video.read_bytes()
    assert detector.received_audio == audio.read_bytes()


def test_every_diarization_segment_is_sent(monkeypatch, video):
    diarization = tuple(
        asd.DiarizationSegment(start_time_ms=i * 10, end_time_ms=i * 10 + 9, speaker_id=i % 3)
        # Spans several batches, so batching is actually tested.
        for i in range(asd.DIARIZATION_BATCH_SIZE * 2 + 7)
    )
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, diarization=diarization)

    assert [
        (s.start_time, s.end_time, s.speaker_id) for s in detector.received_segments
    ] == [(s.start_time_ms, s.end_time_ms, s.speaker_id) for s in diarization]


def test_inputs_are_interleaved_rather_than_sent_one_after_another(monkeypatch, video, audio):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, audio_path=audio)

    # Streaming-mode inference starts before the upload ends, so diarization must not trail
    # the whole video: it has to appear among the first few data messages.
    kinds = detector.request_kinds
    assert kinds[:4] == ["config", "video", "audio", "diarization"]


def test_a_separate_audio_file_is_declared_as_its_own_stream(monkeypatch, video, audio):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, audio_path=audio)

    config = detector.received_config
    assert config.audio_source_config == asd_pb2.AUDIO_SOURCE_CONFIG_SEPARATE_STREAM
    assert config.input_audio_config.encoding == audio_pb2.AUDIO_CODEC_WAV
    assert config.input_video_config.codec == video_pb2.VIDEO_CODEC_H264


def test_without_an_audio_file_the_provider_is_told_to_demux_the_video(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video)

    assert (
        detector.received_config.audio_source_config
        == asd_pb2.AUDIO_SOURCE_CONFIG_EMBEDDED_IN_VIDEO
    )
    assert detector.received_audio == b""


@pytest.mark.parametrize(
    ("suffix", "codec"),
    [
        (".wav", audio_pb2.AUDIO_CODEC_WAV),
        (".mp3", audio_pb2.AUDIO_CODEC_MP3),
        (".opus", audio_pb2.AUDIO_CODEC_OPUS),
        (".WAV", audio_pb2.AUDIO_CODEC_WAV),
    ],
)
def test_audio_codec_is_taken_from_the_file(monkeypatch, video, tmp_path, suffix, codec):
    audio_file = tmp_path / f"track{suffix}"
    audio_file.write_bytes(b"audio")
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, audio_path=audio_file)

    assert detector.received_config.input_audio_config.encoding == codec


def test_threshold_is_sent_only_when_given(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video, speaker_detection_threshold=0.75)

    assert detector.received_config.HasField("speaker_detection_threshold")
    assert detector.received_config.speaker_detection_threshold == 0.75


def test_no_threshold_leaves_the_provider_default_in_place(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video)

    assert not detector.received_config.HasField("speaker_detection_threshold")


def test_credentials_are_sent_as_nvcf_call_metadata(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    run_analysis(detector, monkeypatch, video)

    assert detector.received_metadata["authorization"] == f"Bearer {API_KEY}"
    assert detector.received_metadata["function-id"] == FUNCTION_ID


# --- provider failures --------------------------------------------------------------------


def test_stream_without_any_frame_is_rejected(monkeypatch, video):
    detector = FakeDetector(responses=[echoed_config(), keepalive()])

    with pytest.raises(asd.NvidiaActiveSpeakerInvalidResponse):
        run_analysis(detector, monkeypatch, video)


@pytest.mark.parametrize(
    "code",
    [grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED],
)
def test_rejected_credentials_raise_an_authentication_error(monkeypatch, video, code):
    detector = FakeDetector(abort=(code, "invalid credentials"))

    with pytest.raises(asd.NvidiaActiveSpeakerAuthenticationError):
        run_analysis(detector, monkeypatch, video)


@pytest.mark.parametrize(
    "code",
    [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.RESOURCE_EXHAUSTED],
)
def test_provider_capacity_failures_are_reported_as_unavailable(monkeypatch, video, code):
    detector = FakeDetector(abort=(code, "no capacity"))

    with pytest.raises(asd.NvidiaActiveSpeakerUnavailable):
        run_analysis(detector, monkeypatch, video)


def test_unreachable_provider_is_reported_as_unavailable(monkeypatch, video):
    # Nothing is listening on this port, so the connection attempt fails outright.
    redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(asd.NvidiaActiveSpeakerUnavailable):
        asyncio.run(
            asd.analyze_active_speaker(
                video,
                DIARIZATION,
                api_key=API_KEY,
                function_id=FUNCTION_ID,
                timeout_seconds=5.0,
            )
        )


def test_exceeding_the_deadline_raises_a_timeout(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())], hang=True)

    with pytest.raises(asd.NvidiaActiveSpeakerTimeout):
        run_analysis(detector, monkeypatch, video, timeout_seconds=0.5)


def test_other_provider_failures_stay_generic(monkeypatch, video):
    detector = FakeDetector(abort=(grpc.StatusCode.INTERNAL, "backend exploded"))

    with pytest.raises(asd.NvidiaActiveSpeakerProviderError) as raised:
        run_analysis(detector, monkeypatch, video)

    assert not isinstance(raised.value, asd.NvidiaActiveSpeakerAuthenticationError)
    assert "backend exploded" in str(raised.value)


def test_channel_is_released_after_a_provider_failure(monkeypatch, video):
    detector = FakeDetector(abort=(grpc.StatusCode.UNAUTHENTICATED, "nope"))
    channels: list[RecordingChannel] = []

    with pytest.raises(asd.NvidiaActiveSpeakerAuthenticationError):
        analyze_against(detector, monkeypatch, video, channels)

    assert [channel.closed for channel in channels] == [True]


def test_channel_is_released_after_success(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    _, channels = run_analysis(detector, monkeypatch, video)

    assert [channel.closed for channel in channels] == [True]


# --- local input failures -----------------------------------------------------------------


def test_missing_video_fails_before_any_connection(monkeypatch, tmp_path):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(asd.NvidiaActiveSpeakerLocalFileError):
        asyncio.run(
            asd.analyze_active_speaker(
                tmp_path / "absent.mp4",
                DIARIZATION,
                api_key=API_KEY,
                function_id=FUNCTION_ID,
            )
        )

    assert channels == []


def test_missing_audio_fails_before_any_connection(monkeypatch, video, tmp_path):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(asd.NvidiaActiveSpeakerLocalFileError):
        asyncio.run(
            asd.analyze_active_speaker(
                video,
                DIARIZATION,
                audio_path=tmp_path / "absent.wav",
                api_key=API_KEY,
                function_id=FUNCTION_ID,
            )
        )

    assert channels == []


def test_unreadable_video_mid_stream_is_reported_as_our_failure(monkeypatch, video):
    """A disk that fails after the RPC started must not be blamed on NVIDIA."""

    opened: list = []

    def failing_open(file_path, mode="rb"):
        handle = builtins.open(file_path, mode)
        opened.append(handle)
        if len(opened) > 1:  # the pre-flight check opened it once already
            handle.read = _raise_oserror
        return handle

    def _raise_oserror(*_args):
        raise OSError("disk went away")

    monkeypatch.setattr(asd, "open", failing_open, raising=False)

    detector = FakeDetector(responses=[frame(0, speaker())])

    with pytest.raises(asd.NvidiaActiveSpeakerLocalFileError) as raised:
        run_analysis(detector, monkeypatch, video)

    assert "disk went away" in str(raised.value)


def test_caller_cancellation_is_not_mistaken_for_a_local_failure(monkeypatch, video):
    """The recorded-cause branch must only convert our own failures, never a real cancel."""
    address = f"127.0.0.1:{free_port()}"
    redirect_to(monkeypatch, address)
    detector = FakeDetector(responses=[frame(0, speaker())], hang=True)

    async def scenario():
        server = await serve(detector, address)
        try:
            task = asyncio.ensure_future(
                asd.analyze_active_speaker(
                    video, DIARIZATION, api_key=API_KEY, function_id=FUNCTION_ID
                )
            )
            await asyncio.sleep(0.3)
            task.cancel()
            await task
        finally:
            await server.stop(None)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_unsupported_audio_format_is_rejected_without_contacting_the_provider(
    monkeypatch, video, tmp_path
):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")
    unsupported = tmp_path / "track.aac"
    unsupported.write_bytes(b"aac")

    with pytest.raises(asd.NvidiaActiveSpeakerInputError):
        asyncio.run(
            asd.analyze_active_speaker(
                video,
                DIARIZATION,
                audio_path=unsupported,
                api_key=API_KEY,
                function_id=FUNCTION_ID,
            )
        )

    assert channels == []


def test_empty_diarization_is_rejected_without_contacting_the_provider(monkeypatch, video):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(asd.NvidiaActiveSpeakerInputError):
        asyncio.run(
            asd.analyze_active_speaker(
                video, (), api_key=API_KEY, function_id=FUNCTION_ID
            )
        )

    assert channels == []


@pytest.mark.parametrize(
    "segment",
    [
        asd.DiarizationSegment(start_time_ms=-1, end_time_ms=10, speaker_id=0),
        asd.DiarizationSegment(start_time_ms=0, end_time_ms=-10, speaker_id=0),
        asd.DiarizationSegment(start_time_ms=500, end_time_ms=100, speaker_id=0),
    ],
)
def test_malformed_diarization_is_rejected_without_contacting_the_provider(
    monkeypatch, video, segment
):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")

    with pytest.raises(asd.NvidiaActiveSpeakerInputError):
        asyncio.run(
            asd.analyze_active_speaker(
                video, (segment,), api_key=API_KEY, function_id=FUNCTION_ID
            )
        )

    assert channels == []


# --- credentials --------------------------------------------------------------------------


def test_api_key_never_appears_in_provider_errors(monkeypatch, video):
    detector = FakeDetector(abort=(grpc.StatusCode.UNAUTHENTICATED, "invalid credentials"))

    with pytest.raises(asd.NvidiaActiveSpeakerAuthenticationError) as raised:
        run_analysis(detector, monkeypatch, video)

    assert API_KEY not in str(raised.value)
    assert API_KEY not in repr(raised.value)


def test_api_key_never_appears_in_the_result(monkeypatch, video):
    detector = FakeDetector(responses=[frame(0, speaker())])

    result, _ = run_analysis(detector, monkeypatch, video)

    assert API_KEY not in repr(result)


def test_missing_api_key_is_reported_without_contacting_the_provider(monkeypatch, video):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(asd.NvidiaActiveSpeakerAuthenticationError):
        asyncio.run(asd.analyze_active_speaker(video, DIARIZATION, function_id=FUNCTION_ID))

    assert channels == []


def test_missing_function_id_is_reported_without_contacting_the_provider(monkeypatch, video):
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")
    monkeypatch.delenv("NVIDIA_ASD_FUNCTION_ID", raising=False)

    with pytest.raises(asd.NvidiaActiveSpeakerAuthenticationError):
        asyncio.run(asd.analyze_active_speaker(video, DIARIZATION, api_key=API_KEY))

    assert channels == []


def test_credentials_fall_back_to_the_environment(monkeypatch, video):
    monkeypatch.setenv("NVIDIA_API_KEY", API_KEY)
    monkeypatch.setenv("NVIDIA_ASD_FUNCTION_ID", FUNCTION_ID)

    detector = FakeDetector(responses=[frame(0, speaker())])
    address = f"127.0.0.1:{free_port()}"
    redirect_to(monkeypatch, address)

    async def scenario():
        server = await serve(detector, address)
        try:
            return await asd.analyze_active_speaker(video, DIARIZATION)
        finally:
            await server.stop(None)

    result = asyncio.run(scenario())

    assert result.frames[0].speakers[0].face_detection_confidence == CONFIDENCE
    assert detector.received_metadata["function-id"] == FUNCTION_ID


def test_the_synthetic_video_function_id_is_not_reused(monkeypatch, video):
    """The two NIMs are different deployments behind different function IDs."""
    channels = redirect_to(monkeypatch, f"127.0.0.1:{free_port()}")
    monkeypatch.setenv("NVIDIA_API_KEY", API_KEY)
    monkeypatch.setenv("NVIDIA_SVD_FUNCTION_ID", FUNCTION_ID)
    monkeypatch.delenv("NVIDIA_ASD_FUNCTION_ID", raising=False)

    with pytest.raises(asd.NvidiaActiveSpeakerAuthenticationError):
        asyncio.run(asd.analyze_active_speaker(video, DIARIZATION))

    assert channels == []
