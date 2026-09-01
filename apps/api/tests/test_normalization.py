"""Tests for the ffmpeg subprocess boundary itself.

The endpoint tests replace `_run_ffmpeg`; these exercise it, without needing a real
ffmpeg binary or a video fixture.
"""

import asyncio
import hashlib
from pathlib import Path

import pytest

from app import limits, normalization
from app.media import MediaMetadata


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


@pytest.fixture
def spawned(monkeypatch):
    """Capture the argv `_run_ffmpeg` would execute and hand back a fake process."""
    calls = []

    def spawn(process):
        async def create_subprocess_exec(program, *args, **kwargs):
            calls.append((program, args))
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
        return calls

    return spawn


def metadata(**overrides) -> MediaMetadata:
    fields = {
        "format_name": "matroska,webm",
        "major_brand": None,
        "codec_name": "vp9",
        "width": 1920,
        "height": 1080,
        "duration": 12.34,
        "frame_rate": 30.0,
        "pix_fmt": "yuv420p",
        "constant_frame_rate": True,
    }

    return MediaMetadata(**{**fields, **overrides})


def mp4_metadata(**overrides) -> MediaMetadata:
    """Metadata ffprobe reports for a canonical MP4: the shared demuxer, an MP4 brand."""
    return metadata(
        **{
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "major_brand": "mp42",
            "codec_name": "h264",
            **overrides,
        }
    )


def test_canonical_media_needs_no_normalization():
    assert not normalization.needs_normalization(mp4_metadata())


@pytest.mark.parametrize("brand", ["isom", "mp41", "mp42", "iso2", "avc1"])
def test_every_accepted_mp4_brand_needs_no_normalization(brand):
    assert not normalization.needs_normalization(mp4_metadata(major_brand=brand))


def test_quicktime_brand_needs_normalization_even_with_h264():
    # ffprobe reports the same demuxer name for MOV and MP4; the brand inside the file is
    # what separates them, and the detector does not take QuickTime.
    assert normalization.needs_normalization(mp4_metadata(major_brand="qt"))


def test_a_declared_mime_type_cannot_make_a_quicktime_file_compatible():
    # The regression this guards: a real MOV uploaded as `video/mp4`. Nothing about the
    # declaration reaches this decision, so the `qt` brand still decides.
    assert normalization.needs_normalization(mp4_metadata(major_brand="qt"))


def test_missing_container_brand_needs_normalization():
    # No brand is not evidence of an MP4, so the conservative answer is to normalize.
    assert normalization.needs_normalization(mp4_metadata(major_brand=None))


def test_unrecognized_container_brand_needs_normalization():
    assert normalization.needs_normalization(mp4_metadata(major_brand="3gp4"))


def test_unknown_pixel_format_needs_normalization():
    assert normalization.needs_normalization(mp4_metadata(pix_fmt=None))


def test_variable_frame_rate_needs_normalization():
    assert normalization.needs_normalization(mp4_metadata(constant_frame_rate=False))


def test_ffmpeg_is_executed_without_a_shell(spawned, tmp_path):
    source = tmp_path / "clip with spaces.mkv"
    source.write_bytes(b"payload")
    destination = tmp_path / "out.mp4"
    calls = spawned(FakeProcess())

    asyncio.run(normalization._run_ffmpeg(source, destination, 30.0))

    program, args = calls[0]
    assert program == "ffmpeg"
    # Both paths are their own argv entries, so a filename is never shell syntax.
    assert args[args.index("-i") + 1] == str(source)
    assert args[-1] == str(destination)


def test_ffmpeg_targets_the_canonical_output_shape(spawned, tmp_path):
    calls = spawned(FakeProcess())

    asyncio.run(normalization._run_ffmpeg(tmp_path / "in.mkv", tmp_path / "out.mp4", 30.0))

    _, args = calls[0]
    for flag, value in (
        ("-c:v", "libx264"),
        ("-pix_fmt", "yuv420p"),
        ("-c:a", "aac"),
        # An explicit rate is what makes the output constant frame rate.
        ("-r", "30"),
        ("-movflags", "+faststart"),
        # First video stream only; audio is carried when it exists.
        ("-map", "0:v:0"),
    ):
        assert args[args.index(flag) + 1] == value
    assert "0:a:0?" in args


def test_odd_dimensions_are_padded_to_even_without_scaling(spawned, tmp_path):
    # libx264 with yuv420p rejects odd dimensions; padding keeps the picture itself
    # intact, where scaling or cropping would alter the media under analysis.
    calls = spawned(FakeProcess())

    asyncio.run(normalization._run_ffmpeg(tmp_path / "in.mov", tmp_path / "out.mp4", 25.0))

    _, args = calls[0]
    # A single argv entry: the filter expression is never parsed by a shell.
    assert args[args.index("-vf") + 1] == "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    assert not any(argument.startswith(("scale=", "crop=")) for argument in args)


def test_fractional_source_rate_is_passed_through(spawned, tmp_path):
    calls = spawned(FakeProcess())

    asyncio.run(
        normalization._run_ffmpeg(tmp_path / "in.mkv", tmp_path / "out.mp4", 30000 / 1001)
    )

    _, args = calls[0]
    assert args[args.index("-r") + 1] == "29.97003"


def test_nonzero_exit_is_a_normalization_failure(spawned, tmp_path):
    spawned(FakeProcess(returncode=1, stderr=b"Invalid data found"))

    with pytest.raises(normalization.NormalizationError):
        asyncio.run(normalization._run_ffmpeg(tmp_path / "in.mkv", tmp_path / "out.mp4", 30.0))


def test_timeout_kills_and_reaps_the_child_process(spawned, monkeypatch, tmp_path):
    monkeypatch.setenv(limits.NORMALIZATION_TIMEOUT_VARIABLE, "0.01")
    process = FakeProcess(hang=True)
    spawned(process)

    with pytest.raises(normalization.NormalizationTimeout):
        asyncio.run(normalization._run_ffmpeg(tmp_path / "in.mkv", tmp_path / "out.mp4", 30.0))

    assert process.killed
    assert process.waited


def test_a_child_that_raced_us_to_exit_is_still_reaped(spawned, monkeypatch, tmp_path):
    monkeypatch.setenv(limits.NORMALIZATION_TIMEOUT_VARIABLE, "0.01")
    process = FakeProcess(hang=True)
    process.kill = lambda: (_ for _ in ()).throw(ProcessLookupError())
    spawned(process)

    with pytest.raises(normalization.NormalizationTimeout):
        asyncio.run(normalization._run_ffmpeg(tmp_path / "in.mkv", tmp_path / "out.mp4", 30.0))

    assert process.waited


def test_a_missing_ffmpeg_binary_is_not_reported_as_bad_media(monkeypatch, tmp_path):
    # Real spawn attempt, so this covers the actual OSError the runtime raises.
    monkeypatch.setattr(normalization, "FFMPEG_BINARY", str(tmp_path / "no-such-ffmpeg"))

    with pytest.raises(normalization.NormalizationUnavailable):
        asyncio.run(normalization._run_ffmpeg(tmp_path / "in.mkv", tmp_path / "out.mp4", 30.0))


def test_a_failed_transcode_leaves_no_derivative_behind(monkeypatch, tmp_path):
    created = []

    async def failing_ffmpeg(source, destination, frame_rate):
        created.append(Path(destination))
        Path(destination).write_bytes(b"partial")
        raise normalization.NormalizationError("ffmpeg exited with 1")

    monkeypatch.setattr(normalization, "_run_ffmpeg", failing_ffmpeg)

    with pytest.raises(normalization.NormalizationError):
        asyncio.run(normalization.normalize_to_mp4(tmp_path / "in.mkv", metadata()))

    assert created and not created[0].exists()


def test_the_derivative_is_hashed_from_its_own_bytes(monkeypatch, tmp_path):
    async def fake_ffmpeg(source, destination, frame_rate):
        Path(destination).write_bytes(b"derivative-bytes")

    monkeypatch.setattr(normalization, "_run_ffmpeg", fake_ffmpeg)

    derivative = asyncio.run(normalization.normalize_to_mp4(tmp_path / "in.mkv", metadata()))

    try:
        assert derivative.path.read_bytes() == b"derivative-bytes"
        assert derivative.sha256 == hashlib.sha256(b"derivative-bytes").hexdigest()
        assert derivative.path.suffix == ".mp4"
    finally:
        derivative.path.unlink(missing_ok=True)
