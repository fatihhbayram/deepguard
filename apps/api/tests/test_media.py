"""Tests for the ffprobe subprocess boundary itself.

The endpoint tests replace `_run_ffprobe`; these exercise it, without needing a real
ffprobe binary or a video fixture.
"""

import asyncio
import json

import pytest

from app import limits, media


class FakeProcess:
    """A stand-in child process: never exits on its own unless `output` is set."""

    def __init__(self, *, output=b"", stderr=b"", returncode=0, hang=False):
        self._output = output
        self._stderr = stderr
        self._hang = hang
        self.returncode = None if hang else returncode
        self.killed = False
        self.waited = False

    async def communicate(self):
        if self._hang:
            await asyncio.Event().wait()
        return self._output, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        self.returncode = -9
        return self.returncode


@pytest.fixture
def spawned(monkeypatch):
    """Capture the argv `_run_ffprobe` would execute and hand back a fake process."""
    calls = []

    def spawn(process):
        async def create_subprocess_exec(program, *args, **kwargs):
            calls.append((program, args))
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
        return calls

    return spawn


def test_ffprobe_is_executed_without_a_shell(spawned, tmp_path):
    path = tmp_path / "clip with spaces.mp4"
    path.write_bytes(b"payload")
    calls = spawned(FakeProcess(output=b"{}"))

    asyncio.run(media._run_ffprobe(path))

    program, args = calls[0]
    assert program == "ffprobe"
    # The path is its own argv entry, so a filename can never become shell syntax.
    assert args[-1] == str(path)
    assert "-of" in args and "json" in args


def test_nonzero_exit_is_invalid_media(spawned):
    spawned(FakeProcess(returncode=1, stderr=b"moov atom not found"))

    with pytest.raises(media.MediaProbeError):
        asyncio.run(media._run_ffprobe("/tmp/fake.mp4"))


def test_timeout_kills_and_reaps_the_child_process(spawned, monkeypatch):
    monkeypatch.setenv(limits.FFPROBE_TIMEOUT_VARIABLE, "0.01")
    process = FakeProcess(hang=True)
    spawned(process)

    with pytest.raises(media.MediaProbeError):
        asyncio.run(media._run_ffprobe("/tmp/slow.mp4"))

    assert process.killed
    assert process.waited


def test_a_child_that_raced_us_to_exit_is_still_reaped(spawned, monkeypatch):
    monkeypatch.setenv(limits.FFPROBE_TIMEOUT_VARIABLE, "0.01")
    process = FakeProcess(hang=True)
    process.kill = lambda: (_ for _ in ()).throw(ProcessLookupError())
    spawned(process)

    with pytest.raises(media.MediaProbeError):
        asyncio.run(media._run_ffprobe("/tmp/slow.mp4"))

    assert process.waited


def test_the_container_brand_is_requested_from_ffprobe(spawned, tmp_path):
    # Without this entry the response carries no container evidence at all, and the
    # bypass decision silently falls back to trusting nothing.
    calls = spawned(FakeProcess(output=b"{}"))

    asyncio.run(media._run_ffprobe(tmp_path / "clip.mp4"))

    _, args = calls[0]
    assert "format_tags=major_brand" in args[args.index("-show_entries") + 1]


def probe_json(*, tags=None, **format_extra) -> str:
    stream = {
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "duration": "12.34",
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
    }
    container = {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.34", **format_extra}
    if tags is not None:
        container["tags"] = tags

    return json.dumps({"streams": [stream], "format": container})


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ({"major_brand": "mp42"}, "mp42"),
        # Brands are four bytes, space padded — QuickTime's is literally `qt  `.
        ({"major_brand": "qt  "}, "qt"),
        ({"major_brand": "MP42"}, "mp42"),
        ({"major_brand": ""}, None),
        ({"major_brand": "   "}, None),
        ({"major_brand": 42}, None),
        # An ISOBMFF file whose tags carry no brand, and a container with no tags at all.
        ({"minor_version": "512"}, None),
        (None, None),
    ],
)
def test_the_container_brand_is_extracted_from_the_format_tags(reported, expected):
    assert media._parse(probe_json(tags=reported)).major_brand == expected


def test_an_unparseable_tags_block_is_not_treated_as_a_brand():
    # ffprobe would not emit this, but the parser treats its output as untrusted input.
    assert media._parse(probe_json(tags="mp42")).major_brand is None


def test_a_missing_ffprobe_binary_is_not_reported_as_bad_media(monkeypatch, tmp_path):
    # Real spawn attempt, so this covers the actual OSError the runtime raises.
    monkeypatch.setattr(media, "FFPROBE_BINARY", str(tmp_path / "no-such-ffprobe"))

    with pytest.raises(media.MediaProbeUnavailable):
        asyncio.run(media._run_ffprobe(tmp_path))
