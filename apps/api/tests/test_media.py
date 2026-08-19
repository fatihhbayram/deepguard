"""Tests for the ffprobe subprocess boundary itself.

The endpoint tests replace `_run_ffprobe`; these exercise it, without needing a real
ffprobe binary or a video fixture.
"""

import asyncio

import pytest

from app import media


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
    monkeypatch.setattr(media, "FFPROBE_TIMEOUT_SECONDS", 0.01)
    process = FakeProcess(hang=True)
    spawned(process)

    with pytest.raises(media.MediaProbeError):
        asyncio.run(media._run_ffprobe("/tmp/slow.mp4"))

    assert process.killed
    assert process.waited


def test_a_child_that_raced_us_to_exit_is_still_reaped(spawned, monkeypatch):
    monkeypatch.setattr(media, "FFPROBE_TIMEOUT_SECONDS", 0.01)
    process = FakeProcess(hang=True)
    process.kill = lambda: (_ for _ in ()).throw(ProcessLookupError())
    spawned(process)

    with pytest.raises(media.MediaProbeError):
        asyncio.run(media._run_ffprobe("/tmp/slow.mp4"))

    assert process.waited


def test_a_missing_ffprobe_binary_is_not_reported_as_bad_media(monkeypatch, tmp_path):
    # Real spawn attempt, so this covers the actual OSError the runtime raises.
    monkeypatch.setattr(media, "FFPROBE_BINARY", str(tmp_path / "no-such-ffprobe"))

    with pytest.raises(media.MediaProbeUnavailable):
        asyncio.run(media._run_ffprobe(tmp_path))
