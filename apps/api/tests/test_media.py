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


# Display rotation. A phone records in landscape and writes a display matrix asking for a
# quarter turn, so the coded picture and the displayed one are different shapes. What is
# under test here is only that the rotation is *read* — nothing in this module acts on it,
# and the geometry a detector receives is measured elsewhere, off the artifact itself.


def rotation_probe_json(**stream) -> bytes:
    """One ffprobe document with the fields `_parse` insists on, plus whatever is asked."""
    return json.dumps(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                    "duration": "5.000000",
                    **stream,
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "5.000000",
                "tags": {"major_brand": "isom"},
            },
        }
    ).encode()


def test_media_with_no_rotation_recorded_reports_zero(spawned):
    spawned(FakeProcess(output=rotation_probe_json()))

    assert asyncio.run(media.probe_media("clip.mp4")).display_rotation == 0


@pytest.mark.parametrize(
    ("matrix_rotation", "expected"),
    [
        # ffprobe prints the display matrix counter-clockwise; this column is clockwise, so
        # the two are negations of each other. A quarter turn one way is three the other.
        (90, 270),
        (-90, 90),
        (-180, 180),
        (180, 180),
        (0, 0),
    ],
)
def test_the_display_matrix_is_read_as_clockwise_degrees(spawned, matrix_rotation, expected):
    spawned(
        FakeProcess(
            output=rotation_probe_json(
                side_data_list=[
                    {"side_data_type": "Display Matrix", "rotation": matrix_rotation}
                ]
            )
        )
    )

    assert asyncio.run(media.probe_media("clip.mp4")).display_rotation == expected


def test_the_legacy_rotate_tag_is_read_when_there_is_no_display_matrix(spawned):
    """Older media carries only the QuickTime tag, and that one is already clockwise."""
    spawned(FakeProcess(output=rotation_probe_json(tags={"rotate": "270"})))

    assert asyncio.run(media.probe_media("clip.mp4")).display_rotation == 270


def test_the_display_matrix_wins_over_the_tag(spawned):
    """A file carrying both states one rotation twice, in two conventions.

    The matrix is the authoritative record, and reading the tag instead would invert the
    answer on exactly the media most likely to carry both.
    """
    spawned(
        FakeProcess(
            output=rotation_probe_json(
                tags={"rotate": "270"},
                side_data_list=[{"side_data_type": "Display Matrix", "rotation": 90}],
            )
        )
    )

    assert asyncio.run(media.probe_media("clip.mp4")).display_rotation == 270


@pytest.mark.parametrize(
    "unusable",
    [
        {"tags": {"rotate": "not a number"}},
        {"tags": {"rotate": "45"}},
        {"side_data_list": [{"rotation": "90"}]},
        {"side_data_list": [{"rotation": float("nan")}]},
        {"side_data_list": "not a list"},
        {"side_data_list": [{"side_data_type": "Stereo 3D"}]},
    ],
)
def test_unreadable_rotation_is_zero_rather_than_a_rejection(spawned, unusable):
    """Real video is never refused over a field no detector reads.

    Zero here means "no rotation is recorded", and nothing downstream turns a picture on
    the strength of it — so the cost of an odd display matrix is one reporting field, not a
    rejected upload.
    """
    spawned(FakeProcess(output=rotation_probe_json(**unusable)))

    assert asyncio.run(media.probe_media("clip.mp4")).display_rotation == 0


def test_rotation_is_asked_for_in_the_same_probe(spawned):
    """One ffprobe call, not a second one: both places rotation lives are in the entries."""
    calls = spawned(FakeProcess(output=rotation_probe_json()))

    asyncio.run(media.probe_media("clip.mp4"))

    entries = calls[0][1][calls[0][1].index("-show_entries") + 1]
    assert "stream_tags=rotate" in entries
    assert "stream_side_data_list" in entries


# Dimension probing. The narrow probe that measures the artifact a detector is handed.


def test_dimensions_are_measured_off_the_file(spawned):
    spawned(FakeProcess(output=json.dumps({"streams": [{"width": 1080, "height": 1920}]}).encode()))

    assert asyncio.run(media.probe_dimensions("derivative.mp4")) == (1080, 1920)


def test_the_dimension_probe_asks_for_nothing_else(spawned):
    """A derivative this service just wrote is not in question; only its shape is."""
    calls = spawned(
        FakeProcess(output=json.dumps({"streams": [{"width": 2, "height": 2}]}).encode())
    )

    asyncio.run(media.probe_dimensions("derivative.mp4"))

    entries = calls[0][1][calls[0][1].index("-show_entries") + 1]
    assert entries == "stream=width,height"


@pytest.mark.parametrize(
    "document",
    [
        {"streams": []},
        {"streams": [{"width": 1920}]},
        {"streams": [{"width": 0, "height": 1080}]},
        {"streams": [{"width": "1920", "height": "1080"}]},
    ],
)
def test_a_file_with_no_usable_dimensions_is_a_probe_error(spawned, document):
    spawned(FakeProcess(output=json.dumps(document).encode()))

    with pytest.raises(media.MediaProbeError):
        asyncio.run(media.probe_dimensions("derivative.mp4"))
