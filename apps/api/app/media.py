"""ffprobe-backed validation of admitted media.

This module answers one question: are the uploaded bytes a real video, and what does
downstream processing need to know about them? It deliberately does not decide whether
the media is compatible with any particular detector provider — that decision belongs to
the normalization task (D013).
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from app.limits import ffprobe_timeout_seconds

logger = logging.getLogger(__name__)

FFPROBE_BINARY = "ffprobe"

# One ceiling for media entering DeepGuard, whatever door it comes through — a multipart
# upload or a URL download. It lives here, beside the other facts about admitted media,
# because both the upload route and the downloader need it and neither should have to
# import the other to get it.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Only the fields DeepGuard currently needs; full metadata dumps are both slower and a
# larger parsing surface than this task requires.
FFPROBE_ENTRIES = (
    "stream=codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,duration"
    # Rotation is recorded in two different places by two different generations of muxer,
    # and a phone video can carry either — so both are asked for. See
    # `_parse_display_rotation` for how they are reconciled.
    ":stream_tags=rotate"
    ":stream_side_data_list"
    ":format=format_name,duration"
    ":format_tags=major_brand"
)

# The narrowest probe this module runs: what shape is the picture in these bytes. Used
# against a derivative, where nothing else about the file is in question — it was written
# by our own ffmpeg call moments earlier — so asking for codec, duration or frame timing
# would only widen the parsing surface for facts no caller reads.
FFPROBE_DIMENSION_ENTRIES = "stream=width,height"

# Two rates reported as the same value is the evidence ffprobe offers for constant frame
# timing; anything else is treated as unknown rather than assumed constant.
FRAME_RATE_EQUALITY_TOLERANCE = 1e-6


class MediaProbeError(Exception):
    """The media was admitted but is not usable video."""


class MediaProbeUnavailable(Exception):
    """ffprobe itself could not be executed — a server problem, not a media problem."""


@dataclass(frozen=True)
class MediaMetadata:
    """The minimum a downstream normalization/detection step needs about the original."""

    format_name: str
    # The ISO base media file format brand written into the file itself — `mp42` for an
    # MP4, `qt  ` for a QuickTime file. `format_name` reports one shared demuxer for the
    # whole MOV/MP4 family, so this is the only evidence in the bytes that separates
    # them. None whenever the container carries no brand, which is every non-ISOBMFF
    # format and any ISOBMFF file with an absent or unreadable tag.
    major_brand: str | None
    codec_name: str
    width: int
    height: int
    duration: float
    frame_rate: float
    pix_fmt: str | None
    constant_frame_rate: bool
    # How the container says this picture should be turned for display, in degrees
    # clockwise: 0, 90, 180 or 270. `0` for the overwhelming majority of media, and for
    # anything whose rotation could not be read — an absent display matrix is not evidence
    # of anything but absence, and the picture is shown as encoded either way.
    #
    # Recorded, not acted on. Nothing here rotates anything and nothing downstream reads
    # this to work out what a detector saw: when the media is transcoded, ffmpeg bakes the
    # rotation into the derivative's real geometry, and that geometry is probed off the
    # derivative itself rather than inferred from this number. This column exists so a
    # report can say *why* an original encoded 1920x1080 was analysed as 1080x1920 — it is
    # an explanation of a measured divergence, never the source of one.
    display_rotation: int = 0


async def _run_ffprobe(path: Path, entries: str = FFPROBE_ENTRIES) -> str:
    """Execute ffprobe against the original file and return its stdout.

    No shell is involved: the path is passed as a separate argv entry, so a filename can
    never be interpreted as shell syntax. A probe that outlives the timeout is killed and
    reaped rather than left running.

    The bound is read per call from `app.limits` (R1-T3), so a deployment can widen it
    without a rebuild. It is read once here and carried into the failure message, so what a
    timeout reports is the figure it was actually judged against.
    """
    timeout_seconds = ffprobe_timeout_seconds()

    args = (
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", entries,
        "-of", "json",
        str(path),
    )

    try:
        process = await asyncio.create_subprocess_exec(
            FFPROBE_BINARY,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise MediaProbeUnavailable("ffprobe could not be executed") from error

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        _terminate(process)
        await process.wait()
        raise MediaProbeError(f"ffprobe timed out after {timeout_seconds}s") from None

    if process.returncode != 0:
        # stderr is diagnostic only: it can quote container internals and the temp path.
        logger.warning(
            "ffprobe exited with %s: %s",
            process.returncode,
            stderr.decode("utf-8", "replace").strip(),
        )
        raise MediaProbeError(f"ffprobe exited with {process.returncode}")

    return stdout.decode("utf-8", "replace")


def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        process.kill()
    except ProcessLookupError:
        # The child raced us to exit; the following wait() still reaps it.
        pass


def _parse_duration(value: object) -> float | None:
    """Accept only a finite, strictly positive duration; anything else is unusable."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None

    try:
        duration = float(value)
    except ValueError:
        return None

    if not math.isfinite(duration) or duration <= 0:
        return None

    return duration


def _parse_frame_rate(value: object) -> float | None:
    """Parse an ffprobe frame rate, which is normally a rational such as `30000/1001`.

    Parsed arithmetically rather than evaluated: ffprobe output is untrusted input, and
    `0/0` for "unknown" is a value this has to reject, not crash on.
    """
    if not isinstance(value, str):
        return None

    numerator, separator, denominator = value.partition("/")
    try:
        rate = int(numerator) / int(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return None

    if not math.isfinite(rate) or rate <= 0:
        return None

    return rate


def _is_constant_frame_rate(average: float | None, base: float | None) -> bool:
    """Decide whether the source can be treated as constant frame rate.

    ffprobe reports the averaged rate and the container's base rate separately. They
    agree for constant-rate media and diverge once frame timing varies, so equality is
    the narrow evidence available here. A missing or unknown rate is not evidence of
    anything — it stays False, which makes the caller normalize instead of assuming.
    """
    if average is None or base is None:
        return False

    return math.isclose(average, base, rel_tol=FRAME_RATE_EQUALITY_TOLERANCE)


def _parse_major_brand(tags: object) -> str | None:
    """Extract the container's major brand from the format tags, if it declares one.

    Brands are four bytes, space padded (`qt  `), and their case is not guaranteed by
    every muxer, so the value is trimmed and lower-cased into the token callers compare
    against. Anything missing or not a usable string returns None, which callers must
    read as "the container did not prove what it is" rather than as any particular
    format.
    """
    if not isinstance(tags, dict):
        return None

    brand = tags.get("major_brand")
    if not isinstance(brand, str):
        return None

    return brand.strip().lower() or None


def _parse_display_rotation(stream: object) -> int:
    """Read the container's display rotation, in degrees clockwise, from either place it lives.

    ffprobe reports rotation twice, in two conventions that are the negation of each other,
    and which of them a file carries depends on what muxed it:

      * `side_data_list` holds the ISOBMFF display matrix, and ffprobe prints its `rotation`
        as degrees **counter-clockwise**. This is the authoritative record and the one a
        modern phone writes.
      * `tags.rotate` is the legacy QuickTime string, in degrees **clockwise**.

    A file carrying both reports them consistently — a display matrix of `90` sits beside a
    `rotate` tag of `270` — so the matrix is preferred and the tag is the fallback for older
    media that has only the tag. Both are normalized to clockwise degrees in `[0, 360)`,
    which is the `rotate` convention, so one column means one thing whichever source it came
    from.

    Anything unreadable, absent, or not a right-angle multiple returns `0`. This is not a
    guess dressed up as a fact: `0` here means "no rotation is recorded", nothing downstream
    rotates anything on the strength of it, and the geometry that actually matters is probed
    off the artifact rather than computed from this. Refusing media over an odd display
    matrix would reject real video for a field no detector reads.
    """
    if not isinstance(stream, dict):
        return 0

    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if not isinstance(entry, dict):
                continue

            rotation = entry.get("rotation")
            if isinstance(rotation, (int, float)) and not isinstance(rotation, bool):
                # Counter-clockwise in the matrix, clockwise in this column.
                return _as_right_angle(-rotation)

    tags = stream.get("tags")
    if isinstance(tags, dict):
        rotation = tags.get("rotate")
        if isinstance(rotation, str):
            try:
                # Already clockwise; only the string needs undoing.
                return _as_right_angle(float(rotation))
            except ValueError:
                return 0

    return 0


def _as_right_angle(degrees: float) -> int:
    """Fold a rotation onto one of the four right angles, or `0` if it is not one of them."""
    if not math.isfinite(degrees):
        return 0

    normalized = round(degrees) % 360
    return normalized if normalized in (0, 90, 180, 270) else 0


def _parse_dimension(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return None

    return value


def _parse(output: str) -> MediaMetadata:
    """Turn ffprobe JSON into metadata, rejecting anything downstream cannot rely on."""
    try:
        probe = json.loads(output)
    except json.JSONDecodeError as error:
        raise MediaProbeError("ffprobe returned malformed JSON") from error

    if not isinstance(probe, dict):
        raise MediaProbeError("ffprobe returned an unexpected JSON document")

    streams = probe.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise MediaProbeError("no video stream")

    # `-select_streams v:0` already narrowed this to the primary video stream; P1 needs
    # no multi-stream selection policy.
    stream = streams[0]
    container = probe.get("format") if isinstance(probe.get("format"), dict) else {}

    codec_name = stream.get("codec_name")
    if not isinstance(codec_name, str) or not codec_name:
        raise MediaProbeError("missing codec_name")

    format_name = container.get("format_name")
    if not isinstance(format_name, str) or not format_name:
        raise MediaProbeError("missing format_name")

    width = _parse_dimension(stream.get("width"))
    height = _parse_dimension(stream.get("height"))
    if width is None or height is None:
        raise MediaProbeError("invalid video dimensions")

    # Some containers carry the duration per stream, others only on the format.
    duration = _parse_duration(stream.get("duration")) or _parse_duration(
        container.get("duration")
    )
    if duration is None:
        raise MediaProbeError("no usable duration")

    # r_frame_rate is the narrow fallback for containers that report avg_frame_rate as
    # the unknown `0/0`.
    average_frame_rate = _parse_frame_rate(stream.get("avg_frame_rate"))
    base_frame_rate = _parse_frame_rate(stream.get("r_frame_rate"))
    frame_rate = average_frame_rate or base_frame_rate
    if frame_rate is None:
        raise MediaProbeError("no usable frame rate")

    # An unreported pixel format is not a reason to reject genuine video; it only means
    # compatibility cannot be established from it.
    pix_fmt = stream.get("pix_fmt")

    return MediaMetadata(
        format_name=format_name,
        # A container without a readable brand is not rejected — plenty of real video
        # carries none. It only means compatibility cannot be established from it.
        major_brand=_parse_major_brand(container.get("tags")),
        codec_name=codec_name,
        width=width,
        height=height,
        duration=duration,
        frame_rate=frame_rate,
        pix_fmt=pix_fmt if isinstance(pix_fmt, str) and pix_fmt else None,
        constant_frame_rate=_is_constant_frame_rate(average_frame_rate, base_frame_rate),
        display_rotation=_parse_display_rotation(stream),
    )


def _parse_dimensions(output: str) -> tuple[int, int]:
    """Pull just the picture size out of a narrow probe, rejecting anything unusable."""
    try:
        probe = json.loads(output)
    except json.JSONDecodeError as error:
        raise MediaProbeError("ffprobe returned malformed JSON") from error

    if not isinstance(probe, dict):
        raise MediaProbeError("ffprobe returned an unexpected JSON document")

    streams = probe.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise MediaProbeError("no video stream")

    width = _parse_dimension(streams[0].get("width"))
    height = _parse_dimension(streams[0].get("height"))
    if width is None or height is None:
        raise MediaProbeError("invalid video dimensions")

    return width, height


async def probe_media(path: Path) -> MediaMetadata:
    """Validate the staged original as real video and extract its metadata.

    Runs against the byte-for-byte original staged on disk (D013): nothing is
    transcoded, rewritten or re-downloaded from object storage.
    """
    return _parse(await _run_ffprobe(path))


async def probe_dimensions(path: Path) -> tuple[int, int]:
    """Measure the picture size of the file at `path`, as it is encoded in those bytes.

    The point of this function is that it measures rather than reasons. The dimensions a
    detector sees are a property of the artifact handed to it, and for a transcoded
    derivative they are not the original's: ffmpeg applies the display matrix, so a phone
    video encoded 1920x1080 with a quarter turn is written out as a real 1080x1920 picture,
    and the even-dimension pad can move a side by a pixel besides. Deriving that from the
    original's columns would mean reimplementing ffmpeg's geometry here and hoping the two
    stayed in agreement; probing the file cannot drift from it, because it *is* the file.

    Returns the coded width and height. Raises `MediaProbeError` if the file does not
    contain a readable video stream and `MediaProbeUnavailable` if ffprobe cannot be run —
    both of which callers are expected to treat as "this fact was not recorded", not as a
    finding about the media.
    """
    return _parse_dimensions(
        await _run_ffprobe(path, entries=FFPROBE_DIMENSION_ENTRIES)
    )
