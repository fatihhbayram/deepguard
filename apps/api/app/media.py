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
    ":format=format_name,duration"
    ":format_tags=major_brand"
)

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


async def _run_ffprobe(path: Path) -> str:
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
        "-show_entries", FFPROBE_ENTRIES,
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
    )


async def probe_media(path: Path) -> MediaMetadata:
    """Validate the staged original as real video and extract its metadata.

    Runs against the byte-for-byte original staged on disk (D013): nothing is
    transcoded, rewritten or re-downloaded from object storage.
    """
    return _parse(await _run_ffprobe(path))
