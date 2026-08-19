"""Provider-compatible normalization of validated media (D013).

The uploaded original is a forensic artifact and is never rewritten. When it cannot be
handed to the downstream detector as-is, this module produces a separate derivative in
the canonical shape the NVIDIA Synthetic Video Detector accepts: an MP4 container,
H.264 video, constant frame rate and a broadly decodable pixel format.

Nothing here calls a detector; it only prepares the media a detector could consume.
"""

import asyncio
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.media import MediaMetadata

logger = logging.getLogger(__name__)

FFMPEG_BINARY = "ffmpeg"
FFMPEG_TIMEOUT_SECONDS = 60
DERIVATIVE_TEMP_PREFIX = "deepguard-normalized-"
DERIVATIVE_TEMP_SUFFIX = ".mp4"

# The canonical downstream target. NVIDIA SVD takes MP4/H.264 and does not support
# variable frame rate.
CANONICAL_CONTENT_TYPE = "video/mp4"
CANONICAL_CODEC = "h264"
CANONICAL_PIX_FMT = "yuv420p"

# Rounds each side up to the nearest even number. Even-dimension video pads by zero
# pixels, so the common case is unaffected.
EVEN_DIMENSION_PAD_FILTER = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

HASH_CHUNK_SIZE = 1024 * 1024


class NormalizationError(Exception):
    """The media is real video but could not be transcoded into the canonical form."""


class NormalizationTimeout(NormalizationError):
    """Transcoding outlived the processing limit and was stopped."""


class NormalizationUnavailable(Exception):
    """ffmpeg itself could not be executed — a server problem, not a media problem."""


@dataclass(frozen=True)
class NormalizedMedia:
    """A derivative on local disk, with its own content identity."""

    path: Path
    sha256: str


def needs_normalization(content_type: str, metadata: MediaMetadata) -> bool:
    """Decide whether the original can be sent downstream without transcoding.

    The bypass is deliberately conservative: it is taken only when every fact the
    provider requires is positively known. ffprobe reports one shared demuxer name for
    the whole MOV/MP4 family, so it cannot tell an MP4 from a QuickTime file — the
    admitted `video/mp4` declaration is what distinguishes them, and a MOV is normalized
    rather than assumed compatible. Anything unknown (pixel format, frame timing) counts
    against the bypass, because guessing wrong means shipping media the detector rejects.
    """
    return not (
        content_type == CANONICAL_CONTENT_TYPE
        and metadata.codec_name == CANONICAL_CODEC
        and metadata.pix_fmt == CANONICAL_PIX_FMT
        and metadata.constant_frame_rate
    )


def _format_frame_rate(rate: float) -> str:
    """Render the validated source rate as an explicit constant output rate.

    The source's practical rate is preserved rather than forcing every video to a fixed
    30 fps, which would resample timing for no provider-driven reason.
    """
    return f"{rate:.6f}".rstrip("0").rstrip(".")


async def _run_ffmpeg(source: Path, destination: Path, frame_rate: float) -> None:
    """Transcode the original into the canonical MP4, without a shell.

    Both paths are separate argv entries, so a filename can never be interpreted as
    shell syntax. A transcode that outlives the timeout is killed and reaped rather than
    left running.
    """
    args = (
        "-nostdin",
        "-y",
        "-i", str(source),
        # P1 analyses the primary video stream; audio is carried when it exists.
        "-map", "0:v:0",
        "-map", "0:a:0?",
        # libx264 with yuv420p cannot encode odd dimensions, which is a property of the
        # encoder rather than a defect in the source. One row/column of padding makes an
        # odd side even and leaves already-even video untouched: nothing is scaled or
        # cropped, so the picture itself survives intact.
        "-vf", EVEN_DIMENSION_PAD_FILTER,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", CANONICAL_PIX_FMT,
        # An explicit rate is what makes the output constant frame rate.
        "-r", _format_frame_rate(frame_rate),
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(destination),
    )

    try:
        process = await asyncio.create_subprocess_exec(
            FFMPEG_BINARY,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as error:
        raise NormalizationUnavailable("ffmpeg could not be executed") from error

    try:
        _, stderr = await asyncio.wait_for(
            process.communicate(), timeout=FFMPEG_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        _terminate(process)
        await process.wait()
        raise NormalizationTimeout(
            f"ffmpeg timed out after {FFMPEG_TIMEOUT_SECONDS}s"
        ) from None

    if process.returncode != 0:
        # stderr is diagnostic only: it can quote container internals and the temp path.
        logger.warning(
            "ffmpeg exited with %s: %s",
            process.returncode,
            stderr.decode("utf-8", "replace").strip(),
        )
        raise NormalizationError(f"ffmpeg exited with {process.returncode}")


def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        process.kill()
    except ProcessLookupError:
        # The child raced us to exit; the following wait() still reaps it.
        pass


def _sha256_of(path: Path) -> str:
    """Hash the derivative bytes; the original's hash identifies a different artifact."""
    hasher = hashlib.sha256()
    with open(path, "rb") as derivative:
        while chunk := derivative.read(HASH_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()


async def normalize_to_mp4(source: Path, metadata: MediaMetadata) -> NormalizedMedia:
    """Produce a canonical MP4/H.264 derivative of the staged original.

    The original file is only read. On any failure the half-written derivative is
    removed here, so a failed transcode never leaves a temp file behind.
    """
    descriptor, name = tempfile.mkstemp(
        prefix=DERIVATIVE_TEMP_PREFIX, suffix=DERIVATIVE_TEMP_SUFFIX
    )
    os.close(descriptor)
    destination = Path(name)

    try:
        await _run_ffmpeg(source, destination, metadata.frame_rate)
        return NormalizedMedia(path=destination, sha256=_sha256_of(destination))
    except BaseException:
        try:
            os.unlink(destination)
        except OSError:
            logger.warning("Removing the failed derivative temp file failed.")
        raise
