"""Every deadline this service imposes on an operation it does not control (R1-T3).

The timeouts were always here. What was not here is a way to change one without editing
Python: each bound lived as a literal beside the call it guarded, so a deployment whose
machine transcodes half as fast as the development box had no answer except a code change
and a rebuild. This module is that answer — one place that names the bounds, reads them
from the environment, and hands the same defaults back when nothing is configured.

Three properties are deliberate:

- **the defaults are the previous literals**, to the second. Nothing about an unconfigured
  deployment changes by moving these here, which is what makes the move reviewable: a
  behavioural difference would be a second change hiding inside a refactor;
- **the environment is read per call, never at import.** These are worker-side bounds and
  the worker is a long-lived process, but reading late is what lets a test state a bound
  for one case without reloading a module — and it costs one `os.getenv` on a path that is
  about to spend minutes in ffmpeg or on a gRPC stream;
- **a malformed value raises rather than falling back.** A deployment that wrote
  `DEEPGUARD_FFMPEG_TIMEOUT_SECONDS=15m` meant something by it, and silently running with
  900 instead would leave an operator certain of a bound the process is not keeping.
  `validate()` is how the worker turns that into a refusal at startup rather than a
  surprise on the first job that needs a transcode.

What this module deliberately does not hold is a bound on the analysis as a whole. Every
value here guards one external operation — a subprocess, a socket, an RPC — that can be
stopped cleanly and reported honestly. A ceiling over the whole in-process pipeline could
only be enforced by killing work mid-flight, and the thing that would most often be killed
is local inference holding a model in memory, so there is no clean close to make from it.
The lease in `app.worker` is what bounds a job that stops making progress altogether, and
it does it by declaring the *worker* gone rather than by reaching into it.
"""

import math
import os

# ffprobe, on an upload that has just been staged. Seconds, and short: the probe reads
# container headers rather than decoding, so anything approaching this figure is a file
# that is not going to answer.
FFPROBE_TIMEOUT_VARIABLE = "DEEPGUARD_FFPROBE_TIMEOUT_SECONDS"
DEFAULT_FFPROBE_TIMEOUT_SECONDS = 10.0

# The normalization transcode, in the worker. The reasoning behind fifteen minutes is in
# `app.normalization`, and it is the one bound here most worth a deployment's attention:
# it scales with the machine, not with the service.
NORMALIZATION_TIMEOUT_VARIABLE = "DEEPGUARD_FFMPEG_TIMEOUT_SECONDS"
DEFAULT_NORMALIZATION_TIMEOUT_SECONDS = 900.0

# The ffmpeg call that demuxes analysable audio out of the artifact. Lower than the
# transcode above because it decodes no video, and separate from it because the two are
# different work with different costs — one figure for both would have to be the larger.
AUDIO_EXTRACTION_TIMEOUT_VARIABLE = "DEEPGUARD_AUDIO_EXTRACTION_TIMEOUT_SECONDS"
DEFAULT_AUDIO_EXTRACTION_TIMEOUT_SECONDS = 300.0

# The gRPC deadline on NVIDIA's Synthetic Video Detector, and on the Active Speaker NIM
# beside it. Two variables rather than one: they are separate deployments answering
# separate questions, and a deployment that needs to give one of them longer should not
# have to give both.
NVIDIA_SVD_TIMEOUT_VARIABLE = "DEEPGUARD_NVIDIA_SVD_TIMEOUT_SECONDS"
DEFAULT_NVIDIA_SVD_TIMEOUT_SECONDS = 600.0

NVIDIA_ASD_TIMEOUT_VARIABLE = "DEEPGUARD_NVIDIA_ASD_TIMEOUT_SECONDS"
DEFAULT_NVIDIA_ASD_TIMEOUT_SECONDS = 600.0

# How long one stalled read may hang inside a URL download. It bounds a single socket
# operation rather than the download, which is what yt-dlp offers and what a hung publisher
# actually looks like.
DOWNLOAD_SOCKET_TIMEOUT_VARIABLE = "DEEPGUARD_DOWNLOAD_SOCKET_TIMEOUT_SECONDS"
DEFAULT_DOWNLOAD_SOCKET_TIMEOUT_SECONDS = 30.0


class InvalidTimeout(ValueError):
    """A configured timeout is not a number of seconds this service can act on."""


def _seconds(variable: str, default: float) -> float:
    """Read one bound from the environment, or hand back the default.

    Unset and empty are the same thing and both mean "use the default": a variable set to
    the empty string is what a compose file passing an unset host variable through produces,
    and treating that as a configuration error would fail every deployment that lists the
    variable without setting it.

    Anything else has to parse and has to be a finite, strictly positive number. Zero and
    negatives are refused rather than read as "no limit" — a deployment that wants no limit
    is not something this service offers, and reading `0` as infinity is exactly the kind of
    convention that gets typed by accident. `inf` and `nan` are refused in the same breath
    and for a less obvious reason: `float` accepts both, `nan <= 0` is False, and a bound of
    `nan` would sail through a naive check and then make every `asyncio.wait_for` comparison
    false — a timeout that silently is not one.
    """
    configured = os.getenv(variable, "").strip()
    if not configured:
        return default

    try:
        seconds = float(configured)
    except ValueError:
        raise InvalidTimeout(
            f"{variable} is not a number of seconds: {configured!r}"
        ) from None

    if not math.isfinite(seconds):
        raise InvalidTimeout(
            f"{variable} is not a number of seconds: {configured!r}"
        )

    if seconds <= 0:
        raise InvalidTimeout(f"{variable} must be greater than zero, not {seconds}")

    return seconds


def ffprobe_timeout_seconds() -> float:
    return _seconds(FFPROBE_TIMEOUT_VARIABLE, DEFAULT_FFPROBE_TIMEOUT_SECONDS)


def normalization_timeout_seconds() -> float:
    return _seconds(NORMALIZATION_TIMEOUT_VARIABLE, DEFAULT_NORMALIZATION_TIMEOUT_SECONDS)


def audio_extraction_timeout_seconds() -> float:
    return _seconds(
        AUDIO_EXTRACTION_TIMEOUT_VARIABLE, DEFAULT_AUDIO_EXTRACTION_TIMEOUT_SECONDS
    )


def nvidia_svd_timeout_seconds() -> float:
    return _seconds(NVIDIA_SVD_TIMEOUT_VARIABLE, DEFAULT_NVIDIA_SVD_TIMEOUT_SECONDS)


def nvidia_asd_timeout_seconds() -> float:
    return _seconds(NVIDIA_ASD_TIMEOUT_VARIABLE, DEFAULT_NVIDIA_ASD_TIMEOUT_SECONDS)


def download_socket_timeout_seconds() -> float:
    return _seconds(
        DOWNLOAD_SOCKET_TIMEOUT_VARIABLE, DEFAULT_DOWNLOAD_SOCKET_TIMEOUT_SECONDS
    )


# Every accessor above, so one caller can check the whole set. Listed rather than
# discovered by introspection: a bound that is added and not listed here should be a
# visible omission in a diff, not an invisible one behind a module scan.
ACCESSORS = (
    ffprobe_timeout_seconds,
    normalization_timeout_seconds,
    audio_extraction_timeout_seconds,
    nvidia_svd_timeout_seconds,
    nvidia_asd_timeout_seconds,
    download_socket_timeout_seconds,
)


def validate() -> dict[str, float]:
    """Resolve every bound, raising `InvalidTimeout` on the first one that is malformed.

    Called at worker startup so a typo in a compose file stops the process before it claims
    a job, rather than at whatever point in whatever later analysis first happens to need
    the bound that was mistyped. Returns what resolved, so the worker can log the bounds it
    is actually running under instead of the ones someone believes it read.
    """
    return {accessor.__name__: accessor() for accessor in ACCESSORS}
