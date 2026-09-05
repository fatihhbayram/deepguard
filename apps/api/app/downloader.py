"""Turn a public video URL into an ordinary local file (P10-T1).

Everything downstream of this module is unchanged. A URL becomes a file on disk with a size
and a name, which is exactly what an upload already is by the time `accept_upload` sees one,
so the analysis schema, the worker and the detectors learn nothing new — a downloaded video
is a video. That is the whole point of doing this here: P10 adds a way in, not a second kind
of media.

The hard part is not the download. It is that the URL comes from outside, and a server that
fetches a URL on a caller's behalf is a request forgery engine unless something stops it.
DeepGuard's API container can reach PostgreSQL, MinIO and whatever else shares its Docker
network; `http://minio:9000`, `http://169.254.169.254/` and `http://127.0.0.1:8000/` are all
things a caller must never be able to make this process fetch. Three separate guards answer
that, and they are separate because each one catches what the others cannot:

1. the scheme is checked, so nothing but `http` and `https` is ever attempted at all;
2. the URL's host is resolved and every address it resolves to is checked, which refuses the
   obvious attempt with a clear error before anything is fetched;
3. every name resolution made *during* the download is checked again, at the socket layer,
   which is the only thing that catches a redirect into a private address or a DNS record
   that answers publicly once and privately a moment later.

Guard 2 without guard 3 is security theatre — the check would happen against one resolution
and the connection against another. Guard 3 without guard 2 would work but would report a
blocked address as an opaque download failure, so both are kept.

yt-dlp is used as a library. There is no subprocess and no command string, so the URL is a
Python argument the whole way down and there is no shell for a crafted one to reach.
"""

import ipaddress
import logging
import shutil
import socket
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yt_dlp

from app.limits import download_socket_timeout_seconds
from app.media import MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

# The only two schemes worth allowing. Everything else a URL parser will happily accept is a
# way to reach something that is not a public web server: `file://` reads the container's
# filesystem, `ftp://` and `data:` are neither wanted nor tested, and yt-dlp understands
# enough of them that leaving the door open would be a decision nobody made.
ALLOWED_SCHEMES = frozenset({"http", "https"})

# One ceiling for media entering DeepGuard, whatever door it comes through. Imported rather
# than restated: a URL download allowed to be larger than an upload would be a limit with a
# hole in it, and two constants would drift the first time either was tuned.
MAX_DOWNLOAD_BYTES = MAX_UPLOAD_BYTES

TEMP_DIR_PREFIX = "deepguard-download-"

# What yt-dlp is asked for: one already-muxed file, preferring MP4. Deliberately *not* the
# usual "best video + best audio", which downloads two streams and merges them — that is a
# transcode, the worker already owns transcoding (P4-F2), and doing it here would produce a
# file whose bytes no longer match anything the source served. Provenance is read off these
# bytes later, so what is fetched should be what was published.
#
# This is what every extractor but one gets, and R7-T1 leaves it exactly as it was.
MEDIA_FORMAT = "best[ext=mp4]/best[ext=mov]/best"

# YouTube is the exception, and this is the second time it has had to be. Its player clients
# answer with DASH and HLS: every format they list is video-only or audio-only, so
# `MEDIA_FORMAT` matches nothing and the extractor refuses with "Requested format is not
# available". That is not a bot block and no credential fixes it; it is a catalogue that
# contains nothing of the shape that selector asks for.
#
# P10 answered it by naming the `android` player client, which still lists itag 18 — one
# progressive H.264/AAC MP4, already muxed, nothing to merge. That worked, and it capped every
# YouTube acquisition at 360p, because 360p is the only resolution that client offers in a
# single file. That frame is then the input the face and lip detectors have to work from,
# which is a worse forensic outcome than the byte-preservation rule was buying.
#
# So YouTube gets its own selector instead: the best H.264 video stream at 1080p or below and
# the best AAC audio stream, merged locally by ffmpeg into one MP4. The muxed chain is kept
# behind it, so a YouTube URL that does offer a single file still takes it untouched.
#
# 1080p is a ceiling rather than a preference (`height<=1080`, not `height<=?1080`) because
# `MAX_DOWNLOAD_BYTES` is 100 MiB — a 2160p stream would be fetched only to be refused by the
# size check — and because every rung below it is always offered beside it.
#
# H.264 is asked for by name (`vcodec^=avc1`) rather than left to `[ext=mp4]`, which is not
# the same thing: YouTube also publishes AV1 inside MP4, and it ranks above H.264, so the
# plain extension filter selects `av01` on most videos. That is a codec the detectors were
# never measured against and whose decode support depends on which ffmpeg each of them is
# linked to. The 360p path this replaces served H.264/AAC, so asking for H.264/AAC at a
# larger frame size changes one thing and not two. A second branch drops the codec
# requirement if a video genuinely has no H.264 rendition, since a video that decodes is
# worth more than a preference.
#
# WHAT THIS COSTS, STATED PLAINLY: a merged acquisition is not the bytes YouTube served.
# There is no single file for it to be a copy of; the publisher never produced one. The
# artifact on disk is something this service assembled from two streams it fetched, so it is
# evidence of what was acquired and not a byte-for-byte copy of a published file.
# `DownloadedMedia.assembled` records which of the two happened. Nothing may read
# authenticity, provenance or tampering off the merge in either direction — that ffmpeg wrote
# the container is a fact about DeepGuard's acquisition, not about the video.
#
# Scoped to YouTube by hostname alone. No other extractor's behaviour changes and no provider
# abstraction is introduced; the day a second site needs this, the rule of three applies.
YOUTUBE_MEDIA_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]"
    "/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
    "/best[ext=mp4]/best"
)

# The hosts that mean YouTube, matched exactly or as a parent domain — so `www.`, `m.` and
# `music.` are covered without listing them, and `notyoutube.com` is not.
#
# Read off the submitted URL rather than from yt-dlp's chosen extractor, because the format
# has to be decided before anything is extracted. That is sound here because this is a
# quality decision and not a security one: no guard consults it, and a host that talked its
# way into this branch would get a format string it cannot satisfy, not access to anything.
YOUTUBE_HOSTS = frozenset({"youtube.com", "youtu.be", "youtube-nocookie.com"})

# How long a single stalled read may hang before the download gives up lives in `app.limits`
# as `DEFAULT_DOWNLOAD_SOCKET_TIMEOUT_SECONDS`, unchanged and overridable from the
# environment (R1-T3).

# `live_status` values that mean there is no finished file to fetch. A stream that has not
# started has nothing; one that is running has no end, and yt-dlp would sit there recording
# it until the disk filled.
LIVE_STATUSES = frozenset({"is_live", "is_upcoming", "post_live"})


class DownloadError(Exception):
    """A URL could not be turned into local media."""


class UnsupportedUrl(DownloadError):
    """Not a URL this service will attempt at all — the scheme or the shape is wrong."""


class BlockedAddress(DownloadError):
    """The URL points somewhere this server must not be made to fetch."""


class LiveStreamRejected(DownloadError):
    """The URL is a live or unstarted stream, so there is no finished media behind it."""


class MediaTooLarge(DownloadError):
    """The media behind the URL is larger than DeepGuard accepts."""


class DownloadUnavailable(DownloadError):
    """The URL could not be fetched: the site refused, the extractor failed, the network."""


@dataclass(frozen=True)
class DownloadedMedia:
    """A downloaded video as the ingestion pipeline should see it: a file, named and sized.

    Nothing here records where it came from. That is not an oversight — attributing an
    analysis to a source URL is a column in a table this task is forbidden to touch, and
    inventing a field for it now would be a schema decision made in the wrong place. The
    caller has the URL it passed in.

    Valid only inside the `download` block that produced it. The file is deleted on the way
    out, so anything that needs to survive must be copied or stored before then.
    """

    path: Path
    filename: str
    size_bytes: int
    # Whether this file was assembled here rather than served. False means the source handed
    # over one already-muxed file and these are its bytes; True means the selected formats
    # were a separate video stream and a separate audio stream, and ffmpeg wrote the container
    # on this machine (R7-T1, YouTube only).
    #
    # It is a fact about the acquisition and nothing more. A merged file is not a tampered
    # one and an unmerged file is not an authenticated one, so no detector, verdict or
    # provenance claim may be derived from this flag in either direction. What it exists for
    # is to stop the opposite mistake: describing an assembled artifact as a byte-for-byte
    # copy of something a publisher served, which for a DASH/HLS source is not true and never
    # could be.
    assembled: bool = False


def is_public_address(address: str) -> bool:
    """Whether an IP is one a caller may point this server at.

    Stated as everything that is *not* allowed rather than as an allowlist of public ranges,
    because the set worth refusing is the one that keeps growing: loopback reaches this
    container, private ranges reach PostgreSQL and MinIO on the Docker network, link-local
    reaches the cloud metadata endpoint at 169.254.169.254, and reserved space is whatever
    IANA has not decided about yet.

    The IPv6 disguises — `::ffff:127.0.0.1`, the 6to4 form `2002:7f00:1::`, Teredo — are
    handled by `ipaddress` itself, which reads the embedded IPv4 address and answers about
    that. This function does not unwrap them a second time, because code that duplicates what
    the standard library already does is code that can disagree with it. What it does instead
    is pin the behaviour: every one of those forms is a test in `test_downloader.py`, so a
    Python that ever stopped folding them would fail the suite rather than open a bypass.

    The properties are listed out even though `is_private` subsumes most of them, so a reader
    can check this against a threat without first knowing which ranges Python folds into
    which property.
    """
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        # Not an address at all. Nothing that cannot be parsed gets the benefit of the doubt.
        return False

    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


# --- the socket-layer guard ---------------------------------------------------------------
#
# Checking the URL before the download proves something about one moment. The connection
# happens later, possibly to a different address: a redirect chooses a new host, and a DNS
# record can answer publicly on the first lookup and privately on the second. What is needed
# is a check at the moment of resolution, every time, which is what this is.
#
# `socket.getaddrinfo` is where every one of those resolutions lands — `http.client` reaches
# it through `socket.create_connection`, and an IP literal goes through it too, so a redirect
# straight to `http://127.0.0.1/` is caught by the same code as a hostname.

_guarded = threading.local()
_WRAPPED = "_deepguard_ssrf_guard"


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    """`socket.getaddrinfo`, refusing non-public answers while a download is in progress.

    Inert unless this thread is inside `public_destinations_only`. That matters: the same
    process talks to PostgreSQL and MinIO on private addresses constantly, and a guard that
    applied everywhere would break the application it is protecting.
    """
    results = _real_getaddrinfo(host, port, *args, **kwargs)

    if not getattr(_guarded, "active", False):
        return results

    for *_, sockaddr in results:
        if not is_public_address(sockaddr[0]):
            logger.warning(
                "Refused a download connection to %r, which resolved to a non-public "
                "address.",
                host,
            )
            raise BlockedAddress(f"{host} resolves to an address that is not public.")

    return results


# Installed at import, and permanently. It does nothing outside a guarded thread, so leaving
# it in place costs one attribute lookup per resolution and avoids the race that installing
# and removing a patch around every download would have between two concurrent ones.
#
# The sentinel carries the *true* `getaddrinfo` on the wrapper itself, so re-importing this
# module replaces the previous wrapper rather than wrapping it — which would otherwise leave
# two layers, each consulting a different module's thread-local flag.
_real_getaddrinfo = getattr(socket.getaddrinfo, _WRAPPED, socket.getaddrinfo)
setattr(_guarded_getaddrinfo, _WRAPPED, _real_getaddrinfo)
socket.getaddrinfo = _guarded_getaddrinfo


@contextmanager
def public_destinations_only() -> Iterator[None]:
    """Refuse any name resolution to a non-public address for the length of the block.

    Thread-local, so one download's guard never reaches another thread's database
    connection. The flag is restored rather than cleared on the way out, so nesting — which
    `download` does not do today, but a caller wrapping it could — leaves the outer guard
    standing.
    """
    previous = getattr(_guarded, "active", False)
    _guarded.active = True
    try:
        yield
    finally:
        _guarded.active = previous


def validate_url(url: str) -> str:
    """Check the URL is fetchable at all, and does not obviously point inward.

    The early half of the SSRF defence, and the half that produces a usable error. It cannot
    be the whole of it — what it resolves here is not necessarily what gets connected to
    later — so it is backed by the socket guard rather than trusted on its own.

    Every address the host resolves to has to be public, not merely one of them. A name that
    answers with a public address and a private one is a bypass if any single acceptable
    answer is enough, and there is no legitimate video host that needs 127.0.0.1 in its A
    records.

    Returns the URL unchanged, so it reads as a checkpoint at a call site.
    """
    try:
        parts = urlsplit(url)
    except ValueError as error:
        raise UnsupportedUrl("The URL could not be parsed.") from error

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        # Named without echoing the URL: the scheme is the part that was wrong, and quoting
        # the rest back into an error message is how a `file:///etc/passwd` attempt ends up
        # in a log line that gets pasted somewhere.
        raise UnsupportedUrl(
            f"Only {' and '.join(sorted(ALLOWED_SCHEMES))} URLs can be downloaded."
        )

    try:
        # Both of these parse lazily and both can refuse: a malformed IPv6 literal on the
        # first, a port that is not a number or not in range on the second.
        host = parts.hostname
        port = parts.port
    except ValueError as error:
        raise UnsupportedUrl("The URL's host or port could not be read.") from error

    if not host:
        raise UnsupportedUrl("The URL has no host.")

    try:
        resolved = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except BlockedAddress:
        # The socket guard is already active around this call in a nested `download`, and it
        # has said no. Its answer is the answer.
        raise
    except OSError as error:
        raise UnsupportedUrl(f"{host} could not be resolved.") from error

    for *_, sockaddr in resolved:
        if not is_public_address(sockaddr[0]):
            logger.warning("Refused %r: it resolves to a non-public address.", host)
            raise BlockedAddress(f"{host} resolves to an address that is not public.")

    return url


def is_youtube_url(url: str) -> bool:
    """Whether this URL is one the YouTube-specific format selector applies to.

    Suffix-matched against `YOUTUBE_HOSTS` so subdomains count and lookalikes do not: the
    boundary is a literal dot, which is what separates `m.youtube.com` from
    `youtube.com.example.net`. A trailing root dot is stripped first, since `youtube.com.` is
    the same name and would otherwise slip past.

    Deliberately not asking yt-dlp which extractor it would pick. That answer only exists
    after extraction, and the format string is an option handed to yt-dlp before it starts.
    """
    host = (urlsplit(url).hostname or "").lower().rstrip(".")

    return any(host == known or host.endswith(f".{known}") for known in YOUTUBE_HOSTS)


def _reject_live(info: dict) -> None:
    """Refuse anything that is not a finished recording.

    A live stream has no size and no end. yt-dlp would happily record it until the disk
    filled, and `max_filesize` does not help — that bounds a file whose length is known in
    advance, which is exactly what a stream does not have.
    """
    if info.get("is_live") or info.get("live_status") in LIVE_STATUSES:
        raise LiveStreamRejected("Live streams cannot be analysed.")


def _declared_size(info: dict) -> int | None:
    """What the site says the media weighs, if it says anything.

    `filesize` is exact and often absent; `filesize_approx` is yt-dlp's estimate from the
    bitrate and duration. Either is worth refusing on before spending the bandwidth, and
    neither is trusted afterwards — the bytes on disk are what get checked.

    A merge is asked the same question about the whole acquisition rather than about half of
    it. When two formats were selected the top-level keys describe the video stream alone, so
    a 90 MiB video beside a 15 MiB audio track would declare itself inside a 100 MiB limit and
    then land outside it. Summing is what makes the early refusal cover the file that will
    actually exist — and only when every part declares a size, because a sum with a missing
    term understates and an understated size is the one shape of wrong answer this check must
    not produce. Where a part says nothing, the fall-through below asks the top level and, if
    that says nothing either, the download proceeds to be weighed on disk as it always was.
    """
    parts = info.get("requested_formats")
    if parts:
        sizes = [_declared_size(part) for part in parts]
        if all(size is not None for size in sizes):
            return sum(sizes)

    for key in ("filesize", "filesize_approx"):
        value = info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    return None


def _was_assembled(info: dict) -> bool:
    """Whether yt-dlp selected separate streams, meaning the file on disk was merged here.

    `requested_formats` is yt-dlp's own record of that decision: it is present, holding one
    entry per stream, exactly when the format selector resolved to a `+` combination that
    ffmpeg then muxes into a single output. A single-file selection leaves the key absent.

    Read off the metadata pass, so the answer is known from the same info dict the size and
    live-stream checks come from rather than inferred afterwards from what the directory
    happens to contain.
    """
    return len(info.get("requested_formats") or ()) > 1


def _options(directory: Path, url: str) -> dict:
    """How yt-dlp is configured for one download into one throwaway directory.

    The URL is a parameter because one option depends on it: YouTube gets a selector that may
    resolve to a video stream plus an audio stream, and every other site keeps the muxed-only
    selector it already had. Nothing else here branches, and nothing that branches here is a
    security control — the guards below and around this call are the same either way.

    `max_filesize` is the during-the-download half of the size limit: yt-dlp abandons a file
    that declares itself larger rather than fetching 100 MiB to find out. It is not the whole
    limit, because a server can under-declare or say nothing at all, which is why the bytes
    on disk are weighed afterwards regardless.

    `concurrent_fragment_downloads` is pinned to one for a security reason rather than a
    performance one: the socket guard is thread-local, and a fragment fetched on a worker
    thread yt-dlp started would resolve outside it. One connection at a time keeps every
    resolution on the thread that is guarded.
    """
    return {
        "format": YOUTUBE_MEDIA_FORMAT if is_youtube_url(url) else MEDIA_FORMAT,
        # Which container a merge is written into. Only the YouTube selector can ask for one,
        # so this is inert for every other site — but naming it means the merged file is an
        # MP4 by instruction rather than by yt-dlp's inference, which is what keeps the
        # extension inside the two types the ingestion route admits.
        #
        # ffmpeg does this locally, on bytes already fetched through the guarded socket. It
        # opens no connection of its own and needs no network path.
        "merge_output_format": "mp4",
        "paths": {"home": str(directory)},
        "outtmpl": {"default": "media.%(ext)s"},
        # A URL that happens to name a playlist is one video's worth of work, not a channel's.
        "noplaylist": True,
        "max_filesize": MAX_DOWNLOAD_BYTES,
        "concurrent_fragment_downloads": 1,
        # Read here rather than at import, so a deployment can widen it without a rebuild.
        "socket_timeout": download_socket_timeout_seconds(),
        "retries": 2,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Nothing is cached to disk and nothing phones home. A cache directory shared between
        # analyses is state this service does not want to reason about.
        "cachedir": False,
        "noplaylist_metafiles": True,
    }


def _downloaded_file(directory: Path) -> Path:
    """The single media file yt-dlp left behind, or a failure that says none arrived.

    A directory with nothing in it is the normal shape of a `max_filesize` refusal: yt-dlp
    skips the download and reports success, because from its point of view it did what it was
    told. Reading the directory is how that is noticed.
    """
    files = [entry for entry in directory.iterdir() if entry.is_file()]

    if not files:
        raise MediaTooLarge(
            f"The media is larger than the {MAX_DOWNLOAD_BYTES} byte limit, or was not "
            "downloadable."
        )

    # One file, always. A muxed selection downloads one and a merged selection leaves one:
    # yt-dlp writes the two streams to `media.f<id>.<ext>` parts, muxes them and deletes the
    # parts, so a successful merge is as single-file as a progressive download.
    #
    # More than one file therefore means the merge did not finish — ffmpeg missing, ffmpeg
    # failed, streams it would not put in one container — and the parts are still sitting
    # there. Guessing which of them is the video would be the wrong way to find that out, so
    # this refuses instead, and `download` reports it as an unavailable download exactly as
    # it did before R7-T1.
    if len(files) > 1:
        raise DownloadUnavailable("The download produced more than one file.")

    return files[0]


@contextmanager
def download(url: str) -> Iterator[DownloadedMedia]:
    """Fetch the video behind a public URL, hand it over as a local file, then delete it.

    A context manager rather than a function returning a path, so cleanup is not something a
    caller can forget. The directory goes on every path out of this block — a successful
    analysis, a rejected one, an exception from the caller's own body — because the
    alternative is a container that accumulates every video anyone ever submitted.

    The size limit is enforced three times and each one covers a gap the others leave: what
    the site declares is refused before any bytes move, `max_filesize` abandons a download
    mid-flight, and the file on disk is weighed at the end because a server can declare
    whatever it likes. Only the last of the three is authoritative.
    """
    validate_url(url)

    directory = Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))

    try:
        with public_destinations_only():
            info = _fetch(url, directory)

        media = _downloaded_file(directory)
        size_bytes = media.stat().st_size
        assembled = _was_assembled(info)

        if size_bytes > MAX_DOWNLOAD_BYTES:
            # The declared size was wrong or absent and `max_filesize` had nothing to act on.
            # This is the check that does not depend on the source telling the truth.
            raise MediaTooLarge(
                f"The downloaded media is {size_bytes} bytes, over the "
                f"{MAX_DOWNLOAD_BYTES} byte limit."
            )

        # Which of the two acquisitions this was is worth a log line rather than only a field:
        # it is the difference between bytes a site served and bytes this container assembled,
        # and an operator reading back over an analysis should not have to guess which one is
        # in MinIO.
        logger.info(
            "Downloaded %s bytes of media from %r (%s).",
            size_bytes,
            urlsplit(url).netloc,
            "assembled here from separate video and audio streams"
            if assembled
            else "as served, a single file",
        )

        yield DownloadedMedia(
            path=media,
            # yt-dlp's own name for the file it wrote. It is a fixed template rather than the
            # site's title, so nothing a publisher controls reaches a filename.
            filename=media.name,
            size_bytes=size_bytes,
            assembled=assembled,
        )
    finally:
        # `ignore_errors`, because a directory that has already gone is the outcome wanted
        # and a cleanup that raised would replace the real error with a tidy-up one.
        shutil.rmtree(directory, ignore_errors=True)


def _fetch(url: str, directory: Path) -> dict:
    """Ask yt-dlp about the URL, refuse it if it is live, and download it if it is not.

    Metadata first, in its own call, because the two questions worth asking early — is this a
    stream, and how big is it — both have to be answered before any bytes are fetched rather
    than after.

    Called inside the socket guard, so the metadata request is checked at the same standard
    as the download: an extractor that follows a redirect to a private address is doing it on
    this call as readily as on the next one.
    """
    with yt_dlp.YoutubeDL(_options(directory, url)) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except BlockedAddress:
            raise
        except yt_dlp.utils.DownloadError as error:
            # yt-dlp's message quotes the URL, the extractor and sometimes a response body.
            # It goes to the log; the caller gets the fact and not the transcript.
            logger.warning("yt-dlp could not read %r: %s", url, error)
            raise DownloadUnavailable("The URL could not be read as media.") from None

        if info is None:
            raise DownloadUnavailable("The URL could not be read as media.")

        info = ydl.sanitize_info(info)
        _reject_live(info)

        declared = _declared_size(info)
        if declared is not None and declared > MAX_DOWNLOAD_BYTES:
            raise MediaTooLarge(
                f"The media declares {declared} bytes, over the {MAX_DOWNLOAD_BYTES} "
                "byte limit."
            )

        try:
            ydl.download([url])
        except BlockedAddress:
            raise
        except yt_dlp.utils.DownloadError as error:
            logger.warning("yt-dlp could not download %r: %s", url, error)
            raise DownloadUnavailable("The media could not be downloaded.") from None

    return info
