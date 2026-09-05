"""The URL downloader, and mostly its refusals.

No test here reaches the network. That is not only for speed: the security properties being
checked are about what happens when a *hostile* URL is handed over, and a test that depended
on a real host resolving a particular way would be checking the internet rather than this
module. DNS is faked at `downloader._real_getaddrinfo`, which is the one true resolver behind
the SSRF guard, so a test can say "this name resolves to 10.0.0.1" and mean it.

yt-dlp is faked the same way and for the same reason — the interesting cases are a live
stream, a file that arrives larger than it declared, and an extractor that fails, none of
which a real site can be asked to produce on demand.

The two things that are *not* faked are the ones worth trusting least: `ipaddress`, which
decides what counts as public, and the real `socket.getaddrinfo` wrapper, which is what
actually stops a redirect. Both are exercised directly.
"""

import ipaddress
import shutil
import socket
import threading
from pathlib import Path

import pytest
import yt_dlp

from app import downloader
from app.downloader import (
    BlockedAddress,
    DownloadUnavailable,
    LiveStreamRejected,
    MediaTooLarge,
    UnsupportedUrl,
)

PUBLIC_URL = "https://videos.example.com/clip"
MEDIA_BYTES = b"downloaded-mp4-bytes"


def addrinfo(*addresses):
    """`getaddrinfo`'s answer shape, carrying exactly the addresses a test wants returned."""
    return [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443))
        for address in addresses
    ]


@pytest.fixture
def dns(monkeypatch):
    """Control what every hostname in this module resolves to.

    Patched at `_real_getaddrinfo` rather than at `socket.getaddrinfo`, so the SSRF wrapper
    itself stays in the path and is exercised rather than replaced — the wrapper is the
    thing under test in half of these.
    """

    class Resolver:
        def __init__(self):
            self.answers = {}
            self.default = addrinfo("93.184.216.34")
            self.error = None
            self.lookups = []

        def set(self, host, *addresses):
            self.answers[host] = addrinfo(*addresses)

        def __call__(self, host, port, *args, **kwargs):
            self.lookups.append(host)
            if self.error:
                raise self.error
            return self.answers.get(host, self.default)

    resolver = Resolver()
    monkeypatch.setattr(downloader, "_real_getaddrinfo", resolver)
    return resolver


class FakeYoutubeDL:
    """Stands in for `yt_dlp.YoutubeDL`, writing whatever a test says the site served.

    Only the four things the downloader uses are implemented — the context manager,
    `extract_info`, `sanitize_info` and `download` — so a call this module starts making
    without a test noticing fails here rather than passing silently.
    """

    recorder = None

    def __init__(self, options):
        self.options = options
        FakeYoutubeDL.recorder.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def extract_info(self, url, download=False):
        recorder = FakeYoutubeDL.recorder
        recorder.guarded.append(getattr(downloader._guarded, "active", False))
        recorder.extracted.append((url, download))
        if recorder.extract_error:
            raise recorder.extract_error
        return recorder.info

    def sanitize_info(self, info):
        return info

    def download(self, urls):
        recorder = FakeYoutubeDL.recorder
        recorder.guarded.append(getattr(downloader._guarded, "active", False))
        recorder.downloaded.append(list(urls))
        if recorder.download_error:
            raise recorder.download_error
        if recorder.written is None:
            # What a `max_filesize` refusal looks like: yt-dlp reports success and writes
            # nothing at all.
            return
        home = Path(self.options["paths"]["home"])
        if recorder.files is not None:
            # An exact directory, for the cases where what yt-dlp leaves behind is the point:
            # a merge that finished leaves one file, a merge that did not leaves its parts.
            for name, payload in recorder.files.items():
                (home / name).write_bytes(payload)
            return
        (home / "media.mp4").write_bytes(recorder.written)


@pytest.fixture
def site(monkeypatch):
    """A stand-in video site: what it says about the media, and what it serves."""

    class Recorder:
        def __init__(self):
            self.info = {"id": "clip", "ext": "mp4"}
            self.written = MEDIA_BYTES
            # None means "the one `media.mp4` holding `written`", which is what a progressive
            # download leaves. A dict names the directory contents exactly instead.
            self.files = None
            self.extract_error = None
            self.download_error = None
            self.extracted = []
            self.downloaded = []
            self.instances = []
            # Whether the SSRF guard was up on each call yt-dlp made — the metadata request
            # and every phase of the download, fragments included.
            self.guarded = []

    recorder = Recorder()
    FakeYoutubeDL.recorder = recorder
    monkeypatch.setattr(downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    return recorder


# --- what counts as a public address ------------------------------------------------------
#
# The whole SSRF defence rests on this one predicate, so it is checked against the addresses
# that actually matter rather than a couple of representative ones.


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.1.2.3",
        "0.0.0.0",
        "10.0.0.1",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.1.1",
        # The cloud metadata endpoint. The single most valuable thing an SSRF can reach on a
        # hosted deployment, and it is an ordinary link-local address.
        "169.254.169.254",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "::",
        "fe80::1",
        "fc00::1",
        "ff02::1",
        # IPv4 wearing IPv6 clothes. `ipaddress` reads the embedded address and answers about
        # that; these cases exist so a Python that ever stopped doing so fails here instead
        # of opening a bypass in production.
        "::ffff:127.0.0.1",
        "::ffff:10.0.0.1",
        "::ffff:169.254.169.254",
        # 6to4 wrapping loopback and a private range.
        "2002:7f00:1::1",
        "2002:a00:1::1",
        # Teredo.
        "2001:0:4136:e378:8000:63bf:3fff:fdd2",
    ],
)
def test_non_public_addresses_are_refused(address):
    assert downloader.is_public_address(address) is False


@pytest.mark.parametrize(
    "address",
    ["8.8.8.8", "93.184.216.34", "1.1.1.1", "2606:4700:4700::1111", "2001:4860:4860::8888"],
)
def test_public_addresses_are_allowed(address):
    assert downloader.is_public_address(address) is True


@pytest.mark.parametrize("value", ["", "not-an-address", "999.999.999.999", "localhost"])
def test_anything_unparseable_is_refused(value):
    """No benefit of the doubt. A value that is not an address cannot be shown to be safe."""
    assert downloader.is_public_address(value) is False


def test_the_disguised_forms_really_are_the_addresses_they_wrap():
    """The assumption the parametrized cases above rest on, stated outright.

    If this ever fails, `is_public_address` is being carried by a property that no longer
    means what it says, and the refusals above would be passing for the wrong reason.
    """
    assert ipaddress.ip_address("::ffff:127.0.0.1").ipv4_mapped == ipaddress.ip_address("127.0.0.1")
    assert ipaddress.ip_address("2002:7f00:1::1").sixtofour == ipaddress.ip_address("127.0.0.1")
    assert ipaddress.ip_address("2001:0:4136:e378:8000:63bf:3fff:fdd2").teredo is not None


# --- the scheme gate ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "file://localhost/etc/shadow",
        "ftp://example.com/clip.mp4",
        "gopher://example.com:70/",
        "data:video/mp4;base64,AAAA",
        "dict://example.com:11211/",
        "//example.com/clip.mp4",
        "example.com/clip.mp4",
    ],
)
def test_only_http_and_https_are_attempted(url, dns):
    """Everything else is refused before anything is resolved, let alone fetched.

    yt-dlp understands more schemes than DeepGuard has any reason to accept, and `file://`
    would read this container's own filesystem — the URL is attacker-supplied, so the set of
    things it may name is an allowlist or it is nothing.
    """
    with pytest.raises(UnsupportedUrl):
        downloader.validate_url(url)

    # Refused on the scheme alone: nothing was even looked up.
    assert dns.lookups == []


def test_the_refusal_does_not_quote_the_url_back(dns):
    """A rejected `file:///etc/passwd` should not end up copied into a log or an API body."""
    with pytest.raises(UnsupportedUrl) as refusal:
        downloader.validate_url("file:///etc/passwd")

    assert "etc/passwd" not in str(refusal.value)


@pytest.mark.parametrize("url", ["http://", "https:///clip.mp4"])
def test_a_url_with_no_host_is_refused(url, dns):
    with pytest.raises(UnsupportedUrl):
        downloader.validate_url(url)


def test_an_unreadable_port_is_refused(dns):
    with pytest.raises(UnsupportedUrl):
        downloader.validate_url("http://example.com:not-a-port/clip.mp4")


def test_a_name_that_cannot_be_resolved_is_refused(dns):
    dns.error = socket.gaierror("Name or service not known")

    with pytest.raises(UnsupportedUrl):
        downloader.validate_url(PUBLIC_URL)


# --- the pre-flight address check ---------------------------------------------------------


def test_a_public_host_passes(dns):
    dns.set("videos.example.com", "93.184.216.34")

    assert downloader.validate_url(PUBLIC_URL) == PUBLIC_URL


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/api/v1/analyses",
        "http://localhost:8000/health",
        "http://minio:9000/deepguard",
        "http://postgres:5432/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://[::1]:8000/",
    ],
)
def test_internal_destinations_are_refused(url, dns):
    """The services this container can actually reach, named as a caller would name them.

    These are not hypothetical: `minio:9000` and `postgres:5432` are real hostnames on this
    Docker network, and `127.0.0.1:8000` is DeepGuard's own API.
    """
    dns.set("localhost", "127.0.0.1")
    dns.set("minio", "172.18.0.3")
    dns.set("postgres", "172.18.0.2")
    dns.default = addrinfo("10.0.0.5")

    with pytest.raises(BlockedAddress):
        downloader.validate_url(url)


def test_a_host_answering_with_one_private_address_among_public_ones_is_refused(dns):
    """Every answer has to be public, not just one of them.

    A name that resolves to a public address *and* 127.0.0.1 is a bypass the moment a single
    acceptable answer is enough — the check would pass on one record and the connection would
    be made to another. No real video host needs loopback in its A records.
    """
    dns.set("videos.example.com", "93.184.216.34", "127.0.0.1")

    with pytest.raises(BlockedAddress):
        downloader.validate_url(PUBLIC_URL)


# --- the socket-layer guard ---------------------------------------------------------------
#
# The pre-flight check above proves something about one moment. This is what covers the rest
# of the download: a redirect to a new host, or a name that answers publicly once and
# privately the next time.


def test_a_guarded_resolution_to_a_private_address_is_refused(dns):
    dns.set("rebind.example.com", "127.0.0.1")

    with downloader.public_destinations_only():
        with pytest.raises(BlockedAddress):
            socket.getaddrinfo("rebind.example.com", 443)


def test_a_redirect_to_a_literal_internal_address_is_refused(dns):
    """A redirect straight to `http://127.0.0.1/` never asks DNS anything.

    It still goes through `getaddrinfo`, which is exactly why the guard is installed there
    rather than on hostname lookups — an IP literal and a hostname are the same code path.
    """
    dns.set("127.0.0.1", "127.0.0.1")

    with downloader.public_destinations_only():
        with pytest.raises(BlockedAddress):
            socket.getaddrinfo("127.0.0.1", 80)


def test_a_second_resolution_is_checked_as_well_as_the_first(dns):
    """DNS rebinding: the answer changes between the check and the connection.

    The guard re-checks every resolution rather than remembering a verdict, so a record that
    turns private after passing once is caught on the lookup that matters.
    """
    dns.set("rebind.example.com", "93.184.216.34")

    with downloader.public_destinations_only():
        assert socket.getaddrinfo("rebind.example.com", 443)

        dns.set("rebind.example.com", "127.0.0.1")
        with pytest.raises(BlockedAddress):
            socket.getaddrinfo("rebind.example.com", 443)


def test_the_guard_does_not_apply_outside_a_download(dns):
    """The application it protects talks to private addresses constantly.

    PostgreSQL and MinIO are on the Docker network. A guard that applied process-wide would
    take the product down to secure it.
    """
    dns.set("postgres", "172.18.0.2")

    assert socket.getaddrinfo("postgres", 5432)


def test_the_guard_is_confined_to_the_thread_that_set_it(dns):
    """One download's guard must not reach another thread's database connection."""
    dns.set("postgres", "172.18.0.2")
    other_thread = []

    def resolve_elsewhere():
        try:
            other_thread.append(bool(socket.getaddrinfo("postgres", 5432)))
        except BlockedAddress:
            other_thread.append("blocked")

    with downloader.public_destinations_only():
        with pytest.raises(BlockedAddress):
            socket.getaddrinfo("postgres", 5432)

        thread = threading.Thread(target=resolve_elsewhere)
        thread.start()
        thread.join(timeout=10)

    assert other_thread == [True]


def test_the_guard_is_lifted_again_afterwards(dns):
    dns.set("postgres", "172.18.0.2")

    with downloader.public_destinations_only():
        pass

    assert socket.getaddrinfo("postgres", 5432)


def test_a_guard_inside_a_guard_leaves_the_outer_one_standing(dns):
    dns.set("postgres", "172.18.0.2")

    with downloader.public_destinations_only():
        with downloader.public_destinations_only():
            pass

        with pytest.raises(BlockedAddress):
            socket.getaddrinfo("postgres", 5432)


def test_the_installed_wrapper_is_the_real_resolver_underneath():
    """The patch is in place, and it kept the genuine `getaddrinfo` rather than shadowing it."""
    assert socket.getaddrinfo is downloader._guarded_getaddrinfo
    assert downloader._real_getaddrinfo is not downloader._guarded_getaddrinfo


# --- live streams -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info",
    [
        {"is_live": True},
        {"live_status": "is_live"},
        {"live_status": "is_upcoming"},
        {"live_status": "post_live"},
    ],
)
def test_live_streams_are_refused(info, dns, site, tmp_path):
    """A stream has no end, and `max_filesize` cannot bound something with no declared size.

    yt-dlp would record it until the disk filled.
    """
    site.info = {"id": "stream", **info}

    with pytest.raises(LiveStreamRejected):
        with downloader.download(PUBLIC_URL):
            pass

    # Refused on the metadata alone: nothing was fetched.
    assert site.downloaded == []


def test_a_finished_recording_is_not_mistaken_for_a_stream(dns, site):
    site.info = {"id": "clip", "live_status": "not_live", "is_live": False}

    with downloader.download(PUBLIC_URL) as media:
        assert media.size_bytes == len(MEDIA_BYTES)


# --- the size limit ---------------------------------------------------------------------
#
# Enforced three times, and each of the three tests below removes one of the others to show
# it is not the one doing the work.


def test_media_declaring_more_than_the_limit_is_refused_before_it_is_fetched(dns, site):
    site.info = {"id": "clip", "filesize": downloader.MAX_DOWNLOAD_BYTES + 1}

    with pytest.raises(MediaTooLarge):
        with downloader.download(PUBLIC_URL):
            pass

    # The point of checking the declaration: the bandwidth is never spent.
    assert site.downloaded == []


def test_an_estimated_size_over_the_limit_is_refused_too(dns, site):
    """`filesize` is often absent; `filesize_approx` is what a site usually offers instead."""
    site.info = {"id": "clip", "filesize_approx": downloader.MAX_DOWNLOAD_BYTES * 2}

    with pytest.raises(MediaTooLarge):
        with downloader.download(PUBLIC_URL):
            pass

    assert site.downloaded == []


def test_a_download_that_wrote_nothing_is_refused(dns, site):
    """What a `max_filesize` abort looks like from here.

    yt-dlp reports success and writes no file, because from its point of view it did as it
    was told. An empty directory is the only evidence, so it is read rather than assumed.
    """
    site.info = {"id": "clip"}
    site.written = None

    with pytest.raises(MediaTooLarge):
        with downloader.download(PUBLIC_URL):
            pass


def test_a_file_that_arrives_larger_than_it_declared_is_refused(dns, site, monkeypatch):
    """The check that does not trust the source at all.

    The site declares nothing and serves more than the limit — so neither the declaration
    check nor `max_filesize` has anything to act on, and only weighing the bytes on disk
    catches it. This is the authoritative one of the three.
    """
    monkeypatch.setattr(downloader, "MAX_DOWNLOAD_BYTES", 8)
    site.info = {"id": "clip"}
    site.written = b"considerably longer than eight bytes"

    with pytest.raises(MediaTooLarge) as refusal:
        with downloader.download(PUBLIC_URL):
            pass

    assert str(len(site.written)) in str(refusal.value)


def test_the_limit_is_the_upload_limit(dns, site):
    """One ceiling for media entering DeepGuard, whichever door it came through.

    A URL download allowed to be larger than an upload would be the upload limit with a hole
    in it.
    """
    from app.media import MAX_UPLOAD_BYTES

    assert downloader.MAX_DOWNLOAD_BYTES == MAX_UPLOAD_BYTES


def test_the_limit_is_handed_to_yt_dlp_as_well(dns, site):
    """The during-the-download half: yt-dlp abandons an oversized file mid-flight."""
    with downloader.download(PUBLIC_URL):
        pass

    assert site.instances[0].options["max_filesize"] == downloader.MAX_DOWNLOAD_BYTES


# --- a successful download ----------------------------------------------------------------


def test_a_public_url_becomes_a_local_file(dns, site):
    dns.set("videos.example.com", "93.184.216.34")

    with downloader.download(PUBLIC_URL) as media:
        assert media.path.exists()
        assert media.path.read_bytes() == MEDIA_BYTES
        assert media.size_bytes == len(MEDIA_BYTES)
        assert media.filename == "media.mp4"


def test_the_url_reaches_yt_dlp_as_an_argument(dns, site):
    """Not a command string, and not a shell.

    yt-dlp is a library here, so the URL is a Python value from end to end — a URL full of
    shell metacharacters is data, with nothing to interpret it.
    """
    hostile = "https://videos.example.com/clip;$(id)&&`whoami`"

    with downloader.download(hostile):
        pass

    assert site.extracted == [(hostile, False)]
    assert site.downloaded == [[hostile]]


def test_the_downloaded_name_is_not_the_publishers(dns, site):
    """A filename is a fixed template, so nothing the source controls reaches the filesystem."""
    site.info = {"id": "clip", "title": "../../etc/passwd"}

    with downloader.download(PUBLIC_URL) as media:
        assert media.filename == "media.mp4"
        assert media.path.parent.name.startswith(downloader.TEMP_DIR_PREFIX)


def test_one_video_is_taken_from_a_url_that_names_a_playlist(dns, site):
    with downloader.download(PUBLIC_URL):
        pass

    assert site.instances[0].options["noplaylist"] is True


def test_fragments_are_fetched_on_the_guarded_thread(dns, site):
    """Pinned to one for a security reason, not a performance one.

    The socket guard is thread-local. A fragment fetched on a worker thread yt-dlp started
    would resolve outside it, so concurrency here would be a hole in the SSRF defence.
    """
    with downloader.download(PUBLIC_URL):
        pass

    assert site.instances[0].options["concurrent_fragment_downloads"] == 1


# --- YouTube, and only YouTube (R7-T1) ----------------------------------------------------
#
# Nothing here reaches YouTube. A test that did would be checking YouTube's catalogue on the
# day it ran, which is neither deterministic nor the property worth pinning: what matters is
# that a YouTube URL is recognised as one, that it is handed the merging selector, that a
# merged result is reported as assembled rather than as served bytes, and that every other
# site is left exactly where P10 left it.


YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# What yt-dlp's info dict looks like once the format selector has resolved to two streams.
# `requested_formats` is the record of that decision, and it is present only for a merge.
VIDEO_STREAM = {"format_id": "137", "ext": "mp4", "vcodec": "avc1.640028", "acodec": "none"}
AUDIO_STREAM = {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2"}


def merged_info(**extra):
    """A resolved YouTube extraction that will be merged from a video and an audio stream."""
    return {
        "id": "dQw4w9WgXcQ",
        "ext": "mp4",
        "requested_formats": [dict(VIDEO_STREAM), dict(AUDIO_STREAM)],
        **extra,
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        "https://YouTube.COM/watch?v=dQw4w9WgXcQ",
        "https://youtube.com./watch?v=dQw4w9WgXcQ",
    ],
)
def test_youtube_is_recognised_by_host(url):
    assert downloader.is_youtube_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://notyoutube.com/watch?v=x",
        "https://youtube.com.example.net/watch?v=x",
        "https://myyoutu.be/x",
        "https://videos.example.com/clip?ref=youtube.com",
        "https://example.com/youtube.com",
        "https://192.0.2.1/clip",
    ],
)
def test_a_lookalike_host_is_not_youtube(url):
    """The suffix match is on a dot boundary, so a name that merely contains one is not one.

    `youtube.com.example.net` is the case that matters: it is somebody else's domain, and a
    substring test would hand it the YouTube selector. Nothing unsafe follows from that —
    the selector is a quality decision, not a guard — but a rule scoped to one site should
    actually be scoped to one site.
    """
    assert downloader.is_youtube_url(url) is False


def test_youtube_is_asked_for_streams_it_actually_serves(dns, site):
    """The one selector change, asserted as the exact option yt-dlp receives.

    YouTube's player clients list DASH and HLS: video-only and audio-only formats, and no
    single muxed file above 360p. `MEDIA_FORMAT` matches nothing in that catalogue, which is
    what capped P10's ingestion. This selector asks for the two streams and lets ffmpeg put
    them in one container, with the muxed chain kept behind it as a fallback.

    The `vcodec^=avc1` half is load-bearing and not decoration: without it `[ext=mp4]` selects
    AV1 on most YouTube videos, because AV1-in-MP4 outranks H.264 there. H.264 is what the
    360p path served and what the detectors have been measured on.

    Pinned here because a well-meaning edit that dropped it would put every YouTube
    acquisition back at 360p silently — the download would still succeed.
    """
    site.info = merged_info()

    with downloader.download(YOUTUBE_URL):
        pass

    assert site.instances[0].options["format"] == downloader.YOUTUBE_MEDIA_FORMAT
    assert site.instances[0].options["format"] == (
        "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]"
        "/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]"
        "/best[ext=mp4]/best"
    )


def test_every_other_site_keeps_the_muxed_only_selector(dns, site):
    """R7-T1 is a YouTube change. A direct media URL must not notice it happened.

    The progressive path is the byte-preserving one — one file, served, nothing merged — and
    it stays the default for everything that is not YouTube.
    """
    with downloader.download(PUBLIC_URL):
        pass

    assert site.instances[0].options["format"] == downloader.MEDIA_FORMAT
    assert site.instances[0].options["format"] == "best[ext=mp4]/best[ext=mov]/best"


def test_no_extractor_is_steered_towards_a_player_client(dns, site):
    """P10 pinned YouTube's `android` client to find itag 18. Nothing needs that now.

    That client lists five formats and none above 360p, so keeping it would cap the merge at
    the resolution the merge exists to get past. With it gone, no site is steered at all and
    yt-dlp's own defaults apply everywhere.
    """
    assert not hasattr(downloader, "EXTRACTOR_ARGS")

    with downloader.download(YOUTUBE_URL):
        pass

    assert "extractor_args" not in site.instances[0].options

    with downloader.download(PUBLIC_URL):
        pass

    assert "extractor_args" not in site.instances[1].options


def test_a_merge_is_written_into_an_mp4(dns, site):
    """The ingestion route admits `.mp4` and `.mov` and nothing else.

    Naming the container means the merged file's extension is decided here rather than
    inferred by yt-dlp from whatever pair of streams it chose, which is what keeps a
    successful acquisition from being refused at the door for its suffix.
    """
    site.info = merged_info()

    with downloader.download(YOUTUBE_URL):
        pass

    assert site.instances[0].options["merge_output_format"] == "mp4"


def test_a_merged_acquisition_is_reported_as_assembled(dns, site):
    """The byte-semantics flag, which is the whole forensic point of this task.

    A DASH acquisition is not a copy of a file YouTube served — there is no such file. It is
    two streams this container fetched and muxed, and `assembled` is how the artifact says
    so. Nothing downstream may read authenticity off it in either direction; what it
    prevents is the opposite claim.
    """
    site.info = merged_info()
    site.files = {"media.mp4": MEDIA_BYTES}

    with downloader.download(YOUTUBE_URL) as media:
        assert media.assembled is True
        assert media.filename == "media.mp4"
        assert media.path.read_bytes() == MEDIA_BYTES


def test_a_served_single_file_is_not_reported_as_assembled(dns, site):
    """The other half of the same statement, and the one every non-YouTube URL takes.

    One muxed format, no `requested_formats`, nothing merged: these are the bytes the source
    served, and the flag has to keep saying that or it says nothing at all.
    """
    with downloader.download(PUBLIC_URL) as media:
        assert media.assembled is False


def test_a_youtube_url_that_offers_one_muxed_file_is_not_assembled(dns, site):
    """The fallback tail of the selector, which is still a byte-preserving acquisition.

    `best[ext=mp4]` behind the `+` chain means a YouTube URL that does serve a single file
    takes it untouched. The flag follows what actually happened, not which site it was.
    """
    site.info = {"id": "dQw4w9WgXcQ", "ext": "mp4"}

    with downloader.download(YOUTUBE_URL) as media:
        assert media.assembled is False


def test_a_merge_that_did_not_finish_is_refused(dns, site):
    """ffmpeg missing or failing leaves the two parts behind, and two files is a failure.

    yt-dlp deletes the parts once it has muxed them, so a directory holding
    `media.f137.mp4` beside `media.f140.m4a` means the merge never happened. Handing either
    one to the pipeline would be handing over a video with no sound or a sound file with no
    video, so this refuses rather than guessing.
    """
    site.info = merged_info()
    site.files = {"media.f137.mp4": b"video-stream", "media.f140.m4a": b"audio-stream"}

    with pytest.raises(DownloadUnavailable):
        with downloader.download(YOUTUBE_URL):
            pass


def test_nothing_is_left_behind_when_a_merge_does_not_finish(dns, site):
    site.info = merged_info()
    site.files = {"media.f137.mp4": b"video-stream", "media.f140.m4a": b"audio-stream"}
    before = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))

    with pytest.raises(DownloadUnavailable):
        with downloader.download(YOUTUBE_URL):
            pass

    after = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))
    assert after == before


def test_ffmpeg_is_available_to_merge_with(dns, site):
    """The merge is local post-processing, and it is ffmpeg that does it.

    Asserted in the suite rather than assumed from the Dockerfile, because the suite runs in
    the API container and this is the one dependency the new path adds. Without it every
    YouTube acquisition falls back to whatever the muxed chain can find, which is the 360p
    outcome R7-T1 exists to remove — and it would do that silently.
    """
    assert shutil.which("ffmpeg") is not None


# --- the size ceiling across two streams --------------------------------------------------


def test_the_declared_size_of_a_merge_is_both_streams(dns, site, monkeypatch):
    """A merge is refused early on what the whole file will weigh, not on half of it.

    The top-level `filesize` of a merged extraction describes the video stream alone. A 60
    MiB video beside a 50 MiB audio track declares itself inside a 100 MiB limit and lands
    outside it, so the two are summed before anything is fetched.
    """
    monkeypatch.setattr(downloader, "MAX_DOWNLOAD_BYTES", 100)
    site.info = merged_info(filesize=60)
    site.info["requested_formats"][0]["filesize"] = 60
    site.info["requested_formats"][1]["filesize"] = 50

    with pytest.raises(MediaTooLarge):
        with downloader.download(YOUTUBE_URL):
            pass

    # Refused on the metadata pass. No bytes were spent finding out.
    assert site.downloaded == []


def test_a_merge_inside_the_limit_is_not_refused_early(dns, site, monkeypatch):
    monkeypatch.setattr(downloader, "MAX_DOWNLOAD_BYTES", 100)
    site.info = merged_info()
    site.info["requested_formats"][0]["filesize"] = 60
    site.info["requested_formats"][1]["filesize"] = 5
    site.files = {"media.mp4": MEDIA_BYTES}

    with downloader.download(YOUTUBE_URL) as media:
        assert media.assembled is True


def test_a_merge_that_declares_half_a_size_is_still_weighed_on_disk(dns, site, monkeypatch):
    """A sum with a missing term understates, so it is not used as a refusal.

    What catches it instead is the check that never trusted the source: the file that
    actually arrived is weighed, and that is the authoritative one for a merge exactly as it
    is for a progressive download.
    """
    monkeypatch.setattr(downloader, "MAX_DOWNLOAD_BYTES", 8)
    site.info = merged_info()
    site.info["requested_formats"][0]["filesize"] = 4
    site.files = {"media.mp4": b"far too many bytes to fit"}

    with pytest.raises(MediaTooLarge):
        with downloader.download(YOUTUBE_URL):
            pass

    # The early check had nothing to say, so the download happened and the bytes on disk are
    # what refused it.
    assert site.downloaded == [[YOUTUBE_URL]]


def test_the_limit_reaches_yt_dlp_on_the_youtube_path_too(dns, site):
    site.info = merged_info()

    with downloader.download(YOUTUBE_URL):
        pass

    assert site.instances[0].options["max_filesize"] == downloader.MAX_DOWNLOAD_BYTES


# --- the SSRF guard still covers the whole of a merged download ----------------------------


def test_every_phase_of_a_merged_download_runs_inside_the_guard(dns, site):
    """The guard is thread-local, and a merge fetches more than one thing.

    Two streams and, for DASH, many fragments of each — every one of them a resolution that
    has to be checked. They are all made from this thread by `ydl.download`, which runs
    inside `public_destinations_only`, and `concurrent_fragment_downloads` stays pinned to 1
    so none of them moves onto a worker thread the guard cannot see.
    """
    site.info = merged_info()
    site.files = {"media.mp4": MEDIA_BYTES}

    with downloader.download(YOUTUBE_URL):
        pass

    assert site.guarded == [True, True]
    assert site.instances[0].options["concurrent_fragment_downloads"] == 1


def test_a_youtube_fragment_resolving_inward_is_refused(dns, site):
    """A redirect mid-download is what the socket guard exists for, merge or no merge.

    Raised from inside `ydl.download`, which is where a fragment fetch lives, and it comes
    back out as `BlockedAddress` rather than as a generic failure.
    """
    site.info = merged_info()
    site.download_error = BlockedAddress("10.0.0.1 resolves to an address that is not public.")

    with pytest.raises(BlockedAddress):
        with downloader.download(YOUTUBE_URL):
            pass


def test_a_youtube_url_pointing_inward_never_reaches_the_extractor(dns, site):
    """The host check happens first, and it does not care which site it is.

    A hostname is a hostname: nothing about being YouTube-shaped exempts a URL from
    `validate_url`, and this pins that the new branch did not move the guard.
    """
    dns.set("www.youtube.com", "127.0.0.1")

    with pytest.raises(BlockedAddress):
        with downloader.download(YOUTUBE_URL):
            pass

    assert site.extracted == []


def test_a_youtube_live_stream_is_refused(dns, site):
    """A merge does not make a stream finite. It is refused on the metadata pass as before."""
    site.info = merged_info(live_status="is_live")

    with pytest.raises(LiveStreamRejected):
        with downloader.download(YOUTUBE_URL):
            pass

    assert site.downloaded == []


# --- cleanup ------------------------------------------------------------------------------


def test_the_temporary_file_is_removed_after_a_successful_download(dns, site):
    with downloader.download(PUBLIC_URL) as media:
        directory = media.path.parent
        assert media.path.exists()

    assert not media.path.exists()
    assert not directory.exists()


def test_the_temporary_file_is_removed_when_the_caller_raises(dns, site):
    """A container that ran for a week must not hold every video anyone submitted."""
    seen = {}

    with pytest.raises(RuntimeError):
        with downloader.download(PUBLIC_URL) as media:
            seen["directory"] = media.path.parent
            raise RuntimeError("the caller's own problem")

    assert not seen["directory"].exists()


def test_nothing_is_left_behind_when_the_media_is_too_large(dns, site, monkeypatch):
    monkeypatch.setattr(downloader, "MAX_DOWNLOAD_BYTES", 4)
    site.written = b"far too many bytes"
    before = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))

    with pytest.raises(MediaTooLarge):
        with downloader.download(PUBLIC_URL):
            pass

    after = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))
    assert after == before


def test_nothing_is_left_behind_when_the_download_fails(dns, site):
    site.download_error = yt_dlp.utils.DownloadError("the site said no")
    before = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))

    with pytest.raises(DownloadUnavailable):
        with downloader.download(PUBLIC_URL):
            pass

    after = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))
    assert after == before


def test_no_directory_is_created_for_a_url_that_never_passes_validation(dns, site):
    """Refused URLs cost nothing at all — not even a temp directory to clean up."""
    before = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))

    with pytest.raises(UnsupportedUrl):
        with downloader.download("file:///etc/passwd"):
            pass

    after = set(Path(downloader.tempfile.gettempdir()).glob(f"{downloader.TEMP_DIR_PREFIX}*"))
    assert after == before
    assert site.extracted == []


# --- failures from the site ---------------------------------------------------------------


def test_an_extractor_failure_becomes_an_unavailable_download(dns, site):
    site.extract_error = yt_dlp.utils.DownloadError("ERROR: unable to extract player version")

    with pytest.raises(DownloadUnavailable):
        with downloader.download(PUBLIC_URL):
            pass


def test_the_failure_does_not_carry_yt_dlps_message_out(dns, site):
    """yt-dlp quotes the URL, the extractor and sometimes a response body.

    That belongs in the worker log, not in an exception a caller could end up seeing —
    the same rule the worker's own failures follow.
    """
    site.extract_error = yt_dlp.utils.DownloadError(
        "ERROR: [generic] https://videos.example.com/clip: HTTP Error 403: Forbidden"
    )

    with pytest.raises(DownloadUnavailable) as failure:
        with downloader.download(PUBLIC_URL):
            pass

    assert "403" not in str(failure.value)
    assert "videos.example.com" not in str(failure.value)


def test_a_url_with_nothing_behind_it_is_refused(dns, site):
    site.info = None

    with pytest.raises(DownloadUnavailable):
        with downloader.download(PUBLIC_URL):
            pass


def test_a_blocked_address_during_the_download_is_reported_as_blocked(dns, site):
    """Not flattened into a generic failure.

    `BlockedAddress` says something a `DownloadUnavailable` does not — that the URL pointed
    inward — and an operator reading the log needs to be able to tell those apart.
    """

    def refuse(urls):
        raise BlockedAddress("127.0.0.1 resolves to an address that is not public.")

    site.download_error = None
    FakeYoutubeDL.download = lambda self, urls: refuse(urls)

    try:
        with pytest.raises(BlockedAddress):
            with downloader.download(PUBLIC_URL):
                pass
    finally:
        del FakeYoutubeDL.download


def test_yt_dlp_only_has_backends_the_guard_can_see():
    """The guard's one structural assumption, pinned.

    Blocking at `socket.getaddrinfo` works because every request yt-dlp makes goes through
    Python's socket layer to get there — both of its available backends do, `Urllib`
    directly and `Requests` through urllib3.

    `curl_cffi` is the one that would not. libcurl resolves names in C, never touching
    `socket.getaddrinfo`, so installing it — directly or as somebody else's transitive
    dependency — would turn the SSRF guard off for every download without changing a line of
    this module. That is a failure nobody would notice, so it fails here instead.
    """
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        handlers = {handler.RH_KEY for handler in ydl._request_director.handlers.values()}

    assert handlers <= {"Urllib", "Requests"}, (
        f"yt-dlp gained the {handlers - {'Urllib', 'Requests'}} backend, which may resolve "
        "names outside Python's socket layer and bypass the SSRF guard entirely"
    )
