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
        recorder.extracted.append((url, download))
        if recorder.extract_error:
            raise recorder.extract_error
        return recorder.info

    def sanitize_info(self, info):
        return info

    def download(self, urls):
        recorder = FakeYoutubeDL.recorder
        recorder.downloaded.append(list(urls))
        if recorder.download_error:
            raise recorder.download_error
        if recorder.written is None:
            # What a `max_filesize` refusal looks like: yt-dlp reports success and writes
            # nothing at all.
            return
        home = Path(self.options["paths"]["home"])
        (home / "media.mp4").write_bytes(recorder.written)


@pytest.fixture
def site(monkeypatch):
    """A stand-in video site: what it says about the media, and what it serves."""

    class Recorder:
        def __init__(self):
            self.info = {"id": "clip", "ext": "mp4"}
            self.written = MEDIA_BYTES
            self.extract_error = None
            self.download_error = None
            self.extracted = []
            self.downloaded = []
            self.instances = []

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
