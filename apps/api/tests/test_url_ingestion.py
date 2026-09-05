"""Submitting media by URL: that it is the upload pipeline, and that it fails safely.

The downloader itself is tested in `test_downloader.py` — the SSRF guards, the size ceiling,
the live-stream refusal — and none of that is retested here. What is under test in this file
is the seam: that a downloaded file goes into `accept_upload` rather than into a second
pipeline, that the temporary file is gone whatever happened, that a download failure reaches
the caller as a safe client error with no extractor or socket detail in it, and that the
public URL route keeps the authentication, ownership and throttling the public upload has.

So `app.downloader.download` is replaced throughout. A test that fetched a real URL would be
testing the internet, and one that fetched a fake HTTP server would still be testing yt-dlp,
which already has its own tests here. Everything below is about what DeepGuard does with the
file once it exists — or with the failure, when it does not.

The public tests split the way `test_public_api.py` splits, for the same reason: the route's
behaviour is provable against a fake session, but ownership is a `WHERE` clause and the
concurrency limit is a lock, and neither is a property a fake can demonstrate.
"""

import json
import inspect
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import downloader, media, risk_engine, storage
from app.api.analyses import ALLOWED_CONTENT_TYPES, active_analyses
from app.api.public_v1.analyses import MAX_ACTIVE_ANALYSES
from app.api.url_analyses import MAX_URL_LENGTH, URL_MEDIA_TYPES
from app.auth import generate_api_key
from app.db.models import (
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_QUEUED,
    USER_ROLE_USER,
    Analysis,
    AnalysisJob,
    ApiKey,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine, get_session
from app.downloader import (
    BlockedAddress,
    DownloadedMedia,
    DownloadError,
    DownloadUnavailable,
    LiveStreamRejected,
    MediaTooLarge,
    UnsupportedUrl,
)
from app.main import app
from app.web_auth import require_user
from tests.conftest import DASHBOARD_ORIGIN

INTERNAL_URL = "/api/v1/analyses/url"
PUBLIC_URL = "/api/public/v1/analyses/url"
PUBLIC_UPLOAD_URL = "/api/public/v1/analyses"

SUBMITTED_URL = "https://videos.example.com/clip"
DOWNLOADED_BYTES = b"pretend-this-is-downloaded-video"

integration = pytest.mark.integration


def authorization(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# --- the fake downloader -----------------------------------------------------------------


class RecordingDownloader:
    """Stands in for `download`, writing a real file into a real temporary directory.

    A real file on disk, because that is the whole of what the seam under test consumes: the
    pipeline opens it, reads it in chunks, hashes it and uploads it. Faking the file object
    itself would leave nothing but the wiring proven.

    Every directory it creates is kept, so a test can ask afterwards whether it is still
    there. That is the cleanup assertion: the fake removes the directory on the way out of
    the context manager exactly as the real downloader does, so a route that never left the
    block, or never entered it as a context manager at all, leaves the directory behind.
    """

    def __init__(
        self,
        *,
        filename="media.mp4",
        payload=DOWNLOADED_BYTES,
        error=None,
        assembled=False,
    ):
        self.filename = filename
        self.payload = payload
        self.error = error
        # What the real downloader reports about the acquisition: False for one file a
        # source served, True for one muxed here from a separate video and audio stream.
        self.assembled = assembled
        self.urls = []
        self.directories = []

    @contextmanager
    def __call__(self, url: str):
        self.urls.append(url)

        if self.error is not None:
            raise self.error

        directory = Path(tempfile.mkdtemp(prefix="deepguard-test-download-"))
        self.directories.append(directory)
        path = directory / self.filename
        path.write_bytes(self.payload)

        try:
            yield DownloadedMedia(
                path=path,
                filename=path.name,
                size_bytes=len(self.payload),
                assembled=self.assembled,
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    @property
    def called(self) -> bool:
        return bool(self.urls)

    def left_nothing_behind(self) -> bool:
        return all(not directory.exists() for directory in self.directories)


@pytest.fixture
def download(monkeypatch):
    """The default fake: one MP4 that downloads successfully."""
    fake = RecordingDownloader()
    monkeypatch.setattr(downloader, "download", fake)

    return fake


@pytest.fixture
def assembled_download(monkeypatch):
    """A fake whose acquisition was merged here from separate streams.

    What a YouTube DASH or HLS submission produces. The bytes are the same pretend video —
    the difference this fixture exists for is not in the file, it is in what is true *about*
    the file, which is the fact the pipeline now has to carry.
    """
    fake = RecordingDownloader(assembled=True)
    monkeypatch.setattr(downloader, "download", fake)

    return fake


@pytest.fixture
def failing_download(monkeypatch):
    """A fake whose failure a test chooses."""

    def fail(error: DownloadError) -> RecordingDownloader:
        fake = RecordingDownloader(error=error)
        monkeypatch.setattr(downloader, "download", fake)

        return fake

    return fail


class FakeMinio:
    """Stand-in for the object store, so submission needs no live MinIO."""

    def __init__(self) -> None:
        self.uploads = []

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        pass

    def fput_object(self, bucket, key, file_path, content_type=None):
        self.uploads.append((bucket, key, Path(file_path).read_bytes(), content_type))


@pytest.fixture
def fake_minio(monkeypatch):
    fake = FakeMinio()
    monkeypatch.setattr(storage, "client", fake)

    return fake


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch):
    """Replace only the subprocess call, so real parsing and validation still run."""
    probe = json.dumps(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "12.34",
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "12.34",
                "tags": {"major_brand": "mp42"},
            },
        }
    )

    async def run(path):
        return probe

    monkeypatch.setattr(media, "_run_ffprobe", run)


# --- the fake session --------------------------------------------------------------------


class SubmissionSession:
    """Records what the route persists, and authenticates one key or nobody.

    The same shape `test_public_api.py` uses, and for the same reason: what these tests are
    about is the route, and a database would add nothing to the question of whether a
    downloaded file reaches the pipeline. `active` is a dial that is never moved — the limit
    is proven against real PostgreSQL below, where it can be.
    """

    def __init__(self, api_key: ApiKey | None = None) -> None:
        self.api_key = api_key
        self.active = 0
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement):
        session = self

        class Result:
            def scalar_one_or_none(self):
                return session.api_key

            def scalar_one(self):
                return session.active

        return Result()

    def add(self, instance):
        self.added.append(instance)

    def flush(self):
        for instance in self.added:
            if getattr(instance, "id", None) is None:
                instance.id = uuid.uuid4()

    def commit(self):
        self.flush()
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@contextmanager
def client_over(session: SubmissionSession, user: User | None = None):
    app.dependency_overrides[get_session] = lambda: session
    if user is not None:
        app.dependency_overrides[require_user] = lambda: user

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def dashboard_account() -> User:
    """A signed-in account for the internal route, never persisted."""
    return User(
        id=uuid.uuid4(),
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash="unused",
        role=USER_ROLE_USER,
        is_active=True,
    )


@pytest.fixture
def dashboard_user():
    """The account the internal submissions below are made as."""
    return dashboard_account()


@pytest.fixture
def dashboard(dashboard_user):
    """A client over the internal route, signed in as an ordinary account.

    Since R1-T2 that route demands a web session and records the account as the analysis's
    owner, so the submissions below are made as somebody — which is the state every test
    here was already describing when it said "the dashboard".
    """
    session = SubmissionSession()

    with client_over(session, dashboard_user) as client:
        yield client, session


@pytest.fixture
def anonymous_dashboard():
    """A client over the internal route with no session at all."""
    session = SubmissionSession()

    with client_over(session) as client:
        yield client, session


@pytest.fixture
def submitted_key():
    generated = generate_api_key()
    key = ApiKey(
        id=uuid.uuid4(),
        name="acme-production",
        key_hash=generated.key_hash,
        is_active=True,
    )

    return generated.plaintext, key


@pytest.fixture
def customer(submitted_key):
    """A client over the public route whose session authenticates `submitted_key`."""
    _, key = submitted_key
    session = SubmissionSession(api_key=key)

    with client_over(session) as client:
        yield client, session


@pytest.fixture
def anonymous():
    """A client over the public route whose session authenticates nobody."""
    session = SubmissionSession(api_key=None)

    with client_over(session) as client:
        yield client, session


def persisted(session: SubmissionSession, model):
    return [row for row in session.added if isinstance(row, model)]


def submit(
    client,
    url: str = SUBMITTED_URL,
    key: str | None = None,
    path=INTERNAL_URL,
    origin: str | None = DASHBOARD_ORIGIN,
):
    """One URL submission, to either surface.

    The `Origin` is what a browser attaches to a dashboard POST, and since R1-T2 the internal
    route refuses a submission without an accepted one. It is a parameter rather than a fixed
    header so the CSRF tests can send a foreign origin or none at all, and so the public
    submissions below can be made the way a customer's server actually makes them — with no
    origin at all, which that surface neither needs nor looks at.
    """
    headers = authorization(key) if key else {}
    if origin is not None:
        headers["Origin"] = origin

    return client.post(path, json={"url": url}, headers=headers)


# --- the internal route ------------------------------------------------------------------


def test_a_url_submission_is_accepted_and_queued(dashboard, download, fake_minio):
    client, session = dashboard

    response = submit(client)

    assert response.status_code == 202
    assert response.json()["status"] == ANALYSIS_STATUS_QUEUED
    assert download.urls == [SUBMITTED_URL]


def test_the_submitted_url_media_goes_through_the_upload_pipeline(
    dashboard, download, fake_minio
):
    """The rows and the object a file upload produces, produced by a URL instead.

    This is the whole point of P10: a downloaded video is a video. An analysis, its media
    row and its queued job are committed in one transaction, and the downloaded bytes are
    in the forensic bucket — none of which this route does itself. If a URL submission ever
    grew a pipeline of its own, this is the test that would still pass while the two drifted
    apart, so it asserts the bytes as well as the rows.
    """
    client, session = dashboard

    response = submit(client)

    assert response.status_code == 202
    assert len(persisted(session, Analysis)) == 1
    assert len(persisted(session, AnalysisJob)) == 1
    assert session.commits == 1

    media_row = persisted(session, MediaFile)[0]
    assert media_row.content_type == "video/mp4"
    assert media_row.size_bytes == len(DOWNLOADED_BYTES)
    # ffprobe ran on the downloaded file and its findings were persisted, as for an upload.
    assert media_row.codec_name == "h264"

    (_, _, uploaded, content_type) = fake_minio.uploads[0]
    assert uploaded == DOWNLOADED_BYTES
    assert content_type == "video/mp4"


# --- how the media was acquired, and that the answer survives the request (R7-T1) ---------
#
# The downloader knows whether it assembled the artifact and then goes out of scope. Every
# process that looks at the stored original afterwards — the worker, the C2PA read, a report
# rendered days later — has only the database. So the fact is written to `media_files`, and
# these tests are about it arriving there and staying an acquisition fact.


def test_a_progressive_acquisition_is_persisted_as_not_assembled(
    dashboard, download, fake_minio
):
    """The path every non-YouTube URL takes, and the one that preserves bytes.

    One muxed file, served and stored unchanged, so the stored original really is what the
    source published — which is the claim `was_assembled = false` is there to support.
    """
    client, session = dashboard

    assert submit(client).status_code == 202

    assert persisted(session, MediaFile)[0].was_assembled is False


def test_a_merged_acquisition_is_persisted_as_assembled(
    dashboard, assembled_download, fake_minio
):
    """The DASH/HLS path, which is the only one that can produce an artifact nobody served.

    The row has to say so. Without this column the stored original is indistinguishable from
    a published file, and everything downstream reads it as one.
    """
    client, session = dashboard

    assert submit(client).status_code == 202

    assert persisted(session, MediaFile)[0].was_assembled is True


def test_the_acquisition_is_reported_on_the_response(
    dashboard, assembled_download, fake_minio
):
    """Reported rather than left implicit, on the same response that reports the hash.

    The two belong together: a hash of an assembled artifact is a real, checkable identity
    for the thing DeepGuard stored, and it is not the hash of anything a publisher issued.
    A response that gave the first without the second would invite exactly that reading.
    """
    client, _ = dashboard

    body = submit(client).json()

    assert body["was_assembled"] is True
    assert body["sha256"]


def test_the_acquisition_fact_never_reaches_the_risk_engine():
    """Rule 3 of the review, pinned as a test rather than as a promise in a comment.

    Assembled media is neither more nor less likely to be manipulated than served media. An
    ffmpeg mux is DeepGuard's own plumbing, and a rule that read suspicion into it would be
    manufacturing evidence out of how the file was fetched.

    Asserted two ways, because the comment saying so is not enforcement: the engine takes
    exactly its three calibrated detector evidences and nothing else, and its source never
    mentions the acquisition at all — so a rule could not consult it even if one wanted to.
    """
    assert set(inspect.signature(risk_engine.evaluate).parameters) == {"svd", "face", "lip"}
    assert "assembled" not in inspect.getsource(risk_engine)


def test_the_response_reports_what_an_upload_reports(dashboard, download, fake_minio):
    """The internal response model, unchanged: the dashboard reads one shape, not two."""
    client, session = dashboard

    body = submit(client).json()

    assert set(body) == {
        "id",
        "status",
        "filename",
        "content_type",
        "size_bytes",
        "sha256",
        "storage_key",
        "metadata",
        "was_normalized",
        "was_assembled",
        "derivative_storage_key",
        "derivative_sha256",
    }
    assert body["size_bytes"] == len(DOWNLOADED_BYTES)


def test_the_analysis_is_owned_by_the_signed_in_account(
    dashboard, dashboard_user, download, fake_minio
):
    """The internal route records the account that submitted the URL, and no API key.

    The same pairing the upload route holds to, because a URL is a second door into one
    pipeline and not a second kind of analysis: `api_key_id` stays null, which is what keeps
    what the dashboard commits out of every public read, and `owner_id` is what keeps it out
    of every other account's dashboard.
    """
    client, session = dashboard

    submit(client)

    analysis = persisted(session, Analysis)[0]
    assert analysis.owner_id == dashboard_user.id
    assert analysis.api_key_id is None


def test_an_unauthenticated_url_submission_is_refused_before_anything_is_fetched(
    anonymous_dashboard, download, fake_minio
):
    """No session, no download. The refusal precedes the fetch this server would make.

    That order is the whole point on this route: an unauthenticated caller who could make
    the request run yt-dlp would have an outbound fetch primitive without an account, which
    is worth more to an attacker than the analysis it would have produced.
    """
    client, session = anonymous_dashboard

    response = submit(client)

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert not download.called
    assert session.added == []


def test_the_downloaded_file_is_removed_after_a_successful_analysis(
    dashboard, download, fake_minio
):
    client, _ = dashboard

    assert submit(client).status_code == 202
    assert download.directories
    assert download.left_nothing_behind()


def test_the_downloaded_file_is_removed_when_the_pipeline_refuses_it(
    dashboard, download, fake_minio, monkeypatch
):
    """A refusal from inside the pipeline still leaves nothing on disk.

    The failure is made to happen after the download, which is the case worth pinning: the
    cleanup on the error path is a different path through the route from the cleanup on the
    happy one, and a container that accumulated every rejected video would fill up on
    exactly the submissions nobody was watching.
    """
    client, _ = dashboard

    async def not_video(path):
        return json.dumps({"streams": [], "format": {}})

    monkeypatch.setattr(media, "_run_ffprobe", not_video)

    assert submit(client).status_code == 422
    assert download.directories
    assert download.left_nothing_behind()


def test_a_mov_download_is_declared_as_quicktime(dashboard, monkeypatch, fake_minio):
    fake = RecordingDownloader(filename="media.mov")
    monkeypatch.setattr(downloader, "download", fake)
    client, session = dashboard

    assert submit(client).status_code == 202
    assert persisted(session, MediaFile)[0].content_type == "video/quicktime"


def test_a_download_in_another_container_is_refused(dashboard, monkeypatch, fake_minio):
    """A site that served something other than MP4 or MOV is a 415, not an analysis.

    The downloader asks for one already-muxed file and prefers MP4, but the source decides
    what it serves. Admitting whatever came back would make the URL door accept containers
    the upload door refuses.
    """
    fake = RecordingDownloader(filename="media.webm")
    monkeypatch.setattr(downloader, "download", fake)
    client, session = dashboard

    response = submit(client)

    assert response.status_code == 415
    assert session.added == []
    assert fake_minio.uploads == []
    assert fake.left_nothing_behind()


def test_the_url_door_admits_no_type_the_upload_door_refuses():
    """The two admission sets are pinned together rather than kept in step by hand."""
    assert set(URL_MEDIA_TYPES.values()) <= ALLOWED_CONTENT_TYPES


# --- refusing safely ---------------------------------------------------------------------
#
# Every download failure the caller can cause, and what it is told about it. The messages
# the downloader raises name hosts, extractors and resolved addresses; none of that may
# appear in a response.


LEAKY_TERMS = (
    "127.0.0.1",
    "minio",
    "Traceback",
    "yt_dlp",
    "getaddrinfo",
    "socket",
    "resolves",
)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (UnsupportedUrl("Only http and https URLs can be downloaded."), 400),
        (BlockedAddress("minio resolves to an address that is not public."), 400),
        (LiveStreamRejected("Live streams cannot be analysed."), 422),
        (MediaTooLarge("The downloaded media is 999999999 bytes, over the limit."), 413),
        (DownloadUnavailable("The URL could not be read as media."), 422),
    ],
)
def test_a_download_failure_is_a_safe_client_error(
    dashboard, failing_download, fake_minio, error, expected_status
):
    client, session = dashboard
    failing_download(error)

    response = submit(client)

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert not any(term in detail for term in LEAKY_TERMS)
    # Nothing was committed and nothing reached the bucket: the failure is before the
    # pipeline, not a half-made analysis.
    assert session.added == []
    assert session.commits == 0
    assert fake_minio.uploads == []


def test_a_blocked_address_is_indistinguishable_from_an_unsupported_url(
    dashboard, failing_download, fake_minio
):
    """The SSRF refusal must not become a scanner.

    A distinct answer for `BlockedAddress` would let a caller map the deployment's internal
    names by watching which message came back, so both refusals are the same status and the
    same body.
    """
    client, _ = dashboard

    failing_download(BlockedAddress("minio resolves to an address that is not public."))
    blocked = submit(client)

    failing_download(UnsupportedUrl("nowhere.invalid could not be resolved."))
    unsupported = submit(client)

    assert blocked.status_code == unsupported.status_code == 400
    assert blocked.json() == unsupported.json()


def test_an_oversized_download_names_the_limit(dashboard, failing_download, fake_minio):
    """The ceiling is worth telling the caller: it is the one thing they can act on."""
    client, _ = dashboard
    failing_download(MediaTooLarge("The downloaded media is far too large."))

    response = submit(client)

    assert response.status_code == 413
    assert str(media.MAX_UPLOAD_BYTES) in response.json()["detail"]


def test_an_unrecognized_download_error_is_still_a_client_error(
    dashboard, failing_download, fake_minio
):
    """A later `DownloadError` subclass reaches the caller as a refusal, not as a 500."""

    class FutureFailure(DownloadError):
        pass

    client, _ = dashboard
    failing_download(FutureFailure("something new went wrong"))

    response = submit(client)

    assert response.status_code == 422
    assert "something new" not in response.json()["detail"]


@pytest.mark.parametrize("url", ["", "x" * (MAX_URL_LENGTH + 1)])
def test_an_unusable_url_string_is_refused_before_anything_is_fetched(
    dashboard, download, fake_minio, url
):
    client, _ = dashboard

    response = submit(client, url=url)

    assert response.status_code == 422
    assert not download.called


def test_a_body_without_a_url_is_refused(dashboard, download, fake_minio):
    client, _ = dashboard

    response = client.post(
        INTERNAL_URL, json={}, headers={"Origin": DASHBOARD_ORIGIN}
    )

    assert response.status_code == 422
    assert not download.called


# --- the public route --------------------------------------------------------------------


def test_a_public_url_submission_without_a_key_is_refused(anonymous, download, fake_minio):
    client, session = anonymous

    response = submit(client, key=None, path=PUBLIC_URL)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_an_unauthenticated_submission_never_downloads_anything(
    anonymous, download, fake_minio
):
    """The gate is in front of the download, not behind it.

    A route that fetched the URL and refused afterwards would still answer 401 — while
    having made this server fetch a URL for an unauthenticated caller, which is the request
    forgery the downloader's own guards exist to bound. The cheapest place to stop that is
    before any of it runs.
    """
    client, session = anonymous

    submit(client, key=None, path=PUBLIC_URL)

    assert not download.called
    assert session.added == []
    assert fake_minio.uploads == []


def test_a_public_url_submission_with_an_unknown_key_is_refused(
    anonymous, download, fake_minio
):
    client, _ = anonymous

    response = submit(client, key=generate_api_key().plaintext, path=PUBLIC_URL)

    assert response.status_code == 401
    assert not download.called


def test_a_public_url_submission_is_accepted_and_owned(
    customer, submitted_key, download, fake_minio
):
    client, session = customer
    plaintext, key = submitted_key

    response = submit(client, key=plaintext, path=PUBLIC_URL)

    assert response.status_code == 202
    analysis = persisted(session, Analysis)[0]
    # Written in the same insert as the analysis, which is what every public read filters on.
    assert analysis.api_key_id == key.id
    assert response.json() == {"id": str(analysis.id), "status": ANALYSIS_STATUS_QUEUED}


def test_the_public_url_response_carries_nothing_internal(
    customer, submitted_key, download, fake_minio
):
    """Two fields, exactly as the public upload answers: no storage key, no content identity."""
    client, _ = customer
    plaintext, _ = submitted_key

    body = submit(client, key=plaintext, path=PUBLIC_URL).json()

    assert set(body) == {"id", "status"}


def test_a_public_download_failure_is_also_safe(
    customer, submitted_key, failing_download, fake_minio
):
    client, session = customer
    plaintext, _ = submitted_key
    failing_download(BlockedAddress("minio resolves to an address that is not public."))

    response = submit(client, key=plaintext, path=PUBLIC_URL)

    assert response.status_code == 400
    assert not any(term in response.json()["detail"] for term in LEAKY_TERMS)
    assert session.added == []


# --- ownership and the concurrency limit ---------------------------------------------------
#
# Real PostgreSQL. Ownership is a `WHERE` clause and the limit is a lock taken inside the
# transaction that writes the analysis; a fake session can be told to report either, which
# is exactly why neither is claimed from one.


@pytest.fixture(scope="module")
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def session(database):
    """A real session whose analyses and keys are removed again, in that order.

    Analyses first: `analyses.api_key_id` is `ON DELETE RESTRICT`, so a key cannot be
    deleted while an analysis still names it.
    """
    analyses = []
    keys = []

    with SessionLocal() as db:
        yield db, analyses, keys

        db.rollback()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.flush()
        for key_id in keys:
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()


@pytest.fixture
def live(session):
    """A client over the product app bound to the live session."""
    db, _, _ = session
    app.dependency_overrides[get_session] = lambda: db

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def issue_key(session, *, name="url-customer") -> tuple[str, ApiKey]:
    db, _, keys = session
    generated = generate_api_key()

    key = ApiKey(name=name, key_hash=generated.key_hash, is_active=True)
    db.add(key)
    db.commit()
    keys.append(key.id)

    return generated.plaintext, key


def fill_active(session, key: ApiKey, count: int) -> None:
    """Give a key `count` outstanding analyses, as a real submission leaves them."""
    db, analyses, _ = session

    for _ in range(count):
        analysis = Analysis(status=ANALYSIS_STATUS_QUEUED, api_key_id=key.id)
        db.add(analysis)
        db.commit()
        analyses.append(analysis.id)


@integration
def test_the_acquisition_survives_the_request_that_wrote_it(
    live, session, assembled_download, fake_minio
):
    """Against real PostgreSQL, read back on a statement the request never touched.

    The point of the column is that the answer outlives the process that knew it. The
    downloader reported `assembled` inside a request that has since ended; the worker will
    open this analysis minutes later and has nothing but the row. So the row is expired from
    the session and fetched again, which is as close as a test gets to being the next process.

    The queued job is asserted alongside it because that is the lifecycle the fact has to
    survive: an analysis is committed with its media and its job in one transaction, and a
    column written outside that transaction would be a fact that could go missing.
    """
    db, analyses, _ = session
    plaintext, key = issue_key(session)

    response = submit(live, key=plaintext, path=PUBLIC_URL)

    assert response.status_code == 202
    analysis_id = uuid.UUID(response.json()["id"])
    analyses.append(analysis_id)

    # Nothing of this request's identity map survives into the reads below.
    db.expire_all()

    stored = db.query(MediaFile).filter(MediaFile.analysis_id == analysis_id).one()
    assert stored.was_assembled is True

    job = db.query(AnalysisJob).filter(AnalysisJob.analysis_id == analysis_id).one()
    assert job.status == JOB_STATUS_QUEUED
    # The analysis reached the queue with no risk decision, exactly as any other does. How
    # the media was acquired changed nothing about the work it is owed.
    assert db.query(Analysis).filter(Analysis.id == analysis_id).one().risk_level is None


@integration
def test_a_progressive_acquisition_survives_as_not_assembled(
    live, session, download, fake_minio
):
    """The backward-compatible half, on the same live database.

    Everything submitted before R7-T1 was acquired this way, and the server default that
    backfilled those rows says the same thing this write does.
    """
    db, analyses, _ = session
    plaintext, _ = issue_key(session)

    response = submit(live, key=plaintext, path=PUBLIC_URL)

    assert response.status_code == 202
    analysis_id = uuid.UUID(response.json()["id"])
    analyses.append(analysis_id)
    db.expire_all()

    stored = db.query(MediaFile).filter(MediaFile.analysis_id == analysis_id).one()
    assert stored.was_assembled is False


@integration
def test_a_url_submission_below_the_limit_is_accepted(
    live, session, download, fake_minio
):
    db, analyses, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES - 1)

    response = submit(live, key=plaintext, path=PUBLIC_URL)

    assert response.status_code == 202
    analyses.append(uuid.UUID(response.json()["id"]))
    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES


@integration
def test_a_url_submission_at_the_limit_is_throttled(live, session, download, fake_minio):
    """A URL occupies a slot exactly as an upload does, and the sixth is refused."""
    db, _, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES)

    response = submit(live, key=plaintext, path=PUBLIC_URL)

    assert response.status_code == 429
    assert str(MAX_ACTIVE_ANALYSES) in response.json()["detail"]
    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES


@integration
def test_a_throttled_url_submission_persists_nothing(live, session, download, fake_minio):
    db, _, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES)

    submit(live, key=plaintext, path=PUBLIC_URL)

    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES
    # The download happened before the limit was consulted, and it left nothing behind.
    assert download.left_nothing_behind()


@integration
def test_the_url_limit_counts_uploads_too(live, session, download, fake_minio):
    """One ceiling per key over both doors, because there is one queue behind them.

    A limit that counted URL submissions separately would let a customer run twice the work
    by alternating, which is the starvation the cap exists to prevent.
    """
    db, analyses, _ = session
    plaintext, key = issue_key(session, name="mixed-doors")
    fill_active(session, key, MAX_ACTIVE_ANALYSES - 1)

    upload = live.post(
        PUBLIC_UPLOAD_URL,
        files={"file": ("clip.mp4", b"upload-bytes", "video/mp4")},
        headers=authorization(plaintext),
    )
    assert upload.status_code == 202
    analyses.append(uuid.UUID(upload.json()["id"]))

    throttled = submit(live, key=plaintext, path=PUBLIC_URL)

    assert throttled.status_code == 429
    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES


@integration
def test_a_url_analysis_is_only_readable_by_the_key_that_submitted_it(
    live, session, download, fake_minio
):
    """Ownership isolation, on an analysis that arrived by URL rather than by upload."""
    _, analyses, _ = session
    plaintext, _ = issue_key(session, name="url-owner")
    other, _ = issue_key(session, name="url-stranger")

    submitted = submit(live, key=plaintext, path=PUBLIC_URL)
    assert submitted.status_code == 202
    analysis_id = submitted.json()["id"]
    analyses.append(uuid.UUID(analysis_id))

    mine = live.get(f"{PUBLIC_UPLOAD_URL}/{analysis_id}", headers=authorization(plaintext))
    theirs = live.get(f"{PUBLIC_UPLOAD_URL}/{analysis_id}", headers=authorization(other))

    assert mine.status_code == 200
    assert mine.json()["id"] == analysis_id
    # The same answer an id that was never issued gets.
    assert theirs.status_code == 404
    assert theirs.json() == {"detail": "analysis not found"}
