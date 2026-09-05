import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from minio.error import S3Error

from app import detection, media, normalization, request_limits, storage
from app.api.analyses import CHUNK_SIZE, TEMP_FILE_PREFIX
from app.media import MAX_UPLOAD_BYTES
from app.db.models import USER_ROLE_USER, Analysis, AnalysisJob, MediaFile, User
from app.db.session import get_session
from app.main import app
from app.normalization import DERIVATIVE_TEMP_PREFIX
from app.observability import REQUEST_ID_HEADER
from app.web_auth import require_user
from tests.conftest import DASHBOARD_ORIGIN


class FakeMinio:
    """Stand-in for the MinIO client, so the suite never needs a live object store."""

    def __init__(self, *, bucket_exists=True, failure=None):
        self._bucket_exists = bucket_exists
        self._failure = failure
        self.created_buckets = []
        self.uploads = []
        self.uploaded_bytes = {}
        # Content-addressed objects are never deleted; this proves nothing tries to.
        self.removed = []
        # Number of uploads to let through before failing, for the derivative path.
        self.upload_failure_after = None

    @property
    def stored_keys(self):
        return [key for _, key, _, _ in self.uploads]

    def bucket_exists(self, bucket):
        if self._failure:
            raise self._failure
        return self._bucket_exists

    def make_bucket(self, bucket):
        self.created_buckets.append(bucket)
        self._bucket_exists = True

    def fput_object(self, bucket, key, file_path, content_type=None):
        if self._failure:
            raise self._failure
        if self.upload_failure_after is not None and len(self.uploads) >= self.upload_failure_after:
            raise _s3_error("InternalError")
        # Captured while the file still exists: the request deletes its temp files
        # before responding.
        self.uploaded_bytes[key] = Path(file_path).read_bytes()
        self.uploads.append((bucket, key, file_path, content_type))

    def remove_object(self, bucket, key):
        self.removed.append((bucket, key))


@pytest.fixture
def fake_minio(monkeypatch):
    fake = FakeMinio()
    monkeypatch.setattr(storage, "client", fake)
    return fake


def ffprobe_output(
    *,
    streams=None,
    codec_name="h264",
    width=1920,
    height=1080,
    pix_fmt="yuv420p",
    # The default describes constant-rate media: ffprobe reports the same averaged and
    # base rate. `None` drops the entry, as it does for genuinely absent fields.
    avg_frame_rate="30/1",
    r_frame_rate="30/1",
    stream_duration="12.34",
    format_name="mov,mp4,m4a,3gp,3g2,mj2",
    format_duration="12.34",
    # The default describes a real MP4 container. `None` drops the tag, as a container
    # that declares no brand does.
    major_brand="mp42",
    **stream_extra,
) -> str:
    """Build ffprobe JSON of the shape the real binary emits for the requested entries."""
    if streams is None:
        optional = {
            "pix_fmt": pix_fmt,
            "avg_frame_rate": avg_frame_rate,
            "r_frame_rate": r_frame_rate,
        }
        streams = [
            {
                "codec_name": codec_name,
                "width": width,
                "height": height,
                "duration": stream_duration,
                **{key: value for key, value in optional.items() if value is not None},
                **stream_extra,
            }
        ]

    container = {"format_name": format_name, "duration": format_duration}
    if major_brand is not None:
        container["tags"] = {"major_brand": major_brand}

    return json.dumps({"streams": streams, "format": container})


@pytest.fixture
def fake_ffprobe(monkeypatch):
    """Replace only the subprocess call, so real parsing and validation still run."""

    class Recorder:
        def __init__(self):
            self.output = ffprobe_output()
            self.error = None
            self.paths = []
            self.probed_bytes = []

        async def run(self, path):
            self.paths.append(Path(path))
            # Captured here because the request deletes the file before responding.
            self.probed_bytes.append(Path(path).read_bytes())
            if self.error:
                raise self.error
            return self.output

    recorder = Recorder()
    monkeypatch.setattr(media, "_run_ffprobe", recorder.run)
    return recorder


DERIVATIVE_BYTES = b"normalized-mp4-bytes"


@pytest.fixture(autouse=True)
def fake_ffmpeg(monkeypatch):
    """A tripwire on the transcoder, not a stand-in for one.

    Since P4-F2 the upload decides whether a derivative is needed and never produces one,
    so nothing in this module should reach this fixture. It is autouse for the same reason
    `fake_nvidia` is: a regression that put transcoding back on the request would
    otherwise spend minutes of real ffmpeg time inside the suite, and `calls` staying
    empty is what proves the route left it alone. What normalization does when it really
    runs is `tests/test_worker.py`.
    """

    class Recorder:
        def __init__(self):
            self.calls = []

        async def run(self, source, destination, frame_rate):
            self.calls.append((Path(source), Path(destination), frame_rate))
            raise AssertionError("The upload route must not transcode.")

    recorder = Recorder()
    monkeypatch.setattr(normalization, "_run_ffmpeg", recorder.run)
    return recorder


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """A tripwire on the detector, not a stand-in for one.

    The upload does not detect anything any more, so nothing in this module should reach
    this fixture at all. It stays autouse because a regression that put inference back on
    the request path would otherwise mean real credentials, real network traffic and a
    real bill from the suite — and `paths` staying empty is what proves the route left it
    alone. What detection does when it is called is `tests/test_detection.py`.
    """

    class Recorder:
        def __init__(self):
            self.paths = []

        async def analyze(self, file_path, **kwargs):
            self.paths.append(Path(file_path))
            raise AssertionError("The upload route must not call the detector.")

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_video", recorder.analyze)
    return recorder


class FakeSession:
    """Stand-in for a SQLAlchemy session, so the suite needs no live database.

    The real persistence behaviour — which rows are built, in which order, and what is
    committed or rolled back — is still the code under test. Only the driver is faked;
    `tests/test_persistence.py` covers the schema against real PostgreSQL.
    """

    def __init__(self, *, commit_error=None):
        self.commit_error = commit_error
        self.rollback_error = None
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def add(self, instance):
        self.added.append(instance)

    def add_all(self, instances):
        self.added.extend(instances)

    def flush(self):
        # The real flush is what assigns the Python-side UUID defaults.
        for instance in self.added:
            if getattr(instance, "id", None) is None:
                instance.id = uuid.uuid4()

    def commit(self):
        if self.commit_error is not None:
            raise self.commit_error
        self.flush()
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1
        if self.rollback_error is not None:
            raise self.rollback_error


@pytest.fixture
def fake_session():
    session = FakeSession()
    app.dependency_overrides[get_session] = lambda: session
    yield session
    app.dependency_overrides.clear()


@pytest.fixture
def dashboard_user():
    """The signed-in account these uploads are submitted by.

    Never persisted: the route reads the id off it and writes that id onto the analysis, and
    the session here is a fake that records what it was given rather than a database that
    would hold it to a foreign key. `tests/test_dashboard_authorization.py` does the same
    thing against real rows.
    """
    return User(
        id=uuid.uuid4(),
        email="operator@example.com",
        password_hash="unused",
        role=USER_ROLE_USER,
        is_active=True,
    )


@pytest.fixture
def client(fake_session, dashboard_user):
    app.dependency_overrides[require_user] = lambda: dashboard_user

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def anonymous_client(fake_session):
    """A client carrying no session, for the tests about the upload being refused one."""
    with TestClient(app) as test_client:
        yield test_client


def _temp_uploads() -> set[Path]:
    """Every temp artifact a DeepGuard request can create — original and derivative."""
    temp_dir = Path(tempfile.gettempdir())

    return set(temp_dir.glob(f"{TEMP_FILE_PREFIX}*")) | set(
        temp_dir.glob(f"{DERIVATIVE_TEMP_PREFIX}*")
    )


@pytest.fixture
def new_temp_uploads():
    """Expose the temp files a request left behind, and clean up if one survives.

    From P1-T5 on, a finished request must leave none: the cleanup here is a safety net
    that keeps one failure from leaking into the next test, not expected behavior.
    """
    before = _temp_uploads()

    yield lambda: sorted(_temp_uploads() - before)

    for path in _temp_uploads() - before:
        path.unlink(missing_ok=True)


def _s3_error(code: str) -> S3Error:
    return S3Error(
        code=code,
        message=code,
        resource=storage.ORIGINALS_BUCKET,
        request_id=None,
        host_id=None,
        response=None,
    )


def post_upload(client: TestClient, filename: str, payload, content_type: str):
    """One dashboard upload, as the web application makes it.

    The `Origin` is part of that: since R1-T2 the route refuses a submission that did not
    come from the deployment's own dashboard, and a browser sends the header on every POST.
    Leaving it out here would test a request no browser makes — which is what the CSRF tests
    below deliberately do, and what every other test in this file should not have to.
    """
    return client.post(
        "/api/v1/analyses",
        files={"file": (filename, payload, content_type)},
        headers={"Origin": DASHBOARD_ORIGIN},
    )


def test_an_unauthenticated_upload_is_refused_before_anything_is_read(
    anonymous_client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    """No session, no upload — and nothing stored, staged or probed on the way to the 401.

    The refusal comes from a dependency, so the route body never runs: nothing reaches object
    storage and nothing is staged in one of DeepGuard's own temp files. An upload route that
    stored 100 MiB and then decided the caller was not signed in would be a way to spend this
    server's disk and object storage without an account.
    """
    response = post_upload(anonymous_client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert fake_session.added == []
    assert fake_minio.uploads == []
    assert new_temp_uploads() == []


def test_the_upload_is_owned_by_the_signed_in_account(
    client, fake_session, dashboard_user, new_temp_uploads, fake_minio, fake_ffprobe
):
    """The analysis names the account the session resolved to, and no API key.

    Both halves are the point. The owner is written because the dashboard's isolation rests
    on it, and `api_key_id` stays null because the database refuses a row that carries both
    — an analysis owned by a user *and* a key would be reachable through the public API by a
    customer who never submitted it.
    """
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    persisted = next(row for row in fake_session.added if isinstance(row, Analysis))
    assert persisted.owner_id == dashboard_user.id
    assert persisted.api_key_id is None


def test_the_upload_carries_no_owner_field_a_caller_could_set(
    client, fake_session, dashboard_user, new_temp_uploads, fake_minio, fake_ffprobe
):
    """An `owner_id` in the form is ignored, because the route has nowhere to read one.

    Asserted rather than assumed: this is the shape of the mistake that would let any
    signed-in person file an analysis under somebody else's account, and it costs one test
    to keep a later "just accept the field" from passing silently.
    """
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("clip.mp4", b"payload", "video/mp4")},
        data={"owner_id": str(uuid.uuid4())},
        headers={"Origin": DASHBOARD_ORIGIN},
    )

    assert response.status_code == 202
    persisted = next(row for row in fake_session.added if isinstance(row, Analysis))
    assert persisted.owner_id == dashboard_user.id


def test_declared_mp4_is_accepted(client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe):
    payload = b"probe-says-this-is-video"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 202
    sha256 = hashlib.sha256(payload).hexdigest()
    persisted = next(row for row in fake_session.added if isinstance(row, Analysis))
    assert response.json() == {
        "id": str(persisted.id),
        "status": "queued",
        "filename": "clip.mp4",
        "content_type": "video/mp4",
        "size_bytes": len(payload),
        "sha256": sha256,
        "storage_key": f"originals/{sha256}",
        "metadata": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "major_brand": "mp42",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "duration": 12.34,
            "frame_rate": 30.0,
            "pix_fmt": "yuv420p",
            "constant_frame_rate": True,
        },
        "was_normalized": False,
        # An upload is never assembled: the client sends one file and the pipeline stores
        # exactly it. Only a URL acquisition can be anything else.
        "was_assembled": False,
        "derivative_storage_key": f"originals/{sha256}",
        "derivative_sha256": None,
    }


def test_declared_quicktime_is_accepted(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    response = post_upload(client, "clip.mov", b"0" * 4096, "video/quicktime")

    assert response.status_code == 202
    body = response.json()
    assert body["content_type"] == "video/quicktime"
    assert body["size_bytes"] == 4096


def test_response_does_not_leak_the_temp_path(client, new_temp_uploads, fake_minio, fake_ffprobe):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    assert set(response.json()) == {
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


def test_sha256_is_lowercase_hex_of_the_uploaded_bytes(client, new_temp_uploads, fake_minio, fake_ffprobe):
    # Multi-chunk payload, so an incremental hash is genuinely exercised.
    payload = bytes(range(256)) * (CHUNK_SIZE // 128)

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    sha256 = response.json()["sha256"]
    assert len(sha256) == 64
    assert sha256 == sha256.lower()
    assert sha256 == hashlib.sha256(payload).hexdigest()


def test_accepted_upload_is_staged_to_disk_intact(client, new_temp_uploads, fake_minio, fake_ffprobe):
    payload = bytes(range(256)) * (CHUNK_SIZE // 128)

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 202
    # Observed while the request still held the temp file, which it no longer keeps.
    assert fake_ffprobe.probed_bytes == [payload]


def test_unsupported_declared_mime_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    response = post_upload(client, "notes.txt", b"plain text", "text/plain")

    assert response.status_code == 415
    assert new_temp_uploads() == []


def test_upload_above_the_size_limit_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    oversized = MAX_UPLOAD_BYTES + 1

    # Spooled to disk so the test never holds the whole payload in memory either.
    with SpooledTemporaryFile(max_size=CHUNK_SIZE) as payload:
        written = 0
        while written < oversized:
            block = min(CHUNK_SIZE, oversized - written)
            payload.write(b"\0" * block)
            written += block
        payload.seek(0)

        response = post_upload(client, "big.mp4", payload, "video/mp4")

    assert response.status_code == 413
    # The route's own limit answered, not the request guard in front of it: the body is
    # a MAX_UPLOAD_BYTES + 1 file plus a few hundred bytes of multipart framing, which
    # stays inside the request limit's overhead margin. The two layers stay independent.
    assert response.json() == {
        "detail": f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit."
    }
    assert new_temp_uploads() == []


def test_an_oversized_request_body_is_bounded_before_the_upload_is_parsed(
    client, new_temp_uploads, fake_minio, fake_ffprobe, monkeypatch
):
    """The guard in front of FastAPI answers before an UploadFile is ever produced.

    The real request limit sits above the file limit, so the limit is shrunk here rather
    than posting a body large enough to cross it. What is under test is the layering, not
    the number: nothing behind the guard runs.
    """
    monkeypatch.setattr(request_limits, "MAX_REQUEST_BYTES", 1024)

    response = post_upload(client, "big.mp4", b"0" * 8192, "video/mp4")

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body exceeds the 1024 byte limit."}
    # No parsed upload, so no staged temp file, nothing stored and nothing probed.
    assert new_temp_uploads() == []
    assert fake_minio.uploads == []
    assert fake_ffprobe.paths == []


def test_admitted_upload_is_stored_in_minio(client, new_temp_uploads, fake_minio, fake_ffprobe):
    payload = b"forensic-original"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 202
    sha256 = hashlib.sha256(payload).hexdigest()
    bucket, key, file_path, content_type = fake_minio.uploads[0]
    assert bucket == storage.ORIGINALS_BUCKET == "deepguard-originals"
    assert key == f"originals/{sha256}"
    assert key == response.json()["storage_key"]
    assert content_type == "video/mp4"
    # The staged original is streamed from disk, unchanged.
    assert fake_minio.uploaded_bytes[key] == payload


def test_identical_bytes_resolve_to_the_same_storage_key(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    payload = b"same-bytes"

    first = post_upload(client, "a.mp4", payload, "video/mp4")
    second = post_upload(client, "b.mov", payload, "video/quicktime")

    assert first.json()["storage_key"] == second.json()["storage_key"]


def test_missing_bucket_is_created_before_upload(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_minio._bucket_exists = False

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    assert fake_minio.created_buckets == [storage.ORIGINALS_BUCKET]


def test_concurrently_created_bucket_is_tolerated(client, new_temp_uploads, fake_minio, fake_ffprobe, monkeypatch):
    fake_minio._bucket_exists = False
    monkeypatch.setattr(
        fake_minio,
        "make_bucket",
        lambda bucket: (_ for _ in ()).throw(_s3_error("BucketAlreadyOwnedByYou")),
    )

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202


def test_successful_storage_uploads_the_original_before_probing(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    payload = b"forensic-original"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 202
    assert fake_minio.stored_keys == [response.json()["storage_key"]]
    assert fake_minio.removed == []


def test_storage_failure_returns_a_controlled_503(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_minio._failure = _s3_error("InternalError")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 503
    assert response.json() == {"detail": "media storage unavailable"}


def test_storage_failure_removes_the_temp_file(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_minio._failure = OSError("connection refused")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 503
    assert new_temp_uploads() == []


def test_probe_runs_against_the_staged_original(client, new_temp_uploads, fake_minio, fake_ffprobe):
    payload = b"forensic-original"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 202
    # D013: the probe source is the untouched local original, not a re-download.
    assert len(fake_ffprobe.paths) == 1
    assert fake_ffprobe.paths[0].name.startswith(TEMP_FILE_PREFIX)
    assert fake_ffprobe.probed_bytes == [payload]


def test_integer_frame_rate_is_reported_as_a_number(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = ffprobe_output(avg_frame_rate="25/1", r_frame_rate="25/1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["metadata"]["frame_rate"] == 25.0


def test_fractional_frame_rate_keeps_its_precision(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = ffprobe_output(avg_frame_rate="30000/1001", r_frame_rate="30000/1001")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["metadata"]["frame_rate"] == pytest.approx(30000 / 1001, rel=1e-12)


def test_unknown_average_frame_rate_falls_back_to_r_frame_rate(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # ffprobe reports `0/0` when it cannot average the rate.
    fake_ffprobe.output = ffprobe_output(avg_frame_rate="0/0", r_frame_rate="24/1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["metadata"]["frame_rate"] == 24.0


def test_unusable_frame_rate_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    # Both rates unusable, which is what leaves the media with no frame rate at all. An
    # unusable average alone is not a rejection — `r_frame_rate` is the documented
    # fallback, covered by the test below.
    fake_ffprobe.output = ffprobe_output(avg_frame_rate="0/0", r_frame_rate="0/0")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422


def test_missing_stream_duration_falls_back_to_the_format_duration(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.output = ffprobe_output(stream_duration="N/A", format_duration="7.5")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    assert response.json()["metadata"]["duration"] == 7.5


def test_upload_without_any_usable_duration_is_rejected(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.output = ffprobe_output(stream_duration="N/A", format_duration="-1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422


def test_non_nvidia_compatible_video_is_still_accepted(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # Probing validates that the media is genuine video, not that a provider can consume
    # it; incompatible media is accepted and normalized rather than rejected.
    fake_ffprobe.output = ffprobe_output(codec_name="vp9", format_name="matroska,webm", major_brand=None)

    response = post_upload(client, "clip.mov", b"payload", "video/quicktime")

    assert response.status_code == 202
    metadata = response.json()["metadata"]
    assert metadata["codec_name"] == "vp9"
    assert metadata["format_name"] == "matroska,webm"


def test_ffprobe_failure_is_rejected_as_unprocessable(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.error = media.MediaProbeError("ffprobe exited with 1")

    response = post_upload(client, "fake.mp4", b"just some text", "video/mp4")

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid or unsupported video media"}


def test_malformed_ffprobe_json_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = "{not json"

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid or unsupported video media"}


def test_media_without_a_video_stream_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = ffprobe_output(streams=[])

    response = post_upload(client, "audio.mp4", b"payload", "video/mp4")

    assert response.status_code == 422


def test_zero_dimension_video_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = ffprobe_output(width=0)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422


def test_missing_codec_name_is_rejected(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = ffprobe_output(codec_name="")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422


def test_ffprobe_timeout_is_a_controlled_rejection(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.error = media.MediaProbeError("ffprobe timed out after 10s")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid or unsupported video media"}


def test_missing_ffprobe_binary_is_reported_as_a_server_failure(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.error = media.MediaProbeUnavailable("ffprobe could not be executed")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # The client's media may be perfectly fine; the server is missing its processor.
    assert response.status_code == 503
    assert response.json() == {"detail": "media processor unavailable"}


def test_invalid_media_removes_the_local_temp_file(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = "{not json"

    response = post_upload(client, "fake.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert new_temp_uploads() == []


def test_invalid_media_preserves_the_stored_original(client, new_temp_uploads, fake_minio, fake_ffprobe):
    fake_ffprobe.output = "{not json"

    response = post_upload(client, "fake.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    # The key is content-addressed: identical bytes stored by an earlier analysis
    # resolve to this same object, so deleting it could destroy that analysis's
    # forensic original.
    assert fake_minio.removed == []


def test_canonical_media_is_not_normalized(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    # MP4 container, H.264, yuv420p and a single reported frame rate: every fact the
    # provider needs is known, so transcoding would only cost time and fidelity.
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    assert response.json()["was_normalized"] is False
    assert fake_ffmpeg.calls == []


def test_canonical_media_does_not_create_a_duplicate_derivative(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    body = response.json()
    # Exactly one object: the original. No copy of it under a second key.
    assert fake_minio.stored_keys == [body["storage_key"]]
    assert body["derivative_storage_key"] == body["storage_key"]
    assert body["derivative_sha256"] is None


def test_quicktime_is_normalized_even_when_it_carries_h264(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # ffprobe reports the shared mov/mp4 demuxer for both, so only the brand separates
    # them; NVIDIA SVD takes MP4, so a QuickTime file is normalized.
    fake_ffprobe.output = ffprobe_output(major_brand="qt  ")

    response = post_upload(client, "clip.mov", b"payload", "video/quicktime")

    assert response.status_code == 202
    body = response.json()
    assert body["was_normalized"] is True
    # A derivative is owed, not made: the worker produces it and names it later.
    assert body["derivative_storage_key"] is None
    assert body["derivative_sha256"] is None


def test_quicktime_declared_as_mp4_is_still_normalized(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # A real MOV uploaded as `video/mp4`: the declaration is the client's claim, the
    # `qt  ` brand is the file's own evidence, and the evidence decides.
    fake_ffprobe.output = ffprobe_output(major_brand="qt  ")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    body = response.json()
    assert body["was_normalized"] is True
    assert body["metadata"]["major_brand"] == "qt"
    assert body["derivative_storage_key"] is None


def test_a_container_without_a_brand_is_normalized(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # Otherwise canonical, but the container proves nothing about what it is.
    fake_ffprobe.output = ffprobe_output(major_brand=None)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["was_normalized"] is True
    assert response.json()["metadata"]["major_brand"] is None


def test_webm_vp9_is_normalized(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9", format_name="matroska,webm", major_brand=None)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["was_normalized"] is True


def test_non_h264_mp4_is_normalized(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    fake_ffprobe.output = ffprobe_output(codec_name="hevc")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["was_normalized"] is True


def test_variable_frame_rate_is_normalized(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    # Averaged rate below the container's base rate is what varying frame timing looks
    # like to ffprobe.
    fake_ffprobe.output = ffprobe_output(avg_frame_rate="23.4/1", r_frame_rate="30/1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["was_normalized"] is True
    assert response.json()["metadata"]["constant_frame_rate"] is False


def test_unreported_base_frame_rate_is_normalized(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # Frame timing that cannot be established is never assumed to be constant.
    fake_ffprobe.output = ffprobe_output(r_frame_rate=None)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["was_normalized"] is True


def test_uncommon_pixel_format_is_normalized(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(pix_fmt="yuv444p")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["was_normalized"] is True


def test_success_leaves_no_local_temp_files(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    compatible = post_upload(client, "clip.mp4", b"payload", "video/mp4")
    assert compatible.json()["was_normalized"] is False

    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    normalized = post_upload(client, "clip.mp4", b"other-payload", "video/mp4")
    assert normalized.json()["was_normalized"] is True

    # No P1 step after this reads local media, so neither the original nor the
    # derivative may survive the request.
    assert new_temp_uploads() == []


def _added(session, model):
    return next(row for row in session.added if isinstance(row, model))


def test_successful_upload_persists_the_analysis_and_its_media(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    payload = b"forensic-original"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 202
    assert fake_session.commits == 1
    analysis = _added(fake_session, Analysis)
    media_file = _added(fake_session, MediaFile)
    assert analysis.status == "queued"
    assert media_file.analysis_id == analysis.id
    assert media_file.original_filename == "clip.mp4"
    assert media_file.content_type == "video/mp4"
    assert media_file.size_bytes == len(payload)
    assert media_file.original_sha256 == hashlib.sha256(payload).hexdigest()
    assert media_file.original_storage_key == response.json()["storage_key"]
    assert media_file.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
    assert media_file.codec_name == "h264"
    assert media_file.width == 1920
    assert media_file.height == 1080
    assert media_file.duration == 12.34
    assert media_file.frame_rate == 30.0
    assert media_file.pix_fmt == "yuv420p"
    assert media_file.constant_frame_rate is True
    assert media_file.was_normalized is False
    # An upload is never assembled (R7-T1): the client sends one file and this is it. Only a
    # URL acquisition can be anything else, and only when the source published no single file.
    assert media_file.was_assembled is False
    # No separate artifact exists, so the derivative carries no identity of its own.
    assert media_file.derivative_storage_key == media_file.original_storage_key
    assert media_file.derivative_sha256 is None


def test_response_returns_the_persisted_analysis_id(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.json()["id"] == str(_added(fake_session, Analysis).id)


def test_repeated_upload_of_identical_media_creates_a_second_analysis(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    payload = b"same-bytes"

    first = post_upload(client, "a.mp4", payload, "video/mp4")
    second = post_upload(client, "b.mp4", payload, "video/mp4")

    # Media identity is not analysis identity: the same file may be analysed again.
    assert first.json()["storage_key"] == second.json()["storage_key"]
    assert first.json()["id"] != second.json()["id"]
    assert fake_session.commits == 2


def test_an_upload_owed_a_derivative_persists_no_derivative_identity(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    media_file = _added(fake_session, MediaFile)
    # The decision is recorded because only the probe can make it; the artifact it names
    # does not exist yet, and a key written here would point at nothing.
    assert media_file.was_normalized is True
    assert media_file.derivative_storage_key is None
    assert media_file.derivative_sha256 is None


def test_canonical_media_persists_the_original_as_its_analysis_artifact(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    media_file = _added(fake_session, MediaFile)
    # No second artifact is ever produced for this one, so the answer is already known
    # and is the original's own key — not a placeholder waiting to be replaced.
    assert media_file.was_normalized is False
    assert media_file.derivative_storage_key == media_file.original_storage_key
    assert media_file.derivative_sha256 is None


def test_analysis_is_persisted_only_after_the_media_pipeline_succeeded(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    # Unusable media, which is the last thing that can still stop an upload now that
    # transcoding has moved off the request.
    fake_ffprobe.error = media.MediaProbeError("no video stream")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert fake_session.added == []
    assert fake_session.commits == 0


def _fail_commit(session):
    session.commit_error = OperationalError("INSERT", None, Exception("connection lost"))


def test_persistence_failure_returns_a_controlled_500(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    _fail_commit(fake_session)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500
    # No statement, connection string or driver detail reaches the client.
    assert response.json() == {"detail": "analysis could not be persisted"}


def test_persistence_failure_rolls_the_transaction_back(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    _fail_commit(fake_session)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500
    assert fake_session.rollbacks == 1
    assert fake_session.commits == 0


def test_persistence_failure_preserves_the_stored_objects(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    _fail_commit(fake_session)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500
    # The key is content-addressed and may already belong to another analysis, so a
    # failed request never deletes it.
    assert fake_minio.removed == []
    assert len(fake_minio.stored_keys) == 1


def test_persistence_failure_still_cleans_up_local_temp_files(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    _fail_commit(fake_session)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500
    assert new_temp_uploads() == []


def test_failed_rollback_does_not_mask_the_persistence_failure(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    _fail_commit(fake_session)
    fake_session.rollback_error = OperationalError("ROLLBACK", None, Exception("gone"))

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500
    assert response.json() == {"detail": "analysis could not be persisted"}


def _all_added(session, model):
    return [row for row in session.added if isinstance(row, model)]


def test_the_upload_does_not_call_the_detector(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # The reason the queue exists: NVIDIA can take minutes, and no client is made to hold
    # a connection open through them. The request ends once the work has been recorded.
    assert response.status_code == 202
    assert fake_nvidia.paths == []


def test_the_upload_queues_a_job_for_the_analysis(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    analysis = _added(fake_session, Analysis)
    job = _added(fake_session, AnalysisJob)
    assert job.analysis_id == analysis.id
    assert job.status == "queued"


def test_the_queued_job_carries_the_request_that_asked_for_it(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    """The correlation id crosses from the request into the queue (R1-T4).

    This row is the only thing the API and the worker share. The analysis runs minutes later
    in another process, so an id kept in memory would end with this response and the work it
    queued would be uncorrelated — which is the state this task exists to fix.
    """
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("clip.mp4", b"payload", "video/mp4")},
        headers={"Origin": DASHBOARD_ORIGIN, REQUEST_ID_HEADER: "web-4f21ab"},
    )

    assert response.status_code == 202
    # The web application's id, not one of the API's own: the trace starts at the browser
    # boundary, and a second id minted here would split one request in two.
    assert _added(fake_session, AnalysisJob).request_id == "web-4f21ab"


def test_a_job_queued_without_a_caller_id_carries_the_one_the_api_minted(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    """A caller that sends no id still gets a job that is correlated to something.

    The id the API answers with and the id it wrote down are the same value, which is what
    lets a caller that never sent one still find its own analysis in the worker's log.
    """
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 202
    assert _added(fake_session, AnalysisJob).request_id == response.headers[
        REQUEST_ID_HEADER
    ]


def test_an_unusable_caller_id_never_reaches_the_job_row(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    """What is stored is what the middleware accepted, never what the caller sent.

    The column is bounded and so is the check in front of it, but the reason to prove this
    here is not the width: a value from outside travels from this row into the worker's log
    lines, and the refusal has to happen once, at the boundary, rather than at each of the
    places that later writes it out.
    """
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("clip.mp4", b"payload", "video/mp4")},
        headers={"Origin": DASHBOARD_ORIGIN, REQUEST_ID_HEADER: "A" * 65},
    )

    assert response.status_code == 202
    stored = _added(fake_session, AnalysisJob).request_id
    assert stored != "A" * 65
    assert stored == response.headers[REQUEST_ID_HEADER]


def test_a_queued_job_carries_no_error_message(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # Nothing has failed yet, and an empty string would read as a failure with nothing
    # to say about itself.
    assert _added(fake_session, AnalysisJob).error_message is None


def test_the_response_reports_the_analysis_as_queued(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # No detector has looked at this yet, so "completed" would be a claim about work
    # that has not happened.
    assert response.json()["status"] == "queued"
    assert _added(fake_session, Analysis).status == "queued"


def test_the_job_is_written_in_the_same_transaction_as_the_analysis(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # One commit covers the analysis, its media and its job. An analysis committed
    # without one would be an upload accepted and then silently forgotten: no queue row
    # for a runner to find, and a client holding a 202 for work nobody will ever do.
    assert fake_session.commits == 1
    assert len(_all_added(fake_session, AnalysisJob)) == 1


def test_exactly_one_job_is_queued_per_upload(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    payload = b"same-bytes"

    post_upload(client, "a.mp4", payload, "video/mp4")
    post_upload(client, "b.mp4", payload, "video/mp4")

    # Two analyses of identical media are still two analyses, and each is owed its own
    # detection — but neither is enqueued twice.
    jobs = _all_added(fake_session, AnalysisJob)
    analyses_added = _all_added(fake_session, Analysis)
    assert len(jobs) == 2
    assert {job.analysis_id for job in jobs} == {row.id for row in analyses_added}


def test_no_job_is_queued_when_the_media_pipeline_failed(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.error = media.MediaProbeError("no video stream")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # The media never became an analysis, so there is nothing to queue work against.
    assert response.status_code == 422
    assert _all_added(fake_session, AnalysisJob) == []


def test_a_failed_commit_queues_no_job(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    _fail_commit(fake_session)

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500
    # The job was built and rolled back with everything else: a queued job whose analysis
    # does not exist would be work pointing at nothing.
    assert fake_session.commits == 0
    assert fake_session.rollbacks == 1


def test_the_upload_leaves_nothing_local_for_the_worker_to_read(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    body = response.json()
    # The original is in MinIO and is the only thing the worker needs: it transcodes from
    # those bytes rather than depending on a temp file this request left behind.
    assert response.status_code == 202
    assert new_temp_uploads() == []
    assert fake_minio.stored_keys == [body["storage_key"]]
