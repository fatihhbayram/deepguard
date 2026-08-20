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

from app import media, normalization, nvidia_video, request_limits, storage
from app.api import analyses
from app.api.analyses import CHUNK_SIZE, MAX_UPLOAD_BYTES, TEMP_FILE_PREFIX
from app.db.models import Analysis, AnalysisSignal, MediaFile
from app.db.session import get_session
from app.main import app
from app.normalization import DERIVATIVE_TEMP_PREFIX


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


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Replace only the transcode call, writing plausible derivative bytes.

    Everything around it — the compatibility decision, hashing, key derivation, storage
    and cleanup — is the real code under test.
    """

    class Recorder:
        def __init__(self):
            self.output = DERIVATIVE_BYTES
            self.error = None
            self.calls = []

        async def run(self, source, destination, frame_rate):
            self.calls.append((Path(source), Path(destination), frame_rate))
            if self.error:
                raise self.error
            Path(destination).write_bytes(self.output)

    recorder = Recorder()
    monkeypatch.setattr(normalization, "_run_ffmpeg", recorder.run)
    return recorder


NVIDIA_PROBABILITY = 0.8734567165374756
NVIDIA_LOGIT = 1.9142135381698608
NVIDIA_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """Stand in for the detector, so no test in this module can reach NVIDIA.

    Autouse deliberately: every upload that gets past normalization now calls the
    detector, so a fixture a test forgot to request would mean real credentials, real
    network traffic and a real bill from the suite.

    The recorder captures the file's bytes during the call, because the request deletes
    the artifact before it responds — reading the path afterwards would prove nothing.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.paths = []
            self.analysed_bytes = []
            # What the session had done by the time the detector was called. Detection
            # must happen with no database work outstanding.
            self.session_state = []

        async def analyze(self, file_path, **kwargs):
            self.paths.append(Path(file_path))
            self.analysed_bytes.append(Path(file_path).read_bytes())
            session = app.dependency_overrides.get(get_session)
            if session is not None:
                session = session()
                self.session_state.append((list(session.added), session.commits))
            if self.error:
                raise self.error
            return nvidia_video.NvidiaVideoResult(
                logit=NVIDIA_LOGIT,
                probability=NVIDIA_PROBABILITY,
                total_clips=7,
                csv_data="0,-2.25\n8,3.5\n",
                function_id=NVIDIA_FUNCTION_ID,
                clips=(nvidia_video.NvidiaClipResult(index=0, logit=-2.25),),
            )

    recorder = Recorder()
    monkeypatch.setattr(analyses, "analyze_video", recorder.analyze)
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
def client(fake_session):
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
    return client.post(
        "/api/v1/analyses",
        files={"file": (filename, payload, content_type)},
    )


def test_declared_mp4_is_accepted(client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe):
    payload = b"probe-says-this-is-video"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    sha256 = hashlib.sha256(payload).hexdigest()
    persisted = next(row for row in fake_session.added if isinstance(row, Analysis))
    assert response.json() == {
        "id": str(persisted.id),
        "status": "completed",
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
        "derivative_storage_key": f"originals/{sha256}",
        "derivative_sha256": None,
    }


def test_declared_quicktime_is_accepted(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    response = post_upload(client, "clip.mov", b"0" * 4096, "video/quicktime")

    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "video/quicktime"
    assert body["size_bytes"] == 4096


def test_response_does_not_leak_the_temp_path(client, new_temp_uploads, fake_minio, fake_ffprobe):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
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

    assert response.status_code == 200
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

    assert response.status_code == 200
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

    assert response.status_code == 200
    assert fake_minio.created_buckets == [storage.ORIGINALS_BUCKET]


def test_concurrently_created_bucket_is_tolerated(client, new_temp_uploads, fake_minio, fake_ffprobe, monkeypatch):
    fake_minio._bucket_exists = False
    monkeypatch.setattr(
        fake_minio,
        "make_bucket",
        lambda bucket: (_ for _ in ()).throw(_s3_error("BucketAlreadyOwnedByYou")),
    )

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200


def test_successful_storage_uploads_the_original_before_probing(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    payload = b"forensic-original"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
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

    assert response.status_code == 200
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
    fake_ffprobe.output = ffprobe_output(avg_frame_rate="0/0")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422


def test_missing_stream_duration_falls_back_to_the_format_duration(
    client, new_temp_uploads, fake_minio, fake_ffprobe
):
    fake_ffprobe.output = ffprobe_output(stream_duration="N/A", format_duration="7.5")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
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

    assert response.status_code == 200
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

    assert response.status_code == 200
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

    assert response.status_code == 200
    assert response.json()["was_normalized"] is True
    assert len(fake_ffmpeg.calls) == 1


def test_quicktime_declared_as_mp4_is_still_normalized(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # A real MOV uploaded as `video/mp4`: the declaration is the client's claim, the
    # `qt  ` brand is the file's own evidence, and the evidence decides.
    fake_ffprobe.output = ffprobe_output(major_brand="qt  ")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    body = response.json()
    assert body["was_normalized"] is True
    assert body["metadata"]["major_brand"] == "qt"
    assert len(fake_ffmpeg.calls) == 1


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


def test_normalization_transcodes_the_staged_original_at_its_source_rate(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    payload = b"source-media"
    fake_ffprobe.output = ffprobe_output(codec_name="vp9", avg_frame_rate="24/1", r_frame_rate="24/1")

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    source, destination, frame_rate = fake_ffmpeg.calls[0]
    assert source.name.startswith(TEMP_FILE_PREFIX)
    assert source == fake_ffprobe.paths[0]
    assert destination.name.startswith(DERIVATIVE_TEMP_PREFIX)
    assert destination.suffix == ".mp4"
    # The validated source rate is preserved rather than resampled to a fixed default.
    assert frame_rate == 24.0


def test_odd_dimension_source_is_normalized_rather_than_rejected(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    # Odd dimensions are an encoder constraint, not bad media: the derivative is padded
    # to even sides and the upload succeeds.
    fake_ffprobe.output = ffprobe_output(codec_name="mjpeg", width=321, height=241)

    response = post_upload(client, "odd.mov", b"payload", "video/quicktime")

    assert response.status_code == 200
    body = response.json()
    assert body["was_normalized"] is True
    assert body["metadata"]["width"] == 321
    assert len(fake_ffmpeg.calls) == 1
    assert fake_minio.stored_keys == [body["storage_key"], body["derivative_storage_key"]]


def test_derivative_is_hashed_independently_of_the_original(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    body = response.json()
    derivative_sha256 = hashlib.sha256(DERIVATIVE_BYTES).hexdigest()
    assert body["derivative_sha256"] == derivative_sha256
    assert len(derivative_sha256) == 64
    assert derivative_sha256 == derivative_sha256.lower()
    assert body["derivative_sha256"] != body["sha256"]


def test_derivative_key_is_addressed_on_the_derivative_hash(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    body = response.json()
    assert body["derivative_storage_key"] == f"derivatives/{body['derivative_sha256']}.mp4"
    assert body["sha256"] not in body["derivative_storage_key"]


def test_derivative_is_stored_as_mp4_beside_the_original(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    body = response.json()
    bucket, key, _, content_type = fake_minio.uploads[1]
    assert bucket == storage.ORIGINALS_BUCKET
    assert key == body["derivative_storage_key"]
    assert content_type == "video/mp4"
    assert fake_minio.uploaded_bytes[key] == DERIVATIVE_BYTES


def test_successful_normalization_keeps_both_minio_objects(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    body = response.json()
    # D013: the original stays as the forensic artifact, the derivative as the media a
    # detector can actually consume.
    assert fake_minio.stored_keys == [body["storage_key"], body["derivative_storage_key"]]
    assert fake_minio.removed == []


def test_success_leaves_no_local_temp_files(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    compatible = post_upload(client, "clip.mp4", b"payload", "video/mp4")
    assert compatible.json()["was_normalized"] is False

    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    normalized = post_upload(client, "clip.mp4", b"other-payload", "video/mp4")
    assert normalized.json()["was_normalized"] is True

    # No P1 step after this reads local media, so neither the original nor the
    # derivative may survive the request.
    assert new_temp_uploads() == []


def test_failed_normalization_is_unprocessable(client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert response.json() == {"detail": "video could not be normalized for analysis"}


def test_normalization_timeout_is_a_controlled_rejection(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationTimeout("ffmpeg timed out after 60s")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert response.json() == {"detail": "video could not be normalized for analysis"}


def test_missing_ffmpeg_binary_is_reported_as_a_server_failure(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationUnavailable("ffmpeg could not be executed")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # The client's media may be perfectly fine; the server is missing its processor.
    assert response.status_code == 503
    assert response.json() == {"detail": "media processor unavailable"}


def test_failed_normalization_cleans_up_local_files(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert new_temp_uploads() == []


def test_failed_normalization_preserves_the_stored_original(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    # A transient transcode failure must not delete an object an earlier analysis of
    # the same bytes already refers to.
    assert fake_minio.removed == []


def test_derivative_upload_failure_returns_a_controlled_503(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_minio.upload_failure_after = 1

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 503
    assert response.json() == {"detail": "media storage unavailable"}


def test_derivative_upload_failure_cleans_up_local_files_only(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_minio.upload_failure_after = 1

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 503
    # Local temp files belong to this request alone; the stored objects do not.
    assert new_temp_uploads() == []
    assert fake_minio.removed == []


def _added(session, model):
    return next(row for row in session.added if isinstance(row, model))


def test_successful_upload_persists_the_analysis_and_its_media(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe
):
    payload = b"forensic-original"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    assert fake_session.commits == 1
    analysis = _added(fake_session, Analysis)
    media_file = _added(fake_session, MediaFile)
    assert analysis.status == "completed"
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


def test_normalized_upload_persists_the_derivative_identity(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"original-bytes", "video/mp4")

    body = response.json()
    media_file = _added(fake_session, MediaFile)
    assert media_file.was_normalized is True
    assert media_file.derivative_storage_key == body["derivative_storage_key"]
    assert media_file.derivative_sha256 == hashlib.sha256(DERIVATIVE_BYTES).hexdigest()
    assert media_file.derivative_sha256 != media_file.original_sha256


def test_analysis_is_persisted_only_after_the_media_pipeline_succeeded(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

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
    # Both keys are content-addressed and may already belong to another analysis.
    assert fake_minio.removed == []
    assert len(fake_minio.stored_keys) == 2


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


def test_the_detector_receives_the_original_when_it_was_already_canonical(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    payload = b"already-canonical-mp4"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    # No derivative exists, so the staged original is the provider-compatible artifact.
    assert fake_nvidia.analysed_bytes == [payload]
    assert fake_nvidia.paths[0].name.startswith(TEMP_FILE_PREFIX)


def test_the_detector_receives_the_derivative_when_the_media_was_normalized(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")

    response = post_upload(client, "clip.mp4", b"vp9-original-bytes", "video/mp4")

    assert response.status_code == 200
    # The original is not provider-compatible; sending it would be sending media the
    # detector cannot read.
    assert fake_nvidia.analysed_bytes == [DERIVATIVE_BYTES]
    assert fake_nvidia.paths[0].name.startswith(DERIVATIVE_TEMP_PREFIX)


def test_the_detector_is_not_called_when_the_pipeline_failed_earlier(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 422
    assert fake_nvidia.paths == []


def test_successful_detection_persists_the_nvidia_signal(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    analysis = _added(fake_session, Analysis)
    signal = _added(fake_session, AnalysisSignal)
    assert signal.analysis_id == analysis.id
    assert signal.provider == "nvidia"
    assert signal.signal_type == "synthetic_video"
    assert signal.status == "SUCCESS"
    assert signal.provider_version == NVIDIA_FUNCTION_ID


def test_the_persisted_score_is_nvidias_probability_untouched(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # Not rounded, not rescaled to a percentage, not turned into a DeepGuard number.
    assert _added(fake_session, AnalysisSignal).score == NVIDIA_PROBABILITY


def test_the_signal_carries_no_risk_level(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # A provider score is evidence, not a classification. Nothing here may invent one.
    assert _added(fake_session, AnalysisSignal).risk_level is None


def test_signal_metadata_keeps_the_aggregate_figures_only(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # The clip table and NVIDIA's CSV rendering of it are timeline evidence and grow
    # with the video's length; neither is dumped into this row.
    assert _added(fake_session, AnalysisSignal).signal_metadata == {
        "logit": NVIDIA_LOGIT,
        "total_clips": 7,
    }


def test_detector_timeout_still_returns_the_analysis(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    fake_nvidia.error = nvidia_video.NvidiaProviderTimeout("deadline exceeded")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    assert fake_session.commits == 1
    signal = _added(fake_session, AnalysisSignal)
    assert signal.status == "TIMEOUT"
    # NVIDIA may still have been working; recording 0.0 would be inventing an answer.
    assert signal.score is None
    assert signal.risk_level is None


def test_detector_failure_still_returns_the_analysis(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    fake_nvidia.error = nvidia_video.NvidiaAuthenticationError("NVIDIA rejected the request")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    signal = _added(fake_session, AnalysisSignal)
    assert signal.status == "FAILED"
    assert signal.score is None
    assert signal.provider_version is None


@pytest.mark.parametrize(
    "error",
    [
        nvidia_video.NvidiaProviderUnavailable("NVIDIA is unavailable"),
        nvidia_video.NvidiaProviderInvalidResponse("no final result"),
    ],
)
def test_every_known_provider_failure_is_recorded_rather_than_raised(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia, error
):
    fake_nvidia.error = error

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    assert _added(fake_session, AnalysisSignal).status == "FAILED"


def test_a_failed_signal_records_the_failure_kind_and_nothing_more(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    fake_nvidia.error = nvidia_video.NvidiaAuthenticationError(
        "NVIDIA rejected the request (UNAUTHENTICATED): bad key abc123"
    )

    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # The provider's own message can quote request detail, so only the class name of the
    # failure is persisted. The detail stays in the server log.
    assert _added(fake_session, AnalysisSignal).signal_metadata == {
        "error": "NvidiaAuthenticationError"
    }


def test_detector_failure_still_persists_the_media(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_nvidia.error = nvidia_video.NvidiaProviderUnavailable("NVIDIA is unavailable")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # One detector failing destroys no other evidence: the upload, its hashes and both
    # stored objects survive exactly as they would have.
    assert response.status_code == 200
    assert _added(fake_session, MediaFile).derivative_sha256 == hashlib.sha256(
        DERIVATIVE_BYTES
    ).hexdigest()
    assert len(fake_minio.stored_keys) == 2
    assert fake_minio.removed == []


def test_detection_finishes_before_any_database_work_starts(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # NVIDIA can take minutes. Nothing may be written or held open across that wait, so
    # by the time the detector is called the session must still be untouched.
    assert fake_nvidia.session_state == [([], 0)]
    assert fake_session.commits == 1


def test_a_failed_detection_leaves_no_local_temp_files(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_nvidia.error = nvidia_video.NvidiaProviderTimeout("deadline exceeded")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    assert new_temp_uploads() == []


def test_an_unforeseen_detector_error_still_cleans_up_local_files(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    # Not a provider failure at all — a bug. It is allowed to surface, but it must not
    # leave the staged original or the derivative behind on the way out.
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_nvidia.error = RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert new_temp_uploads() == []


def test_an_unreadable_staged_artifact_is_not_recorded_as_a_provider_failure(
    client, fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    fake_nvidia.error = nvidia_video.NvidiaLocalFileError("cannot read the staged file")

    with pytest.raises(nvidia_video.NvidiaLocalFileError):
        post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # This server prepared that file seconds earlier. Losing it is a fault here, not at
    # NVIDIA, and writing a FAILED signal would blame the provider for our defect and
    # bury a broken machine among routine evidence.
    assert fake_session.added == []
    assert fake_session.commits == 0


def test_an_unreadable_staged_artifact_fails_the_request(
    fake_session, new_temp_uploads, fake_minio, fake_ffprobe, fake_nvidia
):
    fake_nvidia.error = nvidia_video.NvidiaLocalFileError("cannot read the staged file")

    # A client that lets the server's own errors through, so the status a real caller
    # would see is asserted rather than the exception the test client re-raises.
    with TestClient(app, raise_server_exceptions=False) as unguarded:
        response = post_upload(unguarded, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 500


def test_an_unreadable_staged_artifact_still_cleans_up_local_files(
    client, new_temp_uploads, fake_minio, fake_ffprobe, fake_ffmpeg, fake_nvidia
):
    fake_ffprobe.output = ffprobe_output(codec_name="vp9")
    fake_nvidia.error = nvidia_video.NvidiaLocalFileError("cannot read the staged file")

    with pytest.raises(nvidia_video.NvidiaLocalFileError):
        post_upload(client, "clip.mp4", b"payload", "video/mp4")

    # Both the staged original and the derivative go, exactly as on every other path.
    assert new_temp_uploads() == []
