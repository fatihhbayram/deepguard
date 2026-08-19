import hashlib
import tempfile
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pytest
from fastapi.testclient import TestClient

from minio.error import S3Error

from app import storage
from app.api.analyses import CHUNK_SIZE, MAX_UPLOAD_BYTES, TEMP_FILE_PREFIX
from app.main import app


class FakeMinio:
    """Stand-in for the MinIO client, so the suite never needs a live object store."""

    def __init__(self, *, bucket_exists=True, failure=None):
        self._bucket_exists = bucket_exists
        self._failure = failure
        self.created_buckets = []
        self.uploads = []

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
        self.uploads.append((bucket, key, file_path, content_type))


@pytest.fixture
def fake_minio(monkeypatch):
    fake = FakeMinio()
    monkeypatch.setattr(storage, "client", fake)
    return fake


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _temp_uploads() -> set[Path]:
    return set(Path(tempfile.gettempdir()).glob(f"{TEMP_FILE_PREFIX}*"))


@pytest.fixture
def new_temp_uploads():
    """Expose the temp files a request left behind, and clean them up afterwards.

    The endpoint intentionally retains successful uploads for later P1 tasks, so the
    test suite has to remove them itself.
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


def test_declared_mp4_is_accepted(client, new_temp_uploads, fake_minio):
    payload = b"not-a-real-container"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    sha256 = hashlib.sha256(payload).hexdigest()
    assert response.json() == {
        "filename": "clip.mp4",
        "content_type": "video/mp4",
        "size_bytes": len(payload),
        "sha256": sha256,
        "storage_key": f"originals/{sha256}",
    }


def test_declared_quicktime_is_accepted(client, new_temp_uploads, fake_minio):
    response = post_upload(client, "clip.mov", b"0" * 4096, "video/quicktime")

    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "video/quicktime"
    assert body["size_bytes"] == 4096


def test_response_does_not_leak_the_temp_path(client, new_temp_uploads, fake_minio):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    assert set(response.json()) == {
        "filename",
        "content_type",
        "size_bytes",
        "sha256",
        "storage_key",
    }


def test_sha256_is_lowercase_hex_of_the_uploaded_bytes(client, new_temp_uploads, fake_minio):
    # Multi-chunk payload, so an incremental hash is genuinely exercised.
    payload = bytes(range(256)) * (CHUNK_SIZE // 128)

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    sha256 = response.json()["sha256"]
    assert len(sha256) == 64
    assert sha256 == sha256.lower()
    assert sha256 == hashlib.sha256(payload).hexdigest()


def test_accepted_upload_is_written_to_a_temp_file(client, new_temp_uploads, fake_minio):
    payload = bytes(range(256)) * (CHUNK_SIZE // 128)

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    created = new_temp_uploads()
    assert len(created) == 1
    assert created[0].read_bytes() == payload


def test_unsupported_declared_mime_is_rejected(client, new_temp_uploads, fake_minio):
    response = post_upload(client, "notes.txt", b"plain text", "text/plain")

    assert response.status_code == 415
    assert new_temp_uploads() == []


def test_upload_above_the_size_limit_is_rejected(client, new_temp_uploads, fake_minio):
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
    assert new_temp_uploads() == []


def test_admitted_upload_is_stored_in_minio(client, new_temp_uploads, fake_minio):
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
    assert Path(file_path).read_bytes() == payload


def test_identical_bytes_resolve_to_the_same_storage_key(client, new_temp_uploads, fake_minio):
    payload = b"same-bytes"

    first = post_upload(client, "a.mp4", payload, "video/mp4")
    second = post_upload(client, "b.mov", payload, "video/quicktime")

    assert first.json()["storage_key"] == second.json()["storage_key"]


def test_missing_bucket_is_created_before_upload(client, new_temp_uploads, fake_minio):
    fake_minio._bucket_exists = False

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    assert fake_minio.created_buckets == [storage.ORIGINALS_BUCKET]


def test_concurrently_created_bucket_is_tolerated(client, new_temp_uploads, fake_minio, monkeypatch):
    fake_minio._bucket_exists = False
    monkeypatch.setattr(
        fake_minio,
        "make_bucket",
        lambda bucket: (_ for _ in ()).throw(_s3_error("BucketAlreadyOwnedByYou")),
    )

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200


def test_successful_storage_keeps_the_local_temp_file(client, new_temp_uploads, fake_minio):
    payload = b"kept-for-ffprobe"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    retained = new_temp_uploads()
    assert len(retained) == 1
    assert retained[0].read_bytes() == payload


def test_storage_failure_returns_a_controlled_503(client, new_temp_uploads, fake_minio):
    fake_minio._failure = _s3_error("InternalError")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 503
    assert response.json() == {"detail": "media storage unavailable"}


def test_storage_failure_removes_the_temp_file(client, new_temp_uploads, fake_minio):
    fake_minio._failure = OSError("connection refused")

    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 503
    assert new_temp_uploads() == []
