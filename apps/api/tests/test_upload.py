import hashlib
import tempfile
from pathlib import Path
from tempfile import SpooledTemporaryFile

import pytest
from fastapi.testclient import TestClient

from app.api.analyses import CHUNK_SIZE, MAX_UPLOAD_BYTES, TEMP_FILE_PREFIX
from app.main import app


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


def post_upload(client: TestClient, filename: str, payload, content_type: str):
    return client.post(
        "/api/v1/analyses",
        files={"file": (filename, payload, content_type)},
    )


def test_declared_mp4_is_accepted(client, new_temp_uploads):
    payload = b"not-a-real-container"

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    assert response.json() == {
        "filename": "clip.mp4",
        "content_type": "video/mp4",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_declared_quicktime_is_accepted(client, new_temp_uploads):
    response = post_upload(client, "clip.mov", b"0" * 4096, "video/quicktime")

    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "video/quicktime"
    assert body["size_bytes"] == 4096


def test_response_does_not_leak_the_temp_path(client, new_temp_uploads):
    response = post_upload(client, "clip.mp4", b"payload", "video/mp4")

    assert response.status_code == 200
    assert set(response.json()) == {"filename", "content_type", "size_bytes", "sha256"}


def test_sha256_is_lowercase_hex_of_the_uploaded_bytes(client, new_temp_uploads):
    # Multi-chunk payload, so an incremental hash is genuinely exercised.
    payload = bytes(range(256)) * (CHUNK_SIZE // 128)

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    sha256 = response.json()["sha256"]
    assert len(sha256) == 64
    assert sha256 == sha256.lower()
    assert sha256 == hashlib.sha256(payload).hexdigest()


def test_accepted_upload_is_written_to_a_temp_file(client, new_temp_uploads):
    payload = bytes(range(256)) * (CHUNK_SIZE // 128)

    response = post_upload(client, "clip.mp4", payload, "video/mp4")

    assert response.status_code == 200
    created = new_temp_uploads()
    assert len(created) == 1
    assert created[0].read_bytes() == payload


def test_unsupported_declared_mime_is_rejected(client, new_temp_uploads):
    response = post_upload(client, "notes.txt", b"plain text", "text/plain")

    assert response.status_code == 415
    assert new_temp_uploads() == []


def test_upload_above_the_size_limit_is_rejected(client, new_temp_uploads):
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
