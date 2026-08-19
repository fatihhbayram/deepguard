from tempfile import SpooledTemporaryFile

import pytest
from fastapi.testclient import TestClient

from app.api.analyses import CHUNK_SIZE, MAX_UPLOAD_BYTES
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def post_upload(client: TestClient, filename: str, payload, content_type: str):
    return client.post(
        "/api/v1/analyses",
        files={"file": (filename, payload, content_type)},
    )


def test_declared_mp4_is_accepted(client):
    response = post_upload(client, "clip.mp4", b"not-a-real-container", "video/mp4")

    assert response.status_code == 200
    assert response.json() == {
        "filename": "clip.mp4",
        "content_type": "video/mp4",
        "size_bytes": len(b"not-a-real-container"),
    }


def test_declared_quicktime_is_accepted(client):
    response = post_upload(client, "clip.mov", b"0" * 4096, "video/quicktime")

    assert response.status_code == 200
    body = response.json()
    assert body["content_type"] == "video/quicktime"
    assert body["size_bytes"] == 4096


def test_unsupported_declared_mime_is_rejected(client):
    response = post_upload(client, "notes.txt", b"plain text", "text/plain")

    assert response.status_code == 415


def test_upload_above_the_size_limit_is_rejected(client):
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
