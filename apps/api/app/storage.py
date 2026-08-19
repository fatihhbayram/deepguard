import os
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

load_dotenv()

ORIGINALS_BUCKET = "deepguard-originals"
ORIGINALS_PREFIX = "originals/"
# Provider-compatible derivatives live beside the originals under their own prefix; a
# second bucket would add operational surface without changing anything for the MVP.
DERIVATIVES_PREFIX = "derivatives/"
DERIVATIVE_CONTENT_TYPE = "video/mp4"
DERIVATIVE_EXTENSION = ".mp4"

# Already-created races are benign: the bucket we wanted exists either way.
_BUCKET_EXISTS_CODES = frozenset({"BucketAlreadyOwnedByYou", "BucketAlreadyExists"})


def _build_client() -> Minio:
    """Client for the MinIO service on the Docker network, not the host-published port."""
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "deepguard"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "deepguard123"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


client = _build_client()


def _ensure_bucket() -> None:
    if client.bucket_exists(ORIGINALS_BUCKET):
        return

    try:
        client.make_bucket(ORIGINALS_BUCKET)
    except S3Error as error:
        # Two concurrent uploads can both find the bucket missing and both create it.
        if error.code not in _BUCKET_EXISTS_CODES:
            raise


def store_original(path: Path, sha256: str, content_type: str) -> str:
    """Upload the staged original file as-is and return its storage key.

    The key is content-addressed on the SHA-256 of the original bytes, so re-uploading
    identical media resolves to the same object. The file is streamed from disk; the
    bytes are never re-read into memory or re-hashed (D013: the original is a forensic
    artifact and must stay byte-for-byte identical).
    """
    _ensure_bucket()

    key = f"{ORIGINALS_PREFIX}{sha256}"
    client.fput_object(ORIGINALS_BUCKET, key, str(path), content_type=content_type)

    return key


def derivative_key(sha256: str) -> str:
    """The content-addressed key of a derivative, from the derivative's own hash."""
    return f"{DERIVATIVES_PREFIX}{sha256}{DERIVATIVE_EXTENSION}"


def store_derivative(path: Path, sha256: str) -> str:
    """Upload a normalized derivative and return its storage key.

    The key is addressed on the derivative's own SHA-256, not the original's: the two
    are different artifacts and must not share an identity.
    """
    _ensure_bucket()

    key = derivative_key(sha256)
    client.fput_object(
        ORIGINALS_BUCKET, key, str(path), content_type=DERIVATIVE_CONTENT_TYPE
    )

    return key


def remove_stored_object(storage_key: str) -> None:
    """Delete a stored original or derivative again.

    This exists for one case: a pipeline object stored before the request turned out to
    fail, which must not linger as if the upload had been accepted.
    """
    client.remove_object(ORIGINALS_BUCKET, storage_key)
