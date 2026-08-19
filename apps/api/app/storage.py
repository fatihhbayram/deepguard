import os
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

load_dotenv()

ORIGINALS_BUCKET = "deepguard-originals"
ORIGINALS_PREFIX = "originals/"

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


def _ensure_originals_bucket() -> None:
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
    _ensure_originals_bucket()

    key = f"{ORIGINALS_PREFIX}{sha256}"
    client.fput_object(ORIGINALS_BUCKET, key, str(path), content_type=content_type)

    return key


def remove_original(storage_key: str) -> None:
    """Delete a stored original again.

    This exists for exactly one case: an upload that was stored before it turned out not
    to be usable media, which must not linger as a successful forensic artifact.
    """
    client.remove_object(ORIGINALS_BUCKET, storage_key)
