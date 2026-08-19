import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.media import MediaMetadata, MediaProbeError, MediaProbeUnavailable, probe_media
from app.storage import remove_original, store_original

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyses"])

ALLOWED_CONTENT_TYPES = frozenset({"video/mp4", "video/quicktime"})
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
TEMP_FILE_PREFIX = "deepguard-upload-"


def discard_temp_file(path: str | Path) -> None:
    """Remove a staged temp file best-effort, never masking the error being handled."""
    try:
        os.unlink(path)
    except OSError:
        pass


def discard_stored_original(storage_key: str) -> None:
    """Roll back a stored original best-effort, never masking the error being handled."""
    try:
        remove_original(storage_key)
    except Exception:
        logger.exception("Removing the rejected original %s from MinIO failed.", storage_key)


class UploadAdmission(BaseModel):
    """What is genuinely known after admission validation — nothing is persisted yet."""

    filename: str | None
    content_type: str
    size_bytes: int
    sha256: str
    storage_key: str
    metadata: MediaMetadata


@dataclass(frozen=True)
class StoredUpload:
    """Internal result of the single read pass.

    The temp path is deliberately kept out of the API response: it is server-internal
    filesystem layout, not a client-facing contract.
    """

    path: Path
    size_bytes: int
    sha256: str


async def store_upload(file: UploadFile) -> StoredUpload:
    """Consume the upload once in bounded chunks, sizing, hashing and spilling to disk.

    The upload stream is read exactly once: every chunk is counted, checked against the
    limit, fed to SHA-256 and written to the temp file in the same pass. On any failure
    the partial temp file is removed without masking the original error.
    """
    hasher = hashlib.sha256()
    size = 0

    temp_file = tempfile.NamedTemporaryFile(prefix=TEMP_FILE_PREFIX, delete=False)
    try:
        with temp_file:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit.",
                    )

                hasher.update(chunk)
                temp_file.write(chunk)
    except BaseException:
        discard_temp_file(temp_file.name)
        raise

    return StoredUpload(
        path=Path(temp_file.name),
        size_bytes=size,
        sha256=hasher.hexdigest(),
    )


@router.post("/analyses", response_model=UploadAdmission)
async def create_analysis(file: UploadFile) -> UploadAdmission:
    """Admit an upload by declared MIME type and size, then prove it is real media.

    The declared content type is not proof that the bytes are a real MP4/MOV container,
    so admission alone never produces a 200.

    The admitted upload is stored in MinIO as the forensic original, then probed with
    ffprobe to confirm the bytes really are video and to extract the metadata later
    tasks need. On success the temp file is retained for normalization; on any failure
    both the temp file and the stored object are dropped best-effort, since nothing
    persists a reference that could reclaim them later.
    """
    content_type = (file.content_type or "").strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type or 'unknown'}.",
        )

    try:
        stored = await store_upload(file)
    finally:
        await file.close()

    try:
        storage_key = store_original(stored.path, stored.sha256, content_type)
    except Exception:
        discard_temp_file(stored.path)
        # Endpoints, credentials and SDK errors stay in the server log, not the response.
        logger.exception("Storing the original upload in MinIO failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media storage unavailable",
        ) from None

    try:
        metadata = await probe_media(stored.path)
    except MediaProbeUnavailable:
        discard_temp_file(stored.path)
        discard_stored_original(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffprobe is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except MediaProbeError:
        discard_temp_file(stored.path)
        discard_stored_original(storage_key)
        # Rejected content must not stay behind as if it had been accepted.
        logger.info("Rejected admitted upload %s as unusable media.", storage_key, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="invalid or unsupported video media",
        ) from None

    return UploadAdmission(
        filename=file.filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=storage_key,
        metadata=metadata,
    )
