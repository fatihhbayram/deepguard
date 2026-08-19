import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.media import MediaMetadata, MediaProbeError, MediaProbeUnavailable, probe_media
from app.normalization import (
    NormalizationError,
    NormalizationUnavailable,
    needs_normalization,
    normalize_to_mp4,
)
from app.storage import derivative_key, remove_stored_object, store_derivative, store_original

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


def discard_stored_object(storage_key: str) -> None:
    """Roll back a stored object best-effort, never masking the error being handled.

    Nothing persists a reference to these objects yet, so a failed request that left one
    behind would leave it orphaned with no way to reclaim it.
    """
    try:
        remove_stored_object(storage_key)
    except Exception:
        logger.exception("Removing %s from MinIO failed.", storage_key)


class UploadAdmission(BaseModel):
    """What is genuinely known after admission validation — nothing is persisted yet."""

    filename: str | None
    content_type: str
    size_bytes: int
    sha256: str
    storage_key: str
    metadata: MediaMetadata
    was_normalized: bool
    # The object downstream inference should read. When the original is already
    # canonical this is the original's key: no second artifact exists, and inventing a
    # copy of it would only duplicate storage.
    derivative_storage_key: str
    # Present only when a real derivative exists, since it is that artifact's identity.
    derivative_sha256: str | None = None


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


async def create_derivative(
    original_path: Path, storage_key: str, metadata: MediaMetadata
) -> tuple[str, str]:
    """Transcode the original into a stored canonical derivative and describe it.

    Returns the derivative's storage key and its own SHA-256. Any failure cleans up the
    local temp files and rolls the request's MinIO objects back before raising, because
    no persistence exists yet that could reclaim them later. The original file on disk
    is only read; the original MinIO object is preserved on success (D013).
    """
    try:
        derivative = await normalize_to_mp4(original_path, metadata)
    except NormalizationUnavailable:
        discard_temp_file(original_path)
        discard_stored_object(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffmpeg is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except NormalizationError:
        discard_temp_file(original_path)
        discard_stored_object(storage_key)
        logger.info("Could not normalize admitted upload %s.", storage_key, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="video could not be normalized for analysis",
        ) from None

    try:
        key = store_derivative(derivative.path, derivative.sha256)
    except Exception:
        discard_temp_file(derivative.path)
        discard_temp_file(original_path)
        # The upload may have created the object before failing.
        discard_stored_object(derivative_key(derivative.sha256))
        discard_stored_object(storage_key)
        # Endpoints, credentials and SDK errors stay in the server log, not the response.
        logger.exception("Storing the normalized derivative in MinIO failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media storage unavailable",
        ) from None

    discard_temp_file(derivative.path)

    return key, derivative.sha256


@router.post("/analyses", response_model=UploadAdmission)
async def create_analysis(file: UploadFile) -> UploadAdmission:
    """Admit an upload by declared MIME type and size, then prove it is real media.

    The declared content type is not proof that the bytes are a real MP4/MOV container,
    so admission alone never produces a 200.

    The admitted upload is stored in MinIO as the forensic original, then probed with
    ffprobe to confirm the bytes really are video and to extract the metadata later
    tasks need. Media that is not already in the canonical provider shape gets a
    separate normalized derivative (D013) — the original is never rewritten. On any
    failure the temp files and the request's stored objects are dropped best-effort,
    since nothing persists a reference that could reclaim them later.
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
        discard_stored_object(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffprobe is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except MediaProbeError:
        discard_temp_file(stored.path)
        discard_stored_object(storage_key)
        # Rejected content must not stay behind as if it had been accepted.
        logger.info("Rejected admitted upload %s as unusable media.", storage_key, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="invalid or unsupported video media",
        ) from None

    was_normalized = needs_normalization(content_type, metadata)
    if was_normalized:
        canonical_key, derivative_sha256 = await create_derivative(
            stored.path, storage_key, metadata
        )
    else:
        # Already canonical: no transcode, and no duplicate object of the original.
        canonical_key, derivative_sha256 = storage_key, None

    # No P1 step after this reads local media, so nothing may be left on disk.
    discard_temp_file(stored.path)

    return UploadAdmission(
        filename=file.filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        storage_key=storage_key,
        metadata=metadata,
        was_normalized=was_normalized,
        derivative_storage_key=canonical_key,
        derivative_sha256=derivative_sha256,
    )
