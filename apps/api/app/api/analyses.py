import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["analyses"])

ALLOWED_CONTENT_TYPES = frozenset({"video/mp4", "video/quicktime"})
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
TEMP_FILE_PREFIX = "deepguard-upload-"


class UploadAdmission(BaseModel):
    """What is genuinely known after admission validation — nothing is persisted yet."""

    filename: str | None
    content_type: str
    size_bytes: int
    sha256: str


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
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
        raise

    return StoredUpload(
        path=Path(temp_file.name),
        size_bytes=size,
        sha256=hasher.hexdigest(),
    )


@router.post("/analyses", response_model=UploadAdmission)
async def create_analysis(file: UploadFile) -> UploadAdmission:
    """Admission validation only: declared MIME type and upload size.

    The declared content type is not proof that the bytes are a real MP4/MOV container;
    that check belongs to the later ffprobe task.

    The accepted upload is retained as a temp file for the following P1 tasks (MinIO
    storage, ffprobe); its lifecycle beyond this request is defined there.
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

    return UploadAdmission(
        filename=file.filename,
        content_type=content_type,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
    )
