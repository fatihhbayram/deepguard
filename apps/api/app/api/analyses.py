from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["analyses"])

ALLOWED_CONTENT_TYPES = frozenset({"video/mp4", "video/quicktime"})
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class UploadAdmission(BaseModel):
    """What is genuinely known after admission validation — nothing is persisted yet."""

    filename: str | None
    content_type: str
    size_bytes: int


async def measure_upload(file: UploadFile) -> int:
    """Consume the upload in bounded chunks, failing as soon as the limit is exceeded."""
    size = 0
    while chunk := await file.read(CHUNK_SIZE):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit.",
            )

    return size


@router.post("/analyses", response_model=UploadAdmission)
async def create_analysis(file: UploadFile) -> UploadAdmission:
    """Admission validation only: declared MIME type and upload size.

    The declared content type is not proof that the bytes are a real MP4/MOV container;
    that check belongs to the later ffprobe task.
    """
    content_type = (file.content_type or "").strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type: {file.content_type or 'unknown'}.",
        )

    try:
        size_bytes = await measure_upload(file)
    finally:
        await file.close()

    return UploadAdmission(
        filename=file.filename,
        content_type=content_type,
        size_bytes=size_bytes,
    )
