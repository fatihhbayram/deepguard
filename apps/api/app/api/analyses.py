import hashlib
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import ANALYSIS_STATUS_COMPLETED, Analysis, MediaFile
from app.db.session import get_session
from app.media import MediaMetadata, MediaProbeError, MediaProbeUnavailable, probe_media
from app.normalization import (
    NormalizationError,
    NormalizationUnavailable,
    needs_normalization,
    normalize_to_mp4,
)
from app.storage import derivative_key, store_derivative, store_original

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


def report_possible_orphan(*storage_keys: str) -> None:
    """Record objects a failed request may have left behind, without deleting them.

    Storage keys are content-addressed, so an object this request uploaded can be the
    very same object an earlier, already persisted analysis of identical bytes refers
    to. Nothing available here proves the current request created an object exclusively,
    and proving it would take the reference counting or ownership tracking that P1
    deliberately does not have. Destroying a forensic original another analysis still
    references is far worse than leaving an object behind, so the object is preserved
    and the condition is logged for operational follow-up.
    """
    # An original that needed no derivative is reported under one key, not twice.
    logger.warning(
        "Failed request may have left unreferenced MinIO objects behind: %s.",
        ", ".join(dict.fromkeys(storage_keys)),
    )


class CreatedAnalysis(BaseModel):
    """The persisted analysis, as the pipeline established it."""

    id: uuid.UUID
    status: str
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
    local temp files and reports the request's stored objects as possible orphans; they
    are content-addressed and therefore never deleted here. The original file on disk is
    only read; the original MinIO object is preserved either way (D013).
    """
    try:
        derivative = await normalize_to_mp4(original_path, metadata)
    except NormalizationUnavailable:
        discard_temp_file(original_path)
        report_possible_orphan(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffmpeg is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except NormalizationError:
        discard_temp_file(original_path)
        report_possible_orphan(storage_key)
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
        # The upload may have created the derivative object before failing.
        report_possible_orphan(storage_key, derivative_key(derivative.sha256))
        # Endpoints, credentials and SDK errors stay in the server log, not the response.
        logger.exception("Storing the normalized derivative in MinIO failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media storage unavailable",
        ) from None

    discard_temp_file(derivative.path)

    return key, derivative.sha256


def persist_analysis(
    session: Session,
    *,
    filename: str | None,
    content_type: str,
    stored: StoredUpload,
    storage_key: str,
    metadata: MediaMetadata,
    was_normalized: bool,
    derivative_storage_key: str,
    derivative_sha256: str | None,
) -> Analysis:
    """Write the completed analysis and its media in one transaction.

    Called only once the whole pipeline has succeeded, so the row is complete the moment
    it exists. On failure the session is rolled back and the stored objects are reported
    rather than deleted, since they are content-addressed and may be shared.
    """
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    session.add(analysis)
    # Assigns the analysis id the media row needs, still inside the same transaction.
    session.flush()

    session.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename=filename,
            content_type=content_type,
            size_bytes=stored.size_bytes,
            original_sha256=stored.sha256,
            original_storage_key=storage_key,
            format_name=metadata.format_name,
            codec_name=metadata.codec_name,
            width=metadata.width,
            height=metadata.height,
            duration=metadata.duration,
            frame_rate=metadata.frame_rate,
            pix_fmt=metadata.pix_fmt,
            constant_frame_rate=metadata.constant_frame_rate,
            was_normalized=was_normalized,
            derivative_storage_key=derivative_storage_key,
            derivative_sha256=derivative_sha256,
        )
    )
    session.commit()

    return analysis


@router.post("/analyses", response_model=CreatedAnalysis)
async def create_analysis(
    file: UploadFile, session: Session = Depends(get_session)
) -> CreatedAnalysis:
    """Admit an upload by declared MIME type and size, then prove it is real media.

    The declared content type is not proof that the bytes are a real MP4/MOV container,
    so admission alone never produces a 200.

    The admitted upload is stored in MinIO as the forensic original, then probed with
    ffprobe to confirm the bytes really are video and to extract the metadata later
    tasks need. Media that is not already in the canonical provider shape gets a
    separate normalized derivative (D013) — the original is never rewritten. The
    analysis is persisted last, once every step has succeeded and nothing is left on
    local disk. On failure the temp files are dropped; the stored objects are kept,
    because their content-addressed keys may be shared with an earlier analysis.
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
        report_possible_orphan(storage_key)
        # The media may well be fine — this is the server missing its media processor.
        logger.exception("ffprobe is unavailable in this environment.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media processor unavailable",
        ) from None
    except MediaProbeError:
        discard_temp_file(stored.path)
        report_possible_orphan(storage_key)
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

    try:
        analysis = persist_analysis(
            session,
            filename=file.filename,
            content_type=content_type,
            stored=stored,
            storage_key=storage_key,
            metadata=metadata,
            was_normalized=was_normalized,
            derivative_storage_key=canonical_key,
            derivative_sha256=derivative_sha256,
        )
    except SQLAlchemyError:
        try:
            session.rollback()
        except SQLAlchemyError:
            # A rollback that fails is worth knowing about, but the persistence failure
            # is the error being handled and must not be replaced by it.
            logger.exception("Rolling the analysis transaction back failed.")
        # Statements, connection strings and driver errors stay in the server log.
        logger.exception("Persisting the analysis failed.")
        report_possible_orphan(storage_key, canonical_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="analysis could not be persisted",
        ) from None

    return CreatedAnalysis(
        id=analysis.id,
        status=analysis.status,
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
