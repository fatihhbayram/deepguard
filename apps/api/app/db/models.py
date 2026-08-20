"""Persistent shape of an upload that completed the pipeline.

Stored are the analysis record itself, the media facts established by hashing, object
storage, ffprobe and normalization, and one row per detector signal. Risk levels and job
state belong to later phases and have no columns here.

Media identity is not analysis identity. Storage keys and hashes are content-addressed,
so the same bytes can legitimately be uploaded and analysed more than once; none of
those columns is unique.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# The single status P1 can produce: the row is written only after the whole pipeline has
# succeeded, so there is no pending/running lifecycle to model yet.
ANALYSIS_STATUS_COMPLETED = "completed"

# The outcome of asking one detector about one analysis. A provider that answered, a
# provider that refused or broke, and a provider that never answered in time are three
# different forensic facts and are never collapsed into one.
SIGNAL_STATUS_SUCCESS = "SUCCESS"
SIGNAL_STATUS_FAILED = "FAILED"
SIGNAL_STATUS_TIMEOUT = "TIMEOUT"

SHA256_HEX_LENGTH = 64


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    """One upload that was accepted, validated and stored."""

    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MediaFile(Base):
    """The media belonging to an analysis, as the P1 pipeline established it."""

    __tablename__ = "media_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # As uploaded. The filename is whatever the client sent, and may be absent.
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # The forensic original (D013): preserved byte-for-byte, identified by its own hash.
    original_sha256: Mapped[str] = mapped_column(
        String(SHA256_HEX_LENGTH), nullable=False
    )
    original_storage_key: Mapped[str] = mapped_column(String(512), nullable=False)

    # What ffprobe established about the original.
    format_name: Mapped[str] = mapped_column(String(255), nullable=False)
    codec_name: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    frame_rate: Mapped[float] = mapped_column(Float, nullable=False)
    pix_fmt: Mapped[str | None] = mapped_column(String(32), nullable=True)
    constant_frame_rate: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # The object downstream inference should read. When the original was already
    # canonical no second artifact exists, so this is the original's key and the
    # derivative hash stays empty rather than repeating the original's identity.
    was_normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    derivative_storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    derivative_sha256: Mapped[str | None] = mapped_column(
        String(SHA256_HEX_LENGTH), nullable=True
    )


class AnalysisSignal(Base):
    """One detector's answer about one analysis, recorded as that detector gave it.

    Signals are independent evidence: every provider gets its own row, its own status
    and its own untransformed score, and nothing here ever averages two providers into a
    combined number.

    An analysis can carry a signal per provider, so several rows share an `analysis_id`;
    the row is written in the same transaction as the analysis it belongs to.
    """

    __tablename__ = "analysis_signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Who was asked, and what was asked of them — for example "nvidia" and
    # "synthetic_video". One provider can answer more than one question about a video.
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(64), nullable=False)

    # The provider's own figure, on the provider's own scale, exactly as returned. Null
    # whenever the provider produced no number, which is every non-SUCCESS status: a
    # failed detector has no score, and 0.0 would be a fabricated answer.
    score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # DeepGuard's deterministic classification of this signal. Null until a phase exists
    # that owns risk logic — a provider score is not a risk level.
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Which deployment of the detector produced this, so an old signal stays
    # interpretable after the provider ships a new model.
    provider_version: Mapped[str | None] = mapped_column(String(128), nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False)

    # The provider's supporting figures, kept as the small JSON document they are rather
    # than as columns invented per provider. Timeline evidence is not stored here; it
    # belongs in the `analysis_segments` table of a later phase.
    signal_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
