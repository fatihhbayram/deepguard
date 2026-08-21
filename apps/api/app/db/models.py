"""Persistent shape of an upload that has been accepted for analysis.

Stored are the analysis record itself, the media facts established by hashing, object
storage, ffprobe and normalization, the queued work that detection still owes it, and one
row per detector signal. Risk levels belong to a later phase and have no columns here.

Media identity is not analysis identity. Storage keys and hashes are content-addressed,
so the same bytes can legitimately be uploaded and analysed more than once; none of
those columns is unique.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Where an analysis is in its life. The upload commits it `queued` — the media is stored
# and probed, but no detector has looked at it yet — and the phase that runs detection is
# what moves it on. `completed` is not written anywhere until then.
ANALYSIS_STATUS_QUEUED = "queued"
ANALYSIS_STATUS_COMPLETED = "completed"
# Not "the media is fake" — "we could not find out". The detector never answered because
# something on this side broke, and the analysis says so rather than staying `queued`
# behind work that already gave up.
ANALYSIS_STATUS_FAILED = "failed"

# The state of the detection work an analysis is owed. A job is `queued` when the upload
# commits it, `processing` while a runner holds it, and ends `completed` or `failed`.
# Nothing yet moves a job out of `queued`: claiming and running it is P3-T2's job, and
# these four names exist so that task inherits the vocabulary rather than inventing it.
JOB_STATUS_QUEUED = "queued"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

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
    """One upload that was accepted, validated and stored.

    It is committed `queued`: the media is real and safely stored, and detection is still
    outstanding. The row exists long before there is anything to conclude about it.
    """

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

    # Whether this media has to be transcoded before a detector can read it. Decided at
    # upload from the probe, because that decision needs `major_brand` and no column
    # holds it — the worker could not re-derive it from this table. Past tense is
    # deliberate even though the transcode happens later: it describes the artifact the
    # analysis ends up being detected against, and an analysis whose transcode never
    # succeeded never completes.
    was_normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # The object downstream inference should read.
    #
    # Null while a derivative is still owed. Since P4-F2 the transcode runs in the worker,
    # so an upload that needs one commits without it and the worker fills both columns in
    # the transaction that finishes the job — writing a key here beforehand would name an
    # object that does not exist and might never (D020).
    #
    # When the original was already canonical no second artifact is ever produced, so this
    # is the original's own key from the moment of upload and the derivative hash stays
    # empty rather than repeating the original's identity.
    derivative_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    derivative_sha256: Mapped[str | None] = mapped_column(
        String(SHA256_HEX_LENGTH), nullable=True
    )


class AnalysisJob(Base):
    """The detection work an analysis is owed, and how far that work has got.

    The upload no longer waits for a detector. It stages the media, commits the analysis
    and commits this row alongside it in the same transaction, so a committed analysis
    always has a job: an analysis nobody was ever going to look at would be an upload
    silently dropped, and the atomicity is what rules that out.

    One analysis has exactly one job, which the unique constraint enforces rather than
    merely assumes. A retry re-runs this row instead of adding a second one, so the
    question "has this analysis been detected?" always has a single answer.

    Nothing here claims work. Whichever runner eventually moves a job out of `queued`,
    and how it does so safely, is P3-T2 — this table only records the state it moves
    between.
    """

    __tablename__ = "analysis_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    status: Mapped[str] = mapped_column(String(16), nullable=False)

    # Why the job failed, for an operator reading the table. Null for every job that has
    # not failed — an empty string would read as a failure with nothing to say. This is
    # diagnostic text of unbounded length, not a code to branch on.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Touched on every write, so the age of a job's current state is readable — the figure
    # a stuck `processing` job is spotted by.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
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
    # belongs in `analysis_segments`.
    signal_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AnalysisSegment(Base):
    """One piece of within-video evidence behind a signal, as the provider reported it.

    NVIDIA's synthetic-video detector scores the video in clips and reports, per clip, a
    frame index and a raw model logit — and nothing else. Those two facts are the columns
    here, because they are the two facts that exist.

    There is deliberately no `start_time`/`end_time` pair. `docs/planning/DATA_MODEL.md`
    describes a segment in seconds, but this provider reports no times, and converting a
    frame index into one would mean inventing a figure NVIDIA never gave. The frame index
    is stored as the frame index it is; a provider that really does report time ranges can
    add those columns when one exists (D019).

    There is likewise no `score` column. The clip figure is a raw logit on the model's own
    scale, not a probability like `AnalysisSignal.score`, and putting the two in
    identically named columns would invite exactly the comparison that rule 11 forbids.

    Rows hang off the signal rather than the analysis, because a clip logit is only
    meaningful as evidence for the detector run that produced it.
    """

    __tablename__ = "analysis_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    signal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_signals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The frame index of the clip's middle frame, exactly as NVIDIA reported it. The
    # provider's field is a uint32, which outgrows a 32-bit column, so it is stored wide.
    clip_index: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # The provider's raw model output for this clip. Untransformed: not a probability,
    # not rescaled, not rounded.
    logit: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
