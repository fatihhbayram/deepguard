"""Persistent shape of an upload that has been accepted for analysis.

Stored are the analysis record itself, the media facts established by hashing, object
storage, ffprobe and normalization, the queued work that detection still owes it, one
row per detector signal, and — since P7-T3 — the risk decision the analysis ended in,
recorded with the ruleset and calibration it was taken under.

Alongside them is `api_keys` — the credentials the public API authenticates B2B callers
with (P9-T1). Since P9-T2 an analysis may name the key that submitted it, which is what
keeps one customer's analyses out of another's reads.

Since R1-T1 there is a second kind of caller: `users`, the accounts a person signs into the
web application as, and `auth_sessions`, the opaque cookie sessions those sign-ins create.
The two credential families stay apart on purpose — an analysis may name a user or an API
key, never both — and the database enforces that rather than trusting the two code paths
that write the column.

Apart from all of them is `shadow_runs` (R6-T1): what an uncalibrated experimental workload
observed about an analysis, kept in a table no customer-facing reader and no risk rule names.
Its docstring says why that separation is structural rather than a filter.

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
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    true as sa_true,
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

# The state of one experimental workload's run against one analysis (R6-T1). The same four
# names the production job uses, because it is the same life: queued when the production
# analysis finished, processing while a worker holds it, and terminal either way.
#
# Deliberately a second set of constants rather than a reuse of `JOB_STATUS_*`. The two
# vocabularies happen to coincide today and are not the same vocabulary: nothing about a
# shadow run is entitled to follow a change made for production work, and a reader tracing
# where a status came from should land on the table it belongs to.
SHADOW_RUN_STATUS_QUEUED = "queued"
SHADOW_RUN_STATUS_PROCESSING = "processing"
SHADOW_RUN_STATUS_COMPLETED = "completed"
SHADOW_RUN_STATUS_FAILED = "failed"

SHA256_HEX_LENGTH = 64

# What a person may be when they sign in. Two roles and no permission table: the only
# distinction R1-T1 has a use for is "may reach an administrative action at all", and a
# grant model with nothing to grant would be a schema built for a requirement that does not
# exist yet.
USER_ROLE_USER = "USER"
USER_ROLE_ADMIN = "ADMIN"

# The one rule keeping the two credential families apart: an analysis is owned by a signed-in
# user, or by an API key, or by nobody. Never by both. Named here because the migration, the
# model and the test that proves the constraint bites all have to mean the same constraint.
SINGLE_OWNER_CONSTRAINT = "ck_analyses_single_owner"


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    """One upload that was accepted, validated and stored, and what DeepGuard concluded.

    It is committed `queued`: the media is real and safely stored, and detection is still
    outstanding. The row exists long before there is anything to conclude about it.

    The four risk columns are that conclusion, written together in the transaction that
    completes the job (P7-T3). They are the decision *trace*, not a second copy of the
    evidence: no provider score, threshold or clip count is duplicated here, because the
    signal rows remain the forensic record and a figure repeated into this table could
    drift from the one it was copied out of.
    """

    __tablename__ = "analyses"

    # Both ownership columns may be null, and exactly one of them may be set — never two.
    # Written as `NOT (both are present)` rather than as an exclusive-or so that the
    # unowned row stays legal: analyses submitted through the dashboard before there were
    # accounts belong to nobody, and always will.
    #
    # In the database rather than in the two functions that write these columns, because
    # this is the invariant the public API's isolation rests on. A row owned by a user *and*
    # an API key would be reachable through the public API by a customer who never submitted
    # it, and a check that lives only in application code is one code path away from not
    # running.
    __table_args__ = (
        CheckConstraint(
            "NOT (owner_id IS NOT NULL AND api_key_id IS NOT NULL)",
            name=SINGLE_OWNER_CONSTRAINT,
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # What the risk engine concluded: `HIGH`, `MEDIUM` or `UNKNOWN`. `LOW` is measured but
    # not activated in ruleset v1 and is never written here (see `app.risk_engine`).
    #
    # Null means no decision has been taken yet — an analysis still queued or being worked
    # on — and never "we looked and found nothing": that answer is `UNKNOWN`, which is a
    # real classification with a rule behind it.
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Which immutable ruleset produced the level above, which measurement its thresholds
    # came from, and which single rule fired. All three are what make an old decision
    # explainable after the rules move on: the level alone would be unreadable once
    # `p7-v1.0.0` is no longer what runs.
    risk_rules_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    risk_calibration_id: Mapped[str | None] = mapped_column(
        String(SHA256_HEX_LENGTH), nullable=True
    )
    risk_rule_id: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Which API key submitted this analysis, when one did (P9-T2). Null is not missing
    # data: it means the analysis came in through the internal dashboard, which
    # authenticates nobody and owns nothing. Every public read filters on this column, so a
    # null row is unreachable through the public API by construction — no key's id can
    # equal null — and the dashboard keeps seeing everything, as it did before.
    #
    # `RESTRICT` rather than `CASCADE`: deleting a key must never take the analyses it
    # authenticated with it, because those are forensic records that outlive the
    # credential. Retiring a key is `is_active = false`, which leaves this intact.
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    # Which signed-in user submitted this analysis, when one did (R1-T1). The web
    # counterpart of `api_key_id` above, and mutually exclusive with it — see the check
    # constraint at the top of this class.
    #
    # Nothing writes this column yet. R1-T1 establishes the identity foundation and
    # explicitly stops short of enforcing dashboard ownership, so every row is null today
    # and the dashboard keeps reading everything exactly as it did. The column is added now
    # because it is the schema half of the same one logical change, and adding it nullable
    # is a metadata-only alteration of a populated table.
    #
    # `RESTRICT` for the same reason the API key uses it: deleting an account must never
    # take the forensic records it created with it.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
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

    # The request that asked for this analysis, carried across the queue (R1-T4). The API
    # binds an id to every request it serves and writes it here in the same insert as the
    # job; the worker reads it back when it claims the job and binds it to its own logs, so
    # one grep covers the browser request, the API request and the analysis minutes later.
    #
    # This column exists because the queue is the only thing the two processes share. A
    # correlation id held in memory would end with the response, and the work this row
    # describes had not started yet.
    #
    # Nullable, and not backfilled: every job queued before this existed was queued by a
    # request nobody recorded, and inventing an id for one would be inventing the very fact
    # the column is for. Nullable also keeps the column honest about jobs written by
    # anything that runs outside a request.
    #
    # Bounded and character-restricted where it is accepted, not here — see
    # `app.observability.accepted_request_id`. The width matches the ceiling it enforces.
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # When the claim on this job stops being believed, and therefore the one thing that
    # tells a crashed worker apart from a slow one. The worker that claims a job sets this
    # ahead of the database clock and pushes it forward again while it works; a worker that
    # died stops pushing, the timestamp falls into the past, and the job becomes
    # recoverable. Null on every job nobody is running — `queued` never had a lease, and a
    # terminal job's is cleared when it ends.
    #
    # `updated_at` cannot do this job. It moves for any write and stands still through the
    # long middle of a real analysis, so age alone would fail a live worker on a four-minute
    # video and spare a dead one that crashed just after a write. This column is a promise
    # about the future, which is what makes its expiry mean something.
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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

    # Still null on every row, and P7-T3 is where that stopped being "until a later phase".
    # Risk is a decision about the *analysis*, taken from one calibrated signal under a
    # named ruleset, and it lives on `analyses` with the trace that explains it. Writing a
    # level here as well would put a classification beside every provider's number — the
    # provenance reading, the speaker timeline, the audio windows — and invite exactly the
    # per-signal verdicts the risk engine exists to refuse (rule 11).
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Which deployment of the detector produced this, so an old signal stays
    # interpretable after the provider ships a new model.
    #
    # 255 since R5-T2, where a detector arrived whose identity is genuinely two artifacts: the
    # commit its architecture is executed from and the digest of the weights loaded into it,
    # which come from different places and neither of which pins the model alone. The
    # migration that widened it says more.
    provider_version: Mapped[str | None] = mapped_column(String(255), nullable=True)

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


class ApiKey(Base):
    """A credential a B2B caller authenticates the public API with.

    The plaintext key exists once, at the moment it is generated, and is handed to whoever
    asked for it. What is stored here is its SHA-256 digest and nothing else: a database
    dump, a backup or a leaked replica therefore yields no usable credential, and DeepGuard
    itself cannot show a customer their key again — it can only replace it.

    SHA-256 rather than a password hash on purpose. The secret is 256 bits of `secrets`
    output, not a human-chosen password, so there is no dictionary to slow an attacker
    down; the cost a bcrypt-class hash buys would be paid on every single request instead.

    `key_hash` is unique because it identifies the row — authentication hashes the
    presented key and looks it up directly, which is what keeps verification a single
    indexed lookup rather than a scan over every key on file.

    Deactivation is `is_active`, not deletion: a key that authenticated real analyses stays
    on file so those requests remain attributable after it stops working.
    """

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Who the key is for, for an operator reading the table. Not a secret and not an
    # identifier — nothing authenticates by name.
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    key_hash: Mapped[str] = mapped_column(
        String(SHA256_HEX_LENGTH), nullable=False, unique=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_true()
    )

    # When the key last authenticated a request. Null until it ever has — which is every
    # row today, because nothing writes this column yet. Recording it means a write on the
    # hot path of every authenticated request, and that trade belongs to the task that
    # actually needs the figure rather than to the one that adds the column.
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class User(Base):
    """A person who can sign into the web application.

    The account, and nothing beyond it: no profile, no preferences, no organization. R1-T1
    needs to know who is signing in and whether they may reach an administrative action, and
    a wider table would be schema written for requirements that do not exist.

    `email` is stored already normalized — lowercased and stripped — and is unique on that
    normalized form. Uniqueness has to be over the same value the login lookup uses, or
    `Alice@example.com` and `alice@example.com` would be two accounts that both answer to
    one sign-in. `app.web_auth.normalize_email` is the single place that normalization
    happens, and creation and login both go through it.

    `password_hash` is an Argon2id hash, complete with its own parameters and salt in the
    standard encoded form — which is why the column is wide and why nothing else here
    records a salt. Unlike `ApiKey.key_hash` this is deliberately a slow hash: a password is
    human-chosen and therefore guessable, so the cost that would be waste on a 256-bit random
    key is exactly the point here.

    Deactivation is `is_active`, matching `ApiKey`: an account that submitted analyses stays
    on file so those records remain attributable after the person stops being able to sign in.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # 320 characters is the longest an address can be under RFC 3696's 64-character local
    # part and 255-character domain.
    email: Mapped[str] = mapped_column(
        String(320), nullable=False, unique=True, index=True
    )

    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # `USER` or `ADMIN`. Defaulted in the database as well as the model so an account
    # inserted by hand from psql is an ordinary user rather than accidentally privileged —
    # the safe direction for a column whose other value grants administrative access.
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=USER_ROLE_USER
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_true()
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AuthSession(Base):
    """One signed-in web session, identified by the digest of an opaque token.

    The token itself is 256 bits from the system CSPRNG and exists in exactly two places:
    the `Set-Cookie` header that hands it to the browser, and the browser. It is never
    written to this table, never logged, and cannot be recovered from what is stored — the
    same discipline `ApiKey` follows, and for the same reason.

    SHA-256 rather than Argon2, and that is not an inconsistency with `User.password_hash`
    above. The password is human-chosen and needs a slow hash to make guessing expensive;
    this token is 256 random bits, so there is nothing to guess and the cost of a slow hash
    would be paid on every authenticated request for no security gained.

    A session ends in one of two ways and the table records which. `expires_at` is the
    deadline set when it was created; `revoked_at` is a deliberate end — a sign-out, or the
    previous session being displaced by a new sign-in. Revocation is a timestamp rather than
    a deletion so that a row remains to say a session existed and when it stopped.

    Rows are not the authority on *whether* a session is usable; the query in
    `app.web_auth.session_user` is, and it asks for all three conditions at once — not
    revoked, not expired, and the account still active.
    """

    __tablename__ = "auth_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # `CASCADE` here, unlike the `RESTRICT` on the ownership columns: a session is a
    # transient credential, not a forensic record, and there is nothing to preserve about
    # the sessions of an account that no longer exists.

    # Unique because authentication looks a row up by it — one indexed lookup on the digest
    # of the presented cookie, with no plaintext anywhere in the query.
    token_hash: Mapped[str] = mapped_column(
        String(SHA256_HEX_LENGTH), nullable=False, unique=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Null on every session that is still live, and a timestamp on every one deliberately
    # ended. Never "0" or an epoch date — a session that was never revoked has no revocation
    # time, and null is how that is said.
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AnalysisSegment(Base):
    """One piece of within-media evidence behind a signal, as the provider reported it.

    Three evidence sources write rows here and they describe the media in different units,
    so the columns come in groups and a row fills the one its source has facts for. Which
    group is populated follows from the parent signal's `signal_type`; nothing is
    duplicated into another group's columns to make the table look uniform, because a
    filled-in figure a provider never gave is a fabricated one (rule 11).

    Clip evidence — `synthetic_video`. NVIDIA's synthetic-video detector scores the video
    in clips and reports, per clip, a frame index and a raw model logit, and nothing else.
    It reports no times at all, so `start_time`/`end_time` stay null on these rows:
    converting its frame index into seconds would mean inventing a figure NVIDIA never
    gave (D019). This is why `clip_index` and `logit` are nullable rather than required —
    they were `NOT NULL` while this was the only source, and P5-T3 widened them rather
    than making the second source carry placeholder values.

    Temporal evidence — `active_speaker`. NVIDIA's Active Speaker NIM reports per *frame*
    whether a tracked face is speaking, and contiguous runs of that are aggregated into
    real time ranges before they reach this table. Those rows carry `start_time`/`end_time`
    in seconds plus the identity the range is about, and leave `clip_index`/`logit` null:
    there is no clip and no logit in an active-speaker result.

    Audio window evidence — `audio_authenticity`. The local AASIST checkpoint consumes a
    fixed 64600-sample window and emits two raw logits for it, so a recording is cut into
    consecutive windows and each becomes a row. `clip_index` holds the window's position in
    that chronological sequence — the third source to use the column, and the same thing it
    has always held: the index of the unit the parent signal's provider was given, which for
    NVIDIA is a clip identified by its middle frame and here is DeepGuard's own window.
    `logit` and `bona_fide_logit` hold the graph's two outputs in graph order.

    `start_time`/`end_time` are filled on these rows too, and they mean something narrower
    than they do on an active-speaker row: they are `start_sample / 16000` and
    `end_sample / 16000` for the window this codebase cut, which is a record of what was fed
    to the model. AASIST publishes no chunk-to-time mapping and reports no segments, so these
    bounds are DeepGuard preprocessing metadata and are never a claim that the model located
    anything in that interval. The parent signal's `signal_type` is what tells the two
    readings apart. `face_id` and `speaker_label` stay null: there is no face and no voice
    identity in an anti-spoofing result.

    There is deliberately no `score` column for any of them. The clip figure is a raw logit
    on the model's own scale rather than a probability like `AnalysisSignal.score`, an
    active-speaker range has no number attached at all — NVIDIA's per-frame
    `face_detection_confidence` is confidence in having found a face, not in that face
    speaking, and storing it in a `score` column would advertise it as the latter — and the
    audio logits are likewise raw model output with no calibration behind them.

    Rows hang off the signal rather than the analysis, because either kind of evidence is
    only meaningful for the detector run that produced it.
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

    # Which scored unit this row is. On synthetic-video evidence it is the frame index of
    # the clip's middle frame, exactly as NVIDIA reported it; the provider's field is a
    # uint32, which outgrows a 32-bit column, so it is stored wide. On audio evidence it is
    # the window's zero-based position in the chronological sequence this codebase cut.
    # Null on evidence that scores no discrete unit.
    clip_index: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    # Raw model output for the unit above. Untransformed: not a probability, not rescaled,
    # not rounded. NVIDIA's synthetic-video detector emits one figure per clip and fills
    # only `logit`; AASIST emits two per window and fills both columns, in graph order —
    # `logit` is output column 0 and `bona_fide_logit` is output column 1, which is the
    # column the checkpoint's own repository reads as the bona fide score
    # (`clovaai/aasist/main.py:307`). That mapping is the model's, not a threshold or a
    # class this codebase assigned. Null on evidence that carries no logit.
    logit: Mapped[float | None] = mapped_column(Float, nullable=True)
    bona_fide_logit: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The time range this row covers, in seconds from the start of the analysed video.
    # Null on evidence the provider reported no times for.
    start_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Who the range is about, which is what makes an active-speaker segment readable: a
    # time range on its own only says "someone was speaking".
    #
    # `face_id` is NVIDIA's own identifier for the face it tracked across frames, stored
    # wide for the same uint32 reason as `clip_index`. `speaker_label` is pyannote's label
    # for the voice NVIDIA matched that face to — its own string, kept as produced, never
    # the integer this codebase assigned it for NVIDIA's wire format. It is null when
    # NVIDIA matched the face to no diarized voice at all, which is a real observation
    # about the frame rather than missing data.
    face_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    speaker_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ShadowRun(Base):
    """One experimental workload's run against one analysis, and what it observed (R6-T1).

    Shadow mode exists so an uncalibrated detector can be exercised on real traffic before
    anybody is entitled to act on what it says. That entitlement is the whole design problem,
    and it is solved here structurally rather than by discipline: shadow observations live in
    this table and nowhere else, and nothing that answers a customer — the public API, the
    dashboard API, the report — and nothing the risk engine reads names it. `analysis_signals`
    is the forensic record a decision may be taken from; this is not that table and is not a
    sibling of it. An uncalibrated number written beside calibrated ones would eventually be
    read as one.

    So the isolation is not a filter that could be forgotten in one query. There is no join
    from `analyses` to here that any reader traverses, no column on `analyses` pointing at it,
    and no code path that turns a row here into the evidence types `app.risk_engine` accepts.
    `tests/test_shadow_mode.py` asserts that absence over the API and risk-engine modules,
    so a future reader that started selecting this table would fail the suite rather than
    quietly publish an experiment.

    The row is also the queue. Production splits `analysis_jobs` from `analysis_signals`
    because a job carries a lease, a request id and a failure while a signal is one immutable
    reading a detector gave, and the two have genuinely different lives. A shadow run does
    not: one workload runs once per analysis and produces one observation, so a second table
    to hold that observation would be a foreign key between a row and itself. It is one row
    that starts `queued` with nothing observed and ends `completed` carrying what it saw.

    `evidence` is deliberately an opaque JSON document rather than the scored columns
    `analysis_signals` has. Nothing in this codebase interprets it — R6-T1 runs a stub — and
    giving it a `score` column would invite the comparison with a calibrated score that the
    separation exists to prevent. What a shadow observation means is a question for the
    calibration task that will eventually read these rows offline.
    """

    __tablename__ = "shadow_runs"

    # One run per workload per analysis. The uniqueness is what makes re-enqueueing safe:
    # the worker inserts on the way out of a completed job, and a duplicate insert — a
    # re-run, two workers racing, a job concluded twice — is refused by the database rather
    # than quietly doubling the corpus a future calibration would be measured on.
    __table_args__ = (
        UniqueConstraint("analysis_id", "workload", name="uq_shadow_runs_analysis_workload"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Which experimental workload this run is of. A name, not a provider and a signal type:
    # a shadow workload is not yet a detector with an identity worth splitting in two, and it
    # will have one when one is integrated.
    workload: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False)

    # When the claim on this run stops being believed, for the same reason `analysis_jobs`
    # has one: a worker that dies mid-run would otherwise leave the row `processing` forever
    # and this analysis would silently drop out of the shadow corpus. Null on every run
    # nobody is holding.
    #
    # No heartbeat thread renews it, which is the one place this deliberately does less than
    # the production job does. The lease is set generously at claim time and a workload that
    # outlives it is recovered as stale — acceptable while the workload is a stub, and the
    # task that integrates a workload long enough to need renewing is the task that should
    # add the renewal.
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Why the run failed, as a class name rather than the exception's own text — which can
    # quote credentials or SQL, exactly as `app.worker` records a failed job. Null on every
    # run that has not failed.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Which deployment of the experimental workload produced the observation, so a row stays
    # interpretable once the experiment moves on. Null until the run has produced one.
    provider_version: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # What the workload observed, as the workload reported it, uninterpreted. Null on a run
    # that has not finished and on one that failed: an empty document would be an observation
    # nobody made.
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
