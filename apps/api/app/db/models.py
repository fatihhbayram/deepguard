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

Since R8-T5 there is also `admin_audit_events`: an append-only record of the privileged
mutations an administrator makes to somebody else's account. It is the one table here that
deliberately duplicates data — the two email snapshots are copies of `users.email` frozen at
the moment of the change — because an audit row is a historical statement and a join would
let a later rename rewrite what the record says happened.

Since R8-T7 there is `analysis_reviews` alongside them: what a human said about an analysis,
in its own table precisely so that it is not on `analyses`. The forensic columns there are
written once by the worker and no administrative route may reach them; a review is a mutable
opinion, and keeping the two in separate rows is what makes "the detectors said this, a person
said that" readable as two statements instead of one overwritten record.

Media identity is not analysis identity. Storage keys and hashes are content-addressed,
so the same bytes can legitimately be uploaded and analysed more than once; none of
those columns is unique.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
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
    false as sa_false,
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

# How the analysed artifact reached DeepGuard (R7-T12). `upload` is a file a client sent;
# `url` is a file this service fetched from an address a client named. Two doors and no
# third: an analysis whose media arrived some other way does not exist.
#
# The absence of either — a null column — is a third state and not a third method. It means
# the analysis predates this record being kept, and is read as "not recorded" rather than
# resolved to a guess.
ACQUISITION_METHOD_UPLOAD = "upload"
ACQUISITION_METHOD_URL = "url"

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

    # How the original's container says its picture should be turned for display, in degrees
    # clockwise: 0, 90, 180 or 270. Read from the display matrix, or from the legacy
    # `rotate` tag for media that carries only that — see `app.media._parse_display_rotation`
    # for the two conventions and how they are reconciled.
    #
    # Null means the row predates this column, which is not the same fact as `0`. `0` is a
    # probe that ran and found no rotation recorded; null is no probe for rotation having
    # happened at all, and resolving it to `0` would state an upright original on evidence
    # nobody gathered.
    #
    # Nothing reads this to decide anything. It explains the divergence between the two
    # dimension pairs below rather than producing either of them, and no risk rule,
    # threshold or detector sees it.
    display_rotation: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # The picture size of the artifact a detector was actually pointed at, probed from that
    # artifact's own bytes.
    #
    # `width`/`height` above are the *original's* coded dimensions, and for a rotated phone
    # video they are not what any detector saw: ffmpeg bakes the display matrix into the
    # derivative, so a 1920x1080 original becomes a genuine 1080x1920 derivative, and the
    # even-dimension pad can shift a side by a pixel on top of that. Reporting the original's
    # figures as the analysed ones told the reader a detector examined geometry it never
    # received, which is the error this pair exists to end.
    #
    # Never inferred. These are measured off the file — the derivative when one was
    # transcoded, the original when it was canonical enough to be sent as-is — because the
    # alternative is reimplementing ffmpeg's geometry here and trusting it to stay in step.
    #
    # Null in three cases, all of them honest: a row written before this column existed, a
    # job whose transcode never produced a derivative to measure, and a derivative that was
    # produced but could not be probed. A reader must take null as "not recorded" and fall
    # back to reporting the original's encoded size as such — never as the analysed size.
    analyzed_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analyzed_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Whether this media has to be transcoded before a detector can read it. Decided at
    # upload from the probe, because that decision needs `major_brand` and no column
    # holds it — the worker could not re-derive it from this table. Past tense is
    # deliberate even though the transcode happens later: it describes the artifact the
    # analysis ends up being detected against, and an analysis whose transcode never
    # succeeded never completes.
    was_normalized: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # How this media was acquired, and the only column that answers it (R7-T1).
    #
    # False is one file: an upload, or a URL whose source served a single already-muxed
    # artifact, stored byte-for-byte as the forensic original. True is an artifact this
    # service assembled — a separate video stream and a separate audio stream, fetched and
    # then muxed into one container by ffmpeg here — which is the only form YouTube offers
    # above 360p. There is no published file for such an acquisition to be a copy of.
    #
    # Persisted rather than logged because the distinction outlives the request. Provenance
    # is read minutes later, in the worker, off the stored original, and "these are the bytes
    # the source served" is true of one of these artifacts and false of the other. The
    # process holding the file has no other way to tell which it has.
    #
    # It is a fact about the acquisition and nothing more. No risk rule, threshold or verdict
    # reads it and none may: assembled media is neither more nor less authentic than served
    # media, and reading suspicion into an ffmpeg mux would be inventing evidence out of
    # DeepGuard's own plumbing. What it exists to prevent is the opposite error — a report
    # describing an assembled file as the publisher's original bytes.
    #
    # `False` server-side for every row written before this column existed, which is not a
    # guess: those rows could only have been acquired as a single file.
    was_assembled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sa_false()
    )

    # Which door this media came through, and the host it came from (R7-T12). Beside
    # `was_assembled` above, which says how the artifact was put together; these two say
    # where it came from, and the three are only readable together.
    #
    # `acquisition_method` is `ACQUISITION_METHOD_UPLOAD` or `ACQUISITION_METHOD_URL`. Null
    # means the row predates this column: the analysis was acquired by a request that
    # recorded nothing about how, and no evidence survives to say which door it used. That is
    # reported as "not recorded" and never resolved to a default — a row silently called an
    # upload would be a fabricated provenance fact, which is the one error this pair exists
    # to prevent.
    #
    # It is also what makes `was_assembled` readable. On a row with a null method, `false` is
    # only R7-T1's server default and carries no claim about who served the bytes; it becomes
    # an acquisition claim exclusively alongside a recorded method.
    acquisition_method: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # The normalized hostname of a URL submission: lowercased, `www.` stripped, and nothing
    # else of the URL — no scheme, no port, no path, no query, no fragment, no credentials.
    # Null for an upload, where there is no host, and null for a row from before this column.
    #
    # The hostname alone, deliberately. A submitted URL routinely carries a signed expiry, an
    # access token, a session id or a private path in its query string, and a column holding
    # the whole URL would put those in the database, in every backup of it and in front of
    # every reader of the analysis. The host is the part that answers "where did this come
    # from" and the largest part of a URL that carries no secret.
    #
    # It is the host of the address the client submitted and that the SSRF guard validated,
    # not of wherever a redirect chain ended. Recording the endpoint of a redirect would be
    # recording something the submitter never named, and following that chain into the record
    # is exactly the provenance claim this column does not make.
    #
    # No risk rule, threshold or detector reads it. Media fetched from one host is neither
    # more nor less authentic than media fetched from another, and this column asserts
    # nothing about the publisher, the account or the file behind that host.
    source_host: Mapped[str | None] = mapped_column(String(253), nullable=True)

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


# What an audit row says was done. Named as constants rather than written as literals at the
# call sites so that the writer, the reader and the test that proves they agree all name the
# same thing.
#
# Deliberately coarse. The action is "an administrator changed this account", and *what*
# changed is the `changes` payload — a taxonomy that split this into USER_ROLE_CHANGED and
# USER_DEACTIVATED would have to decide what to call the request that does both, and would
# make a reader parse the action name to learn something the payload already states exactly.
AUDIT_ACTION_USER_UPDATED = "USER_UPDATED"

# The two halves of an API key's life, added in R8-T6 and written by `admin_api_keys.py`.
#
# Split into two actions where the account change above is deliberately one, because these are
# genuinely two different events rather than two fields of the same one: a key is created once
# and revoked once, the two carry different `changes` payloads, and an operator reading the log
# for "when did this customer lose access" is asking about exactly one of them. The coarseness
# argument that keeps USER_UPDATED single does not apply — there is no request that does both.
#
# **Neither event carries the key.** Not the plaintext, which exists only in the response to
# the creation request and is never written anywhere, and not `key_hash` either: the digest is
# the credential's stored form, and a log that reprints it has widened what a reader of the
# audit screen can see. What these rows say is which key, by id, and who did it.
AUDIT_ACTION_API_KEY_CREATED = "API_KEY_CREATED"
AUDIT_ACTION_API_KEY_REVOKED = "API_KEY_REVOKED"

# What kind of thing an audit row is about. `target_id` is a string rather than a typed
# foreign key precisely so this column can mean something — and since R8-T6 it earns that: an
# API key is not a user, and a row that names its own type is a row that stays readable when
# it is not.
AUDIT_TARGET_USER = "USER"
AUDIT_TARGET_API_KEY = "API_KEY"
# Since R8-T7 an audit row may also be about an analysis. It is the first target type here
# that is not a credential or an account, and it names the analysis the review was attached
# to — never the review row, which has no identity of its own beyond the analysis it belongs
# to.
AUDIT_TARGET_ANALYSIS = "ANALYSIS"

# The two halves of a human review's life (R8-T7), written by `admin_analyses.py`.
#
# Split into two actions for the reason the API key pair is split and the account change is
# not: the first review of an analysis and a later revision of it are genuinely different
# events. "When was this looked at" and "when was that answer changed" are separate questions,
# and a reader of the log should not have to inspect the `changes` payload to tell which of
# the two a row is.
#
# **Neither event carries the note.** `changes` records that the note moved, as a boolean, and
# not what it moved to — see `AnalysisReview.note` for why the text stays on the review row
# alone. The action names say a review happened; the review says what it says.
AUDIT_ACTION_REVIEW_CREATED = "REVIEW_CREATED"
AUDIT_ACTION_REVIEW_UPDATED = "REVIEW_UPDATED"


class AdminAuditEvent(Base):
    """One privileged change an administrator made, recorded as it happened (R8-T5).

    **Append-only, and that is a property of the application rather than of the table.** There
    is no route that deletes or edits one of these rows, and `app/api/admin_audit.py` offers
    nothing but a GET — the reasoning is stated there. PostgreSQL would happily accept an
    UPDATE; what makes this an audit log is that nothing in the codebase issues one.

    **Written in the same transaction as the change it describes.** `update_user` adds the
    event to the session that holds the mutated `User` and commits both together, so there is
    no window in which the account has been changed and the record of it has not — and a
    request that fails its invariant check rolls back the event along with the change it never
    made. An audit log written afterwards, or by a listener, is a log that disagrees with the
    database exactly when something went wrong, which is the only time anybody reads it.

    **The two email snapshots are copies on purpose.** They are `users.email` as it read at
    the moment of the change, frozen. The obvious alternative — join to `users` at read time —
    would mean that renaming an account silently rewrote the history of what it did, and that
    deleting one blanked it. An audit row is a statement about the past and has to keep saying
    the same thing; that is worth the duplication, and it is the one place in this schema
    where duplication is the correct answer.

    **Nothing here points at another row.** Neither the actor nor the target is a foreign key,
    so no part of this table can be made to disappear, cascade or blank out by something that
    happens to the accounts it describes. That is what lets these rows outlive the lifecycle of
    the people in them, which is the whole proposition of an audit log.

    What is never in here: a password hash, a session token, a cookie, an API key. `changes`
    holds `role` and `is_active` and nothing else — both are already visible to every
    administrator through the account listing, so the audit log does not widen what anybody
    can see. It records who did it and when, which the listing cannot.
    """

    __tablename__ = "admin_audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Who made the change. Taken from the `require_admin` dependency at the call site, never
    # from the request body — an actor a caller can name is not an actor, and this column is
    # the whole point of the table.
    #
    # **Deliberately not a foreign key.** It holds a `users.id` and is not constrained to one,
    # which is the same decision `target_id` makes one field down and is made here for the same
    # reason: an audit event is a historical fact, and a fact that the database can refuse to
    # keep once the account it names is gone is not one. A foreign key would make this table an
    # obstacle to deleting a user — `RESTRICT` by blocking it outright, `CASCADE` by destroying
    # exactly the evidence the log exists to preserve, and `SET NULL` by quietly erasing who
    # did it. All three let the account lifecycle edit the past.
    #
    # What makes the row still readable after such a deletion is that it does not depend on the
    # account at all: `actor_email_snapshot` beside this column was frozen at the time, so the
    # event names the administrator in its own right rather than by pointing at a row that may
    # no longer be there. The id remains the durable handle for correlating events by actor,
    # and it is indexed for that; it is simply not a promise that the account still exists.
    actor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)

    # The actor's address as it read at the time. Nullable because a snapshot is a record of
    # what was known, and a null here would mean it was not — not that the account had no
    # address. See the class docstring for why this is a copy and not a join.
    actor_email_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)

    action: Mapped[str] = mapped_column(String(64), nullable=False)

    # What sort of object was changed, and which one. The id is text rather than a foreign key
    # to `users`: a typed column would have to be replaced the first time something that is
    # not a user becomes auditable, and a foreign key would put this table back in the
    # business of caring whether the row it describes still exists.
    target_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target_id: Mapped[str] = mapped_column(String(255), nullable=False)
    target_email_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # The fields that actually moved, as `{"field": {"old": ..., "new": ...}}`. Only the ones
    # that changed: a payload restating an untouched field would make every row look like a
    # change to everything, and the no-op case would produce a row saying nothing happened.
    #
    # Plain `JSON`, not the `JSONB` the rest of this module uses. Nothing queries inside this
    # document — it is read back whole, by one screen, in the order it was written — and JSONB
    # buys indexable containment operators at the cost of not round-tripping key order. Here
    # the ordering is the only structure the payload has.
    changes: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # The correlation id of the request that made the change, when there was one. Same id the
    # application's logs carry, so an audit row and the log lines around it can be lined up
    # without matching on timestamps.
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Indexed because the only read this table has orders by it, newest first.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


# What a human reviewer has said about an analysis, as workflow state (R8-T7). Two values, and
# the vocabulary is deliberately operational: `REVIEWED` means somebody looked, and
# `NEEDS_FOLLOW_UP` means somebody looked and wants it looked at again.
#
# **Neither of them is a forensic answer, and no value here ever will be.** "Genuine", "fake",
# "confirmed" and the like belong to `Analysis.risk_level`, which is written by the risk engine
# under a named ruleset and a named calibration, and which nothing on this table may
# contradict. A status that reads as a verdict would turn this row into a second, unversioned,
# unexplainable classification of the same media sitting beside the real one — and the day the
# two disagreed, there would be no way to say which of them the report meant.
REVIEW_STATUS_REVIEWED = "REVIEWED"
REVIEW_STATUS_NEEDS_FOLLOW_UP = "NEEDS_FOLLOW_UP"

# Every status that may be stored. Named as a tuple so the validator, the check constraint and
# the test that proves they agree all read the same list.
REVIEW_STATUSES = (REVIEW_STATUS_REVIEWED, REVIEW_STATUS_NEEDS_FOLLOW_UP)

# The third state, which is not in the tuple above because it is not a value: an analysis with
# no `AnalysisReview` row has not been reviewed. It is named here because the API says the word
# in its responses and the screen prints it, and a spelling invented separately in each of
# those places is a spelling that eventually differs.
#
# Not stored, and there is no route that writes it. "Unreviewed" is the absence of the row, so
# every analysis that existed before this table did is already in that state with nothing
# backfilled — and un-reviewing one would be a deletion, which this application does not offer.
REVIEW_STATUS_UNREVIEWED = "UNREVIEWED"

# The longest note the column will hold. Bounded because an unbounded text field on a row an
# administrator can rewrite is an unbounded row an administrator can rewrite; a thousand
# characters is enough for "checked the source video against the broadcaster's upload, the
# timestamps disagree" and short of enough to paste a case file into.
MAX_REVIEW_NOTE_LENGTH = 1000

# The constraint holding `status` to the two names above. Named here because the model, the
# migration and the test that proves it bites all have to mean the same constraint — the same
# reason `SINGLE_OWNER_CONSTRAINT` is a name rather than a string written three times.
REVIEW_STATUS_CONSTRAINT = "ck_analysis_reviews_status"


class AnalysisReview(Base):
    """What a human said about an analysis, kept strictly apart from what the detectors said.

    **This table is mutable and the forensic record is not, which is the whole reason it is a
    table.** `Analysis` holds `risk_level`, `risk_rules_version`, `risk_calibration_id` and
    `risk_rule_id`; `AnalysisSignal` holds what each detector answered. All of those are
    statements about evidence, written once by the worker, and an administrator has no route to
    any of them. The obvious alternative to this table — two more columns on `analyses` — would
    have put a mutable opinion in the same row as the immutable measurement, and the first
    person to write an UPDATE against that row by hand would have had every risk column within
    reach of a typo. Here, the widest mistake anybody can make through this application is to
    the review, and the analysis it describes is untouched by construction.

    So the two layers are readable apart, and stay apart: the report renders the detector
    result and renders the review beside it, and never merges them into one answer.

    **One review per analysis, enforced by making the analysis the key.** There is no history
    of previous notes and no second reviewer — a revision overwrites. That is a real limitation
    and it is the deliberate one: what actually changed, when, and who did it is in
    `admin_audit_events`, written in the same transaction as the change, which is a stronger
    record than a version chain on this table would be and one that cannot be edited from here.
    A second opinion, a threaded discussion, or an assignment queue are all features this
    schema does not have; each of them is a new table when somebody has the requirement.

    `ON DELETE CASCADE` from the analysis. A review of an analysis that no longer exists is not
    a historical fact worth keeping — unlike an audit event, which is about a *person's action*
    and survives everything — it is an annotation with nothing left to annotate.
    """

    __tablename__ = "analysis_reviews"

    # In the database and not only in the request model. The status vocabulary is the one thing
    # about this table that must never widen by accident — see `status` below — and a rule that
    # lives only in a Pydantic model is one route away from not running.
    __table_args__ = (
        CheckConstraint(
            "status IN ('%s')" % "', '".join(REVIEW_STATUSES),
            name=REVIEW_STATUS_CONSTRAINT,
        ),
    )

    # The analysis this is about, and the primary key. One column doing both jobs is what makes
    # "one review per analysis" a property of the table rather than a rule the endpoint
    # remembers: a second insert for the same analysis is a duplicate key, not a second row.
    #
    # A surrogate id plus a unique index would express the same constraint and would also
    # invite a second row later by relaxing the index. There is nothing to relax here.
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # `REVIEWED` or `NEEDS_FOLLOW_UP`. Constrained in the database as well as in the request
    # model, because this is the column whose vocabulary the whole design rests on: a value
    # outside the two — inserted by hand, by a migration, or by a route added later — would be
    # a forensic-sounding word in a governance column, which is exactly what this table exists
    # to keep out. The application would refuse it; the database refuses it too.
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    # Why, in the reviewer's own words. Plain text and nothing else: no Markdown, no HTML, no
    # rendering of any kind on the way in or the way out. The screen prints it as text, which
    # is what makes a note containing `<script>` a note containing `<script>` rather than an
    # administrator-authored injection into another administrator's browser.
    #
    # `nullable=False` with an empty string as the "no note" value, rather than a null. The
    # endpoint's no-op comparison is the reason: two spellings of "nothing" would make `None`
    # and `""` compare unequal and write an audit row for a change that did not happen.
    note: Mapped[str] = mapped_column(String(MAX_REVIEW_NOTE_LENGTH), nullable=False)

    # Who reviewed it, as `users.id` — taken from the session the request authenticated with,
    # never from anything a request body could name.
    #
    # **Deliberately not a foreign key**, the same decision `AdminAuditEvent.actor_id` makes and
    # for a version of the same reason. A reference here would make this table an obstacle to
    # removing an account: `RESTRICT` would block the deletion outright, `CASCADE` would delete
    # the reviews that person wrote — quietly losing governance attached to analyses that are
    # still live — and `SET NULL` would leave a review nobody signed. The id stays the durable
    # handle for "everything this reviewer looked at", and is indexed for exactly that; it is
    # simply not a promise that the account still exists.
    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    # The reviewer's address as it read when they last wrote this review, frozen. What keeps
    # the row readable once the account is renamed or gone, which is the half of the decision
    # above that makes the missing foreign key safe rather than merely convenient.
    #
    # Nullable because a snapshot records what was known, and a null says it was not — not that
    # the account had no address.
    reviewer_email_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # When the review last actually moved. `onupdate` in the model and not only a server
    # default, so a revision written through the ORM carries a new timestamp — and because the
    # endpoint returns before writing when nothing changed, a request that altered nothing
    # leaves this alone. "Last reviewed" therefore means the last time somebody changed their
    # answer, not the last time somebody opened the form.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
