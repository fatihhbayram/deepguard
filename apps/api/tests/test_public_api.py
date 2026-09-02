"""The external B2B API: who may call it, whose analyses they see, and what comes back.

Two kinds of test, split by what they can honestly establish.

The submission tests fake the session. What is under test there is the route — that it
refuses an unauthenticated caller before doing any work, that it reuses the internal
pipeline rather than a copy of it, that it stamps the authenticated key onto the analysis
it commits, and that the response says the two things it promises and nothing more.

The read tests run against real PostgreSQL, because isolation is not a property of the
route. It is a `WHERE` clause, and only a database that actually holds one key's analyses
alongside another's can show that the clause keeps them apart. A fake session asked for one
analysis would hand back whichever row it was given, and a test written on top of that
would pass no matter what the filter said — which is the same trap `test_api_key_auth.py`
had to be pulled out of for the inactive key.
"""

import hashlib
import json
import threading
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from app import media, storage
from app.api.analyses import (
    ActiveAnalysisLimitReached,
    StoredUpload,
    active_analyses,
    persist_analysis,
)
from app.api.public_v1.analyses import MAX_ACTIVE_ANALYSES
from app.auth import generate_api_key
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_QUEUED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    AnalysisJob,
    AnalysisSignal,
    ApiKey,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine, get_session
from app.detection import (
    AASIST_PROVIDER,
    AUDIO_AUTHENTICITY_SIGNAL,
    C2PA_PROVIDER,
    EFFICIENTNET_B7_PROVIDER,
    FACE_MANIPULATION_SIGNAL,
    NVIDIA_PROVIDER,
    PROVENANCE_SIGNAL,
    SYNTHETIC_VIDEO_SIGNAL,
)
from app.main import app
from app.media import MediaMetadata
from app.web_auth import hash_password, require_user
from tests.conftest import DASHBOARD_ORIGIN

SUBMIT_URL = "/api/public/v1/analyses"

# What the public read promises, exactly. Written out rather than derived from the model,
# so adding a field to the response is a deliberate edit here and not something that
# happens to a paying integration by accident.
PUBLIC_FIELDS = {
    "id",
    "status",
    "created_at",
    "risk_level",
    "risk_rules_version",
    "risk_rule_id",
    "risk_calibration_id",
    "signals",
}

PUBLIC_SIGNAL_FIELDS = {"provider", "signal_type", "status", "provider_version", "score"}

# Internal facts the dashboard's own responses carry. None of them belongs to a customer,
# and this is the list the public payloads are checked against.
INTERNAL_FIELDS = {
    "storage_key",
    "derivative_storage_key",
    "derivative_sha256",
    "sha256",
    "original_sha256",
    "metadata",
    "media",
    "size_bytes",
    "declared_content_type",
    "content_type",
    "filename",
    "original_filename",
    "was_normalized",
}


def read_url(analysis_id) -> str:
    return f"{SUBMIT_URL}/{analysis_id}"


def authorization(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


# --- submitting ------------------------------------------------------------------------
#
# A faked session, so no database is needed to say what the route does with an upload.


class SubmissionSession:
    """A session that authenticates one key and records everything the route persists.

    `execute` answers the two kinds of statement the submission path issues: the row
    lookups — the API-key authentication and the `FOR UPDATE` on that same key — through
    `scalar_one_or_none`, and the active-analysis count through `scalar_one`. The rest is
    the same recording session the internal upload tests use, so the rows the pipeline
    builds can be inspected without PostgreSQL.

    `active` is what the count answers, and it is a dial on the fake rather than a fact
    about a database. It is set to zero here and never moved: these tests are about what
    the route does with an upload, and nothing about the limit is claimed from them. The
    limit is a lock, it is proven against real PostgreSQL in the section at the bottom of
    this file, and a fake that could be told to report five would let a test assert a `429`
    while the enforcement was gone.
    """

    def __init__(self, api_key: ApiKey | None) -> None:
        self.api_key = api_key
        self.active = 0
        self.added = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement):
        session = self

        class Result:
            def scalar_one_or_none(self):
                return session.api_key

            def scalar_one(self):
                return session.active

        return Result()

    def add(self, instance):
        self.added.append(instance)

    def flush(self):
        # The real flush is what assigns the Python-side UUID defaults.
        for instance in self.added:
            if getattr(instance, "id", None) is None:
                instance.id = uuid.uuid4()

    def commit(self):
        self.flush()
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeMinio:
    """Stand-in for the object store, so submission needs no live MinIO."""

    def __init__(self) -> None:
        self.uploads = []

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        pass

    def fput_object(self, bucket, key, file_path, content_type=None):
        self.uploads.append((bucket, key, Path(file_path).read_bytes(), content_type))


@pytest.fixture
def fake_minio(monkeypatch):
    fake = FakeMinio()
    monkeypatch.setattr(storage, "client", fake)
    return fake


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch):
    """Replace only the subprocess call, so real parsing and validation still run."""
    probe = json.dumps(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "12.34",
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "12.34",
                "tags": {"major_brand": "mp42"},
            },
        }
    )

    async def run(path):
        return probe

    monkeypatch.setattr(media, "_run_ffprobe", run)


@pytest.fixture
def submitted_key():
    """A key the fake session will authenticate, and the plaintext to present it with."""
    generated = generate_api_key()
    key = ApiKey(
        id=uuid.uuid4(),
        name="acme-production",
        key_hash=generated.key_hash,
        is_active=True,
    )

    return generated.plaintext, key


@pytest.fixture
def submission(submitted_key):
    """A client over the product app whose session authenticates `submitted_key`."""
    _, key = submitted_key
    session = SubmissionSession(api_key=key)
    app.dependency_overrides[get_session] = lambda: session

    with TestClient(app) as client:
        yield client, session

    app.dependency_overrides.clear()


@pytest.fixture
def anonymous():
    """A client whose session authenticates nobody, for the refusal tests."""
    session = SubmissionSession(api_key=None)
    app.dependency_overrides[get_session] = lambda: session

    with TestClient(app) as client:
        yield client, session

    app.dependency_overrides.clear()


def submit(client, key: str | None = None, payload: bytes = b"pretend-this-is-video"):
    return client.post(
        SUBMIT_URL,
        files={"file": ("clip.mp4", payload, "video/mp4")},
        headers=authorization(key) if key else {},
    )


def persisted_analysis(session: SubmissionSession) -> Analysis:
    return next(row for row in session.added if isinstance(row, Analysis))


def test_submitting_without_a_key_is_refused(anonymous, fake_minio):
    client, session = anonymous

    response = submit(client)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_submitting_with_an_unknown_key_is_refused(anonymous, fake_minio):
    client, _ = anonymous

    response = submit(client, generate_api_key().plaintext)

    assert response.status_code == 401


def test_an_unauthenticated_submission_never_reaches_the_pipeline(anonymous, fake_minio):
    """The refusal happens before the upload is read, stored or committed.

    A gate that let the body through and refused afterwards would still answer 401, and
    would still have written an unauthenticated caller's bytes into the forensic bucket.
    """
    client, session = anonymous

    submit(client)

    assert fake_minio.uploads == []
    assert session.added == []
    assert session.commits == 0


def test_a_submission_is_accepted_and_queued(submission, submitted_key, fake_minio):
    client, session = submission
    plaintext, _ = submitted_key

    response = submit(client, plaintext)

    assert response.status_code == 202
    assert response.json() == {
        "id": str(persisted_analysis(session).id),
        "status": "queued",
    }


def test_the_submission_response_carries_nothing_internal(
    submission, submitted_key, fake_minio
):
    """Two fields, and specifically not the storage keys or the content identity.

    The internal route reports all of that to the dashboard, which is inside the trust
    boundary. A customer is not, and a bucket key in a response is an invitation to try it.
    """
    client, _ = submission
    plaintext, _ = submitted_key

    body = submit(client, plaintext).json()

    assert set(body) == {"id", "status"}
    assert INTERNAL_FIELDS.isdisjoint(body)


def test_a_submission_is_owned_by_the_key_that_sent_it(
    submission, submitted_key, fake_minio
):
    client, session = submission
    plaintext, key = submitted_key

    submit(client, plaintext)

    assert persisted_analysis(session).api_key_id == key.id


def test_a_submission_runs_the_existing_pipeline(submission, submitted_key, fake_minio):
    """The public route stores the forensic original and queues a job, exactly as the
    internal one does — it is the same function, and this is what says so behaviourally.
    """
    client, session = submission
    plaintext, _ = submitted_key
    payload = b"public-caller-bytes"

    submit(client, plaintext, payload=payload)

    digest = hashlib.sha256(payload).hexdigest()
    assert [key for _, key, _, _ in fake_minio.uploads] == [f"originals/{digest}"]
    assert [stored for _, _, stored, _ in fake_minio.uploads] == [payload]

    assert any(isinstance(row, MediaFile) for row in session.added)
    assert any(isinstance(row, AnalysisJob) for row in session.added)
    assert session.commits == 1


def test_the_public_route_validates_media_the_same_way(
    submission, submitted_key, fake_minio
):
    """An unsupported type is refused on the public route too, with the internal status."""
    client, session = submission
    plaintext, _ = submitted_key

    response = client.post(
        SUBMIT_URL,
        files={"file": ("notes.txt", b"not video", "text/plain")},
        headers=authorization(plaintext),
    )

    assert response.status_code == 415
    assert session.added == []


def test_a_dashboard_submission_carries_no_api_key(submission, fake_minio):
    """The internal route must never stamp a key onto what it commits.

    If it ever did, the dashboard's uploads would become visible to whichever customer that
    key belonged to. Since R1-T2 the route does record an owner — the signed-in account —
    and that is the other half of the same rule: the two ownership columns are mutually
    exclusive, and this route fills the web one.
    """
    client, session = submission
    user = User(
        id=uuid.uuid4(),
        email="operator@example.com",
        password_hash="unused",
        role=USER_ROLE_USER,
        is_active=True,
    )
    app.dependency_overrides[require_user] = lambda: user

    response = client.post(
        "/api/v1/analyses",
        files={"file": ("clip.mp4", b"dashboard-bytes", "video/mp4")},
        headers={"Origin": DASHBOARD_ORIGIN},
    )

    assert response.status_code == 202
    persisted = persisted_analysis(session)
    assert persisted.api_key_id is None
    assert persisted.owner_id == user.id


# --- reading ---------------------------------------------------------------------------
#
# Real PostgreSQL, because ownership isolation is a query and only a database can answer
# whether one key's analyses stay out of another key's read.

integration = pytest.mark.integration


@pytest.fixture(scope="module")
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def session(database):
    """A real session whose analyses and keys are removed again, in that order.

    Analyses first: `analyses.api_key_id` is `ON DELETE RESTRICT`, so a key cannot be
    deleted while an analysis still names it. The cleanup runs in the order the constraint
    requires, which is also a small demonstration that the constraint is really there.
    """
    analyses = []
    keys = []

    with SessionLocal() as db:
        yield db, analyses, keys

        db.rollback()
        for analysis_id in analyses:
            # Media, job and signal rows go with it through ON DELETE CASCADE.
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        db.flush()
        for key_id in keys:
            db.query(ApiKey).filter(ApiKey.id == key_id).delete()
        db.commit()


@pytest.fixture
def administrator(session):
    """A real administrator account, for the internal routes this file also touches.

    A persisted row rather than a stand-in, because the internal upload writes its id into
    `analyses.owner_id` and PostgreSQL holds that to a foreign key. Everything it owns is
    deleted before it is: `owner_id` is `ON DELETE RESTRICT`, so an analysis this account
    submitted through the route — and therefore never registered for cleanup by name —
    would otherwise block the account's removal and leak both rows into the next test.
    """
    db, _, _ = session

    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password("test-account-password"),
        role=USER_ROLE_ADMIN,
    )
    db.add(user)
    db.commit()

    yield user

    db.rollback()
    db.query(Analysis).filter(Analysis.owner_id == user.id).delete()
    db.flush()
    db.query(User).filter(User.id == user.id).delete()
    db.commit()


@pytest.fixture
def reader(session, administrator):
    """A client over the product app bound to the live session.

    Signed in as an administrator, because this file reaches the internal dashboard routes
    as well as the public ones and those have demanded a web session since R1-T2. The
    administrator is the role whose view of the dashboard is the whole system, which is what
    the tests below about the dashboard still seeing a public analysis are describing.
    """
    db, _, _ = session
    app.dependency_overrides[get_session] = lambda: db
    app.dependency_overrides[require_user] = lambda: administrator

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


def issue_key(session, *, name="customer", is_active=True) -> tuple[str, ApiKey]:
    """Persist a real key and hand back the plaintext to authenticate with."""
    db, _, keys = session
    generated = generate_api_key()

    key = ApiKey(name=name, key_hash=generated.key_hash, is_active=is_active)
    db.add(key)
    db.commit()
    keys.append(key.id)

    return generated.plaintext, key


def store_analysis(
    session,
    *,
    owner: ApiKey | None,
    status=ANALYSIS_STATUS_QUEUED,
    risk_level=None,
    risk_rules_version=None,
    risk_rule_id=None,
    risk_calibration_id=None,
) -> Analysis:
    """Persist an analysis and the media row every read joins onto."""
    db, analyses, _ = session

    analysis = Analysis(
        status=status,
        api_key_id=owner.id if owner is not None else None,
        risk_level=risk_level,
        risk_rules_version=risk_rules_version,
        risk_rule_id=risk_rule_id,
        risk_calibration_id=risk_calibration_id,
    )
    db.add(analysis)
    db.flush()

    digest = hashlib.sha256(str(analysis.id).encode()).hexdigest()
    db.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename="clip.mp4",
            content_type="video/mp4",
            size_bytes=4096,
            original_sha256=digest,
            original_storage_key=f"originals/{digest}",
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
            codec_name="h264",
            width=1920,
            height=1080,
            duration=12.34,
            frame_rate=30.0,
            pix_fmt="yuv420p",
            constant_frame_rate=True,
            was_normalized=False,
            derivative_storage_key=f"originals/{digest}",
        )
    )
    db.commit()
    analyses.append(analysis.id)

    return analysis


def store_signal(session, analysis: Analysis, **values) -> AnalysisSignal:
    db, _, _ = session
    signal = AnalysisSignal(analysis_id=analysis.id, **values)
    db.add(signal)
    db.commit()

    return signal


@integration
def test_reading_without_a_key_is_refused(reader, session):
    plaintext, key = issue_key(session)
    analysis = store_analysis(session, owner=key)

    response = reader.get(read_url(analysis.id))

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


@integration
def test_reading_with_an_unknown_key_is_refused(reader, session):
    _, key = issue_key(session)
    analysis = store_analysis(session, owner=key)

    response = reader.get(
        read_url(analysis.id), headers=authorization(generate_api_key().plaintext)
    )

    assert response.status_code == 401


@integration
def test_a_deactivated_key_cannot_read_its_own_analyses(reader, session):
    """Retiring a key ends its access to the analyses it created, without deleting them."""
    plaintext, key = issue_key(session, name="retired", is_active=False)
    analysis = store_analysis(session, owner=key)

    response = reader.get(read_url(analysis.id), headers=authorization(plaintext))

    assert response.status_code == 401


@integration
def test_a_key_reads_its_own_analysis(reader, session):
    plaintext, key = issue_key(session)
    analysis = store_analysis(session, owner=key)

    response = reader.get(read_url(analysis.id), headers=authorization(plaintext))

    assert response.status_code == 200
    assert response.json()["id"] == str(analysis.id)


@integration
def test_another_keys_analysis_is_not_found(reader, session):
    """The isolation property. Two real keys, two real analyses, one filtered read."""
    _, owner = issue_key(session, name="acme")
    intruder_plaintext, _ = issue_key(session, name="globex")
    analysis = store_analysis(session, owner=owner)

    response = reader.get(
        read_url(analysis.id), headers=authorization(intruder_plaintext)
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "analysis not found"}


@integration
def test_an_unowned_dashboard_analysis_is_not_found(reader, session):
    """Internal analyses carry a null owner, and no key's id equals null."""
    plaintext, _ = issue_key(session)
    analysis = store_analysis(session, owner=None)

    response = reader.get(read_url(analysis.id), headers=authorization(plaintext))

    assert response.status_code == 404


@integration
def test_a_forbidden_analysis_is_indistinguishable_from_one_that_does_not_exist(
    reader, session
):
    """Ownership must not be a probe. If a customer could tell "not yours" from "no such
    id", a public analysis id would be confirmable by anyone holding any key.
    """
    _, owner = issue_key(session, name="acme")
    intruder_plaintext, _ = issue_key(session, name="globex")
    analysis = store_analysis(session, owner=owner)

    forbidden = reader.get(
        read_url(analysis.id), headers=authorization(intruder_plaintext)
    )
    missing = reader.get(
        read_url(uuid.uuid4()), headers=authorization(intruder_plaintext)
    )

    assert (forbidden.status_code, forbidden.text) == (
        missing.status_code,
        missing.text,
    )


@integration
def test_an_administrator_still_sees_a_public_analysis(reader, session):
    """Key ownership narrows the public read; it does not hide the row from an operator.

    The administrator's dashboard is DeepGuard's own view of everything on file, and an
    analysis submitted through the public API must not disappear from it. An ordinary USER
    is a different question and is answered in `tests/test_dashboard_authorization.py`,
    where an API-key analysis is exactly one of the things they cannot reach.
    """
    _, key = issue_key(session)
    analysis = store_analysis(session, owner=key)

    listed = reader.get("/api/v1/analyses").json()

    assert str(analysis.id) in {row["id"] for row in listed}


@integration
def test_a_queued_analysis_reports_no_decision_and_no_signals(reader, session):
    """What a customer sees while polling: queued, and nothing concluded yet.

    Null risk fields here are the absence of a decision, and are a different fact from a
    `risk_level` of `UNKNOWN`, which is a decision the engine took.
    """
    plaintext, key = issue_key(session)
    analysis = store_analysis(session, owner=key, status=ANALYSIS_STATUS_QUEUED)

    body = reader.get(read_url(analysis.id), headers=authorization(plaintext)).json()

    assert body["status"] == "queued"
    assert body["risk_level"] is None
    assert body["risk_rules_version"] is None
    assert body["risk_rule_id"] is None
    assert body["risk_calibration_id"] is None
    assert body["signals"] == []


@integration
def test_a_completed_analysis_reports_the_stored_decision_and_signals(reader, session):
    plaintext, key = issue_key(session)
    calibration = hashlib.sha256(b"calibration").hexdigest()
    analysis = store_analysis(
        session,
        owner=key,
        status=ANALYSIS_STATUS_COMPLETED,
        risk_level="HIGH",
        risk_rules_version="p7-v1.0.0",
        risk_rule_id="R1",
        risk_calibration_id=calibration,
    )
    store_signal(
        session,
        analysis,
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        score=0.91,
        provider_version="1.2.3",
        signal_metadata={"logit": 2.5, "total_clips": 12},
    )
    store_signal(
        session,
        analysis,
        provider=C2PA_PROVIDER,
        signal_type=PROVENANCE_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        provider_version="0.9.0",
        signal_metadata={"manifest_exists": False},
    )
    store_signal(
        session,
        analysis,
        provider=AASIST_PROVIDER,
        signal_type=AUDIO_AUTHENTICITY_SIGNAL,
        status=SIGNAL_STATUS_FAILED,
        provider_version="rev-abc",
        signal_metadata={"error": "RuntimeError"},
    )
    store_signal(
        session,
        analysis,
        provider=EFFICIENTNET_B7_PROVIDER,
        signal_type=FACE_MANIPULATION_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        score=0.77,
        provider_version="repo@rev",
        signal_metadata={"frames_scored": 8},
    )

    body = reader.get(read_url(analysis.id), headers=authorization(plaintext)).json()

    assert body["status"] == "completed"
    assert body["risk_level"] == "HIGH"
    assert body["risk_rules_version"] == "p7-v1.0.0"
    assert body["risk_rule_id"] == "R1"
    assert body["risk_calibration_id"] == calibration

    signals = {signal["signal_type"]: signal for signal in body["signals"]}
    assert set(signals) == {
        SYNTHETIC_VIDEO_SIGNAL,
        PROVENANCE_SIGNAL,
        AUDIO_AUTHENTICITY_SIGNAL,
        FACE_MANIPULATION_SIGNAL,
    }
    assert signals[SYNTHETIC_VIDEO_SIGNAL]["score"] == pytest.approx(0.91)
    assert signals[SYNTHETIC_VIDEO_SIGNAL]["provider"] == NVIDIA_PROVIDER
    assert signals[SYNTHETIC_VIDEO_SIGNAL]["provider_version"] == "1.2.3"
    # A detector that did not answer keeps its own state and gets no score invented for it.
    assert signals[AUDIO_AUTHENTICITY_SIGNAL]["status"] == SIGNAL_STATUS_FAILED
    assert signals[AUDIO_AUTHENTICITY_SIGNAL]["score"] is None
    # Provenance reports no figure at all; null here is "no such measurement", not zero.
    assert signals[PROVENANCE_SIGNAL]["score"] is None
    # The face classifier is the second signal that carries a score, and it crosses the
    # boundary as its own raw figure — not rescaled, and not comparable with NVIDIA's above.
    # It sits beside a HIGH decision it had no part in: only the calibrated signal is read.
    assert signals[FACE_MANIPULATION_SIGNAL]["score"] == pytest.approx(0.77)
    assert signals[FACE_MANIPULATION_SIGNAL]["provider"] == EFFICIENTNET_B7_PROVIDER
    assert signals[FACE_MANIPULATION_SIGNAL]["provider_version"] == "repo@rev"


@integration
def test_the_read_response_carries_exactly_the_public_contract(reader, session):
    plaintext, key = issue_key(session)
    analysis = store_analysis(session, owner=key, status=ANALYSIS_STATUS_COMPLETED)
    store_signal(
        session,
        analysis,
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
        status=SIGNAL_STATUS_SUCCESS,
        score=0.4,
        provider_version="1.2.3",
        signal_metadata={"logit": 1.0, "total_clips": 3},
    )

    body = reader.get(read_url(analysis.id), headers=authorization(plaintext)).json()

    assert set(body) == PUBLIC_FIELDS
    assert set(body["signals"][0]) == PUBLIC_SIGNAL_FIELDS


@integration
def test_the_read_response_leaks_nothing_internal(reader, session):
    """No storage keys, no content hashes, no probed container facts — and no provider
    metadata document, which on a failed signal holds the exception's class name.
    """
    plaintext, key = issue_key(session)
    analysis = store_analysis(session, owner=key, status=ANALYSIS_STATUS_COMPLETED)
    store_signal(
        session,
        analysis,
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
        status=SIGNAL_STATUS_FAILED,
        provider_version="1.2.3",
        signal_metadata={"error": "GrpcTimeoutError"},
    )

    response = reader.get(read_url(analysis.id), headers=authorization(plaintext))
    body = response.json()

    assert INTERNAL_FIELDS.isdisjoint(body)
    assert INTERNAL_FIELDS.isdisjoint(body["signals"][0])
    # The stored diagnostic detail must not have travelled with the signal.
    assert "GrpcTimeoutError" not in response.text
    assert "originals/" not in response.text


@integration
def test_a_malformed_id_is_rejected_before_any_lookup(reader, session):
    plaintext, _ = issue_key(session)

    response = reader.get(read_url("not-a-uuid"), headers=authorization(plaintext))

    assert response.status_code == 422


@integration
def test_an_analysis_is_readable_end_to_end_by_the_key_that_submitted_it(
    reader, session, fake_minio, monkeypatch
):
    """Submit through the public route, then poll it back through the public route.

    The two endpoints are exercised against the same database here rather than separately,
    because the thing worth proving is that the owner the submission writes is the owner
    the read filters on — two halves that could each be correct and still not meet.
    """
    db, analyses, _ = session
    plaintext, key = issue_key(session)

    created = reader.post(
        SUBMIT_URL,
        files={"file": ("clip.mp4", b"end-to-end-bytes", "video/mp4")},
        headers=authorization(plaintext),
    )

    assert created.status_code == 202
    analysis_id = created.json()["id"]
    analyses.append(uuid.UUID(analysis_id))

    polled = reader.get(read_url(analysis_id), headers=authorization(plaintext))

    assert polled.status_code == 200
    assert polled.json()["id"] == analysis_id
    assert polled.json()["status"] == "queued"

    # And the analysis really is owned in the database, not merely readable by luck.
    assert db.get(Analysis, uuid.UUID(analysis_id)).api_key_id == key.id


# --- the concurrency limit ---------------------------------------------------------------
#
# Real PostgreSQL throughout, and for a stronger reason than the reads above: the limit is
# a lock. Its whole purpose is what happens when two requests for one key arrive at the
# same moment, and there is no such moment in a fake session — a single-threaded fake would
# report the limit holding while proving nothing about the only case it exists for.


def fill_active(session, key: ApiKey, count: int) -> list[Analysis]:
    """Give a key `count` analyses that are outstanding, as a real submission leaves them."""
    return [
        store_analysis(session, owner=key, status=ANALYSIS_STATUS_QUEUED)
        for _ in range(count)
    ]


def submit_media(client, plaintext: str, payload: bytes = b"limit-test-bytes"):
    return client.post(
        SUBMIT_URL,
        files={"file": ("clip.mp4", payload, "video/mp4")},
        headers=authorization(plaintext),
    )


@integration
def test_a_key_below_the_limit_is_accepted(reader, session, fake_minio):
    db, analyses, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES - 1)

    response = submit_media(reader, plaintext)

    assert response.status_code == 202
    analyses.append(uuid.UUID(response.json()["id"]))
    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES


@integration
def test_a_key_at_the_limit_is_throttled(reader, session, fake_minio):
    db, _, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES)

    response = submit_media(reader, plaintext)

    assert response.status_code == 429
    assert str(MAX_ACTIVE_ANALYSES) in response.json()["detail"]


@integration
def test_a_throttled_submission_persists_nothing(reader, session, fake_minio):
    """The refusal must leave the database exactly as it found it.

    A 429 that had already committed the analysis would be the worst of both answers: the
    caller told to back off, and the queue filled anyway.
    """
    db, _, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES)

    submit_media(reader, plaintext)

    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES
    assert (
        db.query(Analysis).filter(Analysis.api_key_id == key.id).count()
        == MAX_ACTIVE_ANALYSES
    )


@integration
def test_a_throttled_submission_leaves_the_session_usable(reader, session, fake_minio):
    """The refusal releases the key's lock instead of stranding the transaction.

    The `FOR UPDATE` taken to count is held by an open transaction until something ends it.
    If a 429 left it open, this key's next request — and every other request sharing the
    connection — would block on it. That the very next call answers at all is the check.
    """
    db, _, _ = session
    plaintext, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES)

    assert submit_media(reader, plaintext).status_code == 429

    # A read, then another submission: both would hang on a lock nobody released.
    assert reader.get("/api/v1/analyses").status_code == 200
    assert submit_media(reader, plaintext).status_code == 429


@integration
def test_the_limit_is_per_key(reader, session, fake_minio):
    """One customer at its ceiling must not throttle another.

    The lock is on the submitting key's own row precisely so that two keys never contend,
    and this is the behavioural half of that claim.
    """
    db, analyses, _ = session
    throttled_plaintext, throttled = issue_key(session, name="acme")
    free_plaintext, free = issue_key(session, name="globex")
    fill_active(session, throttled, MAX_ACTIVE_ANALYSES)

    assert submit_media(reader, throttled_plaintext).status_code == 429

    response = submit_media(reader, free_plaintext)

    assert response.status_code == 202
    analyses.append(uuid.UUID(response.json()["id"]))


@integration
def test_finished_analyses_do_not_count_towards_the_limit(reader, session, fake_minio):
    """Only outstanding work is counted. A key that has run a hundred analyses and has
    none in flight is as free as a key that has never run one.
    """
    db, analyses, _ = session
    plaintext, key = issue_key(session)

    for _ in range(MAX_ACTIVE_ANALYSES):
        store_analysis(session, owner=key, status=ANALYSIS_STATUS_COMPLETED)
    for _ in range(MAX_ACTIVE_ANALYSES):
        store_analysis(session, owner=key, status=ANALYSIS_STATUS_FAILED)

    assert active_analyses(db, key.id) == 0

    response = submit_media(reader, plaintext)

    assert response.status_code == 202
    analyses.append(uuid.UUID(response.json()["id"]))


@integration
def test_a_finishing_analysis_frees_a_slot(reader, session, fake_minio):
    """The count is derived from the analyses themselves, so nothing has to release a slot.

    A worker that finishes a job — or one that dies and has its job failed — frees capacity
    by the same act, and no counter can drift away from the truth.
    """
    db, analyses, _ = session
    plaintext, key = issue_key(session)
    active = fill_active(session, key, MAX_ACTIVE_ANALYSES)

    assert submit_media(reader, plaintext).status_code == 429

    active[0].status = ANALYSIS_STATUS_COMPLETED
    db.commit()

    response = submit_media(reader, plaintext)

    assert response.status_code == 202
    analyses.append(uuid.UUID(response.json()["id"]))


@integration
def test_the_dashboard_is_not_throttled(reader, session, fake_minio):
    """The internal route passes no limit, and its analyses are counted for no key.

    DeepGuard's own uploads are not a customer competing for the queue, and throttling them
    would be this task reaching into a route it was not asked to change. Still true now that
    they carry an owner: the ceiling counts analyses per *API key*, and these have none.
    """
    db, analyses, _ = session

    for _ in range(MAX_ACTIVE_ANALYSES + 3):
        response = reader.post(
            "/api/v1/analyses",
            files={"file": ("clip.mp4", b"dashboard-bytes", "video/mp4")},
            headers={"Origin": DASHBOARD_ORIGIN},
        )

        assert response.status_code == 202
        analyses.append(uuid.UUID(response.json()["id"]))


# --- the race ------------------------------------------------------------------------------


def submit_on_its_own_connection(db, key_id: uuid.UUID, results: dict) -> None:
    """One throttled submission through the real persistence transaction.

    `persist_analysis` is called directly rather than through the endpoint. It is the
    function that owns the transaction the limit lives in, and the HTTP layer above it adds
    a thread pool and an event loop between this test and the lock — which is exactly the
    timing these tests need to control.
    """
    try:
        analysis = persist_analysis(
            db,
            filename="race.mp4",
            content_type="video/mp4",
            stored=StoredUpload(
                path=Path("/nonexistent"),
                size_bytes=1024,
                sha256=hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest(),
            ),
            storage_key="originals/race",
            metadata=MediaMetadata(
                format_name="mov,mp4,m4a,3gp,3g2,mj2",
                major_brand="mp42",
                codec_name="h264",
                width=1920,
                height=1080,
                duration=12.34,
                frame_rate=30.0,
                pix_fmt="yuv420p",
                constant_frame_rate=True,
            ),
            was_normalized=False,
            derivative_storage_key="originals/race",
            api_key_id=key_id,
            max_active_analyses=MAX_ACTIVE_ANALYSES,
        )
        results["accepted"].append(analysis.id)
    except ActiveAnalysisLimitReached:
        results["refused"].append(1)
    except BaseException as error:  # pragma: no cover — reported, never swallowed
        results["errors"].append(error)


def concurrent_submission(key_id: uuid.UUID, barrier: threading.Barrier, results: dict):
    """A submission that waits at the barrier with its connection already open.

    The connection is established *before* the barrier on purpose. `SessionLocal()` does
    not connect until the first statement, so a thread that waits first and connects
    afterwards spends its first milliseconds opening a socket to PostgreSQL — and six
    threads doing that arrive at the count spread far enough apart to stop contending at
    all. Warming it here is what makes the barrier release six threads into the same
    moment rather than into six different ones.
    """
    with SessionLocal() as db:
        db.execute(text("SELECT 1"))

        barrier.wait(timeout=10)

        submit_on_its_own_connection(db, key_id, results)


@integration
def test_a_submission_waits_for_the_keys_lock_before_counting(session):
    """The deterministic half of the race, and the one that pins the mechanism down.

    A timing test can only ever say "these six happened to come out right this run". This
    one removes the timing: the key's row is locked here, by this test, and held. A
    submission for that key must then *block* — that it is still running after a second
    and a half is the assertion, and it is the assertion a count-then-insert cannot pass,
    because nothing would stop it reading four and admitting immediately.

    Then the last free slot is taken under that same lock and the lock released, so the
    waiting submission wakes into a database that filled up while it was queued. It has to
    count again and refuse. A `FOR UPDATE` that was taken and then not re-counted — the
    other easy mistake — fails here too, by admitting a sixth analysis.
    """
    db, analyses, _ = session
    _, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES - 1)

    results = {"accepted": [], "refused": [], "errors": []}
    holder = SessionLocal()
    contender = SessionLocal()
    contender.execute(text("SELECT 1"))
    thread = threading.Thread(
        target=submit_on_its_own_connection, args=(contender, key.id, results)
    )

    try:
        # This test now holds what `persist_analysis` must wait for.
        holder.execute(
            select(ApiKey.id).where(ApiKey.id == key.id).with_for_update()
        ).scalar_one()

        thread.start()

        thread.join(timeout=1.5)
        assert thread.is_alive(), (
            "the submission did not wait for the key's lock, so the count and the insert "
            "are not serialized"
        )
        assert results == {"accepted": [], "refused": [], "errors": []}

        # Fill the last slot from inside the locked transaction, then release it.
        latecomer = Analysis(status=ANALYSIS_STATUS_QUEUED, api_key_id=key.id)
        holder.add(latecomer)
        holder.commit()
        analyses.append(latecomer.id)
    finally:
        # Whatever happened above, the lock is dropped, the contender is let go and
        # anything it managed to create is handed to the fixture to remove. A failed
        # assertion must not leave a locked row or an undeletable key behind it.
        holder.close()
        thread.join(timeout=30)
        contender.close()
        analyses.extend(results["accepted"])

    assert not thread.is_alive()
    assert results["errors"] == []
    # It woke up, counted a full database, and refused — rather than inserting the sixth.
    assert results["accepted"] == []
    assert len(results["refused"]) == 1

    db.expire_all()
    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES


@integration
def test_concurrent_submissions_cannot_exceed_the_limit(session):
    """The property the whole design exists for: many at once, and still exactly the limit.

    One free slot and six threads racing for it. A plain count-then-insert passes every
    other test in this section and fails this one — all six would read four, all six would
    admit, and the key would end with ten analyses in flight.

    The threads are released together on a barrier so they contend for real, and each holds
    its own connection, because two threads sharing a session would be serialized by the
    session rather than by PostgreSQL and would prove nothing.
    """
    db, analyses, _ = session
    _, key = issue_key(session)
    fill_active(session, key, MAX_ACTIVE_ANALYSES - 1)

    racers = 6
    barrier = threading.Barrier(racers)
    results = {"accepted": [], "refused": [], "errors": []}

    threads = [
        threading.Thread(target=concurrent_submission, args=(key.id, barrier, results))
        for _ in range(racers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    analyses.extend(results["accepted"])

    assert results["errors"] == []
    assert len(results["accepted"]) == 1
    assert len(results["refused"]) == racers - 1

    db.expire_all()
    assert active_analyses(db, key.id) == MAX_ACTIVE_ANALYSES


@integration
def test_concurrent_submissions_from_different_keys_all_succeed(session):
    """Serializing one key must not serialize the service.

    The same race, but every thread carries its own key. All of them must be admitted: a
    lock taken on something shared — a table, an advisory lock on a constant — would still
    pass the test above and would quietly turn every customer's uploads into one queue.
    """
    db, analyses, _ = session
    keys = [issue_key(session, name=f"customer-{index}")[1] for index in range(6)]

    barrier = threading.Barrier(len(keys))
    results = {"accepted": [], "refused": [], "errors": []}

    threads = [
        threading.Thread(target=concurrent_submission, args=(key.id, barrier, results))
        for key in keys
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    analyses.extend(results["accepted"])

    assert results["errors"] == []
    assert len(results["accepted"]) == len(keys)
    assert results["refused"] == []
