"""What the Ground Truth routes persist, what they audit, and what they may never touch (R12-T2).

Written to the conventions of `test_admin_analyses.py`: real PostgreSQL, real sessions opened
through the real login route, and every row this file creates removed afterwards.

The claims divide into five groups.

*Identity.* Ground Truth is keyed by the bytes' SHA-256. Only the canonical spelling — 64
lowercase hex characters — is accepted, so one file cannot hold two records under two spellings
of its hash; and a hash no media row carries is refused, so nothing is labelled that this system
has never seen.

*The contract.* The body is `GroundTruthContract`. An invalid `source_class` or `label` is a 422,
and so is a body naming `manipulation_family` — the family is derived, never supplied, and never
stored.

*The audit trail.* One event per change that stuck, carrying `old` and `new` for all three
fields whether they moved or not, `old: null` for each on creation, the full 64-character hash as
`target_id`, and the writing administrator — exactly — as the actor.

*Separation.* Recording Ground Truth leaves the verdict, the signals, the human review and the
provenance state of every analysis of those bytes exactly as they were.

*Access.* Both routes refuse an anonymous caller with 401 and a non-administrator with 403.

Counting convention, inherited from `test_admin_audit.py`: the audit table is deployment-wide, so
events are only ever counted for a hash this file generated, which is unique per test.
"""

import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    AUDIT_ACTION_GROUND_TRUTH_CREATED,
    AUDIT_ACTION_GROUND_TRUTH_UPDATED,
    AUDIT_TARGET_GROUND_TRUTH,
    REVIEW_STATUS_NEEDS_FOLLOW_UP,
    SIGNAL_STATUS_SUCCESS,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    Analysis,
    AnalysisReview,
    AnalysisSignal,
    AuthSession,
    GroundTruth,
    MediaFile,
    User,
)
from app.db.session import SessionLocal, engine
from app.detection import PROVENANCE_SIGNAL
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

GROUND_TRUTH_URL = "/api/v1/admin/ground-truth"

# A test fixture, not a credential.
PASSWORD = "correct-horse-battery-staple"

FORBIDDEN_BODY = {"detail": "Insufficient permissions"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}

# Real values on every forensic column, so the separation assertions compare something.
RISK_LEVEL = "HIGH"
RULES_VERSION = "r7-v4.0.0"
CALIBRATION_ID = "c" * 64
RULE_ID = "R4"

# The whole response, written out so widening it is a deliberate edit here.
VISIBLE_FIELDS = {
    "media_sha256",
    "source_class",
    "label",
    "manipulation_family",
    "notes",
    "actor_id",
    "actor_email_snapshot",
    "created_at",
    "updated_at",
}

STATEMENT = {"source_class": "CONTROLLED_TEST", "label": "AI_GENERATED", "notes": "Made here."}


@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    """`TestClient` speaks plain HTTP, and a `Secure` cookie would be dropped over it."""
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "development")


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
    """A real session that removes every row this file created.

    Audit events and Ground Truth are removed by the hashes this file generated; media rows,
    signals and reviews go before their analyses.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []
    hashes: list[str] = []

    with SessionLocal() as db:
        yield db, users, analyses, hashes

        db.rollback()
        for sha256 in hashes:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.target_id == sha256
            ).delete(synchronize_session=False)
            db.query(GroundTruth).filter(GroundTruth.media_sha256 == sha256).delete(
                synchronize_session=False
            )
        for analysis_id in analyses:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.target_id == str(analysis_id)
            ).delete(synchronize_session=False)
            db.query(AnalysisReview).filter(
                AnalysisReview.analysis_id == analysis_id
            ).delete(synchronize_session=False)
            db.query(AnalysisSignal).filter(
                AnalysisSignal.analysis_id == analysis_id
            ).delete(synchronize_session=False)
            db.query(MediaFile).filter(MediaFile.analysis_id == analysis_id).delete(
                synchronize_session=False
            )
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.actor_id == user_id
            ).delete(synchronize_session=False)
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


def make_user(session, *, role: str = USER_ROLE_USER) -> User:
    db, users, _, _ = session

    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def new_hash(session) -> str:
    """A canonical hash unique to this test, registered for cleanup."""
    _, _, _, hashes = session
    sha256 = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    hashes.append(sha256)

    return sha256


def make_analysis(session, sha256: str) -> Analysis:
    """A completed analysis of these bytes, with a risk decision, a detector signal, a
    provenance signal and a media row carrying the hash."""
    db, _, analyses, _ = session

    analysis = Analysis(
        status=ANALYSIS_STATUS_COMPLETED,
        risk_level=RISK_LEVEL,
        risk_rules_version=RULES_VERSION,
        risk_calibration_id=CALIBRATION_ID,
        risk_rule_id=RULE_ID,
    )
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    db.add(
        MediaFile(
            analysis_id=analysis.id,
            original_filename="clip.mov",
            content_type="video/quicktime",
            size_bytes=4096,
            original_sha256=sha256,
            original_storage_key=f"originals/{sha256}",
            format_name="mov,mp4,m4a,3gp,3g2,mj2",
            codec_name="h264",
            width=1920,
            height=1080,
            duration=12.34,
            frame_rate=30000 / 1001,
            pix_fmt="yuv420p",
            constant_frame_rate=True,
            was_normalized=False,
            was_assembled=False,
            acquisition_method="upload",
        )
    )
    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider="fixture",
            signal_type="synthetic_video",
            status=SIGNAL_STATUS_SUCCESS,
            score=0.97,
        )
    )
    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider="c2pa",
            signal_type=PROVENANCE_SIGNAL,
            status=SIGNAL_STATUS_SUCCESS,
            signal_metadata={"manifest_exists": True, "validation_state": "Valid"},
        )
    )
    db.add(
        AnalysisReview(
            analysis_id=analysis.id,
            status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
            note="Checked by hand.",
            reviewer_id=uuid.uuid4(),
            reviewer_email_snapshot="reviewer@example.com",
        )
    )
    db.commit()

    return analysis


def signed_in(user: User) -> TestClient:
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


def put_ground_truth(client: TestClient, sha256: str, body: dict):
    return client.put(
        f"{GROUND_TRUTH_URL}/{sha256}", json=body, headers={"Origin": DASHBOARD_ORIGIN}
    )


def get_ground_truth(client: TestClient, sha256: str):
    return client.get(f"{GROUND_TRUTH_URL}/{sha256}")


def events_for(session, sha256: str) -> list[AdminAuditEvent]:
    """Every event about these bytes, oldest first."""
    db, _, _, _ = session
    db.commit()

    return list(
        db.execute(
            select(AdminAuditEvent)
            .where(AdminAuditEvent.target_id == sha256)
            .order_by(AdminAuditEvent.created_at, AdminAuditEvent.id)
        )
        .scalars()
        .all()
    )


def stored_count(session, sha256: str) -> int:
    db, _, _, _ = session
    db.commit()

    return len(
        db.execute(select(GroundTruth).where(GroundTruth.media_sha256 == sha256))
        .scalars()
        .all()
    )


def forensic_state(session, analysis: Analysis) -> tuple:
    """Everything about an analysis Ground Truth must never change: the verdict columns, every
    signal row with its score and metadata (provenance included), and the human review."""
    db, _, _, _ = session
    db.commit()
    db.expire_all()

    verdict = db.execute(
        select(
            Analysis.status,
            Analysis.risk_level,
            Analysis.risk_rules_version,
            Analysis.risk_calibration_id,
            Analysis.risk_rule_id,
        ).where(Analysis.id == analysis.id)
    ).one()

    signals = db.execute(
        select(
            AnalysisSignal.signal_type,
            AnalysisSignal.status,
            AnalysisSignal.score,
            AnalysisSignal.signal_metadata,
        )
        .where(AnalysisSignal.analysis_id == analysis.id)
        .order_by(AnalysisSignal.signal_type)
    ).all()

    review = db.execute(
        select(
            AnalysisReview.status,
            AnalysisReview.analyst_assessment,
            AnalysisReview.note,
            AnalysisReview.reviewer_id,
            AnalysisReview.updated_at,
        ).where(AnalysisReview.analysis_id == analysis.id)
    ).one()

    return (tuple(verdict), [tuple(signal) for signal in signals], tuple(review))


# --- identity: canonical hash, no orphans -----------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    [
        lambda h: h.upper(),
        lambda h: h[:63],
        lambda h: h + "0",
        lambda h: h[:63] + "g",
        lambda h: "sha256:" + h,
    ],
    ids=["uppercase", "short", "long", "non-hex", "prefixed"],
)
def test_a_non_canonical_hash_is_refused_on_both_routes(session, spelling):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    assert put_ground_truth(client, spelling(sha256), STATEMENT).status_code == 422
    assert get_ground_truth(client, spelling(sha256)).status_code == 422
    assert stored_count(session, sha256) == 0
    assert events_for(session, sha256) == []


def test_ground_truth_for_bytes_nobody_uploaded_is_refused(session):
    sha256 = new_hash(session)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = put_ground_truth(client, sha256, STATEMENT)

    assert response.status_code == 404
    assert response.json() == {"detail": "media not found"}
    assert stored_count(session, sha256) == 0
    assert events_for(session, sha256) == []
    assert get_ground_truth(client, sha256).json() == {"detail": "media not found"}


def test_known_bytes_with_no_record_are_not_reported_as_unknown(session):
    """No record is a 404 of its own, never a 200 claiming `UNKNOWN` on nobody's behalf."""
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = get_ground_truth(client, sha256)

    assert response.status_code == 404
    assert response.json() == {"detail": "ground truth not recorded"}


def test_one_record_serves_every_upload_of_the_same_bytes(session):
    """Keyed by the hash: two analyses of the same bytes share one record."""
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    assert put_ground_truth(client, sha256, STATEMENT).status_code == 200
    assert stored_count(session, sha256) == 1


def test_the_table_has_no_family_column_and_no_reference_to_media(database):
    inspector = inspect(database)

    columns = {column["name"] for column in inspector.get_columns("ground_truth")}

    assert columns == {
        "media_sha256",
        "source_class",
        "label",
        "notes",
        "actor_id",
        "actor_email_snapshot",
        "created_at",
        "updated_at",
    }
    assert inspector.get_pk_constraint("ground_truth")["constrained_columns"] == [
        "media_sha256"
    ]
    assert inspector.get_foreign_keys("ground_truth") == []


# --- persistence --------------------------------------------------------------------------


def test_put_then_get_returns_the_recorded_statement(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)

    written = put_ground_truth(client, sha256, STATEMENT)
    read = get_ground_truth(client, sha256)

    assert written.status_code == 200
    assert read.status_code == 200
    assert set(read.json()) == VISIBLE_FIELDS
    assert read.json() == written.json()

    body = read.json()
    assert body["media_sha256"] == sha256
    assert body["source_class"] == "CONTROLLED_TEST"
    assert body["label"] == "AI_GENERATED"
    assert body["manipulation_family"] == "GENERATED_VIDEO"
    assert body["notes"] == "Made here."
    assert body["actor_id"] == str(admin.id)
    assert body["actor_email_snapshot"] == admin.email


def test_a_revision_overwrites_and_rederives_the_family(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    put_ground_truth(client, sha256, STATEMENT)
    response = put_ground_truth(
        client, sha256, {"source_class": "OWNER_KNOWN", "label": "GENUINE"}
    )

    assert response.status_code == 200
    body = get_ground_truth(client, sha256).json()
    assert body["source_class"] == "OWNER_KNOWN"
    assert body["label"] == "GENUINE"
    assert body["manipulation_family"] == "NONE"
    # A PUT is the whole statement: omitting notes clears them.
    assert body["notes"] is None
    assert stored_count(session, sha256) == 1


# --- the contract -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"source_class": "DETECTOR", "label": "GENUINE"},
        {"source_class": "owner_known", "label": "GENUINE"},
        {"source_class": "OWNER_KNOWN", "label": "FAKE"},
        {"source_class": "OWNER_KNOWN", "label": "HIGH"},
        {"source_class": "OWNER_KNOWN"},
        {"label": "GENUINE"},
    ],
)
def test_an_invalid_taxonomy_is_refused(session, body):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    assert put_ground_truth(client, sha256, body).status_code == 422
    assert stored_count(session, sha256) == 0
    assert events_for(session, sha256) == []


@pytest.mark.parametrize("family", ["GENERATED_VIDEO", "NONE", None])
def test_a_body_naming_the_family_is_refused(session, family):
    """Even a family that agrees with the label: it is derived, never supplied."""
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = put_ground_truth(
        client, sha256, {**STATEMENT, "manipulation_family": family}
    )

    assert response.status_code == 422
    assert stored_count(session, sha256) == 0


@pytest.mark.parametrize("field", ["actor_id", "risk_level", "media_sha256"])
def test_a_body_naming_anything_else_is_refused(session, field):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = put_ground_truth(client, sha256, {**STATEMENT, field: "x"})

    assert response.status_code == 422
    assert stored_count(session, sha256) == 0


def test_notes_carrying_a_nul_byte_are_refused_not_a_500(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = put_ground_truth(client, sha256, {**STATEMENT, "notes": "a\x00b"})

    assert response.status_code == 422
    assert stored_count(session, sha256) == 0


# --- the audit trail ----------------------------------------------------------------------


def test_creation_audits_all_three_fields_with_null_old_values(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)

    put_ground_truth(client, sha256, STATEMENT)

    [event] = events_for(session, sha256)
    assert event.action == AUDIT_ACTION_GROUND_TRUTH_CREATED
    assert event.target_type == AUDIT_TARGET_GROUND_TRUTH
    assert event.target_id == sha256
    assert len(event.target_id) == 64
    assert event.target_email_snapshot is None
    assert event.changes == {
        "source_class": {"old": None, "new": "CONTROLLED_TEST"},
        "label": {"old": None, "new": "AI_GENERATED"},
        "notes": {"old": None, "new": "Made here."},
    }


def test_an_update_audits_all_three_fields_including_unchanged_ones(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    put_ground_truth(client, sha256, STATEMENT)
    # Only the label moves; source_class and notes must still appear, old == new.
    put_ground_truth(client, sha256, {**STATEMENT, "label": "FACE_SWAP"})

    created, updated = events_for(session, sha256)
    assert created.action == AUDIT_ACTION_GROUND_TRUTH_CREATED
    assert updated.action == AUDIT_ACTION_GROUND_TRUTH_UPDATED
    assert updated.target_id == sha256
    assert updated.changes == {
        "source_class": {"old": "CONTROLLED_TEST", "new": "CONTROLLED_TEST"},
        "label": {"old": "AI_GENERATED", "new": "FACE_SWAP"},
        "notes": {"old": "Made here.", "new": "Made here."},
    }


def test_the_audit_actor_is_exactly_the_administrator_who_wrote_it(session):
    """Two administrators in turn: each event names its own writer, and the record names the
    last."""
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    first = make_user(session, role=USER_ROLE_ADMIN)
    second = make_user(session, role=USER_ROLE_ADMIN)

    put_ground_truth(signed_in(first), sha256, STATEMENT)
    put_ground_truth(signed_in(second), sha256, {**STATEMENT, "notes": "Revised."})

    created, updated = events_for(session, sha256)
    assert (created.actor_id, created.actor_email_snapshot) == (first.id, first.email)
    assert (updated.actor_id, updated.actor_email_snapshot) == (second.id, second.email)

    db, _, _, _ = session
    db.expire_all()
    record = db.get(GroundTruth, sha256)
    assert (record.actor_id, record.actor_email_snapshot) == (second.id, second.email)


def test_a_request_that_changes_nothing_writes_nothing(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    first = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(first)

    put_ground_truth(client, sha256, STATEMENT)
    response = put_ground_truth(
        signed_in(make_user(session, role=USER_ROLE_ADMIN)), sha256, STATEMENT
    )

    assert response.status_code == 200
    assert len(events_for(session, sha256)) == 1
    # The record still names who actually wrote it.
    assert response.json()["actor_id"] == str(first.id)


def test_a_refused_request_writes_no_event(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    put_ground_truth(client, sha256, {**STATEMENT, "label": "FAKE"})

    assert events_for(session, sha256) == []


# --- separation ---------------------------------------------------------------------------


def test_recording_ground_truth_leaves_every_other_record_untouched(session):
    """Verdict, signals (provenance included) and the human review, before and after both a
    creation and a revision, for every analysis of the bytes."""
    sha256 = new_hash(session)
    analyses = [make_analysis(session, sha256), make_analysis(session, sha256)]
    before = [forensic_state(session, analysis) for analysis in analyses]
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    assert put_ground_truth(client, sha256, STATEMENT).status_code == 200
    assert [forensic_state(session, analysis) for analysis in analyses] == before

    assert (
        put_ground_truth(
            client, sha256, {"source_class": "OWNER_KNOWN", "label": "GENUINE"}
        ).status_code
        == 200
    )
    assert [forensic_state(session, analysis) for analysis in analyses] == before


def test_recording_ground_truth_writes_no_review_or_analysis_event(session):
    """The only audit rows are about the bytes; none names an analysis."""
    sha256 = new_hash(session)
    analysis = make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    put_ground_truth(client, sha256, STATEMENT)

    db, _, _, _ = session
    db.commit()
    assert (
        db.execute(
            select(AdminAuditEvent).where(AdminAuditEvent.target_id == str(analysis.id))
        )
        .scalars()
        .all()
        == []
    )


# --- access -------------------------------------------------------------------------------


def test_an_anonymous_caller_is_refused(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = TestClient(app)

    get = get_ground_truth(client, sha256)
    put = put_ground_truth(client, sha256, STATEMENT)

    assert (get.status_code, get.json()) == (401, UNAUTHENTICATED_BODY)
    assert (put.status_code, put.json()) == (401, UNAUTHENTICATED_BODY)
    assert stored_count(session, sha256) == 0


def test_a_non_administrator_is_refused(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    admin_client = signed_in(make_user(session, role=USER_ROLE_ADMIN))
    put_ground_truth(admin_client, sha256, STATEMENT)
    client = signed_in(make_user(session, role=USER_ROLE_USER))

    get = get_ground_truth(client, sha256)
    put = put_ground_truth(client, sha256, {**STATEMENT, "label": "GENUINE"})

    assert (get.status_code, get.json()) == (403, FORBIDDEN_BODY)
    assert (put.status_code, put.json()) == (403, FORBIDDEN_BODY)
    assert len(events_for(session, sha256)) == 1


def test_a_put_without_the_dashboard_origin_is_refused(session):
    sha256 = new_hash(session)
    make_analysis(session, sha256)
    client = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = client.put(f"{GROUND_TRUTH_URL}/{sha256}", json=STATEMENT)

    assert response.status_code == 403
    assert stored_count(session, sha256) == 0
