"""User feedback: who may write it, what a resubmission does, and what it may never touch (R13-T1).

Written to the conventions of `test_ground_truth_api.py`, whose fixtures it reuses: real
PostgreSQL, real sessions through the real login route, every row removed afterwards.

*Ownership.* Only `Analysis.owner_id == session user` may write or read their feedback. Another
user, an administrator who is not the owner, and anybody at all on an analysis with no owner (a
legacy upload or an API-key submission) get the concealing 404. The admin route reads, never
writes.

*Upsert.* One row per (user, analysis), enforced by the database, concurrent submissions
included. An identical resubmission is a 200 that leaves `updated_at` alone.

*Isolation.* Submitting or revising feedback leaves the verdict, the analysis status, every
signal, the human review and Ground Truth exactly as they were, and writes no audit event.

*Evaluation independence.* The operational metrics and the operational export read the same
with feedback in the database as without it, and neither script names the table.
"""

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    USER_ROLE_ADMIN,
    AdminAuditEvent,
    ApiKey,
    GroundTruth,
    User,
    UserFeedback,
)
from app.db.session import SessionLocal
from app.main import app
from app.web_auth import SESSION_COOKIE_NAME
from tests.conftest import DASHBOARD_ORIGIN
from tests.test_export_operational_dataset import load_script, seeded  # noqa: F401
from tests.test_ground_truth_api import (  # noqa: F401
    database,
    forensic_state,
    make_analysis,
    make_user,
    new_hash,
    plain_http_environment,
    session,
    signed_in,
)
from tests.test_r12_invariants import assert_identical_exports, read_only_export

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]

FEEDBACK = {"assessment": "DISAGREE", "claimed_label": "GENUINE", "notes": "It is my own video."}


def feedback_url(analysis_id) -> str:
    return f"/api/v1/analyses/{analysis_id}/feedback"


def admin_feedback_url(analysis_id) -> str:
    return f"/api/v1/admin/analyses/{analysis_id}/feedback"


def post_feedback(client, analysis_id, body: dict):
    return client.post(feedback_url(analysis_id), json=body, headers={"Origin": DASHBOARD_ORIGIN})


def owned_analysis(session, owner: User):
    """A completed analysis with a verdict, signals, a review and a media hash, owned by `owner`."""
    db = session[0]
    analysis = make_analysis(session, new_hash(session))
    analysis.owner_id = owner.id
    db.commit()
    return analysis


def stored_rows(session, analysis_id) -> list[UserFeedback]:
    db = session[0]
    db.commit()
    db.expire_all()
    return list(
        db.execute(select(UserFeedback).where(UserFeedback.analysis_id == analysis_id))
        .scalars()
        .all()
    )


def ground_truth_state(session, sha256: str) -> tuple | None:
    db = session[0]
    db.commit()
    db.expire_all()
    row = db.execute(
        select(
            GroundTruth.source_class,
            GroundTruth.label,
            GroundTruth.notes,
            GroundTruth.actor_id,
            GroundTruth.updated_at,
        ).where(GroundTruth.media_sha256 == sha256)
    ).one_or_none()
    return None if row is None else tuple(row)


def audit_count(session, analysis_id) -> int:
    db = session[0]
    db.commit()
    return len(
        db.execute(
            select(AdminAuditEvent.id).where(AdminAuditEvent.target_id == str(analysis_id))
        ).all()
    )


# --- ownership ------------------------------------------------------------------------------


def test_the_owner_submits_and_reads_their_feedback(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    client = signed_in(owner)

    empty = client.get(feedback_url(analysis.id))
    assert empty.status_code == 200
    assert empty.json()["assessment"] is None and empty.json()["created_at"] is None

    response = post_feedback(client, analysis.id, FEEDBACK)
    assert response.status_code == 200, response.text
    body = response.json()
    assert {k: body[k] for k in FEEDBACK} == FEEDBACK
    assert body["analysis_id"] == str(analysis.id)

    assert client.get(feedback_url(analysis.id)).json() == body
    [row] = stored_rows(session, analysis.id)
    assert row.user_id == owner.id


def test_another_user_can_neither_write_nor_read(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    stranger = signed_in(make_user(session))

    assert post_feedback(stranger, analysis.id, FEEDBACK).status_code == 404
    assert stranger.get(feedback_url(analysis.id)).status_code == 404
    assert stored_rows(session, analysis.id) == []


def test_an_administrator_who_is_not_the_owner_cannot_write_user_feedback(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    admin = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = post_feedback(admin, analysis.id, FEEDBACK)

    assert response.status_code == 404
    assert response.json() == {"detail": "analysis not found"}
    assert admin.get(feedback_url(analysis.id)).status_code == 404
    assert stored_rows(session, analysis.id) == []


def test_an_analysis_with_no_owner_takes_no_feedback_from_anybody(session):
    """Legacy uploads carry no owner at all; neither a user nor an administrator reaches them."""
    analysis = make_analysis(session, new_hash(session))
    assert analysis.owner_id is None and analysis.api_key_id is None

    for client in (
        signed_in(make_user(session)),
        signed_in(make_user(session, role=USER_ROLE_ADMIN)),
    ):
        assert post_feedback(client, analysis.id, FEEDBACK).status_code == 404
        assert client.get(feedback_url(analysis.id)).status_code == 404

    assert stored_rows(session, analysis.id) == []


def test_an_api_key_analysis_takes_no_feedback_from_anybody(session):
    db = session[0]
    key = ApiKey(name="r13-t1", key_hash=hashlib.sha256(uuid.uuid4().bytes).hexdigest())
    db.add(key)
    db.commit()
    analysis = make_analysis(session, new_hash(session))
    analysis.api_key_id = key.id
    db.commit()

    try:
        for client in (
            signed_in(make_user(session)),
            signed_in(make_user(session, role=USER_ROLE_ADMIN)),
        ):
            assert post_feedback(client, analysis.id, FEEDBACK).status_code == 404
            assert client.get(feedback_url(analysis.id)).status_code == 404
        assert stored_rows(session, analysis.id) == []
    finally:
        db.rollback()
        analysis.api_key_id = None
        db.commit()
        db.delete(key)
        db.commit()


def test_an_administrator_who_owns_the_analysis_may_give_feedback(session):
    """The rule is ownership, not role: an administrator's own upload is theirs to comment on."""
    admin_user = make_user(session, role=USER_ROLE_ADMIN)
    analysis = owned_analysis(session, admin_user)

    assert post_feedback(signed_in(admin_user), analysis.id, FEEDBACK).status_code == 200


def test_anonymous_and_cross_origin_submissions_are_refused(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    client = signed_in(owner)

    anonymous = TestClient(app)
    assert post_feedback(anonymous, analysis.id, FEEDBACK).status_code == 401
    assert anonymous.get(feedback_url(analysis.id)).status_code == 401
    assert client.post(feedback_url(analysis.id), json=FEEDBACK).status_code == 403
    assert (
        client.post(
            feedback_url(analysis.id), json=FEEDBACK, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert stored_rows(session, analysis.id) == []


def test_an_unknown_analysis_is_a_404(session):
    client = signed_in(make_user(session))
    assert post_feedback(client, uuid.uuid4(), FEEDBACK).status_code == 404


# --- the request contract -------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"assessment": "TRUE"},
        {"assessment": "agree"},
        {"assessment": "AGREE", "claimed_label": "FAKE"},
        {"assessment": "AGREE", "claimed_label": "AUDIO_MANIPULATION"},
        {"assessment": "AGREE", "risk_level": "HIGH"},
        {"assessment": "AGREE", "user_id": str(uuid.uuid4())},
        {"assessment": "AGREE", "notes": "x" * 1001},
        {"assessment": "AGREE", "notes": "bad\x00byte"},
        {"claimed_label": "GENUINE"},
    ],
    ids=[
        "unknown-assessment",
        "lowercase",
        "unknown-label",
        "ground-truth-only-label",
        "forensic-field",
        "identity-field",
        "too-long",
        "control-character",
        "no-assessment",
    ],
)
def test_a_body_outside_the_contract_is_a_422(session, body):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)

    assert post_feedback(signed_in(owner), analysis.id, body).status_code == 422
    assert stored_rows(session, analysis.id) == []


def test_blank_notes_are_stored_as_none(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)

    response = post_feedback(signed_in(owner), analysis.id, {"assessment": "UNSURE", "notes": "  \n"})

    assert response.status_code == 200
    assert response.json()["notes"] is None
    assert response.json()["claimed_label"] is None


# --- upsert and no-op -----------------------------------------------------------------------


def test_an_identical_resubmission_is_a_no_op(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    client = signed_in(owner)
    first = post_feedback(client, analysis.id, FEEDBACK).json()

    # Trailing whitespace normalizes away, so this is the same statement.
    again = post_feedback(client, analysis.id, {**FEEDBACK, "notes": FEEDBACK["notes"] + "  "})

    assert again.status_code == 200
    assert again.json() == first
    [row] = stored_rows(session, analysis.id)
    assert row.updated_at == datetime.fromisoformat(first["updated_at"].replace("Z", "+00:00"))
    assert row.updated_at == row.created_at


def test_a_changed_submission_updates_the_one_row(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    client = signed_in(owner)
    first = post_feedback(client, analysis.id, FEEDBACK).json()

    revised = post_feedback(client, analysis.id, {"assessment": "AGREE"}).json()

    assert revised["assessment"] == "AGREE"
    # A PUT-like statement: omitted optional fields are cleared, not kept.
    assert revised["claimed_label"] is None and revised["notes"] is None
    assert revised["created_at"] == first["created_at"]
    assert revised["updated_at"] > first["updated_at"]
    assert len(stored_rows(session, analysis.id)) == 1


def test_concurrent_submissions_leave_exactly_one_row(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    # One session shared by six clients: signing in again would revoke the previous session.
    cookie = signed_in(owner).cookies.get(SESSION_COOKIE_NAME)
    clients = [TestClient(app, cookies={SESSION_COOKIE_NAME: cookie}) for _ in range(6)]
    bodies = [{"assessment": a} for a in ("AGREE", "DISAGREE", "UNSURE") * 2]

    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        statuses = list(
            pool.map(lambda pair: post_feedback(pair[0], analysis.id, pair[1]).status_code,
                     zip(clients, bodies))
        )

    assert statuses == [200] * len(clients)
    assert len(stored_rows(session, analysis.id)) == 1


# --- the database invariants ----------------------------------------------------------------


def test_the_database_refuses_a_second_row_for_the_same_user_and_analysis(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    db = session[0]

    db.add(UserFeedback(analysis_id=analysis.id, user_id=owner.id, assessment="AGREE"))
    db.commit()
    db.add(UserFeedback(analysis_id=analysis.id, user_id=owner.id, assessment="UNSURE"))
    with pytest.raises(IntegrityError, match="uq_user_feedback_user_analysis"):
        db.commit()
    db.rollback()


@pytest.mark.parametrize(
    ("column", "value", "constraint"),
    [
        ("assessment", "CONFIRMED", "ck_user_feedback_assessment"),
        ("claimed_label", "FAKE", "ck_user_feedback_claimed_label"),
    ],
)
def test_the_database_refuses_values_outside_the_vocabulary(session, column, value, constraint):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    db = session[0]

    values = {"assessment": "AGREE", column: value}
    db.add(UserFeedback(analysis_id=analysis.id, user_id=owner.id, **values))
    with pytest.raises(IntegrityError, match=constraint):
        db.commit()
    db.rollback()


def test_deleting_the_analysis_deletes_its_feedback(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    assert post_feedback(signed_in(owner), analysis.id, FEEDBACK).status_code == 200
    db = session[0]

    db.execute(text("DELETE FROM analysis_signals WHERE analysis_id = :id"), {"id": analysis.id})
    db.execute(text("DELETE FROM analysis_reviews WHERE analysis_id = :id"), {"id": analysis.id})
    db.execute(text("DELETE FROM media_files WHERE analysis_id = :id"), {"id": analysis.id})
    db.execute(text("DELETE FROM analyses WHERE id = :id"), {"id": analysis.id})
    db.commit()

    assert stored_rows(session, analysis.id) == []


def test_a_user_with_feedback_cannot_be_deleted(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    db = session[0]
    # A second account's row, written directly, so `analyses.owner_id` is not what blocks it.
    other = make_user(session)
    db.add(UserFeedback(analysis_id=analysis.id, user_id=other.id, assessment="UNSURE"))
    db.commit()

    with pytest.raises(IntegrityError, match="user_feedback_user_id_fkey"):
        db.execute(text("DELETE FROM users WHERE id = :id"), {"id": other.id})
        db.commit()
    db.rollback()
    db.query(UserFeedback).filter(UserFeedback.user_id == other.id).delete()
    db.commit()


# --- the admin read -------------------------------------------------------------------------


def test_the_admin_reads_every_users_feedback_and_cannot_write(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)
    assert post_feedback(signed_in(owner), analysis.id, FEEDBACK).status_code == 200
    admin = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    response = admin.get(admin_feedback_url(analysis.id))

    assert response.status_code == 200
    [entry] = response.json()
    assert entry["user_id"] == str(owner.id)
    assert entry["user_email"] == owner.email
    assert {k: entry[k] for k in FEEDBACK} == FEEDBACK

    for method in ("post", "put", "patch", "delete"):
        refused = getattr(admin, method)(
            admin_feedback_url(analysis.id), headers={"Origin": DASHBOARD_ORIGIN}
        )
        assert refused.status_code == 405, method
    assert len(stored_rows(session, analysis.id)) == 1


def test_the_admin_read_is_admin_only(session):
    owner = make_user(session)
    analysis = owned_analysis(session, owner)

    assert TestClient(app).get(admin_feedback_url(analysis.id)).status_code == 401
    assert signed_in(owner).get(admin_feedback_url(analysis.id)).status_code == 403


def test_the_admin_read_of_an_analysis_without_feedback_is_empty_and_unknown_is_404(session):
    analysis = make_analysis(session, new_hash(session))
    admin = signed_in(make_user(session, role=USER_ROLE_ADMIN))

    assert admin.get(admin_feedback_url(analysis.id)).json() == []
    assert admin.get(admin_feedback_url(uuid.uuid4())).status_code == 404


# --- isolation ------------------------------------------------------------------------------


def test_feedback_leaves_verdict_status_signals_review_and_ground_truth_unchanged(session):
    owner = make_user(session)
    sha256 = new_hash(session)
    analysis = make_analysis(session, sha256)
    db = session[0]
    analysis.owner_id = owner.id
    db.add(
        GroundTruth(
            media_sha256=sha256, source_class="CONTROLLED_TEST", label="AI_GENERATED",
            actor_id=uuid.uuid4(),
        )
    )
    db.commit()
    client = signed_in(owner)

    before = forensic_state(session, analysis)
    truth_before = ground_truth_state(session, sha256)
    audits_before = audit_count(session, analysis.id)

    # A claim that contradicts Ground Truth, then a revision, then a no-op.
    for body in (FEEDBACK, {"assessment": "AGREE", "claimed_label": "OTHER"}, {"assessment": "AGREE", "claimed_label": "OTHER"}):
        assert post_feedback(client, analysis.id, body).status_code == 200

        assert forensic_state(session, analysis) == before
        assert ground_truth_state(session, sha256) == truth_before
        assert audit_count(session, analysis.id) == audits_before


def test_a_genuine_claim_does_not_create_ground_truth(session):
    owner = make_user(session)
    sha256 = new_hash(session)
    analysis = make_analysis(session, sha256)
    analysis.owner_id = owner.id
    session[0].commit()

    assert post_feedback(signed_in(owner), analysis.id, FEEDBACK).status_code == 200

    assert ground_truth_state(session, sha256) is None


# --- evaluation independence ----------------------------------------------------------------


EVALUATOR_SCRIPTS = ("operational_metrics.py", "export_operational_dataset.py", "replay.py")


@pytest.mark.parametrize("name", EVALUATOR_SCRIPTS)
def test_no_evaluator_names_user_feedback(name):
    source = (REPO_ROOT / "scripts" / "eval" / name).read_text()
    assert "UserFeedback" not in source
    assert "user_feedback" not in source


@pytest.fixture
def feedback_on_seeded(seeded):  # noqa: F811
    """Contradicting feedback on every seeded analysis, removed before `seeded` tears down."""
    from app.db.models import Analysis, MediaFile
    from app.web_auth import hash_password

    with SessionLocal() as db:
        user = User(email=f"{uuid.uuid4().hex}@example.com", password_hash=hash_password("x" * 12))
        db.add(user)
        db.commit()
        analysis_ids = db.execute(
            select(Analysis.id)
            .join(MediaFile, MediaFile.analysis_id == Analysis.id)
            .where(MediaFile.original_sha256.in_(list(seeded.values())))
        ).scalars().all()

        def add():
            for analysis_id in analysis_ids:
                db.add(UserFeedback(
                    analysis_id=analysis_id, user_id=user.id, assessment="DISAGREE",
                    claimed_label="FACE_SWAP", notes="Contradicts Ground Truth on purpose.",
                ))
            db.commit()

        yield add

        db.rollback()
        db.query(UserFeedback).filter(UserFeedback.user_id == user.id).delete()
        db.query(User).filter(User.id == user.id).delete()
        db.commit()


def test_feedback_does_not_move_operational_metrics(seeded, feedback_on_seeded):  # noqa: F811
    om = load_script("operational_metrics")

    def measure():
        with SessionLocal() as db:
            db.execute(text("SET TRANSACTION READ ONLY"))
            rows, facts = om.load_rows(db)
            db.rollback()
        return om.evaluate(rows), facts

    before = measure()
    feedback_on_seeded()
    after = measure()

    assert after == before
    assert any(item["media_sha256"] == seeded["swap"] for item in before[0]["snapshot"])


def test_feedback_does_not_move_the_operational_export(seeded, feedback_on_seeded, tmp_path):  # noqa: F811
    exporter = load_script("export_operational_dataset")

    before = read_only_export(exporter, tmp_path / "before")
    feedback_on_seeded()
    after = read_only_export(exporter, tmp_path / "after")

    assert seeded["swap"] in (tmp_path / "before" / "corpus.json").read_text()
    assert_identical_exports(tmp_path / "before", before, tmp_path / "after", after)
