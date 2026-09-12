"""What a human review may say about an analysis, and what it may never touch.

The sibling of `test_admin_audit.py` and written to its conventions: real PostgreSQL, real
sessions opened through the real login route, one client per account, and every row this file
creates removed in constraint order afterwards. A role check demonstrated against a stubbed
session proves only that the stub agreed.

The claims here divide into four groups, and the first is the reason the feature exists.

*Isolation.* A review must not be able to reach the forensic record. This is asserted the only
way it can be asserted from outside: by reading every risk column and every signal row before
a review is written and again after, and requiring them to be identical — and by sending a
request that names `risk_level` outright and requiring a 422 rather than a silently dropped
field. A review of an analysis leaves the analysis byte-for-byte as it was.

*The taxonomy.* Only `REVIEWED` and `NEEDS_FOLLOW_UP` are storable. `FAKE`, `GENUINE` and the
rest are refused by name in a test, because the thing being prevented is not a typo — it is a
second, unversioned classification of the media sitting beside the risk engine's.

*The audit trail.* One event per change that stuck, zero for a change that did not, and never
the note's text in the payload. The last is asserted by putting a distinctive string in a note
and requiring it to appear nowhere in the serialized event.

*Durability of the row.* `reviewer_id` is not a foreign key, so deleting the reviewer must
leave the review standing and readable by its snapshot. `analysis_id` is one, with `CASCADE`,
so deleting the analysis must take the review with it. Both are asserted by actually deleting.

Counting convention, inherited from `test_admin_audit.py`: the audit table is deployment-wide
and other tests write to it, so nothing here asserts an absolute row count. Events are always
counted for a specific analysis id that this file created, which is unique per test.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    AUDIT_ACTION_REVIEW_CREATED,
    AUDIT_ACTION_REVIEW_UPDATED,
    AUDIT_TARGET_ANALYSIS,
    MAX_REVIEW_NOTE_LENGTH,
    REVIEW_STATUS_NEEDS_FOLLOW_UP,
    REVIEW_STATUS_REVIEWED,
    REVIEW_STATUS_UNREVIEWED,
    SIGNAL_STATUS_SUCCESS,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    Analysis,
    AnalysisReview,
    AnalysisSignal,
    AuthSession,
    User,
)
from app.db.session import SessionLocal, engine
from app.main import app
from app.web_auth import ENVIRONMENT_VARIABLE, SESSION_COOKIE_NAME, hash_password
from tests.conftest import DASHBOARD_ORIGIN

pytestmark = pytest.mark.integration

ADMIN_ANALYSES_URL = "/api/v1/admin/analyses"

# A test fixture, not a credential: nothing outside this file uses it and no deployment ships
# it.
PASSWORD = "correct-horse-battery-staple"

FORBIDDEN_BODY = {"detail": "Insufficient permissions"}
UNAUTHENTICATED_BODY = {"detail": "Not authenticated"}

# The forensic decision every analysis in this file is created carrying. Real values, because
# the isolation assertions are only meaningful against columns that actually hold something —
# a review that failed to overwrite four nulls would prove nothing.
RISK_LEVEL = "HIGH"
RULES_VERSION = "r7-v4.0.0"
CALIBRATION_ID = "c" * 64
RULE_ID = "R4"

# The fields a review payload carries, and the entire set of them. Written out here rather
# than read back off the first response, so that widening the contract has to be a deliberate
# edit to this line rather than something a test quietly accepts.
VISIBLE_FIELDS = {
    "analysis_id",
    "status",
    "note",
    "reviewer_id",
    "reviewer_email_snapshot",
    "created_at",
    "updated_at",
}


@pytest.fixture(autouse=True)
def plain_http_environment(monkeypatch):
    """`TestClient` speaks plain HTTP, and a browser's jar discards a `Secure` cookie sent over
    it. Without this the sign-ins below would succeed and the very next request would arrive
    with no cookie at all — a failure appearing nowhere near its cause."""
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

    Order is load-bearing in two places. Audit events are deleted by the analysis id they name
    before the analysis goes, not because anything references it — `target_id` is text by
    design — but so the deletion can still find them. And reviews go before analyses: the
    cascade would take them anyway, which is exactly what one test asserts, so this file does
    not rely on it for its own cleanup.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, analyses

        db.rollback()
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
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(AdminAuditEvent).filter(
                AdminAuditEvent.actor_id == user_id
            ).delete(synchronize_session=False)
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


def make_user(session, *, role: str = USER_ROLE_USER) -> User:
    """A persisted account, registered for cleanup."""
    db, users, _ = session

    user = User(
        # Unique per call, so two runs against the same database cannot collide on the unique
        # index over the email.
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def make_analysis(session) -> Analysis:
    """A completed analysis carrying a real risk decision and one signal row.

    Both halves matter for the isolation assertions: the four risk columns are what a review
    must not overwrite, and the signal is what it must not touch either.
    """
    db, _, analyses = session

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
        AnalysisSignal(
            analysis_id=analysis.id,
            provider="fixture",
            signal_type="synthetic_video",
            status=SIGNAL_STATUS_SUCCESS,
            score=0.97,
        )
    )
    db.commit()

    return analysis


def signed_in(user: User) -> TestClient:
    """A client holding a real session for this account, opened through the login route."""
    client = TestClient(app)
    response = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": PASSWORD}
    )

    assert response.status_code == 200, "the fixture account could not sign in"
    assert client.cookies.get(SESSION_COOKIE_NAME), "no session cookie was issued"

    return client


def administrator(session) -> TestClient:
    """A signed-in administrator, which is what every mutation below is made through."""
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def review_url(analysis_id) -> str:
    return f"{ADMIN_ANALYSES_URL}/{analysis_id}/review"


def put_review(client: TestClient, analysis_id, **body):
    """A review, sent the way the dashboard sends one.

    The origin header is not optional: the route carries `require_same_origin`, and a request
    without it is refused before any of the logic this file is about.
    """
    return client.put(
        review_url(analysis_id), json=body, headers={"Origin": DASHBOARD_ORIGIN}
    )


def events_for(session, analysis: Analysis) -> list[AdminAuditEvent]:
    """Every recorded event about one analysis, newest first, read from the database.

    Scoped to an analysis this file created — never a count of the whole table. Other tests
    write audit rows too, and an assertion about "how many events exist" would pass alone and
    fail in a suite.
    """
    db, _, _ = session
    db.commit()  # Drop this session's snapshot so the API's committed writes are visible.

    return list(
        db.execute(
            select(AdminAuditEvent)
            .where(AdminAuditEvent.target_id == str(analysis.id))
            .order_by(AdminAuditEvent.created_at.desc(), AdminAuditEvent.id.desc())
        )
        .scalars()
        .all()
    )


def count_for(session, analysis: Analysis) -> int:
    """How many events stand against one analysis."""
    db, _, _ = session
    db.commit()

    return db.execute(
        select(func.count(AdminAuditEvent.id)).where(
            AdminAuditEvent.target_id == str(analysis.id)
        )
    ).scalar_one()


def forensic_state(session, analysis: Analysis) -> tuple:
    """Everything about this analysis that a review must never change.

    The four risk columns, the status, and every signal row with its score. Read fresh from
    the database each time, so comparing two of these compares what was actually committed
    rather than two views of one identity-mapped object.
    """
    db, _, _ = session
    db.commit()
    db.expire_all()

    row = db.execute(
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
            AnalysisSignal.signal_type, AnalysisSignal.status, AnalysisSignal.score
        )
        .where(AnalysisSignal.analysis_id == analysis.id)
        .order_by(AnalysisSignal.signal_type)
    ).all()

    return (tuple(row), [tuple(signal) for signal in signals])


# --- isolation: the forensic record is out of reach -----------------------------------


def test_a_review_leaves_every_forensic_field_exactly_as_it_was(session):
    """The claim the whole design rests on, asserted by comparison rather than by inspection."""
    analysis = make_analysis(session)
    before = forensic_state(session, analysis)

    client = administrator(session)
    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_NEEDS_FOLLOW_UP, note="Source disputed."
    )

    assert response.status_code == 200
    assert forensic_state(session, analysis) == before


def test_a_revision_leaves_every_forensic_field_exactly_as_it_was(session):
    """The second write is the one that issues an UPDATE, so it is asserted separately."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="First pass.")
    before = forensic_state(session, analysis)

    put_review(
        client, analysis.id, status=REVIEW_STATUS_NEEDS_FOLLOW_UP, note="Second pass."
    )

    assert forensic_state(session, analysis) == before


def test_a_request_naming_a_forensic_field_is_refused(session):
    """`extra="forbid"`, asserted against the field it exists to refuse.

    A 422 and not a 200-with-the-field-ignored. The difference is whether an operator who sent
    `risk_level` is told it does not exist or is left believing they set it.
    """
    analysis = make_analysis(session)
    before = forensic_state(session, analysis)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="", risk_level="LOW"
    )

    assert response.status_code == 422
    assert forensic_state(session, analysis) == before


def test_a_request_naming_the_reviewer_is_refused(session):
    """The identity comes from the session. There is no field for a caller to put one in."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        note="",
        reviewer_id=str(uuid.uuid4()),
    )

    assert response.status_code == 422


# --- the taxonomy ---------------------------------------------------------------------


@pytest.mark.parametrize("status", [REVIEW_STATUS_REVIEWED, REVIEW_STATUS_NEEDS_FOLLOW_UP])
def test_both_operational_statuses_are_accepted(session, status):
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(client, analysis.id, status=status, note="")

    assert response.status_code == 200
    assert response.json()["status"] == status


@pytest.mark.parametrize(
    "status", ["FAKE", "GENUINE", "REAL", "CONFIRMED", "HIGH", "reviewed", ""]
)
def test_a_status_that_reads_as_a_verdict_is_refused(session, status):
    """Not a spelling test. Each of these is an answer about the media, and the only thing in
    this system entitled to give one is the risk engine."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(client, analysis.id, status=status, note="")

    assert response.status_code == 422


def test_unreviewed_cannot_be_written(session):
    """The third state is the absence of a row, not a value. Nothing may store it — which also
    means there is no route that un-reviews an analysis."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(client, analysis.id, status=REVIEW_STATUS_UNREVIEWED, note="")

    assert response.status_code == 422


# --- the read side --------------------------------------------------------------------


def test_an_analysis_nobody_has_reviewed_reads_as_unreviewed(session):
    """A 200 and not a 404. The state is real and every analysis in the deployment is in it."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = client.get(review_url(analysis.id))

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == REVIEW_STATUS_UNREVIEWED
    assert payload["note"] == ""
    assert payload["reviewer_id"] is None
    assert payload["reviewer_email_snapshot"] is None
    assert payload["created_at"] is None
    assert payload["updated_at"] is None


def test_an_unknown_analysis_is_not_found_on_read(session):
    client = administrator(session)

    response = client.get(review_url(uuid.uuid4()))

    assert response.status_code == 404


def test_an_unknown_analysis_is_not_found_on_write(session):
    """404 before any insert. The alternative is an orphan the foreign key would reject as a
    500 — a server fault for a request that was merely wrong."""
    client = administrator(session)

    response = put_review(client, uuid.uuid4(), status=REVIEW_STATUS_REVIEWED, note="")

    assert response.status_code == 404


def test_a_malformed_analysis_id_is_rejected_by_validation(session):
    client = administrator(session)

    assert client.get(f"{ADMIN_ANALYSES_URL}/not-a-uuid/review").status_code == 422


def test_a_review_payload_carries_the_recorded_fields_and_no_others(session):
    analysis = make_analysis(session)
    client = administrator(session)
    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")

    payload = client.get(review_url(analysis.id)).json()

    assert set(payload) == VISIBLE_FIELDS


def test_a_stored_review_reads_back_as_it_was_written(session):
    analysis = make_analysis(session)
    admin = make_user(session, role=USER_ROLE_ADMIN)
    client = signed_in(admin)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
        note="Broadcaster upload disagrees on timestamps.",
    )
    payload = client.get(review_url(analysis.id)).json()

    assert payload["analysis_id"] == str(analysis.id)
    assert payload["status"] == REVIEW_STATUS_NEEDS_FOLLOW_UP
    assert payload["note"] == "Broadcaster upload disagrees on timestamps."
    assert payload["reviewer_id"] == str(admin.id)
    assert payload["reviewer_email_snapshot"] == admin.email


# --- the write side -------------------------------------------------------------------


def test_the_reviewer_is_the_authenticated_administrator(session):
    """Taken from the session, never from the body — and the body has no field for it anyway."""
    analysis = make_analysis(session)
    admin = make_user(session, role=USER_ROLE_ADMIN)

    put_review(
        signed_in(admin), analysis.id, status=REVIEW_STATUS_REVIEWED, note="Mine."
    )

    db, _, _ = session
    db.commit()
    review = db.get(AnalysisReview, analysis.id)

    assert review.reviewer_id == admin.id
    assert review.reviewer_email_snapshot == admin.email


def test_a_revision_replaces_the_reviewer_with_the_one_who_made_it(session):
    """There is one review per analysis and it is signed by whoever last wrote it. Who wrote
    the previous one is in the audit log, which is where history lives."""
    analysis = make_analysis(session)
    first = make_user(session, role=USER_ROLE_ADMIN)
    second = make_user(session, role=USER_ROLE_ADMIN)

    put_review(signed_in(first), analysis.id, status=REVIEW_STATUS_REVIEWED, note="A")
    put_review(
        signed_in(second), analysis.id, status=REVIEW_STATUS_NEEDS_FOLLOW_UP, note="B"
    )

    db, _, _ = session
    db.commit()
    review = db.get(AnalysisReview, analysis.id)
    db.refresh(review)

    assert review.reviewer_id == second.id
    assert review.reviewer_email_snapshot == second.email


def test_a_note_is_stored_verbatim_including_anything_that_looks_like_markup(session):
    """Stored as typed. It is not escaped, not sanitized and not rewritten — the guarantee is
    that nothing ever renders it as markup, which is a property of the reader, not of the
    column. Rewriting somebody's note would be worse: the record would no longer say what they
    wrote."""
    analysis = make_analysis(session)
    client = administrator(session)
    note = "<script>alert(1)</script> & <b>bold</b>"

    response = put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note=note)

    assert response.status_code == 200
    assert response.json()["note"] == note
    assert client.get(review_url(analysis.id)).json()["note"] == note


def test_a_note_is_stripped_once_and_stored_stripped(session):
    """One normalization. Two would disagree about whether anything changed."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="   padded   "
    )

    assert response.json()["note"] == "padded"


def test_a_note_of_only_whitespace_is_no_note(session):
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="   ")

    assert response.json()["note"] == ""


def test_a_note_at_the_limit_is_accepted(session):
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="x" * MAX_REVIEW_NOTE_LENGTH
    )

    assert response.status_code == 200


def test_a_note_over_the_limit_is_refused(session):
    """Refused here and not by PostgreSQL. A value this model accepted and the column rejected
    would surface as a 500 on a request whose only fault was length."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        note="x" * (MAX_REVIEW_NOTE_LENGTH + 1),
    )

    assert response.status_code == 422


def test_a_note_carrying_a_control_character_is_refused(session):
    """The NUL byte is the one that must not reach PostgreSQL, which cannot store it at all."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="before\x00after"
    )

    assert response.status_code == 422


def test_a_note_may_contain_newlines_and_tabs(session):
    """Both are ordinary text. Refusing them would make a two-paragraph note impossible."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="one\n\ttwo"
    )

    assert response.status_code == 200
    assert response.json()["note"] == "one\n\ttwo"


def test_the_note_may_be_omitted(session):
    """"Reviewed, nothing to add" is a body with one field."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = client.put(
        review_url(analysis.id),
        json={"status": REVIEW_STATUS_REVIEWED},
        headers={"Origin": DASHBOARD_ORIGIN},
    )

    assert response.status_code == 200
    assert response.json()["note"] == ""


def test_a_cross_origin_write_is_refused(session):
    """The route carries `require_same_origin`, like every other dashboard mutation."""
    analysis = make_analysis(session)
    client = administrator(session)

    response = client.put(
        review_url(analysis.id),
        json={"status": REVIEW_STATUS_REVIEWED, "note": ""},
        headers={"Origin": "https://elsewhere.example"},
    )

    assert response.status_code == 403

    db, _, _ = session
    db.commit()
    assert db.get(AnalysisReview, analysis.id) is None


# --- the audit trail ------------------------------------------------------------------


def test_a_first_review_writes_exactly_one_creation_event(session):
    analysis = make_analysis(session)
    admin = make_user(session, role=USER_ROLE_ADMIN)

    put_review(
        signed_in(admin), analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked."
    )

    events = events_for(session, analysis)
    assert len(events) == 1
    assert events[0].action == AUDIT_ACTION_REVIEW_CREATED
    assert events[0].target_type == AUDIT_TARGET_ANALYSIS
    assert events[0].target_id == str(analysis.id)
    assert events[0].actor_id == admin.id
    assert events[0].actor_email_snapshot == admin.email
    # An analysis is not a person and has no address. Null rather than the reviewer's, which
    # would make the target of the event read as the administrator who wrote it.
    assert events[0].target_email_snapshot is None


def test_a_first_review_records_the_move_out_of_unreviewed(session):
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")

    changes = events_for(session, analysis)[0].changes
    assert changes["status"] == {
        "old": REVIEW_STATUS_UNREVIEWED,
        "new": REVIEW_STATUS_REVIEWED,
    }
    assert changes["note_changed"] is True


def test_a_revision_writes_an_update_event(session):
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="First.")
    put_review(client, analysis.id, status=REVIEW_STATUS_NEEDS_FOLLOW_UP, note="Second.")

    events = events_for(session, analysis)
    assert len(events) == 2
    assert events[0].action == AUDIT_ACTION_REVIEW_UPDATED
    assert events[0].changes["status"] == {
        "old": REVIEW_STATUS_REVIEWED,
        "new": REVIEW_STATUS_NEEDS_FOLLOW_UP,
    }
    assert events[0].changes["note_changed"] is True


def test_an_event_omits_the_status_when_only_the_note_moved(session):
    """The house convention: a payload restating an untouched field makes every row look like
    a change to everything."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="First.")
    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Reworded.")

    changes = events_for(session, analysis)[0].changes
    assert "status" not in changes
    assert changes["note_changed"] is True


def test_note_changed_is_false_when_only_the_status_moved(session):
    """Present either way, unlike `status`, precisely because the wording is withheld: without
    it a reader could not tell "unchanged" from "not recorded"."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Same words.")
    put_review(
        client, analysis.id, status=REVIEW_STATUS_NEEDS_FOLLOW_UP, note="Same words."
    )

    changes = events_for(session, analysis)[0].changes
    assert changes["note_changed"] is False
    assert changes["status"]["new"] == REVIEW_STATUS_NEEDS_FOLLOW_UP


def test_no_event_ever_carries_the_text_of_a_note(session):
    """Asserted against the serialized event rather than against the keys, so a future field
    that happened to include the note would fail this too."""
    analysis = make_analysis(session)
    client = administrator(session)
    secret = "distinctive-prose-that-must-not-be-copied"

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note=secret)
    put_review(client, analysis.id, status=REVIEW_STATUS_NEEDS_FOLLOW_UP, note=secret)

    for event in events_for(session, analysis):
        assert secret not in str(event.changes)


def test_a_request_that_changes_nothing_writes_no_event(session):
    """The screen submits both controls on every save, so this is the ordinary case."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")
    assert count_for(session, analysis) == 1

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked."
    )

    assert response.status_code == 200
    assert count_for(session, analysis) == 1


def test_a_request_that_changes_nothing_does_not_move_the_timestamp(session):
    """"Last reviewed" means the last time somebody changed their answer, not the last time
    somebody opened the form."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")
    before = client.get(review_url(analysis.id)).json()["updated_at"]

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")

    assert client.get(review_url(analysis.id)).json()["updated_at"] == before


def test_a_save_differing_only_in_whitespace_is_a_no_op(session):
    """The comparison is against the stripped note, which is the one the model produced."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")
    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="  Checked.  ")

    assert count_for(session, analysis) == 1


def test_a_no_op_does_not_reassign_the_reviewer(session):
    """A second administrator saving an unchanged form does not become the reviewer. Nothing
    was written, so nothing was signed."""
    analysis = make_analysis(session)
    first = make_user(session, role=USER_ROLE_ADMIN)
    second = make_user(session, role=USER_ROLE_ADMIN)

    put_review(signed_in(first), analysis.id, status=REVIEW_STATUS_REVIEWED, note="A")
    put_review(signed_in(second), analysis.id, status=REVIEW_STATUS_REVIEWED, note="A")

    db, _, _ = session
    db.commit()
    review = db.get(AnalysisReview, analysis.id)
    db.refresh(review)

    assert review.reviewer_id == first.id


def test_a_refused_request_writes_neither_a_review_nor_an_event(session):
    """The refusal and the event are in one transaction, so there is no arrangement in which
    the write is rejected and the record of it survives."""
    analysis = make_analysis(session)
    client = administrator(session)

    assert put_review(client, analysis.id, status="FAKE", note="x").status_code == 422

    db, _, _ = session
    db.commit()
    assert db.get(AnalysisReview, analysis.id) is None
    assert count_for(session, analysis) == 0


# --- authorization --------------------------------------------------------------------


def test_an_anonymous_caller_cannot_read_a_review(session):
    analysis = make_analysis(session)

    response = TestClient(app).get(review_url(analysis.id))

    assert response.status_code == 401
    assert response.json() == UNAUTHENTICATED_BODY


def test_an_anonymous_caller_cannot_write_a_review(session):
    analysis = make_analysis(session)

    response = TestClient(app).put(
        review_url(analysis.id),
        json={"status": REVIEW_STATUS_REVIEWED, "note": ""},
        headers={"Origin": DASHBOARD_ORIGIN},
    )

    assert response.status_code == 401

    db, _, _ = session
    db.commit()
    assert db.get(AnalysisReview, analysis.id) is None


def test_an_ordinary_user_cannot_read_a_review(session):
    """Who has looked at which case is itself operational information, so the GET is guarded
    exactly as the PUT is."""
    analysis = make_analysis(session)
    client = signed_in(make_user(session))

    response = client.get(review_url(analysis.id))

    assert response.status_code == 403
    assert response.json() == FORBIDDEN_BODY


def test_an_ordinary_user_cannot_write_a_review(session):
    analysis = make_analysis(session)
    client = signed_in(make_user(session))

    response = put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="x")

    assert response.status_code == 403

    db, _, _ = session
    db.commit()
    assert db.get(AnalysisReview, analysis.id) is None
    assert count_for(session, analysis) == 0


def test_an_ordinary_user_cannot_learn_whether_an_analysis_exists(session):
    """The 403 arrives before the lookup, so a non-administrator cannot use this route to probe
    for ids: an analysis that exists and one that does not answer identically."""
    analysis = make_analysis(session)
    client = signed_in(make_user(session))

    assert client.get(review_url(analysis.id)).status_code == 403
    assert client.get(review_url(uuid.uuid4())).status_code == 403


# --- durability of the row ------------------------------------------------------------


def test_a_review_survives_the_deletion_of_the_account_that_wrote_it(session):
    """`reviewer_id` is not a foreign key, so the account can go. What keeps the row readable
    is the snapshot, which is the other half of that decision."""
    analysis = make_analysis(session)
    admin = make_user(session, role=USER_ROLE_ADMIN)
    address = admin.email

    put_review(signed_in(admin), analysis.id, status=REVIEW_STATUS_REVIEWED, note="Mine.")

    db, _, _ = session
    db.commit()
    db.query(AuthSession).filter(AuthSession.user_id == admin.id).delete()
    db.query(User).filter(User.id == admin.id).delete()
    db.commit()

    review = db.get(AnalysisReview, analysis.id)
    db.refresh(review)

    assert review is not None, "deleting the reviewer must not delete the review"
    assert review.reviewer_id == admin.id
    assert review.reviewer_email_snapshot == address


def test_deleting_an_analysis_takes_its_review_with_it(session):
    """`ON DELETE CASCADE`. A review of an analysis that no longer exists annotates nothing —
    unlike an audit event, which is about a person's action and survives everything."""
    analysis = make_analysis(session)
    client = administrator(session)
    put_review(client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="Checked.")

    db, _, _ = session
    db.commit()
    db.query(AnalysisSignal).filter(AnalysisSignal.analysis_id == analysis.id).delete()
    db.query(Analysis).filter(Analysis.id == analysis.id).delete()
    db.commit()

    assert db.get(AnalysisReview, analysis.id) is None
    # The audit event does not go with it. It is a record of what a person did, and the
    # analysis's removal does not un-do the doing.
    assert count_for(session, analysis) == 1


def test_the_database_refuses_a_status_outside_the_taxonomy(session):
    """The check constraint, asserted directly. The API refuses one too — this is the layer
    underneath, for the insert that does not come through the API."""
    analysis = make_analysis(session)
    db, _, _ = session

    db.add(
        AnalysisReview(
            analysis_id=analysis.id,
            status="FAKE",
            note="",
            reviewer_id=uuid.uuid4(),
        )
    )

    with pytest.raises(SQLAlchemyError):
        db.commit()

    db.rollback()


def test_one_analysis_cannot_hold_two_reviews(session):
    """The analysis is the primary key, so "one review per analysis" is a property of the
    table rather than a rule the endpoint remembers."""
    analysis = make_analysis(session)
    db, _, _ = session

    for _ in range(2):
        db.add(
            AnalysisReview(
                analysis_id=analysis.id,
                status=REVIEW_STATUS_REVIEWED,
                note="",
                reviewer_id=uuid.uuid4(),
            )
        )

    with pytest.raises(SQLAlchemyError):
        db.commit()

    db.rollback()
