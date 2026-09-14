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

*The analyst assessment.* Added in R9-T7, and the fifth group because it is a second axis
rather than more of the first. What a reviewer made of the automated assessment is recorded
beside the workflow status and never merged into it: the two are separate columns, separate
controls and separate diffs in the audit log. The claims about it are the claims about the
review generally, made again against the new field — it is storable in exactly three spellings
and in no fourth, null is a distinct state from `UNDETERMINED` and is never coerced into it,
and writing one leaves the forensic record untouched including the two values that are derived
from it rather than stored, the decision coverage and the provenance. `TRUE`, `FALSE`, `TP`,
`FP` and `CONFIRMED` are refused by name for the reason `FAKE` and `GENUINE` are: an opinion
that reads as a verdict, or as a claim about whether the detector was right, would be a second
classification of the media or a ground truth label this system does not hold.

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
    ANALYST_ASSESSMENT_AGREES,
    ANALYST_ASSESSMENT_DISAGREES,
    ANALYST_ASSESSMENT_UNDETERMINED,
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
from app.detection import PROVENANCE_SIGNAL, SYNTHETIC_VIDEO_SIGNAL
from app.main import app
from app.provenance_status import provenance_state
from app.risk_engine import (
    RULE_V5_INCONCLUSIVE_PARTIAL,
    RULES_VERSION_V5,
    VERDICT_INCONCLUSIVE,
)
from app.risk_trace import RULESET_V5, PersistedSignal, build_trace
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
    # The analyst's opinion of the automated assessment (R9-T7), on its own axis from the
    # workflow status above and never merged into it.
    "analyst_assessment",
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


# --- the analyst assessment (R9-T7) ---------------------------------------------------

# The three storable assessments, and the fourth state that is the absence of one. Written out
# rather than imported as a tuple from the model, so that widening the vocabulary has to be a
# deliberate edit to this line — the same reason `VISIBLE_FIELDS` is spelled out above.
#
# `None` sits in the same list because it is a value a request may send and a value the payload
# may carry, and every claim below has to hold for it too. It is not a fourth spelling: it means
# no opinion was recorded, which is what every review written before this field existed says.
ASSESSMENTS = (
    ANALYST_ASSESSMENT_AGREES,
    ANALYST_ASSESSMENT_DISAGREES,
    ANALYST_ASSESSMENT_UNDETERMINED,
    None,
)

# Words that must never be storable in the assessment column, asserted by name.
#
# Two different kinds of wrong, deliberately mixed. `TRUE`, `FALSE` and `CONFIRMED` are answers
# about the media, and the only thing in this system entitled to give one is the risk engine
# under a named ruleset. `TP` and `FP` are claims about whether the engine was *right*, which
# is a question about ground truth — this deployment holds none, and an analyst pressing a
# button in a form does not create any.
#
# `AGREES` is here for a third reason: it is the correct concept under a spelling short enough
# to be read as a verdict once it is out of context and in a column heading. The vocabulary is
# long on purpose and the abbreviation of it is not an alias for it.
FORBIDDEN_ASSESSMENTS = ("TRUE", "FALSE", "TP", "FP", "TN", "FN", "CONFIRMED", "AGREES")


# The calibrated identity the v5 ruleset requires before it will read a synthetic-video score
# at all. Taken off the ruleset rather than retyped: a different deployment of the same model is
# an uncalibrated one, so a fixture that invented a provider version would produce a reading the
# coverage model declines to count and a coverage of 0/2 that proves less than it appears to.
V5_SVD = RULESET_V5.signals[0]


def make_decided_analysis(session) -> Analysis:
    """An analysis carrying a v5 decision, a usable detector reading and a provenance reading.

    Richer than `make_analysis` and for one reason: two of the things a review must not alter
    are not columns. The decision coverage and the provenance status are *derived* from the
    persisted signal rows every time they are read, so asserting that a review leaves them
    alone requires rows they can actually be derived from — against an analysis with no
    provenance signal, "the provenance did not change" is a statement about two nulls. The
    coverage needs `r9-v5.0.0` specifically, because that is the only ruleset that freezes a
    `decision_total`; every earlier version states no coverage at all.

    The record is internally coherent rather than merely well-formed. The synthetic-video
    reading is usable and below its measured threshold, the face detector produced nothing, and
    one usable reading out of a frozen denominator of two is exactly the partial coverage that
    `R9-300` concludes `INCONCLUSIVE` on. The provenance reading carries a manifest, so the
    provenance axes are substantive too.

    `INCONCLUSIVE` is also the only v5 verdict that fits the `risk_level` column as it stands —
    see the note on `test_the_fixture_actually_has_something_to_leave_alone`.
    """
    db, _, analyses = session

    analysis = Analysis(
        status=ANALYSIS_STATUS_COMPLETED,
        risk_level=VERDICT_INCONCLUSIVE,
        risk_rules_version=RULES_VERSION_V5,
        risk_calibration_id=RULESET_V5.calibration_id,
        risk_rule_id=RULE_V5_INCONCLUSIVE_PARTIAL,
    )
    db.add(analysis)
    db.flush()
    analyses.append(analysis.id)

    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider=V5_SVD.provider,
            signal_type=SYNTHETIC_VIDEO_SIGNAL,
            status=SIGNAL_STATUS_SUCCESS,
            provider_version=V5_SVD.provider_version,
            # Below the threshold, which is a reading and therefore usable. "Did not reach it"
            # and "reached it" are both coverage; only the absence of a reading is not.
            score=V5_SVD.threshold - 0.1,
            signal_metadata={V5_SVD.count_key: 8},
        )
    )
    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider="fixture",
            signal_type=PROVENANCE_SIGNAL,
            status=SIGNAL_STATUS_SUCCESS,
            provider_version="fixture-c2pa-1",
            score=None,
            signal_metadata={"manifest_exists": True},
        )
    )
    db.commit()

    return analysis


def derived_state(session, analysis: Analysis) -> tuple:
    """The two forensic values that are computed rather than stored, as a reader would see them.

    `forensic_state` above compares the columns and the signal rows. This compares what those
    rows *mean* — the decision coverage and the two provenance axes — because a reader of a
    report never sees the rows, and a change that left every column identical while moving a
    derived value would be a change to the forensic answer that a column-by-column comparison
    would pass.

    Both are produced here the way the API produces them, by calling the same two functions on
    the persisted rows. Nothing is recomputed by hand: a second implementation of the coverage
    model written in a test would be the copy the whole R9 line exists to prevent, and it would
    agree with itself rather than with the application.
    """
    db, _, _ = session
    db.commit()
    db.expire_all()

    rows = db.execute(
        select(AnalysisSignal).where(AnalysisSignal.analysis_id == analysis.id)
    ).scalars()

    signals = {
        row.signal_type: PersistedSignal(
            provider=row.provider,
            signal_type=row.signal_type,
            status=row.status,
            provider_version=row.provider_version,
            score=row.score,
            metadata=row.signal_metadata,
        )
        for row in rows
    }

    decision = db.execute(
        select(
            Analysis.risk_level,
            Analysis.risk_rule_id,
            Analysis.risk_rules_version,
            Analysis.risk_calibration_id,
        ).where(Analysis.id == analysis.id)
    ).one()

    trace = build_trace(
        risk_level=decision.risk_level,
        rule_id=decision.risk_rule_id,
        rules_version=decision.risk_rules_version,
        calibration_id=decision.risk_calibration_id,
        signals=signals,
    )

    provenance = signals.get(PROVENANCE_SIGNAL)
    state = provenance_state(
        provenance.status if provenance is not None else None,
        (provenance.metadata or {}).get("manifest_exists")
        if provenance is not None
        else None,
    )

    coverage = None if trace is None else trace.decision_coverage

    return (
        None
        if coverage is None
        else (coverage.usable, coverage.total, coverage.status, coverage.is_complete),
        state.status,
        state.availability,
    )


def stored_assessment(session, analysis: Analysis) -> str | None:
    """The assessment as the column holds it, read fresh rather than off a cached object."""
    db, _, _ = session
    db.commit()
    db.expire_all()

    return db.execute(
        select(AnalysisReview.analyst_assessment).where(
            AnalysisReview.analysis_id == analysis.id
        )
    ).scalar_one()


@pytest.mark.parametrize("assessment", ASSESSMENTS)
def test_an_assessment_round_trips_through_the_api_and_the_column(session, assessment):
    """Every storable assessment, and the absence of one, survives a write and a read.

    Three assertions rather than one, because they can fail apart: the response to the write,
    the response to an independent read, and the column itself. A payload echoing back what it
    was sent while storing something else is the failure this shape catches.

    `None` is parametrized alongside the three words deliberately. It is not the untested
    default — it is the state a legacy review is in and the state clearing one returns to, and
    it has to round-trip as itself rather than arriving back as `UNDETERMINED`.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    written = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=assessment,
        note="Checked.",
    )

    assert written.status_code == 200
    assert written.json()["analyst_assessment"] == assessment
    assert set(written.json()) == VISIBLE_FIELDS

    read_back = client.get(review_url(analysis.id))

    assert read_back.status_code == 200
    assert read_back.json()["analyst_assessment"] == assessment
    assert stored_assessment(session, analysis) == assessment


def test_an_omitted_assessment_is_the_absence_of_one_and_not_a_default_opinion(session):
    """A body that does not mention the field stores null, not a value chosen for the reviewer.

    The dashboard always submits the control, so this is the shape an API caller sends. It is
    asserted because the alternative — defaulting to `UNDETERMINED`, or to agreement — would
    have the API record an opinion nobody expressed, which is the one thing this column must
    never do.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, note="No opinion offered."
    )

    assert response.status_code == 200
    assert response.json()["analyst_assessment"] is None
    assert stored_assessment(session, analysis) is None


def test_the_two_axes_are_independent(session):
    """Either may move without the other, which is the whole reason they are two columns.

    A case closed by somebody who disagreed with it and a case left open by somebody who agreed
    are both ordinary, and a design that folded the opinion into the status could express
    neither.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    agreed_but_open = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="",
    ).json()

    assert agreed_but_open["status"] == REVIEW_STATUS_NEEDS_FOLLOW_UP
    assert agreed_but_open["analyst_assessment"] == ANALYST_ASSESSMENT_AGREES

    disagreed_but_closed = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_DISAGREES,
        note="",
    ).json()

    assert disagreed_but_closed["status"] == REVIEW_STATUS_REVIEWED
    assert disagreed_but_closed["analyst_assessment"] == ANALYST_ASSESSMENT_DISAGREES


# --- legacy reviews, which carry no assessment at all -----------------------------------


def legacy_review(session, analysis: Analysis, reviewer: uuid.UUID) -> None:
    """A review written the way one was written before this column existed.

    Inserted through the ORM without naming `analyst_assessment`, which is exactly what a row
    migrated from before R9-T7 looks like: nothing was backfilled, so the column is null.
    """
    db, _, _ = session
    db.add(
        AnalysisReview(
            analysis_id=analysis.id,
            status=REVIEW_STATUS_REVIEWED,
            note="Looked at before the field existed.",
            reviewer_id=reviewer,
            reviewer_email_snapshot="departed@example.com",
        )
    )
    db.commit()


def test_a_legacy_review_reads_back_with_no_assessment_rather_than_an_error(session):
    """The row every existing deployment is full of, read through the new payload.

    A 200 carrying `REVIEWED` and a null, and the complete field set. The failure this guards
    against is a reader that requires the new field and turns the entire history of the
    deployment into an error, which would hide the reviews rather than the missing opinions.
    """
    analysis = make_analysis(session)
    legacy_review(session, analysis, uuid.uuid4())
    client = administrator(session)

    response = client.get(review_url(analysis.id))

    assert response.status_code == 200
    assert set(response.json()) == VISIBLE_FIELDS
    assert response.json()["status"] == REVIEW_STATUS_REVIEWED
    assert response.json()["analyst_assessment"] is None


def test_a_legacy_review_is_not_migrated_by_being_read(session):
    """Reading one leaves it exactly as it was. A read that wrote would be a backfill."""
    analysis = make_analysis(session)
    legacy_review(session, analysis, uuid.uuid4())
    client = administrator(session)

    client.get(review_url(analysis.id))

    assert stored_assessment(session, analysis) is None
    assert count_for(session, analysis) == 0


def test_a_legacy_review_can_be_resaved_without_acquiring_an_opinion(session):
    """Saving one unchanged is a no-op, including on the field it does not have.

    The screen preselects "Not recorded" for a null, so pressing save on a legacy review sends
    the status it already had and no assessment. That must write nothing at all — a reviewer
    who opened an old case and saved it must not have an opinion attributed to them, and the
    audit log must not gain an entry saying something moved.
    """
    analysis = make_analysis(session)
    legacy_review(session, analysis, uuid.uuid4())
    client = administrator(session)

    response = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=None,
        note="Looked at before the field existed.",
    )

    assert response.status_code == 200
    assert response.json()["analyst_assessment"] is None
    assert stored_assessment(session, analysis) is None
    assert count_for(session, analysis) == 0


def test_a_legacy_review_keeps_its_status_when_an_assessment_is_added(session):
    """Adding an opinion to an old review moves one axis and leaves the other alone."""
    analysis = make_analysis(session)
    legacy_review(session, analysis, uuid.uuid4())
    client = administrator(session)

    response = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_DISAGREES,
        note="Looked at before the field existed.",
    )

    assert response.json()["status"] == REVIEW_STATUS_REVIEWED
    assert response.json()["analyst_assessment"] == ANALYST_ASSESSMENT_DISAGREES

    changes = events_for(session, analysis)[0].changes

    assert "status" not in changes
    assert changes["analyst_assessment"] == {
        "old": None,
        "new": ANALYST_ASSESSMENT_DISAGREES,
    }


# --- isolation, again, for the second axis ----------------------------------------------


@pytest.mark.parametrize("assessment", ASSESSMENTS)
def test_an_assessment_leaves_every_forensic_value_exactly_as_it_was(session, assessment):
    """The claim that matters most about this field, asserted for every value it may take.

    Both snapshots, because they catch different failures. `forensic_state` compares the risk
    columns and the detector rows; `derived_state` compares the decision coverage and the two
    provenance axes, which are computed from those rows and are what a report actually prints.

    Disagreement is in the parameter list and is not a special case. An analyst saying the
    automated assessment is wrong writes one nullable column on the review and changes nothing
    about the verdict, the coverage, the provenance or the evidence — the record still says
    exactly what the engine decided, and what a human thought of it is stored beside that.
    """
    analysis = make_decided_analysis(session)
    before_columns = forensic_state(session, analysis)
    before_derived = derived_state(session, analysis)

    client = administrator(session)
    response = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=assessment,
        note="Reviewed against the source.",
    )

    assert response.status_code == 200
    assert forensic_state(session, analysis) == before_columns
    assert derived_state(session, analysis) == before_derived


def test_the_fixture_actually_has_something_to_leave_alone(session):
    """The isolation assertions above are only meaningful against substantive values.

    Without this, a change that made `derived_state` return three nulls would make every
    comparison above pass by comparing nothing to nothing. This is the test that fails instead,
    and it pins the exact coverage rather than merely requiring one: 1 usable reading out of the
    version's frozen denominator of 2, reported partial.

    A note for whoever wires `evaluate_v5` into the worker. The fixture above stores
    `INCONCLUSIVE` because it is the only v5 verdict that fits `analyses.risk_level`, which is
    `String(16)` — `MANIPULATION_DETECTED` is 21 characters and
    `NO_CALIBRATED_MANIPULATION_SIGNAL` is 33, and neither can be persisted today. Nothing
    writes a v5 verdict yet, so this is latent rather than broken, and widening the column is
    that task's migration and not this one's.
    """
    analysis = make_decided_analysis(session)
    coverage, status, availability = derived_state(session, analysis)

    assert coverage == (1, 2, "partial", False)
    assert status == "PROVENANCE_PRESENT"
    assert availability == "AVAILABLE"


@pytest.mark.parametrize("assessment", ASSESSMENTS)
def test_revising_an_assessment_leaves_every_forensic_value_exactly_as_it_was(
    session, assessment
):
    """The second write is the one that issues an UPDATE, so it is asserted separately."""
    analysis = make_decided_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="First pass.",
    )
    before_columns = forensic_state(session, analysis)
    before_derived = derived_state(session, analysis)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
        analyst_assessment=assessment,
        note="Second pass.",
    )

    assert forensic_state(session, analysis) == before_columns
    assert derived_state(session, analysis) == before_derived


# --- the audit trail carries both axes, and still never the note ------------------------


def test_the_two_axes_are_separate_diff_keys(session):
    """One event, two named movements, told apart in the payload.

    Separate keys rather than one compound entry, and that is the requirement rather than the
    tidy option: "closed the case" and "disagreed with the detector" are different statements,
    and the audit log is the one record that has to be able to distinguish them.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
        analyst_assessment=ANALYST_ASSESSMENT_DISAGREES,
        note="Both moved.",
    )

    events = events_for(session, analysis)

    assert len(events) == 1
    assert events[0].action == AUDIT_ACTION_REVIEW_CREATED
    assert events[0].target_type == AUDIT_TARGET_ANALYSIS
    assert events[0].changes["status"] == {
        "old": REVIEW_STATUS_UNREVIEWED,
        "new": REVIEW_STATUS_NEEDS_FOLLOW_UP,
    }
    assert events[0].changes["analyst_assessment"] == {
        "old": None,
        "new": ANALYST_ASSESSMENT_DISAGREES,
    }


def test_an_event_omits_the_assessment_when_only_the_status_moved(session):
    """The house convention, applied to the new field: an untouched axis is not restated.

    An event that named both every time would make every row look like a change to everything,
    which is the failure that makes an audit log unreadable rather than merely verbose.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="Steady.",
    )
    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="Steady.",
    )

    changes = events_for(session, analysis)[0].changes

    assert changes["status"]["new"] == REVIEW_STATUS_NEEDS_FOLLOW_UP
    assert "analyst_assessment" not in changes
    assert changes["note_changed"] is False


def test_an_event_omits_the_status_when_only_the_assessment_moved(session):
    """The mirror of the test above, and the one that proves the two are genuinely separate."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="Steady.",
    )
    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_UNDETERMINED,
        note="Steady.",
    )

    changes = events_for(session, analysis)[0].changes

    assert "status" not in changes
    assert changes["analyst_assessment"] == {
        "old": ANALYST_ASSESSMENT_AGREES,
        "new": ANALYST_ASSESSMENT_UNDETERMINED,
    }
    assert changes["note_changed"] is False


def test_no_event_carries_the_text_of_a_note_when_an_assessment_moves(session):
    """The existing claim, made again on the path that writes the new field.

    A distinctive string goes into the note and must appear nowhere in any serialized event
    for this analysis. Asserted against the whole payload rather than against the keys it is
    known to have, so a field added later that happened to carry the wording would fail here.
    """
    secret = f"note-body-{uuid.uuid4().hex}"
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_DISAGREES,
        note=secret,
    )
    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_NEEDS_FOLLOW_UP,
        analyst_assessment=ANALYST_ASSESSMENT_UNDETERMINED,
        note=f"{secret}-revised",
    )

    events = events_for(session, analysis)

    assert len(events) == 2
    for event in events:
        assert secret not in repr(event.changes)
        assert set(event.changes) <= {"status", "analyst_assessment", "note_changed"}


def test_an_exact_no_op_on_both_axes_writes_no_event(session):
    """Nothing moved, so nothing is recorded — including the timestamp.

    The screen submits every control on every save, so "open the form and press save" is the
    ordinary case rather than a strange one. A log in which a dozen entries say a review was
    updated to what it already said cannot answer when the review actually changed.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="Settled.",
    )
    after_first = count_for(session, analysis)
    stamped = client.get(review_url(analysis.id)).json()["updated_at"]

    repeated = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="Settled.",
    )

    assert repeated.status_code == 200
    assert repeated.json()["analyst_assessment"] == ANALYST_ASSESSMENT_AGREES
    assert count_for(session, analysis) == after_first
    assert client.get(review_url(analysis.id)).json()["updated_at"] == stamped


# --- clearing an assessment back to nothing ---------------------------------------------


def test_an_assessment_can_be_cleared_and_the_clearing_is_audited(session):
    """Returning to "no opinion recorded" is a real revision and is logged as one.

    A PUT is the whole review, so clearing is expressed by sending no assessment — which is
    what makes the field the one part of this body that can be unset as well as set. An
    operator who chose the wrong option must be able to take it back, and the record must say
    that they did.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_DISAGREES,
        note="Steady.",
    )

    cleared = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=None,
        note="Steady.",
    )

    assert cleared.status_code == 200
    assert cleared.json()["analyst_assessment"] is None
    assert stored_assessment(session, analysis) is None

    changes = events_for(session, analysis)[0].changes

    assert changes["analyst_assessment"] == {
        "old": ANALYST_ASSESSMENT_DISAGREES,
        "new": None,
    }
    assert "status" not in changes
    assert events_for(session, analysis)[0].action == AUDIT_ACTION_REVIEW_UPDATED


def test_clearing_an_assessment_twice_writes_one_event(session):
    """The second clear moves nothing, so it is a no-op like any other."""
    analysis = make_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_AGREES,
        note="",
    )
    put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, analyst_assessment=None, note=""
    )
    after_clearing = count_for(session, analysis)

    put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, analyst_assessment=None, note=""
    )

    assert count_for(session, analysis) == after_clearing


def test_clearing_an_assessment_leaves_the_forensic_record_alone(session):
    """Clearing writes a null over a column. It reaches nothing else."""
    analysis = make_decided_analysis(session)
    client = administrator(session)

    put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=ANALYST_ASSESSMENT_DISAGREES,
        note="",
    )
    before_columns = forensic_state(session, analysis)
    before_derived = derived_state(session, analysis)

    put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, analyst_assessment=None, note=""
    )

    assert forensic_state(session, analysis) == before_columns
    assert derived_state(session, analysis) == before_derived


# --- the vocabulary, refused by name ----------------------------------------------------


@pytest.mark.parametrize("forbidden", FORBIDDEN_ASSESSMENTS)
def test_a_verdict_or_a_correctness_label_is_refused(session, forbidden):
    """422, and nothing written — neither a review nor an audit row.

    These are refused not because they are misspelled but because of what they would mean. A
    verdict here would be a second, unversioned classification of the media sitting beside the
    risk engine's, with nothing to say which the report meant. A correctness label would be a
    ground truth claim, and this system holds no ground truth against which the detector could
    be marked right or wrong.
    """
    analysis = make_analysis(session)
    before = forensic_state(session, analysis)
    client = administrator(session)

    response = put_review(
        client,
        analysis.id,
        status=REVIEW_STATUS_REVIEWED,
        analyst_assessment=forbidden,
        note="",
    )

    assert response.status_code == 422
    assert forensic_state(session, analysis) == before
    assert count_for(session, analysis) == 0
    assert client.get(review_url(analysis.id)).json()["status"] == REVIEW_STATUS_UNREVIEWED


def test_an_empty_assessment_is_refused_rather_than_read_as_no_opinion(session):
    """The empty string is not a spelling of null, and the API does not accept it as one.

    The form sends `""` for "not recorded" and the route handler in front of the API converts
    it to a JSON null. That conversion is the browser's half of the contract and belongs there;
    the API's vocabulary has three words and none of them is the empty string, so a caller that
    sends one is told so rather than having it interpreted.
    """
    analysis = make_analysis(session)
    client = administrator(session)

    response = put_review(
        client, analysis.id, status=REVIEW_STATUS_REVIEWED, analyst_assessment="", note=""
    )

    assert response.status_code == 422
    assert count_for(session, analysis) == 0


def test_the_database_refuses_an_assessment_outside_the_taxonomy(session):
    """The check constraint, asserted by going around the API entirely.

    The application would refuse the value; this proves the database refuses it too. A rule
    that lives only in a Pydantic model is one route, one migration or one hand-written UPDATE
    away from not running, and the column this protects is the one a forensic-sounding word
    must never reach.
    """
    analysis = make_analysis(session)
    db, _, _ = session

    db.add(
        AnalysisReview(
            analysis_id=analysis.id,
            status=REVIEW_STATUS_REVIEWED,
            analyst_assessment="CONFIRMED",
            note="",
            reviewer_id=uuid.uuid4(),
        )
    )

    with pytest.raises(SQLAlchemyError):
        db.commit()

    db.rollback()


def test_the_database_accepts_a_null_assessment(session):
    """The other half of the constraint, which must permit the state every legacy row is in."""
    analysis = make_analysis(session)
    db, _, _ = session

    db.add(
        AnalysisReview(
            analysis_id=analysis.id,
            status=REVIEW_STATUS_REVIEWED,
            note="",
            reviewer_id=uuid.uuid4(),
        )
    )
    db.commit()

    assert stored_assessment(session, analysis) is None
