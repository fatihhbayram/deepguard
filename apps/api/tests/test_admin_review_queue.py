"""R14-T3: the admin review queue, `GET /api/v1/admin/analyses/queue`.

What is asserted:

*Routing.* `/analyses/queue` reaches the queue and is never parsed as an `{analysis_id}`; the
dynamic review route beside it still works for a real id and still refuses `queue` as one.

*One row per analysis.* An analysis with ten failed signals and three disagreeing feedbacks,
from three different accounts as the one-per-account constraint requires, appears exactly once
and is counted exactly once. Asserted on the statement itself, restricted to the seeded rows, and
through the live route by walking every page.

*Classifier reuse.* The `decision` filter selects rows by `risk_bucket`, the function the
operational summary places analyses with, so a legacy `HIGH` and a legacy row holding a v5
verdict never match a decision, and `UNDECIDED` matches both spellings of "no decision".

*Unreviewed.* `unreviewed_only` (the default) removes an analysis whose review is `REVIEWED`
and keeps one with no review or one marked `NEEDS_FOLLOW_UP`.

*Separation.* The queue reads; it writes nothing, and no evaluator reads it.

The live queue covers every analysis in the database, so route-level assertions are made on the
seeded ids found by walking every page rather than on absolute totals.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api import admin_analyses, admin_analytics
from app.api.admin_analyses import (
    REVIEW_QUEUE_MAX_LIMIT,
    pairs_for_decision,
    review_queue_filters,
)
from app.db.models import (
    REVIEW_STATUS_NEEDS_FOLLOW_UP,
    REVIEW_STATUS_REVIEWED,
    REVIEW_STATUS_UNREVIEWED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    AdminAuditEvent,
    Analysis,
    AnalysisReview,
    AnalysisSignal,
    AuthSession,
    User,
    UserFeedback,
)
from app.db.session import SessionLocal
from app.main import app
from app.risk_engine import VERDICT_NO_SIGNAL
from app.web_auth import hash_password
from tests.test_admin_analytics import (  # noqa: F401
    LEGACY_LAST,
    PASSWORD,
    V5,
    database,
    plain_http_environment,
    signed_in,
)

pytestmark = pytest.mark.integration

QUEUE_URL = "/api/v1/admin/analyses/queue"

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_ROOT = REPO_ROOT / "scripts" / "eval"


@pytest.fixture
def session(database):
    """A real session that removes every row this file created, in constraint order.

    Analyses first: feedback, signals and reviews all cascade from the analysis, and
    `user_feedback.user_id` is `RESTRICT`, so no user can go while their feedback remains.
    """
    users: list[uuid.UUID] = []
    analyses: list[uuid.UUID] = []

    with SessionLocal() as db:
        yield db, users, analyses

        db.rollback()
        for analysis_id in analyses:
            db.query(Analysis).filter(Analysis.id == analysis_id).delete()
        for user_id in users:
            db.query(AuthSession).filter(AuthSession.user_id == user_id).delete()
            db.query(User).filter(User.id == user_id).delete()
        db.commit()


def make_user(session, *, role: str = USER_ROLE_USER) -> User:
    db, users, _ = session

    user = User(
        email=f"{uuid.uuid4().hex}@example.com",
        password_hash=hash_password(PASSWORD),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    users.append(user.id)

    return user


def make_analysis(
    session,
    *,
    risk_level: str | None = None,
    rules_version: str | None = None,
    created_at: datetime | None = None,
) -> Analysis:
    db, _, analyses = session

    analysis = Analysis(
        status="completed", risk_level=risk_level, risk_rules_version=rules_version
    )
    if created_at is not None:
        analysis.created_at = created_at

    db.add(analysis)
    db.commit()
    analyses.append(analysis.id)

    return analysis


def give_feedback(session, analysis: Analysis, *, assessment: str) -> None:
    """One feedback row from a fresh account — one per account per analysis, as the schema
    requires, so several on one analysis means several accounts."""
    db, _, _ = session

    db.add(
        UserFeedback(
            analysis_id=analysis.id,
            user_id=make_user(session).id,
            assessment=assessment,
        )
    )
    db.commit()


def make_signal(session, analysis: Analysis, *, status: str) -> None:
    """One detector's answer, each from its own provider."""
    db, _, _ = session

    db.add(
        AnalysisSignal(
            analysis_id=analysis.id,
            provider=f"test-provider-{uuid.uuid4().hex[:12]}",
            signal_type="deepfake",
            status=status,
        )
    )
    db.commit()


def make_review(session, analysis: Analysis, *, status: str) -> None:
    db, _, _ = session

    db.add(
        AnalysisReview(
            analysis_id=analysis.id,
            status=status,
            note="",
            reviewer_id=uuid.uuid4(),
        )
    )
    db.commit()


def administrator(session) -> TestClient:
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def walk(client: TestClient, **params) -> list[dict]:
    """Every item the queue returns for these filters, page by page."""
    items: list[dict] = []
    offset = 0
    total = None

    while True:
        response = client.get(
            QUEUE_URL, params={**params, "limit": REVIEW_QUEUE_MAX_LIMIT, "offset": offset}
        )
        assert response.status_code == 200, response.text
        page = response.json()

        # The total is the same on every page of one walk.
        assert total is None or page["total"] == total
        total = page["total"]

        items.extend(page["items"])
        offset += REVIEW_QUEUE_MAX_LIMIT
        if offset >= total:
            break

    # The pages add up to the total, with no row on two of them.
    assert len(items) == total
    assert len({item["analysis_id"] for item in items}) == total

    return items


def ids_among(items: list[dict], seeded) -> list[str]:
    """The seeded analyses in a walk, in the order the queue gave them, repeats kept."""
    wanted = {str(analysis.id) for analysis in seeded}
    return [item["analysis_id"] for item in items if item["analysis_id"] in wanted]


# --- authorization and routing --------------------------------------------------------------


def test_an_anonymous_caller_cannot_read_the_queue(database):
    assert TestClient(app).get(QUEUE_URL).status_code == 401


def test_an_ordinary_user_cannot_read_the_queue(session):
    client = signed_in(make_user(session))

    assert client.get(QUEUE_URL).status_code == 403


def test_the_queue_path_is_not_captured_by_the_analysis_id_routes(session):
    """`queue` is not a UUID: had a dynamic route captured it, this would be a 422 about
    `analysis_id` or a 404/405, never a page."""
    client = administrator(session)

    response = client.get(QUEUE_URL)

    assert response.status_code == 200, response.text
    assert set(response.json()) == {"items", "total", "limit", "offset"}

    # And the dynamic routes are still there and still dynamic: a real id reaches the review,
    # and `queue` in the id's place is refused as a malformed id rather than served.
    analysis = make_analysis(session)
    review = client.get(f"/api/v1/admin/analyses/{analysis.id}/review")
    assert review.status_code == 200
    assert review.json()["status"] == REVIEW_STATUS_UNREVIEWED

    captured = client.get("/api/v1/admin/analyses/queue/review")
    assert captured.status_code == 422
    assert captured.json()["detail"][0]["loc"] == ["path", "analysis_id"]


def test_an_unknown_filter_value_is_refused(session):
    client = administrator(session)

    for params in (
        {"decision": "HIGH"},
        {"decision": "NEEDS_REVIEW"},
        {"feedback_assessment": "WRONG"},
        {"limit": 0},
        {"limit": REVIEW_QUEUE_MAX_LIMIT + 1},
        {"offset": -1},
    ):
        assert client.get(QUEUE_URL, params=params).status_code == 422, params


# --- one row per analysis ---------------------------------------------------------------------


@pytest.fixture
def crowded(session):
    """One analysis with ten failed signals and three disagreeing feedbacks from three
    accounts, beside a control analysis with neither."""
    crowded = make_analysis(session, risk_level="MANIPULATION_DETECTED", rules_version=V5)
    for _ in range(10):
        make_signal(session, crowded, status=SIGNAL_STATUS_FAILED)
    for _ in range(3):
        give_feedback(session, crowded, assessment="DISAGREE")

    control = make_analysis(session, risk_level="MANIPULATION_DETECTED", rules_version=V5)
    make_signal(session, control, status=SIGNAL_STATUS_SUCCESS)

    return crowded, control


def test_the_statement_counts_and_pages_one_row_per_analysis(session, crowded):
    """On the statement the route runs, restricted to the two seeded analyses."""
    db, _, _ = session
    crowded_analysis, control = crowded
    seeded = Analysis.id.in_([crowded_analysis.id, control.id])

    clauses = review_queue_filters(
        pairs_for_decision(db, "MANIPULATION_DETECTED"),
        feedback_assessment="DISAGREE",
        signal_error=True,
        unreviewed_only=True,
    )

    count = select(func.count()).select_from(Analysis).where(*clauses, seeded)
    rows = select(Analysis.id).where(*clauses, seeded).limit(10)

    assert db.execute(count).scalar_one() == 1
    assert db.execute(rows).scalars().all() == [crowded_analysis.id]

    # No join to undo and nothing to de-duplicate: the statement is over `analyses` alone.
    compiled = str(select(Analysis.id).where(*clauses)).upper()
    # (`IS NOT DISTINCT FROM` is the null-safe pair match, not a de-duplication.)
    assert " JOIN " not in compiled
    assert "SELECT DISTINCT" not in compiled
    assert "EXISTS" in compiled


def test_the_route_lists_a_crowded_analysis_exactly_once(session, crowded):
    crowded_analysis, _ = crowded
    client = administrator(session)

    items = walk(client, feedback_assessment="DISAGREE", signal_error="true")
    assert ids_among(items, crowded) == [str(crowded_analysis.id)]

    # Unfiltered, too: still once.
    assert ids_among(walk(client, unreviewed_only="false"), crowded).count(
        str(crowded_analysis.id)
    ) == 1

    # The item carries counts, not a feedback record.
    item = next(i for i in items if i["analysis_id"] == str(crowded_analysis.id))
    assert item["feedback_counts"] == {"AGREE": 0, "DISAGREE": 3, "UNSURE": 0}
    assert item["has_disagree_feedback"] is True
    assert item["signal_error_counts"] == {SIGNAL_STATUS_FAILED: 10, SIGNAL_STATUS_TIMEOUT: 0}
    assert item["has_signal_errors"] is True
    assert set(item) == {
        "analysis_id",
        "created_at",
        "status",
        "media_sha256s",
        "risk_rules_version",
        "decision",
        "recorded_risk_level",
        "unrecognised_risk_state",
        "review_status",
        "analyst_assessment",
        "feedback_counts",
        "has_disagree_feedback",
        "signal_error_counts",
        "has_signal_errors",
    }


def test_the_total_moves_by_one_for_a_crowded_analysis(session):
    client = administrator(session)
    filters = {"feedback_assessment": "DISAGREE", "signal_error": "true"}
    before = client.get(QUEUE_URL, params=filters).json()["total"]

    analysis = make_analysis(session, risk_level="INCONCLUSIVE", rules_version=V5)
    for _ in range(10):
        make_signal(session, analysis, status=SIGNAL_STATUS_TIMEOUT)
    for _ in range(3):
        give_feedback(session, analysis, assessment="DISAGREE")

    after = client.get(QUEUE_URL, params=filters).json()["total"]

    assert after == before + 1


def test_pages_are_newest_first_and_do_not_overlap(session):
    # Dated a year ahead, so they are the three newest rows in any database.
    ahead = datetime.now(timezone.utc) + timedelta(days=365)
    seeded = [
        make_analysis(session, created_at=ahead + timedelta(seconds=offset))
        for offset in range(3)
    ]
    client = administrator(session)

    pages = [
        client.get(QUEUE_URL, params={"limit": 1, "offset": offset}).json()["items"]
        for offset in range(3)
    ]

    assert [page[0]["analysis_id"] for page in pages] == [
        str(analysis.id) for analysis in reversed(seeded)
    ]


# --- filters ------------------------------------------------------------------------------------


def test_the_feedback_filter_matches_only_that_assessment(session):
    agreed = make_analysis(session)
    give_feedback(session, agreed, assessment="AGREE")
    disagreed = make_analysis(session)
    give_feedback(session, disagreed, assessment="AGREE")
    give_feedback(session, disagreed, assessment="DISAGREE")
    silent = make_analysis(session)
    seeded = (agreed, disagreed, silent)

    client = administrator(session)

    assert ids_among(walk(client, feedback_assessment="DISAGREE"), seeded) == [
        str(disagreed.id)
    ]
    assert set(ids_among(walk(client, feedback_assessment="AGREE"), seeded)) == {
        str(agreed.id),
        str(disagreed.id),
    }
    assert set(ids_among(walk(client), seeded)) == {str(a.id) for a in seeded}


def test_the_signal_error_filter_matches_failed_and_timed_out_only(session):
    failed = make_analysis(session)
    make_signal(session, failed, status=SIGNAL_STATUS_FAILED)
    timed_out = make_analysis(session)
    make_signal(session, timed_out, status=SIGNAL_STATUS_TIMEOUT)
    succeeded = make_analysis(session)
    make_signal(session, succeeded, status=SIGNAL_STATUS_SUCCESS)
    seeded = (failed, timed_out, succeeded)

    client = administrator(session)

    assert set(ids_among(walk(client, signal_error="true"), seeded)) == {
        str(failed.id),
        str(timed_out.id),
    }


def test_unreviewed_only_excludes_reviewed_and_keeps_follow_up(session):
    unreviewed = make_analysis(session)
    reviewed = make_analysis(session)
    make_review(session, reviewed, status=REVIEW_STATUS_REVIEWED)
    follow_up = make_analysis(session)
    make_review(session, follow_up, status=REVIEW_STATUS_NEEDS_FOLLOW_UP)
    seeded = (unreviewed, reviewed, follow_up)

    client = administrator(session)

    # The default is `unreviewed_only=true`.
    assert set(ids_among(walk(client), seeded)) == {str(unreviewed.id), str(follow_up.id)}

    everything = walk(client, unreviewed_only="false")
    by_id = {item["analysis_id"]: item for item in everything}
    assert set(ids_among(everything, seeded)) == {str(a.id) for a in seeded}
    assert by_id[str(unreviewed.id)]["review_status"] == REVIEW_STATUS_UNREVIEWED
    assert by_id[str(reviewed.id)]["review_status"] == REVIEW_STATUS_REVIEWED
    assert by_id[str(follow_up.id)]["review_status"] == REVIEW_STATUS_NEEDS_FOLLOW_UP


# --- classifier reuse ---------------------------------------------------------------------------


@pytest.fixture
def taxonomy(session):
    return {
        "v5_detected": make_analysis(
            session, risk_level="MANIPULATION_DETECTED", rules_version=V5
        ),
        "v5_inconclusive": make_analysis(session, risk_level="INCONCLUSIVE", rules_version=V5),
        "legacy_high": make_analysis(session, risk_level="HIGH", rules_version=LEGACY_LAST),
        # Contradicts its own stamp: a v5 word under a legacy ruleset. Unrecognised, not a
        # decision.
        "legacy_verdict": make_analysis(
            session, risk_level="MANIPULATION_DETECTED", rules_version=LEGACY_LAST
        ),
        "v5_undecided": make_analysis(session, risk_level=None, rules_version=V5),
        "never_decided": make_analysis(session, risk_level=None, rules_version=None),
        "legacy_null": make_analysis(session, risk_level=None, rules_version=LEGACY_LAST),
    }


def test_the_decision_filter_separates_v5_verdicts_from_legacy_levels(session, taxonomy):
    client = administrator(session)
    seeded = taxonomy.values()

    def matched(decision: str) -> set[str]:
        return set(ids_among(walk(client, decision=decision), seeded))

    assert matched("MANIPULATION_DETECTED") == {str(taxonomy["v5_detected"].id)}
    assert matched("INCONCLUSIVE") == {str(taxonomy["v5_inconclusive"].id)}
    assert matched(VERDICT_NO_SIGNAL) == set()
    assert matched("UNDECIDED") == {
        str(taxonomy["v5_undecided"].id),
        str(taxonomy["never_decided"].id),
    }


def test_each_item_carries_the_verdict_in_exactly_one_field(session, taxonomy):
    client = administrator(session)
    by_id = {item["analysis_id"]: item for item in walk(client)}

    def verdict(name: str) -> tuple:
        item = by_id[str(taxonomy[name].id)]
        return (
            item["decision"],
            item["recorded_risk_level"],
            item["unrecognised_risk_state"],
        )

    assert verdict("v5_detected") == ("MANIPULATION_DETECTED", None, None)
    assert verdict("legacy_high") == (None, "HIGH", None)
    assert verdict("legacy_verdict") == (None, None, f"{LEGACY_LAST}/MANIPULATION_DETECTED")
    assert verdict("never_decided") == ("UNDECIDED", None, None)
    assert verdict("legacy_null") == (None, None, f"{LEGACY_LAST}/(none)")


def test_the_decision_filter_is_decided_by_the_shared_classifier(session, taxonomy, monkeypatch):
    """Reuse, not a copy: the queue's classifier is the operational summary's, and replacing it
    changes what the filter matches."""
    assert admin_analyses.risk_bucket is admin_analytics.risk_bucket
    assert admin_analyses.DECISIONS is admin_analytics.DECISIONS

    db, _, _ = session
    assert ("HIGH", LEGACY_LAST) not in pairs_for_decision(db, "MANIPULATION_DETECTED")

    # A classifier that called every legacy HIGH a detection would make the filter match it.
    def reclassify(risk_level, rules_version):
        if (risk_level, rules_version) == ("HIGH", LEGACY_LAST):
            return admin_analytics.BUCKET_DECISIONS, "MANIPULATION_DETECTED"
        return admin_analytics.risk_bucket(risk_level, rules_version)

    monkeypatch.setattr(admin_analyses, "risk_bucket", reclassify)

    assert ("HIGH", LEGACY_LAST) in pairs_for_decision(db, "MANIPULATION_DETECTED")


# --- separation ---------------------------------------------------------------------------------


def test_reading_the_queue_writes_nothing(session, crowded):
    db, _, _ = session
    crowded_analysis, _ = crowded
    client = administrator(session)

    def snapshot():
        db.expire_all()
        analysis = db.get(Analysis, crowded_analysis.id)
        return (
            analysis.status,
            analysis.risk_level,
            analysis.risk_rules_version,
            analysis.risk_calibration_id,
            analysis.risk_rule_id,
            *(
                db.execute(select(func.count()).select_from(table)).scalar_one()
                for table in (AdminAuditEvent, AnalysisReview, UserFeedback, AnalysisSignal)
            ),
        )

    before = snapshot()
    walk(client, unreviewed_only="false")
    walk(client, feedback_assessment="DISAGREE", signal_error="true", decision="UNDECIDED")

    assert snapshot() == before


def test_no_evaluator_reads_the_queue():
    """Queue membership is operational and must not reach FP/FN/TP/TN scoring."""
    for path in EVALUATOR_ROOT.rglob("*.py"):
        text = path.read_text()
        assert "admin_analyses" not in text, path
        assert "analyses/queue" not in text, path
