"""R14-T2: user feedback on the operational summary, and what its numbers are counting.

The feedback half of `/api/v1/admin/analytics`, written to the conventions of
`test_admin_analytics.py` and reusing its client helpers. Every assertion made against the live
route is a **difference** across a seed, for the reason that file's docstring gives: the payload
aggregates every row in the deployment, and absolute numbers would hold alone and fail in a suite.
The contract itself — seeded buckets, a null rate at zero, exact arithmetic — is asserted
absolutely against the pure functions the route is built from.

What is asserted:

*Window.* Feedback is windowed on its own `updated_at`: a record created eight days ago and
changed today is counted, one created eight days ago and left alone is not.

*Arithmetic.* The disagreement rate is `DISAGREE / total_feedback` for its bucket, asserted with
three analyses owned by three different users — one feedback each, as production allows.

*Zero denominators.* Every known decision and recorded risk level is present with its whole shape
even when nobody gave feedback on it: `total_feedback` 0 and `disagreement_rate` null.

*Placement.* Feedback is placed by the verdict on its analysis through `risk_bucket`, the same
function that places the analyses — a legacy `HIGH` lands in `feedback_by_recorded_risk_level`.

*Separation.* No evaluator reads user feedback.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.api import admin_analytics
from app.api.admin_analytics import (
    DECISIONS,
    FEEDBACK_DISAGREE,
    RECORDED_RISK_LEVELS,
    WINDOW_DAYS,
    claimed_label_counts,
    feedback_buckets,
    risk_bucket,
)
from app.db.models import (
    FEEDBACK_ASSESSMENTS,
    FEEDBACK_CLAIMED_LABELS,
    USER_ROLE_ADMIN,
    USER_ROLE_USER,
    Analysis,
    AuthSession,
    User,
    UserFeedback,
)
from app.db.session import SessionLocal
from app.web_auth import hash_password
from tests.test_admin_analytics import (  # noqa: F401
    LEGACY_LAST,
    PASSWORD,
    V5,
    database,
    plain_http_environment,
    read,
    signed_in,
)
from tests.test_operational_summary_taxonomy import (
    PAYLOAD as TAXONOMY_PAYLOAD,
    WEB_ANALYTICS_PAGE,
    parse_with_node,
    requires_node,
    requires_web,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_ROOT = REPO_ROOT / "scripts" / "eval"

FEEDBACK_BUCKETS = (
    "feedback_by_decision",
    "feedback_by_recorded_risk_level",
    "feedback_by_unrecognised_risk_state",
)

EMPTY_BUCKET = {
    "total_feedback": 0,
    "AGREE": 0,
    "DISAGREE": 0,
    "UNSURE": 0,
    "disagreement_rate": None,
}


# --- the contract, on the pure functions ----------------------------------------------------


def test_the_disagreement_spelling_is_one_of_the_schema_assessments():
    assert FEEDBACK_DISAGREE in FEEDBACK_ASSESSMENTS


def test_an_empty_window_seeds_every_known_bucket_with_a_null_rate():
    """The zero-denominator contract, exactly: every known key, its whole shape, a null rate."""
    decisions, recorded, unrecognised = feedback_buckets([])

    assert decisions == {name: EMPTY_BUCKET for name in DECISIONS}
    assert recorded == {level: EMPTY_BUCKET for level in RECORDED_RISK_LEVELS}
    assert unrecognised == {}


def test_the_rate_is_disagree_over_the_buckets_own_feedback():
    decisions, recorded, _ = feedback_buckets(
        [
            ("MANIPULATION_DETECTED", V5, "DISAGREE", 2),
            ("MANIPULATION_DETECTED", V5, "AGREE", 1),
            ("HIGH", LEGACY_LAST, "UNSURE", 4),
        ]
    )

    assert decisions["MANIPULATION_DETECTED"] == {
        "total_feedback": 3,
        "AGREE": 1,
        "DISAGREE": 2,
        "UNSURE": 0,
        "disagreement_rate": 2 / 3,
    }
    # Another bucket's feedback is not in this bucket's denominator, and vice versa.
    assert recorded["HIGH"]["total_feedback"] == 4
    assert recorded["HIGH"]["disagreement_rate"] == 0
    assert decisions["INCONCLUSIVE"] == EMPTY_BUCKET


def test_feedback_is_placed_by_the_same_function_as_the_analyses(monkeypatch):
    """Reuse, not a copy of the rules: replace `risk_bucket` and the feedback follows it."""
    calls = []

    def spy(risk_level, rules_version):
        calls.append((risk_level, rules_version))
        return risk_bucket(risk_level, rules_version)

    monkeypatch.setattr(admin_analytics, "risk_bucket", spy)

    feedback_buckets([("HIGH", LEGACY_LAST, "AGREE", 1)])
    admin_analytics.risk_buckets([("HIGH", LEGACY_LAST, 1)])

    assert calls == [("HIGH", LEGACY_LAST), ("HIGH", LEGACY_LAST)]


def test_an_unrecognised_pairing_gets_its_own_bucket_only_when_it_has_feedback():
    _, _, unrecognised = feedback_buckets([("MEDIUM", V5, "DISAGREE", 1)])

    assert unrecognised == {
        f"{V5}/MEDIUM": {
            **EMPTY_BUCKET,
            "total_feedback": 1,
            "DISAGREE": 1,
            "disagreement_rate": 1.0,
        }
    }


def test_a_missing_claimed_label_is_a_count_and_not_a_key():
    labels, without = claimed_label_counts([(None, 5), ("GENUINE", 2)])

    assert labels == {
        label: (2 if label == "GENUINE" else 0) for label in FEEDBACK_CLAIMED_LABELS
    }
    assert without == 5
    assert None not in labels and "null" not in labels and "None" not in labels


# --- the live route -------------------------------------------------------------------------


@pytest.fixture
def session(database):
    """A real session that removes every row this file created, in constraint order.

    Analyses first: `user_feedback.analysis_id` cascades, and both `analyses.owner_id` and
    `user_feedback.user_id` are `RESTRICT`, so no user can go while an analysis or feedback of
    theirs remains. Then sessions, then users.
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


def make_owned_analysis(
    session,
    *,
    risk_level: str | None,
    rules_version: str | None,
    created_at: datetime | None = None,
) -> tuple[Analysis, User]:
    """An analysis owned by a fresh user — the only person who can give feedback on it."""
    db, _, analyses = session
    owner = make_user(session)

    analysis = Analysis(
        status="completed",
        risk_level=risk_level,
        risk_rules_version=rules_version,
        owner_id=owner.id,
    )
    if created_at is not None:
        analysis.created_at = created_at

    db.add(analysis)
    db.commit()
    analyses.append(analysis.id)

    return analysis, owner


def give_feedback(
    session,
    analysis: Analysis,
    owner: User,
    *,
    assessment: str,
    claimed_label: str | None = None,
    written_at: datetime | None = None,
) -> UserFeedback:
    """The owner's one feedback row on their analysis, optionally backdated on both timestamps."""
    db, _, _ = session

    feedback = UserFeedback(
        analysis_id=analysis.id,
        user_id=owner.id,
        assessment=assessment,
        claimed_label=claimed_label,
    )
    if written_at is not None:
        feedback.created_at = written_at
        feedback.updated_at = written_at

    db.add(feedback)
    db.commit()

    return feedback


def feedback_moved(before: dict, after: dict) -> dict[tuple[str, str, str], int]:
    """Every feedback-bucket count that changed across a seed, and by how much."""
    changes = {}
    for bucket in FEEDBACK_BUCKETS:
        for key in set(before[bucket]) | set(after[bucket]):
            old = before[bucket].get(key, EMPTY_BUCKET)
            new = after[bucket].get(key, EMPTY_BUCKET)
            for field in ("total_feedback", *FEEDBACK_ASSESSMENTS):
                delta = new.get(field, 0) - old.get(field, 0)
                if delta:
                    changes[(bucket, key, field)] = delta
    return changes


def administrator(session):
    return signed_in(make_user(session, role=USER_ROLE_ADMIN))


def test_feedback_created_long_ago_and_changed_today_is_counted(session):
    """The window is the feedback's `updated_at`. The analysis and the feedback are both eight
    days old, and a genuine revision through the ORM brings the feedback — only — into it."""
    outside = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS + 1)
    analysis, owner = make_owned_analysis(
        session, risk_level="MANIPULATION_DETECTED", rules_version=V5, created_at=outside
    )
    feedback = give_feedback(session, analysis, owner, assessment="AGREE", written_at=outside)

    client = administrator(session)
    before = read(client)

    db, _, _ = session
    feedback.assessment = "DISAGREE"
    db.commit()

    after = read(client)

    assert after["feedback_total"] == before["feedback_total"] + 1
    assert after["feedback_by_assessment"]["DISAGREE"] == (
        before["feedback_by_assessment"]["DISAGREE"] + 1
    )
    assert feedback_moved(before, after) == {
        ("feedback_by_decision", "MANIPULATION_DETECTED", "total_feedback"): 1,
        ("feedback_by_decision", "MANIPULATION_DETECTED", "DISAGREE"): 1,
    }
    # The analysis itself is still outside its own window.
    assert after["analyses_total"] == before["analyses_total"]


def test_feedback_created_long_ago_and_left_alone_is_not_counted(session):
    outside = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS + 1)
    client = administrator(session)
    before = read(client)

    analysis, owner = make_owned_analysis(
        session, risk_level="MANIPULATION_DETECTED", rules_version=V5
    )
    give_feedback(session, analysis, owner, assessment="DISAGREE", written_at=outside)

    after = read(client)

    assert after["feedback_total"] == before["feedback_total"]
    assert after["feedback_by_assessment"] == before["feedback_by_assessment"]
    assert feedback_moved(before, after) == {}


def test_disagreement_rate_across_three_owners_of_three_analyses(session):
    """Production-realistic: one feedback per analysis, each from that analysis's own owner."""
    client = administrator(session)
    before = read(client)

    for assessment in ("DISAGREE", "DISAGREE", "AGREE"):
        analysis, owner = make_owned_analysis(
            session, risk_level="MANIPULATION_DETECTED", rules_version=V5
        )
        give_feedback(session, analysis, owner, assessment=assessment)

    after = read(client)
    old = before["feedback_by_decision"]["MANIPULATION_DETECTED"]
    new = after["feedback_by_decision"]["MANIPULATION_DETECTED"]

    assert new["total_feedback"] == old["total_feedback"] + 3
    assert new["DISAGREE"] == old["DISAGREE"] + 2
    assert new["AGREE"] == old["AGREE"] + 1
    assert new["disagreement_rate"] == new["DISAGREE"] / new["total_feedback"]
    # On a database with no other feedback on this verdict in the window — the test database —
    # that is exactly two in three.
    if old["total_feedback"] == 0:
        assert new["disagreement_rate"] == pytest.approx(0.666, abs=1e-3)


def test_every_known_bucket_is_on_the_wire_with_its_whole_shape(session):
    """The zero-denominator contract, on the live payload: every canonical key present, and
    every bucket nobody gave feedback on reads 0 with a null rate rather than being omitted."""
    payload = read(administrator(session))

    assert set(payload["feedback_by_decision"]) >= set(DECISIONS)
    assert set(payload["feedback_by_recorded_risk_level"]) >= set(RECORDED_RISK_LEVELS)

    empty_seen = 0
    for field in FEEDBACK_BUCKETS:
        for key, bucket in payload[field].items():
            assert set(bucket) >= {"total_feedback", "disagreement_rate", *FEEDBACK_ASSESSMENTS}
            if bucket["total_feedback"] == 0:
                assert bucket["disagreement_rate"] is None, (field, key)
                empty_seen += 1
            else:
                assert bucket["disagreement_rate"] == (
                    bucket[FEEDBACK_DISAGREE] / bucket["total_feedback"]
                )
    # Seven canonical buckets; a window with feedback on every one of them is not this suite's.
    assert empty_seen > 0


def test_feedback_on_a_legacy_high_is_a_recorded_risk_level(session):
    client = administrator(session)
    before = read(client)

    analysis, owner = make_owned_analysis(session, risk_level="HIGH", rules_version=LEGACY_LAST)
    give_feedback(session, analysis, owner, assessment="UNSURE")

    after = read(client)

    assert feedback_moved(before, after) == {
        ("feedback_by_recorded_risk_level", "HIGH", "total_feedback"): 1,
        ("feedback_by_recorded_risk_level", "HIGH", "UNSURE"): 1,
    }


def test_a_missing_claimed_label_is_counted_without_a_null_key(session):
    client = administrator(session)
    before = read(client)

    analysis, owner = make_owned_analysis(
        session, risk_level="MANIPULATION_DETECTED", rules_version=V5
    )
    give_feedback(session, analysis, owner, assessment="AGREE", claimed_label=None)

    after = read(client)

    assert after["feedback_without_claimed_label"] == (
        before["feedback_without_claimed_label"] + 1
    )
    assert after["feedback_by_claimed_label"] == before["feedback_by_claimed_label"]
    assert set(after["feedback_by_claimed_label"]) == set(FEEDBACK_CLAIMED_LABELS)


# --- the web half of the contract -----------------------------------------------------------


def with_feedback(**fields) -> dict:
    return {**TAXONOMY_PAYLOAD, **fields}


WEB_BUCKETS = {
    "MANIPULATION_DETECTED": {
        "total_feedback": 3,
        "AGREE": 1,
        "DISAGREE": 2,
        "UNSURE": 0,
        "disagreement_rate": 2 / 3,
    },
    "INCONCLUSIVE": EMPTY_BUCKET,
}

WEB_CASES = {
    "full": with_feedback(
        feedback_total=3,
        feedback_by_assessment={"AGREE": 1, "DISAGREE": 2, "UNSURE": 0},
        feedback_by_claimed_label={"GENUINE": 2},
        feedback_without_claimed_label=1,
        feedback_by_decision=WEB_BUCKETS,
    ),
    "rate_on_an_empty_bucket": with_feedback(
        feedback_by_decision={"INCONCLUSIVE": {**EMPTY_BUCKET, "disagreement_rate": 0}}
    ),
    "null_rate_on_a_non_empty_bucket": with_feedback(
        feedback_by_decision={
            "MANIPULATION_DETECTED": {
                **WEB_BUCKETS["MANIPULATION_DETECTED"],
                "disagreement_rate": None,
            }
        }
    ),
    "rate_above_one": with_feedback(
        feedback_by_decision={
            "MANIPULATION_DETECTED": {
                **WEB_BUCKETS["MANIPULATION_DETECTED"],
                "disagreement_rate": 2,
            }
        }
    ),
    "missing_without_claimed_label": {
        k: v for k, v in TAXONOMY_PAYLOAD.items() if k != "feedback_without_claimed_label"
    },
}


@requires_node
def test_the_web_parses_feedback_buckets_as_the_api_sent_them():
    parsed = parse_with_node(WEB_CASES)

    full = parsed["full"]
    assert full["feedback_total"] == 3
    assert full["feedback_without_claimed_label"] == 1
    assert full["feedback_by_decision"]["MANIPULATION_DETECTED"] == {
        "total_feedback": 3,
        "disagreement_rate": 2 / 3,
        "assessments": {"AGREE": 1, "DISAGREE": 2, "UNSURE": 0},
    }
    assert full["feedback_by_decision"]["INCONCLUSIVE"]["disagreement_rate"] is None

    for case in (
        "rate_on_an_empty_bucket",
        "null_rate_on_a_non_empty_bucket",
        "rate_above_one",
        "missing_without_claimed_label",
    ):
        assert parsed[case] is None, case


@requires_web
def test_the_feedback_section_carries_its_disclaimers_and_stays_apart():
    source = WEB_ANALYTICS_PAGE.read_text(encoding="utf-8")
    section = source.split("function FeedbackActivity(", 1)[1].split("\nfunction ", 1)[0]
    flat = " ".join(section.split())

    assert "User feedback activity" in section
    assert (
        "This reflects the current state of user feedback modified in the last 7 days, not a"
        " submission count or event log." in flat
    )
    assert (
        "User feedback represents unverified end-user opinions. Disagreement rate is not a"
        " False Positive or Error rate." in flat
    )
    # Drawn after the detector table, as its own section — never inside the risk distributions.
    summary = source.split("function Summary(", 1)[1]
    assert summary.index("<Detectors") < summary.index("<FeedbackActivity")
    # No evaluation vocabulary is used to label it, and the page computes no rate of its own.
    for term in ("FPR", "TPR", "Accuracy", "Precision", "Recall", "Confusion"):
        assert term not in section, term
    assert "DISAGREE /" not in source and "/ total_feedback" not in source


# --- separation from evaluation -------------------------------------------------------------


def test_no_evaluator_reads_user_feedback():
    """Feedback is opinion, and the evaluators score detectors against Ground Truth only. No
    evaluation script may name the table or the model."""
    sources = sorted(EVALUATOR_ROOT.rglob("*.py"))
    assert (EVALUATOR_ROOT / "operational_metrics.py") in sources

    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "user_feedback" not in text, path
        assert "UserFeedback" not in text, path


def test_the_analytics_route_does_not_reach_into_evaluation_or_ground_truth():
    """The dependency runs one way: the summary reads feedback, and nothing about evaluation or
    Ground Truth is imported alongside it."""
    imports = [
        line
        for line in Path(admin_analytics.__file__).read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    ]

    for line in imports:
        assert "eval" not in line, line
        assert "operational_metrics" not in line, line
        assert "ground_truth" not in line, line
    assert "GroundTruth" not in Path(admin_analytics.__file__).read_text(encoding="utf-8")
