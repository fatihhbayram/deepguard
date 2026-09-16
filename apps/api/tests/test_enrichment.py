"""Deep Evidence Enrichment, and the boundary it may not cross (R10-T2).

What this module is for is the *separation*, not the detectors. `tests/test_lip_forensics.py`
and the modules beside it already prove what each evidence-only detector does; nothing here
re-checks a score. These tests check the four claims R10 rests on, and every one of them is a
claim about what enrichment *cannot* do:

- it cannot write a decision field, `analyses.status`, the deciding detectors' evidence, the
  media identity or a review — at the persistence boundary, and again at the database;
- it cannot delay the Fast Decision Path, because a decision job is claimed on every poll
  before enrichment is looked for;
- it cannot invalidate a decision by failing, crashing or being retried;
- it cannot make a legacy analysis look like an enriched one, or an unenriched one.

The state projections are checked as pure functions because that is what they are, and the
claiming and guarding are checked against real PostgreSQL because locking and triggers cannot
be demonstrated against anything else.
"""

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DatabaseError, SQLAlchemyError

from app import detection, enrichment, risk_engine, worker
from app.db.models import (
    Analysis,
    AnalysisEnrichmentTask,
    AnalysisJob,
    AnalysisReview,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import SessionLocal, engine
from app.enrichment_guard import ProtectedWriteRejected, enrichment_session

RULES_V5 = "r9-v5.0.0"

# The two detectors that decide under `r9-v5.0.0`. Named here so the tests that prove the
# guard refuses them do not have to agree with the guard by importing the same tuple it uses —
# a check that shares its subject's constant proves less than one that restates it.
SVD_COMPONENT = ("nvidia", "synthetic_video")
B7_COMPONENT = ("efficientnet-b7", "face_manipulation")

LIP_COMPONENT = ("lipforensics", "lip_forensics")
AASIST_COMPONENT = ("aasist", "audio_authenticity")

ALLOWED = frozenset(enrichment.EVIDENCE_ONLY_COMPONENTS[RULES_V5])


# --------------------------------------------------------------------------------------
# Membership: derived from the ruleset, not held as a list that can drift from it
# --------------------------------------------------------------------------------------


def test_the_declared_rulesets_are_the_rulesets_the_engine_decides_under():
    """The transcribed key has to name a ruleset the engine actually has.

    `app.enrichment` writes the version as a literal rather than importing it, because §7.2
    forbids the enrichment path any reference to the risk engine. This test is what makes that
    literal safe: a test may import both, because a test decides nothing about anybody's media.
    """
    assert risk_engine.RULES_VERSION_V5 in enrichment.EVIDENCE_ONLY_COMPONENTS


def test_the_deciding_detectors_are_never_deep_evidence():
    """Membership and the decision are disjoint, under every declared ruleset.

    The single most dangerous defect available here: a component that is both decided-from and
    enriched would let the enrichment path rewrite the evidence a verdict was taken from, and
    every other guard in this module is downstream of that not happening.
    """
    for version, components in enrichment.EVIDENCE_ONLY_COMPONENTS.items():
        assert SVD_COMPONENT not in components, version
        assert B7_COMPONENT not in components, version


def test_lipforensics_is_deep_evidence_under_the_live_ruleset():
    """R9-T1 made it `evidence_only`, so R10 runs it off the fast path (§4.1)."""
    assert LIP_COMPONENT in enrichment.evidence_only_components(RULES_V5)


def test_a_ruleset_with_no_declared_roles_is_refused_rather_than_guessed():
    """An undeclared ruleset raises instead of defaulting to a membership nobody assigned.

    A default would be wrong quietly — either enriching under roles the ruleset never made, or
    silently enriching nothing — and quiet is the one thing a role assignment must not be.
    """
    with pytest.raises(enrichment.UnknownRuleset):
        enrichment.evidence_only_components("r11-v6.0.0")


def test_the_enrichment_path_holds_no_reference_to_the_risk_engine():
    """Invariant I4, asserted over the modules rather than trusted (§7.2).

    "Not `evaluate_v5`, not any other entry point. Not to confirm the verdict, not to log a
    comparison, not behind a feature flag, not in shadow mode against the persisted value."
    The absence is the enforcement, and an absence is invisible to whoever later adds a call —
    so it is asserted here, in the shape `tests/test_shadow_mode.py` already uses to assert the
    shadow isolation.

    Over the parsed syntax tree rather than over the text, because both modules *discuss* the
    risk engine at length in their docstrings — the prohibition is most of what they are for,
    and a check that could not tell an explanation from a call would have forced those modules
    to stop explaining themselves.
    """
    import ast
    import inspect

    from app import enrichment_guard

    for module in (enrichment, enrichment_guard):
        tree = ast.parse(inspect.getsource(module))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "risk_engine" not in alias.name, module.__name__
            elif isinstance(node, ast.ImportFrom):
                assert "risk_engine" not in (node.module or ""), module.__name__
                for alias in node.names:
                    assert alias.name != "evaluate_v5", module.__name__
            elif isinstance(node, ast.Name):
                assert node.id not in {"risk_engine", "evaluate_v5"}, module.__name__
            elif isinstance(node, ast.Attribute):
                assert node.attr not in {"risk_engine", "evaluate_v5"}, module.__name__


# --------------------------------------------------------------------------------------
# The trigger: requested, deferred, and the difference between deferred and legacy
# --------------------------------------------------------------------------------------


def test_the_default_policy_queues_enrichment_with_the_decision(monkeypatch):
    """`immediate` unless a deployment says otherwise.

    The default is R10's compatibility story: the same six signals are produced for the same
    media as before this task, and only the waiting changed. A default of `deferred` would
    have shipped a latency change as a silent reduction in what a report contains.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    assert enrichment.policy() == enrichment.POLICY_IMMEDIATE


def test_a_deployment_may_defer_enrichment(monkeypatch):
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, "deferred")
    assert enrichment.policy() == enrichment.POLICY_DEFERRED


def test_an_unreadable_policy_enriches_rather_than_going_quiet(monkeypatch):
    """The fail-open direction, which is deliberate and is the opposite of a security switch.

    A misspelled value here costs a report nothing; a misspelled value that silently stopped
    enriching would cost every report its supplementary evidence with nothing saying so.
    """
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, "deffered")
    assert enrichment.policy() == enrichment.POLICY_IMMEDIATE


# --------------------------------------------------------------------------------------
# The two axes, as pure projections (§5)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,risk_level,expected",
    [
        ("completed", "MANIPULATION_DETECTED", enrichment.DECIDED),
        # A verdict, not a decision failure. Conflating the two would let a real
        # classification be presented as a system fault (§5.1).
        ("completed", "INCONCLUSIVE", enrichment.DECIDED),
        ("failed", None, enrichment.DECISION_FAILED),
        ("queued", None, enrichment.DECISION_PENDING),
    ],
)
def test_the_decision_axis_is_the_status_column_it_always_was(status, risk_level, expected):
    """§5.4: every existing reader keeps its current meaning for free."""
    assert enrichment.decision_state(status, risk_level) == expected


def test_an_analysis_with_no_task_rows_is_legacy_and_not_enriched():
    """§8.1, and the distinction the whole `not_requested` status exists to preserve.

    A legacy `completed` row is not `ENRICHMENT_COMPLETE`. That would assert every Deep
    Evidence component terminated *and* succeeded, which no pre-R10 row can support — one
    could always carry a failed evidence-only signal and still reach `completed`.
    """
    assert (
        enrichment.enrichment_state(enrichment.DECIDED, [])
        == enrichment.LEGACY_SINGLE_STAGE
    )


def test_deferred_enrichment_is_distinguishable_from_a_legacy_analysis():
    """The two look identical unless deferral is a row, which is why it is one."""
    deferred = enrichment.enrichment_state(enrichment.DECIDED, ["not_requested"] * 4)

    assert deferred == enrichment.ENRICHMENT_NOT_REQUESTED
    assert deferred != enrichment.LEGACY_SINGLE_STAGE


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["queued", "queued"], enrichment.ENRICHMENT_PENDING),
        # An abstention is terminal and is a success, so an analysis whose components all
        # abstained is fully enriched — there was nothing more to get (§5.2).
        (["abstained", "abstained"], enrichment.ENRICHMENT_COMPLETE),
        (["completed", "abstained"], enrichment.ENRICHMENT_COMPLETE),
        # And it does not drag a genuinely failed component up with it.
        (["abstained", "failed"], enrichment.ENRICHMENT_PARTIAL),
        # Nor is it itself dragged down: one abstention among failures is still a success, so
        # this is partial rather than `ENRICHMENT_FAILED`.
        (["abstained", "failed", "failed"], enrichment.ENRICHMENT_PARTIAL),
        # Still owed, because an abstention terminates and a queued component does not.
        (["abstained", "queued"], enrichment.ENRICHMENT_PENDING),
        (["processing", "queued"], enrichment.ENRICHMENT_PROCESSING),
        (["completed", "processing"], enrichment.ENRICHMENT_PROCESSING),
        (["completed", "completed"], enrichment.ENRICHMENT_COMPLETE),
        # §7.4's worked example: LipForensics failed, AASIST succeeded. The verdict stands,
        # unchanged and unqualified, and the report says plainly what is unavailable.
        (["failed", "completed"], enrichment.ENRICHMENT_PARTIAL),
        (["failed", "failed"], enrichment.ENRICHMENT_FAILED),
        # Asked for in part and deferred in part: something is still owed, so neither a
        # terminal state nor `NOT_REQUESTED` would be true.
        (["completed", "not_requested"], enrichment.ENRICHMENT_PENDING),
    ],
)
def test_the_enrichment_axis_is_derived_from_the_component_rows(statuses, expected):
    """§5.4: derived, so `ENRICHMENT_PARTIAL` can never assert what the rows do not say."""
    assert enrichment.enrichment_state(enrichment.DECIDED, statuses) == expected


@pytest.mark.parametrize(
    "decision", [enrichment.DECISION_FAILED, enrichment.DECISION_PENDING]
)
def test_an_undecided_analysis_has_no_enrichment_reading(decision):
    """§8.1 gives `failed` and `queued` rows no enrichment axis at all.

    An analysis that never got a verdict has nothing for supplementary evidence to supplement,
    and `ENRICHMENT_FAILED` beside a decision that never happened would invite reading the two
    as one outcome.
    """
    assert (
        enrichment.enrichment_state(decision, ["failed", "failed"])
        == enrichment.ENRICHMENT_NOT_APPLICABLE
    )


# --------------------------------------------------------------------------------------
# Abstention is terminal and is not a failure (§5.2)
# --------------------------------------------------------------------------------------


def test_the_abstention_identities_are_bound_to_the_exception_classes():
    """The distinction is drawn from what `app.detection` already persists, not from literals.

    `ABSTENTION_ERRORS` holds `__name__`s taken off the real exception classes, so a rename
    follows automatically and a deletion is an ImportError rather than a set that silently
    stops matching. This test states which three they are, because the *membership* is the
    forensic judgement and it should not be possible to widen it without a test changing.
    """
    from app.effort import EffortNoFaceDetected
    from app.lip_forensics import LipForensicsNoTrackedFace
    from app.speaker_diarization import SpeakerDiarizationAudioError

    assert detection.ABSTENTION_ERRORS == {
        LipForensicsNoTrackedFace.__name__,
        EffortNoFaceDetected.__name__,
        SpeakerDiarizationAudioError.__name__,
    }


def test_a_documented_abstention_is_terminal_but_not_a_failure():
    """§5.2: "It was asked, it answered, and its answer is evidence."

    Checked for each of the three abstentions the pipeline can produce, because they arrive by
    three different routes — no trackable face, no face at all, and no audio stream.
    """
    from app.effort import EffortNoFaceDetected
    from app.lip_forensics import LipForensicsNoTrackedFace
    from app.speaker_diarization import SpeakerDiarizationAudioError

    for error in (
        LipForensicsNoTrackedFace,
        EffortNoFaceDetected,
        SpeakerDiarizationAudioError,
    ):
        signal = AnalysisSignal(
            provider=LIP_COMPONENT[0],
            signal_type=LIP_COMPONENT[1],
            status="FAILED",
            signal_metadata={"error": error.__name__},
        )

        status, reason = enrichment._component_outcome(signal)

        assert status == "abstained", error.__name__
        assert status != "failed", error.__name__
        # The reason travels with it: "why is there no lip evidence on this report" is a
        # question an operator asks of an abstention too.
        assert reason == error.__name__


def test_a_real_detector_failure_is_still_a_failure():
    """The other side of the same mapping, and the one that must not widen.

    A missing checkpoint is a fact about this deployment, not about the media, and an
    enrichment axis that reported it as a success would be claiming evidence nobody has.
    """
    from app.effort import EffortModelUnavailable
    from app.lip_forensics import LipForensicsModelUnavailable

    for error in (LipForensicsModelUnavailable, EffortModelUnavailable):
        signal = AnalysisSignal(
            provider=LIP_COMPONENT[0],
            signal_type=LIP_COMPONENT[1],
            status="FAILED",
            signal_metadata={"error": error.__name__},
        )

        assert enrichment._component_outcome(signal) == ("failed", error.__name__)


def test_a_timeout_is_a_failure_and_can_never_abstain():
    """A timeout describes this worker, not the media, and is stated rather than inferred.

    The guard is explicit in `_component_outcome` because the failure mode it prevents is
    quiet: a `TIMEOUT` row that happened to carry an abstention-shaped metadata key would
    otherwise report this machine's load to an analyst as a fact about someone's video.
    """
    from app.lip_forensics import LipForensicsNoTrackedFace

    timed_out = AnalysisSignal(
        provider=LIP_COMPONENT[0],
        signal_type=LIP_COMPONENT[1],
        status="TIMEOUT",
        signal_metadata={"error": "NvidiaProviderTimeout"},
    )
    assert enrichment._component_outcome(timed_out) == ("failed", "NvidiaProviderTimeout")

    # Even wearing an abstention's name, which is the case the status check exists for.
    disguised = AnalysisSignal(
        provider=LIP_COMPONENT[0],
        signal_type=LIP_COMPONENT[1],
        status="TIMEOUT",
        signal_metadata={"error": LipForensicsNoTrackedFace.__name__},
    )
    assert disguised.status == "TIMEOUT"
    assert enrichment._component_outcome(disguised)[0] == "failed"


def test_a_successful_reading_is_never_downgraded_to_an_abstention():
    """Only a `FAILED` signal can abstain.

    A `SUCCESS` row carrying an abstention-shaped key is a defect in a detector, and it is read
    as the reading it claims to be rather than quietly reinterpreted here.
    """
    from app.lip_forensics import LipForensicsNoTrackedFace

    scored = AnalysisSignal(
        provider=LIP_COMPONENT[0],
        signal_type=LIP_COMPONENT[1],
        status="SUCCESS",
        score=0.4,
        signal_metadata={"error": LipForensicsNoTrackedFace.__name__},
    )

    assert enrichment._component_outcome(scored) == ("completed", None)


def test_an_abstention_changes_nothing_about_the_evidence_or_the_verdict():
    """The mapping reads the signal row; it does not touch it.

    R9's coverage arithmetic treats an abstention as unresolved coverage on purpose — a
    detector that abstains on difficult media must not *buy* complete coverage by declining to
    look — and R10 does not get to soften that by reading the same row differently for a
    different question.
    """
    from app.lip_forensics import LipForensicsNoTrackedFace

    signal = AnalysisSignal(
        provider=LIP_COMPONENT[0],
        signal_type=LIP_COMPONENT[1],
        status="FAILED",
        score=None,
        signal_metadata={"error": LipForensicsNoTrackedFace.__name__},
    )
    before = (signal.status, signal.score, dict(signal.signal_metadata), signal.risk_level)

    enrichment._component_outcome(signal)

    assert (
        signal.status,
        signal.score,
        dict(signal.signal_metadata),
        signal.risk_level,
    ) == before
    # The forensic status stays `FAILED`: the reading produced no score, which is what that
    # means on a signal row, and the enrichment axis is a different question about it.
    assert signal.status == "FAILED"


# --------------------------------------------------------------------------------------
# Fixtures for everything that needs real PostgreSQL
# --------------------------------------------------------------------------------------


@pytest.fixture
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def analysis(database):
    """A decided analysis with its media, exactly as the fast decision path leaves one.

    Committed `completed` with a verdict, because that is the only state enrichment is ever
    queued against: §4.2 makes Deep Evidence something that happens to an analysis that already
    has an answer.
    """
    created = []

    def make(*, status="completed", risk_level="NO_CALIBRATED_MANIPULATION_SIGNAL",
             was_normalized=False, derivative_storage_key=None):
        with SessionLocal() as session:
            row = Analysis(
                status=status,
                risk_level=risk_level,
                risk_rules_version=RULES_V5 if risk_level else None,
                risk_calibration_id=("c" * 64) if risk_level else None,
                risk_rule_id="R9-200" if risk_level else None,
            )
            session.add(row)
            session.flush()

            digest = uuid.uuid4().hex
            session.add(
                MediaFile(
                    analysis_id=row.id,
                    original_filename="clip.mp4",
                    content_type="video/mp4",
                    size_bytes=1024,
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
                    was_normalized=was_normalized,
                    derivative_storage_key=derivative_storage_key,
                )
            )
            session.add(AnalysisJob(analysis_id=row.id, status="completed"))
            session.commit()
            created.append(row.id)

            return row.id

    yield make

    with SessionLocal() as session:
        for analysis_id in created:
            # Media, job, tasks, signals and segments all go through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


def read_analysis(analysis_id):
    with SessionLocal() as session:
        return session.get(Analysis, analysis_id)


def read_tasks(analysis_id):
    """This analysis's task rows, keyed by component."""
    with SessionLocal() as session:
        return {
            (task.provider, task.signal_type): task
            for task in session.query(AnalysisEnrichmentTask)
            .filter_by(analysis_id=analysis_id)
            .all()
        }


def read_signals(analysis_id):
    with SessionLocal() as session:
        return {
            (signal.provider, signal.signal_type): signal
            for signal in session.query(AnalysisSignal)
            .filter_by(analysis_id=analysis_id)
            .all()
        }


# --------------------------------------------------------------------------------------
# Queueing, deferring and asking
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_queueing_writes_one_row_per_deep_evidence_component(analysis, monkeypatch):
    """One row per component, because `ENRICHMENT_PARTIAL` has to be able to name one."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        queued = enrichment.enqueue(session, analysis_id, RULES_V5)

    tasks = read_tasks(analysis_id)

    assert queued == len(tasks)
    assert set(tasks) == set(enrichment.evidence_only_components(RULES_V5))
    assert {task.status for task in tasks.values()} == {"queued"}
    # The role assignment is frozen on the row: a later ruleset must not be able to rewrite
    # what this analysis's components were, any more than it can rewrite its verdict.
    assert {task.rules_version for task in tasks.values()} == {RULES_V5}


@pytest.mark.integration
def test_a_disabled_detector_gets_no_task_row_at_all(analysis, monkeypatch):
    """Effort's rollback switch, at the row level (`app.effort.is_enabled`).

    A detector this deployment has switched off is not evidence-only-and-failing; it is a
    detector nothing asked anything. So it gets no task, writes no signal and counts in no
    enrichment state — which is what keeps the rollback clean: an analysis run with Effort off
    is indistinguishable from one run before Effort existed.

    The switch is an operational fact about this deployment and the membership is a forensic
    fact about the ruleset, which is why they are two questions and this one cannot quietly
    become a role assignment.
    """
    from app.effort import ENABLED_VARIABLE

    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    monkeypatch.setenv(ENABLED_VARIABLE, "false")
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)

    tasks = read_tasks(analysis_id)

    assert ("effort", "face_forgery") not in tasks
    # And the components that are not switchable are all still there.
    assert LIP_COMPONENT in tasks
    assert AASIST_COMPONENT in tasks


@pytest.mark.integration
def test_deferring_writes_the_rows_and_asks_for_none_of_them(analysis, monkeypatch):
    """`ENRICHMENT_NOT_REQUESTED` is a state, and §5.2 is explicit that it is not a failure."""
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, "deferred")
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_NOT_REQUESTED,
        )

    assert {task.status for task in read_tasks(analysis_id).values()} == {"not_requested"}

    # And nothing claims it, which is the whole of "deferred".
    with SessionLocal() as session:
        assert enrichment.claim(session) is None


@pytest.mark.integration
def test_asking_for_deferred_enrichment_queues_it(analysis, monkeypatch):
    """The internal trigger, and the whole of it (§10, question 5).

    Deliberately not given a product name here. R10-T3 decides what a user sees and what it is
    called; this task owes that one a mechanism.
    """
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, "deferred")
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        requested = enrichment.request(session, analysis_id)

        assert requested == len(read_tasks(analysis_id))
        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_PENDING,
        )

    assert {task.status for task in read_tasks(analysis_id).values()} == {"queued"}


@pytest.mark.integration
def test_asking_again_does_not_re_run_what_already_finished(analysis, monkeypatch):
    """`request` moves `not_requested` and nothing else.

    Re-running a component that already terminated is not what "please start the enrichment
    that was deferred" means, and a request that quietly re-ran finished work would make the
    enrichment axis oscillate out of a terminal state.
    """
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, "deferred")
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        session.execute(
            AnalysisEnrichmentTask.__table__.update()
            .where(AnalysisEnrichmentTask.analysis_id == analysis_id)
            .values(status="completed")
        )
        session.commit()

        assert enrichment.request(session, analysis_id) == 0

    assert {task.status for task in read_tasks(analysis_id).values()} == {"completed"}


@pytest.mark.integration
def test_a_legacy_analysis_is_read_as_single_stage_rather_than_enriched(analysis):
    """§8.1, against real rows: no migration gave this analysis any two-axis state."""
    analysis_id = analysis()

    with SessionLocal() as session:
        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.LEGACY_SINGLE_STAGE,
        )


@pytest.mark.integration
def test_queueing_never_raises_into_the_decision_that_just_completed(analysis, monkeypatch):
    """The load-bearing property of `enqueue`, not loose error handling.

    An analysis that has been decided, published and closed must not retroactively fail
    because its supplementary evidence could not be *scheduled*. That would be the enrichment
    path destroying a fast decision through the one door §7.4 does not name.
    """
    analysis_id = analysis()

    with SessionLocal() as session:
        # A ruleset nothing declares roles for: the failure `enqueue` is most likely to meet.
        assert enrichment.enqueue(session, analysis_id, "r11-v6.0.0") == 0

    assert read_tasks(analysis_id) == {}
    assert read_analysis(analysis_id).status == "completed"
    assert read_analysis(analysis_id).risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"


# --------------------------------------------------------------------------------------
# Claiming, leasing and recovery
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_claiming_takes_every_queued_component_of_one_analysis(analysis, monkeypatch):
    """One claim, one download, one WAV extraction — and still a state per component."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        claimed = enrichment.claim(session)

    assert claimed is not None
    assert claimed.analysis_id == analysis_id
    assert claimed.rules_version == RULES_V5
    assert {component.component for component in claimed.components} == set(
        enrichment.evidence_only_components(RULES_V5)
    )
    # Media that needed no derivative is read as the original: it *is* the artifact every
    # deciding detector saw, and re-encoding it would hand these detectors a different file.
    assert claimed.storage_key.startswith("originals/")

    tasks = read_tasks(analysis_id)
    assert {task.status for task in tasks.values()} == {"processing"}
    # A row that was `processing` for even one commit without a deadline would be a task no
    # recovery could ever reach.
    assert all(task.lease_expires_at is not None for task in tasks.values())


@pytest.mark.integration
def test_enrichment_reads_the_derivative_the_decision_was_taken_from(analysis, monkeypatch):
    """Never a second transcode: the fast path already made and uploaded this artifact."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis(
        was_normalized=True, derivative_storage_key="derivatives/abc123"
    )

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        claimed = enrichment.claim(session)

    assert claimed.storage_key == "derivatives/abc123"


@pytest.mark.integration
def test_a_decision_with_no_readable_artifact_fails_its_components(analysis, monkeypatch):
    """A transcode that failed leaves nothing for Deep Evidence to read.

    The components record that and no signal row is invented to explain it: nothing asked
    these detectors anything, and a `FAILED` signal would be a finding nobody made.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis(was_normalized=True, derivative_storage_key=None)

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        assert enrichment.process_one(session) is True

    tasks = read_tasks(analysis_id)
    assert {task.status for task in tasks.values()} == {"failed"}
    assert {task.error_message for task in tasks.values()} == {
        enrichment.MISSING_ARTIFACT_ERROR
    }
    assert read_signals(analysis_id) == {}
    # And the verdict is exactly where the fast decision left it.
    assert read_analysis(analysis_id).risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"


@pytest.mark.integration
def test_a_stale_enrichment_lease_is_recovered_without_touching_the_analysis(
    analysis, monkeypatch
):
    """Invariant I11: an enrichment worker crash never moves an analysis out of `DECIDED`.

    The recovery statement cannot reach `analyses` — that is what makes this structural rather
    than a promise — so what is checked here is the outcome that proves it.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        enrichment.claim(session)
        # The worker died: the lease it promised to renew has fallen into the past.
        session.execute(
            AnalysisEnrichmentTask.__table__.update()
            .where(AnalysisEnrichmentTask.analysis_id == analysis_id)
            .values(lease_expires_at=func.now() - timedelta(minutes=5))
        )
        session.commit()

        recovered = enrichment.recover_stale_tasks(session)

    assert recovered == len(read_tasks(analysis_id))

    tasks = read_tasks(analysis_id)
    assert {task.status for task in tasks.values()} == {"failed"}
    assert {task.error_message for task in tasks.values()} == {
        enrichment.STALE_LEASE_ERROR
    }

    decided = read_analysis(analysis_id)
    assert decided.status == "completed"
    assert decided.risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"
    assert decided.risk_rule_id == "R9-200"


# --------------------------------------------------------------------------------------
# Resource priority: the fast path is never behind enrichment
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_a_queued_decision_job_is_claimed_before_any_enrichment(analysis, monkeypatch):
    """§4.2: Deep Evidence must never contend in a way that can starve the Fast Decision Path.

    The priority is structural rather than a weighting: `app.worker.run` asks for a job on
    every poll and only reaches enrichment when there was none. So the property to demonstrate
    is that a poll with both kinds of work available takes the decision job and leaves the
    enrichment task untouched — repeatedly, which is what "cannot be starved" means.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)

    enriching = analysis()
    with SessionLocal() as session:
        enrichment.enqueue(session, enriching, RULES_V5)

    # Three analyses still owed a decision, queued behind a full enrichment backlog.
    pending = [analysis(status="queued", risk_level=None) for _ in range(3)]
    with SessionLocal() as session:
        session.execute(
            AnalysisJob.__table__.update()
            .where(AnalysisJob.analysis_id.in_(pending))
            .values(status="queued")
        )
        session.commit()

    claimed_analyses = []
    for _ in range(3):
        with SessionLocal() as session:
            claimed = worker.claim_job(session)
            assert claimed is not None, "a queued decision job was passed over"
            claimed_analyses.append(claimed.analysis_id)

    assert sorted(claimed_analyses) == sorted(pending)
    # Untouched throughout: nothing claimed the enrichment while decision work existed.
    assert {task.status for task in read_tasks(enriching).values()} == {"queued"}


@pytest.mark.integration
def test_enrichment_is_only_claimed_once_the_decision_queue_is_empty(analysis, monkeypatch):
    """The other half of the same property: secondary work is not starved either, just last."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        # No decision job is queued, so the loop's first question comes back empty.
        assert worker.claim_job(session) is None
        assert enrichment.claim(session) is not None


# --------------------------------------------------------------------------------------
# The persistence boundary (§6.1), enforced rather than conventional
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_the_boundary_refuses_an_orm_write_to_the_analysis(analysis):
    """The decision fields and `analyses.status` are out of reach, not merely unasked for."""
    analysis_id = analysis()

    with enrichment_session(ALLOWED) as session:
        row = session.get(Analysis, analysis_id)
        row.risk_level = "MANIPULATION_DETECTED"

        with pytest.raises(ProtectedWriteRejected):
            session.flush()


@pytest.mark.integration
def test_the_boundary_refuses_a_core_update_of_the_analysis(analysis):
    """The shape a copy-paste from `app.worker._set_status` would take."""
    analysis_id = analysis()

    with enrichment_session(ALLOWED) as session:
        with pytest.raises(ProtectedWriteRejected):
            session.execute(
                Analysis.__table__.update()
                .where(Analysis.id == analysis_id)
                .values(status="completed")
            )


@pytest.mark.integration
def test_the_boundary_refuses_an_idempotent_rewrite_of_a_decision_field(analysis):
    """§6.1: "the prohibition is on the write, not on the delta".

    Writing the same value back is still a second write to a decision field, it still moves
    `updated_at`-style metadata, and it still means the enrichment path holds code that
    touches the verdict. The database trigger below cannot see this case — the values do not
    change — which is exactly why the two layers are not the same check.
    """
    analysis_id = analysis()
    unchanged = read_analysis(analysis_id).risk_level

    with enrichment_session(ALLOWED) as session:
        with pytest.raises(ProtectedWriteRejected):
            session.execute(
                Analysis.__table__.update()
                .where(Analysis.id == analysis_id)
                .values(risk_level=unchanged)
            )


@pytest.mark.integration
def test_the_boundary_refuses_the_decision_job_the_media_and_a_review(analysis):
    """The other three tables §6.1 names, refused by table rather than by column."""
    analysis_id = analysis()

    with enrichment_session(ALLOWED) as session:
        with pytest.raises(ProtectedWriteRejected):
            session.execute(
                AnalysisJob.__table__.update()
                .where(AnalysisJob.analysis_id == analysis_id)
                .values(status="queued")
            )

    with enrichment_session(ALLOWED) as session:
        with pytest.raises(ProtectedWriteRejected):
            session.execute(
                MediaFile.__table__.update()
                .where(MediaFile.analysis_id == analysis_id)
                .values(original_sha256="d" * 64)
            )

    with enrichment_session(ALLOWED) as session:
        session.add(AnalysisReview(analysis_id=analysis_id, status="REVIEWED"))
        with pytest.raises(ProtectedWriteRejected):
            session.flush()


@pytest.mark.integration
def test_the_boundary_refuses_the_deciding_detectors_evidence(analysis):
    """SVD and B7 rows are the evidence the verdict rests on (§6.1).

    Refused by an allowlist read from the ruleset rather than by a denylist naming these two,
    so that a ruleset which promotes a detector back into the decision stops admitting it
    without this code being edited (§3.2).
    """
    analysis_id = analysis()

    for provider, signal_type in (SVD_COMPONENT, B7_COMPONENT):
        with enrichment_session(ALLOWED) as session:
            session.add(
                AnalysisSignal(
                    analysis_id=analysis_id,
                    provider=provider,
                    signal_type=signal_type,
                    status="SUCCESS",
                    score=0.9,
                )
            )
            with pytest.raises(ProtectedWriteRejected):
                session.flush()


@pytest.mark.integration
def test_the_boundary_refuses_a_risk_level_on_a_signal_row(analysis):
    """§6.2: `analysis_signals.risk_level` stays null on every row.

    A level written beside a provider's score would be the per-detector verdict the risk engine
    exists to refuse (rule 11).
    """
    analysis_id = analysis()

    with enrichment_session(ALLOWED) as session:
        session.add(
            AnalysisSignal(
                analysis_id=analysis_id,
                provider=LIP_COMPONENT[0],
                signal_type=LIP_COMPONENT[1],
                status="SUCCESS",
                score=0.4,
                risk_level="MANIPULATION_DETECTED",
            )
        )
        with pytest.raises(ProtectedWriteRejected):
            session.flush()


@pytest.mark.integration
def test_the_boundary_refuses_a_segment_of_a_signal_it_does_not_own(analysis):
    """Segments are writable only beneath an enrichment signal of this same analysis (§6.2)."""
    analysis_id = analysis()

    with SessionLocal() as session:
        deciding = AnalysisSignal(
            analysis_id=analysis_id,
            provider=SVD_COMPONENT[0],
            signal_type=SVD_COMPONENT[1],
            status="SUCCESS",
            score=0.9,
        )
        session.add(deciding)
        session.commit()
        deciding_id = deciding.id

    with enrichment_session(ALLOWED) as session:
        session.add(AnalysisSegment(signal_id=deciding_id, clip_index=0, logit=1.0))
        with pytest.raises(ProtectedWriteRejected):
            session.flush()


@pytest.mark.integration
def test_the_boundary_refuses_core_dml_against_signals(analysis):
    """An allowlist that waved through statements it could not inspect would not be one."""
    analysis_id = analysis()

    with enrichment_session(ALLOWED) as session:
        with pytest.raises(ProtectedWriteRejected):
            session.execute(
                AnalysisSignal.__table__.update()
                .where(AnalysisSignal.analysis_id == analysis_id)
                .values(score=0.1)
            )


@pytest.mark.integration
def test_the_boundary_allows_the_evidence_enrichment_is_for(analysis, monkeypatch):
    """§6.2, from the other side: the allowed writes must actually go through.

    A guard that refused everything would pass every test above and deliver nothing.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)

    with enrichment_session(ALLOWED) as session:
        signal = AnalysisSignal(
            analysis_id=analysis_id,
            provider=LIP_COMPONENT[0],
            signal_type=LIP_COMPONENT[1],
            status="SUCCESS",
            score=0.42,
            provider_version="lip-1",
        )
        session.add(signal)
        session.flush()
        session.add(AnalysisSegment(signal_id=signal.id, clip_index=0, logit=1.5))
        session.execute(
            AnalysisEnrichmentTask.__table__.update()
            .where(AnalysisEnrichmentTask.analysis_id == analysis_id)
            .values(status="completed")
        )
        session.commit()

    assert read_signals(analysis_id)[LIP_COMPONENT].score == 0.42
    assert {task.status for task in read_tasks(analysis_id).values()} == {"completed"}


# --------------------------------------------------------------------------------------
# The database guard (§6.3), which does not care which code path is executing
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_a_decided_analysis_cannot_have_its_verdict_changed_by_anything(analysis):
    """§6.3, enforced by PostgreSQL rather than by whoever holds the session.

    This is the guarantee the application guard cannot give: a statement from a connection that
    never went through `enrichment_session` — a future caller, a psql prompt, a worker back
    from the dead — is refused all the same.
    """
    analysis_id = analysis()

    for column, value in (
        ("risk_level", "MANIPULATION_DETECTED"),
        ("risk_rule_id", "R9-101"),
        ("risk_rules_version", "r7-v4.0.0"),
        ("risk_calibration_id", "e" * 64),
        ("status", "failed"),
    ):
        with SessionLocal() as session:
            with pytest.raises(DatabaseError):
                session.execute(
                    text(f"UPDATE analyses SET {column} = :value WHERE id = :id"),
                    {"value": value, "id": analysis_id},
                )
                session.commit()
            session.rollback()

    decided = read_analysis(analysis_id)
    assert decided.risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"
    assert decided.status == "completed"


@pytest.mark.integration
def test_an_undecided_analysis_may_still_be_published(analysis):
    """The guard must not break the one write that is supposed to happen.

    It fires on `risk_level IS NOT NULL`, which is precisely "this analysis has been decided"
    — so the first verdict publication passes, and only a second is refused.
    """
    analysis_id = analysis(status="queued", risk_level=None)

    with SessionLocal() as session:
        session.execute(
            Analysis.__table__.update()
            .where(Analysis.id == analysis_id)
            .values(
                status="completed",
                risk_level="MANIPULATION_DETECTED",
                risk_rules_version=RULES_V5,
                risk_calibration_id="f" * 64,
                risk_rule_id="R9-101",
            )
        )
        session.commit()

    assert read_analysis(analysis_id).risk_level == "MANIPULATION_DETECTED"


@pytest.mark.integration
def test_a_failed_analysis_is_not_frozen_by_the_guard(analysis):
    """`DECISION_FAILED` rows have null decision fields and are not what §6.3 freezes.

    Stated as a test because the alternative reading — freeze everything terminal — would have
    blocked the recovery path that fails an analysis whose worker died, and that is a write
    this system depends on.
    """
    analysis_id = analysis(status="queued", risk_level=None)

    with SessionLocal() as session:
        session.execute(
            Analysis.__table__.update()
            .where(Analysis.id == analysis_id)
            .values(status="failed")
        )
        session.commit()

    assert read_analysis(analysis_id).status == "failed"


# --------------------------------------------------------------------------------------
# Resilience: enrichment failures, retries and crashes never reach the decision (§7.4)
# --------------------------------------------------------------------------------------


@pytest.fixture
def fake_components(monkeypatch):
    """Stand in for the four evidence-only detectors, so no test here loads a checkpoint."""

    class Recorder:
        def __init__(self):
            self.lip_error = None
            self.lip_abstains = False
            self.effort_abstains = False
            self.audio_abstains = False
            self.calls = []

        def detect_lip_forensics(self, path):
            self.calls.append("lip")
            if self.lip_error:
                raise self.lip_error
            if self.lip_abstains:
                # Recorded exactly as `app.detection` records it: a `FAILED` row with the
                # abstention's own exception class and no score. The detector was asked and
                # answered; what it answered is that there was nothing here to score.
                return AnalysisSignal(
                    provider=LIP_COMPONENT[0],
                    signal_type=LIP_COMPONENT[1],
                    status="FAILED",
                    score=None,
                    signal_metadata={"error": "LipForensicsNoTrackedFace"},
                )
            return AnalysisSignal(
                provider=LIP_COMPONENT[0],
                signal_type=LIP_COMPONENT[1],
                status="SUCCESS",
                score=0.31,
                provider_version="lip-1",
            )

        def detect_effort(self, path):
            self.calls.append("effort")
            # A genuine failure by default — a checkpoint this deployment does not have — so
            # that the tests which need a real failure beside an abstention have one.
            return AnalysisSignal(
                provider="effort",
                signal_type="face_forgery",
                status="FAILED",
                signal_metadata={
                    "error": (
                        "EffortNoFaceDetected"
                        if self.effort_abstains
                        else "EffortModelUnavailable"
                    )
                },
            )

        async def analyse_audio(self, path, frame_rate):
            self.calls.append("audio")
            if self.audio_abstains:
                # The one abstention that reaches two components at once: both were waiting on
                # the same extraction, and the media carries no audio to extract.
                reason = {"error": "SpeakerDiarizationAudioError"}
                return (
                    AnalysisSignal(
                        provider="nvidia",
                        signal_type="active_speaker",
                        status="FAILED",
                        signal_metadata=reason,
                    ),
                    [],
                ), (
                    AnalysisSignal(
                        provider=AASIST_COMPONENT[0],
                        signal_type=AASIST_COMPONENT[1],
                        status="FAILED",
                        signal_metadata=reason,
                    ),
                    [],
                )
            speaker = AnalysisSignal(
                provider="nvidia",
                signal_type="active_speaker",
                status="SUCCESS",
                provider_version="asd-1",
            )
            audio = AnalysisSignal(
                provider=AASIST_COMPONENT[0],
                signal_type=AASIST_COMPONENT[1],
                status="SUCCESS",
                score=0.12,
                provider_version="aasist-1",
            )
            return (speaker, []), (audio, [AnalysisSegment(start_time=0.0, end_time=1.0, logit=0.5)])

    recorder = Recorder()
    monkeypatch.setattr(detection, "detect_lip_forensics", recorder.detect_lip_forensics)
    monkeypatch.setattr(detection, "detect_effort", recorder.detect_effort)
    monkeypatch.setattr(detection, "analyse_audio", recorder.analyse_audio)
    return recorder


@pytest.fixture
def fake_artifact(monkeypatch, tmp_path):
    """Hand the enrichment path a local file instead of reaching MinIO."""
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"not really a video")

    def fetch(storage_key, path):
        path.write_bytes(artifact.read_bytes())

    monkeypatch.setattr(enrichment, "fetch_object", fetch)
    return artifact


@pytest.mark.integration
def test_a_completed_enrichment_leaves_the_verdict_exactly_as_it_found_it(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """Invariant I3: no Deep Evidence outcome changes any of the five decision fields."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()
    before = read_analysis(analysis_id)
    frozen = (
        before.status,
        before.risk_level,
        before.risk_rules_version,
        before.risk_calibration_id,
        before.risk_rule_id,
    )

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        assert enrichment.process_one(session) is True

    after = read_analysis(analysis_id)
    assert (
        after.status,
        after.risk_level,
        after.risk_rules_version,
        after.risk_calibration_id,
        after.risk_rule_id,
    ) == frozen

    signals = read_signals(analysis_id)
    assert signals[LIP_COMPONENT].score == 0.31
    assert signals[AASIST_COMPONENT].score == 0.12


@pytest.mark.integration
def test_a_component_that_failed_is_partial_and_the_verdict_still_stands(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """§7.4's worked example, end to end.

    Effort's stub records a `FAILED` signal, the other three succeed. The analysis is `DECIDED`
    and `ENRICHMENT_PARTIAL`, the verdict is unchanged and unqualified, and the failure is
    readable per component rather than as a summary nobody can trace back.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        enrichment.process_one(session)

        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_PARTIAL,
        )

    tasks = read_tasks(analysis_id)
    assert tasks[("effort", "face_forgery")].status == "failed"
    # An operator can read *why*, which is what makes `ENRICHMENT_PARTIAL` traceable.
    assert tasks[("effort", "face_forgery")].error_message == "EffortModelUnavailable"
    assert tasks[LIP_COMPONENT].status == "completed"

    assert read_analysis(analysis_id).risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"


@pytest.mark.integration
def test_a_component_that_crashes_outright_costs_only_that_component(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """A detector that raises rather than recording a signal — the unhandled case.

    `app.detection` turns almost everything into a `FAILED` signal, so what reaches this path
    is what that layer does not catch: a timeout, a torch that broke, a checkpoint that is not
    there. Two things must be true of it, and they pull in different directions.

    It must not reach the decision. The analysis stays `DECIDED` with the verdict, the rule and
    the ruleset the fast path published, because the enrichment path cannot reach those columns
    (invariant I11).

    And it must not cost the readings taken beside it. LipForensics collapsing says nothing
    about the AASIST reading produced from the same artifact moments later — each is a
    complete, independent finding (rule 11) — so the claim ends `ENRICHMENT_PARTIAL` with the
    failure named on one row, not `ENRICHMENT_FAILED` with three good readings thrown away to
    tidy up. That is the same judgement `app.worker.AnalysisTimedOut` makes on the other path.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    fake_components.lip_error = RuntimeError("the checkpoint exploded")
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        # True: there was work, it was claimed, and every component of it was concluded.
        assert enrichment.process_one(session) is True

        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_PARTIAL,
        )

    tasks = read_tasks(analysis_id)
    assert tasks[LIP_COMPONENT].status == "failed"
    # The exception's class name, never its text: that can quote credentials, paths or SQL.
    assert tasks[LIP_COMPONENT].error_message == "RuntimeError"
    # No signal row for it. Nothing was asked and answered, so there is no finding to record.
    assert LIP_COMPONENT not in read_signals(analysis_id)

    # The readings beside it stand, evidence and all.
    assert tasks[AASIST_COMPONENT].status == "completed"
    assert read_signals(analysis_id)[AASIST_COMPONENT].score == 0.12

    decided = read_analysis(analysis_id)
    assert decided.status == "completed"
    assert decided.risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"
    assert decided.risk_rule_id == "R9-200"


@pytest.mark.integration
def test_a_claim_that_cannot_be_run_at_all_leaves_the_decision_untouched(
    analysis, fake_components, monkeypatch
):
    """The other failure shape: the artifact could not be fetched, so nothing ran.

    Every claimed component is failed, the loop keeps running, and the analysis is exactly as
    the fast decision left it. This is the difference between an enrichment that failed and an
    analysis that failed.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    def unreachable(storage_key, path):
        raise OSError("the object store is not there")

    monkeypatch.setattr(enrichment, "fetch_object", unreachable)

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        assert enrichment.process_one(session) is True

        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_FAILED,
        )

    assert {task.status for task in read_tasks(analysis_id).values()} == {"failed"}
    assert {task.error_message for task in read_tasks(analysis_id).values()} == {"OSError"}
    assert read_signals(analysis_id) == {}

    decided = read_analysis(analysis_id)
    assert decided.status == "completed"
    assert decided.risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"


@pytest.mark.integration
def test_a_broken_enrichment_does_not_stop_the_worker_loop(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """`process_one` swallows its own failures, as `app.shadow.process_one` does.

    A loop that crashed out of enrichment would have stopped claiming decision jobs, which
    would be a strange way to lose the independence R10 exists to create.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    def explode(session):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(enrichment, "recover_stale_tasks", explode)

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        # Reported as "nothing was done" rather than raised into the loop.
        assert enrichment.process_one(session) is False


@pytest.mark.integration
def test_re_running_a_component_replaces_its_evidence_rather_than_doubling_it(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """§7.3: enrichment retries are idempotent per component.

    Evidence from two different executions of the same detector never coexists, and the
    segments of a re-run component go with their parent signal through the cascade.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        enrichment.process_one(session)

        # Ask for the whole thing again, as a retry would.
        session.execute(
            AnalysisEnrichmentTask.__table__.update()
            .where(AnalysisEnrichmentTask.analysis_id == analysis_id)
            .values(status="queued", error_message=None)
        )
        session.commit()

        enrichment.process_one(session)

    with SessionLocal() as session:
        rows = session.execute(
            select(func.count())
            .select_from(AnalysisSignal)
            .where(
                AnalysisSignal.analysis_id == analysis_id,
                AnalysisSignal.provider == AASIST_COMPONENT[0],
                AnalysisSignal.signal_type == AASIST_COMPONENT[1],
            )
        ).scalar()

        segments = session.execute(
            select(func.count())
            .select_from(AnalysisSegment)
            .join(AnalysisSignal, AnalysisSegment.signal_id == AnalysisSignal.id)
            .where(AnalysisSignal.analysis_id == analysis_id)
        ).scalar()

    assert rows == 1, "a retry accumulated a second signal row for one component"
    assert segments == 1, "a retry left segments from two executions side by side"


@pytest.mark.integration
def test_an_abstaining_component_ends_terminal_and_not_failed(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """The whole mapping, end to end and against real rows.

    LipForensics abstains, Effort fails for real, the audio pair succeeds. All four terminate,
    so the enrichment axis settles — and it settles on `ENRICHMENT_PARTIAL` because of Effort
    alone. Had the abstention been collapsed into a failure, this analysis would have been
    indistinguishable from one where two components broke.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    fake_components.lip_abstains = True
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        enrichment.process_one(session)

        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_PARTIAL,
        )

    tasks = read_tasks(analysis_id)
    assert tasks[LIP_COMPONENT].status == "abstained"
    assert tasks[LIP_COMPONENT].error_message == "LipForensicsNoTrackedFace"
    # The genuine failure beside it is still a failure, and still traceable.
    assert tasks[("effort", "face_forgery")].status == "failed"
    assert tasks[("effort", "face_forgery")].error_message == "EffortModelUnavailable"
    assert tasks[AASIST_COMPONENT].status == "completed"

    # And the evidence is written exactly as it always was: a `FAILED` signal with the
    # abstention's reason and no score. Nothing about the forensic record changed shape to
    # carry the enrichment distinction.
    signal = read_signals(analysis_id)[LIP_COMPONENT]
    assert signal.status == "FAILED"
    assert signal.score is None
    assert signal.signal_metadata == {"error": "LipForensicsNoTrackedFace"}
    assert signal.risk_level is None

    # The verdict is untouched, which is the claim every test in this module ends on.
    assert read_analysis(analysis_id).risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"
    assert read_analysis(analysis_id).risk_rule_id == "R9-200"


@pytest.mark.integration
def test_an_analysis_whose_components_all_abstained_is_fully_enriched(
    analysis, fake_components, fake_artifact, monkeypatch
):
    """A silent, faceless clip is completely enriched, not partially.

    There was nothing more to get, and `ENRICHMENT_COMPLETE` is the honest reading of that.
    R10-T3 must not render it as supplementary evidence having failed.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    fake_components.lip_abstains = True
    fake_components.effort_abstains = True
    fake_components.audio_abstains = True
    analysis_id = analysis()

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        enrichment.process_one(session)

        assert enrichment.analysis_states(session, analysis_id) == (
            enrichment.DECIDED,
            enrichment.ENRICHMENT_COMPLETE,
        )

    assert {task.status for task in read_tasks(analysis_id).values()} == {"abstained"}


@pytest.mark.integration
def test_a_component_whose_artifact_never_existed_manufactures_no_signal_row(
    analysis, monkeypatch
):
    """The regression pin: a prerequisite that could not be produced writes no evidence.

    A `FAILED` signal row means a source was asked about this media and had no answer. ASD and
    AASIST were never asked here — the transcode failed, so the artifact they read was never
    produced — and manufacturing rows for them would record findings nobody made, and would
    put those detectors' names on a gap they had no part in.

    What is recorded instead is the enrichment task's own reason, which is where a fact about
    the *execution* belongs. §7.4 asks that an enrichment failure be readable per component by
    an operator, and this is that, without inventing forensic evidence to carry it.
    """
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)
    analysis_id = analysis(was_normalized=True, derivative_storage_key=None)

    with SessionLocal() as session:
        enrichment.enqueue(session, analysis_id, RULES_V5)
        assert enrichment.process_one(session) is True

    # No evidence rows at all, for any component.
    assert read_signals(analysis_id) == {}

    tasks = read_tasks(analysis_id)
    assert {task.status for task in tasks.values()} == {"failed"}
    assert {task.error_message for task in tasks.values()} == {
        enrichment.MISSING_ARTIFACT_ERROR
    }
    # Not an abstention: nothing was asked, so nothing answered. The two must not merge.
    assert "abstained" not in {task.status for task in tasks.values()}

    assert read_analysis(analysis_id).risk_level == "NO_CALIBRATED_MANIPULATION_SIGNAL"
