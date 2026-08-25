"""The risk engine, and the orchestration that stores what it decided.

Two halves, and the split is deliberate.

The first half runs the rules directly. They are pure functions over a frozen dataclass —
no database, no clock, no configuration — so every branch, boundary and degenerate input
is reachable without staging anything, and the tests read as the rule table they check.

The second half runs `conclude_job` against real PostgreSQL, because what is being checked
there is not arithmetic but a property of the stored row: that the decision, the ruleset,
the calibration and the rule that fired all survive the write, and that no amount of
changing, removing or failing the *other* three evidence sources moves the classification
by a band. Those tests write the signal rows themselves rather than driving the detectors,
so the SVD evidence under test is exact and the context signals around it can be varied
freely — which is the whole point of the isolation check.

**No threshold is mocked anywhere in this module.** `T_HIGH = 0.98` is a calibrated
constant with a 340-sample study behind it (P7-T2), not a knob; a test that patched it
would be checking that comparison operators work rather than that DeepGuard classifies
media the way it was calibrated to.
"""

import dataclasses
import inspect
import math

import pytest
from sqlalchemy.exc import MultipleResultsFound, SQLAlchemyError

from app import risk_engine, worker
from app.db.models import (
    Analysis,
    AnalysisJob,
    AnalysisSignal,
)
from app.db.session import SessionLocal, engine
from app.risk_engine import RiskDecision, SvdEvidence, evaluate

# The deployment the calibration binds to, restated rather than imported. If the constant in
# `app.risk_engine` is ever edited, these tests must fail rather than agree with the edit:
# the binding is a fact about a measurement that was taken, not a value the code may choose.
VALIDATED_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"
EXPECTED_RULES_VERSION = "p7-v1.0.0"
EXPECTED_CALIBRATION_ID = "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c"
EXPECTED_T_HIGH = 0.98

# NVIDIA's aggregate figures for a call that succeeded, as `detect_synthetic_video` writes
# them into the signal's metadata.
TOTAL_CLIPS = 7
LOGIT = 1.9142135381698608


def svd(
    *,
    score=0.5,
    provider="nvidia",
    signal_type="synthetic_video",
    status="SUCCESS",
    provider_version=VALIDATED_FUNCTION_ID,
    total_clips=TOTAL_CLIPS,
) -> SvdEvidence:
    """Persisted synthetic-video evidence from the calibrated deployment, unless varied.

    Defaults are a healthy, eligible signal, so every test below states only the one thing
    it is about and nothing else can drift underneath it.
    """
    return SvdEvidence(
        provider=provider,
        signal_type=signal_type,
        status=status,
        provider_version=provider_version,
        score=score,
        total_clips=total_clips,
    )


# --------------------------------------------------------------------------------------
# The calibration the rules are bound to
# --------------------------------------------------------------------------------------


def test_the_ruleset_names_itself_and_the_calibration_behind_it():
    """The two identifiers every stored decision is only readable through."""
    assert risk_engine.RULES_VERSION == EXPECTED_RULES_VERSION
    assert risk_engine.CALIBRATION_ID == EXPECTED_CALIBRATION_ID


def test_the_thresholds_are_the_measured_ones():
    assert risk_engine.T_HIGH == EXPECTED_T_HIGH
    assert risk_engine.CALIBRATED_PROVIDER == "nvidia"
    assert risk_engine.CALIBRATED_SIGNAL_TYPE == "synthetic_video"
    assert risk_engine.CALIBRATED_PROVIDER_VERSION == VALIDATED_FUNCTION_ID


def test_no_minimum_duration_guard_exists():
    """`D_MIN` was removed, not given a value.

    P7-T2 §8 found no contract basis and no empirical basis for one, and the degeneracy it
    was speculatively guarding against is caught by `R012` on the provider's own clip count
    instead. A constant reintroduced here would be an invented number.
    """
    assert not hasattr(risk_engine, "D_MIN")


def test_t_low_is_recorded_but_is_not_a_boundary():
    """The measured value is part of the calibration identity and decides nothing.

    Both sides of 0.05 must classify identically, because there is no LOW band in v1.
    """
    assert risk_engine.T_LOW == 0.05

    below = evaluate(svd(score=0.04))
    above = evaluate(svd(score=0.06))

    assert below.risk_level == above.risk_level == "MEDIUM"
    assert below.rule_id == above.rule_id == "R200"


# --------------------------------------------------------------------------------------
# R100 — CALIBRATED_HIGH
# --------------------------------------------------------------------------------------


def test_a_score_exactly_at_the_threshold_is_high():
    """`>=`, not `>`. 0.98 is inside the HIGH band by the rule's own definition."""
    decision = evaluate(svd(score=0.98))

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R100"


@pytest.mark.parametrize("score", [0.9800001, 0.9979, 0.999999, 1.0])
def test_a_score_above_the_threshold_is_high(score):
    decision = evaluate(svd(score=score))

    assert decision.risk_level == "HIGH"
    assert decision.rule_id == "R100"


# --------------------------------------------------------------------------------------
# R200 — INDETERMINATE_BAND
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("score", [0.9799999, 0.9704, 0.5123, 0.2063, 0.0128])
def test_a_score_below_the_threshold_is_medium(score):
    decision = evaluate(svd(score=score))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R200"


def test_the_bottom_of_the_providers_scale_is_medium():
    """0.0 is a real reading at the floor of `expit`, not missing evidence.

    With LOW disabled it lands in the indeterminate band like any other sub-threshold score.
    Classifying it as UNKNOWN would confuse "the provider said the lowest thing it can say"
    with "the provider said nothing".
    """
    decision = evaluate(svd(score=0.0))

    assert decision.risk_level == "MEDIUM"
    assert decision.rule_id == "R200"


# --------------------------------------------------------------------------------------
# R010 — UNVALIDATED_PROVIDER
# --------------------------------------------------------------------------------------


def test_a_missing_synthetic_video_signal_is_unknown():
    decision = evaluate(None)

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT", "UNAVAILABLE", None])
def test_a_signal_that_did_not_succeed_is_unknown(status):
    """A detector that refused, timed out or was unavailable produced no reading.

    The score is set high on purpose: a stale or placeholder figure sitting beside a
    non-SUCCESS status must not be able to reach HIGH.
    """
    decision = evaluate(svd(score=0.999, status=status))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "provider_version",
    [
        None,
        "",
        "f286f937-05c4-454b-8312-fba67a2a6fa7",
        "0.90.14",
    ],
)
def test_an_unrecognised_provider_version_is_unknown(provider_version):
    decision = evaluate(svd(score=0.999, provider_version=provider_version))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "provider_version",
    [
        f"{VALIDATED_FUNCTION_ID}-preview",
        f"{VALIDATED_FUNCTION_ID} ",
        f" {VALIDATED_FUNCTION_ID}",
        f"nvcf:{VALIDATED_FUNCTION_ID}",
        f"prefix-{VALIDATED_FUNCTION_ID}-suffix",
        VALIDATED_FUNCTION_ID.upper(),
        VALIDATED_FUNCTION_ID[:-1],
    ],
)
def test_a_version_containing_the_validated_id_is_still_unknown(provider_version):
    """Exact equality, never substring, prefix, suffix or case-insensitive matching.

    Every value here contains or nearly is the validated function id and none of them names
    the deployment that was measured. The operating point was chosen against 0.0096 of
    margin over observed genuine media (P7-T2 §11), so a different deployment answering
    behind a similar-looking identifier is precisely what `R010` exists to refuse.
    """
    decision = evaluate(svd(score=0.999, provider_version=provider_version))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize("provider", ["aasist", "c2pa", "NVIDIA", "", None])
def test_a_signal_from_another_provider_is_unknown(provider):
    decision = evaluate(svd(score=0.999, provider=provider))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


@pytest.mark.parametrize(
    "signal_type", ["active_speaker", "audio_authenticity", "provenance", None]
)
def test_a_signal_answering_another_question_is_unknown(signal_type):
    """Only the direct-risk question is calibrated.

    NVIDIA answers two, and the thresholds were measured against exactly one of them.
    """
    decision = evaluate(svd(score=0.999, signal_type=signal_type))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R010"


# --------------------------------------------------------------------------------------
# R012 — INVALID_DIRECT_EVIDENCE
# --------------------------------------------------------------------------------------


def test_a_null_score_on_an_eligible_signal_is_unknown():
    """Eligible, so `R010` passes; then there is no number to compare."""
    decision = evaluate(svd(score=None))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("score", [-0.0001, -1.0, -1e-12, 1.0000001, 2.0, 1e6])
def test_a_score_outside_the_providers_scale_is_unknown(score):
    """`probability = expit(logit)` cannot leave [0, 1]; anything that did is not it."""
    decision = evaluate(svd(score=score))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_a_non_finite_score_is_unknown(score):
    """NaN compares false against every threshold, so it must be rejected explicitly.

    Left to fall through, the band would be decided by which comparison happened to be
    written first rather than by the evidence.
    """
    decision = evaluate(svd(score=score))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("total_clips", [0, -1, None])
def test_evidence_aggregated_over_no_clips_is_unknown(total_clips):
    """The aggregate is a figure over an empty table, not a reading of the video.

    The score is set high so that a degenerate clip count cannot be overridden by it.
    """
    decision = evaluate(svd(score=0.999, total_clips=total_clips))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("total_clips", ["7", 7.0, True, [], {"n": 7}])
def test_a_clip_count_that_is_not_a_whole_number_is_unknown(total_clips):
    """It comes from a JSON document, so no column guarantees its type.

    `True` is in the list because Python would let it pass `> 0` while counting nothing.
    """
    decision = evaluate(svd(score=0.999, total_clips=total_clips))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


@pytest.mark.parametrize("score", ["0.99", None, [], True])
def test_a_score_that_is_not_a_number_is_unknown(score):
    decision = evaluate(svd(score=score))

    assert decision.risk_level == "UNKNOWN"
    assert decision.rule_id == "R012"


# --------------------------------------------------------------------------------------
# Properties of the ruleset as a whole
# --------------------------------------------------------------------------------------


def test_low_is_never_emitted_anywhere_on_the_providers_scale():
    """Swept across the whole of [0, 1], including well below the measured `T_LOW`.

    v1 has no LOW branch, and the sweep is what proves it rather than the absence of the
    word from the source.
    """
    levels = {
        evaluate(svd(score=index / 1000)).risk_level for index in range(0, 1001)
    }

    assert levels == {"MEDIUM", "HIGH"}
    assert "LOW" not in levels


def test_the_ruleset_emits_only_the_three_v1_levels():
    """Across every branch reachable in this module, including the degenerate ones."""
    decisions = [
        evaluate(None),
        evaluate(svd(score=0.999, status="FAILED")),
        evaluate(svd(score=0.999, provider_version="other")),
        evaluate(svd(score=None)),
        evaluate(svd(score=2.0)),
        evaluate(svd(score=math.nan)),
        evaluate(svd(score=0.999, total_clips=0)),
        evaluate(svd(score=0.0)),
        evaluate(svd(score=0.5)),
        evaluate(svd(score=0.98)),
        evaluate(svd(score=1.0)),
    ]

    assert {decision.risk_level for decision in decisions} <= {
        "HIGH",
        "MEDIUM",
        "UNKNOWN",
    }


def test_every_decision_carries_the_ruleset_and_calibration_that_made_it():
    """Including the UNKNOWNs. A decision with no trace is not explainable later."""
    for evidence in [None, svd(score=0.0), svd(score=0.99), svd(score=None)]:
        decision = evaluate(evidence)

        assert decision.rules_version == EXPECTED_RULES_VERSION
        assert decision.calibration_id == EXPECTED_CALIBRATION_ID


def test_only_the_four_documented_rules_can_fire():
    rules = {
        evaluate(evidence).rule_id
        for evidence in [
            None,
            svd(score=0.999, status="FAILED"),
            svd(score=None),
            svd(score=0.99),
            svd(score=0.1),
        ]
    }

    assert rules == {"R010", "R012", "R100", "R200"}


def test_the_same_evidence_always_yields_the_same_decision():
    """Stateless and deterministic: nothing accumulates between calls."""
    evidence = svd(score=0.9799999)

    assert evaluate(evidence) == evaluate(evidence) == evaluate(svd(score=0.9799999))


def test_a_decision_is_immutable_once_made():
    decision = evaluate(svd(score=0.99))

    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.risk_level = "MEDIUM"


# --------------------------------------------------------------------------------------
# Context isolation, in the shape of the input itself
# --------------------------------------------------------------------------------------


def test_the_engine_has_nowhere_to_put_a_context_signal():
    """The strongest isolation check there is: the fields do not exist.

    C2PA, active speaker and AASIST cannot influence a classification they cannot be passed
    to. This test fails the moment someone widens the input to admit one, which is exactly
    when the design decision behind P7 v1 would be quietly reversed.
    """
    fields = {field.name for field in dataclasses.fields(SvdEvidence)}

    assert fields == {
        "provider",
        "signal_type",
        "status",
        "provider_version",
        "score",
        "total_clips",
    }


def test_the_engine_takes_no_argument_but_the_direct_risk_evidence():
    """No session, no analysis id, no second signal: there is nothing else to pass it."""
    assert list(inspect.signature(evaluate).parameters) == ["evidence"]


# --------------------------------------------------------------------------------------
# Orchestration, against real PostgreSQL
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
def analysed(database):
    """An analysis whose evidence is already persisted and whose job is still `processing`.

    Exactly the state `conclude_job` is called in: the detectors have run, every signal row
    is committed, and the only thing left is the classification. The signals are written
    here rather than produced by driving the detectors, so the direct-risk evidence under
    test is exact and the context signals around it can be varied at will.
    """
    created = []

    def stage(*, svd_signal=True, context=(), **svd_kwargs):
        with SessionLocal() as session:
            analysis = Analysis(status="queued")
            session.add(analysis)
            session.flush()

            if svd_signal:
                session.add(persisted_signal(analysis.id, **svd_kwargs))
            for signal in context:
                signal.analysis_id = analysis.id
                session.add(signal)

            session.add(AnalysisJob(analysis_id=analysis.id, status="processing"))
            session.commit()
            created.append(analysis.id)

            return worker.ClaimedJob(
                job_id=session.query(AnalysisJob)
                .filter_by(analysis_id=analysis.id)
                .one()
                .id,
                analysis_id=analysis.id,
                original_storage_key="originals/unused",
                normalization_required=False,
                frame_rate=30.0,
            )

    yield stage

    with SessionLocal() as session:
        for analysis_id in created:
            # Signals and the job go with it through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


def persisted_signal(
    analysis_id,
    *,
    score=0.5,
    status="SUCCESS",
    provider_version=VALIDATED_FUNCTION_ID,
    total_clips=TOTAL_CLIPS,
) -> AnalysisSignal:
    """A synthetic-video row exactly as `detect_synthetic_video` writes one."""
    return AnalysisSignal(
        analysis_id=analysis_id,
        provider="nvidia",
        signal_type="synthetic_video",
        status=status,
        score=score,
        provider_version=provider_version,
        signal_metadata={"logit": LOGIT, "total_clips": total_clips},
    )


def context_signals(*, provenance=True, active_speaker=True, audio=True, failed=False):
    """The three signals that must never touch a classification, in a stated arrangement."""
    status = "FAILED" if failed else "SUCCESS"
    signals = []

    if provenance:
        signals.append(
            AnalysisSignal(
                provider="c2pa",
                signal_type="provenance",
                status=status,
                provider_version="0.90.14",
                signal_metadata={
                    "manifest_exists": not failed,
                    "validation_state": "Valid" if not failed else None,
                    "signature_issuer": "Test Signing Cert" if not failed else None,
                },
            )
        )
    if active_speaker:
        signals.append(
            AnalysisSignal(
                provider="nvidia",
                signal_type="active_speaker",
                status=status,
                provider_version="f286f937-05c4-454b-8312-fba67a2a6fa7",
                signal_metadata={"total_speaking_segments": 0 if failed else 12},
            )
        )
    if audio:
        signals.append(
            AnalysisSignal(
                provider="aasist",
                signal_type="audio_authenticity",
                status=status,
                provider_version="SpeechAntiSpoofingBenchmarks/AASIST@16774d45",
                signal_metadata={"total_audio_windows": 0 if failed else 3},
            )
        )

    return signals


def read_analysis(analysis_id) -> Analysis:
    with SessionLocal() as reader:
        return reader.query(Analysis).filter_by(id=analysis_id).one()


def read_job(job_id) -> AnalysisJob:
    with SessionLocal() as reader:
        return reader.query(AnalysisJob).filter_by(id=job_id).one()


def read_signals(analysis_id) -> dict[str, AnalysisSignal]:
    with SessionLocal() as reader:
        rows = reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).all()

    return {row.signal_type: row for row in rows}


@pytest.mark.integration
def test_the_engine_reads_the_evidence_the_database_actually_holds(analysed):
    """Not the values a detector returned in memory — the committed row.

    A classification derived from anything else could disagree with the evidence stored
    beside it, and nothing in the database would say so.
    """
    claimed = analysed(score=0.9931, total_clips=1829)

    with SessionLocal() as session:
        evidence = worker.persisted_svd_evidence(session, claimed.analysis_id)

    assert evidence == SvdEvidence(
        provider="nvidia",
        signal_type="synthetic_video",
        status="SUCCESS",
        provider_version=VALIDATED_FUNCTION_ID,
        score=0.9931,
        total_clips=1829,
    )


@pytest.mark.integration
def test_an_analysis_with_no_synthetic_video_signal_reads_back_as_absent(analysed):
    claimed = analysed(svd_signal=False, context=context_signals())

    with SessionLocal() as session:
        assert worker.persisted_svd_evidence(session, claimed.analysis_id) is None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kwargs", "risk_level", "rule_id"),
    [
        ({"score": 0.9979}, "HIGH", "R100"),
        ({"score": 0.98}, "HIGH", "R100"),
        ({"score": 0.9704}, "MEDIUM", "R200"),
        ({"score": 0.0}, "MEDIUM", "R200"),
        ({"score": 0.999, "status": "FAILED"}, "UNKNOWN", "R010"),
        ({"score": 0.999, "provider_version": "other-function"}, "UNKNOWN", "R010"),
        ({"score": None}, "UNKNOWN", "R012"),
        ({"score": 0.999, "total_clips": 0}, "UNKNOWN", "R012"),
    ],
)
def test_the_decision_and_its_whole_trace_are_persisted(
    analysed, kwargs, risk_level, rule_id
):
    """Level, ruleset, calibration and the rule that fired — all four, exactly.

    The trace is what makes a decision explainable after the thresholds move on, so each
    branch is checked at the stored row rather than at the return value.
    """
    claimed = analysed(**kwargs)

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == risk_level
    assert analysis.risk_rule_id == rule_id
    assert analysis.risk_rules_version == EXPECTED_RULES_VERSION
    assert analysis.risk_calibration_id == EXPECTED_CALIBRATION_ID


@pytest.mark.integration
def test_an_analysis_with_no_direct_risk_evidence_at_all_is_unknown(analysed):
    claimed = analysed(svd_signal=False, context=context_signals())

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"


@pytest.mark.integration
def test_risk_evaluation_is_the_last_step_before_the_job_completes(analysed):
    """Nothing is classified while the job is still in progress, and nothing completes
    without a classification."""
    claimed = analysed(score=0.5)

    before = read_analysis(claimed.analysis_id)
    assert before.status == "queued"
    assert before.risk_level is None
    assert before.risk_rules_version is None
    assert before.risk_calibration_id is None
    assert before.risk_rule_id is None

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    assert read_job(claimed.job_id).status == "completed"
    assert read_analysis(claimed.analysis_id).status == "completed"
    assert read_analysis(claimed.analysis_id).risk_level == "MEDIUM"


@pytest.mark.integration
def test_the_forensic_signals_are_untouched_by_the_classification(analysed):
    """The decision goes on the analysis. No signal row is rewritten to carry a verdict."""
    claimed = analysed(score=0.99, context=context_signals())

    before = {
        signal_type: (row.status, row.score, row.provider_version, row.risk_level)
        for signal_type, row in read_signals(claimed.analysis_id).items()
    }

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    after = {
        signal_type: (row.status, row.score, row.provider_version, row.risk_level)
        for signal_type, row in read_signals(claimed.analysis_id).items()
    }

    assert after == before
    # In particular, the per-signal column stays null on every one of them: risk is a
    # decision about the analysis, taken under a named ruleset, not a label per provider.
    assert all(row.risk_level is None for row in read_signals(claimed.analysis_id).values())


# --------------------------------------------------------------------------------------
# Context isolation, against stored evidence
# --------------------------------------------------------------------------------------

# Identical direct-risk evidence, each entry a different arrangement of everything else.
# Removed, present, failed, and every combination in between.
CONTEXT_ARRANGEMENTS = {
    "all present and successful": context_signals(),
    "all present and failed": context_signals(failed=True),
    "none at all": [],
    "no provenance": context_signals(provenance=False),
    "no active speaker": context_signals(active_speaker=False),
    "no audio authenticity": context_signals(audio=False),
    "provenance only": context_signals(active_speaker=False, audio=False),
    "provenance failed": context_signals(active_speaker=False, audio=False, failed=True),
    "active speaker only": context_signals(provenance=False, audio=False),
    "active speaker failed": context_signals(provenance=False, audio=False, failed=True),
    "audio only": context_signals(provenance=False, active_speaker=False),
    "audio failed": context_signals(provenance=False, active_speaker=False, failed=True),
}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("score", "risk_level", "rule_id"),
    [(0.9931, "HIGH", "R100"), (0.4646, "MEDIUM", "R200")],
)
def test_no_arrangement_of_the_context_signals_changes_the_classification(
    analysed, score, risk_level, rule_id
):
    """The isolation requirement, checked exhaustively against stored evidence.

    Twelve arrangements of C2PA, active speaker and AASIST — present, absent, successful,
    failed — over one unchanged synthetic-video score. All twelve must land on the same
    band by the same rule. Nothing here is averaged, voted on, weighted or combined, and
    this test is what would fail the moment something started to be.
    """
    decisions = {}

    for name, context in CONTEXT_ARRANGEMENTS.items():
        claimed = analysed(
            score=score,
            # Rebuilt per arrangement: an ORM instance cannot be attached twice.
            context=[
                AnalysisSignal(
                    provider=signal.provider,
                    signal_type=signal.signal_type,
                    status=signal.status,
                    provider_version=signal.provider_version,
                    signal_metadata=signal.signal_metadata,
                )
                for signal in context
            ],
        )

        with SessionLocal() as session:
            worker.conclude_job(session, claimed)

        analysis = read_analysis(claimed.analysis_id)
        decisions[name] = (
            analysis.risk_level,
            analysis.risk_rule_id,
            analysis.risk_rules_version,
            analysis.risk_calibration_id,
        )

    expected = (risk_level, rule_id, EXPECTED_RULES_VERSION, EXPECTED_CALIBRATION_ID)

    assert decisions == {name: expected for name in CONTEXT_ARRANGEMENTS}


@pytest.mark.integration
def test_context_signals_cannot_rescue_an_uncalibrated_provider(analysed):
    """UNKNOWN is not a gap for other evidence to fill in.

    An analysis with a perfect provenance chain, a full speaker timeline and clean audio
    windows still classifies UNKNOWN when the direct-risk signal came from a deployment the
    thresholds were never measured against.
    """
    claimed = analysed(
        score=0.9999,
        provider_version=f"{VALIDATED_FUNCTION_ID}-v2",
        context=context_signals(),
    )

    with SessionLocal() as session:
        worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------


@pytest.mark.integration
def test_duplicate_direct_risk_evidence_raises_rather_than_classifying(analysed):
    """A defect in how evidence was written must not be papered over with a verdict.

    Two synthetic-video signals on one analysis means something wrote evidence twice.
    Picking one and classifying it would publish a decision over half the evidence and hide
    the defect; raising sends it to the worker's own handler, which logs it and fails the
    job with the forensic rows intact.
    """
    claimed = analysed(score=0.5)

    with SessionLocal() as session:
        session.add(persisted_signal(claimed.analysis_id, score=0.99))
        session.commit()

    with SessionLocal() as session:
        with pytest.raises(MultipleResultsFound):
            worker.conclude_job(session, claimed)

    analysis = read_analysis(claimed.analysis_id)

    # No fabricated classification, and no half-finished completion either.
    assert analysis.risk_level is None
    assert analysis.status == "queued"
    assert read_job(claimed.job_id).status == "processing"
    # The evidence that provoked it survives untouched.
    with SessionLocal() as reader:
        assert (
            reader.query(AnalysisSignal)
            .filter_by(analysis_id=claimed.analysis_id, signal_type="synthetic_video")
            .count()
            == 2
        )


@pytest.mark.integration
def test_a_failed_job_records_no_classification(analysed):
    """`fail_job` writes no risk columns. Null is the absence of a decision, and it is
    deliberately not `UNKNOWN`, which is a conclusion an explicit rule reached."""
    claimed = analysed(score=0.99)

    with SessionLocal() as session:
        worker.fail_job(session, claimed, RuntimeError("storage is unreachable"))

    analysis = read_analysis(claimed.analysis_id)

    assert analysis.status == "failed"
    assert analysis.risk_level is None
    assert analysis.risk_rules_version is None
    assert analysis.risk_calibration_id is None
    assert analysis.risk_rule_id is None


def test_the_decision_type_is_the_only_thing_persisted():
    """Four fields, and no room for a provider figure to be copied alongside them.

    The signal rows remain the forensic record; a score duplicated onto the analysis could
    drift from the row it came from (rule 11).
    """
    fields = {field.name for field in dataclasses.fields(RiskDecision)}

    assert fields == {"risk_level", "rules_version", "calibration_id", "rule_id"}


def test_an_analysis_row_carries_no_detector_figures():
    """The risk columns on `analyses` are a decision trace, not a copy of the evidence."""
    risk_columns = {
        column.name for column in Analysis.__table__.columns if column.name.startswith("risk")
    }

    assert risk_columns == {
        "risk_level",
        "risk_rules_version",
        "risk_calibration_id",
        "risk_rule_id",
    }
