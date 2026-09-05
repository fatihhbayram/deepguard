"""Unit coverage for the derived Risk Trace (R7-T3).

What is under test is an *explanation* of a decision that has already been taken, so every
test here starts from persisted columns rather than from evidence to be classified. Three
properties carry the whole module and each has its own section below: the trace is read
under the ruleset version the row names and no other, it never re-runs the risk engine, and
it never turns a detector's silence into a statement about the media.

The historical fixtures are spelled out as literals rather than imported from
`app.risk_engine`, for the same reason `app.risk_trace` does not import that module: a test
that read today's thresholds could not tell the difference between a trace that is
version-aware and one that merely happens to agree with the current release.
"""

import ast
from pathlib import Path

import pytest

from app import risk_engine, risk_trace
from app.api.analyses import RiskTrace as RiskTraceResponse
from app.api.analyses import analysis_risk_trace
from app.api.public_v1.analyses import PublicAnalysis
from app.risk_trace import PersistedSignal, build_trace

V1_VERSION = "p7-v1.0.0"
V1_CALIBRATION = "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c"
V1_T_HIGH = 0.98

V2_VERSION = "r4-v2.0.0"
V2_CALIBRATION = "cab2ea262bb7e41cb87e49bdb3dad53ecd0f02248035a993f9fcb033363afd1e"

V3_VERSION = "r5-v3.0.0"
V3_CALIBRATION = "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"

SVD_T_HIGH = 0.9550971388816833
FACE_T_HIGH = 0.9867589175701141
LIP_T_HIGH = 0.22962537594139576

SVD_VERSION = "847b6e53-0133-452d-ab85-d7acf3ace723"
FACE_VERSION = (
    "tomas-gajarsky/facetorch-deepfake-efficientnet-b7@4acc494f37eb63d7457166eff2acb45c5b04b9a6"
)
LIP_VERSION = (
    "https://github.com/ahaliassos/LipForensics"
    "@d0bf5553bfb9676f1771d590472b26a3a76de894"
    "+4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
)


def svd(score=0.5, status="SUCCESS", total_clips=7, provider_version=SVD_VERSION):
    return PersistedSignal(
        provider="nvidia",
        signal_type="synthetic_video",
        status=status,
        provider_version=provider_version,
        score=score,
        metadata={"total_clips": total_clips},
    )


def face(score=0.5, status="SUCCESS", frames_scored=6, provider_version=FACE_VERSION):
    return PersistedSignal(
        provider="efficientnet-b7",
        signal_type="face_manipulation",
        status=status,
        provider_version=provider_version,
        score=score,
        metadata={"frames_scored": frames_scored},
    )


def lip(score=0.5, status="SUCCESS", windows_scored=3, provider_version=LIP_VERSION):
    return PersistedSignal(
        provider="lipforensics",
        signal_type="lip_forensics",
        status=status,
        provider_version=provider_version,
        score=score,
        metadata={"windows_scored": windows_scored},
    )


def trace(level, rule, version, calibration, **signals):
    return build_trace(
        risk_level=level,
        rule_id=rule,
        rules_version=version,
        calibration_id=calibration,
        signals={s.signal_type: s for s in signals.values()},
    )


def by_signal(result):
    return {c.signal: c for c in result.contributions}


# The persisted decision is copied, never recomputed. Whatever the evidence beside it says,
# the four columns come back as they were stored.


def test_the_persisted_decision_is_reported_unchanged():
    result = trace("MEDIUM", "R200", V1_VERSION, V1_CALIBRATION, svd=svd(score=0.79))

    assert result.risk_level == "MEDIUM"
    assert result.rule_id == "R200"
    assert result.rules_version == V1_VERSION
    assert result.calibration_id == V1_CALIBRATION


def test_a_stored_decision_is_reported_even_when_the_score_beside_it_would_flag_today():
    """A v1 row scoring above today's threshold but below its own is still MEDIUM.

    0.96 clears the R4-T1 operating point of 0.9551 and sits under P7's 0.98. The trace has
    no vote: the level is the persisted one, and the contribution is read against 0.98.
    """
    result = trace("MEDIUM", "R200", V1_VERSION, V1_CALIBRATION, svd=svd(score=0.96))

    assert result.risk_level == "MEDIUM"
    contribution = by_signal(result)["synthetic_video"]
    assert contribution.threshold == V1_T_HIGH
    assert contribution.condition == risk_trace.CONDITION_THRESHOLD_NOT_REACHED


def test_an_analysis_with_no_decision_has_no_trace():
    """Null `risk_level` is the absence of a decision, not `UNKNOWN`, and explains nothing."""
    assert (
        build_trace(
            risk_level=None,
            rule_id=None,
            rules_version=None,
            calibration_id=None,
            signals={},
        )
        is None
    )


# Version-aware semantics. One ruleset per version, and each explains only its own decisions.


def test_a_v1_decision_lists_only_the_detector_v1_read():
    """v1 classified from one detector. Rows from detectors that did not exist then are not
    part of the reasoning behind a decision taken without them."""
    result = trace(
        "HIGH",
        "R100",
        V1_VERSION,
        V1_CALIBRATION,
        svd=svd(score=0.99),
        face=face(score=0.999),
        lip=lip(score=0.9),
    )

    assert [c.signal for c in result.contributions] == ["synthetic_video"]
    contribution = result.contributions[0]
    assert contribution.threshold == V1_T_HIGH
    assert contribution.condition == risk_trace.CONDITION_THRESHOLD_REACHED
    assert contribution.role == risk_trace.ROLE_DECISIVE


def test_a_v2_decision_lists_both_of_the_detectors_v2_read():
    result = trace(
        "HIGH",
        "R102",
        V2_VERSION,
        V2_CALIBRATION,
        svd=svd(score=0.99),
        face=face(score=0.999),
        lip=lip(score=0.9),
    )

    assert [c.signal for c in result.contributions] == [
        "synthetic_video",
        "face_manipulation",
    ]
    assert by_signal(result)["synthetic_video"].threshold == SVD_T_HIGH
    assert by_signal(result)["face_manipulation"].threshold == FACE_T_HIGH
    assert all(
        c.condition == risk_trace.CONDITION_THRESHOLD_REACHED
        and c.role == risk_trace.ROLE_DECISIVE
        for c in result.contributions
    )


def test_a_v3_decision_lists_all_three_detectors_with_their_own_thresholds():
    result = trace(
        "HIGH",
        "R103",
        V3_VERSION,
        V3_CALIBRATION,
        svd=svd(score=0.1),
        face=face(score=0.2),
        lip=lip(score=0.91),
    )

    contributions = by_signal(result)
    assert set(contributions) == {"synthetic_video", "face_manipulation", "lip_forensics"}
    assert contributions["lip_forensics"].threshold == LIP_T_HIGH
    assert contributions["lip_forensics"].condition == (
        risk_trace.CONDITION_THRESHOLD_REACHED
    )
    assert contributions["lip_forensics"].role == risk_trace.ROLE_DECISIVE
    # The two that did not reach their own thresholds decided nothing, and neither of them
    # held the mouth-dynamics finding back — the rules are disjunctive.
    assert contributions["synthetic_video"].role == risk_trace.ROLE_CONSIDERED
    assert contributions["face_manipulation"].role == risk_trace.ROLE_CONSIDERED


def test_the_same_rule_id_means_different_things_under_different_versions():
    """`R200` is one detector below its threshold in v1, two in v2, three in v3."""
    summaries = {
        version: build_trace(
            risk_level="MEDIUM",
            rule_id="R200",
            rules_version=version,
            calibration_id=calibration,
            signals={},
        ).rule_summary
        for version, calibration in (
            (V1_VERSION, V1_CALIBRATION),
            (V2_VERSION, V2_CALIBRATION),
            (V3_VERSION, V3_CALIBRATION),
        )
    }

    assert len(set(summaries.values())) == 3
    assert "single available signal" in summaries[V1_VERSION]
    assert "Both calibrated detectors" in summaries[V2_VERSION]
    assert "All three calibrated detectors" in summaries[V3_VERSION]


def test_the_same_high_rule_id_is_read_under_the_version_that_fired_it():
    """`R102` is "both detectors" under v2 and "two or more" under v3."""
    v2 = trace("HIGH", "R102", V2_VERSION, V2_CALIBRATION)
    v3 = trace("HIGH", "R102", V3_VERSION, V3_CALIBRATION)

    assert v2.rule_summary != v3.rule_summary
    assert "Both" in v2.rule_summary
    assert "Two or more" in v3.rule_summary


def test_the_same_score_is_read_against_the_threshold_of_the_version_that_decided():
    """0.96 reached the operating point R4-T1 measured and did not reach P7's."""
    under_v1 = trace("MEDIUM", "R200", V1_VERSION, V1_CALIBRATION, svd=svd(score=0.96))
    under_v3 = trace("HIGH", "R100", V3_VERSION, V3_CALIBRATION, svd=svd(score=0.96))

    assert by_signal(under_v1)["synthetic_video"].condition == (
        risk_trace.CONDITION_THRESHOLD_NOT_REACHED
    )
    assert by_signal(under_v3)["synthetic_video"].condition == (
        risk_trace.CONDITION_THRESHOLD_REACHED
    )


def test_a_rule_id_that_version_never_had_is_left_unexplained():
    """v1 had no `R103`. Inventing a meaning for it would be a guess about the rules."""
    result = trace("HIGH", "R103", V1_VERSION, V1_CALIBRATION, svd=svd(score=0.99))

    assert result.rule_id == "R103"
    assert result.rule_summary is None


# Today's numbers cannot reach a historical decision.


def test_changing_the_current_thresholds_does_not_reinterpret_a_stored_analysis(
    monkeypatch,
):
    """The engine's constants are moved under the trace's feet; nothing shifts.

    This is the property the whole module exists for. A recalibration — or a fixture, or a
    typo — that moved `SVD_T_HIGH` must not change what an already-decided analysis is
    reported to have been decided from.
    """
    before = trace("MEDIUM", "R201", V3_VERSION, V3_CALIBRATION, svd=svd(score=0.5))

    monkeypatch.setattr(risk_engine, "SVD_T_HIGH", 0.1)
    monkeypatch.setattr(risk_engine, "FACE_T_HIGH", 0.1)
    monkeypatch.setattr(risk_engine, "LIP_T_HIGH", 0.1)
    monkeypatch.setattr(risk_engine, "CALIBRATION_ID", "0" * 64)
    monkeypatch.setattr(risk_engine, "RULES_VERSION", "r9-v9.0.0")

    after = trace("MEDIUM", "R201", V3_VERSION, V3_CALIBRATION, svd=svd(score=0.5))

    assert after == before
    assert by_signal(after)["synthetic_video"].threshold == SVD_T_HIGH
    assert by_signal(after)["synthetic_video"].condition == (
        risk_trace.CONDITION_THRESHOLD_NOT_REACHED
    )


def test_the_trace_module_does_not_import_the_risk_engine():
    """Structural, not behavioural: the dependency cannot exist, so it cannot be used."""
    source = Path(risk_trace.__file__).read_text()
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not any("risk_engine" in name for name in imported)


def test_building_a_trace_never_evaluates(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the trace re-ran the risk engine")

    monkeypatch.setattr(risk_engine, "evaluate", refuse)

    result = trace(
        "HIGH",
        "R100",
        V3_VERSION,
        V3_CALIBRATION,
        svd=svd(score=0.99),
        face=face(score=0.5),
        lip=lip(score=0.1),
    )

    assert result.risk_level == "HIGH"


def test_the_current_ruleset_entry_matches_the_engine_that_writes_it():
    """A drift guard, and the one place the two modules are compared.

    If the engine ships a new ruleset version, this fails until `app.risk_trace` learns it —
    which is the intended outcome: an unknown version degrades safely, but shipping one
    unnoticed would leave every new decision unexplained.
    """
    current = risk_trace.RULESETS[risk_engine.RULES_VERSION]

    assert current.calibration_id == risk_engine.CALIBRATION_ID
    thresholds = {s.signal_type: s.threshold for s in current.signals}
    assert thresholds == {
        risk_engine.SVD_SIGNAL_TYPE: risk_engine.SVD_T_HIGH,
        risk_engine.FACE_SIGNAL_TYPE: risk_engine.FACE_T_HIGH,
        risk_engine.LIP_SIGNAL_TYPE: risk_engine.LIP_T_HIGH,
    }
    rule_ids = {
        risk_engine.RULE_NO_CALIBRATED_EVIDENCE,
        risk_engine.RULE_INVALID_CALIBRATED_EVIDENCE,
        risk_engine.RULE_HIGH_SYNTHETIC_VIDEO,
        risk_engine.RULE_HIGH_FACE_MANIPULATION,
        risk_engine.RULE_HIGH_MULTIPLE_SOURCES,
        risk_engine.RULE_HIGH_MOUTH_DYNAMICS,
        risk_engine.RULE_INDETERMINATE_ALL_SOURCES,
        risk_engine.RULE_INDETERMINATE_PARTIAL_SOURCES,
    }
    assert rule_ids <= set(current.rules)


# Silence. A detector that contributed no reading is reported as exactly that, and never as
# a finding about the media.


@pytest.mark.parametrize(
    ("signal", "reason"),
    [
        (None, risk_trace.UNAVAILABLE_NO_READING),
        (face(status="FAILED"), risk_trace.UNAVAILABLE_DETECTOR_DID_NOT_REPORT),
        (face(status="TIMEOUT"), risk_trace.UNAVAILABLE_DETECTOR_DID_NOT_REPORT),
        (
            face(provider_version="some-other-checkpoint"),
            risk_trace.UNAVAILABLE_UNCALIBRATED_DEPLOYMENT,
        ),
        (face(score=None), risk_trace.UNAVAILABLE_UNREADABLE_FIGURES),
        (face(frames_scored=0), risk_trace.UNAVAILABLE_UNREADABLE_FIGURES),
        (face(frames_scored="six"), risk_trace.UNAVAILABLE_UNREADABLE_FIGURES),
        (face(score=1.4), risk_trace.UNAVAILABLE_UNREADABLE_FIGURES),
        (face(score=float("nan")), risk_trace.UNAVAILABLE_UNREADABLE_FIGURES),
    ],
)
def test_a_detector_that_contributed_no_reading_is_unavailable(signal, reason):
    signals = {} if signal is None else {"face_manipulation": signal}
    result = build_trace(
        risk_level="MEDIUM",
        rule_id="R201",
        rules_version=V3_VERSION,
        calibration_id=V3_CALIBRATION,
        signals=signals,
    )

    contribution = by_signal(result)["face_manipulation"]
    assert contribution.condition == risk_trace.CONDITION_UNAVAILABLE
    assert contribution.unavailable_reason == reason
    # The two things it must never become: a threshold comparison, or a reason for a level.
    assert contribution.condition != risk_trace.CONDITION_THRESHOLD_NOT_REACHED
    assert contribution.role == risk_trace.ROLE_CONSIDERED
    # Neither figure survives into the trace. The decision could not use this reading, so
    # staging a number against an operating point would imply a comparison nothing made.
    assert contribution.score is None
    assert contribution.threshold is None
    # What is kept is the identity of the detector that did not contribute.
    assert contribution.signal == "face_manipulation"
    assert contribution.provider == "efficientnet-b7"


def test_an_unavailable_detector_never_becomes_a_finding_of_absence(monkeypatch):
    """No rendering of a silent detector may read as evidence about the media."""
    result = build_trace(
        risk_level="UNKNOWN",
        rule_id="R010",
        rules_version=V3_VERSION,
        calibration_id=V3_CALIBRATION,
        signals={"face_manipulation": face(status="FAILED")},
    )

    rendered = RiskTraceResponse.model_validate(result, from_attributes=True)
    # The interpretive fields only. A provider's own identity is not this module's wording —
    # `facetorch-deepfake-efficientnet-b7` is the checkpoint's name, recorded as it is.
    wording = " ".join(
        part.lower()
        for part in [rendered.risk_level, rendered.rule_summary or ""]
        + [
            f"{c.condition} {c.unavailable_reason or ''} {c.role}"
            for c in rendered.contributions
        ]
    )

    for word in ("authentic", "genuine", "real", "fake", "unmanipulated", "clean", "safe"):
        assert word not in wording


def test_the_trace_module_states_no_truth_labels():
    """The vocabulary is fixed at three levels; no fourth label exists to fall back to."""
    assert risk_trace.RISK_HIGH == "HIGH"
    assert risk_trace.RISK_MEDIUM == "MEDIUM"
    assert risk_trace.RISK_UNKNOWN == "UNKNOWN"

    levels = {
        value
        for name, value in vars(risk_trace).items()
        if name.startswith("RISK_") and isinstance(value, str)
    }
    assert levels == {"HIGH", "MEDIUM", "UNKNOWN"}


def test_unknown_stays_unknown():
    """`UNKNOWN` is a decision with a rule behind it, and the trace explains it as one."""
    result = build_trace(
        risk_level="UNKNOWN",
        rule_id="R012",
        rules_version=V3_VERSION,
        calibration_id=V3_CALIBRATION,
        signals={"synthetic_video": svd(score=2.0)},
    )

    assert result.risk_level == "UNKNOWN"
    assert "could not be read" in result.rule_summary
    assert all(c.role == risk_trace.ROLE_CONSIDERED for c in result.contributions)


def test_no_unavailable_contribution_carries_a_figure_under_any_ruleset():
    """The invariant, swept across all three versions and every way a reading can be lost.

    A persisted score that the engine refused to read is not evidence the decision rested
    on, and a threshold beside it would say a comparison happened. Both are withheld here
    even when the signal row holds them; the raw values stay on the analysis's own signal
    evidence, where they are a provider's output rather than a term in a decision.
    """
    lost = [
        None,
        svd(status="FAILED", score=0.99),
        svd(status="TIMEOUT", score=0.99),
        svd(provider_version="redeployed-function-id", score=0.99),
        svd(score=None),
        svd(score=0.99, total_clips=0),
        svd(score=0.99, total_clips="seven"),
        svd(score=7.0),
        svd(score=float("nan")),
    ]

    for version, calibration in (
        (V1_VERSION, V1_CALIBRATION),
        (V2_VERSION, V2_CALIBRATION),
        (V3_VERSION, V3_CALIBRATION),
    ):
        for signal in lost:
            result = build_trace(
                risk_level="UNKNOWN",
                rule_id="R010",
                rules_version=version,
                calibration_id=calibration,
                signals={} if signal is None else {"synthetic_video": signal},
            )
            contribution = by_signal(result)["synthetic_video"]

            assert contribution.condition == risk_trace.CONDITION_UNAVAILABLE
            assert contribution.score is None
            assert contribution.threshold is None
            assert contribution.unavailable_reason is not None
            assert contribution.signal == "synthetic_video"
            assert contribution.provider == "nvidia"


def test_an_available_reading_still_carries_its_figure_and_its_threshold():
    """The withholding is about unavailable readings only; a used reading is shown in full."""
    result = trace("HIGH", "R100", V3_VERSION, V3_CALIBRATION, svd=svd(score=0.99))
    contribution = by_signal(result)["synthetic_video"]

    assert contribution.score == 0.99
    assert contribution.threshold == SVD_T_HIGH


def test_the_serialized_contract_withholds_the_figures_too():
    """Asserted on the response model, since that is what R7-T4 will render."""
    result = build_trace(
        risk_level="UNKNOWN",
        rule_id="R010",
        rules_version=V3_VERSION,
        calibration_id=V3_CALIBRATION,
        signals={"face_manipulation": face(status="FAILED", score=0.999)},
    )

    rendered = RiskTraceResponse.model_validate(result, from_attributes=True)
    payload = {c.signal: c for c in rendered.contributions}["face_manipulation"]

    assert payload.condition == "unavailable"
    assert payload.score is None
    assert payload.threshold is None
    assert payload.provider == "efficientnet-b7"
    assert payload.unavailable_reason == "detector_did_not_report"


# Legacy and incomplete metadata degrade rather than guess.


def test_an_unknown_ruleset_version_is_reported_without_interpretation():
    result = build_trace(
        risk_level="HIGH",
        rule_id="R100",
        rules_version="p3-v0.9.0",
        calibration_id="f" * 64,
        signals={"synthetic_video": svd(score=0.99)},
    )

    assert result.risk_level == "HIGH"
    assert result.rules_version == "p3-v0.9.0"
    assert result.rule_summary is None
    assert result.contributions == ()
    assert result.interpreted is False


def test_a_missing_ruleset_version_is_reported_without_interpretation():
    result = build_trace(
        risk_level="UNKNOWN",
        rule_id=None,
        rules_version=None,
        calibration_id=None,
        signals={"synthetic_video": svd(score=0.99)},
    )

    assert result.risk_level == "UNKNOWN"
    assert result.rule_summary is None
    assert result.interpreted is False


def test_a_calibration_identity_that_is_not_the_versions_own_withholds_the_thresholds():
    """The rule meanings survive; the numbers do not.

    A decision naming a calibration this ruleset was not measured under cannot be shown
    against this ruleset's thresholds, and showing them anyway is exactly the silent
    substitution the task forbids.
    """
    result = build_trace(
        risk_level="HIGH",
        rule_id="R100",
        rules_version=V3_VERSION,
        calibration_id="0" * 64,
        signals={"synthetic_video": svd(score=0.99)},
    )

    contribution = by_signal(result)["synthetic_video"]
    assert contribution.threshold is None
    assert contribution.condition == risk_trace.CONDITION_NOT_INTERPRETED
    assert contribution.unavailable_reason == (
        risk_trace.UNAVAILABLE_THRESHOLD_UNRESOLVED
    )
    assert contribution.score == 0.99
    assert result.rule_summary is not None
    assert result.interpreted is False


# The API contract.


def test_the_public_contract_is_not_widened_by_the_internal_trace():
    """R7-T3 is an internal contract. `/api/public/v1` renders its own model and keeps it."""
    assert "risk_trace" not in PublicAnalysis.model_fields


def test_the_serialization_layer_reads_the_row_it_was_given(monkeypatch):
    """`analysis_risk_trace` maps joined columns onto the trace and classifies nothing."""
    from tests.test_analysis_listing import listing_row

    monkeypatch.setattr(
        risk_engine,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("the read path re-ran the risk engine"),
    )

    rendered = analysis_risk_trace(listing_row())

    assert isinstance(rendered, RiskTraceResponse)
    # The fixture row is a `p7-v1.0.0` decision, so one detector is in scope.
    assert [c.signal for c in rendered.contributions] == ["synthetic_video"]
    assert rendered.contributions[0].threshold == V1_T_HIGH


def test_a_row_with_no_decision_carries_no_trace():
    from tests.test_analysis_listing import listing_row

    assert (
        analysis_risk_trace(
            listing_row(
                risk_level=None,
                risk_rule_id=None,
                risk_rules_version=None,
                risk_calibration_id=None,
            )
        )
        is None
    )


def test_a_missing_signal_row_reaches_the_trace_as_an_unavailable_reading():
    from tests.test_analysis_listing import listing_row

    rendered = analysis_risk_trace(
        listing_row(
            risk_rules_version=V3_VERSION,
            risk_calibration_id=V3_CALIBRATION,
            risk_level="MEDIUM",
            risk_rule_id="R201",
            lip_forensics_provider=None,
            lip_forensics_signal_type=None,
            lip_forensics_status=None,
            lip_forensics_score=None,
            lip_forensics_provider_version=None,
            lip_forensics_metadata=None,
        )
    )

    lip_contribution = next(
        c for c in rendered.contributions if c.signal == "lip_forensics"
    )
    assert lip_contribution.condition == risk_trace.CONDITION_UNAVAILABLE
    assert lip_contribution.unavailable_reason == risk_trace.UNAVAILABLE_NO_READING
    assert lip_contribution.score is None
    assert lip_contribution.threshold is None
