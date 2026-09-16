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
import re
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

# v4 is decided under v3's calibration identity, deliberately and as a matter of contract:
# R7-T6 changed the rules and not one measurement, so the artifacts a v4 decision rests on are
# the artifacts a v3 decision rested on. The two versions are told apart by `rules_version`,
# which is why a stored decision carries both columns. Spelled as its own name rather than
# reusing `V3_CALIBRATION` so the tests below say which version they mean.
V4_VERSION = "r7-v4.0.0"
V4_CALIBRATION = "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"

# v5 shares that calibration identity for the same reason v4 shares v3's: R9-T2 rewrote the
# vocabulary and the coverage arithmetic and moved no threshold, so the artifacts a v5 decision
# rests on are the artifacts a v4 decision rested on. `rules_version` is what tells them apart,
# and it is the only field a consumer may resolve the verdict vocabulary through.
V5_VERSION = "r9-v5.0.0"
V5_CALIBRATION = "a74f6b9dbc64cead34cb8e31a03791228cdeb19497e8e5e0bc1a67c0337fc5f7"

MANIPULATION_DETECTED = "MANIPULATION_DETECTED"
NO_SIGNAL = "NO_CALIBRATED_MANIPULATION_SIGNAL"
INCONCLUSIVE = "INCONCLUSIVE"

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


def test_a_v4_decision_lists_the_same_three_detectors_with_the_same_thresholds():
    """v4 reads everything v3 read, against the numbers v3 read it against.

    The withdrawal of `R103` took a rule out of the ruleset and no detector out of the trace.
    LipForensics is still listed, still banded against R5-T3's operating point, and a score
    above it is still reported as the crossing it is — `threshold_reached` is a statement about
    a number and a threshold, and that statement did not stop being true.
    """
    result = trace(
        "MEDIUM",
        "R200",
        V4_VERSION,
        V4_CALIBRATION,
        svd=svd(score=0.1),
        face=face(score=0.2),
        lip=lip(score=0.91),
    )

    contributions = by_signal(result)
    assert result.interpreted is True
    assert set(contributions) == {"synthetic_video", "face_manipulation", "lip_forensics"}
    assert contributions["synthetic_video"].threshold == SVD_T_HIGH
    assert contributions["face_manipulation"].threshold == FACE_T_HIGH
    assert contributions["lip_forensics"].threshold == LIP_T_HIGH
    assert contributions["lip_forensics"].condition == (
        risk_trace.CONDITION_THRESHOLD_REACHED
    )


def test_a_v4_mouth_dynamics_crossing_is_never_reported_as_decisive():
    """The role field, on the one arrangement where v3 and v4 disagree about it.

    A HIGH taken from the synthetic-video detector, with the mouth-dynamics score above its own
    threshold beside it. Under v3 both were `decisive` — both rules could take the level, and
    `R102` said so. Under v4 only NVIDIA's finding could have produced this level, so only
    NVIDIA's is decisive, and the crossing is `considered`: read, reported, and not credited
    with a decision it was not entitled to make.
    """
    result = by_signal(
        trace(
            "HIGH",
            "R100",
            V4_VERSION,
            V4_CALIBRATION,
            svd=svd(score=0.99),
            face=face(score=0.2),
            lip=lip(score=0.91),
        )
    )

    assert result["synthetic_video"].condition == risk_trace.CONDITION_THRESHOLD_REACHED
    assert result["synthetic_video"].role == risk_trace.ROLE_DECISIVE
    assert result["lip_forensics"].condition == risk_trace.CONDITION_THRESHOLD_REACHED
    assert result["lip_forensics"].role == risk_trace.ROLE_CONSIDERED
    assert result["face_manipulation"].role == risk_trace.ROLE_CONSIDERED


def test_the_same_evidence_reads_as_decisive_under_v3_and_considered_under_v4():
    """One stored mouth-dynamics crossing, two versions, two honest readings of it.

    This is the whole of what `decisional` buys, in one comparison. The score, the threshold,
    the deployment and the calibration identity are identical on both sides; the only thing that
    differs is the ruleset version the decision was taken under, and that is exactly the thing
    that decides whether this detector could have concluded anything.
    """
    stored = dict(svd=svd(score=0.1), face=face(score=0.2), lip=lip(score=0.91))

    under_v3 = by_signal(trace("HIGH", "R103", V3_VERSION, V3_CALIBRATION, **stored))
    under_v4 = by_signal(trace("MEDIUM", "R200", V4_VERSION, V4_CALIBRATION, **stored))

    assert under_v3["lip_forensics"].score == under_v4["lip_forensics"].score
    assert under_v3["lip_forensics"].threshold == under_v4["lip_forensics"].threshold
    assert under_v3["lip_forensics"].condition == under_v4["lip_forensics"].condition

    assert under_v3["lip_forensics"].role == risk_trace.ROLE_DECISIVE
    assert under_v4["lip_forensics"].role == risk_trace.ROLE_CONSIDERED


def test_r103_exists_in_v3_and_in_no_other_version():
    """The immutability requirement, stated as the smallest fact that carries it.

    A stored `r5-v3.0.0` row naming `R103` must keep reading as v3's sentence for as long as
    that row exists. Two things protect it: v3's table is never edited, and no later version
    may reuse the id for anything. The second is what this asserts — a v4 rule wearing `R103`
    would silently rewrite every historical decision that names it.
    """
    assert "R103" in risk_trace.RULESET_V3.rules
    assert risk_trace.RULESET_V3.rules["R103"] == (
        "The calibrated mouth-dynamics score reached its measured threshold."
    )

    for ruleset in risk_trace.RULESETS.values():
        if ruleset.rules_version != V3_VERSION:
            assert "R103" not in ruleset.rules


def test_a_stored_r103_decision_still_reads_exactly_as_v3_wrote_it():
    """The persisted row, rendered after R7-T6 shipped. Nothing about it moved.

    The level, the rule, the summary, the threshold the score was read against and the role the
    detector played are all v3's. This analysis is not re-evaluated, not re-explained and not
    migrated: it was decided under those rules, and the trace's job is to say what they said.
    """
    result = trace(
        "HIGH",
        "R103",
        V3_VERSION,
        V3_CALIBRATION,
        svd=svd(score=0.1),
        face=face(score=0.2),
        lip=lip(score=0.91),
    )

    assert result.risk_level == "HIGH"
    assert result.rule_id == "R103"
    assert result.rules_version == V3_VERSION
    assert result.interpreted is True
    assert result.rule_summary == (
        "The calibrated mouth-dynamics score reached its measured threshold."
    )

    lip_contribution = by_signal(result)["lip_forensics"]
    assert lip_contribution.threshold == LIP_T_HIGH
    assert lip_contribution.condition == risk_trace.CONDITION_THRESHOLD_REACHED
    assert lip_contribution.role == risk_trace.ROLE_DECISIVE


def test_the_v3_ruleset_entry_is_byte_for_byte_what_it_was():
    """A guard on the historical table itself, not on one trace built from it.

    Every field v3 was measured and written with, restated as a literal. R7-T6 added a version
    beside it and edited nothing inside it; if a later change ever does, this fails rather than
    quietly re-explaining every stored v3 decision.
    """
    v3 = risk_trace.RULESET_V3

    assert v3.rules_version == V3_VERSION
    assert v3.calibration_id == V3_CALIBRATION
    assert {s.signal_type: s.threshold for s in v3.signals} == {
        "synthetic_video": SVD_T_HIGH,
        "face_manipulation": FACE_T_HIGH,
        "lip_forensics": LIP_T_HIGH,
    }
    # All three could take a decision under v3. That is what `R103` was.
    assert all(s.decisional for s in v3.signals)
    assert set(v3.rules) == {
        "R010",
        "R012",
        "R100",
        "R101",
        "R102",
        "R103",
        "R200",
        "R201",
    }


def test_only_the_mouth_dynamics_detector_is_non_decisional_and_only_after_v3():
    """Which detector each version could take its top conclusion from, across the whole table.

    v5 inherits v4's answer exactly: R9-T2 changed the vocabulary and the coverage arithmetic and
    did not readmit the mouth-dynamics detector to the decision. v3 remains the one version it
    could decide under, and that entry is never edited.
    """
    decisional = {
        version: {s.signal_type for s in ruleset.signals if s.decisional}
        for version, ruleset in risk_trace.RULESETS.items()
    }

    assert decisional == {
        V1_VERSION: {"synthetic_video"},
        V2_VERSION: {"synthetic_video", "face_manipulation"},
        V3_VERSION: {"synthetic_video", "face_manipulation", "lip_forensics"},
        V4_VERSION: {"synthetic_video", "face_manipulation"},
        V5_VERSION: {"synthetic_video", "face_manipulation"},
    }


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
            (V4_VERSION, V4_CALIBRATION),
        )
    }

    assert len(set(summaries.values())) == 4
    assert "single available signal" in summaries[V1_VERSION]
    assert "Both calibrated detectors" in summaries[V2_VERSION]
    assert "All three calibrated detectors were readable and none reached" in (
        summaries[V3_VERSION]
    )
    # v4's `R200` is the one a mouth-dynamics crossing can sit inside. Saying "none reached its
    # threshold" on such a decision would be false, and this is where that would show.
    assert "neither detector this ruleset takes a decision from" in summaries[V4_VERSION]
    assert "does not decide this level" in summaries[V4_VERSION]


def test_the_same_high_rule_id_is_read_under_the_version_that_fired_it():
    """`R102` is "both detectors" in v2, "two or more" in v3, "both deciding ones" in v4.

    Three statements behind four characters, and the v3/v4 pair is the sharp one: under v3 a
    mouth-dynamics crossing could be one of the two that agreed, and under v4 it cannot be.
    A stored v3 `R102` must keep meaning what it meant.
    """
    v2 = trace("HIGH", "R102", V2_VERSION, V2_CALIBRATION)
    v3 = trace("HIGH", "R102", V3_VERSION, V3_CALIBRATION)
    v4 = trace("HIGH", "R102", V4_VERSION, V4_CALIBRATION)

    assert len({v2.rule_summary, v3.rule_summary, v4.rule_summary}) == 3
    assert "Both" in v2.rule_summary
    assert "Two or more" in v3.rule_summary
    assert "Both detectors this ruleset takes a decision from" in v4.rule_summary


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
        risk_engine.RULE_INDETERMINATE_ALL_SOURCES,
        risk_engine.RULE_INDETERMINATE_PARTIAL_SOURCES,
    }
    assert rule_ids <= set(current.rules)

    # Every rule the engine can fire *under this ruleset version* is explained by this version's
    # table, and nothing else is. The equality is the half that matters after R7-T6: `R103` is
    # not among the engine's rule constants any more, and an entry for it here would be a
    # sentence no decision can name.
    #
    # Scoped to the v4 table by the same prefix the engine separates the two tables with. The
    # `r9-v5.0.0` constants R9-T2 added belong to `evaluate_v5`, which has no caller and writes
    # no row yet; the trace entry that will explain them is R9-T4's, and comparing them against
    # `RULESETS[r7-v4.0.0]` would assert that one ruleset's rules are the other's.
    declared = {
        value
        for name, value in vars(risk_engine).items()
        if name.startswith("RULE_")
        and not name.startswith("RULE_V5_")
        and isinstance(value, str)
    }
    assert declared == set(current.rules)


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


# --------------------------------------------------------------------------------------
# The report's own version-aware rendering (R7-T6)
# --------------------------------------------------------------------------------------
#
# The report is a second reader of the same persisted columns, and it carries its own copy of
# "which detectors could this version decide from" — a rationale table keyed by ruleset, and two
# `decides` expressions that set what each detector's panel claims. Those are presentation, not
# rules, but they are the sentences a reader actually sees, so a version the report has never
# heard of is a report that describes the decision wrongly while looking complete.
#
# Asserted from the sources rather than by rendering them, which is how this repository already
# checks properties of the web application from the backend suite (`test_shadow_mode`). What is
# being checked is a mapping, and the mapping is visible in the text.

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
WEB_ANALYSIS = WEB_ROOT / "app" / "analysis.ts"
WEB_REPORT = WEB_ROOT / "app" / "app" / "report" / "[id]" / "page.tsx"

requires_web = pytest.mark.skipif(
    not WEB_ROOT.exists(), reason="the web application is not present"
)


def _decides_expressions(source: str) -> list[str]:
    """Every `const decides = …;` in the report, in the order the file declares them.

    Two of them: the face-manipulation panel's and the mouth-dynamics panel's, in that order.
    Each says which rulesets that panel may describe as having decided anything.
    """
    return [chunk.split(";", 1)[0] for chunk in source.split("const decides =")[1:]]


def _v4_rationale_rule_ids(source: str) -> set[str]:
    """The rule ids `V4_RATIONALES` gives a description for."""
    block = source.split("const V4_RATIONALES", 1)[1].split("\n};", 1)[0]
    return set(re.findall(r"^  (R\d{3}): \{", block, flags=re.MULTILINE))


@requires_web
def test_the_report_knows_the_ruleset_the_engine_now_writes():
    """A version the report has no entry for renders every new analysis unexplained."""
    source = WEB_ANALYSIS.read_text(encoding="utf-8")

    assert f'export const RULES_VERSION_V4 = "{risk_engine.RULES_VERSION}";' in source
    assert "[RULES_VERSION_V4]: V4_RATIONALES," in source

    # The historical entries are still registered, and still keyed by their own versions.
    for constant, version in (
        ("RULES_VERSION_V1", V1_VERSION),
        ("RULES_VERSION_V2", V2_VERSION),
        ("RULES_VERSION_V3", V3_VERSION),
    ):
        assert f'export const {constant} = "{version}";' in source
        assert f"[{constant}]: V" in source


@requires_web
def test_the_reports_v4_rationales_cover_every_rule_v4_can_fire_and_no_others():
    """The report explains exactly the rules that exist, which is what excludes `R103`.

    An entry for a rule the engine cannot fire would be dead prose; a missing entry is a real
    decision rendered with no explanation beside it. `R103` is the one that must be absent: it
    is v3's rule id, its description lives in `V3_RATIONALES` for the stored rows that name it,
    and a v4 entry would be a second and contradictory meaning for the same four characters.
    """
    rule_ids = _v4_rationale_rule_ids(WEB_ANALYSIS.read_text(encoding="utf-8"))

    declared = {
        value
        for name, value in vars(risk_engine).items()
        # v4's own rule ids and no others. `RULE_V5_*` names the `r9-v5.0.0` table, which is a
        # separate vocabulary with a disjoint set of ids (`R9-100` and the rest); an entry for
        # one of them in `V4_RATIONALES` would explain a v4 decision with a v5 sentence.
        if name.startswith("RULE_")
        and not name.startswith("RULE_V5_")
        and isinstance(value, str)
    }

    assert rule_ids == declared
    assert "R103" not in rule_ids
    assert not any(rule_id.startswith("R9-") for rule_id in rule_ids)


@requires_web
def test_the_report_treats_the_two_deciding_detectors_as_deciding_under_v4():
    """Which panels may say they decided, per ruleset, read off the report itself.

    The face-manipulation panel decides under v2, v3 and v4 — R7-T6 changed nothing about that
    detector. The mouth-dynamics panel decides under v3 alone: not under the rulesets that had
    no threshold for it, and not under v4, which measured one and withdrew it.
    """
    face_decides, lip_decides = _decides_expressions(
        WEB_REPORT.read_text(encoding="utf-8")
    )

    assert "RULES_VERSION_V2" in face_decides
    assert "RULES_VERSION_V3" in face_decides
    assert "RULES_VERSION_V4" in face_decides

    assert "RULES_VERSION_V3" in lip_decides
    assert "RULES_VERSION_V4" not in lip_decides
    assert "RULES_VERSION_V2" not in lip_decides


# --- R9-T5: the operational summary the report opens with ------------------------------------
#
# The same source-assertion approach as the block above, for the same reason: what is being
# checked is a mapping and a refusal, and both are visible in the text. Rendering the component
# would check React; reading it checks the property the component exists to guarantee.

# The three sentences, exactly as R9-T5 locked them. Transcribed here so that the test owns an
# independent copy: a test that read the wording out of the file it is checking would pass on any
# wording at all, including a reassuring rewrite of it.
LOCKED_WORDING = {
    MANIPULATION_DETECTED: (
        "Manipulation detected",
        "One or more calibrated decision detectors reached their operating point.",
        "This identifies calibrated manipulation evidence. It does not establish the original "
        "source or provenance of the media.",
    ),
    NO_SIGNAL: (
        "No calibrated manipulation signal detected",
        "The completed decision detectors produced no threshold-reaching manipulation signal.",
        "This does not prove that the media is authentic, genuine, or source-verified.",
    ),
    INCONCLUSIVE: (
        "Inconclusive",
        "InspectRoot could not complete the calibrated automated assessment because one or more "
        "required decision detectors did not produce a usable reading.",
        "This result is neither evidence of manipulation nor evidence of authenticity.",
    ),
}


def _operational_summary(source: str) -> str:
    """The body of the `OperationalSummary` component, from its signature to the next one."""
    body = source.split("function OperationalSummary(", 1)[1]
    return body.split("\nfunction ", 1)[0]


def _v5_verdict_helper(source: str) -> str:
    """The body of the `v5Verdict` helper, from its signature to its closing brace.

    R9 moved the version guard out of `OperationalSummary` and into this one function, because
    the condition is asked twice — once by the summary and once by the page body, which drops
    the legacy classification card when the summary has already said what the card would repeat.
    Two copies of that condition could disagree. The guard is therefore asserted where it now
    lives, and the summary is asserted to delegate to it rather than to restate it.
    """
    body = source.split("function v5Verdict(", 1)[1]
    return body.split("\n}", 1)[0]


@requires_web
def test_the_report_says_each_v5_verdict_in_the_words_r9_t5_locked():
    """All three sentences of all three verdicts, verbatim.

    The wording is the deliverable of this task. `NO_CALIBRATED_MANIPULATION_SIGNAL` is the one
    that carries the weight: it is the verdict a reader most wants to hear as "the media is
    fine", and its clarification is the sentence that refuses to let it be read that way.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")

    assert f'export const RULES_VERSION_V5 = "{risk_engine.RULES_VERSION_V5}";' in source

    for verdict, (title, meaning, clarification) in LOCKED_WORDING.items():
        block = source.split(f"  {verdict}: {{", 1)[1].split("\n  },", 1)[0]

        assert f'title: "{title}"' in block, verdict
        for sentence in (meaning, clarification):
            # Prettier may wrap a long string onto its own line; the sentence is what matters.
            assert f'"{sentence}"' in block, (verdict, sentence)


@requires_web
def test_the_reports_verdict_table_covers_exactly_the_v5_vocabulary():
    """No verdict left without wording, and no wording for a verdict the engine cannot reach.

    A missing entry is a v5 decision rendered with no operational summary at all. An extra one is
    a sentence waiting to be shown for a verdict nobody took — and the extras that would be
    reached for here are precisely the reassuring ones.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    listed = set(re.findall(r'^  "([A-Z_]+)",$', source.split(
        "export const V5_VERDICTS = [", 1
    )[1].split("] as const;", 1)[0], flags=re.MULTILINE))

    assert listed == {
        risk_engine.VERDICT_MANIPULATION_DETECTED,
        risk_engine.VERDICT_NO_SIGNAL,
        risk_engine.VERDICT_INCONCLUSIVE,
    }

    table = source.split("V5_VERDICT_WORDING: Record<V5Verdict, VerdictWording> = {", 1)[1]
    table = table.split("\n};", 1)[0]
    assert set(re.findall(r"^  ([A-Z_]+): \{", table, flags=re.MULTILINE)) == listed


@requires_web
def test_the_operational_summary_reads_coverage_and_never_computes_it():
    """The `complete`/`partial` word is printed from the API, not worked out in the browser.

    This is the R9-T4 property applied to the one place a reader actually sees coverage. A
    component that derived completeness from the fraction would hold a second copy of the
    coverage model, and the day that model moves the report and the record disagree — with the
    report looking authoritative while being wrong.
    """
    body = _operational_summary(WEB_REPORT.read_text(encoding="utf-8"))

    assert "coverage.usable" in body
    assert "coverage.total" in body
    assert "coverage.status" in body

    # The comparison itself, in the spellings it could plausibly be written in.
    for computed in (
        "usable ===",
        "usable ==",
        "usable !==",
        "usable <",
        "usable >",
        "? \"complete\"",
        "? \"partial\"",
    ):
        assert computed not in body, computed


@requires_web
def test_the_operational_summary_compares_no_score_against_any_threshold():
    """No figure and no operating point reaches this block, so no comparison can be made in it.

    The summary answers what was detected, not how narrowly. A score printed here would invite
    exactly the arithmetic the whole decision model exists to have already done.
    """
    body = _operational_summary(WEB_REPORT.read_text(encoding="utf-8"))

    for figure in ("score", "threshold", "T_HIGH", "toFixed", "Number("):
        assert figure not in body, figure


@requires_web
def test_the_operational_summary_is_withheld_from_every_pre_v5_report():
    """A v1-v4 decision renders as it always did: no v5 sentence, no coverage line.

    Those versions answered in `HIGH`/`MEDIUM`/`UNKNOWN` and took their decisions without a
    denominator. v5 wording over a stored `MEDIUM` would describe a decision nobody took, and the
    guard that prevents it is an equality against v5's own version string — not a "not legacy"
    test, which a sixth ruleset would silently pass.
    """
    source = WEB_REPORT.read_text(encoding="utf-8")
    body = _operational_summary(source)
    guard = _v5_verdict_helper(source)

    # The summary does not carry the condition itself. It asks the single helper that holds it,
    # so the page cannot grow a second copy of the guard that could disagree with the first.
    assert "v5Verdict(" in body
    assert "return null;" in body

    # And the guard is an equality against v5's own version string — not a "not legacy" test,
    # which a sixth ruleset would silently pass.
    assert "trace.rules_version !== RULES_VERSION_V5" in guard
    assert "return null;" in guard
    # And an unknown verdict under v5 is withheld too, rather than shown under a borrowed entry.
    assert "isV5Verdict(trace.risk_level)" in guard

    # Neither the guard nor the sentences it gates may name a legacy ruleset or a legacy level.
    for legacy in ("RULES_VERSION_V4", "RULES_VERSION_V3", "RULES_VERSION_V2", "HIGH", "MEDIUM"):
        assert legacy not in body, legacy
        assert legacy not in guard, legacy


@requires_web
def test_the_operational_wording_never_reassures():
    """The words this summary may not contain, checked over the locked table itself.

    "No calibrated manipulation signal detected" is the product's whole discipline in one line:
    the system reports what its detectors did, and never upgrades their silence into a finding
    that the media is genuine.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    table = source.split("V5_VERDICT_WORDING: Record<V5Verdict, VerdictWording> = {", 1)[1]
    table = table.split("\n};", 1)[0].lower()

    # Each of these appears in the table only inside a sentence that denies it.
    assert table.count("authentic") == 2
    assert "does not prove that the media is authentic" in table
    assert "neither evidence of manipulation nor evidence of authenticity" in table

    for reassurance in ("is real", "is genuine", " fake", "verified authentic", "clean"):
        assert reassurance not in table, reassurance


@requires_web
def test_the_report_never_reads_one_vocabulary_through_the_others_table():
    """`riskLabel` picks its table by ruleset version, so neither vocabulary borrows the other.

    A stored `MEDIUM` read through the v5 table, or a `MANIPULATION_DETECTED` read through
    `RISK_LABELS`, would put a name on a decision nobody took. Both fall through to `Unsupported`
    instead, which is the only thing actually known about a level a version cannot express.
    """
    resolver = WEB_ANALYSIS.read_text(encoding="utf-8").split(
        "export function classificationLabel(", 1
    )[1].split("\n}", 1)[0]

    assert "rulesVersion === RULES_VERSION_V5" in resolver
    assert "isV5Verdict(level) ? V5_VERDICT_WORDING[level].title : UNSUPPORTED" in resolver
    assert "isSupportedRiskLevel(level) ? RISK_LABELS[level] : UNSUPPORTED" in resolver

    # The report states its own "no decision" sentence and then delegates the vocabulary.
    source = WEB_REPORT.read_text(encoding="utf-8")
    label = source.split("function riskLabel(", 1)[1].split("\n}", 1)[0]

    assert "return classificationLabel(level, rulesVersion);" in label
    assert "RISK_LABELS" not in label
    assert "V5_VERDICT_WORDING" not in label

    # Both call sites pass the version the decision was taken under — the analysis row's on the
    # classification card, the trace's on the breakdown — rather than defaulting to either table.
    assert source.count("riskLabel(level, rulesVersion)") == 1
    assert source.count("riskLabel(trace.risk_level, trace.rules_version)") == 1
    assert "riskLabel(level)" not in source


@requires_web
def test_the_v5_report_does_not_explain_its_verdict_twice():
    """The legacy notes under the classification card are withheld from a v5 decision.

    Those three sentences are written about `UNKNOWN`, about a null level and about a level
    outside the allowlist. None is true of a v5 verdict, and printing one directly beneath the
    operational summary would have the report contradict itself on the same screen.
    """
    section = WEB_REPORT.read_text(encoding="utf-8").split(
        "function RiskSection(", 1
    )[1].split("\nfunction ", 1)[0]

    assert "const legacyVocabulary = rulesVersion !== RULES_VERSION_V5;" in section
    assert "{!legacyVocabulary ? null : level === null ? (" in section

    # The record fields are not withheld: rule, ruleset and calibration are shown for every
    # decision, v5 included, because they are what the row actually holds.
    assert 'label="Rule fired"' in section
    assert 'label="Ruleset version"' in section
    assert 'label="Calibration ID"' in section


WEB_ADMIN_ANALYSIS = (
    WEB_ROOT / "app" / "admin" / "analyses" / "[id]" / "page.tsx"
)


@requires_web
def test_the_admin_card_titles_a_v5_verdict_through_the_same_resolver_as_the_report():
    """Neither screen owns a copy of the vocabulary, so the two cannot disagree about a row.

    This is the R9-T5 fix. The card knew `HIGH`, `MEDIUM` and `UNKNOWN` and nothing else, so
    every v5 verdict reached it as `Unsupported` — while the report for the same analysis said
    `Manipulation detected`. An operator comparing the two screens had no way to tell which was
    wrong, which is a worse failure than either screen being wrong on its own.
    """
    source = WEB_ADMIN_ANALYSIS.read_text(encoding="utf-8")

    assert "classificationLabel(level, analysis.risk_rules_version)" in source

    # The private lookup this card used to make is gone, not merely bypassed.
    assert "RISK_LABELS[level]" not in source
    assert "isSupportedRiskLevel" not in source
    # And the card still says its own sentence for a row that was never decided.
    assert "level === null" in source
    assert "NO_DECISION" in source


@requires_web
def test_the_admin_card_prints_coverage_without_computing_it():
    """The same three API fields the report prints, and the same refusal to derive the word."""
    source = WEB_ADMIN_ANALYSIS.read_text(encoding="utf-8")

    assert "analysis.risk_trace?.decision_coverage ?? null" in source
    assert "${coverage.usable}/${coverage.total} ${coverage.status}" in source
    # Omitted entirely on a decision that states no coverage — never rendered as `0/0`.
    assert "{coverage !== null && (" in source

    for computed in ("usable ===", "usable ==", "? \"complete\"", "? \"partial\""):
        assert computed not in source, computed


@requires_web
def test_the_admin_card_compares_no_score_against_any_threshold():
    """The card gained a trace field and no arithmetic with it."""
    card = WEB_ADMIN_ANALYSIS.read_text(encoding="utf-8").split(
        "function ForensicResult(", 1
    )[1].split("\nfunction ", 1)[0]

    for figure in ("score", "threshold", "toFixed", "Number("):
        assert figure not in card, figure


@requires_web
def test_the_admin_card_keeps_the_human_review_on_the_other_side_of_the_page():
    """The forensic half and the review half stay separate cards, as R8-T7 drew them.

    The fix added a fact to the evidence card. It must not have moved the boundary this page is
    built around: what a detector committed and what a colleague said are different kinds of
    statement, and an operator has to be able to tell them apart at a glance.
    """
    source = WEB_ADMIN_ANALYSIS.read_text(encoding="utf-8")

    assert source.count("function ForensicResult(") == 1
    assert source.count("function HumanReview(") == 1
    assert source.count("<ForensicResult analysis={analysisResult.analysis} />") == 1
    assert source.count("<HumanReview review={reviewResult.review} />") == 1

    # The review half reads the review and never the forensic decision.
    review = source.split("function HumanReview(", 1)[1].split("\nfunction ", 1)[0]
    for forensic in ("risk_level", "risk_trace", "classificationLabel", "coverage"):
        assert forensic not in review, forensic


@requires_web
def test_the_printed_report_uses_the_same_summary_component():
    """One component, rendered once. There is no second decision path for the PDF.

    The report prints by being printed — the browser renders this markup to paper — so the only
    way the document could say something different from the screen is a second component or a
    `print:hidden` on this one. Neither exists, and this is what says so.
    """
    source = WEB_REPORT.read_text(encoding="utf-8")

    assert source.count("function OperationalSummary(") == 1
    assert source.count("<OperationalSummary trace={analysis.risk_trace} />") == 1

    body = _operational_summary(source)
    assert "print:hidden" not in body
    assert "hidden print:" not in body


@requires_web
def test_a_v4_report_never_falls_back_to_the_p7_wording():
    """The two places the report describes the ruleset's scope in prose.

    Both were written as a chain ending in v1's sentence — "only the synthetic-video detector
    contributes" — which is what an unrecognised version would have rendered. It is true of a
    `p7-v1.0.0` decision and false of a v4 one, and a fallback that is false while reading as
    deliberate is worse than no description at all.

    The count is the assertion: each chain must test `RULES_VERSION_V4` before it can reach its
    final branch.
    """
    source = WEB_REPORT.read_text(encoding="utf-8")

    # The import, the scope-of-the-model chain, and three `analysis.risk_rules_version` chains:
    # the face-manipulation panel's `decides`, the mouth-dynamics panel's `evidenceOnly`, and
    # the independent-evidence introduction. R9 added the second of those three — v4 is one of
    # the two rulesets under which the mouth-dynamics model is calibrated and still may not
    # decide — which is why both totals below moved by one and the `ruleset ===` count did not.
    assert source.count("RULES_VERSION_V4") == 5
    assert source.count("ruleset === RULES_VERSION_V4") == 1
    assert source.count("analysis.risk_rules_version === RULES_VERSION_V4") == 3

    # The sentence a v4 decision must never reach. It is still there, and still the last branch
    # for the version it is true of.
    assert "Only the synthetic-video detector contributes" in source


@requires_web
def test_the_report_never_says_the_mouth_dynamics_model_reached_this_level_under_v4():
    """The one claim R7-T6 exists to stop the report from making.

    v4's table may describe this detector as evidence and must never mark it `decided`. The
    check is on the role, which is the field the panel renders as a contribution: `decided` is
    the vocabulary's word for "this is what produced the level", and v4 has no rule that can.
    `V3_RATIONALES` keeps its own `decided` entry for `R103`, untouched, and this does not look
    at it.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    v4_block = source.split("const V4_RATIONALES", 1)[1].split("\n};", 1)[0]

    mouth_entries = re.findall(
        r"mouthDynamics: \{\s*role: \"(\w+)\"", v4_block
    ) + re.findall(r"mouthDynamics: \{\s*\n\s*role: \"(\w+)\"", v4_block)

    assert mouth_entries
    assert "decided" not in mouth_entries

    # `evidence` on every rule that could have taken a level, and it is its own role for a
    # reason. `unread` renders as "Not read by this ruleset", which is false — v4 reads this
    # detector, and its readability is what separates an `R200` from an `R201`. `below` and
    # `quiet` would be false too, on exactly the reports that matter: the R7-T5 regression is a
    # score *above* the threshold measured for it.
    assert set(mouth_entries) == {"evidence", "unreadable", "unavailable"}
    assert "unread" not in mouth_entries

    # v3 is untouched: its `R103` still says this detector decided, because it did.
    v3_block = source.split("const V3_RATIONALES", 1)[1].split("\n};", 1)[0]
    assert 'role: "decided"' in v3_block.split("R103:", 1)[1]


# --- R9-T4: the decision trace API under `r9-v5.0.0` -----------------------------------------
#
# One property runs through every test below and is worth stating once: the trace is
# *explanatory*. It reads persisted columns and persisted evidence and says what the ruleset
# that took the decision made of them. It does not re-decide, and a coverage figure that sits
# oddly beside a verdict is a fact about the record rather than a correction to be applied.


def v5(level, rule, **signals):
    """A v5 trace over the persisted evidence given, under v5's own calibration identity."""
    return trace(level, rule, V5_VERSION, V5_CALIBRATION, **signals)


def coverage_of(result):
    return (result.decision_coverage.usable, result.decision_coverage.total)


def test_both_deciding_detectors_readable_is_two_of_two():
    """`D_usable == D_total`, and it is a count of readings rather than of findings.

    Neither detector reached its threshold here and the coverage is complete anyway: a reading
    below an operating point is a reading. The verdict beside it is the persisted one and is
    what separates this from the case above — `2/2` sits under a `MANIPULATION_DETECTED` just
    as readily.
    """
    result = v5(NO_SIGNAL, "R9-200", svd=svd(score=0.5), face=face(score=0.5))

    assert coverage_of(result) == (2, 2)
    assert {c.signal for c in result.decision_eligible_detectors} == {
        "synthetic_video",
        "face_manipulation",
    }
    assert all(
        c.condition == risk_trace.CONDITION_THRESHOLD_NOT_REACHED
        for c in result.decision_eligible_detectors
    )


@pytest.mark.parametrize(
    "face_row",
    [
        None,
        face(status="FAILED"),
        face(status="TIMEOUT"),
        # An abstention: recorded as a `FAILED` row with no score, because no sampled frame held
        # a face. A statement about the media and not a finding about it, and it completes no
        # coverage.
        face(status="FAILED", score=None),
    ],
    ids=["no_row", "failed", "timeout", "abstained"],
)
def test_a_deciding_detector_that_produced_no_reading_is_one_of_two(face_row):
    """Coverage the detector never obtained is missing coverage, not a zero-valued reading."""
    signals = {"synthetic_video": svd(score=0.5)}
    if face_row is not None:
        signals["face_manipulation"] = face_row

    result = build_trace(
        risk_level=INCONCLUSIVE,
        rule_id="R9-300",
        rules_version=V5_VERSION,
        calibration_id=V5_CALIBRATION,
        signals=signals,
    )

    assert coverage_of(result) == (1, 2)
    contribution = by_signal(result)["face_manipulation"]
    assert contribution.condition == risk_trace.CONDITION_UNAVAILABLE
    assert contribution.condition != risk_trace.CONDITION_THRESHOLD_NOT_REACHED
    assert contribution.score is None


@pytest.mark.parametrize(
    "svd_row",
    [
        # The whole point of the case: the provider said `SUCCESS` in every one of these.
        svd(provider_version="847b6e53-0133-452d-ab85-d7acf3ace723-preview"),
        svd(score=None),
        svd(score=1.4),
        svd(score=float("nan")),
        svd(total_clips=0),
        svd(total_clips="seven"),
    ],
    ids=[
        "uncalibrated_deployment",
        "no_score",
        "score_out_of_range",
        "score_not_a_number",
        "zero_units",
        "unreadable_units",
    ],
)
def test_success_is_not_enough_to_count_as_coverage(svd_row):
    """`SUCCESS != usable_reading`, asserted on the status that most looks like one.

    This is the case a loose coverage model gets wrong. Counting `decisional=True` rows whose
    status is `SUCCESS` would report `2/2` for every row here — including a score from a
    deployment no operating point was ever measured on, which is the reading R6-T1 exists to
    keep out of a verdict.
    """
    result = build_trace(
        risk_level=INCONCLUSIVE,
        rule_id="R9-300",
        rules_version=V5_VERSION,
        calibration_id=V5_CALIBRATION,
        signals={"synthetic_video": svd_row, "face_manipulation": face(score=0.5)},
    )

    assert svd_row.status == "SUCCESS"
    assert coverage_of(result) == (1, 2)
    assert by_signal(result)["synthetic_video"].condition == (
        risk_trace.CONDITION_UNAVAILABLE
    )


def test_neither_deciding_detector_readable_is_zero_of_two():
    """The denominator survives an analysis in which nothing was read."""
    result = v5(INCONCLUSIVE, "R9-301", svd=svd(status="FAILED"), lip=lip(score=0.9))

    assert coverage_of(result) == (0, 2)
    assert result.decision_coverage.total == 2


@pytest.mark.parametrize(
    "lip_row",
    [
        None,
        lip(score=0.9),
        lip(score=0.01),
        lip(status="FAILED"),
        lip(status="TIMEOUT"),
        lip(score=None),
        lip(windows_scored=0),
        lip(provider_version="https://github.com/ahaliassos/LipForensics@deadbeef"),
    ],
    ids=[
        "no_row",
        "above_its_threshold",
        "below_its_threshold",
        "failed",
        "timeout",
        "no_score",
        "zero_windows",
        "uncalibrated_deployment",
    ],
)
def test_the_evidence_only_detector_moves_neither_coverage_nor_explanation(lip_row):
    """R9-T1 invariant 1, asserted across every state this detector can be in.

    It is outside the coverage arithmetic rather than a zero inside it: its absence removes no
    coverage, its failure removes no coverage, and a score above the operating point R5-T3
    measured for it adds none and decides nothing. Everything that explains the verdict —
    coverage, the rule's sentence, and how the two deciding detectors stood — is identical in
    all eight.
    """
    signals = {"synthetic_video": svd(score=0.99), "face_manipulation": face(score=0.5)}
    if lip_row is not None:
        signals["lip_forensics"] = lip_row

    result = build_trace(
        risk_level=MANIPULATION_DETECTED,
        rule_id="R9-101",
        rules_version=V5_VERSION,
        calibration_id=V5_CALIBRATION,
        signals=signals,
    )
    baseline = v5(
        MANIPULATION_DETECTED, "R9-101", svd=svd(score=0.99), face=face(score=0.5)
    )

    assert coverage_of(result) == coverage_of(baseline) == (2, 2)
    assert result.risk_level == baseline.risk_level
    assert result.rule_summary == baseline.rule_summary
    assert result.decision_eligible_detectors == baseline.decision_eligible_detectors

    # It is reported — banded against its own measured threshold, as v4 already bands it — and
    # reported strictly as supplementary. It is never in the decision-eligible list, whatever
    # it scored.
    eligible = {c.signal for c in result.decision_eligible_detectors}
    assert [c.signal for c in result.supplementary_evidence] == ["lip_forensics"]
    assert "lip_forensics" not in eligible
    assert all(
        c.role == risk_trace.ROLE_CONSIDERED for c in result.supplementary_evidence
    )


def test_the_two_lists_partition_the_contributions_under_every_ruleset():
    """Every contribution in exactly one list, and the union unchanged — v5 and legacy alike."""
    for version, calibration in (
        (V1_VERSION, V1_CALIBRATION),
        (V2_VERSION, V2_CALIBRATION),
        (V3_VERSION, V3_CALIBRATION),
        (V4_VERSION, V4_CALIBRATION),
        (V5_VERSION, V5_CALIBRATION),
    ):
        result = trace(
            "MEDIUM" if version != V5_VERSION else INCONCLUSIVE,
            None,
            version,
            calibration,
            svd=svd(score=0.5),
            face=face(score=0.5),
            lip=lip(score=0.5),
        )

        eligible = result.decision_eligible_detectors
        supplementary = result.supplementary_evidence

        assert set(eligible) | set(supplementary) == set(result.contributions), version
        assert len(eligible) + len(supplementary) == len(result.contributions), version
        assert not set(eligible) & set(supplementary), version

    # The split is the persisted version's own: the same detector, decision-eligible under v3
    # and supplementary under v4 and v5, with the same stored score on every side of that line.
    for version, calibration in (
        (V4_VERSION, V4_CALIBRATION),
        (V5_VERSION, V5_CALIBRATION),
    ):
        result = trace("MEDIUM", None, version, calibration, lip=lip(score=0.9))
        assert "lip_forensics" in {c.signal for c in result.supplementary_evidence}

    v3_result = trace("MEDIUM", "R200", V3_VERSION, V3_CALIBRATION, lip=lip(score=0.9))
    assert "lip_forensics" in {c.signal for c in v3_result.decision_eligible_detectors}


def test_the_denominator_is_the_frozen_expectation_and_not_the_rows_present():
    """A detector that was never invoked stays in the denominator.

    The defect this rules out is the one R9-T1 section 2.4 names as the most dangerous available
    to a coverage model: if `D_total` came from the evidence present, an analysis on which a
    detector never ran would report complete coverage — the detector would *buy* coverage by
    being absent.
    """
    nothing_ran = build_trace(
        risk_level=INCONCLUSIVE,
        rule_id="R9-301",
        rules_version=V5_VERSION,
        calibration_id=V5_CALIBRATION,
        signals={},
    )

    assert coverage_of(nothing_ran) == (0, 2)
    assert risk_trace.RULESET_V5.decision_total == 2


def test_the_frozen_denominator_must_match_the_detectors_it_counts():
    """A version whose literal and table disagree fails loudly at construction.

    The literal is the contract and is never `len()` of anything. What must not happen quietly
    is the two drifting apart — a later edit that demotes a deciding detector without moving the
    number would leave every coverage figure overstating itself by one.
    """
    with pytest.raises(ValueError, match="out of step"):
        risk_trace.Ruleset(
            rules_version="r9-v9.9.9",
            calibration_id=V5_CALIBRATION,
            signals=risk_trace.RULESET_V5.signals,
            rules={},
            decision_total=3,
        )


def test_a_hit_beside_incomplete_coverage_is_reported_as_both():
    """The trace explains the verdict; it does not second-guess it.

    `MANIPULATION_DETECTED` on 1/2 coverage is the engine's own rule showing through — a hit is
    never softened by the other detector failing, because R4-T1 measured that a quiet detector
    carries no information about the family the flagging one is calibrated for. The trace's job
    is to report the hit *and* the missing coverage, and to change neither.
    """
    result = v5(
        MANIPULATION_DETECTED, "R9-101", svd=svd(score=0.99), face=face(status="FAILED")
    )

    assert result.risk_level == MANIPULATION_DETECTED
    assert coverage_of(result) == (1, 2)
    assert by_signal(result)["synthetic_video"].role == risk_trace.ROLE_DECISIVE
    assert "whatever the other decision-eligible detector did" in result.rule_summary


def test_the_detector_that_produced_a_v5_verdict_is_reported_as_decisive():
    """`MANIPULATION_DETECTED` is v5's word for the position `HIGH` names in the legacy table.

    A trace that compared the persisted verdict against the literal `HIGH` would report the very
    detector that produced it as merely `considered`, which is a false attribution in the
    opposite direction from the one `decisional` guards.
    """
    result = v5(
        MANIPULATION_DETECTED, "R9-100", svd=svd(score=0.99), face=face(score=0.99)
    )

    roles = {c.signal: c.role for c in result.decision_eligible_detectors}
    assert roles == {
        "synthetic_video": risk_trace.ROLE_DECISIVE,
        "face_manipulation": risk_trace.ROLE_DECISIVE,
    }

    # And not under a verdict no threshold produced: the same evidence read back under a
    # persisted `INCONCLUSIVE` marks nothing decisive.
    inconclusive = v5(INCONCLUSIVE, "R9-300", svd=svd(score=0.99), face=face(score=0.99))
    assert all(c.role == risk_trace.ROLE_CONSIDERED for c in inconclusive.contributions)


def test_the_v5_trace_never_recalculates_the_persisted_verdict(monkeypatch):
    """The persisted columns come back untouched, whatever the evidence beside them says.

    The evidence here reaches both operating points and the stored verdict is `INCONCLUSIVE` —
    a combination the engine would not produce. The trace reports it exactly as stored, because
    an inconsistency in the record is something a reader must be able to see; a trace that
    quietly corrected it would be the one place the record could be rewritten unnoticed.
    """
    for name in ("evaluate", "evaluate_v5"):
        monkeypatch.setattr(
            risk_engine,
            name,
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("the trace re-ran the risk engine")
            ),
        )

    result = v5(INCONCLUSIVE, "R9-301", svd=svd(score=0.99), face=face(score=0.99))

    assert result.risk_level == INCONCLUSIVE
    assert result.rule_id == "R9-301"
    assert result.rules_version == V5_VERSION
    assert result.calibration_id == V5_CALIBRATION
    assert "None of the expected decision coverage" in result.rule_summary
    # The coverage is read off the evidence and is free to disagree with the rule beside it.
    assert coverage_of(result) == (2, 2)


def test_moving_todays_v5_constants_does_not_reinterpret_a_stored_v5_decision(
    monkeypatch,
):
    """The engine's v5 constants are moved under the trace's feet; nothing shifts."""
    before = v5(NO_SIGNAL, "R9-200", svd=svd(score=0.5), face=face(score=0.5))

    monkeypatch.setattr(risk_engine, "SVD_T_HIGH", 0.1)
    monkeypatch.setattr(risk_engine, "FACE_T_HIGH", 0.1)
    monkeypatch.setattr(risk_engine, "D_TOTAL_V5", 99)
    monkeypatch.setattr(risk_engine, "RULES_VERSION_V5", "r9-v9.0.0")

    after = v5(NO_SIGNAL, "R9-200", svd=svd(score=0.5), face=face(score=0.5))

    assert after == before
    assert coverage_of(after) == (2, 2)
    assert by_signal(after)["synthetic_video"].threshold == SVD_T_HIGH


# Legacy integrity: v1-v4 still resolve, and v5's coverage model is not retrofitted onto them.


@pytest.mark.parametrize(
    ("version", "calibration", "level", "rule"),
    [
        (V1_VERSION, V1_CALIBRATION, "MEDIUM", "R200"),
        (V2_VERSION, V2_CALIBRATION, "MEDIUM", "R200"),
        (V3_VERSION, V3_CALIBRATION, "HIGH", "R103"),
        (V4_VERSION, V4_CALIBRATION, "MEDIUM", "R200"),
    ],
)
def test_a_legacy_decision_still_resolves_and_states_no_coverage(
    version, calibration, level, rule
):
    """Every stored version still reads, and none of them acquires a coverage claim.

    v4's `R200` is "all three detectors were readable and neither deciding one reached its
    threshold" — a sentence about three detectors, taken under rules that counted no
    denominator. Explaining it as "2/2" would be a coverage claim v4 never made, and R9-T1
    invariant 4 forbids retrofitting the R9 model onto a decision taken without it.
    """
    result = trace(
        level,
        rule,
        version,
        calibration,
        svd=svd(score=0.5),
        face=face(score=0.5),
        lip=lip(score=0.9),
    )

    assert result.interpreted is True
    assert result.rule_summary is not None
    assert result.decision_coverage is None
    assert result.decision_eligible_detectors


def test_the_legacy_trace_is_byte_for_byte_what_it_was_before_r9():
    """The fields that existed before R9-T4 are unchanged on a v4 trace.

    Asserted against literals rather than against a regenerated expectation, so a change in how
    the trace is assembled cannot move a stored decision's explanation and agree with itself.
    """
    result = trace(
        "HIGH",
        "R100",
        V4_VERSION,
        V4_CALIBRATION,
        svd=svd(score=0.99),
        lip=lip(score=0.9),
    )

    contributions = by_signal(result)

    assert result.risk_level == "HIGH"
    assert result.interpreted is True
    assert len(result.contributions) == 3
    assert contributions["synthetic_video"].threshold == SVD_T_HIGH
    assert contributions["synthetic_video"].condition == (
        risk_trace.CONDITION_THRESHOLD_REACHED
    )
    assert contributions["synthetic_video"].role == risk_trace.ROLE_DECISIVE
    # A mouth-dynamics crossing under v4 is still `considered` and still not decisive.
    assert contributions["lip_forensics"].condition == (
        risk_trace.CONDITION_THRESHOLD_REACHED
    )
    assert contributions["lip_forensics"].role == risk_trace.ROLE_CONSIDERED
    assert contributions["face_manipulation"].condition == (
        risk_trace.CONDITION_UNAVAILABLE
    )


def test_an_uninterpretable_v5_trace_reports_no_coverage_rather_than_zero():
    """A calibration identity v5 was not measured under withholds the fraction entirely.

    Nothing was read against an operating point, so every detector is `not_interpreted` and a
    count of usable readings would come out `0` — indistinguishable from the genuinely uncovered
    analysis `R9-301` describes, and a far stronger statement than "this could not be read".
    """
    result = build_trace(
        risk_level=NO_SIGNAL,
        rule_id="R9-200",
        rules_version=V5_VERSION,
        calibration_id="0" * 64,
        signals={
            "synthetic_video": svd(score=0.5),
            "face_manipulation": face(score=0.5),
        },
    )

    assert result.interpreted is False
    assert result.decision_coverage is None
    assert all(
        c.condition == risk_trace.CONDITION_NOT_INTERPRETED
        for c in result.decision_eligible_detectors
    )


def test_an_unknown_ruleset_version_carries_none_of_the_r9_fields():
    """A version this build cannot explain gets no coverage and no detector roles invented."""
    result = build_trace(
        risk_level="SOMETHING_ELSE",
        rule_id="R9-999",
        rules_version="r99-v9.0.0",
        calibration_id=V5_CALIBRATION,
        signals={"synthetic_video": svd(score=0.99)},
    )

    assert result.interpreted is False
    assert result.decision_coverage is None
    assert result.decision_eligible_detectors == ()
    assert result.supplementary_evidence == ()


def test_the_v5_rule_ids_are_disjoint_from_every_legacy_table():
    """No v5 sentence can rewrite what a stored legacy row said, and none of them reuses an id."""
    v5_ids = set(risk_trace.RULESET_V5.rules)

    assert v5_ids == {"R9-100", "R9-101", "R9-102", "R9-200", "R9-300", "R9-301"}

    for version, ruleset in risk_trace.RULESETS.items():
        if version == V5_VERSION:
            continue
        assert not v5_ids & set(ruleset.rules), version


def test_the_v5_entry_matches_the_engine_that_writes_it():
    """The drift guard for v5, on the same terms as the one for the current ruleset.

    `RULESET_V5` is a transcription, and a transcription that has fallen behind is worse than an
    absent one: it would explain every v5 decision confidently and wrongly.
    """
    entry = risk_trace.RULESETS[risk_engine.RULES_VERSION_V5]

    assert entry.calibration_id == risk_engine.CALIBRATION_ID
    assert entry.decision_total == risk_engine.D_TOTAL_V5
    assert entry.decisive_level == risk_engine.VERDICT_MANIPULATION_DETECTED

    thresholds = {s.signal_type: s.threshold for s in entry.signals}
    assert thresholds == {
        risk_engine.SVD_SIGNAL_TYPE: risk_engine.SVD_T_HIGH,
        risk_engine.FACE_SIGNAL_TYPE: risk_engine.FACE_T_HIGH,
        risk_engine.LIP_SIGNAL_TYPE: risk_engine.LIP_T_HIGH,
    }

    decisional = {s.signal_type for s in entry.signals if s.decisional}
    assert decisional == {risk_engine.SVD_SIGNAL_TYPE, risk_engine.FACE_SIGNAL_TYPE}

    assert set(entry.rules) == {
        risk_engine.RULE_V5_HIGH_MULTIPLE,
        risk_engine.RULE_V5_HIGH_SVD,
        risk_engine.RULE_V5_HIGH_FACE,
        risk_engine.RULE_V5_NO_SIGNAL,
        risk_engine.RULE_V5_INCONCLUSIVE_PARTIAL,
        risk_engine.RULE_V5_INCONCLUSIVE_ALL,
    }


def test_the_v5_coverage_the_trace_reports_is_the_coverage_the_engine_counted():
    """The two modules agree on `D_usable` across every combination of the two deciding rows.

    The one comparison of the two implementations, and it has to exist: the trace computes
    usability from persisted columns without importing the engine, so nothing structural stops
    the two definitions from drifting. What is asserted is the count, over the same evidence,
    every way round.
    """
    rows = {
        "usable_low": (svd(score=0.5), face(score=0.5)),
        "usable_high": (svd(score=0.99), face(score=0.99)),
        "failed": (svd(status="FAILED"), face(status="FAILED")),
        "uncalibrated": (
            svd(provider_version="not-the-deployment"),
            face(provider_version="not-the-deployment"),
        ),
        "unreadable": (svd(score=None), face(score=None)),
        "zero_units": (svd(total_clips=0), face(frames_scored=0)),
        "missing": (None, None),
    }

    for svd_key, (svd_row, _) in rows.items():
        for face_key, (_, face_row) in rows.items():
            engine_usable = sum(
                (
                    risk_engine.is_eligible_svd(_svd_evidence(svd_row))
                    and risk_engine.is_usable_svd(_svd_evidence(svd_row)),
                    risk_engine.is_eligible_face(_face_evidence(face_row))
                    and risk_engine.is_usable_face(_face_evidence(face_row)),
                )
            )

            signals = {s.signal_type: s for s in (svd_row, face_row) if s is not None}
            result = build_trace(
                risk_level=INCONCLUSIVE,
                rule_id="R9-300",
                rules_version=V5_VERSION,
                calibration_id=V5_CALIBRATION,
                signals=signals,
            )

            assert result.decision_coverage.usable == engine_usable, (
                svd_key,
                face_key,
            )


def _svd_evidence(persisted):
    """The same persisted row, in the shape the engine's usability predicates take."""
    if persisted is None:
        return None
    metadata = persisted.metadata if isinstance(persisted.metadata, dict) else {}
    return risk_engine.SvdEvidence(
        provider=persisted.provider,
        signal_type=persisted.signal_type,
        status=persisted.status,
        provider_version=persisted.provider_version,
        score=persisted.score,
        total_clips=metadata.get("total_clips"),
    )


def _face_evidence(persisted):
    if persisted is None:
        return None
    metadata = persisted.metadata if isinstance(persisted.metadata, dict) else {}
    return risk_engine.FaceEvidence(
        provider=persisted.provider,
        signal_type=persisted.signal_type,
        status=persisted.status,
        provider_version=persisted.provider_version,
        score=persisted.score,
        frames_scored=metadata.get("frames_scored"),
    )


# The API contract, as `/api/v1/analyses` serves it.


def test_the_api_trace_carries_the_r9_fields():
    """The three fields R9-T4 adds survive the pydantic boundary with their meanings intact."""
    rendered = RiskTraceResponse.model_validate(
        v5(NO_SIGNAL, "R9-200", svd=svd(score=0.5), face=face(score=0.5)),
        from_attributes=True,
    )

    assert rendered.risk_level == NO_SIGNAL
    assert rendered.decision_coverage.usable == 2
    assert rendered.decision_coverage.total == 2
    assert [c.signal for c in rendered.decision_eligible_detectors] == [
        "synthetic_video",
        "face_manipulation",
    ]
    # The mouth-dynamics detector is listed by v5 and is always supplementary — here with no row
    # at all, which is `unavailable` and still not part of any coverage count.
    assert [c.signal for c in rendered.supplementary_evidence] == ["lip_forensics"]
    assert rendered.supplementary_evidence[0].condition == "unavailable"


def test_the_api_trace_states_coverage_completeness_so_no_consumer_computes_it():
    """`is_complete` and `status` ride with the fraction, for both of its outcomes.

    R9-T5 renders `Decision coverage: X/Y complete` in a browser and again in a PDF. Neither
    may reach that last word by comparing `usable` against `total`: that comparison is the
    coverage model, and a copy of it downstream is a copy that can disagree with the record.
    So the API states the answer and the wording of the answer, and the report prints them.
    """
    complete = RiskTraceResponse.model_validate(
        v5(NO_SIGNAL, "R9-200", svd=svd(score=0.5), face=face(score=0.5)),
        from_attributes=True,
    )

    assert (complete.decision_coverage.usable, complete.decision_coverage.total) == (2, 2)
    assert complete.decision_coverage.is_complete is True
    assert complete.decision_coverage.status == "complete"

    partial = RiskTraceResponse.model_validate(
        v5(INCONCLUSIVE, "R9-300", svd=svd(status="FAILED"), face=face(score=0.5)),
        from_attributes=True,
    )

    assert (partial.decision_coverage.usable, partial.decision_coverage.total) == (1, 2)
    assert partial.decision_coverage.is_complete is False
    assert partial.decision_coverage.status == "partial"


def test_coverage_completeness_says_nothing_about_what_was_found():
    """Complete coverage sits under a detection exactly as readily as under no signal.

    The word beside the fraction describes how much of the expected reading was obtained. A
    reader who took `complete` as reassurance would be reading the coverage line as a verdict,
    and the one place that could have encouraged it is this pairing.
    """
    detected = RiskTraceResponse.model_validate(
        v5(MANIPULATION_DETECTED, "R9-100", svd=svd(score=0.99), face=face(score=0.5)),
        from_attributes=True,
    )

    assert detected.risk_level == MANIPULATION_DETECTED
    assert detected.decision_coverage.status == "complete"


def test_the_api_trace_states_no_coverage_for_a_legacy_decision():
    """Serialized, a legacy trace carries `decision_coverage: null` — never `0`."""
    rendered = RiskTraceResponse.model_validate(
        trace("MEDIUM", "R200", V4_VERSION, V4_CALIBRATION, svd=svd(score=0.5)),
        from_attributes=True,
    )

    assert rendered.model_dump()["decision_coverage"] is None
    assert [c.signal for c in rendered.supplementary_evidence] == ["lip_forensics"]
