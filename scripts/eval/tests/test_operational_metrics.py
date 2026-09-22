"""R12-T4: what a stored verdict is scored as, against which Ground Truth, and how often."""

import ast
import importlib.util
from pathlib import Path

import pytest

from eval import operational_metrics as om

API_APP = Path(__file__).resolve().parents[3] / "apps" / "api" / "app"

SHA_A = "a" * 64
SHA_B = "b" * 64
CAL_1 = "1" * 64
CAL_2 = "2" * 64
V5 = "r9-v5.0.0"


def row(
    analysis_id="00000000-0000-0000-0000-000000000001",
    *,
    sha=SHA_A,
    rules=V5,
    calibration=CAL_1,
    verdict="MANIPULATION_DETECTED",
    source_class="CONTROLLED_TEST",
    label="AI_GENERATED",
    created_at="2026-09-22T10:00:00+00:00",
):
    return om.EvaluationRow(
        analysis_id=analysis_id,
        analysis_created_at=created_at,
        analysis_status="completed",
        media_sha256=sha,
        risk_rules_version=rules,
        risk_calibration_id=calibration,
        risk_level=verdict,
        risk_rule_id=None,
        gt_source_class=source_class,
        gt_label=label,
    )


def only_group(report):
    assert len(report["groups"]) == 1
    return report["groups"][0]


# --- deduplication -------------------------------------------------------------------------


def test_duplicate_analyses_of_one_unit_do_not_inflate_n():
    rows = [
        row("id-1", created_at="2026-09-20T10:00:00+00:00", verdict="INCONCLUSIVE"),
        row("id-2", created_at="2026-09-22T10:00:00+00:00", verdict="MANIPULATION_DETECTED"),
        row("id-3", created_at="2026-09-21T10:00:00+00:00", verdict="INCONCLUSIVE"),
    ]
    report = om.evaluate(rows)
    group = only_group(report)

    assert report["input"]["analysis_rows"] == 3
    assert report["input"]["evaluation_units"] == 1
    assert group["in_scope_units"] == 1
    assert group["duplicate_analyses"] == 2
    # The latest analysis represents the unit, and only its verdict is scored.
    assert group["confusion"]["TP"] == 1
    assert group["abstentions"] == 0
    [unit] = report["snapshot"]
    assert unit["analysis_id"] == "id-2"
    assert unit["duplicate_analysis_ids"] == ["id-3", "id-1"]


def test_representative_choice_is_deterministic_on_a_timestamp_tie():
    same_time = "2026-09-22T10:00:00+00:00"
    forward = [row("id-a", created_at=same_time), row("id-b", created_at=same_time)]
    units_forward = om.deduplicate(forward)
    units_reverse = om.deduplicate(list(reversed(forward)))
    assert units_forward[0].representative.analysis_id == "id-b"
    assert units_reverse[0].representative.analysis_id == "id-b"


def test_same_media_same_ruleset_different_calibration_is_two_units():
    rows = [
        row("id-1", calibration=CAL_1, verdict="MANIPULATION_DETECTED"),
        row("id-2", calibration=CAL_2, verdict="NO_CALIBRATED_MANIPULATION_SIGNAL"),
    ]
    report = om.evaluate(rows)

    assert report["input"]["evaluation_units"] == 2
    assert report["input"]["duplicate_analyses"] == 0
    # Two calibrations are two measurements, and they are never pooled.
    by_calibration = {g["risk_calibration_id"]: g for g in report["groups"]}
    assert set(by_calibration) == {CAL_1, CAL_2}
    assert by_calibration[CAL_1]["confusion"]["TP"] == 1
    assert by_calibration[CAL_2]["confusion"]["FN"] == 1
    assert all(g["duplicate_analyses"] == 0 for g in report["groups"])


def test_different_media_under_one_ruleset_and_calibration_are_different_units():
    report = om.evaluate([row("id-1", sha=SHA_A), row("id-2", sha=SHA_B)])
    assert only_group(report)["in_scope_units"] == 2


# --- Ground Truth eligibility --------------------------------------------------------------


@pytest.mark.parametrize("label", ["GENUINE", "AI_GENERATED", "FACE_SWAP", "UNKNOWN"])
def test_unknown_source_class_is_ineligible_whatever_the_label(label):
    report = om.evaluate([row(source_class="UNKNOWN", label=label)])
    group = only_group(report)
    assert group["exclusions"]["ground_truth_ineligible"] == 1
    assert group["in_scope_units"] == 0
    assert sum(group["confusion"].values()) == 0


@pytest.mark.parametrize("source_class", ["OWNER_KNOWN", "CONTROLLED_TEST", "EXTERNAL_VERIFIED"])
def test_known_source_classes_are_eligible(source_class):
    report = om.evaluate([row(source_class=source_class)])
    assert only_group(report)["confusion"]["TP"] == 1


def test_unknown_label_is_excluded_not_scored():
    report = om.evaluate([row(label="UNKNOWN")])
    group = only_group(report)
    assert group["exclusions"]["ground_truth_label_unknown"] == 1
    assert group["in_scope_units"] == 0


def test_analysis_without_ground_truth_is_counted_and_not_listed():
    report = om.evaluate([row(source_class=None, label=None)])
    assert only_group(report)["exclusions"]["no_ground_truth"] == 1
    assert report["snapshot"] == []


# --- scope ---------------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["AUDIO_MANIPULATION", "OTHER_MANIPULATION"])
@pytest.mark.parametrize(
    "verdict", ["NO_CALIBRATED_MANIPULATION_SIGNAL", "MANIPULATION_DETECTED", "INCONCLUSIVE"]
)
def test_unsupported_manipulation_family_is_never_a_false_negative(label, verdict):
    report = om.evaluate([row(label=label, verdict=verdict)])
    group = only_group(report)
    assert group["exclusions"]["unsupported_for_ruleset"] == 1
    assert group["confusion"] == {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    assert group["abstentions"] == 0


def test_ruleset_without_a_contract_is_not_scored_by_analogy():
    report = om.evaluate([row(rules="r7-v4.0.0", verdict="MEDIUM")])
    group = only_group(report)
    assert group["scope_contract"] is False
    assert group["exclusions"]["ruleset_without_scope_contract"] == 1
    assert group["in_scope_units"] == 0


def test_scope_is_read_from_the_ruleset_that_decided_the_row():
    """A later ruleset's wider coverage must not reach back into an older row."""
    wider = om.ScopeContract(
        rules_version="r99-v9.0.0",
        positive_verdicts=frozenset({"MANIPULATION_DETECTED"}),
        negative_verdicts=frozenset({"NO_CALIBRATED_MANIPULATION_SIGNAL"}),
        abstention_verdicts=frozenset({"INCONCLUSIVE"}),
        supported_positive_labels=frozenset(
            {"AI_GENERATED", "FACE_SWAP", "AUDIO_MANIPULATION", "OTHER_MANIPULATION"}
        ),
        supported_negative_labels=frozenset({"GENUINE"}),
        unsupported_labels=frozenset(),
        excluded_labels=frozenset({"UNKNOWN"}),
        rationale="test",
    )
    wider.validate()
    contracts = {**om.SCOPE_CONTRACTS, wider.rules_version: wider}
    old = row("id-1", rules=V5, label="AUDIO_MANIPULATION", verdict="NO_CALIBRATED_MANIPULATION_SIGNAL")
    new = row("id-2", rules="r99-v9.0.0", label="AUDIO_MANIPULATION", verdict="NO_CALIBRATED_MANIPULATION_SIGNAL")
    assert om.classify(old, contracts) == om.UNSUPPORTED_FOR_RULESET
    assert om.classify(new, contracts) == om.FN


def test_a_contract_that_leaves_a_label_unaccounted_for_is_refused():
    broken = om.ScopeContract(
        rules_version="broken",
        positive_verdicts=frozenset({"X"}),
        negative_verdicts=frozenset({"Y"}),
        abstention_verdicts=frozenset({"Z"}),
        supported_positive_labels=frozenset({"AI_GENERATED"}),
        supported_negative_labels=frozenset({"GENUINE"}),
        unsupported_labels=frozenset(),
        excluded_labels=frozenset({"UNKNOWN"}),
        rationale="test",
    )
    with pytest.raises(ValueError):
        broken.validate()


# --- verdict mapping -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "verdict", "outcome"),
    [
        ("AI_GENERATED", "MANIPULATION_DETECTED", "TP"),
        ("FACE_SWAP", "MANIPULATION_DETECTED", "TP"),
        ("GENUINE", "MANIPULATION_DETECTED", "FP"),
        ("GENUINE", "NO_CALIBRATED_MANIPULATION_SIGNAL", "TN"),
        ("AI_GENERATED", "NO_CALIBRATED_MANIPULATION_SIGNAL", "FN"),
        ("FACE_SWAP", "NO_CALIBRATED_MANIPULATION_SIGNAL", "FN"),
    ],
)
def test_the_v5_confusion_cells(label, verdict, outcome):
    assert om.classify(row(label=label, verdict=verdict)) == outcome


@pytest.mark.parametrize("verdict", ["INCONCLUSIVE", None])
@pytest.mark.parametrize("label", ["AI_GENERATED", "FACE_SWAP", "GENUINE"])
def test_inconclusive_or_null_verdict_is_an_abstention_not_a_false_negative(verdict, label):
    report = om.evaluate([row(label=label, verdict=verdict)])
    group = only_group(report)
    assert group["abstentions"] == 1
    assert group["confusion"] == {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    assert group["decided_units"] == 0
    assert group["rates"]["abstention_rate"]["observed"] == 1.0
    assert group["rates"]["recall"]["n"] == 0


def test_no_recorded_decision_is_its_own_exclusion():
    report = om.evaluate([row(rules=None, calibration=None, verdict=None)])
    assert only_group(report)["exclusions"]["no_decision_recorded"] == 1


def test_a_verdict_outside_the_contract_raises_rather_than_being_bucketed():
    with pytest.raises(ValueError):
        om.classify(row(verdict="HIGH"))


def test_a_ground_truth_value_outside_the_vocabulary_raises():
    with pytest.raises(ValueError):
        om.classify(row(label="DEEPFAKE"))
    with pytest.raises(ValueError):
        om.classify(row(source_class="ANALYST"))


# --- metrics -------------------------------------------------------------------------------


def test_rates_and_bounds_come_from_the_confusion_matrix():
    rows = (
        [row(f"tp-{i}", sha=f"{i:064x}", label="AI_GENERATED") for i in range(3)]
        + [row(f"fn-{i}", sha=f"{i + 10:064x}", label="FACE_SWAP", verdict="NO_CALIBRATED_MANIPULATION_SIGNAL") for i in range(1)]
        + [row(f"tn-{i}", sha=f"{i + 20:064x}", label="GENUINE", verdict="NO_CALIBRATED_MANIPULATION_SIGNAL") for i in range(4)]
        + [row(f"fp-{i}", sha=f"{i + 30:064x}", label="GENUINE") for i in range(1)]
        + [row(f"ab-{i}", sha=f"{i + 40:064x}", label="GENUINE", verdict="INCONCLUSIVE") for i in range(1)]
    )
    group = only_group(om.evaluate(rows))
    rates = group["rates"]

    assert group["confusion"] == {"TP": 3, "FP": 1, "TN": 4, "FN": 1}
    assert (rates["precision"]["k"], rates["precision"]["n"]) == (3, 4)
    assert (rates["recall"]["k"], rates["recall"]["n"]) == (3, 4)
    assert (rates["specificity"]["k"], rates["specificity"]["n"]) == (4, 5)
    assert (rates["false_positive_rate"]["k"], rates["false_positive_rate"]["n"]) == (1, 5)
    assert (rates["false_negative_rate"]["k"], rates["false_negative_rate"]["n"]) == (1, 4)
    assert (rates["abstention_rate"]["k"], rates["abstention_rate"]["n"]) == (1, 10)
    assert rates["false_positive_rate"]["upper_95_one_sided"] == pytest.approx(
        om.stats.clopper_pearson_upper(1, 5)
    )
    assert group["by_ground_truth_label"]["FACE_SWAP"]["FN"] == 1


def test_an_empty_denominator_is_undefined_not_zero():
    group = only_group(om.evaluate([row(label="GENUINE", verdict="NO_CALIBRATED_MANIPULATION_SIGNAL")]))
    assert group["rates"]["recall"]["observed"] is None
    assert group["rates"]["false_positive_rate"]["observed"] == 0.0
    assert group["rates"]["false_positive_rate"]["upper_95_one_sided"] > 0.0


def test_snapshot_carries_the_inputs_and_no_dataset_split():
    report = om.evaluate([row("id-1", verdict="INCONCLUSIVE")])
    [unit] = report["snapshot"]
    assert {
        "media_sha256",
        "ground_truth_source_class",
        "ground_truth_label",
        "analysis_id",
        "verdict",
        "risk_rules_version",
        "risk_calibration_id",
        "analysis_created_at",
        "outcome",
        "dataset_split",
    } <= set(unit)
    assert unit["dataset_split"] is None
    assert only_group(report)["dataset_split_status"] == "unavailable"


# --- the contract against the production sources -------------------------------------------


def _literal_values(module_path: Path, name: str) -> tuple[str, ...]:
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(elt) for elt in node.value.slice.elts)
    raise AssertionError(f"{name} not found in {module_path}")


def test_ground_truth_vocabulary_matches_the_r12_t1_contract():
    source = API_APP / "ground_truth.py"
    assert _literal_values(source, "SourceClass") == om.SOURCE_CLASSES
    assert _literal_values(source, "GroundTruthLabel") == om.GROUND_TRUTH_LABELS


def test_v5_contract_uses_the_risk_engine_verdict_strings():
    spec = importlib.util.spec_from_file_location("r12t4_risk_engine", API_APP / "risk_engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    contract = om.SCOPE_CONTRACTS[engine.RULES_VERSION_V5]
    assert contract.positive_verdicts == {engine.VERDICT_MANIPULATION_DETECTED}
    assert contract.negative_verdicts == {engine.VERDICT_NO_SIGNAL}
    assert contract.abstention_verdicts == {engine.VERDICT_INCONCLUSIVE}
