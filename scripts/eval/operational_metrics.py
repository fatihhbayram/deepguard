"""R12-T4: score stored automated verdicts against recorded Ground Truth, offline.

    set -a; . ./.env; set +a
    PYTHONPATH=apps/api:scripts apps/api/.venv/bin/python scripts/eval/operational_metrics.py \
        --output ../deepguard-corpus/r12t4/operational_metrics.json

This script measures. It changes nothing: no file under `apps/` is written, no endpoint is
added, and the database session it opens is `READ ONLY` at the transaction level, so a bug in
here cannot become a write to the forensic record.

**The join.** `GroundTruth` is keyed by bytes, not by a row: `ground_truth.media_sha256` holds
the same value as `media_files.original_sha256`, and neither table has a foreign key or ORM
relationship to the other (see the `GroundTruth` docstring for why). There is no
`Analysis.media` attribute either. The path is therefore

    analyses ⟶ media_files  ON media_files.analysis_id = analyses.id
             ⟶ ground_truth ON ground_truth.media_sha256 = media_files.original_sha256

with the last step an outer join, so analyses nobody has labelled are counted rather than
silently lost. The join is written against the imported models, not against table-name
strings, so a renamed column breaks this script at import rather than returning nothing.

**The evaluation unit** is `(media_sha256, risk_rules_version, risk_calibration_id)`. The same
bytes submitted five times under one ruleset and one calibration are one observation of that
ruleset, not five: the latest analysis (by `created_at`, then `id`) represents the unit and
the others are counted as `duplicate_analyses`, never added to `n`. The same bytes under two
calibration ids are two units, because they are two different measurements.

**Metrics are never pooled across rulesets, calibrations or splits.** Each `(rules_version,
calibration_id, dataset_split)` triple is its own group with its own confusion matrix. A verdict
only means what its ruleset says it means, and adding a v4 `HIGH` to a v5
`MANIPULATION_DETECTED` would count two different claims as one.

**What a unit may be scored against** is decided in this order, and the first rule that applies
is the unit's outcome:

1. no Ground Truth recorded for the bytes → `no_ground_truth`;
2. `source_class = UNKNOWN` → `ground_truth_ineligible`, *whatever the label says*: a label
   nobody is in a position to know is not a truth to measure against;
3. `label = UNKNOWN` → `ground_truth_label_unknown`;
4. no decision recorded (`risk_rules_version` is null: queued, running or failed) →
   `no_decision_recorded`. There is no ruleset to read the missing verdict under, so it is
   not an abstention *of* any ruleset;
5. the ruleset has no entry in `SCOPE_CONTRACTS` → `ruleset_without_scope_contract`;
6. the label is outside the ruleset's claimed coverage → `unsupported_for_ruleset`. A detector
   is not charged a false negative for a manipulation family it never claimed to see;
7. otherwise the verdict decides: positive, negative, or `abstention` (`INCONCLUSIVE`, or a
   null verdict under a known ruleset). An abstention is not a false negative; it is reported
   as its own rate beside the confusion matrix.

A Ground Truth value outside the R12-T1 vocabulary, or a verdict outside its ruleset's
vocabulary, raises rather than being bucketed: either one means the contract below is wrong,
and a count would hide that.

**Dataset split** (R12-T5) is read, not inferred. The split belongs to a source lineage
(`lineage_splits`), and a file reaches it through its governance record:

    media_files ⟶ media_governance ON media_governance.media_sha256 = media_files.original_sha256
                ⟶ lineage_splits   ON lineage_splits.source_lineage_id = media_governance.source_lineage_id

both as outer joins. Groups are keyed by `(risk_rules_version, risk_calibration_id,
dataset_split)`, so a calibration split and a holdout split are never pooled into one matrix.
Bytes with no governance record carry `dataset_split: "unavailable"` and form their own group;
they are never assigned a split by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import stats

SCHEMA_VERSION = "deepguard/r12-t4-operational-metrics/2"
TASK = "R12-T4"

REPO_ROOT = Path(__file__).resolve().parents[2]

# The R12-T5 split vocabulary (`app.dataset_governance.DatasetSplit`), restated for the same
# reason as the Ground Truth vocabulary below, and the value a unit with no governance carries.
DATASET_SPLITS = ("CALIBRATION", "VALIDATION", "TEST", "HOLDOUT")
SPLIT_UNAVAILABLE = "unavailable"

# The R12-T1 vocabulary (`app.ground_truth`), restated so this module runs without pydantic.
# `tests/test_operational_metrics.py` checks it against the source.
SOURCE_CLASSES = ("OWNER_KNOWN", "CONTROLLED_TEST", "EXTERNAL_VERIFIED", "UNKNOWN")
ELIGIBLE_SOURCE_CLASSES = frozenset({"OWNER_KNOWN", "CONTROLLED_TEST", "EXTERNAL_VERIFIED"})
GROUND_TRUTH_LABELS = (
    "GENUINE",
    "AI_GENERATED",
    "FACE_SWAP",
    "AUDIO_MANIPULATION",
    "OTHER_MANIPULATION",
    "UNKNOWN",
)

# Unit outcomes. The four confusion cells and the abstention are "in scope"; everything else is
# an exclusion with a named reason.
TP, FP, TN, FN, ABSTENTION = "TP", "FP", "TN", "FN", "abstention"
IN_SCOPE_OUTCOMES = (TP, FP, TN, FN, ABSTENTION)
NO_GROUND_TRUTH = "no_ground_truth"
GROUND_TRUTH_INELIGIBLE = "ground_truth_ineligible"
GROUND_TRUTH_LABEL_UNKNOWN = "ground_truth_label_unknown"
NO_DECISION_RECORDED = "no_decision_recorded"
RULESET_WITHOUT_SCOPE_CONTRACT = "ruleset_without_scope_contract"
UNSUPPORTED_FOR_RULESET = "unsupported_for_ruleset"
EXCLUSION_OUTCOMES = (
    NO_GROUND_TRUTH,
    GROUND_TRUTH_INELIGIBLE,
    GROUND_TRUTH_LABEL_UNKNOWN,
    NO_DECISION_RECORDED,
    RULESET_WITHOUT_SCOPE_CONTRACT,
    UNSUPPORTED_FOR_RULESET,
)


@dataclass(frozen=True)
class ScopeContract:
    """What one immutable ruleset claims it can tell apart, and how its verdicts read.

    Keyed by `risk_rules_version`, never by "whatever runs now": a historical row is scored
    against the claims of the ruleset that decided it, so widening a later ruleset's coverage
    cannot retroactively turn an old row's out-of-scope miss into a false negative.

    The four label sets and the three verdict sets must each partition their vocabulary;
    `validate` refuses a contract that leaves a label or verdict unaccounted for.
    """

    rules_version: str
    positive_verdicts: frozenset[str]
    negative_verdicts: frozenset[str]
    abstention_verdicts: frozenset[str]
    supported_positive_labels: frozenset[str]
    supported_negative_labels: frozenset[str]
    unsupported_labels: frozenset[str]
    excluded_labels: frozenset[str]
    rationale: str

    def validate(self) -> None:
        label_sets = (
            self.supported_positive_labels,
            self.supported_negative_labels,
            self.unsupported_labels,
            self.excluded_labels,
        )
        covered = [label for labels in label_sets for label in labels]
        if sorted(covered) != sorted(GROUND_TRUTH_LABELS):
            raise ValueError(
                f"{self.rules_version}: scope contract does not partition the Ground Truth "
                f"labels exactly once each ({sorted(covered)})"
            )
        verdict_sets = (self.positive_verdicts, self.negative_verdicts, self.abstention_verdicts)
        verdicts = [verdict for verdicts in verdict_sets for verdict in verdicts]
        if len(verdicts) != len(set(verdicts)):
            raise ValueError(f"{self.rules_version}: a verdict is mapped twice")

    def as_dict(self) -> dict:
        return {
            "rules_version": self.rules_version,
            "positive_verdicts": sorted(self.positive_verdicts),
            "negative_verdicts": sorted(self.negative_verdicts),
            "abstention_verdicts": sorted(self.abstention_verdicts) + [None],
            "supported_positive_labels": sorted(self.supported_positive_labels),
            "supported_negative_labels": sorted(self.supported_negative_labels),
            "unsupported_labels": sorted(self.unsupported_labels),
            "excluded_labels": sorted(self.excluded_labels),
            "rationale": self.rationale,
        }


# The evaluation scope contract. One entry per ruleset that may be scored; a ruleset absent from
# here is reported under `ruleset_without_scope_contract` and never scored by analogy.
#
# `r9-v5.0.0` decides from two calibrated, decision-eligible detectors: NVIDIA SVD (generated
# video) and EfficientNet-B7 (face appearance). Audio is evidence-only under v5 and nothing in
# it claims "other" manipulation, so those two labels are outside its coverage.
#
# The v4-era rulesets (`p7-v1.0.0` … `r7-v4.0.0`, verdicts `HIGH`/`MEDIUM`/`UNKNOWN`) are left
# out deliberately: `MEDIUM` was never "no manipulation", so there is no honest negative to map
# it to, and inventing one here would be a new claim about rules that no longer run.
SCOPE_CONTRACTS: dict[str, ScopeContract] = {
    "r9-v5.0.0": ScopeContract(
        rules_version="r9-v5.0.0",
        positive_verdicts=frozenset({"MANIPULATION_DETECTED"}),
        negative_verdicts=frozenset({"NO_CALIBRATED_MANIPULATION_SIGNAL"}),
        abstention_verdicts=frozenset({"INCONCLUSIVE"}),
        supported_positive_labels=frozenset({"AI_GENERATED", "FACE_SWAP"}),
        supported_negative_labels=frozenset({"GENUINE"}),
        unsupported_labels=frozenset({"AUDIO_MANIPULATION", "OTHER_MANIPULATION"}),
        excluded_labels=frozenset({"UNKNOWN"}),
        rationale=(
            "Decision-eligible detectors under r9-v5.0.0 are NVIDIA SVD (generated video) and "
            "EfficientNet-B7 (face appearance). Audio is evidence-only; no rule claims other "
            "manipulation families."
        ),
    ),
}

for _contract in SCOPE_CONTRACTS.values():
    _contract.validate()


@dataclass(frozen=True)
class EvaluationRow:
    """One analysis as read from the database, with the Ground Truth for its bytes if any."""

    analysis_id: str
    analysis_created_at: str
    analysis_status: str
    media_sha256: str
    risk_rules_version: str | None
    risk_calibration_id: str | None
    risk_level: str | None
    risk_rule_id: str | None
    gt_source_class: str | None = None
    gt_label: str | None = None
    gt_updated_at: str | None = None
    dataset_split: str | None = None

    @property
    def unit_key(self) -> tuple[str, str | None, str | None]:
        return (self.media_sha256, self.risk_rules_version, self.risk_calibration_id)

    @property
    def has_ground_truth(self) -> bool:
        return self.gt_source_class is not None

    @property
    def split(self) -> str:
        """The recorded split, or `unavailable`; a value outside the vocabulary raises."""
        if self.dataset_split is None:
            return SPLIT_UNAVAILABLE
        if self.dataset_split not in DATASET_SPLITS:
            raise ValueError(
                f"unknown dataset_split {self.dataset_split!r} for {self.media_sha256}"
            )
        return self.dataset_split


@dataclass
class EvaluationUnit:
    representative: EvaluationRow
    duplicates: list[EvaluationRow] = field(default_factory=list)
    outcome: str | None = None


def _recency(row: EvaluationRow) -> tuple[str, str]:
    # ISO-8601 in UTC sorts lexically; the id breaks ties so the choice never depends on
    # database return order.
    return (row.analysis_created_at, row.analysis_id)


def deduplicate(rows: list[EvaluationRow]) -> list[EvaluationUnit]:
    """Collapse analyses into evaluation units, keeping the latest as the representative."""
    by_key: dict[tuple, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        by_key[row.unit_key].append(row)
    units = []
    for key in sorted(by_key, key=lambda k: tuple("" if part is None else part for part in k)):
        members = sorted(by_key[key], key=_recency, reverse=True)
        units.append(EvaluationUnit(representative=members[0], duplicates=members[1:]))
    return units


def classify(row: EvaluationRow, contracts: dict[str, ScopeContract] = SCOPE_CONTRACTS) -> str:
    """The outcome of one unit, by the ordered rules in the module docstring."""
    if not row.has_ground_truth:
        return NO_GROUND_TRUTH
    if row.gt_source_class not in SOURCE_CLASSES:
        raise ValueError(f"unknown Ground Truth source_class {row.gt_source_class!r}")
    if row.gt_label not in GROUND_TRUTH_LABELS:
        raise ValueError(f"unknown Ground Truth label {row.gt_label!r}")
    if row.gt_source_class not in ELIGIBLE_SOURCE_CLASSES:
        return GROUND_TRUTH_INELIGIBLE
    if row.gt_label == "UNKNOWN":
        return GROUND_TRUTH_LABEL_UNKNOWN
    if row.risk_rules_version is None:
        return NO_DECISION_RECORDED
    contract = contracts.get(row.risk_rules_version)
    if contract is None:
        return RULESET_WITHOUT_SCOPE_CONTRACT
    if row.gt_label in contract.unsupported_labels or row.gt_label in contract.excluded_labels:
        return UNSUPPORTED_FOR_RULESET

    truth_positive = row.gt_label in contract.supported_positive_labels
    verdict = row.risk_level
    if verdict is None or verdict in contract.abstention_verdicts:
        return ABSTENTION
    if verdict in contract.positive_verdicts:
        return TP if truth_positive else FP
    if verdict in contract.negative_verdicts:
        return FN if truth_positive else TN
    raise ValueError(
        f"verdict {verdict!r} on analysis {row.analysis_id} is not in the "
        f"{row.risk_rules_version} scope contract"
    )


def _rates(counts: Counter) -> dict:
    tp, fp, tn, fn, abstained = (counts[o] for o in (TP, FP, TN, FN, ABSTENTION))
    return {
        "precision": stats.rate(tp, tp + fp).as_dict(),
        "recall": stats.rate(tp, tp + fn).as_dict(),
        "specificity": stats.rate(tn, tn + fp).as_dict(),
        "false_positive_rate": stats.rate(fp, tn + fp).as_dict(),
        "false_negative_rate": stats.rate(fn, tp + fn).as_dict(),
        "abstention_rate": stats.rate(abstained, tp + fp + tn + fn + abstained).as_dict(),
    }


def _unit_snapshot(unit: EvaluationUnit) -> dict:
    row = unit.representative
    return {
        "media_sha256": row.media_sha256,
        "risk_rules_version": row.risk_rules_version,
        "risk_calibration_id": row.risk_calibration_id,
        "analysis_id": row.analysis_id,
        "analysis_created_at": row.analysis_created_at,
        "analysis_status": row.analysis_status,
        "verdict": row.risk_level,
        "risk_rule_id": row.risk_rule_id,
        "ground_truth_source_class": row.gt_source_class,
        "ground_truth_label": row.gt_label,
        "ground_truth_updated_at": row.gt_updated_at,
        "outcome": unit.outcome,
        "duplicate_analysis_ids": [d.analysis_id for d in unit.duplicates],
        "dataset_split": row.split,
    }


def evaluate(
    rows: list[EvaluationRow], contracts: dict[str, ScopeContract] = SCOPE_CONTRACTS
) -> dict:
    """Deduplicate, classify and count. Pure: no database, no clock, no filesystem."""
    units = deduplicate(rows)
    for unit in units:
        unit.outcome = classify(unit.representative, contracts)

    groups: dict[tuple, list[EvaluationUnit]] = defaultdict(list)
    for unit in units:
        row = unit.representative
        groups[(row.risk_rules_version, row.risk_calibration_id, row.split)].append(unit)

    group_reports = []
    for (rules_version, calibration_id, split) in sorted(
        groups, key=lambda k: tuple("" if part is None else part for part in k)
    ):
        members = groups[(rules_version, calibration_id, split)]
        counts = Counter(unit.outcome for unit in members)
        by_label: dict[str, Counter] = defaultdict(Counter)
        for unit in members:
            if unit.outcome in IN_SCOPE_OUTCOMES:
                by_label[unit.representative.gt_label][unit.outcome] += 1
        in_scope = sum(counts[o] for o in IN_SCOPE_OUTCOMES)
        group_reports.append(
            {
                "risk_rules_version": rules_version,
                "risk_calibration_id": calibration_id,
                "scope_contract": rules_version in contracts,
                "dataset_split": split,
                "units": len(members),
                "analyses": sum(1 + len(unit.duplicates) for unit in members),
                "duplicate_analyses": sum(len(unit.duplicates) for unit in members),
                "in_scope_units": in_scope,
                "decided_units": in_scope - counts[ABSTENTION],
                "confusion": {o: counts[o] for o in (TP, FP, TN, FN)},
                "abstentions": counts[ABSTENTION],
                "exclusions": {o: counts[o] for o in EXCLUSION_OUTCOMES},
                "by_ground_truth_label": {
                    label: {o: by_label[label][o] for o in IN_SCOPE_OUTCOMES}
                    for label in sorted(by_label)
                },
                "rates": _rates(counts),
            }
        )

    totals = Counter(unit.outcome for unit in units)
    return {
        "input": {
            "analysis_rows": len(rows),
            "evaluation_units": len(units),
            "duplicate_analyses": sum(len(unit.duplicates) for unit in units),
            "outcomes": {o: totals[o] for o in IN_SCOPE_OUTCOMES + EXCLUSION_OUTCOMES},
        },
        "groups": group_reports,
        # Only units with a Ground Truth record: the no-GT population is summarised by its counts
        # above and in each group, not listed, since nothing about it was measured.
        "snapshot": [_unit_snapshot(unit) for unit in units if unit.outcome != NO_GROUND_TRUTH],
    }


# ---------------------------------------------------------------------------------------------
# Database and provenance. Everything above this line is pure and is what the tests exercise.
# ---------------------------------------------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def load_rows(session) -> tuple[list[EvaluationRow], dict]:
    """Read every analysis with its media hash and any Ground Truth for those bytes."""
    from sqlalchemy import func, select

    from app.db.models import Analysis, GroundTruth, LineageSplit, MediaFile, MediaGovernance

    statement = (
        select(
            Analysis.id,
            Analysis.created_at,
            Analysis.status,
            Analysis.risk_rules_version,
            Analysis.risk_calibration_id,
            Analysis.risk_level,
            Analysis.risk_rule_id,
            MediaFile.original_sha256,
            GroundTruth.source_class,
            GroundTruth.label,
            GroundTruth.updated_at,
            LineageSplit.dataset_split,
        )
        .join(MediaFile, MediaFile.analysis_id == Analysis.id)
        .outerjoin(GroundTruth, GroundTruth.media_sha256 == MediaFile.original_sha256)
        .outerjoin(MediaGovernance, MediaGovernance.media_sha256 == MediaFile.original_sha256)
        .outerjoin(
            LineageSplit, LineageSplit.source_lineage_id == MediaGovernance.source_lineage_id
        )
        .order_by(Analysis.created_at, Analysis.id)
    )
    raw = session.execute(statement).all()

    analyses_total = session.execute(select(func.count()).select_from(Analysis)).scalar_one()
    analyses_with_media = {r.id for r in raw}

    # `media_files.analysis_id` is indexed but not unique. An analysis whose rows name two
    # different hashes has no single byte identity to score, so it is set aside and counted.
    hashes_by_analysis: dict = defaultdict(set)
    for r in raw:
        hashes_by_analysis[r.id].add(r.original_sha256)
    ambiguous = {aid for aid, hashes in hashes_by_analysis.items() if len(hashes) > 1}

    rows: list[EvaluationRow] = []
    seen: set = set()
    for r in raw:
        if r.id in ambiguous or r.id in seen:
            continue
        seen.add(r.id)
        rows.append(
            EvaluationRow(
                analysis_id=str(r.id),
                analysis_created_at=_iso(r.created_at),
                analysis_status=r.status,
                media_sha256=r.original_sha256,
                risk_rules_version=r.risk_rules_version,
                risk_calibration_id=r.risk_calibration_id,
                risk_level=r.risk_level,
                risk_rule_id=r.risk_rule_id,
                gt_source_class=r.source_class,
                gt_label=r.label,
                gt_updated_at=_iso(r.updated_at),
                dataset_split=r.dataset_split,
            )
        )

    query_facts = {
        "join": (
            "analyses JOIN media_files ON media_files.analysis_id = analyses.id "
            "LEFT OUTER JOIN ground_truth ON ground_truth.media_sha256 = "
            "media_files.original_sha256 "
            "LEFT OUTER JOIN media_governance ON media_governance.media_sha256 = "
            "media_files.original_sha256 "
            "LEFT OUTER JOIN lineage_splits ON lineage_splits.source_lineage_id = "
            "media_governance.source_lineage_id"
        ),
        "analyses_total": analyses_total,
        "analyses_without_media": analyses_total - len(analyses_with_media),
        "analyses_with_ambiguous_media_identity": sorted(str(a) for a in ambiguous),
        "ground_truth_records_total": session.execute(
            select(func.count()).select_from(GroundTruth)
        ).scalar_one(),
        "media_governance_records_total": session.execute(
            select(func.count()).select_from(MediaGovernance)
        ).scalar_one(),
    }
    return rows, query_facts


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def provenance() -> dict:
    script = Path(__file__).resolve()
    dirty = _git("status", "--porcelain", "--", str(script.relative_to(REPO_ROOT)))
    return {
        "script": str(script.relative_to(REPO_ROOT)),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "git_commit": _git("rev-parse", "HEAD"),
        "script_uncommitted": None if dirty is None else bool(dirty),
        "python": platform.python_version(),
    }


def build_report(rows: list[EvaluationRow], query_facts: dict, database: dict) -> dict:
    result = evaluate(rows)
    return {
        "schema": SCHEMA_VERSION,
        "task": TASK,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance(),
        "database": database,
        "query": query_facts,
        "evaluation_unit": ["media_sha256", "risk_rules_version", "risk_calibration_id"],
        "representative_selection": "latest analysis by (created_at, id); others are duplicates",
        "pooling": (
            "none: one confusion matrix per "
            "(risk_rules_version, risk_calibration_id, dataset_split)"
        ),
        "dataset_split_source": (
            "lineage_splits.dataset_split via media_governance.source_lineage_id; "
            f"{SPLIT_UNAVAILABLE!r} where no governance is recorded"
        ),
        "confidence_bounds": "exact one-sided 95% Clopper-Pearson (scripts/eval/stats.py)",
        "scope_contracts": {k: c.as_dict() for k, c in sorted(SCOPE_CONTRACTS.items())},
        **result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    from sqlalchemy import text

    from app.db.session import SessionLocal, database_url

    url = database_url()
    database = {
        "driver": url.drivername,
        "host": url.host,
        "port": url.port,
        "database": url.database,
    }
    with SessionLocal() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        rows, query_facts = load_rows(session)
        session.rollback()

    report = build_report(rows, query_facts, database)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")

    totals = report["input"]
    print(
        f"{totals['analysis_rows']} analyses → {totals['evaluation_units']} units "
        f"({totals['duplicate_analyses']} duplicates); outcomes {totals['outcomes']}"
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
