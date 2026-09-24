"""R12-T6: export governed, Ground-Truth-labelled production media as a replayable corpus.

    set -a; . ./.env; set +a
    PYTHONPATH=apps/api:scripts apps/api/.venv/bin/python \
        scripts/eval/export_operational_dataset.py --out-dir ../deepguard-corpus/r12t6/export \
        [--split HOLDOUT]

and then, one results file per detector slot:

    PYTHONPATH=scripts python3 scripts/eval/replay.py \
        --corpus ../deepguard-corpus/r12t6/export/corpus.json \
        --svd-run ../deepguard-corpus/r12t6/export/results_nvidia_synthetic_video_….json \
        --output ../deepguard-corpus/r12t6/replay.json

This script reads. The database session is `READ ONLY` at the transaction level, nothing under
`apps/` is written, and no threshold, ruleset or calibration is touched. What it writes goes to a
new or empty `--out-dir`, and it refuses anything else, so an export is never merged into an
older one.

**A record is exported only when everything it needs was recorded by a person.** The universe is
the `ground_truth` table. For each record, in this order, the first rule that applies decides:

1. `source_class = UNKNOWN` → `ground_truth_ineligible`;
2. `label = UNKNOWN` → `unknown_label`;
3. a label with no benchmark counterpart (`AUDIO_MANIPULATION`, `OTHER_MANIPULATION`) →
   `unsupported_label`. `GENUINE`, `AI_GENERATED` and `FACE_SWAP` map to `real`, `synthetic` and
   `face_swap` (`eval.corpus`), and nothing else is mapped;
4. no governance record, or any of the eight corpus fields null (`license`,
   `permission_status`, `redistributable`, `private`, `stratum_primary`, `source`,
   `acquisition_type`, `benchmark_family`), or a derived file whose `transformations` were never
   recorded → `insufficient_corpus_metadata`. `False` is a statement and is exported; null is
   not, and is never filled in;
5. no media row for the bytes → `no_media_file`;
6. a storage key that is not the content-addressed `originals/<media_sha256>` →
   `unusable_storage_path`.

A record outside `--split` is not an exclusion; it is counted under `split_filter`.

**Field mapping** into `eval.corpus.CorpusItem`, with nothing derived that the database does not
hold: `clip_id` and `sha256` are the media hash; `family` is `MediaGovernance.benchmark_family`
and nothing else (never the Ground Truth family); `label_provenance` is
`GroundTruth.source_class`; `split` is the lineage's `dataset_split`; `derivative_of` is
`derived_from_sha256`; `base_media_id` is the root of that chain; `derivation` is `"none"` for a
file with no parent and no recorded steps, else the recorded `transformations` joined with `+`;
`bytes` and `path` come from the representative media row. `strata` stays empty: no column
records it.

**The path is the MinIO object key, verbatim.** `original_storage_key` is `originals/<sha256>` in
the `deepguard-originals` bucket. It is not converted to anything. `replay.py` never reads
`path`; a harness that opens files has to resolve it against a mirror of that bucket, and
`export_metadata.json` says so under `path_semantics`.

**One media row represents the bytes.** The same bytes uploaded twice are one corpus item. The
representative is the earliest upload — `media_files` has no timestamp of its own, so its
analysis's `created_at` — with the media row's `id` breaking ties; the others are counted.

**Signals.** Only the three detector slots `replay.py` feeds into the rules are exported:
`nvidia/synthetic_video`, `efficientnet-b7/face_manipulation`, `lipforensics/lip_forensics`.
Every other `(provider, signal_type)` is `unsupported_signal_schema`, as is any status other than
`SUCCESS` (→ `"ok"`) and `FAILED` (→ `"error"`, which `replay.py` reads back as `FAILED`). A
signal with no stored `provider_version` — most `FAILED` rows — is `missing_provider_version`:
its exact detector identity cannot be established, so it goes into no results file. It is never
given a guessed version, the version of a successful run, or an "unversioned" identity. A
`SUCCESS` without a score or without its count (`total_clips`, `frames_scored`,
`windows_scored`) is `invalid_success_payload`: no zero is written in its place.

Runs are deduplicated on `(media_sha256, provider, provider_version, signal_type)`, the latest by
`(created_at, id)` representing the rest. Exactly one `results_*.json` is written per
`(provider, provider_version, signal_type)`; two versions of one detector are two files, never
one. The file states `provider`, `provider_version` (as stored) and
`signal_type` at the top level, and `replay.py` uses that `provider_version` as is. The file
has no `model_provenance`: nothing the database did not record as provenance is written as
provenance. Each clip's count is copied from the signal's stored metadata into an explicit
top-level `evidence_counts`, keyed by the metadata name (`total_clips`, `frames_scored`,
`windows_scored`).

**Leakage** is checked by `eval.corpus.leakage_findings` over the four operational splits before
anything is written. Real leakage stops the export with nothing written. A derivative whose
parent is governed but was not exported is reported as `missing_exported_parent` and does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.corpus import (
    LABEL_FACE_SWAP,
    LABEL_REAL,
    LABEL_SYNTHETIC,
    OPERATIONAL_SPLITS,
    CorpusItem,
    leakage_findings,
    write_corpus,
    write_manifest,
)

TASK = "R12-T6"
METADATA_SCHEMA = "deepguard/r12-t6-export-metadata/1"
RESULTS_SCHEMA = "deepguard/r12-t6-results/1"

REPO_ROOT = Path(__file__).resolve().parents[2]

# The R12-T1 vocabulary, restated so the pure half runs without pydantic (as in
# `operational_metrics.py`). A value outside it raises: the contract would be wrong, not the row.
SOURCE_CLASSES = ("OWNER_KNOWN", "CONTROLLED_TEST", "EXTERNAL_VERIFIED", "UNKNOWN")
GROUND_TRUTH_LABELS = (
    "GENUINE",
    "AI_GENERATED",
    "FACE_SWAP",
    "AUDIO_MANIPULATION",
    "OTHER_MANIPULATION",
    "UNKNOWN",
)
LABEL_MAP = {
    "GENUINE": LABEL_REAL,
    "AI_GENERATED": LABEL_SYNTHETIC,
    "FACE_SWAP": LABEL_FACE_SWAP,
}

# The eight corpus fields R12-T5A records on `media_governance`; a null in any one excludes.
CORPUS_METADATA_FIELDS = (
    "license",
    "permission_status",
    "redistributable",
    "private",
    "stratum_primary",
    "source",
    "acquisition_type",
    "benchmark_family",
)

STORAGE_BUCKET = "deepguard-originals"
STORAGE_PREFIX = "originals/"

# Exclusion reasons, in the order they are decided.
GROUND_TRUTH_INELIGIBLE = "ground_truth_ineligible"
UNKNOWN_LABEL = "unknown_label"
UNSUPPORTED_LABEL = "unsupported_label"
INSUFFICIENT_CORPUS_METADATA = "insufficient_corpus_metadata"
NO_MEDIA_FILE = "no_media_file"
UNUSABLE_STORAGE_PATH = "unusable_storage_path"
UNSUPPORTED_SIGNAL_SCHEMA = "unsupported_signal_schema"
INVALID_SUCCESS_PAYLOAD = "invalid_success_payload"
MISSING_PROVIDER_VERSION = "missing_provider_version"
SIGNAL_EXCLUSION_REASONS = (
    UNSUPPORTED_SIGNAL_SCHEMA,
    MISSING_PROVIDER_VERSION,
    INVALID_SUCCESS_PAYLOAD,
)
EXCLUSION_REASONS = (
    INSUFFICIENT_CORPUS_METADATA,
    GROUND_TRUTH_INELIGIBLE,
    UNKNOWN_LABEL,
    UNSUPPORTED_LABEL,
    UNSUPPORTED_SIGNAL_SCHEMA,
    INVALID_SUCCESS_PAYLOAD,
    NO_MEDIA_FILE,
    UNUSABLE_STORAGE_PATH,
    MISSING_PROVIDER_VERSION,
)

MEDIA_REPRESENTATIVE_POLICY = (
    "earliest upload: owning analyses.created_at ASC, then media_files.id ASC "
    "(media_files has no created_at of its own)"
)
SIGNAL_REPRESENTATIVE_POLICY = (
    "per (media_sha256, provider, provider_version, signal_type): "
    "latest analysis_signals.created_at, then analysis_signals.id, descending"
)


@dataclass(frozen=True)
class ReplaySlot:
    """A detector `replay.py` feeds into the rules, and the name of the clip's count."""

    provider: str
    signal_type: str
    metadata_count_key: str


# Restated from `apps/api/app/risk_engine.py` (`*_PROVIDER`, `*_SIGNAL_TYPE`) and from the count
# names `replay.build_evidence` reads. The tests check both against the source.
REPLAY_SLOTS = {
    ("nvidia", "synthetic_video"): ReplaySlot("nvidia", "synthetic_video", "total_clips"),
    ("efficientnet-b7", "face_manipulation"): ReplaySlot(
        "efficientnet-b7", "face_manipulation", "frames_scored"
    ),
    ("lipforensics", "lip_forensics"): ReplaySlot(
        "lipforensics", "lip_forensics", "windows_scored"
    ),
}
STATUS_MAP = {"SUCCESS": "ok", "FAILED": "error"}


class ExportError(Exception):
    """The export cannot be written honestly; nothing has been."""


@dataclass(frozen=True)
class GroundTruthRow:
    media_sha256: str
    source_class: str
    label: str


@dataclass(frozen=True)
class GovernanceRow:
    media_sha256: str
    source_lineage_id: str
    dataset_split: str
    derived_from_sha256: str | None
    transformations: tuple[str, ...] | None
    license: str | None
    permission_status: str | None
    redistributable: bool | None
    private: bool | None
    stratum_primary: str | None
    source: str | None
    acquisition_type: str | None
    benchmark_family: str | None


@dataclass(frozen=True)
class MediaRow:
    media_file_id: str
    analysis_id: str
    analysis_created_at: str
    media_sha256: str
    size_bytes: int
    storage_key: str


@dataclass(frozen=True)
class SignalRow:
    signal_id: str
    analysis_id: str
    created_at: str
    media_sha256: str
    provider: str
    signal_type: str
    provider_version: str | None
    status: str
    score: float | None
    metadata: dict | None

    @property
    def identity(self) -> tuple[str, str | None, str]:
        return (self.provider, self.provider_version, self.signal_type)


# ---------------------------------------------------------------------------------------------
# The corpus. Pure: no database, no clock, no filesystem.
# ---------------------------------------------------------------------------------------------


def representative_media(media: list[MediaRow]) -> tuple[dict[str, MediaRow], int]:
    """The earliest upload of each set of bytes, by `(analysis created_at, media row id)`."""
    by_sha: dict[str, list[MediaRow]] = defaultdict(list)
    for row in media:
        by_sha[row.media_sha256].append(row)
    chosen = {}
    duplicates = 0
    for sha, rows in by_sha.items():
        ordered = sorted(rows, key=lambda r: (r.analysis_created_at, r.media_file_id))
        chosen[sha] = ordered[0]
        duplicates += len(ordered) - 1
    return chosen, duplicates


def _metadata_complete(governance: GovernanceRow) -> bool:
    if any(getattr(governance, name) is None for name in CORPUS_METADATA_FIELDS):
        return False
    # A derivative with no recorded steps has no honest `derivation` to write.
    if governance.derived_from_sha256 is not None and not governance.transformations:
        return False
    return True


def _derivation(governance: GovernanceRow) -> str:
    if governance.transformations:
        return "+".join(governance.transformations)
    return "none"


def _chain_root(sha: str, governance: dict[str, GovernanceRow]) -> str:
    seen = {sha}
    current = sha
    while True:
        record = governance.get(current)
        parent = record.derived_from_sha256 if record else None
        if parent is None:
            return current
        if parent in seen:
            raise ExportError(f"derivation cycle through {parent}")
        seen.add(parent)
        current = parent


def storage_path_usable(media: MediaRow) -> bool:
    """Only the content-addressed key names these bytes; anything else is not exported."""
    return media.storage_key == f"{STORAGE_PREFIX}{media.media_sha256}"


def classify(
    truth: GroundTruthRow,
    governance: GovernanceRow | None,
    media: MediaRow | None,
) -> str | None:
    """The exclusion reason for one Ground Truth record, or None when it is exported."""
    if truth.source_class not in SOURCE_CLASSES:
        raise ExportError(f"unknown source_class {truth.source_class!r} for {truth.media_sha256}")
    if truth.label not in GROUND_TRUTH_LABELS:
        raise ExportError(f"unknown label {truth.label!r} for {truth.media_sha256}")
    if truth.source_class == "UNKNOWN":
        return GROUND_TRUTH_INELIGIBLE
    if truth.label == "UNKNOWN":
        return UNKNOWN_LABEL
    if truth.label not in LABEL_MAP:
        return UNSUPPORTED_LABEL
    if governance is None or not _metadata_complete(governance):
        return INSUFFICIENT_CORPUS_METADATA
    if governance.dataset_split not in OPERATIONAL_SPLITS:
        raise ExportError(
            f"unknown dataset_split {governance.dataset_split!r} for {truth.media_sha256}"
        )
    if media is None:
        return NO_MEDIA_FILE
    if not storage_path_usable(media):
        return UNUSABLE_STORAGE_PATH
    return None


def corpus_item(
    truth: GroundTruthRow,
    governance: GovernanceRow,
    media: MediaRow,
    all_governance: dict[str, GovernanceRow],
) -> CorpusItem:
    return CorpusItem(
        clip_id=truth.media_sha256,
        path=media.storage_key,
        label=LABEL_MAP[truth.label],
        family=governance.benchmark_family,
        stratum_primary=governance.stratum_primary,
        source_lineage_id=governance.source_lineage_id,
        base_media_id=_chain_root(truth.media_sha256, all_governance),
        derivative_of=governance.derived_from_sha256,
        derivation=_derivation(governance),
        source=governance.source,
        acquisition_type=governance.acquisition_type,
        label_provenance=truth.source_class,
        license=governance.license,
        permission_status=governance.permission_status,
        redistributable=governance.redistributable,
        private=governance.private,
        sha256=truth.media_sha256,
        bytes=media.size_bytes,
        split=governance.dataset_split,
    )


def build_corpus(
    truths: list[GroundTruthRow],
    governance: list[GovernanceRow],
    media: list[MediaRow],
    split_filter: str | None = None,
) -> dict:
    """Decide every Ground Truth record, build the exported items, and check leakage."""
    if split_filter is not None and split_filter not in OPERATIONAL_SPLITS:
        raise ExportError(f"unknown split filter {split_filter!r}")
    by_sha = {row.media_sha256: row for row in governance}
    chosen, media_duplicates = representative_media(media)

    exclusions: Counter = Counter()
    excluded: dict[str, str] = {}
    outside_filter: list[str] = []
    items: list[CorpusItem] = []
    for truth in sorted(truths, key=lambda t: t.media_sha256):
        record = by_sha.get(truth.media_sha256)
        reason = classify(truth, record, chosen.get(truth.media_sha256))
        if reason is not None:
            exclusions[reason] += 1
            excluded[truth.media_sha256] = reason
            continue
        if split_filter is not None and record.dataset_split != split_filter:
            outside_filter.append(truth.media_sha256)
            continue
        items.append(corpus_item(truth, record, chosen[truth.media_sha256], by_sha))

    exported = {item.clip_id for item in items}
    # Governed bytes that exist but were not exported, for whatever reason — including having no
    # Ground Truth at all. A derivative pointing at one of these is reported, not failed.
    known_unexported = frozenset(sha for sha in by_sha if sha not in exported)
    leakage = leakage_findings(items, missing_exported_parents=known_unexported)

    return {
        "items": items,
        "leakage": leakage,
        "exclusion_counts": {
            reason: exclusions[reason]
            for reason in EXCLUSION_REASONS
            if reason not in SIGNAL_EXCLUSION_REASONS
        },
        "excluded": excluded,
        "outside_split_filter": outside_filter,
        "media_file_duplicates": media_duplicates,
        "governed_without_ground_truth": len(
            set(by_sha) - {truth.media_sha256 for truth in truths}
        ),
    }


# ---------------------------------------------------------------------------------------------
# The signals. Pure.
# ---------------------------------------------------------------------------------------------


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def signal_exclusion(signal: SignalRow) -> str | None:
    """Why a (deduplicated) signal cannot be exported losslessly, or None."""
    slot = REPLAY_SLOTS.get((signal.provider, signal.signal_type))
    if slot is None or signal.status not in STATUS_MAP:
        return UNSUPPORTED_SIGNAL_SCHEMA
    if signal.provider_version is None:
        return MISSING_PROVIDER_VERSION
    if signal.status == "SUCCESS":
        count = (signal.metadata or {}).get(slot.metadata_count_key)
        if signal.score is None or not _is_count(count):
            return INVALID_SUCCESS_PAYLOAD
    return None


def build_results(signals: list[SignalRow], exported_shas: set[str]) -> dict:
    """Deduplicate, filter and group the signals of exported media into one run per identity."""
    relevant = [s for s in signals if s.media_sha256 in exported_shas]
    by_key: dict[tuple, list[SignalRow]] = defaultdict(list)
    for signal in relevant:
        by_key[(signal.media_sha256, *signal.identity)].append(signal)

    duplicates = 0
    exclusions: Counter = Counter()
    excluded_by_schema: Counter = Counter()
    groups: dict[tuple, list[SignalRow]] = defaultdict(list)
    for key in sorted(by_key, key=lambda k: tuple("" if part is None else part for part in k)):
        members = sorted(by_key[key], key=lambda s: (s.created_at, s.signal_id), reverse=True)
        duplicates += len(members) - 1
        chosen = members[0]
        reason = signal_exclusion(chosen)
        if reason is not None:
            exclusions[reason] += 1
            excluded_by_schema[f"{reason}:{chosen.provider}/{chosen.signal_type}"] += 1
            continue
        groups[chosen.identity].append(chosen)

    runs = {identity: results_payload(identity, members) for identity, members in groups.items()}
    return {
        "runs": runs,
        "signal_duplicates": duplicates,
        "exclusion_counts": {reason: exclusions[reason] for reason in SIGNAL_EXCLUSION_REASONS},
        "excluded_by_provider": dict(sorted(excluded_by_schema.items())),
        "signals_for_unexported_media": len(signals) - len(relevant),
    }


def results_payload(identity: tuple[str, str, str], members: list[SignalRow]) -> dict:
    provider, provider_version, signal_type = identity
    if not isinstance(provider_version, str):
        raise ExportError(f"no exact detector identity for {identity}")
    slot = REPLAY_SLOTS[(provider, signal_type)]
    counts = {}
    clips = []
    for signal in sorted(members, key=lambda s: s.media_sha256):
        metadata = signal.metadata or {}
        if slot.metadata_count_key in metadata:
            counts[signal.media_sha256] = metadata[slot.metadata_count_key]
        clips.append(
            {
                "clip_id": signal.media_sha256,
                "status": STATUS_MAP[signal.status],
                "stored_status": signal.status,
                "score": signal.score,
                "signal_id": signal.signal_id,
                "analysis_id": signal.analysis_id,
                "signal_created_at": signal.created_at,
            }
        )
    return {
        "schema_version": RESULTS_SCHEMA,
        "provider": provider,
        "provider_version": provider_version,
        "signal_type": signal_type,
        "evidence_counts": {slot.metadata_count_key: counts},
        "run": {"source": "analysis_signals (R12-T6 export)"},
        "clips": clips,
    }


_UNSAFE = re.compile(r"[^A-Za-z0-9.-]+")


def _safe(part: str, limit: int = 48) -> str:
    text = _UNSAFE.sub("_", part).strip("._-")
    return (text or "empty")[:limit]


def results_filename(identity: tuple[str, str, str]) -> str:
    """A path-safe name that is still unique: the readable parts, then a digest of the exact ones.

    Sanitizing is lossy (`a/b` and `a_b` read the same), so the digest of the untouched identity
    is what keeps two identities from ever sharing a file.
    """
    provider, provider_version, signal_type = identity
    digest = hashlib.sha256(json.dumps(list(identity)).encode("utf-8")).hexdigest()[:12]
    return (
        f"results_{_safe(provider)}_{_safe(signal_type)}_{_safe(provider_version)}_{digest}.json"
    )


# ---------------------------------------------------------------------------------------------
# Writing. Everything is decided before the first byte goes to disk.
# ---------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_export(
    out_dir: Path,
    corpus: dict,
    results: dict,
    *,
    split_filter: str | None,
    exporter_identity: dict,
    source_db: dict,
    query_facts: dict | None = None,
) -> dict:
    """Write corpus, manifest, one results file per identity and the metadata; return the latter."""
    leakage = corpus["leakage"]
    if leakage["leaked"]:
        raise ExportError(
            "leakage across splits; nothing written: "
            + json.dumps({k: v for k, v in leakage.items() if v and k != "clips"}, default=str)
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ExportError(f"{out_dir} is not empty; an export is never written over another")
    names = {identity: results_filename(identity) for identity in results["runs"]}
    if len(set(names.values())) != len(names):
        raise ExportError("two signal identities map to one results filename")

    out_dir.mkdir(parents=True, exist_ok=True)
    items = corpus["items"]
    write_corpus(items, out_dir / "corpus.json", extra={"export_task": TASK})
    write_manifest(items, out_dir / "manifest.csv")
    artifacts = {}
    for identity, payload in sorted(
        results["runs"].items(), key=lambda kv: names[kv[0]]
    ):
        path = out_dir / names[identity]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        artifacts[names[identity]] = {
            "provider": identity[0],
            "provider_version": identity[1],
            "signal_type": identity[2],
            "clips": len(payload["clips"]),
        }

    exclusion_counts = {**corpus["exclusion_counts"], **results["exclusion_counts"]}
    metadata = {
        "schema": METADATA_SCHEMA,
        "task": TASK,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "exporter_identity": exporter_identity,
        "source_db": source_db,
        "exported_sha_count": len(items),
        "exclusion_counts": {reason: exclusion_counts[reason] for reason in EXCLUSION_REASONS},
        "signal_exclusions_by_provider": results["excluded_by_provider"],
        "representative_selection": {
            "media_files": MEDIA_REPRESENTATIVE_POLICY,
            "signals": SIGNAL_REPRESENTATIVE_POLICY,
        },
        "duplicate_counts": {
            "media_files": corpus["media_file_duplicates"],
            "signals": results["signal_duplicates"],
        },
        "split_filter": {
            "split": split_filter,
            "outside_filter": len(corpus["outside_split_filter"]),
        },
        "leakage": leakage,
        "results_artifacts": artifacts,
        "path_semantics": (
            f"CorpusItem.path is the MinIO object key in bucket {STORAGE_BUCKET!r}, exported "
            "verbatim; it is not relative to this directory"
        ),
        "counts_note": (
            "governed_without_ground_truth and signals_for_unexported_media are not exclusions: "
            "they were never candidates"
        ),
        "governed_without_ground_truth": corpus["governed_without_ground_truth"],
        "signals_for_unexported_media": results["signals_for_unexported_media"],
        **({"query": query_facts} if query_facts is not None else {}),
        "artifact_hashes": {
            name: _sha256_file(out_dir / name)
            for name in ["corpus.json", "manifest.csv", *sorted(artifacts)]
        },
    }
    (out_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


# ---------------------------------------------------------------------------------------------
# Database and provenance.
# ---------------------------------------------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def load_rows(session) -> tuple[list, list, list, list, dict]:
    """Read Ground Truth, governance, media and signals. Call inside a READ ONLY transaction."""
    from sqlalchemy import func, select

    from app.db.models import (
        Analysis,
        AnalysisSignal,
        GroundTruth,
        LineageSplit,
        MediaFile,
        MediaGovernance,
    )

    truths = [
        GroundTruthRow(r.media_sha256, r.source_class, r.label)
        for r in session.execute(
            select(GroundTruth.media_sha256, GroundTruth.source_class, GroundTruth.label)
        )
    ]
    governance = [
        GovernanceRow(
            media_sha256=g.media_sha256,
            source_lineage_id=g.source_lineage_id,
            dataset_split=split,
            derived_from_sha256=g.derived_from_sha256,
            transformations=None if g.transformations is None else tuple(g.transformations),
            **{name: getattr(g, name) for name in CORPUS_METADATA_FIELDS},
        )
        for g, split in session.execute(
            select(MediaGovernance, LineageSplit.dataset_split).join(
                LineageSplit,
                LineageSplit.source_lineage_id == MediaGovernance.source_lineage_id,
            )
        )
    ]
    shas = {t.media_sha256 for t in truths}
    media = [
        MediaRow(
            media_file_id=str(r.id),
            analysis_id=str(r.analysis_id),
            analysis_created_at=_iso(r.created_at),
            media_sha256=r.original_sha256,
            size_bytes=r.size_bytes,
            storage_key=r.original_storage_key,
        )
        for r in session.execute(
            select(
                MediaFile.id,
                MediaFile.analysis_id,
                MediaFile.original_sha256,
                MediaFile.size_bytes,
                MediaFile.original_storage_key,
                Analysis.created_at,
            )
            .join(Analysis, Analysis.id == MediaFile.analysis_id)
            .where(MediaFile.original_sha256.in_(shas))
        )
    ]

    # An analysis whose media rows name two different hashes has no single byte identity to
    # attach its signals to (as in `operational_metrics.py`); its signals are set aside.
    ambiguous = {
        analysis_id
        for (analysis_id,) in session.execute(
            select(MediaFile.analysis_id)
            .group_by(MediaFile.analysis_id)
            .having(func.count(func.distinct(MediaFile.original_sha256)) > 1)
        )
    }
    seen: set = set()
    signals = []
    for r in session.execute(
        select(AnalysisSignal, MediaFile.original_sha256)
        .join(MediaFile, MediaFile.analysis_id == AnalysisSignal.analysis_id)
        .where(MediaFile.original_sha256.in_(shas))
    ):
        signal, sha = r
        if signal.analysis_id in ambiguous or signal.id in seen:
            continue
        seen.add(signal.id)
        signals.append(
            SignalRow(
                signal_id=str(signal.id),
                analysis_id=str(signal.analysis_id),
                created_at=_iso(signal.created_at),
                media_sha256=sha,
                provider=signal.provider,
                signal_type=signal.signal_type,
                provider_version=signal.provider_version,
                status=signal.status,
                score=signal.score,
                metadata=signal.signal_metadata,
            )
        )

    facts = {
        "ground_truth_records": len(truths),
        "media_governance_records": len(governance),
        "analyses_with_ambiguous_media_identity": len(ambiguous),
    }
    return truths, governance, media, signals, facts


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def exporter_identity() -> dict:
    script = Path(__file__).resolve()
    dirty = _git("status", "--porcelain", "--", str(script.relative_to(REPO_ROOT)))
    return {
        "script": str(script.relative_to(REPO_ROOT)),
        "script_sha256": _sha256_file(script),
        "git_commit": _git("rev-parse", "HEAD"),
        "script_uncommitted": None if dirty is None else bool(dirty),
        "python": platform.python_version(),
    }


def export(session, out_dir: Path, split_filter: str | None, source_db: dict) -> dict:
    truths, governance, media, signals, facts = load_rows(session)
    corpus = build_corpus(truths, governance, media, split_filter)
    results = build_results(signals, {item.clip_id for item in corpus["items"]})
    return write_export(
        out_dir,
        corpus,
        results,
        split_filter=split_filter,
        exporter_identity=exporter_identity(),
        source_db=source_db,
        query_facts=facts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=OPERATIONAL_SPLITS)
    arguments = parser.parse_args(argv)

    from sqlalchemy import text

    from app.db.session import SessionLocal, database_url

    url = database_url()
    source_db = {
        "driver": url.drivername,
        "host": url.host,
        "port": url.port,
        "database": url.database,
    }
    with SessionLocal() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        try:
            metadata = export(session, arguments.out_dir, arguments.split, source_db)
        except ExportError as error:
            print(f"export refused: {error}", file=sys.stderr)
            return 1
        finally:
            session.rollback()

    print(
        f"exported {metadata['exported_sha_count']} media; "
        f"exclusions {metadata['exclusion_counts']}; "
        f"{len(metadata['results_artifacts'])} results artifacts"
    )
    print(f"wrote {arguments.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
