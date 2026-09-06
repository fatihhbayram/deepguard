"""The R7-T5 corpus: what a clip carries with it, and how clips are grouped and split.

Pure data and pure functions. Nothing here downloads, decodes or scores anything — that is
`build_corpus.py`'s and the benchmark harness's work — so the record schema, the lineage
grouping and the split rule can be read, tested and reasoned about without a network or a GPU.

**The unit of observation is a lineage, not a file.** A false-positive rate counted over files
counts the same underlying recording once per re-encode, and a corpus built from social
transcodes and simulated degradations is mostly re-encodes. Every record therefore carries a
`source_lineage_id` naming the underlying media it came from, and every rate in this task is
computed over distinct lineages. Derivatives inflate the stress applied to a detector; they do
not inflate the sample size.

**Splitting is by lineage and only by lineage.** `split_by_lineage` assigns whole lineages, so
a clip and its transcodes, crops and recompressions always land in the same split. That is the
only construction under which the evaluation split is a hold-out at all: a threshold derived on
one 1080p original and tested on its own 540p transcode has been tested on its training data
wearing a different codec.

**Provenance travels with every clip.** Source, licence, permission and redistribution state
are fields of the record rather than a README paragraph, because the corpus manifest is the
thing that gets copied around and a licence that lives somewhere else does not. `private`
records are the ones that must never leave this machine: `redistributable` is false, their media
lives outside the repository, and the reporting layer is expected to identify them by
`source_lineage_id` and hash alone.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The schema the artifacts are written under. Bumped when a field's meaning changes, so a
# reader of an old corpus.json is never silently reinterpreted under new rules.
CORPUS_SCHEMA = "r7-t5-corpus-1"

# The two splits, named once. `calibration` is where distributions may be observed and
# candidate operating points derived; `evaluation` is touched by measurement only.
SPLIT_CALIBRATION = "calibration"
SPLIT_EVALUATION = "evaluation"

# The benchmark harness's ground-truth vocabulary, unchanged (scripts/benchmark/README.md).
# `real` is the negative class; everything else is a manipulation family.
LABEL_REAL = "real"
LABEL_FACE_SWAP = "face_swap"
LABEL_SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class CorpusItem:
    """One clip, and everything that has to be true about it for a number to mean something.

    `clip_id` names the file; `source_lineage_id` names the recording behind it. Those are
    different identities and conflating them is the mistake this whole module exists to
    prevent — `mavos:en:1017` is one lineage whether it appears once or in nine degradations.

    `base_media_id` is the clip_id of the lineage's own base clip: for a base clip it is its
    own id, for a derivative it is the clip it was derived from. It is redundant with
    `derivative_of` for one-level derivations and stays anyway, because a two-level derivation
    (a transcode of a crop) would otherwise have no field naming the original.

    `strata` is a list rather than a single label because real media is several things at once
    — a phone clip can be low-light *and* shaky *and* small-faced — and a stratum table that
    forces one label per clip either invents a precedence order or throws information away.
    `stratum_primary` is what the clip was acquired or constructed *for*, and is what the
    per-stratum breakdown is keyed on.

    `redistributable` is the field that decides whether a clip's bytes may be copied anywhere
    at all, and `private` whether it may be named. They are separate: a permissively licensed
    research clip is redistributable and nameable, and a consented personal recording is
    neither.
    """

    clip_id: str
    path: str
    label: str
    family: str
    stratum_primary: str
    source_lineage_id: str
    base_media_id: str
    derivative_of: str | None
    derivation: str
    source: str
    acquisition_type: str
    label_provenance: str
    license: str
    permission_status: str
    redistributable: bool
    private: bool
    sha256: str
    bytes: int
    strata: list[str] = field(default_factory=list)
    split: str | None = None

    @property
    def is_manipulated(self) -> bool:
        """The benchmark's orientation, restated once: anything but `real` is positive."""
        return self.label != LABEL_REAL

    @property
    def is_base(self) -> bool:
        """Whether this clip is the lineage's own base rather than a derivative of it."""
        return self.derivative_of is None


class CorpusError(Exception):
    """The corpus is not in a state any measurement may be taken over."""


def lineages(items: list[CorpusItem]) -> dict[str, list[CorpusItem]]:
    """Group clips by the recording behind them, in first-seen order."""
    grouped: dict[str, list[CorpusItem]] = {}
    for item in items:
        grouped.setdefault(item.source_lineage_id, []).append(item)
    return grouped


def split_by_lineage(
    items: list[CorpusItem],
    assignment: dict[str, str],
) -> list[CorpusItem]:
    """Stamp every clip with its lineage's split, refusing anything the assignment misses.

    Takes an explicit lineage → split mapping rather than deriving one from a hash of the id,
    because the split here is not arbitrary: FaceForensics++ is deliberately held out of the
    evaluation side (it is the corpus LipForensics was trained and previously calibrated on),
    and the generated families are divided so that the evaluation side carries generator
    families the calibration side never saw. A hash cannot express either intent, and a split
    that cannot express intent cannot measure *cross-dataset* robustness.

    Raises rather than defaulting an unmapped lineage into a split. A clip that quietly lands
    in the evaluation set because nobody named it is exactly the leak this task must not have.
    """
    stamped = []
    for item in items:
        split = assignment.get(item.source_lineage_id)
        if split is None:
            raise CorpusError(
                f"lineage {item.source_lineage_id!r} (clip {item.clip_id!r}) has no split "
                "assignment; every lineage must be assigned explicitly"
            )
        if split not in (SPLIT_CALIBRATION, SPLIT_EVALUATION):
            raise CorpusError(
                f"lineage {item.source_lineage_id!r} assigned to unknown split {split!r}"
            )
        stamped.append(CorpusItem(**{**asdict(item), "split": split}))
    return stamped


def leakage_findings(items: list[CorpusItem]) -> dict:
    """Everything that would make the two splits not independent, as data rather than prose.

    Four questions are asked, and all four are asked every time so the artifact records the
    ones that came back empty as well:

    - does any `source_lineage_id` appear in both splits? This is the check the task requires,
      and it is the one that catches a mis-assigned derivative;
    - does any clip's `sha256` appear in both splits? Identical bytes under two clip ids are
      one measurement counted twice no matter what their lineage fields claim, and this catches
      a lineage id that was assigned wrongly at build time;
    - does any clip's `sha256` appear twice *anywhere*? A duplicate inside one split does not
      leak across the boundary but does inflate that split's n;
    - is any derivative separated from its base, or pointing at a base that does not exist?

    Returns a report rather than raising, so the caller can write the artifact before deciding
    what to do about it. `leaked` is the single boolean a gate should read.
    """
    by_split: dict[str, set[str]] = {SPLIT_CALIBRATION: set(), SPLIT_EVALUATION: set()}
    digests: dict[str, dict[str, list[str]]] = {}
    by_id = {item.clip_id: item for item in items}
    unassigned = [item.clip_id for item in items if item.split is None]

    for item in items:
        if item.split in by_split:
            by_split[item.split].add(item.source_lineage_id)
        digests.setdefault(item.sha256, {}).setdefault(item.split or "unassigned", []).append(
            item.clip_id
        )

    shared_lineages = sorted(by_split[SPLIT_CALIBRATION] & by_split[SPLIT_EVALUATION])

    cross_split_digests = sorted(
        digest for digest, splits in digests.items() if len(splits) > 1
    )
    repeated_digests = sorted(
        digest
        for digest, splits in digests.items()
        if sum(len(ids) for ids in splits.values()) > 1
    )

    orphans = []
    straddling = []
    for item in items:
        if item.derivative_of is None:
            continue
        base = by_id.get(item.derivative_of)
        if base is None:
            orphans.append(item.clip_id)
            continue
        if base.split != item.split or base.source_lineage_id != item.source_lineage_id:
            straddling.append(item.clip_id)

    return {
        "clips": len(items),
        "unassigned_clips": unassigned,
        "lineages_calibration": len(by_split[SPLIT_CALIBRATION]),
        "lineages_evaluation": len(by_split[SPLIT_EVALUATION]),
        "shared_lineage_ids": shared_lineages,
        "cross_split_sha256": [
            {"sha256": digest, "clips": digests[digest]} for digest in cross_split_digests
        ],
        "repeated_sha256": [
            {"sha256": digest, "clips": digests[digest]} for digest in repeated_digests
        ],
        "derivatives_without_base": orphans,
        "derivatives_split_from_base": straddling,
        "leaked": bool(
            shared_lineages
            or cross_split_digests
            or orphans
            or straddling
            or unassigned
        ),
    }


def corpus_digest(items: list[CorpusItem]) -> str:
    """One digest over exactly which clips this corpus holds and which bytes they are."""
    joined = "\n".join(
        f"{item.clip_id}:{item.sha256}:{item.split}"
        for item in sorted(items, key=lambda entry: entry.clip_id)
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


MANIFEST_COLUMNS = (
    "clip_id",
    "path",
    "label",
    "audio_path",
    "family",
    "stratum",
    "split",
    "source_lineage_id",
)


def write_manifest(items: list[CorpusItem], path: Path) -> str:
    """Write the benchmark-compatible manifest and return its SHA-256.

    The harness needs `clip_id`, `path` and `label` and ignores the rest; the rest is here so
    that a `results.json` can be joined back to a stratum and a split without re-reading
    `corpus.json`. Paths resolve against the manifest's own directory, which is what lets the
    corpus live outside the repository.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for item in sorted(items, key=lambda entry: entry.clip_id):
        writer.writerow(
            [
                item.clip_id,
                item.path,
                item.label,
                "",
                item.family,
                item.stratum_primary,
                item.split or "",
                item.source_lineage_id,
            ]
        )
    raw = buffer.getvalue().encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def write_corpus(items: list[CorpusItem], path: Path, extra: dict | None = None) -> None:
    """Write the full provenance record for the corpus."""
    payload = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_digest": corpus_digest(items),
        "counts": summarise(items),
        **(extra or {}),
        "items": [asdict(item) for item in sorted(items, key=lambda e: e.clip_id)],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_corpus(path: Path) -> list[CorpusItem]:
    """Read a corpus.json back into records, refusing a schema this code does not know."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CORPUS_SCHEMA:
        raise CorpusError(
            f"{path} is schema {payload.get('schema_version')!r}, not {CORPUS_SCHEMA!r}"
        )
    return [CorpusItem(**entry) for entry in payload["items"]]


def summarise(items: list[CorpusItem]) -> dict:
    """Counts a reader needs before any metric: clips, lineages, and both broken down."""

    def tally(subset: list[CorpusItem]) -> dict:
        return {
            "clips": len(subset),
            "base_clips": sum(1 for item in subset if item.is_base),
            "lineages": len({item.source_lineage_id for item in subset}),
            "by_label": _counted(item.label for item in subset),
            "by_stratum": _counted(item.stratum_primary for item in subset),
            "by_family": _counted(item.family for item in subset),
        }

    return {
        "all": tally(items),
        SPLIT_CALIBRATION: tally([i for i in items if i.split == SPLIT_CALIBRATION]),
        SPLIT_EVALUATION: tally([i for i in items if i.split == SPLIT_EVALUATION]),
        "private_clips": sum(1 for item in items if item.private),
        "non_redistributable_clips": sum(1 for item in items if not item.redistributable),
    }


def _counted(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
