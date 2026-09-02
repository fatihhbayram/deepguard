"""Ground-truth dataset ingestion for the offline benchmark.

A benchmark is only as trustworthy as the labels behind it, so this module is
deliberately strict: it validates the whole manifest before a single model call is
made, and reports *every* problem it found rather than the first. A run that dies on
clip 300 of 400 because of a typo has wasted the expensive part of the work.

**Manifest format — CSV, one row per clip.**

```csv
clip_id,path,label,audio_path
ls_1272,clips/real_ls_1272.mp4,real,
xtts_01,clips/synth_xtts_01.mp4,synthetic,clips/synth_xtts_01.wav
```

| column | required | meaning |
|---|---|---|
| `path` | yes | media file, absolute or relative *to the manifest's own directory* |
| `label` | yes | one of the ground-truth labels below |
| `clip_id` | no | stable identifier; defaults to `path` as written |
| `audio_path` | no | separate audio track, for the R2 audio anti-spoof evaluation track |

Unknown columns are ignored, so a corpus may carry provenance notes of its own.

CSV only, on purpose. JSON manifests were considered and left out: every corpus in
reach is a spreadsheet export, one reader is one thing to keep correct, and a second
format can be added the day a dataset actually arrives in it (AGENTS.md, YAGNI).

**Paths resolve against the manifest, not the working directory.** A corpus directory
is therefore relocatable and a run reproduces from anywhere on the machine.

**The manifest is read exactly once.** `load_manifest` returns a `Dataset` carrying the
clips *and* the SHA-256 of the bytes they were parsed from, because a fingerprint taken
in a second read describes whatever the file happened to contain at that moment rather
than the ground truth the run actually measured.
"""

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

# The ground-truth vocabulary, and the single place where a label becomes a binary
# target. `real` is the only genuine label; everything else is a manipulation family.
# Families are kept distinct rather than collapsed to "fake" at ingestion time because
# the per-family breakdown is the interesting half of a detector evaluation — a model
# that catches synthesis and misses face swaps is a very different proposition from one
# that is uniformly mediocre, and that difference disappears if the labels are merged
# on the way in.
GENUINE_LABEL = "real"
MANIPULATED_LABELS = frozenset({"synthetic", "face_swap", "audio_spoof"})
KNOWN_LABELS = frozenset({GENUINE_LABEL}) | MANIPULATED_LABELS


class ManifestError(Exception):
    """The manifest could not be read, or did not describe a usable dataset."""


@dataclass(frozen=True)
class Clip:
    """One labelled item of media, with its ground truth already resolved."""

    clip_id: str
    path: Path
    label: str
    audio_path: Path | None = None

    @property
    def is_manipulated(self) -> bool:
        """The binary ground truth, with *manipulated* as the positive class."""
        return self.label != GENUINE_LABEL


@dataclass(frozen=True)
class Dataset:
    """Clips, together with the identity of the manifest they were parsed from.

    The two travel as one value on purpose. `manifest_sha256` is the digest of the
    exact bytes `clips` came from, captured in the same read, so a caller cannot
    accidentally pair records with the fingerprint of a manifest that has since been
    edited. That pairing is the whole traceability claim a `results.json` makes.
    """

    clips: list[Clip] = field(default_factory=list)
    manifest_path: Path = Path()
    manifest_sha256: str = ""


def load_manifest(manifest_path: Path) -> Dataset:
    """Read and validate a manifest into a `Dataset`, ordered by `clip_id`.

    Raises `ManifestError` — with every problem listed — if the manifest is missing,
    malformed, empty, carries an unknown label, repeats a `clip_id`, or points at media
    that is not on disk. Ordering is fixed here so that two runs over the same corpus
    produce per-clip records in the same sequence and the artifacts diff cleanly.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")

    base = manifest_path.parent

    # One read, and both the records and their fingerprint derive from it. Hashing the
    # file separately -- before the run or, worse, after it -- allows a manifest edited
    # mid-run to produce clips parsed from one ground truth stamped with the digest of
    # another, which is precisely the traceability the digest is there to provide.
    raw = manifest_path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    # `utf-8-sig` strips a leading byte-order mark, and is plain UTF-8 when there is
    # none. Spreadsheet exports are the expected source of these manifests and Excel's
    # "CSV UTF-8" writes a BOM, which would otherwise glue itself to the *first* column
    # name: a BOM'd `path` column reads as missing, and a BOM'd `clip_id` column reads
    # as absent, silently demoting every clip_id to its path and changing both the
    # record ordering and any clip_id-derived score. Only the parse tolerates the BOM;
    # the fingerprint above hashes the raw bytes, so two encodings of the same rows
    # remain distinguishable in the artifact.
    handle = io.StringIO(raw.decode("utf-8-sig"), newline="")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ManifestError(f"manifest is empty: {manifest_path}")
    missing = {"path", "label"} - set(reader.fieldnames)
    if missing:
        raise ManifestError(
            f"manifest {manifest_path} is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )
    rows = list(reader)

    problems: list[str] = []
    clips: list[Clip] = []
    seen_ids: set[str] = set()

    # Row 1 is the header, so the first data row is line 2 — quoting the file's own
    # line numbers is what makes a validation error fixable without counting rows.
    for line_number, row in enumerate(rows, start=2):
        raw_path = (row.get("path") or "").strip()
        label = (row.get("label") or "").strip()
        clip_id = (row.get("clip_id") or "").strip() or raw_path
        raw_audio = (row.get("audio_path") or "").strip()

        if not raw_path:
            problems.append(f"line {line_number}: empty `path`")
            continue
        if label not in KNOWN_LABELS:
            problems.append(
                f"line {line_number}: unknown label {label!r} "
                f"(expected one of {', '.join(sorted(KNOWN_LABELS))})"
            )
            continue
        if clip_id in seen_ids:
            problems.append(f"line {line_number}: duplicate clip_id {clip_id!r}")
            continue
        seen_ids.add(clip_id)

        media = _resolve(base, raw_path)
        if not media.is_file():
            problems.append(f"line {line_number}: media file not found: {media}")
            continue

        audio: Path | None = None
        if raw_audio:
            audio = _resolve(base, raw_audio)
            if not audio.is_file():
                problems.append(f"line {line_number}: audio file not found: {audio}")
                continue

        clips.append(Clip(clip_id=clip_id, path=media, label=label, audio_path=audio))

    if problems:
        raise ManifestError(
            f"{manifest_path} has {len(problems)} problem(s):\n  - "
            + "\n  - ".join(problems)
        )
    if not clips:
        raise ManifestError(f"manifest describes no clips: {manifest_path}")

    return Dataset(
        clips=sorted(clips, key=lambda clip: clip.clip_id),
        manifest_path=manifest_path,
        manifest_sha256=fingerprint,
    )


def _resolve(base: Path, raw: Path | str) -> Path:
    """Resolve a manifest path entry against the manifest's directory."""
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def label_counts(clips: list[Clip]) -> dict[str, int]:
    """How many clips carry each ground-truth label, for the run's composition record."""
    counts: dict[str, int] = {}
    for clip in clips:
        counts[clip.label] = counts.get(clip.label, 0) + 1
    return dict(sorted(counts.items()))

