"""Assemble the R7-T5 cross-dataset corpus, reproducibly, with its provenance and its splits.

    PYTHONPATH=scripts python3 scripts/eval/build_corpus.py \
        --output-dir ../deepguard-corpus/r7t5 \
        --private-dir ../deepguard-corpus-r7t5-private

R5-T3 measured LipForensics' operating point against twenty genuine clips from one benchmark
family. This corpus exists to ask whether that operating point survives contact with media the
benchmark families do not contain: five languages of in-the-wild footage, the platform
transcodes real uploads arrive under, low light, handheld motion, decimated frame rates, and two
real-world lineages that already produced a HIGH in production.

**Reproducibility.** Every remote source names a repository *and the commit read*, member
selection inside an archive is deterministic (sorted, evenly spaced, `.mp4` only, size-capped),
and derivatives are produced by pinned filter graphs and encoder settings. Two runs of this
script against the same revisions produce the same corpus digest, modulo the ffmpeg build, which
is recorded.

**Splits are decided here and by name.** `assign_splits` maps every lineage to a split
explicitly and `corpus.split_by_lineage` refuses anything it missed. The evaluation side
deliberately carries source families the calibration side does not — FaceForensics++ is held
entirely on the calibration side because it is the corpus LipForensics was trained on and
previously calibrated against, and the generated and swapped families are divided so the
evaluation side is answering about generators it was not tuned on. A hash-based split would
have distributed both families evenly across the boundary and measured nothing cross-dataset.

**Private media never enters the repository or the corpus directory.** The two real-world
regression lineages are referenced where they already live, under `--private-dir`, and their
records carry `private: true`, `redistributable: false` and no original filename. What travels
into any artifact is a lineage id, a digest and a measured score.

Standard library only, plus ffmpeg on PATH for the derivatives. Nothing under `apps/` is read,
imported or written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.fetch_corpus import BUFFER_BYTES, RESOLVE_URL, _RangeFile, select_members
from eval import derive
from eval.corpus import (
    LABEL_FACE_SWAP,
    LABEL_REAL,
    LABEL_SYNTHETIC,
    SPLIT_CALIBRATION,
    SPLIT_EVALUATION,
    CorpusError,
    CorpusItem,
    split_by_lineage,
    write_corpus,
    write_manifest,
)

# Members above this are skipped, as in R4-T1: one 300 MB clip would dominate the download and
# every detector's wall clock while contributing exactly one lineage.
MAX_MEMBER_BYTES = 30 * 1024 * 1024


@dataclass(frozen=True)
class RemoteSource:
    """One mirrored archive and how much of it this corpus takes.

    `lineage_namespace` is the field R4-T1's equivalent did not need. Two clips are the same
    lineage when they came from the same underlying recording, and within one MAVOS-DD archive
    the member number is the only handle on that recording there is — so the namespace plus the
    member's number is the lineage id, and clips from unrelated archives can never collide into
    one by accident.
    """

    source_id: str
    repo: str
    revision: str
    member: str
    label: str
    family: str
    stratum: str
    acquisition_type: str
    lineage_namespace: str
    count: int
    license: str
    permission_status: str
    redistributable: bool


# --- The genuine side ----------------------------------------------------------------------
#
# Ninety lineages per language from MAVOS-DD's real splits. Five languages rather than one
# because a false-positive rate measured on English talking heads is a claim about English
# talking heads: the recording conditions, broadcast conventions, camera hardware and typical
# upload paths differ enough between these pools that a detector can behave differently in each,
# and R5-T3's twenty-clip genuine sample could not have shown it either way.
#
# Only `data_001.zip` is read per language. The second archive of each repository re-uses the
# same member numbering, so a clip drawn from both would be two records under one lineage id
# with no way to tell whether it is genuinely one recording — and the whole split guarantee
# rests on that id.
GENUINE_SOURCES = tuple(
    RemoteSource(
        f"mavos_real_{code}",
        repo,
        revision,
        "data_001.zip",
        LABEL_REAL,
        f"mavos_real_{code}",
        "genuine_in_the_wild",
        "in_the_wild_upload",
        f"mavos-real:{code}",
        90,
        "MAVOS-DD dataset terms (research use)",
        "public research mirror, pinned revision",
        False,
    )
    for code, repo, revision in (
        ("en", "34data/mavos-dd-english_real", "02cd6215882b2c18117ba133874cc471a79a4a9e"),
        ("de", "34data/mavos-dd-german_real", "846ef3709843fbff294214b10cf0cebd79371d72"),
        ("hi", "34data/mavos-dd-hindi_real", "53cb91016b0435a34c99497cf21f70761b8651ea"),
        ("zh", "34data/mavos-dd-mandarin_real", "274199e0f50dfd41c936bba0de5e9b593de3bba1"),
        ("ar", "34data/mavos-dd-arabic-real", "542a345fb9d21af63b0b04055dc2deaaf8851814"),
    )
)

# --- The manipulated side ------------------------------------------------------------------
#
# Additional swap and generated media beyond what R4-T1 already holds locally, drawn so the
# evaluation split has enough manipulated lineages for a primary-task rate to mean anything.
# The families are the ones R4-T1 measured as hardest for one detector or the other, on
# purpose: a robustness study that samples only the easy families measures its own sampling.
MANIPULATED_SOURCES = (
    RemoteSource(
        "inswapper_x", "34data/v15-human-vid-mavos-dd-english_inswapper",
        "30433869a189a03acab231ab89836dfe11331970", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_inswapper", "face_swap", "generated_manipulation",
        "mavos-inswapper", 20,
        "MAVOS-DD dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
    RemoteSource(
        "roop_x", "34data/v15-human-vid-mavos-dd-english_roop",
        "9366303b174d3c10aad2faa806ed3835f29f1a3c", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_roop", "face_swap", "generated_manipulation",
        "mavos-roop", 20,
        "MAVOS-DD dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
    RemoteSource(
        "sonic_x", "34data/MAVOS-DD-english_sonic",
        "15557eea4b9aa0d13af944ad335942211c809dfd", "data_001.zip",
        LABEL_SYNTHETIC, "talkinghead_sonic", "generated", "generated_manipulation",
        "mavos-sonic", 10,
        "MAVOS-DD dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
)

# --- What is already on this machine -------------------------------------------------------
#
# R4-T1's corpus is copied in wholesale rather than re-fetched. It is the corpus the two
# adopted thresholds and LipForensics' own operating point were measured against, so carrying
# it here is what makes "the same clip, before and after" answerable at all — and every clip in
# it already has a recorded source string, byte count and digest.
LOCAL_CORPUS_ID = "r4t1"

# Stratum names for the local corpus's three strata, mapped to this task's vocabulary. The
# genuine ones are separated from the in-the-wild pool because they are a different kind of
# genuine: benchmark-sourced, uniformly encoded, and already used to derive the thresholds
# under test.
LOCAL_STRATA = {
    "genuine_face": "genuine_benchmark",
    "face_swap": "face_swap",
    "generated": "generated",
}


def _remote_namespaces() -> dict[str, str]:
    """Repository -> lineage namespace, for every archive this task knows how to address.

    R4-T1 drew fifteen of its genuine clips and ten of its swaps from the same pinned archives
    this corpus draws from, so a clip can arrive here twice: once copied from that corpus and
    once fetched fresh. Under two different lineage namespaces those two records would be two
    independent lineages, and a false-positive rate would count one recording twice — on the
    genuine side, in the split the acceptance target is stated over.

    Mapping both back onto one namespace by *repository* is what makes them collide into a
    single lineage, which is what lets `dedupe` drop the copy and the leakage check verify the
    result rather than take it on trust.
    """
    return {
        source.repo: source.lineage_namespace
        for source in (*GENUINE_SOURCES, *MANIPULATED_SOURCES)
    }

# The two real-world lineages that produced a HIGH in production on media nobody manipulated.
# Identified here by digest only. The originals stay under `--private-dir`; no filename, no
# container metadata and no personal detail from them reaches any artifact this script writes.
@dataclass(frozen=True)
class PrivateSource:
    """A locally held, non-redistributable lineage, named by what it is rather than by whom.

    `note` is the only description that travels, and it is written to be true and impersonal:
    acquisition path and encoding shape, which is what a reader needs to judge whether the
    stratum generalises, and nothing that identifies a person or a file.
    """

    lineage_id: str
    filename: str
    stratum: str
    acquisition_type: str
    note: str


PRIVATE_SOURCES = (
    PrivateSource(
        "private:regression-1",
        "candA.mp4",
        "genuine_phone_direct",
        "consumer_phone_capture",
        "Consumer phone capture shared through a messaging app: 848x480 with a 90-degree "
        "rotation flag, ~30 fps, ~19 s, AAC at 64 kbit/s. Benign; no manipulation of any kind.",
    ),
    PrivateSource(
        "private:regression-2",
        "candB.mp4",
        "genuine_social_transcode",
        "platform_transcode_download",
        "Benign footage retrieved through a platform download path: 320x240 at 15 fps, ~19 s. "
        "Benign; no manipulation of any kind.",
    ),
)

# How many evaluation-split genuine lineages carry the constructed degradations, and how many
# manipulated ones carry the platform ladder. Both are deliberately far below the number of
# lineages available: a derivative adds no lineage to the denominator of the headline rate, so
# every one of them is bought with detector wall-clock and nothing else. Forty is what makes a
# per-stratum figure worth printing at all — with zero observed false positives it supports a
# one-sided 95% upper bound near 7%, which is a weak statement, and the report says so rather
# than spending a day of GPU time pretending otherwise.
GENUINE_DERIVATIVE_LINEAGES = 40
MANIPULATED_DERIVATIVE_LINEAGES = 12


class BuildError(Exception):
    """The corpus could not be assembled, so it was not assembled."""


def _read_archive(source: RemoteSource):
    url = RESOLVE_URL.format(
        repo=source.repo, revision=source.revision, member=source.member
    )
    try:
        handle = io.BufferedReader(_RangeFile(url), buffer_size=BUFFER_BYTES)
        return zipfile.ZipFile(handle), url
    except Exception as error:  # noqa: BLE001 - urllib and zipfile fail in many ways
        raise BuildError(f"{source.source_id}: cannot read {url}: {error}") from error


def _member_number(name: str) -> str:
    """The member's own number, which is the only handle on its underlying recording.

    `real_1017.mp4`, `MAVOS-DD-hindi_real_1017.mp4` and `inswapper_998.mp4` all reduce to their
    trailing integer. Two members of one archive with the same number are the same recording;
    two members of different archives are only ever compared within one `lineage_namespace`,
    so the numbering spaces of unrelated repositories never meet.
    """
    stem = Path(name).stem
    tail = stem.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise BuildError(f"member {name!r} does not end in a number; cannot derive a lineage")
    return tail


def fetch_remote(source: RemoteSource, clips_dir: Path) -> list[CorpusItem]:
    """Take this source's share of an archive, one corpus record per clip."""
    archive, url = _read_archive(source)
    with archive:
        catalog = [
            (info.filename, info.file_size)
            for info in archive.infolist()
            if not info.is_dir() and info.file_size <= MAX_MEMBER_BYTES
        ]
        chosen = select_members(catalog, source.count)
        if len(chosen) < source.count:
            raise BuildError(
                f"{source.source_id}: wanted {source.count} clips, "
                f"{len(chosen)} eligible members in {source.member}"
            )
        items = []
        for index, member in enumerate(chosen):
            payload = archive.read(member)
            clip_id = f"{source.source_id}_{index:03d}"
            (clips_dir / f"{clip_id}.mp4").write_bytes(payload)
            lineage = f"{source.lineage_namespace}:{_member_number(member)}"
            items.append(
                CorpusItem(
                    clip_id=clip_id,
                    path=f"clips/{clip_id}.mp4",
                    label=source.label,
                    family=source.family,
                    stratum_primary=source.stratum,
                    strata=[source.stratum],
                    source_lineage_id=lineage,
                    base_media_id=clip_id,
                    derivative_of=None,
                    derivation="none",
                    source=f"{source.repo}@{source.revision}#{source.member}:{member}",
                    acquisition_type=source.acquisition_type,
                    label_provenance="dataset label, as published by the source repository",
                    license=source.license,
                    permission_status=source.permission_status,
                    redistributable=source.redistributable,
                    private=False,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    bytes=len(payload),
                )
            )
        print(f"  {source.source_id}: {len(items)} clips")
        return items


def copy_local(local_root: Path, clips_dir: Path) -> list[CorpusItem]:
    """Copy R4-T1's corpus in, carrying its recorded provenance forward.

    Lineage ids are namespaced by family. FaceForensics++ names a fake after both the source
    and the target video (`Deepfakes_100_077`), so a swap and a real clip in that corpus can
    share an underlying recording; every FaceForensics++ clip is assigned to the calibration
    split for that reason among others, which makes the imprecision unable to cross the split
    boundary whatever it is.
    """
    corpus_dir = local_root / "r4t1_calibration"
    sources_path = corpus_dir / "sources.json"
    manifest_path = corpus_dir / "manifest.csv"
    if not sources_path.is_file() or not manifest_path.is_file():
        raise BuildError(f"no R4-T1 corpus at {corpus_dir}")

    recorded = {
        entry["clip_id"]: entry
        for entry in json.loads(sources_path.read_text(encoding="utf-8"))["clips"]
    }
    namespaces = _remote_namespaces()

    items = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            original_id = row["clip_id"].strip()
            entry = recorded.get(original_id)
            if entry is None:
                raise BuildError(f"{original_id} is in the R4-T1 manifest but not its sources")
            media = corpus_dir / row["path"].strip()
            if not media.is_file():
                raise BuildError(f"missing media {media}")
            payload = media.read_bytes()
            clip_id = f"{LOCAL_CORPUS_ID}_{original_id}"
            (clips_dir / f"{clip_id}.mp4").write_bytes(payload)
            family = entry["family"]
            stratum = LOCAL_STRATA[entry["stratum"]]
            lineage = _local_lineage(entry, family, original_id, namespaces)
            items.append(
                CorpusItem(
                    clip_id=clip_id,
                    path=f"clips/{clip_id}.mp4",
                    label=entry["label"],
                    family=family,
                    stratum_primary=stratum,
                    strata=[stratum],
                    source_lineage_id=lineage,
                    base_media_id=clip_id,
                    derivative_of=None,
                    derivation="none",
                    source=f"{LOCAL_CORPUS_ID}:{original_id} ({entry['source']})",
                    acquisition_type=(
                        "benchmark_corpus"
                        if entry["label"] == LABEL_REAL
                        else "generated_manipulation"
                    ),
                    label_provenance="R4-T1 corpus label, carried forward unchanged",
                    license="source dataset terms (research use)",
                    permission_status="already held locally under R4-T1",
                    redistributable=False,
                    private=False,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    bytes=len(payload),
                )
            )
    print(f"  {LOCAL_CORPUS_ID}: {len(items)} clips")
    return items


def _local_lineage(
    entry: dict, family: str, original_id: str, namespaces: dict[str, str]
) -> str:
    """The lineage id for one clip copied out of R4-T1's corpus.

    A clip whose recorded source names an archive this task also fetches from is resolved into
    that archive's namespace, so the two copies are one lineage. Everything else — the
    FaceForensics++ clips and the generator families with no fresh fetch — keeps a
    family-scoped id, which is unambiguous because nothing else in this corpus addresses those
    archives at all.
    """
    source = entry.get("source", "")
    if "@" in source and "#" in source and ":" in source:
        repo = source.split("@", 1)[0]
        namespace = namespaces.get(repo)
        if namespace is not None:
            member = source.rsplit(":", 1)[1]
            return f"{namespace}:{_member_number(member)}"
    return f"{family}:{original_id}"


def dedupe(items: list[CorpusItem], clips_dir: Path) -> tuple[list[CorpusItem], list[dict]]:
    """Drop clips whose bytes are already in the corpus, keeping the first occurrence.

    Two records of one file are two data points drawn from one measurement, and on the genuine
    side that is the denominator of the acceptance target. The fetched copy is dropped in
    favour of the one already recorded, and the dropped clip's file is removed from the corpus
    directory so the directory and the manifest cannot disagree.

    What survives here still has to pass `leakage_check`, which asks the same question again
    over the finished splits. This is the build doing the obvious half; that is the audit.
    """
    kept: list[CorpusItem] = []
    dropped: list[dict] = []
    seen: dict[str, CorpusItem] = {}
    for item in items:
        first = seen.get(item.sha256)
        if first is not None:
            dropped.append(
                {
                    "clip_id": item.clip_id,
                    "duplicate_of": first.clip_id,
                    "sha256": item.sha256,
                    "lineage_agreed": first.source_lineage_id == item.source_lineage_id,
                    "lineage_kept": first.source_lineage_id,
                    "lineage_dropped": item.source_lineage_id,
                }
            )
            if not item.private:
                (clips_dir / f"{item.clip_id}.mp4").unlink(missing_ok=True)
            continue
        seen[item.sha256] = item
        kept.append(item)
    return kept, dropped


def register_private(private_dir: Path) -> list[CorpusItem]:
    """Record the two held-out real-world lineages without moving a byte of them.

    `path` points at `--private-dir`, which is outside both the repository and the corpus
    directory, so building a redistributable archive of the corpus directory cannot pick them
    up by accident. Nothing derived from the file name is stored.
    """
    items = []
    for source in PRIVATE_SOURCES:
        media = private_dir / source.filename
        if not media.is_file():
            raise BuildError(
                f"private lineage {source.lineage_id} expects media at {media}, which is absent"
            )
        payload = media.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        clip_id = f"private_{source.lineage_id.split(':')[-1]}"
        items.append(
            CorpusItem(
                clip_id=clip_id,
                path=str(media.resolve()),
                label=LABEL_REAL,
                family="real_world_regression",
                stratum_primary=source.stratum,
                strata=[source.stratum, "genuine_in_the_wild"],
                source_lineage_id=source.lineage_id,
                base_media_id=clip_id,
                derivative_of=None,
                derivation="none",
                source=f"private, held locally; sha256 {digest}",
                acquisition_type=source.acquisition_type,
                label_provenance=(
                    "benign by direct inspection; no manipulation applied or claimed"
                ),
                license="not licensed for redistribution",
                permission_status="private; local evaluation only",
                redistributable=False,
                private=True,
                sha256=digest,
                bytes=len(payload),
            )
        )
        print(f"  {source.lineage_id}: registered by digest, media not copied")
    return items


def build_derivatives(
    bases: list[CorpusItem],
    derivations: tuple[str, ...],
    clips_dir: Path,
) -> list[CorpusItem]:
    """Produce the degradations for a chosen set of base clips.

    A derivative inherits label, lineage and provenance from its base and changes exactly two
    things: its stratum, and the fact that it is not redistributable. The second is inherited
    upward rather than downward — a derivative of a clip that may not be redistributed may not
    be either, and a derivative of a private clip is itself private.
    """
    produced = []
    for base in bases:
        source_path = Path(base.path)
        if not source_path.is_absolute():
            source_path = clips_dir.parent / base.path
        for derivation_id in derivations:
            derivation = derive.DERIVATIONS_BY_ID[derivation_id]
            clip_id = f"{base.clip_id}__{derivation_id}"
            destination = clips_dir / f"{clip_id}.mp4"
            derive.derive(source_path, destination, derivation)
            payload = destination.read_bytes()
            stratum = (
                derivation.stratum
                if base.label == LABEL_REAL
                else f"{base.stratum_primary}_{derivation_id}"
            )
            produced.append(
                replace(
                    base,
                    clip_id=clip_id,
                    path=f"clips/{clip_id}.mp4",
                    stratum_primary=stratum,
                    strata=sorted({*base.strata, stratum, "constructed_derivative"}),
                    base_media_id=base.base_media_id,
                    derivative_of=base.clip_id,
                    derivation=derivation_id,
                    source=f"derived from {base.clip_id} by {derivation_id}",
                    label_provenance=(
                        f"inherited from {base.clip_id}; the derivation changes encoding and "
                        "acquisition conditions only and cannot change what was recorded"
                    ),
                    redistributable=False,
                    sha256=hashlib.sha256(payload).hexdigest(),
                    bytes=len(payload),
                )
            )
        print(f"  {base.clip_id}: {len(derivations)} derivatives")
    return produced


def assign_splits(items: list[CorpusItem]) -> dict[str, str]:
    """Map every lineage to a split, by rule, so the mapping can be read and argued with.

    The rules, in the order they apply:

    - **Every FaceForensics++ lineage is calibration.** It is the corpus LipForensics was
      trained on and the corpus R5-T3 chose the threshold under test on. A hold-out that
      contains it is not measuring generalisation, it is re-reading the training set.
    - **The in-the-wild genuine pool is divided per language by member number**, the lower
      third to calibration and the upper two thirds to evaluation. Contiguous blocks rather
      than an interleave: if two members of one archive were ever cut from one recording they
      are far more likely to be adjacent in the numbering than to straddle a single boundary,
      so a block split has one place it can go wrong per language instead of a hundred. The
      evaluation side gets the larger share because the predeclared power target is stated
      over the evaluation genuine lineages, and a target that the split makes unreachable is
      not a target.
    - **The manipulated families are divided by family**, so the evaluation side carries
      generator and swap families the calibration side never saw. That is what makes the
      measurement cross-dataset rather than cross-sample.
    - **Both private lineages are evaluation.** They are real-world hold-outs by nature, and
      they are counted in the genuine evaluation denominator like every other lineage.
    """
    assignment: dict[str, str] = {}
    per_language: dict[str, list[str]] = {}

    for lineage in sorted({item.source_lineage_id for item in items}):
        if lineage.startswith("private:"):
            assignment[lineage] = SPLIT_EVALUATION
        elif lineage.startswith("ffpp_"):
            assignment[lineage] = SPLIT_CALIBRATION
        elif lineage.startswith("mavos-real:"):
            per_language.setdefault(lineage.split(":")[1], []).append(lineage)
        elif lineage.startswith(("mavos-inswapper:", "mavos-roop:", "mavos-sonic:",
                                 "t2v_veo3:", "t2v_sora2:", "t2v_kling:")):
            assignment[lineage] = SPLIT_EVALUATION
        elif lineage.startswith(("talkinghead_echomimic:", "talkinghead_memo:",
                                 "reenactment_liveportrait:")):
            assignment[lineage] = SPLIT_CALIBRATION
        else:
            raise BuildError(f"no split rule covers lineage {lineage!r}")

    for code, pool in per_language.items():
        ordered = sorted(pool, key=lambda value: int(value.rsplit(":", 1)[1]))
        boundary = len(ordered) // 3
        for lineage in ordered[:boundary]:
            assignment[lineage] = SPLIT_CALIBRATION
        for lineage in ordered[boundary:]:
            assignment[lineage] = SPLIT_EVALUATION

    return assignment


def choose_derivative_bases(
    items: list[CorpusItem], assignment: dict[str, str]
) -> tuple[list[CorpusItem], list[CorpusItem]]:
    """Pick which evaluation lineages carry the degradations, spread and deterministically.

    Evaluation-side only. The degradation strata exist to be measured, and every rate in this
    task is measured on the evaluation side; putting them on calibration lineages would spend
    the same GPU hours producing numbers no conclusion may be drawn from.

    Genuine bases are taken evenly spaced across each language's evaluation block so the
    degradation strata are multilingual rather than an artefact of whichever pool sorts first.
    The two private lineages are always included: they are the cases this task exists to
    explain, and asking what a transcode does to them is most of the question.
    """
    genuine_by_language: dict[str, list[CorpusItem]] = {}
    private: list[CorpusItem] = []
    manipulated: list[CorpusItem] = []
    for item in items:
        if not item.is_base or assignment.get(item.source_lineage_id) != SPLIT_EVALUATION:
            continue
        if item.private:
            private.append(item)
        elif item.label == LABEL_REAL and item.source_lineage_id.startswith("mavos-real:"):
            code = item.source_lineage_id.split(":")[1]
            genuine_by_language.setdefault(code, []).append(item)
        elif item.label != LABEL_REAL:
            manipulated.append(item)

    genuine: list[CorpusItem] = list(private)
    per_language = max(1, (GENUINE_DERIVATIVE_LINEAGES - len(private)) // max(
        1, len(genuine_by_language)
    ))
    for code in sorted(genuine_by_language):
        pool = sorted(genuine_by_language[code], key=lambda entry: entry.clip_id)
        genuine.extend(_evenly_spaced(pool, per_language))

    manipulated_pool = sorted(manipulated, key=lambda entry: entry.clip_id)
    return genuine, _evenly_spaced(manipulated_pool, MANIPULATED_DERIVATIVE_LINEAGES)


def _evenly_spaced(pool: list, count: int) -> list:
    """`count` entries spread over `pool`, deterministically and without a seed."""
    if count >= len(pool):
        return list(pool)
    if count <= 0:
        return []
    if count == 1:
        return [pool[0]]
    step = (len(pool) - 1) / (count - 1)
    return [pool[int(round(index * step))] for index in range(count)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_corpus",
        description="Build the R7-T5 cross-dataset robustness corpus.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--private-dir",
        required=True,
        type=Path,
        help="directory holding the private regression lineages; never copied from",
    )
    parser.add_argument(
        "--local-corpus-root", type=Path, default=Path.home() / "deepguard-corpus"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-derivatives",
        action="store_true",
        help="assemble base clips only, for a fast structural check of the splits",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        print(f"{output} is not empty; pass --overwrite to rebuild", file=sys.stderr)
        return 1
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    clips = output / "clips"
    clips.mkdir(parents=True)

    duplicates: list[dict] = []
    try:
        print("Genuine, in the wild:")
        items = []
        for source in GENUINE_SOURCES:
            items.extend(fetch_remote(source, clips))
        print("Manipulated, additional:")
        for source in MANIPULATED_SOURCES:
            items.extend(fetch_remote(source, clips))
        print("Local R4-T1 corpus:")
        items.extend(copy_local(args.local_corpus_root, clips))
        print("Private regression lineages:")
        items.extend(register_private(args.private_dir))

        items, duplicates = dedupe(items, clips)
        if duplicates:
            print(f"Dropped {len(duplicates)} duplicate clip(s) by digest.")

        assignment = assign_splits(items)

        derivatives: list[CorpusItem] = []
        if not args.skip_derivatives:
            genuine_bases, manipulated_bases = choose_derivative_bases(items, assignment)
            print(f"Derivatives, genuine ({len(genuine_bases)} lineages):")
            derivatives.extend(
                build_derivatives(
                    genuine_bases,
                    tuple(d.derivation_id for d in derive.DERIVATIONS),
                    clips,
                )
            )
            print(f"Derivatives, manipulated ({len(manipulated_bases)} lineages):")
            derivatives.extend(
                build_derivatives(manipulated_bases, derive.MANIPULATED_DERIVATIONS, clips)
            )
        items.extend(derivatives)

        stamped = split_by_lineage(items, assignment)
    except (BuildError, CorpusError, derive.DeriveError) as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 1

    manifest_sha = write_manifest(stamped, output / "manifest.csv")
    write_corpus(
        stamped,
        output / "corpus.json",
        extra={
            "task": "R7-T5",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha256": manifest_sha,
            "ffmpeg": derive.ffmpeg_version(),
            "duplicates_dropped": duplicates,
            "derivations": [
                {
                    "derivation_id": d.derivation_id,
                    "stratum": d.stratum,
                    "video_filter": d.video_filter,
                    "crf": d.crf,
                    "preset": d.preset,
                    "audio_bitrate": d.audio_bitrate,
                    "rationale": d.rationale,
                }
                for d in derive.DERIVATIONS
            ],
            "private_lineages": [
                {
                    "source_lineage_id": source.lineage_id,
                    "stratum": source.stratum,
                    "acquisition_type": source.acquisition_type,
                    "note": source.note,
                }
                for source in PRIVATE_SOURCES
            ],
            "split_assignment": dict(sorted(assignment.items())),
        },
    )
    print(f"\n{len(stamped)} clips, {len(assignment)} lineages -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
