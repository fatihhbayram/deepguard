"""R15-T5: the offline, governed manifest of the R15 out-of-sample media.

    cd apps/api && PYTHONPATH=. .venv/bin/python ../../scripts/eval/r15_t5_oos_manifest.py \
        --output ~/deepguard-corpus/r15_oos_manifest.json \
        --checks ~/deepguard-corpus/r15_oos_manifest_checks.txt \
        --root ~/deepguard-corpus --root ~/deepguard-p4-media \
        --root ~/deepguard-corpus-r7t5-private --root ~/deepguard --root ~/.cache/huggingface \
        --db-container deepguard-postgres-1

Reads the R15-T1 audio and R15-T2 genuine video records as their tasks froze them and writes one
JSON manifest. Nothing is uploaded, nothing is written to the database, no detector runs. The
database is only *read*, inside a READ ONLY transaction, and only when `--db-container` is given,
to check that no manifest hash is already held there.

**The vocabulary is imported, not restated.** `DatasetSplit`, `GroundTruthLabel` and
`SourceClass` come from the api package, and every record's governance fields are passed through
`MediaGovernanceContract`, so a split, label or field the platform would refuse cannot be
written here.

**Lineage and grouping are two things.** `source_lineage_id` keeps its R12 meaning: the source
recording a file descends from. Every file here is an original as its publisher or owner holds it
(uncoded ASVspoof rows, camera originals, transcodes of unknown parentage), so each file is its
own lineage. Who is speaking or on camera is a different fact, held in `speaker_id` (audio) and
`subject_id` (video), which are manifest-only grouping metadata and not governance fields.

**The split belongs to the speaker.** The leakage unit for audio is the speaker: a zero-shot TTS
or VC spoof is built to sound like its target speaker, so a speaker whose bona fide speech is in
one split and whose spoofs are in another leaks voice identity across the split. The split is a
pure function of `speaker_id`, the same for the primary and the auxiliary group, so the 34
speakers present in both land in one split by construction rather than by bookkeeping. The
genuine video is one subject in one session and is never divided. How the speaker constraint is
kept if this is ever imported into the database is an ingestion question; this task does not
bend lineage to answer it.

**Primary and auxiliary never mix.** `primary_oos` holds only the 550 family-novel audio files
and the 6 frozen camera originals. The 50 known-family audio files (A19, A29) and the 6
auxiliary transcodes sit in `auxiliary_known_family`. The 4 excluded video files appear only as
hashes that must stay out.

**Deterministic.** No timestamps, records sorted by hash, keys sorted, fixed separators: a rerun
on the same inputs is byte-identical.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import typing
from pathlib import Path

from app.dataset_governance import DatasetSplit, MediaGovernanceContract
from app.ground_truth import GroundTruthLabel, SourceClass

MANIFEST_SCHEMA = "r15-t5-oos-manifest-1"

HOME = Path.home()
AUDIO_ROOT = HOME / "deepguard-corpus/r15_oos_raw/audio"
VIDEO_RECORD = HOME / "deepguard-corpus/r15_oos_raw/genuine_video"
# R15-T2 verified the clips in place; the record directory's clips/ is empty by design.
VIDEO_SOURCE = HOME / "deepguard-p4-media/R15-T2-Real-Video"

SPLITS = typing.get_args(DatasetSplit)
LABELS = typing.get_args(GroundTruthLabel)
SOURCE_CLASSES = typing.get_args(SourceClass)

# Speaker → split. A salted SHA-256 of the speaker id, so the assignment depends on nothing but
# the id: not on file order, not on which group the speaker appears in, not on what else is in
# the manifest. Half the hash space goes to each split.
SPLIT_SALT = "r15-t5:speaker-split:v1"
AUDIO_SPLITS = ("CALIBRATION", "HOLDOUT")
# One subject, 6 clips, one session: too few to calibrate on, and splitting them would put one
# face on both sides. Evaluation only.
VIDEO_SPLIT = "HOLDOUT"

# R15-T1 SOURCES.txt §5 N2: Malafide attacks optimised against an AASIST countermeasure, the
# deployed audio detector's architecture. They must be reportable on their own.
AASIST_TARGETED = {"A18", "A20", "A23", "A30"}

ASVSPOOF_SOURCE = (
    "ASVspoof 5 evaluation partition, track_1 protocol; Zenodo record 14498691 v0.1, "
    "flac_E_aa.tar, uncoded rows"
)
ASVSPOOF_PERMISSION = (
    "ODC-By 1.0 s3.1 grant (commercial use included); attribution notice on public use of "
    "results; not redistributed by DeepGuard"
)
VIDEO_PERMISSION = (
    "owner written consent 2026-09-30/2026-10-01: internal forensic evaluation; R15 face-swap "
    "target-base use; public redistribution NOT granted"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    """`sha256  name` lines → {name: sha256}; `#` lines are headers."""
    sums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        if name in sums:
            raise SystemExit(f"{path}: {name} listed twice")
        sums[name] = digest
    return sums


def read_protocol(path: Path) -> dict[str, dict[str, str]]:
    """The official track_1 key lines, by file name (README §3 column order)."""
    columns = (
        "speaker_id", "file", "gender", "codec", "codec_q", "codec_seed",
        "attack_tag", "attack_label", "key", "tmp",
    )
    rows: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        row = dict(zip(columns, line.split(), strict=True))
        rows[row["file"] + ".flac"] = row
    return rows


def read_matrix(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="") as fh:
        return {row["attack_id"]: row for row in csv.DictReader(fh, delimiter="\t")}


def speaker_split(speaker_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{speaker_id}".encode()).digest()
    return AUDIO_SPLITS[digest[0] >> 7]


def governed(record: dict) -> dict:
    """Validate the governance fields against the platform's own contract, then the GT axes."""
    governance = {k: record[k] for k in MediaGovernanceContract.model_fields}
    MediaGovernanceContract(**governance)
    if record["label"] not in LABELS:
        raise SystemExit(f"{record['file_hash']}: label {record['label']!r} not canonical")
    if record["source_class"] not in SOURCE_CLASSES:
        raise SystemExit(f"{record['file_hash']}: source_class not canonical")
    return record


def stratum_for(attack_id: str, kind: str) -> str:
    if kind == "TTS":
        return "audio_tts"
    if kind == "VC":
        return "audio_vc"
    if kind == "ADV":
        return (
            "audio_adversarial_aasist_targeted"
            if attack_id in AASIST_TARGETED
            else "audio_adversarial_asv_targeted"
        )
    raise SystemExit(f"{attack_id}: unknown attack kind {kind!r}")


def audio_records(group: str, matrix: dict[str, dict[str, str]]) -> list[dict]:
    base = AUDIO_ROOT / group
    sums = read_sums(base / "SHA256SUMS.txt")
    protocol = read_protocol(base / "protocol_excerpt.eval.track_1.tsv")
    if set(sums) != set(protocol):
        raise SystemExit(f"{group}: SHA256SUMS and protocol excerpt name different files")

    records = []
    for name, digest in sums.items():
        path = base / "asvspoof5_eval" / name
        row = protocol[name]
        if row["codec"] != "-":
            raise SystemExit(f"{name}: coded row in an uncoded selection")

        bonafide = row["key"] == "bonafide"
        if bonafide != (row["attack_label"] == "bonafide") or row["key"] not in {
            "bonafide", "spoof",
        }:
            raise SystemExit(f"{name}: KEY and ATTACK_LABEL disagree")

        family = matrix[row["attack_label"]]
        speaker = row["speaker_id"]
        attack_id = None if bonafide else row["attack_label"]

        records.append(governed({
            "file_path": str(path),
            "file_hash": digest,
            "media_type": "audio",
            "label": "GENUINE" if bonafide else "AUDIO_MANIPULATION",
            # The label is the publisher's official protocol key, not ours.
            "source_class": "EXTERNAL_VERIFIED",
            "speaker_id": speaker,
            "subject_id": None,
            "attack_id": attack_id,
            "attack_family": None if bonafide else family["generator_or_attack_family"],
            # Each uncoded protocol row is its own source utterance.
            "source_lineage_id": f"asvspoof5:{row['file']}",
            "dataset_split": speaker_split(speaker),
            "recording_identity": f"asvspoof5:{row['file']}",
            # Uncoded protocol rows (CODEC "-"): bytes as published, no step applied.
            "transformations": [],
            "derived_from_sha256": None,
            # The attack system as its publisher names it (arXiv:2502.08857 Table 2).
            "generation_pipeline": (
                None if bonafide
                else f"ASVspoof 5 {attack_id}: {family['generator_or_attack_family']}"
            ),
            "license": (
                "CC BY 4.0 (bona fide contents, attribution required); ODC-By 1.0 "
                "(ASVspoof 5 database, attribution required)"
                if bonafide else "ODC-By 1.0 (ASVspoof 5 database, attribution required)"
            ),
            "permission_status": ASVSPOOF_PERMISSION,
            "stratum_primary": (
                "genuine_read_speech" if bonafide
                else stratum_for(attack_id, family["kind"])
            ),
            "source": ASVSPOOF_SOURCE,
            "acquisition_type": "benchmark_corpus" if bonafide else "generated_manipulation",
            "benchmark_family": "asvspoof5_eval_track1",
            "redistributable": True,
            "private": False,
        }))
    return records


def video_records(sums_file: str, primary: bool) -> list[dict]:
    records = []
    for name, digest in read_sums(VIDEO_RECORD / "metadata" / sums_file).items():
        stem = name.rsplit(".", 1)[0]
        records.append(governed({
            "file_path": str(VIDEO_SOURCE / name),
            "file_hash": digest,
            "media_type": "video",
            "label": "GENUINE",
            # The owner made the recordings and says so (R15-T2 SOURCES.txt §5-6).
            "source_class": "OWNER_KNOWN",
            "speaker_id": None,
            # One subject, one session: the grouping that keeps one face on one side.
            "subject_id": "r15-t2:owner-subject-1",
            "attack_id": None,
            "attack_family": None,
            # Each clip is a separate take (R15-T2 SOURCES.txt §7), so its own lineage.
            "source_lineage_id": f"r15-t2:{stem}",
            "dataset_split": VIDEO_SPLIT,
            "recording_identity": f"r15-t2:{stem}",
            # Camera originals: none. Auxiliary transcodes: steps unknown, so not recorded.
            "transformations": [] if primary else None,
            "derived_from_sha256": None,
            "generation_pipeline": None,
            "license": "none; owner-held recording used under written consent",
            "permission_status": VIDEO_PERMISSION,
            "stratum_primary": "genuine_phone_direct" if primary else "genuine_social_transcode",
            "source": (
                "owner self-recorded, iPhone 14 camera original (R15-T2 batch B3)" if primary
                else "owner-supplied undated transcode (R15-T2 batch B1/B2)"
            ),
            "acquisition_type": "consumer_phone_capture",
            "benchmark_family": "r15_t2_genuine_video",
            "redistributable": False,
            "private": True,
        }))
    return records


def by_hash(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: r["file_hash"])


def summarise(records: list[dict]) -> dict:
    counts: dict[str, collections.Counter] = {
        "by_split": collections.Counter(),
        "by_label": collections.Counter(),
        "by_split_and_label": collections.Counter(),
        "by_attack_id_and_split": collections.Counter(),
        "by_stratum_primary": collections.Counter(),
    }
    lineages: dict[str, set[str]] = collections.defaultdict(set)
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for r in records:
        counts["by_split"][r["dataset_split"]] += 1
        counts["by_label"][r["label"]] += 1
        counts["by_split_and_label"][f"{r['dataset_split']}/{r['label']}"] += 1
        counts["by_stratum_primary"][r["stratum_primary"]] += 1
        if r["attack_id"]:
            counts["by_attack_id_and_split"][f"{r['attack_id']}/{r['dataset_split']}"] += 1
        lineages[r["dataset_split"]].add(r["source_lineage_id"])
        groups[r["dataset_split"]].add(r["speaker_id"] or r["subject_id"])
    out = {k: dict(sorted(v.items())) for k, v in counts.items()}
    out["files"] = len(records)
    out["lineages_by_split"] = {k: len(v) for k, v in sorted(lineages.items())}
    out["speakers_or_subjects_by_split"] = {k: len(v) for k, v in sorted(groups.items())}
    return out


def check(condition: bool, message: str, log: list[str]) -> None:
    log.append(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        print("\n".join(log), file=sys.stderr)
        raise SystemExit(1)


def disk_overlaps(hashes: set[str], sizes: set[int], roots: list[Path], exclude: list[Path]):
    """Every regular file under `roots` whose bytes equal a manifest file. Only size-equal files
    can share a SHA-256, so only those are hashed: exact, not a sample."""
    excluded = [os.path.realpath(p) for p in exclude]
    scanned = hashed = 0
    hits = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            real = os.path.realpath(dirpath)
            if any(real == e or real.startswith(e + os.sep) for e in excluded):
                dirnames[:] = []
                continue
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                scanned += 1
                try:
                    if os.path.getsize(path) not in sizes:
                        continue
                    hashed += 1
                    if sha256_file(Path(path)) in hashes:
                        hits.append(path)
                except OSError:
                    continue
    return scanned, hashed, sorted(hits)


DB_QUERY = """
BEGIN TRANSACTION READ ONLY;
SELECT original_sha256 FROM media_files
UNION ALL SELECT derivative_sha256 FROM media_files WHERE derivative_sha256 IS NOT NULL
UNION ALL SELECT media_sha256 FROM ground_truth
UNION ALL SELECT media_sha256 FROM media_governance;
COMMIT;
"""


def db_hashes(container: str) -> list[str]:
    out = subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-U", "deepguard", "-d", "deepguard",
         "-At", "-v", "ON_ERROR_STOP=1"],
        input=DB_QUERY, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if len(line) == 64]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--checks", required=True, type=Path)
    ap.add_argument("--root", action="append", default=[], type=Path,
                    help="tree to scan for byte-identical copies (repeatable)")
    ap.add_argument("--db-container", help="read the DB's hashes in a READ ONLY transaction")
    args = ap.parse_args(argv)

    log: list[str] = []
    matrix = read_matrix(AUDIO_ROOT / "metadata/family_novelty_matrix.tsv")
    eligible = {a for a, row in matrix.items() if row["oos_eligible"] == "yes"}

    primary_audio = by_hash(audio_records("oos_population", matrix))
    aux_audio = by_hash(audio_records("known_family_new_samples", matrix))
    primary_video = by_hash(video_records("FROZEN_PRIMARY_SHA256SUMS.txt", primary=True))
    aux_video = by_hash(video_records("AUXILIARY_SHA256SUMS.txt", primary=False))
    excluded = read_sums(VIDEO_RECORD / "metadata/EXCLUDED_SHA256SUMS.txt")
    everything = primary_audio + aux_audio + primary_video + aux_video

    # --- composition ---------------------------------------------------------------------------
    check(len(primary_audio) == 550 and len(aux_audio) == 50,
          f"audio counts: primary {len(primary_audio)}, auxiliary {len(aux_audio)}", log)
    check(len(primary_video) == 6 and len(aux_video) == 6,
          f"video counts: primary {len(primary_video)}, auxiliary {len(aux_video)}", log)
    acquired = set(read_sums(AUDIO_ROOT / "metadata/SHA256SUMS.txt").values())
    check({r["file_hash"] for r in primary_audio + aux_audio} == acquired,
          "audio primary + auxiliary == the 600 acquired (metadata/SHA256SUMS.txt)", log)
    all_video = set(read_sums(VIDEO_RECORD / "metadata/SHA256SUMS.txt").values())
    check({r["file_hash"] for r in primary_video + aux_video} | set(excluded.values())
          == all_video, "video primary + auxiliary + excluded == the 16 placed", log)
    hashes = [r["file_hash"] for r in everything]
    check(len(set(hashes)) == len(hashes), f"{len(hashes)} records, 0 duplicate hashes", log)
    check(not set(hashes) & set(excluded.values()), "0 excluded video hashes in the manifest", log)

    # --- primary vs auxiliary --------------------------------------------------------------------
    primary_attacks = {r["attack_id"] for r in primary_audio if r["attack_id"]}
    aux_attacks = {r["attack_id"] for r in aux_audio}
    check(primary_attacks <= eligible and len(primary_attacks) == 14,
          f"primary audio attacks are the 14 family-novel ones: {sorted(primary_attacks)}", log)
    check(aux_attacks == {a for a in matrix if a != "bonafide"} - eligible == {"A19", "A29"},
          f"auxiliary audio attacks are the known families only: {sorted(aux_attacks)}", log)
    check(not any(r["label"] == "GENUINE" for r in aux_audio),
          "no bona fide audio in the auxiliary group", log)
    check(all(r["stratum_primary"] == "genuine_phone_direct" for r in primary_video)
          and all(r["stratum_primary"] == "genuine_social_transcode" for r in aux_video),
          "primary video = camera originals only; transcodes only in auxiliary", log)
    check({r["attack_id"] for r in everything if r["stratum_primary"]
           == "audio_adversarial_aasist_targeted"} == AASIST_TARGETED,
          f"AASIST-targeted attacks isolated by stratum: {sorted(AASIST_TARGETED)}", log)

    # --- bytes ---------------------------------------------------------------------------------
    mismatched = [r["file_path"] for r in everything
                  if sha256_file(Path(r["file_path"])) != r["file_hash"]]
    check(not mismatched, f"all {len(everything)} files re-hashed; match the frozen SHA256SUMS"
          + (f" (mismatch: {mismatched})" if mismatched else ""), log)

    # --- leakage -------------------------------------------------------------------------------
    split_of: dict[str, set[str]] = collections.defaultdict(set)
    for r in everything:
        split_of[r["source_lineage_id"]].add(r["dataset_split"])
    check(all(len(s) == 1 for s in split_of.values()),
          f"{len(split_of)} lineages, each in exactly one split", log)
    check(len(split_of) == len(everything)
          and all(r["recording_identity"] == r["source_lineage_id"] for r in everything),
          "lineage = the file's own source recording (one per file); none keyed on a speaker "
          "or subject", log)
    subjects: dict[str, set[str]] = collections.defaultdict(set)
    for r in primary_video + aux_video:
        subjects[r["subject_id"]].add(r["dataset_split"])
    check(all(len(s) == 1 for s in subjects.values()),
          f"{len(subjects)} video subject(s), each in exactly one split", log)
    speakers: dict[str, set[str]] = collections.defaultdict(set)
    for r in primary_audio + aux_audio:
        speakers[r["speaker_id"]].add(r["dataset_split"])
    check(all(len(s) == 1 for s in speakers.values()),
          f"{len(speakers)} speakers, each in exactly one split (both groups together)", log)
    shared = {r["speaker_id"] for r in primary_audio} & {r["speaker_id"] for r in aux_audio}
    both = ({r["speaker_id"] for r in primary_audio if r["label"] == "GENUINE"}
            & {r["speaker_id"] for r in primary_audio if r["label"] != "GENUINE"})
    check(len(shared) == 34, f"{len(shared)} speakers shared primary/auxiliary, none divided", log)
    check(len(both) == 41, f"{len(both)} primary speakers with bona fide and spoof, none divided",
          log)
    check(all(r["dataset_split"] in SPLITS for r in everything),
          f"every split is a canonical DatasetSplit literal {SPLITS}", log)
    per_attack = collections.Counter((r["attack_id"], r["dataset_split"])
                                     for r in primary_audio if r["attack_id"])
    check(all(per_attack[(a, s)] > 0 for a in primary_attacks for s in AUDIO_SPLITS),
          "every primary attack has files in both CALIBRATION and HOLDOUT", log)

    # --- novelty against what DeepGuard already holds --------------------------------------------
    if args.root:
        sizes = {os.path.getsize(r["file_path"]) for r in everything}
        # R15's own acquisition directories hold the manifest's files and would find them;
        # novelty is measured against everything else only.
        own = [AUDIO_ROOT, VIDEO_SOURCE, VIDEO_RECORD]
        log.append("      disk reference roots: " + ", ".join(map(str, args.root)))
        log.append("      excluded from the reference set (R15's own sources): "
                   + ", ".join(map(str, own)))
        scanned, hashed, hits = disk_overlaps(set(hashes), sizes, args.root, exclude=own)
        check(not hits, f"disk: {scanned} files under {len(args.root)} roots (R15 sources "
              f"excluded), {hashed} size-equal hashed, {len(hits)} byte-identical"
              + (f" {hits}" if hits else ""), log)
        # Positive control: the same scan with nothing excluded must find every manifest file
        # at its own path and nowhere else. A scanner that cannot see them proves nothing.
        _, _, seen = disk_overlaps(set(hashes), sizes, args.root, exclude=[])
        check(seen == sorted(r["file_path"] for r in everything),
              f"control: same roots, nothing excluded -> {len(seen)} byte-identical, exactly "
              f"the {len(everything)} manifest files at their own paths", log)
    if args.db_container:
        stored = db_hashes(args.db_container)
        overlap = sorted(set(stored) & set(hashes))
        check(not overlap, f"database (READ ONLY): {len(stored)} recorded hashes, "
              f"{len(overlap)} in the manifest" + (f" {overlap}" if overlap else ""), log)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "task": "R15-T5",
        "canonical_vocabulary": {
            "dataset_split": list(SPLITS),
            "label": list(LABELS),
            "source_class": list(SOURCE_CLASSES),
            "governance_fields": list(MediaGovernanceContract.model_fields),
        },
        "split_policy": {
            "audio": (
                "split = CALIBRATION if sha256('" + SPLIT_SALT + ":' + speaker_id)[0] < 0x80 "
                "else HOLDOUT; grouped on speaker_id, so every file of a speaker, bona fide or "
                "spoof, primary or auxiliary, is in one split"
            ),
            "video": f"grouped on subject_id; one subject, one session: all clips {VIDEO_SPLIT}",
            "lineage": (
                "source_lineage_id keeps its R12 meaning (the source recording a file descends "
                "from); every file here is its own lineage. speaker_id / subject_id are "
                "manifest-only grouping metadata, not governance fields"
            ),
            "unused_splits": [s for s in SPLITS if s not in (*AUDIO_SPLITS, VIDEO_SPLIT)],
        },
        "primary_oos": {
            "audio": primary_audio,
            "video": primary_video,
        },
        "auxiliary_known_family": {
            "audio_known_family_new_samples": aux_audio,
            "video_auxiliary_genuine_reference": aux_video,
        },
        "excluded_must_not_enter": {
            "video": sorted(excluded.values()),
        },
        "inputs": {
            str(p): sha256_file(p) for p in sorted([
                AUDIO_ROOT / "metadata/family_novelty_matrix.tsv",
                AUDIO_ROOT / "metadata/SHA256SUMS.txt",
                AUDIO_ROOT / "oos_population/SHA256SUMS.txt",
                AUDIO_ROOT / "oos_population/protocol_excerpt.eval.track_1.tsv",
                AUDIO_ROOT / "known_family_new_samples/SHA256SUMS.txt",
                AUDIO_ROOT / "known_family_new_samples/protocol_excerpt.eval.track_1.tsv",
                VIDEO_RECORD / "metadata/FROZEN_PRIMARY_SHA256SUMS.txt",
                VIDEO_RECORD / "metadata/AUXILIARY_SHA256SUMS.txt",
                VIDEO_RECORD / "metadata/EXCLUDED_SHA256SUMS.txt",
                VIDEO_RECORD / "metadata/SHA256SUMS.txt",
            ])
        },
        "summary": {
            "primary_oos": summarise(primary_audio + primary_video),
            "auxiliary_known_family": summarise(aux_audio + aux_video),
        },
    }

    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    args.output.write_text(text)
    log.append(f"manifest sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    args.checks.write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
