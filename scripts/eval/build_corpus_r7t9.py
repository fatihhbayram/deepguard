"""Assemble the R7-T9 corpus: a third independent sample, new pools, new families.

    PYTHONPATH=scripts python3 scripts/eval/build_corpus_r7t9.py \
        --output-dir ../deepguard-corpus/r7t9 \
        --prior-corpus ../deepguard-corpus/r7t5/corpus.json \
        --prior-corpus ../deepguard-corpus/r7t8/corpus.json \
        --private-dir ../deepguard-corpus-r7t5-private

R7-T8 answered the question R7-T7 could not: Effort's low-false-positive region survives contact
with new media, and the midpoint rule that picked an operating point inside it did not. This
corpus exists so the replacement selection rule can be tested on media that neither previous
corpus contained, with the power to support a 1% false-positive bound on BOTH sides of the
study rather than only on the validation side.

**Two differences from R7-T8's corpus, and both are requirements rather than improvements.**

*Power on both splits.* R7-T8 needed 299 readable genuine lineages in validation. Here the
threshold is chosen under a false-positive constraint measured on the calibration split, so the
calibration split needs the same power: a constraint measured over eighty lineages is not a
constraint. Both splits are therefore sized to leave at least 299 readable genuine lineages
after abstention.

*Core acquisition coverage.* The false-positive bound is a claim about the media the detector
consented to read. R7-T8 found a genuine pool that abstained 30 times out of 30, and its 0.83%
bound was therefore a statement about media with a plainly visible face rather than about
genuine media. This corpus names three core acquisition domains in advance - direct phone or
consumer capture, web and social acquired video, broadcast and television - and carries at least
thirty independent lineages of each on both sides, so the coverage gate has something to measure.
A deliberately hard long-tail pool is kept as a declared non-core stratum so the abstention
finding stays visible rather than being designed out.

**Everything is new, and the build refuses to be otherwise.** `_ffpp_excluded_ids` reads the
FaceForensics++ originals reachable from BOTH prior corpora - 314 of them - and refuses any
member naming one. Pools that share a lineage namespace draw from disjoint blocks of it. Each
pool declares the upstream dataset it is built on, so a manipulated pool sitting on top of a
genuine pool a previous task read is caught by a relationship no identifier expresses.

**Splits are decided by name, before anything is scored.** `assign_splits` maps every family
explicitly and whole pools and whole families are held out, so the validation side carries
genuine domains and manipulation methods the threshold never saw.

**Private media never enters the repository or the corpus directory.** The two real-world
regression lineages are the one declared re-use in this task, they are referenced where they
already live under `--private-dir`, they are held out of BOTH headline populations, and the
calibration split never sees them so they cannot influence threshold selection.

Standard library only, plus ffmpeg on PATH for the derivatives. Nothing under `apps/` is read,
imported or written.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.fetch_corpus import BUFFER_BYTES, RESOLVE_URL, _RangeFile
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

# Unchanged from R7-T5 and R7-T8: one 300 MB member would dominate the download and the
# detector's wall clock while contributing exactly one lineage.
MAX_MEMBER_BYTES = 30 * 1024 * 1024

# How many member payloads are pulled from one archive at a time. The mirror serves each member
# over its own HTTP range request and a single-threaded build spends almost all of its wall clock
# waiting on those, so the fetch is parallel. Nothing about WHICH members are taken changes: the
# selection above is deterministic and the results are reassembled in that order. Each worker
# opens its own `ZipFile` over its own range-backed handle, because a `ZipFile` is a single
# cursor and sharing one across threads would interleave reads.
FETCH_WORKERS = 8

# The regression lineages are the only declared re-use in this corpus, and this family name is
# what keeps them out of the headline denominators. `analyze_effort_r7t9` keys on it.
REGRESSION_FAMILY = "real_world_regression"

# Stratum labels. The first three are the core acquisition domains the protocol names in
# advance and the coverage gate is evaluated over; the last two are declared non-core.
STRATUM_PHONE = "genuine_phone_direct"
STRATUM_WEB_SOCIAL = "genuine_web_social"
STRATUM_BROADCAST = "genuine_broadcast_tv"
STRATUM_HIGH_RES = "genuine_high_resolution"
STRATUM_LONG_TAIL = "genuine_long_tail"


# The dialogue number inside a MELD member name, whichever of the pool's two spellings is used.
_MELD_DIALOGUE = re.compile(r"dia(\d+)")


class BuildError(Exception):
    """The corpus could not be assembled, so it was not assembled."""


# --- Lineage rules -------------------------------------------------------------------------
#
# Each source declares which rule applies to it, so a reader can check the claim "this is one
# recording" per pool rather than trusting one regex to have guessed right on nine naming
# conventions at once.


def _rule_stem(stem: str) -> str:
    """The whole stem. For pools whose member name is already one recording's handle."""
    return stem


def _rule_trailing_int(stem: str) -> str:
    """`real_1017` -> `1017`. One archive's member number is its handle on one recording."""
    tail = stem.replace("-", "_").rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise BuildError(f"member {stem!r} does not end in a number; cannot derive a lineage")
    return tail


def _rule_ffpp_target(stem: str) -> str:
    """`001_870` -> `001`, the FaceForensics++ target original, which supplies the video."""
    parts = stem.split("_")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise BuildError(f"member {stem!r} is not a FaceForensics++ <target>_<source> name")
    return parts[0]


def _rule_meld_dialogue(stem: str) -> str:
    """`dev_dia101_utt0` -> `dev_dia101`, the scene several utterances were cut from.

    Utterances of one dialogue are cuts of one continuous recording. Counting them separately
    would inflate a false-positive denominator by the cutting, which is the error the lineage
    unit exists to prevent.

    The pool ships three upstream subsets in one archive and spells a handful of the held-out
    ones differently - `test_final_videos_testdia93_utt9` beside `test_dia93_utt2`. Those are
    the same scene under two names, so the rule reads the subset from the leading token and the
    dialogue number from the last `dia<N>` in the stem, and both spellings collapse onto one
    lineage. A rule that took the name at face value would have counted one recording twice.
    """
    match = _MELD_DIALOGUE.search(stem)
    subset = stem.split("_", 1)[0]
    if match is None or subset not in ("train", "dev", "test"):
        raise BuildError(f"member {stem!r} is not a MELD <subset>...dia<N>_utt<M> name")
    return f"{subset}_dia{int(match.group(1))}"


def _rule_youtube_id(stem: str) -> str:
    """`-0M7SoM-zMM_1.16-33.00_2.0fps` -> `-0M7SoM-zMM`, the upload several cuts came from."""
    return stem.split("_", 1)[0]


def _rule_smd_uuid(stem: str) -> str:
    """`smd_real_03465397-...` -> the uuid. One upload, named by its own identifier."""
    parts = stem.split("_")
    if len(parts) < 3:
        raise BuildError(f"member {stem!r} is not an `smd_<label>_<uuid>` name")
    return "_".join(parts[2:])


def _rule_strip_clip_index(stem: str) -> str:
    """`Abuse001_x264_0004` -> `Abuse001_x264`, the source video behind several cut clips."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


LINEAGE_RULES = {
    "stem": _rule_stem,
    "trailing_int": _rule_trailing_int,
    "ffpp_target": _rule_ffpp_target,
    "meld_dialogue": _rule_meld_dialogue,
    "youtube_id": _rule_youtube_id,
    "smd_uuid": _rule_smd_uuid,
    "strip_clip_index": _rule_strip_clip_index,
}


@dataclass(frozen=True)
class RemoteSource:
    """One mirrored archive, how much of it this corpus takes, and what a lineage is in it."""

    source_id: str
    repo: str
    revision: str
    member: str
    label: str
    family: str
    stratum: str
    acquisition_type: str
    lineage_namespace: str
    lineage_rule: str
    count: int
    split: str
    license: str
    permission_status: str
    redistributable: bool
    # The upstream dataset(s) this pool's media is built on, by name. Screened against every
    # prior corpus by the independence check: a relationship no identifier expresses.
    parent_datasets: tuple[str, ...] = ()
    # Only members whose name begins with one of these prefixes are eligible. Used where one
    # archive carries several disjoint blocks that belong on different sides of the study - the
    # broadcast pool ships its upstream train/dev/test subsets in one file - or where a pool
    # holds a category this corpus deliberately does not take. Declared here, before anything is
    # scored, and recorded in the screening record.
    member_prefixes: tuple[str, ...] = ()
    # Sources sharing a namespace draw from disjoint blocks of it, claimed in build order.
    shares_namespace: bool = False
    # Families named after FaceForensics++ originals are screened against the ids the prior
    # corpora used. Everything else has no relationship to that numbering.
    ffpp_named: bool = False
    note: str = ""


def _consumed_ids(stem: str) -> set[str]:
    """Every FaceForensics++ original a member's name refers to, zero-padded.

    A swap or reenactment names the recording it took the video from and the one it took the
    face from. Both are spent by that clip: another family reusing either is not sampling an
    independent recording, whichever position it puts it in. FaceForensics++ originals are
    numbered 000-999, so a longer run of digits is a frame range and is not claimed as one.
    """
    return {token.zfill(3) for token in stem.split("_") if token.isdigit() and len(token) <= 3}


@dataclass(frozen=True)
class PrivateSource:
    """A locally held, non-redistributable lineage, named by what it is rather than by whom."""

    lineage_id: str
    filename: str
    stratum: str
    acquisition_type: str
    note: str


# The one declared re-use in this corpus. Carried forward from R7-T5/R7-T7/R7-T8 because the
# task requires these two lineages to be re-read under the newly derived operating point. They
# are excluded from every headline denominator by their family name, and they are assigned to
# neither split's headline population - `assign_splits` puts them on the validation side so they
# are scored in the same pass, and `analyze_effort_r7t9._independent` removes them from every
# rate computed there.
PRIVATE_SOURCES = (
    PrivateSource(
        "private:regression-1",
        "candA.mp4",
        STRATUM_PHONE,
        "consumer_phone_capture",
        "Consumer phone capture shared through a messaging app: 848x480 with a 90-degree "
        "rotation flag, ~30 fps, ~19 s, AAC at 64 kbit/s. Benign; no manipulation of any kind.",
    ),
    PrivateSource(
        "private:regression-2",
        "candB.mp4",
        STRATUM_WEB_SOCIAL,
        "platform_transcode_download",
        "Benign footage retrieved through a platform download path: 320x240 at 15 fps, ~19 s. "
        "Benign; no manipulation of any kind.",
    ),
)

# Lineages the independence gate implicated and the build therefore refuses.
#
# The first gate run over this corpus reported twenty-four candidate pairs inside the
# high-similarity band. The predeclared 256-bit corroboration test refuted twelve of them and
# confirmed twelve, and a confirmed pair is a finding: the protocol says an unresolved or
# confirmed high-similarity candidate fails the check and may not be whitelisted.
#
# **The remedy is the corpus, not the check.** The corroboration threshold was fixed in advance
# and is not revised here, even though the evidence suggests it is set conservatively - the four
# known-duplicate positive controls sit at 0 of 256 bits while every confirmed pair sits between
# 27 and 63, with no gap separating them from the refuted band at 66 to 92, and several confirmed
# pairs are between media that cannot physically share a recording (one generated clip matching
# both a sign-language recording and a FaceForensics++ original). Loosening a gate after seeing
# which pairs it caught is the manoeuvre the predeclaration exists to prevent, so instead the ten
# implicated R7-T9 lineages are dropped. That costs six genuine validation lineages out of 537
# and buys an independence claim that rests on the check as written.
#
# Dropping happens after selection rather than before it, so every surviving clip keeps the
# member it would have had anyway and a rebuild can reuse the bytes already fetched.
LEAKAGE_EXCLUDED_LINEAGES = {
    "ffpp-orig:427",
    "ffpp-orig:740",
    "meld:test_dia270",
    "t2v-gen3:gen3_00102",
    "t2v-gen3:gen3_00721",
    "v-librasil:Ab#U00f3bora_Articulador2",
    "v-librasil:Almo#U00e7o_Articulador1",
    "v-librasil:Ao redor_Articulador2",
    "v-librasil:Aranha_Articulador2",
    "v-librasil:Barbear_Articulador3",
}


# How many validation-split lineages carry the constructed degradations. Well below what is
# available, on purpose: a derivative adds no lineage to the headline denominator, so every one
# of them is bought with detector wall-clock and nothing else.
GENUINE_DERIVATIVE_LINEAGES = 24
MANIPULATED_DERIVATIVE_LINEAGES = 12


def _ffpp_excluded_ids(prior_corpora: list[Path]) -> set[str]:
    """Every FaceForensics++ original id any prior task touched, from its own corpus record.

    Read out of the artifacts rather than transcribed, so it cannot drift from what those tasks
    actually held. Three spellings are in play across the two predecessors and all three are
    read: R7-T7's `ffpp_deepfakes:ffpp_dev_Deepfakes_100_077` and `ffpp_real:ffpp_dev_real_000`,
    R7-T8's `ffpp-orig:143`, and the `<target>_<source>` member names both recorded in `source`.
    """
    excluded: set[str] = set()
    for path in prior_corpora:
        payload = json.loads(path.read_text(encoding="utf-8"))
        found_here: set[str] = set()
        for entry in payload["items"]:
            lineage = entry["source_lineage_id"]
            key = lineage.split(":", 1)[1] if ":" in lineage else lineage
            if lineage.startswith("ffpp_") or lineage.startswith("ffpp-orig:"):
                for token in key.replace("-", "_").split("_"):
                    if token.isdigit() and len(token) <= 3:
                        found_here.add(token.zfill(3))
            source = entry.get("source", "")
            if "#" in source:
                stem = Path(source.rsplit(":", 1)[-1]).stem
                parts = stem.split("_")
                if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
                    found_here |= {
                        part.zfill(3) for part in parts[:2] if len(part) <= 3
                    }
        if not found_here:
            raise BuildError(f"no FaceForensics++ lineages found in {path}")
        print(f"  {path}: {len(found_here)} FaceForensics++ originals refused")
        excluded |= found_here
    return excluded


def _load_cache(corpus_json: Path | None) -> dict[str, tuple[Path, str, int]]:
    """Bytes already fetched under the same pinned coordinates, from an earlier build.

    Keyed on `repo@revision#archive:member`, the full address of one immutable remote file, so a
    hit is the same bytes by construction. The digest and byte count are recomputed from the file
    anyway and the finished corpus goes back through the independence check, so this saves a
    download and verifies nothing less than a fresh fetch would.
    """
    if corpus_json is None or not corpus_json.is_file():
        return {}
    payload = json.loads(corpus_json.read_text(encoding="utf-8"))
    root = corpus_json.parent
    cache: dict[str, tuple[Path, str, int]] = {}
    for entry in payload["items"]:
        if entry["derivative_of"] is not None or entry["private"]:
            continue
        media = Path(entry["path"])
        if not media.is_absolute():
            media = root / entry["path"]
        if media.is_file():
            cache[entry["source"]] = (media, entry["sha256"], entry["bytes"])
    return cache


def _read_archive(source: RemoteSource):
    url = RESOLVE_URL.format(repo=source.repo, revision=source.revision, member=source.member)
    try:
        handle = io.BufferedReader(_RangeFile(url), buffer_size=BUFFER_BYTES)
        return zipfile.ZipFile(handle), url
    except Exception as error:  # noqa: BLE001 - urllib and zipfile fail in many ways
        raise BuildError(f"{source.source_id}: cannot read {url}: {error}") from error


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


def fetch_remote(
    source: RemoteSource,
    clips_dir: Path,
    excluded_ffpp: set[str],
    claimed: dict[str, set[str]],
    cache: dict[str, tuple[Path, str, int]] | None = None,
) -> tuple[list[CorpusItem], dict]:
    """Take this source's share of an archive, one corpus record per clip.

    Three filters run before selection and each is recorded in the returned screening record
    rather than applied silently: size; FaceForensics++ exclusion against the union of both
    prior corpora, for the families named after those originals; and the namespace claim, so
    families sharing a lineage namespace draw from disjoint blocks of it.

    Members are then chosen evenly spaced over what survives, sorted by name - deterministic
    given the pinned revision, with no seed to record.
    """
    rule = LINEAGE_RULES[source.lineage_rule]
    archive, url = _read_archive(source)
    with archive:
        all_members = [i for i in archive.infolist() if not i.is_dir()]
        sized = sorted(
            i.filename
            for i in all_members
            if i.filename.lower().endswith(".mp4")
            and i.file_size <= MAX_MEMBER_BYTES
            and not Path(i.filename).name.startswith("._")
            and (
                not source.member_prefixes
                or Path(i.filename).name.startswith(source.member_prefixes)
                or i.filename.startswith(source.member_prefixes)
            )
        )

        refused_ffpp: list[str] = []
        refused_claimed: list[str] = []
        eligible: list[tuple[str, str, frozenset]] = []
        seen_here: set[str] = set()
        pool = claimed.setdefault(source.lineage_namespace, set())

        for name in sized:
            stem = Path(name).stem
            if source.ffpp_named and _consumed_ids(stem) & excluded_ffpp:
                refused_ffpp.append(stem)
                continue
            key = rule(stem)
            consumed = _consumed_ids(stem) if source.ffpp_named else {key}
            if key in seen_here or seen_here & consumed:
                continue  # a second clip over a recording this source already offers
            if source.shares_namespace and (key in pool or pool & consumed):
                refused_claimed.append(stem)
                continue
            seen_here |= consumed
            seen_here.add(key)
            eligible.append((name, key, frozenset(consumed)))

        if len(eligible) < source.count:
            raise BuildError(
                f"{source.source_id}: wanted {source.count} lineages, "
                f"{len(eligible)} eligible after screening {len(sized)} members "
                f"({len(refused_ffpp)} refused as prior-task FaceForensics++ originals, "
                f"{len(refused_claimed)} already claimed in this namespace)"
            )

        chosen = _evenly_spaced(eligible, source.count)

        local = threading.local()

        def payload_for(entry: tuple[int, tuple[str, str, frozenset]]) -> tuple[int, bytes, bool]:
            index, (member, _key, _consumed) = entry
            coordinate = f"{source.repo}@{source.revision}#{source.member}:{member}"
            cached = (cache or {}).get(coordinate)
            if cached is not None:
                return index, cached[0].read_bytes(), True
            handle = getattr(local, "archive", None)
            if handle is None:
                handle, _ = _read_archive(source)
                local.archive = handle
            return index, handle.read(member), False

        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool_executor:
            fetched = dict(
                (index, (data, was_cached))
                for index, data, was_cached in pool_executor.map(
                    payload_for, list(enumerate(chosen))
                )
            )

        items = []
        reused = 0
        for index, (member, key, consumed) in enumerate(chosen):
            coordinate = f"{source.repo}@{source.revision}#{source.member}:{member}"
            payload, was_cached = fetched[index]
            reused += 1 if was_cached else 0
            clip_id = f"{source.source_id}_{index:03d}"
            (clips_dir / f"{clip_id}.mp4").write_bytes(payload)
            pool |= set(consumed)
            pool.add(key)
            items.append(
                CorpusItem(
                    clip_id=clip_id,
                    path=f"clips/{clip_id}.mp4",
                    label=source.label,
                    family=source.family,
                    stratum_primary=source.stratum,
                    strata=[source.stratum],
                    source_lineage_id=f"{source.lineage_namespace}:{key}",
                    base_media_id=clip_id,
                    derivative_of=None,
                    derivation="none",
                    source=coordinate,
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
        screening = {
            "source_id": source.source_id,
            "repo": source.repo,
            "revision": source.revision,
            "member": source.member,
            "url": url,
            "members_in_archive": len(all_members),
            "members_eligible_by_size": len(sized),
            "refused_as_prior_task_ffpp_original": len(refused_ffpp),
            "refused_as_claimed_lineage": len(refused_claimed),
            "distinct_lineages_available": len(eligible),
            "taken": len(items),
            "reused_from_cache": reused,
            "lineage_rule": source.lineage_rule,
            "lineage_namespace": source.lineage_namespace,
            "stratum": source.stratum,
            "split": source.split,
            "parent_datasets": list(source.parent_datasets),
            "member_prefixes": list(source.member_prefixes),
            "note": source.note,
        }
        print(
            f"  {source.source_id}: {len(items)} clips "
            f"({len(eligible)} eligible, {len(refused_ffpp)} ffpp-refused, "
            f"{len(refused_claimed)} claim-refused, {reused} reused from cache)"
        )
        return items, screening


def register_private(private_dir: Path) -> list[CorpusItem]:
    """Record the held-out real-world lineages without moving a byte of them."""
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
                family=REGRESSION_FAMILY,
                stratum_primary=source.stratum,
                strata=[source.stratum, "declared_reuse", "regression_fixture"],
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


def dedupe(items: list[CorpusItem], clips_dir: Path) -> tuple[list[CorpusItem], list[dict]]:
    """Drop clips whose bytes are already in the corpus, keeping the first occurrence."""
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


def build_derivatives(
    bases: list[CorpusItem],
    derivations: tuple[str, ...],
    clips_dir: Path,
) -> list[CorpusItem]:
    """Produce the degradations for a chosen set of base clips."""
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
    """Map every lineage to a split, from the split its source pool declared.

    The split is a property of the *source*, not of a hash or a slice: whole pools and whole
    families are held out, so the validation side carries genuine acquisition domains and
    manipulation methods the calibration side never saw. The private regression lineages ride
    with the validation pass so they are scored under the frozen threshold; they are removed
    from every rate computed there by family name, and they are never in the calibration split,
    so they cannot influence threshold selection.

    Raises rather than defaulting an unmapped lineage. A clip that quietly lands in a split
    because nobody named it is exactly the leak this task must not have.
    """
    declared: dict[str, str] = {}
    for source in (*GENUINE_SOURCES, *MANIPULATED_SOURCES, *GENERATED_SOURCES):
        if declared.setdefault(source.family, source.split) != source.split:
            raise BuildError(
                f"family {source.family!r} is declared on two different splits; a family is "
                "held out whole or not at all"
            )

    assignment: dict[str, str] = {}
    for item in items:
        lineage = item.source_lineage_id
        split = SPLIT_EVALUATION if item.family == REGRESSION_FAMILY else declared.get(item.family)
        if split is None:
            raise BuildError(f"no split rule covers lineage {lineage!r} (family {item.family!r})")
        previous = assignment.get(lineage)
        if previous is not None and previous != split:
            raise BuildError(
                f"lineage {lineage!r} is claimed by both splits; the namespace blocks are not "
                "disjoint and the build cannot proceed"
            )
        assignment[lineage] = split
    return assignment


def choose_derivative_bases(
    items: list[CorpusItem], assignment: dict[str, str]
) -> tuple[list[CorpusItem], list[CorpusItem]]:
    """Pick which validation lineages carry the degradations, spread and deterministically.

    Validation-side only: every headline rate is measured there. Genuine bases are taken evenly
    spaced within each genuine pool so the degradation strata span the acquisition domains
    rather than being an artefact of whichever pool sorts first, and both private lineages are
    always included - asking what a transcode does to them is most of the regression question.
    """
    genuine_by_family: dict[str, list[CorpusItem]] = {}
    private: list[CorpusItem] = []
    manipulated: list[CorpusItem] = []
    for item in items:
        if not item.is_base or assignment.get(item.source_lineage_id) != SPLIT_EVALUATION:
            continue
        if item.private:
            private.append(item)
        elif item.label == LABEL_REAL:
            genuine_by_family.setdefault(item.family, []).append(item)
        else:
            manipulated.append(item)

    genuine: list[CorpusItem] = list(private)
    per_family = max(
        1, (GENUINE_DERIVATIVE_LINEAGES - len(private)) // max(1, len(genuine_by_family))
    )
    for family in sorted(genuine_by_family):
        pool = sorted(genuine_by_family[family], key=lambda entry: entry.clip_id)
        genuine.extend(_evenly_spaced(pool, per_family))

    return genuine, _evenly_spaced(
        sorted(manipulated, key=lambda entry: entry.clip_id), MANIPULATED_DERIVATIVE_LINEAGES
    )


# The three acquisition domains the protocol names as core, and the gate is evaluated over.
CORE_STRATA = frozenset({STRATUM_PHONE, STRATUM_WEB_SOCIAL, STRATUM_BROADCAST})


# --- The genuine side ----------------------------------------------------------------------
#
# Eleven pools over five strata, and the three core strata carry two independent pools each so
# that "web and social video" is not one platform wearing one name. R7-T7's genuine side was
# five MAVOS-DD languages; R7-T8's was six pools it had never read; none of those eleven pools
# appears here.
#
# **How split assignment was decided, before anything was scored.** Whole pools are held out,
# and for each core stratum the two pools were placed on opposite sides so that both splits
# carry all three core domains. Where one archive supplies both sides - the broadcast pool is
# the only genuine television source in this mirror that contains faces at all - the upstream
# subsets are used as the boundary and the lineage is the dialogue, so no recording appears on
# both sides. That the same production supplies both broadcast blocks is the one intra-corpus
# source-family overlap in this corpus, it is declared here rather than discovered later, and
# the perceptual gate still runs across the boundary at the recording level.
#
# **What is deliberately kept and known to abstain.** The long-tail pools are consumer footage
# in which a face is present but is not the subject. R7-T8 found a pool of that kind abstaining
# 30 times out of 30 and reported it; keeping two here means the same finding stays visible
# instead of being designed out. They are declared NON-CORE, they carry no part of the coverage
# gate, and they are inside every headline genuine denominator, where their abstentions cost
# readable n rather than buying it.
GENUINE_SOURCES = (
    RemoteSource(
        "slovo", "34data/slovo",
        "c45cbe040e4983220f8d1fd290091859caa9392b", "data_001.zip",
        LABEL_REAL, "slovo_consumer_capture", STRATUM_PHONE, "consumer_camera_capture",
        "slovo", "stem", 140, SPLIT_CALIBRATION,
        "Slovo dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Slovo",),
        note="Russian sign-language video recorded by crowdsourced contributors on their own "
             "cameras and uploaded directly. Frontal, close, hand-held consumer capture - the "
             "acquisition path a clip takes when someone records themselves and sends the file.",
    ),
    RemoteSource(
        "smd_real", "34data/social-media-deepfakes-real",
        "f08398c1c3a3c0bee86ec241c34337283470c2e9", "v22-real-social-media-deepfakes_001.zip",
        LABEL_REAL, "social_media_real", STRATUM_WEB_SOCIAL, "web_social_download",
        "smd-real", "smd_uuid", 100, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Social Media Deepfakes (genuine half)",),
        note="Genuine video collected from social platforms for a deepfake benchmark: platform "
             "transcodes, mixed aspect ratios, mixed quality. The closest pool here to what "
             "actually arrives through a share link, and one of the validation side's two "
             "independent web platforms.",
    ),
    RemoteSource(
        "douyin_real", "34data/lanyuer",
        "b42c4b144b670d50783511fcb68602940e6ead23", "data_001.zip",
        LABEL_REAL, "douyin_short_video", STRATUM_WEB_SOCIAL, "web_social_download",
        "douyin", "stem", 60, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("lanyuer short-video collection",),
        note="Short-video platform uploads, named by the platform's own video id. A second "
             "platform in the same stratum, so the web/social figure is not one site's encoder.",
    ),
    RemoteSource(
        "bili_real", "34data/lanyuer-bili",
        "2d13242cea646eba2d1dc0a5281ce6d854d3f66d", "v22-real-lanyuer-bili_001.zip",
        LABEL_REAL, "bilibili_upload", STRATUM_WEB_SOCIAL, "web_social_download",
        "bili", "stem", 50, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("lanyuer short-video collection",),
        note="Uploads from a second Chinese platform, collected alongside the short-video pool "
             "above and therefore the same source family as it. Both sit on the calibration "
             "side for that reason: they are enough to measure a false-positive constraint on "
             "and too narrow to gate acquisition coverage on, which is why the validation side "
             "carries two genuinely independent web platforms instead.",
    ),
    RemoteSource(
        "meld_train", "34data/meld",
        "fa2dcc0782ab094a497620d0eb78da5cb5f76573", "data_001.zip",
        LABEL_REAL, "broadcast_series_train", STRATUM_BROADCAST, "broadcast_television",
        "meld", "meld_dialogue", 130, SPLIT_CALIBRATION,
        "MELD dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, parent_datasets=("MELD",),
        member_prefixes=("train_",),
        note="Utterance-level cuts from a broadcast television series: studio lighting, "
             "shot-reverse-shot editing, broadcast-grade encoding. The lineage is the dialogue, "
             "because several utterances of one dialogue are cuts of one continuous scene.",
    ),
    RemoteSource(
        "consumer_longtail_cal", "34data/real-life-violence",
        "372d48337be06c7abae6391c221707ae5be68327", "real-life-violence_0001.zip",
        LABEL_REAL, "consumer_longtail_scenes", STRATUM_LONG_TAIL, "consumer_camera_capture",
        "rlv-nonviolence", "stem", 40, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Real Life Violence Situations (non-violent half)",),
        member_prefixes=("NonViolence/",),
        note="Everyday consumer footage where a face is present but is not the subject: "
             "distance, motion, off-axis heads, heavy recompression. Declared non-core and kept "
             "because a pool that abstains is a finding, not a defect.",
    ),
    RemoteSource(
        "v_librasil", "34data/v-librasil",
        "508ac8a30b4c5002375dc29bf69c4328ea241e29", "v-librasil_0001.zip",
        LABEL_REAL, "librasil_consumer_capture", STRATUM_PHONE, "consumer_camera_capture",
        "v-librasil", "stem", 120, SPLIT_EVALUATION,
        "V-Librasil dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("V-Librasil",),
        note="Brazilian sign-language video recorded by contributors on their own cameras. The "
             "validation counterpart of the calibration consumer-capture pool: same acquisition "
             "path, different contributors, different language, different continent.",
    ),
    RemoteSource(
        "tiktok_real", "34data/tiktok-10m",
        "a9b65995121364409e31768f24977c318f620e80", "data_001.zip",
        LABEL_REAL, "tiktok_short_video", STRATUM_WEB_SOCIAL, "web_social_download",
        "tiktok-10m", "stem", 100, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("TikTok video collection",),
        note="Short-video platform uploads as downloaded: vertical framing, aggressive "
             "transcoding, overlays and effects, and often no face at all. Expected to abstain "
             "at a materially higher rate than the calibration side's web pools, and kept in "
             "the core stratum for exactly that reason - it is what web media is.",
    ),
    RemoteSource(
        "meld_devtest", "34data/meld",
        "fa2dcc0782ab094a497620d0eb78da5cb5f76573", "data_001.zip",
        LABEL_REAL, "broadcast_series_devtest", STRATUM_BROADCAST, "broadcast_television",
        "meld", "meld_dialogue", 100, SPLIT_EVALUATION,
        "MELD dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, parent_datasets=("MELD",),
        member_prefixes=("dev_", "test_"),
        note="The same broadcast series, upstream's held-out dialogues. Disjoint scenes from "
             "the calibration block by construction and by the shared-namespace claim, and the "
             "declared intra-corpus source-family overlap of this corpus: there is no second "
             "genuine television pool with faces in this mirror, and a broadcast stratum "
             "present on only one side would leave the coverage gate unmeasurable.",
    ),
    RemoteSource(
        "hallo3_real", "34data/v14-real-hallo3-training-data",
        "c100ff46c11617d5bfc6bdba72d842112506ec1c", "data_001.zip",
        LABEL_REAL, "high_resolution_talking_head", STRATUM_HIGH_RES, "curated_capture",
        "hallo3", "stem", 80, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Hallo3 training corpus (genuine)",),
        note="Well-lit, high-resolution, centred face video. Declared non-core: it is the "
             "easiest case for a spatial detector and the furthest from how media arrives, and "
             "it is here so the score distribution has its clean end represented.",
    ),
    RemoteSource(
        "phone_longtail_val", "34data/fall-video",
        "34eed9be4f962c6df17acde0c33eacc77813dcfd", "fall-video_0001.zip",
        LABEL_REAL, "phone_longtail_scenes", STRATUM_LONG_TAIL, "consumer_phone_capture",
        "fall-video", "stem", 40, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("fall-video",),
        note="Phone-captured room footage, named by the camera's own timestamp. Phone "
             "acquisition, but the face is small, distant and often turned away. Declared "
             "non-core for that reason and reported separately: it measures what happens to "
             "phone media the detector cannot read, which the core phone stratum cannot show.",
    ),
)


# --- The manipulated side ------------------------------------------------------------------
#
# Effort's checkpoint is trained on FaceForensics++ Deepfakes / FaceSwap / Face2Face /
# NeuralTextures. None of the nine families below is one of those, and none appears in R7-T5,
# R7-T7 or R7-T8. Seven are built on FaceForensics++ *originals*, which the exclusion set keeps
# disjoint from both predecessors, and two are built on other corpora entirely so that the
# intended-target population is not one benchmark's pair list wearing seven names.
#
# The calibration and validation sides carry disjoint families, so the operating point is
# chosen on manipulation methods the validation side never sees.
MANIPULATED_SOURCES = (
    RemoteSource(
        "fomm", "34data/fomm",
        "41ca893469fbd6005e9e521e967d856b36b6d294", "data_001.zip",
        LABEL_FACE_SWAP, "reenact_fomm", "face_reenactment", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="First Order Motion Model: a real face driven by another recording's motion. The "
             "identity is not replaced, which is the sub-type a spatial swap detector has least "
             "reason to catch.",
    ),
    RemoteSource(
        "tpsm", "34data/tpsm",
        "d673b6f6222303d5b2fa0b8600a20ebe9ccdc4b9", "data_001.zip",
        LABEL_FACE_SWAP, "reenact_tpsm", "face_reenactment", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="Thin-Plate Spline Motion Model reenactment.",
    ),
    RemoteSource(
        "lia", "34data/lia",
        "9dbfcbd6184d3b8a2ede68ebcd4da9d206d30124", "data_001.zip",
        LABEL_FACE_SWAP, "reenact_lia", "face_reenactment", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="Latent Image Animator reenactment, driven in a learned latent space.",
    ),
    RemoteSource(
        "smd_fake", "34data/social-media-deepfakes-fake",
        "f4b76fc47eb9e5bff66eceaa89185e51821f943d", "v22-fake-social-media-deepfakes_001.zip",
        LABEL_FACE_SWAP, "inthewild_social_deepfake", "face_manipulation_in_the_wild",
        "generated_manipulation",
        "smd-fake", "smd_uuid", 35, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Social Media Deepfakes (manipulated half)",),
        note="Deepfakes collected from social platforms rather than generated for a benchmark: "
             "unknown tools, real distribution paths, platform recompression on top. Placed on "
             "the same side as the genuine half of its own collection, so no manipulation can "
             "sit across the split boundary from the recording it may have been built on.",
    ),
    RemoteSource(
        "e4s", "34data/v13-e4s",
        "e60da488e2dd4d3818bf2b3e47764d241bfe887a", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_e4s", "face_swap", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 25, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="E4S, a fine-grained region-level identity swap. Unseen family at threshold "
             "selection, and the validation side's identity-replacement case.",
    ),
    RemoteSource(
        "mcnet", "34data/mcnet",
        "d6aaf7834841b2ebb73e1152341116f57fbd5c43", "data_001.zip",
        LABEL_FACE_SWAP, "reenact_mcnet", "face_reenactment", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="MCNet reenactment, with an explicit appearance memory. Unseen at selection.",
    ),
    RemoteSource(
        "hyperreenact", "34data/hyperreenact",
        "c649b3bf4da754aa7f6372660fff97e9f62fb94f", "data_001.zip",
        LABEL_FACE_SWAP, "reenact_hyperreenact", "face_reenactment", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="HyperReenact, reenactment through a hypernetwork-refined StyleGAN generator. "
             "Unseen at selection.",
    ),
    RemoteSource(
        "dagan", "34data/dagan",
        "85f16b3bd48d7257ffb3e442ee14a108bfa71cf8", "data_001.zip",
        LABEL_FACE_SWAP, "reenact_dagan", "face_reenactment", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        parent_datasets=("FaceForensics++ originals",),
        note="DaGAN, depth-aware reenactment. Unseen at selection.",
    ),
    RemoteSource(
        "lavdf", "34data/lav-df",
        "738669206e5be96171074e386d8a676d90433ffe", "data_001.zip",
        LABEL_FACE_SWAP, "lipsync_lavdf", "face_manipulation_lipsync", "generated_manipulation",
        "lav-df", "stem", 35, SPLIT_CALIBRATION,
        "LAV-DF dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("LAV-DF",),
        note="LAV-DF: short content-driven edits in which the mouth region of a real face is "
             "redriven and the rest of the recording is untouched. Built on a corpus neither "
             "predecessor read, and reported as its own family rather than folded into a swap "
             "rate. It is the calibration side's non-FaceForensics++ family, so the operating "
             "point is not chosen on one benchmark's pair list alone.",
    ),
)


# --- The out-of-distribution side ----------------------------------------------------------
#
# Fully generated video from four text-to-video systems no prior task used. Outside the
# checkpoint's training distribution by construction, reported separately, and never added into
# a primary-task figure. R7-T8 measured 92% abstention here and drew no conclusion from two
# readable observations; the same small allocation is kept so the same question can be asked
# without spending validation wall-clock on media that mostly does not contain a face.
GENERATED_SOURCES = (
    RemoteSource(
        "t2v_mochi1", "34data/v14-fake-mochi1",
        "3673de803d2fa0f052877c22b49940da2a20a7ba", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_mochi1", "generated", "generated_manipulation",
        "t2v-mochi1", "stem", 12, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Mochi-1 generations",),
    ),
    RemoteSource(
        "t2v_allegro", "34data/v14-fake-allegro",
        "72ef4dd61240ccf87de8be9e8fad7197887145fe", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_allegro", "generated", "generated_manipulation",
        "t2v-allegro", "stem", 12, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Allegro generations",),
    ),
    RemoteSource(
        "t2v_pyramidflow", "34data/v14-fake-pyramidflow",
        "710873a567b527247a888a508a255e8128750aba", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_pyramidflow", "generated", "generated_manipulation",
        "t2v-pyramidflow", "stem", 12, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Pyramid Flow generations",),
    ),
    RemoteSource(
        "t2v_gen3", "34data/videogen-rewardbench-gen3",
        "557aaf0a0cb5aa193ec7e7bf05c9327d17e8ef69",
        "v22-fake-videogen-rewardbench-gen3_001.zip",
        LABEL_SYNTHETIC, "t2v_gen3", "generated", "generated_manipulation",
        "t2v-gen3", "stem", 12, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, parent_datasets=("Gen-3 generations",),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_corpus_r7t9",
        description="Build the R7-T9 independent FPR-constrained operating point corpus.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prior-corpus",
        required=True,
        action="append",
        type=Path,
        help="a prior task's corpus.json, read to derive the FaceForensics++ exclusion set; "
             "repeatable, and every one given is screened against",
    )
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument(
        "--reuse-from",
        type=Path,
        help="an earlier corpus.json whose already-fetched bytes may be reused for members "
             "addressed by identical pinned coordinates; digests are recomputed regardless",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-derivatives", action="store_true")
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
    screening: list[dict] = []
    try:
        print("FaceForensics++ originals refused, by prior corpus:")
        excluded = _ffpp_excluded_ids(args.prior_corpus)
        print(f"  union refused: {len(excluded)}")

        cache = _load_cache(args.reuse_from)
        if cache:
            print(f"reuse cache: {len(cache)} members addressable from {args.reuse_from}")
        claimed: dict[str, set[str]] = {}
        items: list[CorpusItem] = []
        print("Genuine, independent pools:")
        for source in GENUINE_SOURCES:
            fetched, record = fetch_remote(source, clips, excluded, claimed, cache)
            items.extend(fetched)
            screening.append(record)
        print("Manipulated, new families:")
        for source in MANIPULATED_SOURCES:
            fetched, record = fetch_remote(source, clips, excluded, claimed, cache)
            items.extend(fetched)
            screening.append(record)
        print("Generated, out of distribution:")
        for source in GENERATED_SOURCES:
            fetched, record = fetch_remote(source, clips, excluded, claimed, cache)
            items.extend(fetched)
            screening.append(record)
        print("Declared re-use, regression lineages:")
        items.extend(register_private(args.private_dir))

        refused = [item for item in items if item.source_lineage_id in LEAKAGE_EXCLUDED_LINEAGES]
        if refused:
            items = [i for i in items if i.source_lineage_id not in LEAKAGE_EXCLUDED_LINEAGES]
            for item in refused:
                (clips / f"{item.clip_id}.mp4").unlink(missing_ok=True)
            print(
                f"Refused {len(refused)} clip(s) over "
                f"{len({i.source_lineage_id for i in refused})} lineage(s) implicated by the "
                "independence gate."
            )

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

    # What the coverage gate will be evaluated over, printed at build time so a shortfall is
    # visible before anything is scored rather than after.
    print("Genuine lineages by split and stratum (before abstention):")
    coverage: dict[tuple[str, str], set[str]] = {}
    for item in stamped:
        if item.label != LABEL_REAL or not item.is_base or item.family == REGRESSION_FAMILY:
            continue
        coverage.setdefault((item.split, item.stratum_primary), set()).add(
            item.source_lineage_id
        )
    for (split, stratum), lineages_seen in sorted(coverage.items()):
        core = stratum in CORE_STRATA
        print(
            f"  {split:11s} {stratum:26s} {len(lineages_seen):4d}"
            f"{'  [core]' if core else ''}"
        )

    manifest_sha = write_manifest(stamped, output / "manifest.csv")
    write_corpus(
        stamped,
        output / "corpus.json",
        extra={
            "task": "R7-T9",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha256": manifest_sha,
            "ffmpeg": derive.ffmpeg_version() if not args.skip_derivatives else "not invoked",
            "protocol": "scripts/eval/r7t9_protocol.json",
            "reuse_cache": str(args.reuse_from) if args.reuse_from else None,
            "core_strata": sorted(CORE_STRATA),
            "independence": {
                "prior_corpora_read": [str(path) for path in args.prior_corpus],
                "ffpp_original_ids_refused": sorted(excluded),
                "declared_reuse": [source.lineage_id for source in PRIVATE_SOURCES],
                "screening": screening,
                "source_family_declarations": {
                    source.source_id: {
                        "repo": source.repo,
                        "family": source.family,
                        "parent_datasets": list(source.parent_datasets),
                    }
                    for source in (
                        *GENUINE_SOURCES,
                        *MANIPULATED_SOURCES,
                        *GENERATED_SOURCES,
                    )
                },
            },
            "leakage_excluded_lineages": {
                "lineages": sorted(LEAKAGE_EXCLUDED_LINEAGES),
                "clips_refused": [item.clip_id for item in refused],
                "why": (
                    "confirmed by the predeclared 256-bit perceptual corroboration test as "
                    "sharing a recording with a prior task's corpus, or unresolved by it; the "
                    "check was not loosened, the media was removed"
                ),
            },
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
    print(f"wrote {output/'manifest.csv'} and {output/'corpus.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
