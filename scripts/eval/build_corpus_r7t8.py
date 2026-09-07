"""Assemble the R7-T8 independent corpus: new lineages, new families, same protocol.

    PYTHONPATH=scripts python3 scripts/eval/build_corpus_r7t8.py \
        --output-dir ../deepguard-corpus/r7t8 \
        --r7t7-corpus ../deepguard-corpus/r7t5/corpus.json \
        --private-dir ../deepguard-corpus-r7t5-private

R7-T7 derived a candidate operating threshold of 0.96612 on one calibration sample and
observed zero false positives at it on one evaluation sample. Both samples came from a single
corpus. This corpus exists so that the same question can be asked of media that corpus never
contained: whether the operating region is a property of Effort or a property of that sample.

**Everything here is new, and the build refuses to be otherwise.** The genuine side is drawn
from source pools R7-T7 did not read at all — three MAVOS-DD languages it never fetched, plus
web/social face video, high-resolution face video, a Chinese broadcast-derived pool, a
multilingual dubbing pool, and long-tail consumer/surveillance footage. The manipulated side is
six swap and autoencoder families and one lip-sync family, none of which appears in R7-T7 and
none of which is a FaceForensics++ manipulation Effort was trained on. The generated side is
four text-to-video generators R7-T7 did not use.

**The FaceForensics++ trap, and how it is closed.** Five of the swap families are built on
FaceForensics++ originals and name a clip `<target>_<source>.mp4`. R7-T7's calibration split
held forty FaceForensics++ swaps and thirty-nine FaceForensics++ reals, so a new *method*
applied to an original R7-T7 already used would be a new file over an old recording — the exact
leak this study cannot have. `_ffpp_excluded_ids` reads those ids out of the R7-T7 corpus and
`fetch_remote` refuses any member whose target or source id is among them. The five families
additionally draw from disjoint blocks of target ids, so `ffpp-orig:143` cannot appear twice
under two family names and be counted as two independent lineages.

**Splits are decided by name, before anything is scored.** `assign_splits` maps every lineage
explicitly, and whole source pools rather than slices of one pool are held out: the validation
side carries genuine pools and manipulated families the calibration side never saw. The
predeclared power target is stated over validation genuine lineages, so the validation side gets
the larger share by design.

**Private media never enters the repository or the corpus directory.** The two real-world
regression lineages are the one declared re-use in this task (`r7t8_protocol.json`), they are
referenced where they already live under `--private-dir`, and they are excluded from every
headline denominator by their family name.

Standard library only, plus ffmpeg on PATH for the derivatives. Nothing under `apps/` is read,
imported or written.
"""

from __future__ import annotations

import argparse
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

# Unchanged from R7-T5: one 300 MB member would dominate the download and the detector's wall
# clock while contributing exactly one lineage.
MAX_MEMBER_BYTES = 30 * 1024 * 1024

# The regression lineages are the only declared re-use in this corpus, and this family name is
# what keeps them out of the headline denominators. `analyze_effort_r7t8` keys on it.
REGRESSION_FAMILY = "real_world_regression"


class BuildError(Exception):
    """The corpus could not be assembled, so it was not assembled."""


# --- Lineage rules -------------------------------------------------------------------------
#
# A `source_lineage_id` names the underlying recording, and every pool names its members
# differently. Rather than one clever parser, each source declares which of these rules applies
# to it, so a reader can check the claim "this is one recording" per pool instead of trusting a
# regex to have guessed right on six naming conventions at once.


def _rule_trailing_int(stem: str) -> str:
    """`real_1017` -> `1017`. One archive's member number is its handle on one recording."""
    tail = stem.replace("-", "_").rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise BuildError(f"member {stem!r} does not end in a number; cannot derive a lineage")
    return tail


def _rule_stem(stem: str) -> str:
    """The whole stem. For pools where the prefix carries meaning (`ar_1` is not `en_1`)."""
    return stem


def _rule_ffpp_target(stem: str) -> str:
    """`001_870` -> `001`, the FaceForensics++ target original, which supplies the video.

    Two swaps of one target by two methods are two files over one recording. Keying the lineage
    on the target is what stops them being counted as two independent observations, and it is
    what the R7-T7 exclusion set is compared against.
    """
    parts = stem.split("_")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise BuildError(f"member {stem!r} is not a FaceForensics++ <target>_<source> name")
    return parts[0]


def _rule_dfdm_pair(stem: str) -> str:
    """`id0_id1_0000_4dfaker` -> `id0_id1_0000`, the identity pair and sequence.

    DFDM ships one source sequence through five autoencoder architectures under one name plus a
    method suffix. The recording is the part before the suffix.
    """
    parts = stem.split("_")
    if len(parts) < 4:
        raise BuildError(f"member {stem!r} is not a DFDM <pair>_<seq>_<method> name")
    return "_".join(parts[:3])


def _rule_strip_clip_index(stem: str) -> str:
    """`Abuse001_x264_0004` -> `Abuse001_x264`, the source video behind several cut clips.

    This pool is cut from long videos, several clips per source. Without this rule four cuts of
    one recording would be four genuine lineages and the false-positive denominator would be
    inflated by the cutting, which is precisely the error the lineage unit exists to prevent.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _rule_ffpp_leading(stem: str) -> str:
    """`001_bLGTrz1Zolk_50015` -> `001`, the FaceForensics++ original the clip was built on.

    FaceForensics++ originals are YouTube clips, and some derived corpora keep the original id
    followed by the YouTube id and a frame range rather than the `<target>_<source>` pair form.
    The leading integer is still the FaceForensics++ original, which is what has to stay disjoint
    from R7-T7. This rule exists because assuming otherwise put a lip-sync clip built on
    FaceForensics++ original 001 into the validation split while R7-T7 held that same recording -
    the perceptual pass caught it, and this is the fix.
    """
    head = stem.split("_", 1)[0]
    if not head.isdigit():
        raise BuildError(f"member {stem!r} does not begin with a FaceForensics++ original id")
    return head


def _consumed_ids(stem: str) -> set[str]:
    """Every FaceForensics++ original a member's name refers to, zero-padded.

    A swap file names the recording it took the video from and the recording it took the face
    from. Both are spent by that clip: another family reusing either one is not sampling an
    independent recording, whichever position it puts it in.
    """
    consumed: set[str] = set()
    for token in stem.split("_"):
        # FaceForensics++ originals are numbered 000-999. A longer run of digits in one of these
        # names is a frame range or a YouTube artefact, not an original, and must not be claimed
        # as one.
        if token.isdigit() and len(token) <= 3:
            consumed.add(token.zfill(3))
    return consumed


LINEAGE_RULES = {
    "trailing_int": _rule_trailing_int,
    "stem": _rule_stem,
    "ffpp_target": _rule_ffpp_target,
    "dfdm_pair": _rule_dfdm_pair,
    "strip_clip_index": _rule_strip_clip_index,
    "ffpp_leading": _rule_ffpp_leading,
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
    # Sources sharing a namespace draw from disjoint blocks of it, claimed in build order.
    shares_namespace: bool = False
    # Families whose members are named after FaceForensics++ originals are screened against the
    # ids R7-T7 used. Everything else has no relationship to that numbering.
    ffpp_named: bool = False
    note: str = ""


# --- The genuine side ----------------------------------------------------------------------
#
# Six pools, chosen so that "genuine media" here is not one recording condition wearing six
# names. R7-T7's genuine side was five MAVOS-DD languages and one benchmark; a false-positive
# rate measured on that is a claim about talking heads uploaded to one platform. These add
# web/social face video at consumer quality (CelebV-HQ), high-resolution studio-grade face video
# (DH-FaceVid-1K), a Chinese pool assembled from broadcast and short-video sources (FMFCC-V),
# multilingual dubbing-source footage (PolyglotFake reals), and long-tail consumer and
# surveillance footage where the face is small, off-axis or barely lit (UCF-Crime cuts). The
# three MAVOS-DD languages are new pools in new namespaces, and they are the bridge that makes
# the R7-T7 comparison a comparison rather than an apples-to-oranges swap of the whole domain.
GENUINE_SOURCES = (
    RemoteSource(
        "mavos_real_ro", "34data/MAVOS-DD-romanian_real",
        "a5d97027f78d4ad405702e27e80b3f7d971137a3", "data_001.zip",
        LABEL_REAL, "mavos_real_ro", "genuine_in_the_wild", "in_the_wild_upload",
        "mavos-real:ro", "trailing_int", 90, SPLIT_CALIBRATION,
        "MAVOS-DD dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="Romanian in-the-wild uploads. A MAVOS-DD language R7-T7 did not fetch.",
    ),
    RemoteSource(
        "fmfcc_real", "34data/fmfcc-v-real",
        "aa9b8eb72ea687327971351acf47d8fef2d9bda1", "data_001.zip",
        LABEL_REAL, "fmfcc_v_real", "genuine_in_the_wild", "in_the_wild_upload",
        "fmfcc-v-real", "trailing_int", 50, SPLIT_CALIBRATION,
        "FMFCC-V dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="Chinese genuine pool from the FMFCC-V challenge: broadcast and short-video "
             "sources, heavier and more varied compression than the MAVOS-DD pools.",
    ),
    RemoteSource(
        "polyglot_real", "34data/polyglotfake-real",
        "41a72206d3df0c06228577f7e84938ceb3a847e5", "data_001.zip",
        LABEL_REAL, "polyglotfake_real", "genuine_in_the_wild", "in_the_wild_upload",
        "polyglotfake-real", "stem", 40, SPLIT_CALIBRATION,
        "PolyglotFake dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="The genuine half of a multilingual dubbing corpus. Members are language-prefixed "
             "and the prefix is part of the identity, so the lineage is the whole stem.",
    ),
    RemoteSource(
        "mavos_real_ru", "34data/MAVOS-DD-russian_real",
        "2a7cfd8a253f8b0f89eb8aac4c377554248956b7", "data_001.zip",
        LABEL_REAL, "mavos_real_ru", "genuine_in_the_wild", "in_the_wild_upload",
        "mavos-real:ru", "trailing_int", 90, SPLIT_EVALUATION,
        "MAVOS-DD dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="Russian in-the-wild uploads. A MAVOS-DD language R7-T7 did not fetch.",
    ),
    RemoteSource(
        "mavos_real_es", "34data/MAVOS-DD-spanish_real",
        "3a8340cc8ad58e2b963a39c6e2c9dc1a270b8855", "data_001.zip",
        LABEL_REAL, "mavos_real_es", "genuine_in_the_wild", "in_the_wild_upload",
        "mavos-real:es", "trailing_int", 90, SPLIT_EVALUATION,
        "MAVOS-DD dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="Spanish in-the-wild uploads. A MAVOS-DD language R7-T7 did not fetch.",
    ),
    RemoteSource(
        "celebv_hq", "34data/v15-human-vid-celebv-hq",
        "6d498baa95b0b2b4c1791c63fbed1907ec459a9e", "data_001.zip",
        LABEL_REAL, "celebv_hq", "genuine_web_social", "web_social_download",
        "celebv-hq", "trailing_int", 120, SPLIT_EVALUATION,
        "CelebV-HQ dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="Face video scraped from the open web and re-encoded to consumer quality. The "
             "closest pool here to what actually arrives through a share link.",
    ),
    RemoteSource(
        "dh_facevid", "34data/v14-real-dh-facevid-1k-0001",
        "90a1c7d8145816db90e5e72aac8b338370dab376", "data_001.zip",
        LABEL_REAL, "dh_facevid_1k", "genuine_high_resolution", "curated_capture",
        "dh-facevid-1k:0001", "trailing_int", 70, SPLIT_EVALUATION,
        "DH-FaceVid-1K dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="High-resolution, well-lit face video. The opposite end of the quality range from "
             "the surveillance pool, and the case where a spatial detector has the most to see.",
    ),
    RemoteSource(
        "ytclips_real", "34data/v14-real-youtubeclips",
        "80b0c1fde56cd52013827990d202636af027ded9", "data_001.zip",
        LABEL_REAL, "ucf_crime_clips", "genuine_long_tail", "web_social_download",
        "ucf-crime", "strip_clip_index", 30, SPLIT_EVALUATION,
        "UCF-Crime dataset terms (research use)", "public research mirror, pinned revision",
        False,
        note="Consumer and surveillance footage: small faces, off-axis, low light, heavy "
             "compression. Expected to produce abstentions rather than readings, and included "
             "for exactly that reason - the abstention rate on real-world footage is a finding.",
    ),
)


# --- The manipulated side ------------------------------------------------------------------
#
# Effort's checkpoint is trained on FaceForensics++ Deepfakes/FaceSwap/Face2Face/NeuralTextures.
# None of the seven families below is one of those, and none appears in R7-T7. Five are
# generator-based swaps applied to FaceForensics++ *originals* (which the exclusion set keeps
# disjoint from R7-T7's), four are DFDM autoencoder architectures over identity pairs, and one
# is lip-sync, which manipulates a real face without replacing it and is the sub-type a spatial
# swap detector has the least reason to catch. The calibration and validation sides carry
# disjoint families, so the validation side is answering about methods the threshold never saw.
MANIPULATED_SOURCES = (
    RemoteSource(
        "facedancer", "34data/facedancer",
        "e94c502456fefe7752483e047ecf83cd5f26f4a6", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_facedancer", "face_swap", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        note="FaceDancer, a single-stage adaptive feature-fusion swap.",
    ),
    RemoteSource(
        "fsgan", "34data/fsgan",
        "213a04799e8b7739f3590901fff7b6ce61e79825", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_fsgan", "face_swap", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        note="FSGAN, subject-agnostic swap and reenactment.",
    ),
    RemoteSource(
        "dfdm_dfaker", "34data/v15-human-vid-dfdm_cfr23-dfaker",
        "39e97430005bd8f3d236c190950004c86370aeb2", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_dfdm_dfaker", "face_swap", "generated_manipulation",
        "dfdm", "dfdm_pair", 15, SPLIT_CALIBRATION,
        "DFDM dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True,
        note="DFDM / Dfaker autoencoder architecture, CRF 23.",
    ),
    RemoteSource(
        "dfdm_iae", "34data/v15-human-vid-dfdm_cfr23-iae",
        "195d7c17731389aba734633eaeb6b5f358649c38", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_dfdm_iae", "face_swap", "generated_manipulation",
        "dfdm", "dfdm_pair", 15, SPLIT_CALIBRATION,
        "DFDM dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True,
        note="DFDM / IAE autoencoder architecture, CRF 23.",
    ),
    RemoteSource(
        "blendface", "34data/blendface",
        "20a5fdd858d71fdbbcf686c3340fc572cddebcd8", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_blendface", "face_swap", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        note="BlendFace, identity-disentangled swap. Unseen family at threshold selection.",
    ),
    RemoteSource(
        "simswap", "34data/swimswap",
        "97fe5311c0a1bcd97e3cc514715e88a3a674fda3", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_simswap", "face_swap", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        note="SimSwap, arbitrary-identity swap. Unseen family at threshold selection.",
    ),
    RemoteSource(
        "mobileswap", "34data/mobileswap",
        "d4655fad090be2775d088b80c29bc243230d6578", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_mobileswap", "face_swap", "generated_manipulation",
        "ffpp-orig", "ffpp_target", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        note="MobileSwap, a lightweight on-device-class swap. Unseen family at selection, and "
             "the cheapest of the seven to run at scale, which is why it is worth asking about.",
    ),
    RemoteSource(
        "dfdm_dfl_h128", "34data/v15-human-vid-dfdm_cfr23-dfl-h128",
        "e3ee18eb8bd848033cb28f7475665a262281a151", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_dfdm_dfl_h128", "face_swap", "generated_manipulation",
        "dfdm", "dfdm_pair", 15, SPLIT_EVALUATION,
        "DFDM dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True,
        note="DFDM / DeepFaceLab H128 architecture, CRF 23. Unseen at threshold selection.",
    ),
    RemoteSource(
        "dfdm_lightweight", "34data/v15-human-vid-dfdm_cfr23-lightweight",
        "dfa7d13ccbf8118a35dcd1a79cf439b2ab2ab2ab", "data_001.zip",
        LABEL_FACE_SWAP, "faceswap_dfdm_lightweight", "face_swap", "generated_manipulation",
        "dfdm", "dfdm_pair", 15, SPLIT_EVALUATION,
        "DFDM dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True,
        note="DFDM / Lightweight architecture, CRF 23. Unseen at threshold selection.",
    ),
    RemoteSource(
        "wav2lip", "34data/wav2lip",
        "73b563d8f30fd3470e5a4e10fd09d309eb2b23fe", "data_001.zip",
        LABEL_FACE_SWAP, "lipsync_wav2lip", "face_manipulation_lipsync",
        "generated_manipulation",
        "ffpp-orig", "ffpp_leading", 20, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False, shares_namespace=True, ffpp_named=True,
        note="Wav2Lip: the mouth region of a real face redriven by audio, the rest of the "
             "recording untouched. Face manipulation, but not a swap, and reported as its own "
             "family rather than folded into the swap rate. Built on FaceForensics++ originals "
             "under a `<original>_<youtube id>_<range>` name, so it is screened against R7-T7's "
             "originals and shares the FaceForensics++ namespace like every other family that "
             "consumes one.",
    ),
)


# --- The out-of-distribution side ----------------------------------------------------------
#
# Fully generated video from four text-to-video systems R7-T7 did not use. Outside the
# checkpoint's training distribution by construction, reported separately, and never added into
# a primary-task figure.
GENERATED_SOURCES = (
    RemoteSource(
        "t2v_hunyuan", "34data/gen-videos-hunyuanvideo",
        "063249f0fab03017fcbe012997afab36d53886ef", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_hunyuanvideo", "generated", "generated_manipulation",
        "t2v-hunyuan", "stem", 12, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
    RemoteSource(
        "t2v_ltx", "34data/v14-fake-ltxvideo",
        "4a2f179581177a0fa150fa311b76278fe4707e4a", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_ltxvideo", "generated", "generated_manipulation",
        "t2v-ltx", "stem", 12, SPLIT_CALIBRATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
    RemoteSource(
        "t2v_wan2", "34data/gen-videos-wan2",
        "660cd80d16ac7cc962d892461e6b1e506809a3c7", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_wan2", "generated", "generated_manipulation",
        "t2v-wan2", "stem", 12, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
    RemoteSource(
        "t2v_cogvideox", "34data/v14-fake-cogvideox",
        "fdff2865d8e9d78905541bdc916d6591848152f5", "data_001.zip",
        LABEL_SYNTHETIC, "t2v_cogvideox", "generated", "generated_manipulation",
        "t2v-cogvideox", "stem", 12, SPLIT_EVALUATION,
        "source dataset terms (research use)", "public research mirror, pinned revision",
        False,
    ),
)


@dataclass(frozen=True)
class PrivateSource:
    """A locally held, non-redistributable lineage, named by what it is rather than by whom."""

    lineage_id: str
    filename: str
    stratum: str
    acquisition_type: str
    note: str


# The one declared re-use in this corpus. Carried forward from R7-T5/R7-T7 because the task
# requires these two lineages to be re-read under the newly derived threshold, and excluded from
# every headline denominator by their family name.
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

# How many validation-split lineages carry the constructed degradations. Well below what is
# available, on purpose: a derivative adds no lineage to the headline denominator, so every one
# of them is bought with detector wall-clock and nothing else.
GENUINE_DERIVATIVE_LINEAGES = 24
MANIPULATED_DERIVATIVE_LINEAGES = 12


def _ffpp_excluded_ids(r7t7_corpus: Path) -> set[str]:
    """Every FaceForensics++ original id R7-T7 touched, from its own corpus record.

    Read out of the artifact rather than transcribed, so it cannot drift from what R7-T7
    actually held. `ffpp_deepfakes:ffpp_dev_Deepfakes_100_077` contributes both `100` and `077`
    - the target and the source - because a later method swapping 077 onto some other target is
    still a manipulation of a recording R7-T7 used. `ffpp_real:ffpp_dev_real_000` contributes
    `000`.
    """
    payload = json.loads(r7t7_corpus.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    for entry in payload["items"]:
        lineage = entry["source_lineage_id"]
        if not lineage.startswith("ffpp_"):
            continue
        for token in lineage.split(":", 1)[1].split("_"):
            if token.isdigit():
                excluded.add(token.zfill(3))
    if not excluded:
        raise BuildError(f"no FaceForensics++ lineages found in {r7t7_corpus}")
    return excluded


def _load_cache(corpus_json: Path | None) -> dict[str, tuple[Path, str, int]]:
    """Bytes already fetched under the same pinned coordinates, from an earlier build.

    Keyed on `repo@revision#archive:member`, which is the full address of one immutable remote
    file, so a hit is the same bytes by construction and not by hope. The digest and byte count
    are recomputed from the file anyway and written into the new corpus record, and the finished
    corpus goes back through the independence check, so this saves a download and verifies
    nothing less than a fresh fetch would.

    Exists because the first build of this corpus found a real leak in one manipulated family.
    Re-fetching six hundred untouched genuine clips to fix twenty lip-sync ones would have cost
    an hour of network time and changed no byte of them.
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
    url = RESOLVE_URL.format(
        repo=source.repo, revision=source.revision, member=source.member
    )
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

    Three filters run before selection, and each is recorded in the returned screening record
    rather than applied silently:

    - size, as in R7-T5;
    - **FaceForensics++ exclusion**, for the families named after those originals: a member is
      refused if either its target or its source id is one R7-T7 used;
    - **namespace claim**, for the families that share a lineage namespace: a member is refused
      if another family in this build already took that lineage, so the five swap families draw
      from disjoint blocks of target ids and the four DFDM families from disjoint identity
      pairs.

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
            if i.filename.lower().endswith(".mp4") and i.file_size <= MAX_MEMBER_BYTES
        )

        refused_ffpp: list[str] = []
        refused_claimed: list[str] = []
        eligible: list[tuple[str, str]] = []
        seen_here: set[str] = set()
        pool = claimed.setdefault(source.lineage_namespace, set())

        for name in sized:
            stem = Path(name).stem
            if source.ffpp_named and _consumed_ids(stem) & excluded_ffpp:
                refused_ffpp.append(stem)
                continue
            key = rule(stem)
            # A swap consumes BOTH originals, not only the one the lineage is keyed on.
            # FaceForensics++ pairs videos it selected for visual similarity and pairs them
            # reciprocally, so `433_498` and `498_433` are two clips over the same two chosen-
            # to-be-alike recordings. Claiming the target alone let one of those land in
            # calibration and the other in validation - which the perceptual pass then found at
            # Hamming distance 0. Claiming every original a member names closes it.
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
                f"({len(refused_ffpp)} refused as R7-T7 FaceForensics++ originals, "
                f"{len(refused_claimed)} already claimed in this namespace)"
            )

        chosen = _evenly_spaced(eligible, source.count)
        items = []
        reused = 0
        for index, (member, key, consumed) in enumerate(chosen):
            coordinate = f"{source.repo}@{source.revision}#{source.member}:{member}"
            cached = (cache or {}).get(coordinate)
            clip_id = f"{source.source_id}_{index:03d}"
            if cached is not None:
                payload = cached[0].read_bytes()
                reused += 1
            else:
                payload = archive.read(member)
            (clips_dir / f"{clip_id}.mp4").write_bytes(payload)
            lineage = f"{source.lineage_namespace}:{key}"
            pool |= consumed
            pool.add(key)
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
            "refused_as_r7t7_ffpp_original": len(refused_ffpp),
            "refused_as_claimed_lineage": len(refused_claimed),
            "distinct_lineages_available": len(eligible),
            "taken": len(items),
            "reused_from_cache": reused,
            "lineage_rule": source.lineage_rule,
            "lineage_namespace": source.lineage_namespace,
            "split": source.split,
            "note": source.note,
        }
        print(
            f"  {source.source_id}: {len(items)} clips "
            f"({len(eligible)} eligible, {len(refused_ffpp)} ffpp-refused, "
            f"{len(refused_claimed)} claim-refused, {reused} reused from cache)"
        )
        return items, screening


def register_private(private_dir: Path) -> list[CorpusItem]:
    """Record the two held-out real-world lineages without moving a byte of them."""
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
                strata=[source.stratum, "declared_reuse"],
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

    The split is a property of the *source*, not of a hash or a slice: whole pools are held out,
    so the validation side carries genuine domains and manipulated families the calibration side
    never saw. That is what makes this a stability study rather than a re-read of one sample
    under two names. The private regression lineages are validation, as in R7-T7.

    Raises rather than defaulting an unmapped lineage. A clip that quietly lands in the
    validation set because nobody named it is exactly the leak this task must not have.
    """
    declared: dict[str, str] = {}
    for source in (*GENUINE_SOURCES, *MANIPULATED_SOURCES, *GENERATED_SOURCES):
        declared[source.family] = source.split

    assignment: dict[str, str] = {}
    for item in items:
        lineage = item.source_lineage_id
        if item.family == REGRESSION_FAMILY:
            split = SPLIT_EVALUATION
        else:
            split = declared.get(item.family)
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

    Validation-side only: the degradation strata exist to be measured, and every rate in this
    task is measured on the validation side. Genuine bases are taken evenly spaced within each
    genuine pool so the degradation strata span the domains rather than being an artefact of
    whichever pool sorts first, and both private lineages are always included - asking what a
    transcode does to them is most of the regression question.
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

    manipulated_pool = sorted(manipulated, key=lambda entry: entry.clip_id)
    return genuine, _evenly_spaced(manipulated_pool, MANIPULATED_DERIVATIVE_LINEAGES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_corpus_r7t8",
        description="Build the R7-T8 independent calibration-stability corpus.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--r7t7-corpus",
        required=True,
        type=Path,
        help="the R7-T7 corpus.json, read to derive the FaceForensics++ exclusion set",
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
        excluded = _ffpp_excluded_ids(args.r7t7_corpus)
        print(f"R7-T7 FaceForensics++ originals refused: {len(excluded)}")

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
            "task": "R7-T8",
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest_sha256": manifest_sha,
            "ffmpeg": derive.ffmpeg_version() if not args.skip_derivatives else "not invoked",
            "protocol": "scripts/eval/r7t8_protocol.json",
            "reuse_cache": str(args.reuse_from) if args.reuse_from else None,
            "independence": {
                "r7t7_corpus_read": str(args.r7t7_corpus),
                "ffpp_original_ids_refused": sorted(excluded),
                "declared_reuse": [source.lineage_id for source in PRIVATE_SOURCES],
                "screening": screening,
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
