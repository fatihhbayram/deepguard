"""Assemble the R4-T1 calibration corpus, reproducibly, from pinned sources.

    python3 scripts/calibration/fetch_corpus.py --output-dir ../deepguard-corpus/r4t1_calibration

R4 calibrates two detectors that answer different questions — NVIDIA's synthetic-video
detector looks for *generated video*, EfficientNet-B7 looks for a *manipulated face* — so
a corpus that carries only one of those manipulation kinds cannot measure either one
honestly, and can say nothing at all about what the pair does when they disagree. This
script builds one that carries both, plus the genuine media any false-positive figure has
to be measured against.

**Everything here is pinned.** Each remote source names a dataset repository *and the
commit that was read*, never a branch: a mirror that gains or loses a file later produces
a different corpus, and a calibration whose corpus cannot be reconstructed is a number
without a measurement behind it. Member selection inside an archive is deterministic too —
members sorted by name, `.mp4` only, taken at evenly spaced indices — so the same
revision always yields the same clips in the same order.

**Only the bytes that are needed are downloaded.** The mirrors publish ~900 MB zip
archives; this reads the central directory and then each selected member over HTTP range
requests, so a ten-clip sample costs tens of megabytes rather than a gigabyte. `zipfile`
does the parsing — the only thing added is a seekable file object backed by `Range`.

**The local FaceForensics++ clips are copied in, not referenced.** The corpus is one
directory with one digest over its own bytes; a manifest pointing at clips that live in
another corpus's directory would let a later edit there change what this calibration
measured without changing anything the artifact records.

Standard library only, and it writes nothing outside `--output-dir`.
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# HuggingFace serves dataset files from this path shape. The revision goes in the URL, so
# the bytes fetched are the bytes of that commit even if the branch has moved since.
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{member}"

# Members larger than this are skipped during selection. A calibration sample does not
# need the corpus's longest videos, and NVIDIA's detector is streamed the whole file: one
# 300 MB member would dominate both the download and the provider call time while
# contributing exactly one data point.
MAX_MEMBER_BYTES = 30 * 1024 * 1024

# Read granularity for the range-backed file object. The zip central directory is read in
# small pieces and the member payloads in large ones; a 1 MiB buffer keeps the number of
# HTTP requests down without holding an archive in memory.
BUFFER_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RemoteSource:
    """One mirrored archive, and how many clips to take from it.

    `label` is the benchmark's ground-truth vocabulary (`real`, `face_swap`,
    `synthetic`). `family` names the generator or manipulation technique, and `stratum`
    the domain the clip belongs to. The last two are extra manifest columns — the
    benchmark harness ignores unknown columns — and they exist because a pooled figure
    over a corpus like this one hides the finding that matters: the two detectors fail on
    different families, and only a per-family breakdown shows it.
    """

    source_id: str
    repo: str
    revision: str
    member: str
    label: str
    family: str
    stratum: str
    count: int


# The corpus, as a table rather than as prose.
#
# Three strata, chosen so that each detector is measured both inside and outside the
# domain it was built for:
#
# - `genuine_face`   real talking-head footage, in five languages plus the FF++ source
#                    videos. Every false-positive figure in the calibration is measured
#                    here and nowhere else.
# - `face_swap`      a face composited into otherwise authentic footage. This is
#                    EfficientNet-B7's domain, and P7-T2 measured it as the domain NVIDIA's
#                    detector is weakest in — three swap families scored *below* the
#                    genuine median. Both mirror families here (`inswapper`, `roop`) are
#                    from that group on purpose: a calibration should include the cases
#                    that are known to be hard, not avoid them.
# - `generated`      video that was generated rather than edited — audio-driven talking
#                    heads, portrait reenactment, and text-to-video. This is NVIDIA's
#                    domain. The T2V sets carry essentially no faces, which is not a flaw
#                    in the sample: it is the clearest available case of evidence one
#                    detector can read and the other cannot, and R4-T2 has to decide what
#                    the risk engine says about exactly that.
REMOTE_SOURCES = (
    RemoteSource("mavos_real_en", "34data/mavos-dd-english_real",
                 "02cd6215882b2c18117ba133874cc471a79a4a9e", "data_001.zip",
                 "real", "mavos_real", "genuine_face", 3),
    RemoteSource("mavos_real_de", "34data/mavos-dd-german_real",
                 "846ef3709843fbff294214b10cf0cebd79371d72", "data_001.zip",
                 "real", "mavos_real", "genuine_face", 3),
    RemoteSource("mavos_real_hi", "34data/mavos-dd-hindi_real",
                 "53cb91016b0435a34c99497cf21f70761b8651ea", "data_001.zip",
                 "real", "mavos_real", "genuine_face", 3),
    RemoteSource("mavos_real_zh", "34data/mavos-dd-mandarin_real",
                 "274199e0f50dfd41c936bba0de5e9b593de3bba1", "data_001.zip",
                 "real", "mavos_real", "genuine_face", 3),
    RemoteSource("mavos_real_ar", "34data/mavos-dd-arabic-real",
                 "542a345fb9d21af63b0b04055dc2deaaf8851814", "data_001.zip",
                 "real", "mavos_real", "genuine_face", 3),
    RemoteSource("inswapper_en", "34data/v15-human-vid-mavos-dd-english_inswapper",
                 "30433869a189a03acab231ab89836dfe11331970", "data_001.zip",
                 "face_swap", "faceswap_inswapper", "face_swap", 5),
    RemoteSource("roop_en", "34data/v15-human-vid-mavos-dd-english_roop",
                 "9366303b174d3c10aad2faa806ed3835f29f1a3c", "data_001.zip",
                 "face_swap", "faceswap_roop", "face_swap", 5),
    RemoteSource("echomimic_en", "34data/v15-human-vid-mavos-dd-english_echomimic",
                 "f56bf178a9521207f5ebc546ac47b00114901e4e", "data_001.zip",
                 "synthetic", "talkinghead_echomimic", "generated", 10),
    RemoteSource("sonic_en", "34data/MAVOS-DD-english_sonic",
                 "15557eea4b9aa0d13af944ad335942211c809dfd", "data_001.zip",
                 "synthetic", "talkinghead_sonic", "generated", 10),
    RemoteSource("memo_de", "34data/MAVOS-DD-german_memo",
                 "e6903fdfb063f32e5e9d32ad28539e59a78b7807", "data_001.zip",
                 "synthetic", "talkinghead_memo", "generated", 10),
    RemoteSource("liveportrait_en", "34data/v15-human-vid-mavos-dd-english_liveportrait",
                 "e344ba113bd728897390871fd9103809b8957080", "data_001.zip",
                 "synthetic", "reenactment_liveportrait", "generated", 10),
    RemoteSource("veo3", "34data/gen-videos-veo3",
                 "dad55b648cf7da90584e69245b8eb22da0929ba4", "data_001.zip",
                 "synthetic", "t2v_veo3", "generated", 5),
    RemoteSource("sora2", "34data/gen-videos-sora2",
                 "82a7b46ee58f417a233b7e89e705b5065039df3a", "data_001.zip",
                 "synthetic", "t2v_sora2", "generated", 5),
    RemoteSource("kling", "34data/gen-videos-kling",
                 "07d2e8ff55f94b745749bd94e9af80b6546d272a", "data_001.zip",
                 "synthetic", "t2v_kling", "generated", 5),
)

# The FaceForensics++ clips already on this machine, from R3-T1. Both splits are taken:
# the benchmark's train/test division was there to keep an operating point off the data it
# was measured on, and this task selects thresholds over the whole corpus at once, so
# holding a split back here would only shrink the sample. Family comes from the file name
# prefix — `Deepfakes` is the autoencoder swap, `FaceSwap` the graphics-based one, and the
# two are different techniques that FF++ keeps separate for exactly this reason.
LOCAL_SPLITS = (
    ("ffpp_dev", "ff++_c23_r3t1"),
    ("ffpp_test", "ff++_c23_test_r3t1"),
)
LOCAL_FAMILIES = {
    "Deepfakes": ("face_swap", "ffpp_deepfakes", "face_swap"),
    "FaceSwap": ("face_swap", "ffpp_faceswap", "face_swap"),
    "real": ("real", "ffpp_real", "genuine_face"),
}


class FetchError(Exception):
    """A source could not be read, so the corpus was not built."""


class _RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP `Range` requests.

    `zipfile` needs `seek`/`read` and will happily work over this: it reads the end of the
    archive to find the central directory, then jumps straight to the members it is asked
    for. The alternative is downloading 900 MB to keep 20 of them.
    """

    def __init__(self, url: str):
        self.position = 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is None:
                raise FetchError(f"{url} does not report a length; cannot range-read it")
            self.size = int(length)
            # HuggingFace redirects file downloads to a CDN. Following the redirect once
            # here and addressing the final URL afterwards keeps every subsequent range
            # request on the host that actually serves the bytes.
            self.url = response.url

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        else:
            self.position = self.size + offset
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        last = self.position + size - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{last}"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = response.read()
        self.position += len(payload)
        return payload

    def readinto(self, buffer) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)


def select_members(names_and_sizes: list[tuple[str, int]], count: int) -> list[str]:
    """Choose `count` members from an archive, deterministically and spread out.

    Sorted by name and taken at evenly spaced indices rather than at random or from the
    head: an archive's first members are often one recording session, and a random sample
    would need a seed recorded somewhere for the corpus to be rebuildable. This needs
    neither — the archive revision alone determines the result.
    """
    eligible = sorted(
        name for name, size in names_and_sizes
        if name.lower().endswith(".mp4") and size <= MAX_MEMBER_BYTES
    )
    if not eligible:
        return []
    if count >= len(eligible):
        return eligible
    # Evenly spaced over the whole range, including both ends.
    step = (len(eligible) - 1) / (count - 1) if count > 1 else 0
    indices = sorted({int(round(index * step)) for index in range(count)})
    return [eligible[index] for index in indices]


def fetch_remote(source: RemoteSource, clips_dir: Path) -> list[dict]:
    """Extract this source's selected members into `clips_dir`, one record each."""
    url = RESOLVE_URL.format(
        repo=source.repo, revision=source.revision, member=source.member
    )
    try:
        handle = io.BufferedReader(_RangeFile(url), buffer_size=BUFFER_BYTES)
        archive = zipfile.ZipFile(handle)
    except Exception as error:  # noqa: BLE001 - urllib and zipfile fail in many ways
        raise FetchError(f"{source.source_id}: cannot read {url}: {error}") from error

    with archive:
        catalog = [
            (info.filename, info.file_size)
            for info in archive.infolist()
            if not info.is_dir()
        ]
        chosen = select_members(catalog, source.count)
        if len(chosen) < source.count:
            raise FetchError(
                f"{source.source_id}: wanted {source.count} clip(s) but only "
                f"{len(chosen)} eligible member(s) in {source.member}"
            )
        records = []
        for index, member in enumerate(chosen):
            payload = archive.read(member)
            clip_id = f"{source.source_id}_{index:02d}"
            destination = clips_dir / f"{clip_id}.mp4"
            destination.write_bytes(payload)
            records.append(
                {
                    "clip_id": clip_id,
                    "label": source.label,
                    "family": source.family,
                    "stratum": source.stratum,
                    "source": f"{source.repo}@{source.revision}#{source.member}:{member}",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            print(f"  {clip_id}  {len(payload) / 1e6:.1f} MB  {member}")
        return records


def copy_local(split_id: str, split_dir: Path, clips_dir: Path) -> list[dict]:
    """Copy one on-disk FaceForensics++ split in, one record per clip."""
    manifest = split_dir / "manifest.csv"
    if not manifest.is_file():
        raise FetchError(f"{split_id}: no manifest at {manifest}")
    records = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            original_id = (row.get("clip_id") or "").strip()
            prefix = original_id.split("_")[0]
            if prefix not in LOCAL_FAMILIES:
                raise FetchError(f"{split_id}: unrecognised clip id {original_id!r}")
            label, family, stratum = LOCAL_FAMILIES[prefix]
            media = split_dir / (row.get("path") or "").strip()
            if not media.is_file():
                raise FetchError(f"{split_id}: missing media {media}")
            payload = media.read_bytes()
            clip_id = f"{split_id}_{original_id}"
            (clips_dir / f"{clip_id}.mp4").write_bytes(payload)
            records.append(
                {
                    "clip_id": clip_id,
                    "label": label,
                    "family": family,
                    "stratum": stratum,
                    "source": f"{split_dir.name}:{original_id} "
                              f"({(row.get('source') or '').strip()})",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    print(f"  {split_id}: {len(records)} clip(s)")
    return records


def drop_duplicates(records: list[dict], clips_dir: Path) -> tuple[list[dict], list[dict]]:
    """Remove clips whose bytes already appear in the corpus, keeping the first.

    The two FaceForensics++ splits were sampled independently and can name the same
    source video. One file counted twice is one measurement counted twice, and on a
    corpus this size a duplicated genuine clip moves a false-positive rate visibly.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    seen: dict[str, str] = {}
    for record in records:
        first = seen.get(record["sha256"])
        if first is not None:
            dropped.append({**record, "duplicate_of": first})
            (clips_dir / f"{record['clip_id']}.mp4").unlink(missing_ok=True)
            continue
        seen[record["sha256"]] = record["clip_id"]
        kept.append(record)
    return kept, dropped


def corpus_digest(records: list[dict]) -> str:
    """One digest naming every clip in the corpus and its exact bytes.

    `clip_id:sha256` per clip, in clip_id order. Two corpora with this digest hold the
    same clips under the same names; anything added, removed or re-encoded changes it.
    """
    joined = "\n".join(
        f"{record['clip_id']}:{record['sha256']}"
        for record in sorted(records, key=lambda item: item["clip_id"])
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def write_manifest(records: list[dict], path: Path) -> str:
    """Write the benchmark manifest and return its SHA-256."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["clip_id", "path", "label", "audio_path", "family", "stratum"])
    for record in sorted(records, key=lambda item: item["clip_id"]):
        writer.writerow(
            [
                record["clip_id"],
                f"clips/{record['clip_id']}.mp4",
                record["label"],
                "",
                record["family"],
                record["stratum"],
            ]
        )
    raw = buffer.getvalue().encode("utf-8")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fetch_corpus",
        description="Build the R4-T1 calibration corpus from pinned sources.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="directory to build the corpus in"
    )
    parser.add_argument(
        "--local-corpus-root",
        type=Path,
        default=Path.home() / "deepguard-corpus",
        help="directory holding the R3-T1 FaceForensics++ splits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow rebuilding into a directory that already holds a manifest",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.output_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        print(f"refusing to overwrite {manifest_path} (pass --overwrite)", file=sys.stderr)
        return 1

    clips_dir = args.output_dir / "clips"
    if clips_dir.exists() and args.overwrite:
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    try:
        for split_id, split_name in LOCAL_SPLITS:
            print(f"{split_id} (local)")
            records += copy_local(split_id, args.local_corpus_root / split_name, clips_dir)
        for source in REMOTE_SOURCES:
            print(f"{source.source_id} <- {source.repo}@{source.revision[:12]}")
            records += fetch_remote(source, clips_dir)
    except FetchError as error:
        print(error, file=sys.stderr)
        return 1

    records, duplicates = drop_duplicates(records, clips_dir)
    manifest_sha256 = write_manifest(records, manifest_path)

    label_counts: dict[str, int] = {}
    stratum_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for record in records:
        label_counts[record["label"]] = label_counts.get(record["label"], 0) + 1
        stratum_counts[record["stratum"]] = stratum_counts.get(record["stratum"], 0) + 1
        family_counts[record["family"]] = family_counts.get(record["family"], 0) + 1

    sources = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "corpus_digest": corpus_digest(records),
        "manifest_sha256": manifest_sha256,
        "clip_count": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "remote_sources": [
            {
                "source_id": source.source_id,
                "repo": source.repo,
                "revision": source.revision,
                "member": source.member,
                "label": source.label,
                "family": source.family,
                "stratum": source.stratum,
                "requested": source.count,
            }
            for source in REMOTE_SOURCES
        ],
        "local_splits": [
            {"split_id": split_id, "directory": name} for split_id, name in LOCAL_SPLITS
        ],
        "duplicates_dropped": duplicates,
        "clips": sorted(records, key=lambda item: item["clip_id"]),
    }
    (args.output_dir / "sources.json").write_text(
        json.dumps(sources, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"\n{len(records)} clip(s), {len(duplicates)} duplicate(s) dropped\n"
        f"labels {sources['label_counts']}\n"
        f"strata {sources['stratum_counts']}\n"
        f"corpus digest {sources['corpus_digest']}\n"
        f"wrote {manifest_path}\nwrote {args.output_dir / 'sources.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
