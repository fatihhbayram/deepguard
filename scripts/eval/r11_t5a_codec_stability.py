"""R11-T5A: AASIST measurement stability across three codec/container transformations.

This script measures. It does not judge.

It takes the 12 frozen R11-T2A WAV derivatives, re-hashes them against their frozen hashes,
encodes each one into exactly three codec/container variants (AAC/m4a, Opus/webm, MP3),
decodes every variant back to the only input shape the detector accepts, runs the unmodified
`audio_detector.analyze_audio_authenticity(...)` over each decoded file, and compares the raw
logits against the R11-T3A baseline frozen by R11-T4A.

No threshold, calibration, vote, label, FPR, accuracy or risk-engine rule appears anywhere in
this file, and none may be added to it. MAD, max deviation and Pearson correlation here are
descriptions of how far two measurement arrays sit apart; they are not scores and they carry
no operating point.

Two facts shape the whole design:

1. `audio_detector._read_pcm16_mono` opens its input with the stdlib `wave` module and refuses
   anything that is not 16 kHz mono 16-bit PCM. A compressed file cannot be handed to it. The
   transformation under test is therefore a round trip -- encode to the variant, decode back to
   PCM -- and the lineage recorded per variant has four stages:
   baseline wav -> encoded derivative -> decoded wav (the inference input) -> inference result.
   Nothing is padded, trimmed or delay-compensated by this task at any stage.

2. The detector's windows are cut at `window_index * window_samples`. Every full window's
   `start_sample`/`end_sample` pair therefore matches across two files *by construction*, even
   when a lossy encoder has shifted the audio content inside those boundaries. Accepting a
   boundary match on its own would be index alignment wearing a sample-boundary costume. So a
   window is treated as comparable only when its boundaries match AND the file carries no net
   content offset (`variant.total_samples == baseline.total_samples`). Anything else is
   reported as `not_comparable_due_to_alignment_shift` with the reason that made it so.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "deepguard/r11-t5a-codec-stability/1"
TASK = "R11-T5A"
FROZEN_INPUT_TASK = "R11-T2A"
BASELINE_TASK = "R11-T3A"
PRESERVATION_TASK = "R11-T4A"

HASH_CHUNK_BYTES = 1024 * 1024

NOT_OBSERVABLE = "not_observable_from_current_detector_contract"
UNEXPOSED_WINDOW_FIELDS = ("windows_attempted", "windows_failed", "windows_skipped")

NOT_COMPARABLE = "not_comparable_due_to_alignment_shift"
UNDEFINED = "undefined"

LOGIT_CHANNELS = ("logit_0", "logit_1", "bona_fide_logit")


class StabilityHalt(RuntimeError):
    """A frozen hash did not match, or an invariant this task must not violate was broken."""


# --- Transformations ---------------------------------------------------------------------------
#
# Exactly three, fixed here and nowhere else. Every argument is spelled out rather than defaulted
# so the manifest can carry the literal argv that produced each byte on disk.
#
# `-map_metadata -1 -fflags +bitexact -flags:a +bitexact` strips the muxer's clock-derived and
# build-derived metadata (mp4 creation time, encoder version strings). It does not touch the
# encoded audio; it is what makes a derivative's SHA-256 a reproducible function of its input
# rather than of the minute it was produced.


@dataclass(frozen=True)
class Variant:
    variant_id: str
    codec: str
    container: str
    muxer: str
    encoder: str
    bitrate: str
    suffix: str

    def encode_argv(self, source: Path, target: Path) -> list[str]:
        return [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-map_metadata", "-1",
            "-fflags", "+bitexact",
            "-flags:a", "+bitexact",
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", self.encoder,
            "-b:a", self.bitrate,
            "-f", self.muxer,
            str(target),
        ]

    def decode_argv(self, source: Path, target: Path) -> list[str]:
        """Back to the exact shape `audio_detector` accepts: 16 kHz mono PCM s16le WAV.

        No `-af`, no trim, no delay compensation. Whatever the encoder added to the front or the
        back of the recording survives into the decoded file and is measured, not corrected.
        """
        return [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-map_metadata", "-1",
            "-fflags", "+bitexact",
            "-flags:a", "+bitexact",
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            str(target),
        ]


VARIANTS = (
    Variant("aac_m4a", "aac", "m4a", "mp4", "aac", "128k", ".m4a"),
    Variant("opus_webm", "opus", "webm", "webm", "libopus", "64k", ".webm"),
    Variant("mp3", "mp3", "mp3", "mp3", "libmp3lame", "128k", ".mp3"),
)

# Opus decodes at 48 kHz by specification regardless of the rate handed to the encoder, so a
# probe of the encoded stream reports 48000 for this variant. The invariant this task actually
# enforces is on the file the detector reads: every decoded WAV is asserted to be 16 kHz mono
# PCM s16le before inference. Recorded rather than silently reconciled.
OPUS_RATE_NOTE = (
    "Opus is defined at 48 kHz internally; the encoded stream probes as 48000 Hz even though "
    "the encoder was fed 16 kHz mono. The 16 kHz mono invariant is enforced and asserted on the "
    "decoded inference input, which is the file the detector reads."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(payload: dict) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def run_argv(argv: list[str]) -> None:
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise StabilityHalt(
            f"command failed ({result.returncode}): {' '.join(argv)}\n{result.stderr.strip()}"
        )


def ffmpeg_identity() -> dict:
    version = subprocess.run(
        ["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, check=True
    ).stdout
    banner = version.splitlines()[0].strip()
    return {
        "version_banner": banner,
        "version": banner.split(" ")[2] if len(banner.split(" ")) > 2 else banner,
        "ffprobe_banner": subprocess.run(
            ["ffprobe", "-hide_banner", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0].strip(),
        "bitexact_flags_applied": ["-map_metadata -1", "-fflags +bitexact", "-flags:a +bitexact"],
        "bitexact_note": (
            "Applied so a derivative's SHA-256 is a function of its input and the encoder "
            "settings only, not of the wall clock. The encoded audio is unaffected."
        ),
    }


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries",
            "stream=codec_name,codec_type,sample_rate,channels,sample_fmt,bit_rate,duration,"
            "start_time,initial_padding:format=format_name,duration,bit_rate,size",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise StabilityHalt(f"ffprobe failed for {path}: {result.stderr.strip()}")

    parsed = json.loads(result.stdout)
    streams = parsed.get("streams") or []
    if not streams:
        raise StabilityHalt(f"ffprobe found no audio stream in {path}")
    stream = streams[0]
    fmt = parsed.get("format") or {}

    def as_int(value):
        return int(value) if value not in (None, "", "N/A") else None

    def as_float(value):
        return float(value) if value not in (None, "", "N/A") else None

    return {
        "codec_name": stream.get("codec_name"),
        "sample_rate": as_int(stream.get("sample_rate")),
        "channels": as_int(stream.get("channels")),
        "sample_fmt": stream.get("sample_fmt"),
        "stream_bit_rate": as_int(stream.get("bit_rate")),
        "stream_duration_seconds": as_float(stream.get("duration")),
        "stream_start_time_seconds": as_float(stream.get("start_time")),
        "initial_padding_samples": as_int(stream.get("initial_padding")),
        "format_name": fmt.get("format_name"),
        "format_duration_seconds": as_float(fmt.get("duration")),
        "format_bit_rate": as_int(fmt.get("bit_rate")),
        "size_bytes": as_int(fmt.get("size")),
    }


# --- Frozen input gate -------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenInput:
    recording_id: str
    panel: str
    person: str
    stem: str
    wav_path: Path
    sha256: str
    manifest_record: dict


def verify_frozen_inputs(manifest: dict, manifest_root: Path) -> list[FrozenInput]:
    """Re-hash every R11-T2A derivative before a single byte is encoded. Halt on any mismatch."""
    frozen: list[FrozenInput] = []
    for record in sorted(manifest["records"], key=lambda item: item["derivative"]["relative_path"]):
        derivative = record["derivative"]
        wav_path = manifest_root / derivative["relative_path"]
        if not wav_path.is_file():
            raise StabilityHalt(f"frozen derivative missing: {wav_path}")

        actual = sha256_file(wav_path)
        if actual != derivative["sha256"]:
            raise StabilityHalt(
                f"hash mismatch for {derivative['relative_path']}: "
                f"manifest {derivative['sha256']}, on disk {actual}"
            )
        if wav_path.stat().st_size != derivative["size_bytes"]:
            raise StabilityHalt(f"size mismatch for {derivative['relative_path']}")

        frozen.append(
            FrozenInput(
                recording_id=record["recording_id"],
                panel=record["panel"],
                person=record["person"],
                stem=wav_path.stem,
                wav_path=wav_path,
                sha256=actual,
                manifest_record=record,
            )
        )

    if len(frozen) != manifest["record_count"]:
        raise StabilityHalt(
            f"manifest declares {manifest['record_count']} records, verified {len(frozen)}"
        )
    return frozen


def verify_baseline(baseline_path: Path, t4a_manifest: dict) -> tuple[dict, str]:
    """Read the R11-T3A baseline from the R11-T4A preservation root and gate it on T4A's hash."""
    actual = sha256_file(baseline_path)
    preserved = {
        item["filename"]: item for item in t4a_manifest["preserved_artifacts"]
    }
    entry = preserved.get(baseline_path.name)
    if entry is None:
        raise StabilityHalt(
            f"{baseline_path.name} is not among the {PRESERVATION_TASK} preserved artifacts"
        )
    if entry["sha256"] != actual:
        raise StabilityHalt(
            f"baseline hash mismatch: {PRESERVATION_TASK} froze {entry['sha256']}, "
            f"on disk {actual}"
        )
    baseline = json.loads(baseline_path.read_bytes())
    if baseline.get("task") != BASELINE_TASK:
        raise StabilityHalt(f"expected a {BASELINE_TASK} baseline, got {baseline.get('task')!r}")
    return baseline, actual


# --- Detector ----------------------------------------------------------------------------------


def load_detector(detector_path: Path):
    """Import `audio_detector.py` by path, exactly as R11-T3A did, and hash the file it loaded.

    `app/__init__.py` validates Postgres and MinIO configuration at import time; this
    measurement needs neither. The module imports nothing from its own package, so loading it
    by path yields the identical module object.
    """
    spec = importlib.util.spec_from_file_location("deepguard_audio_detector", detector_path)
    if spec is None or spec.loader is None:
        raise StabilityHalt(f"could not load the detector at {detector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_metadata(detector, model_path: Path, detector_path: Path) -> dict:
    import numpy
    import onnxruntime

    return {
        "device": "cpu",
        "execution_provider_used": "CPUExecutionProvider",
        "execution_providers_available": sorted(onnxruntime.get_available_providers()),
        "inference_runtime": "onnxruntime",
        "onnxruntime_version": onnxruntime.__version__,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg": ffmpeg_identity(),
        "detector_source": {
            "path": str(detector_path),
            "sha256": sha256_file(detector_path),
            "entrypoint": "analyze_audio_authenticity",
            "modified_by_this_task": False,
        },
        "model": {
            "repository": detector.MODEL_REPOSITORY,
            "revision": detector.MODEL_REVISION,
            "filename": detector.MODEL_FILENAME,
            "sha256_expected": detector.MODEL_SHA256,
            "sha256_on_disk": sha256_file(model_path),
            "path": str(model_path),
            "window_samples_expected": detector.EXPECTED_WINDOW_SAMPLES,
            "sample_rate": detector.MODEL_SAMPLE_RATE,
            "channels": detector.MODEL_CHANNELS,
        },
    }


# --- Phase 1: transform --------------------------------------------------------------------------


def transform_one(frozen: FrozenInput, variant: Variant, work_dir: Path) -> dict:
    """Encode one frozen WAV into one variant, then decode it straight back to PCM.

    Both argv lists are recorded verbatim, as are the hashes of every file the pair produced.
    """
    encoded_dir = work_dir / "encoded" / variant.variant_id
    decoded_dir = work_dir / "decoded" / variant.variant_id
    encoded_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir.mkdir(parents=True, exist_ok=True)

    encoded_path = encoded_dir / f"{frozen.stem}{variant.suffix}"
    decoded_path = decoded_dir / f"{frozen.stem}.wav"

    encode_argv = variant.encode_argv(frozen.wav_path, encoded_path)
    run_argv(encode_argv)
    decode_argv = variant.decode_argv(encoded_path, decoded_path)
    run_argv(decode_argv)

    encoded_probe = probe(encoded_path)
    decoded_probe = probe(decoded_path)

    # The invariant is enforced where it can be enforced: on the file the detector will read.
    if decoded_probe["sample_rate"] != 16000 or decoded_probe["channels"] != 1:
        raise StabilityHalt(
            f"decoded inference input for {frozen.stem}/{variant.variant_id} is not 16 kHz mono: "
            f"{decoded_probe['sample_rate']} Hz, {decoded_probe['channels']} ch"
        )
    if decoded_probe["codec_name"] != "pcm_s16le":
        raise StabilityHalt(
            f"decoded inference input for {frozen.stem}/{variant.variant_id} is "
            f"{decoded_probe['codec_name']}, not pcm_s16le"
        )

    record = {
        "recording_id": frozen.recording_id,
        "panel": frozen.panel,
        "person": frozen.person,
        "stem": frozen.stem,
        "variant_id": variant.variant_id,
        "transform": {
            "codec": variant.codec,
            "container": variant.container,
            "muxer": variant.muxer,
            "encoder": variant.encoder,
            "requested_bitrate": variant.bitrate,
            "requested_sample_rate": 16000,
            "requested_channels": 1,
            "attribution": (
                "Combined codec+container transform. Any deviation observed under this variant "
                "is the effect of the codec and the container together and is not attributed to "
                "either one alone."
            ),
            "encoder_delay_compensated_by_this_task": False,
            "audio_padded_or_trimmed_by_this_task": False,
        },
        "commands": {
            "encode": encode_argv,
            "encode_shell": " ".join(encode_argv),
            "decode": decode_argv,
            "decode_shell": " ".join(decode_argv),
        },
        "source_frozen_wav": {
            "relative_path": frozen.manifest_record["derivative"]["relative_path"],
            "absolute_path": str(frozen.wav_path),
            "sha256": frozen.sha256,
            "sha256_source": f"{FROZEN_INPUT_TASK} frozen manifest, re-verified by this task",
            "size_bytes": frozen.wav_path.stat().st_size,
        },
        "encoded_derivative": {
            "absolute_path": str(encoded_path),
            "relative_path": str(encoded_path.relative_to(work_dir)),
            "sha256": sha256_file(encoded_path),
            "size_bytes": encoded_path.stat().st_size,
            "probe": encoded_probe,
        },
        "decoded_inference_input": {
            "absolute_path": str(decoded_path),
            "relative_path": str(decoded_path.relative_to(work_dir)),
            "sha256": sha256_file(decoded_path),
            "size_bytes": decoded_path.stat().st_size,
            "probe": decoded_probe,
        },
    }

    if variant.variant_id == "opus_webm":
        record["transform"]["sample_rate_note"] = OPUS_RATE_NOTE

    return record


def run_transform_phase(frozen_inputs: list[FrozenInput], work_dir: Path) -> list[dict]:
    records: list[dict] = []
    total = len(frozen_inputs) * len(VARIANTS)
    index = 0
    for frozen in frozen_inputs:
        for variant in VARIANTS:
            index += 1
            print(f"[transform {index}/{total}] {frozen.stem} -> {variant.variant_id}", flush=True)
            records.append(transform_one(frozen, variant, work_dir))
    return records


# --- Phase 2: measure ----------------------------------------------------------------------------


def measure_one(record: dict, detector, model_path: Path) -> dict:
    """Run the unmodified detector over one decoded variant and read its return value off."""
    decoded_path = Path(record["decoded_inference_input"]["absolute_path"])
    evidence = detector.analyze_audio_authenticity(decoded_path, model_path=model_path)

    observed = {
        "total_samples": evidence.total_samples,
        "sample_rate": evidence.sample_rate,
        "channels": evidence.channels,
        "window_samples": evidence.window_samples,
        "window_padding_scheme": evidence.window_padding_scheme,
        "windows_returned": len(evidence.windows),
        "model_repository": evidence.model_repository,
        "model_revision": evidence.model_revision,
        "model_sha256": evidence.model_sha256,
    }
    windows = [
        {
            "window_index": window.window_index,
            "start_sample": window.start_sample,
            "end_sample": window.end_sample,
            "padded_samples": window.padded_samples,
            "logits": list(window.logits),
            "bona_fide_logit": window.bona_fide_logit,
        }
        for window in evidence.windows
    ]

    inference = {
        "observed": observed,
        "derived_from_observed": {
            "decoded_duration_seconds": evidence.total_samples / evidence.sample_rate,
            "derivation": "total_samples / sample_rate",
        },
        "not_observable": {field: NOT_OBSERVABLE for field in UNEXPOSED_WINDOW_FIELDS},
        "windows": windows,
    }
    # The inference hash closes the lineage chain: source -> derivative -> inference. It is the
    # SHA-256 of this record's canonical serialization, so the numbers cannot be edited later
    # without the manifest disagreeing.
    inference["inference_sha256"] = sha256_bytes(canonical_json(inference))
    return inference


# --- Phase 3: alignment and stability ------------------------------------------------------------
#
# A length delta between the baseline and a decoded variant says nothing on its own about where
# the extra samples went. An encoder that pads the tail of its last frame leaves the recording's
# content at the same sample positions; one whose priming samples survive decoding moves every
# sample in the file. The first case leaves most windows genuinely comparable, the second leaves
# none. So the leading offset is measured rather than inferred from the length.

OFFSET_PROBE_SAMPLES = 262144
OFFSET_MAX_LAG_SAMPLES = 4800  # 0.3 s at 16 kHz; far beyond any codec's priming delay.
OFFSET_ANCHOR_FRACTIONS = (0.25, 0.5, 0.75)


def read_pcm16(path: Path):
    """Read a 16 kHz mono PCM16 WAV as float64, by the same contract the detector enforces."""
    import numpy as np
    import wave as wave_module

    with wave_module.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != 16000
        ):
            raise StabilityHalt(f"{path} is not 16 kHz mono PCM16")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype("float64")


def estimate_leading_offset(baseline_path: Path, variant_path: Path) -> dict:
    """Measure how far the decoded variant's content sits from the baseline's, in samples.

    Cross-correlates three widely separated segments of the two decoded waveforms over a lag
    window of +/- 0.3 s. Three anchors rather than one because a single agreeing lag could be a
    property of that stretch of audio; three that disagree mean the file is not shifted by a
    constant and cannot be compared at all.

    Returns the consensus lag in samples, positive when the variant runs late. Nothing is
    corrected: the number decides which windows are reported as comparable, and no audio is
    moved.
    """
    import numpy as np

    baseline = read_pcm16(baseline_path)
    variant = read_pcm16(variant_path)

    usable = min(baseline.size, variant.size)
    probe = min(OFFSET_PROBE_SAMPLES, max(usable // 8, 1))
    lag = OFFSET_MAX_LAG_SAMPLES

    anchors = []
    for fraction in OFFSET_ANCHOR_FRACTIONS:
        start = int(usable * fraction)
        start = max(lag, min(start, usable - probe - lag))
        if start < lag or probe <= 0 or start + probe + lag > usable:
            continue

        reference = baseline[start : start + probe]
        window = variant[start - lag : start + probe + lag]
        reference = reference - reference.mean()
        window = window - window.mean()

        if not np.any(reference) or not np.any(window):
            anchors.append({"anchor_start_sample": start, "lag_samples": None,
                            "reason": "silent_segment"})
            continue

        size = 1
        while size < window.size + reference.size:
            size *= 2
        spectrum = np.fft.rfft(window, size) * np.conjugate(np.fft.rfft(reference, size))
        correlation = np.fft.irfft(spectrum, size)[: window.size - reference.size + 1]
        peak = int(np.argmax(correlation))
        anchors.append(
            {
                "anchor_start_sample": start,
                "lag_samples": peak - lag,
                "peak_correlation": float(correlation[peak]),
            }
        )

    measured = [entry["lag_samples"] for entry in anchors if entry.get("lag_samples") is not None]
    distinct = sorted(set(measured))
    if not measured:
        consensus, agreement = None, "no_usable_anchor"
    elif len(distinct) == 1:
        consensus, agreement = distinct[0], "unanimous"
    else:
        consensus, agreement = None, "anchors_disagree"

    return {
        "method": (
            "FFT cross-correlation of the two decoded waveforms at three anchors, "
            f"probe {OFFSET_PROBE_SAMPLES} samples, lag window +/-{OFFSET_MAX_LAG_SAMPLES} samples"
        ),
        "anchors": anchors,
        "distinct_lags_observed": distinct,
        "agreement": agreement,
        "leading_offset_samples": consensus,
        "leading_offset_seconds": None if consensus is None else consensus / 16000,
        "compensated_by_this_task": False,
    }


def pearson(left: list[float], right: list[float]):
    """Pearson r over two equal-length arrays, or the string `undefined`.

    Undefined when fewer than two pairs survive alignment, or when either array has zero
    variance -- in both cases the coefficient is not defined rather than zero.
    """
    count = len(left)
    if count < 2:
        return UNDEFINED

    mean_left = sum(left) / count
    mean_right = sum(right) / count
    dev_left = [value - mean_left for value in left]
    dev_right = [value - mean_right for value in right]

    ss_left = sum(value * value for value in dev_left)
    ss_right = sum(value * value for value in dev_right)
    if ss_left <= 0.0 or ss_right <= 0.0:
        return UNDEFINED

    covariance = sum(a * b for a, b in zip(dev_left, dev_right))
    return covariance / ((ss_left ** 0.5) * (ss_right ** 0.5))


def channel_values(window: dict, channel: str) -> float:
    if channel == "logit_0":
        return window["logits"][0]
    if channel == "logit_1":
        return window["logits"][1]
    return window["bona_fide_logit"]


def align_and_measure(baseline_record: dict, variant_inference: dict, offset: dict) -> dict:
    """Pair variant windows with baseline windows on exact sample boundaries.

    A window is comparable only when both conditions hold:

      1. the file carries no leading content offset -- the measured lag between the two decoded
         waveforms is exactly zero, so sample n of the variant is sample n of the baseline; and
      2. a baseline window exists at the identical `start_sample`, with the identical
         `end_sample` and `padded_samples`.

    Condition 1 is the one that does the work. The detector cuts windows at
    `window_index * window_samples`, so `start_sample` pairs match across any two files by
    construction; a boundary check on its own would re-admit index alignment under a new name.
    When the lag is non-zero the whole file is reported as not comparable, because the only way
    to compare it would be to shift the audio, and this task compensates for nothing.

    Condition 2 then excludes the windows whose boundaries genuinely differ -- in practice the
    tail, where an encoder's trailing padding changes `end_sample` and `padded_samples` on the
    final window even though everything before it is sample-for-sample aligned.
    """
    baseline_windows = baseline_record["windows"]
    baseline_total = baseline_record["observed"]["total_samples"]
    variant_windows = variant_inference["windows"]
    variant_total = variant_inference["observed"]["total_samples"]

    leading_offset = offset["leading_offset_samples"]
    length_delta = variant_total - baseline_total
    by_start = {window["start_sample"]: window for window in baseline_windows}

    if leading_offset is None:
        file_reason = "leading_offset_not_determined"
    elif leading_offset != 0:
        file_reason = "leading_content_offset_nonzero"
    else:
        file_reason = None

    pairs: list[dict] = []
    not_comparable: list[dict] = []

    for window in variant_windows:
        start = window["start_sample"]
        baseline_window = by_start.get(start)

        if file_reason is not None:
            reason = file_reason
        elif baseline_window is None:
            reason = "no_baseline_window_at_this_start_sample"
        elif baseline_window["end_sample"] != window["end_sample"]:
            reason = "end_sample_mismatch"
        elif baseline_window["padded_samples"] != window["padded_samples"]:
            reason = "padded_samples_mismatch"
        else:
            reason = None

        if reason is not None:
            not_comparable.append(
                {
                    "window_index": window["window_index"],
                    "start_sample": start,
                    "end_sample": window["end_sample"],
                    "padded_samples": window["padded_samples"],
                    "baseline_start_sample": (
                        baseline_window["start_sample"] if baseline_window else None
                    ),
                    "baseline_end_sample": (
                        baseline_window["end_sample"] if baseline_window else None
                    ),
                    "baseline_padded_samples": (
                        baseline_window["padded_samples"] if baseline_window else None
                    ),
                    "status": NOT_COMPARABLE,
                    "reason": reason,
                }
            )
            continue

        pairs.append({"baseline": baseline_window, "variant": window})

    baseline_starts = {window["start_sample"] for window in baseline_windows}
    variant_starts = {window["start_sample"] for window in variant_windows}
    baseline_only = sorted(baseline_starts - variant_starts)

    statistics = {}
    for channel in LOGIT_CHANNELS:
        if not pairs:
            statistics[channel] = {
                "aligned_window_count": 0,
                "mean_absolute_deviation": UNDEFINED,
                "max_absolute_deviation": UNDEFINED,
                "max_absolute_deviation_at_start_sample": None,
                "pearson_correlation": UNDEFINED,
            }
            continue

        baseline_values = [channel_values(pair["baseline"], channel) for pair in pairs]
        variant_values = [channel_values(pair["variant"], channel) for pair in pairs]
        deltas = [
            abs(variant_value - baseline_value)
            for baseline_value, variant_value in zip(baseline_values, variant_values)
        ]
        peak_index = max(range(len(deltas)), key=deltas.__getitem__)

        statistics[channel] = {
            "aligned_window_count": len(pairs),
            "mean_absolute_deviation": sum(deltas) / len(deltas),
            "max_absolute_deviation": deltas[peak_index],
            "max_absolute_deviation_at_start_sample": pairs[peak_index]["variant"]["start_sample"],
            "pearson_correlation": pearson(baseline_values, variant_values),
        }

    reason_counts: dict[str, int] = {}
    for entry in not_comparable:
        reason_counts[entry["reason"]] = reason_counts.get(entry["reason"], 0) + 1

    return {
        "alignment": {
            "rule": (
                "A window is comparable only if the measured leading content offset between the "
                "two decoded waveforms is exactly 0 samples AND a baseline window exists at the "
                "identical start_sample with identical end_sample and padded_samples. Index "
                "position alone is never sufficient."
            ),
            "leading_offset": offset,
            "baseline_total_samples": baseline_total,
            "variant_total_samples": variant_total,
            "length_delta_samples": length_delta,
            "length_delta_seconds": length_delta / 16000,
            "length_delta_note": (
                "Decoded length difference. Read together with leading_offset: a non-zero length "
                "delta at a zero leading offset is trailing padding, which moves no earlier "
                "sample and disqualifies only the windows whose own boundaries changed."
            ),
            "baseline_windows_returned": len(baseline_windows),
            "variant_windows_returned": len(variant_windows),
            "aligned_window_count": len(pairs),
            "not_comparable_window_count": len(not_comparable),
            "not_comparable_reasons": reason_counts,
            "baseline_start_samples_absent_from_variant": baseline_only,
            "file_level_status": (
                "aligned" if pairs and not not_comparable
                else "partially_aligned" if pairs
                else NOT_COMPARABLE
            ),
        },
        "statistics": statistics,
        "not_comparable_windows": not_comparable,
        "aligned_pairs": pairs,
    }


# --- rec02__L_p04 ---------------------------------------------------------------------------------

ANOMALY_STEM = "rec02__L_p04"


def anomaly_block(t4a_manifest: dict, results: list[dict]) -> dict:
    """Carry the frozen R11-T3A facts for `rec02__L_p04`, then record what each variant did to it.

    Nothing here is reconciled, explained away or generalised. The carried facts are lineage;
    the per-variant rows are this task's observations, kept separate from them.
    """
    carried = t4a_manifest.get("lineage", {}).get(ANOMALY_STEM)
    per_variant = {}
    for result in results:
        if result["stem"] != ANOMALY_STEM:
            continue
        inference = result["inference"]
        windows = inference["windows"]
        per_variant[result["variant_id"]] = {
            "observed_total_samples": inference["observed"]["total_samples"],
            "observed_decoded_duration_seconds": (
                inference["derived_from_observed"]["decoded_duration_seconds"]
            ),
            "observed_windows_returned": inference["observed"]["windows_returned"],
            "observed_final_window_padded_samples": windows[-1]["padded_samples"] if windows else None,
            "window_padding_scheme": inference["observed"]["window_padding_scheme"],
            "length_delta_samples": result["stability"]["alignment"]["length_delta_samples"],
            "leading_offset_samples":
                result["stability"]["alignment"]["leading_offset"]["leading_offset_samples"],
            "aligned_window_count": result["stability"]["alignment"]["aligned_window_count"],
            "file_level_status": result["stability"]["alignment"]["file_level_status"],
        }

    return {
        "stem": ANOMALY_STEM,
        "carried_baseline_reference": {
            "source_task": PRESERVATION_TASK,
            "observed_in": BASELINE_TASK,
            "facts": carried,
            "note": (
                "Carried verbatim as lineage. R11-T5A did not re-analyse, reinterpret or "
                "extrapolate from these facts, and made no assumption about window loss."
            ),
        },
        "observed_under_each_variant": per_variant,
    }


# --- Pooled per-variant description ----------------------------------------------------------------


def pooled_by_variant(results: list[dict]) -> dict:
    """Pool the aligned per-window absolute deviations across files, per variant.

    A description of the aligned windows this run produced and nothing more. Files with no
    aligned windows contribute nothing and are counted separately, so a small pool is visible
    as a small pool rather than hidden inside an average.
    """
    pooled: dict[str, dict] = {}
    for variant in VARIANTS:
        rows = [result for result in results if result["variant_id"] == variant.variant_id]
        entry = {
            "files_processed": len(rows),
            "files_with_aligned_windows": sum(
                1 for row in rows if row["stability"]["alignment"]["aligned_window_count"] > 0
            ),
            "files_fully_not_comparable": sum(
                1 for row in rows
                if row["stability"]["alignment"]["file_level_status"] == NOT_COMPARABLE
            ),
            "aligned_window_total": sum(
                row["stability"]["alignment"]["aligned_window_count"] for row in rows
            ),
            "not_comparable_window_total": sum(
                row["stability"]["alignment"]["not_comparable_window_count"] for row in rows
            ),
            "length_delta_samples_by_file": {
                row["stem"]: row["stability"]["alignment"]["length_delta_samples"]
                for row in rows
            },
            "leading_offset_samples_by_file": {
                row["stem"]:
                    row["stability"]["alignment"]["leading_offset"]["leading_offset_samples"]
                for row in rows
            },
            "channels": {},
        }
        for channel in LOGIT_CHANNELS:
            deltas: list[float] = []
            for row in rows:
                for pair in row["stability"]["aligned_pairs"]:
                    deltas.append(
                        abs(
                            channel_values(pair["variant"], channel)
                            - channel_values(pair["baseline"], channel)
                        )
                    )
            entry["channels"][channel] = {
                "pooled_window_count": len(deltas),
                "pooled_mean_absolute_deviation": (
                    sum(deltas) / len(deltas) if deltas else UNDEFINED
                ),
                "pooled_max_absolute_deviation": max(deltas) if deltas else UNDEFINED,
            }
        pooled[variant.variant_id] = entry
    return pooled


# --- Serialization ---------------------------------------------------------------------------------

CSV_COLUMNS = (
    "stem", "recording_id", "panel", "person", "variant_id", "codec", "container",
    "window_index", "start_sample", "end_sample", "padded_samples",
    "comparability", "not_comparable_reason",
    "baseline_logit_0", "baseline_logit_1", "baseline_bona_fide_logit",
    "variant_logit_0", "variant_logit_1", "variant_bona_fide_logit",
    "delta_logit_0", "delta_logit_1", "delta_bona_fide_logit",
)


def serialize_csv(results: list[dict]) -> bytes:
    """One row per variant window. Floats via repr() so the CSV round-trips to the same doubles."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)

    for result in sorted(results, key=lambda item: (item["stem"], item["variant_id"])):
        rows: list[tuple] = []
        for pair in result["stability"]["aligned_pairs"]:
            baseline, variant = pair["baseline"], pair["variant"]
            rows.append(
                (
                    variant["window_index"], variant["start_sample"], variant["end_sample"],
                    variant["padded_samples"], "aligned", "",
                    repr(baseline["logits"][0]), repr(baseline["logits"][1]),
                    repr(baseline["bona_fide_logit"]),
                    repr(variant["logits"][0]), repr(variant["logits"][1]),
                    repr(variant["bona_fide_logit"]),
                    repr(variant["logits"][0] - baseline["logits"][0]),
                    repr(variant["logits"][1] - baseline["logits"][1]),
                    repr(variant["bona_fide_logit"] - baseline["bona_fide_logit"]),
                )
            )
        for entry in result["stability"]["not_comparable_windows"]:
            rows.append(
                (
                    entry["window_index"], entry["start_sample"], entry["end_sample"],
                    entry["padded_samples"], NOT_COMPARABLE, entry["reason"],
                    "", "", "", "", "", "", "", "", "",
                )
            )

        for row in sorted(rows, key=lambda item: item[0]):
            writer.writerow(
                [
                    result["stem"], result["recording_id"], result["panel"], result["person"],
                    result["variant_id"], result["transform"]["codec"],
                    result["transform"]["container"], *row,
                ]
            )

    return buffer.getvalue().encode("utf-8")


SEMANTIC_FLAGS = {
    "classification_performed": False,
    "labels_assigned": False,
    "calibration_applied": False,
    "threshold_applied": False,
    "false_positive_rate_computed": False,
    "accuracy_computed": False,
    "voting_applied": False,
    "verdict_produced": False,
    "risk_engine_invoked": False,
    "rulesets_consulted": False,
    "database_written": False,
    "inference_performed": True,
    "detector_source_modified": False,
    "audio_padded_or_trimmed_by_this_task": False,
    "encoder_delay_compensated_by_this_task": False,
}

PURPOSE = (
    "Raw AASIST logit stability across three combined codec+container transformations of the 12 "
    "frozen Real-TurnTurk derivatives, measured against the R11-T3A baseline on strictly "
    "boundary-aligned windows. MAD, max deviation and Pearson correlation here describe the "
    "distance between two measurement arrays. They are not scores, carry no operating point, "
    "and no threshold, label, verdict or risk-engine rule is present in this artifact."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="R11-T5A codec/container stability measurement")
    parser.add_argument("--phase", choices=("transform", "measure", "all"), default="all")
    parser.add_argument("--t2a-manifest", required=True, type=Path)
    parser.add_argument("--t3a-baseline", required=True, type=Path)
    parser.add_argument("--t4a-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("/models/aasist.onnx"))
    parser.add_argument("--detector", type=Path, default=Path("/app/app/audio_detector.py"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    transform_manifest_path = args.out_dir / "r11_t5a_transform_manifest.json"

    t2a_path = args.t2a_manifest.resolve()
    t2a_sha256 = sha256_file(t2a_path)
    t2a = json.loads(t2a_path.read_bytes())
    if t2a.get("task") != FROZEN_INPUT_TASK:
        raise StabilityHalt(f"expected an {FROZEN_INPUT_TASK} manifest, got {t2a.get('task')!r}")

    # Hash gate closes before a single byte is encoded.
    frozen_inputs = verify_frozen_inputs(t2a, t2a_path.parent)
    print(f"verified {len(frozen_inputs)} frozen inputs against {FROZEN_INPUT_TASK} hashes", flush=True)

    if args.phase in ("transform", "all"):
        transform_records = run_transform_phase(frozen_inputs, args.work_dir)
        if len(transform_records) != len(frozen_inputs) * len(VARIANTS):
            raise StabilityHalt("derivative count does not match sources x variants")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "task": TASK,
            "artifact": "transformation_metadata_manifest",
            "purpose": PURPOSE,
            "dataset": t2a["dataset"],
            "upstream": {
                "frozen_input_task": FROZEN_INPUT_TASK,
                "frozen_input_manifest_path": str(t2a_path),
                "frozen_input_manifest_sha256": t2a_sha256,
            },
            "ffmpeg": ffmpeg_identity(),
            "variants": [
                {
                    "variant_id": variant.variant_id,
                    "codec": variant.codec,
                    "container": variant.container,
                    "muxer": variant.muxer,
                    "encoder": variant.encoder,
                    "requested_bitrate": variant.bitrate,
                    "requested_sample_rate": 16000,
                    "requested_channels": 1,
                }
                for variant in VARIANTS
            ],
            "invariants": {
                "inference_input_is_16k_mono_pcm_s16le": True,
                "asserted_per_file_before_inference": True,
                "opus_sample_rate_note": OPUS_RATE_NOTE,
                "round_trip_rationale": (
                    "audio_detector._read_pcm16_mono accepts only 16 kHz mono 16-bit PCM WAV, so "
                    "a compressed file cannot be handed to it. The transform under test is the "
                    "encode/decode round trip and the decoded WAV is the inference input."
                ),
            },
            "semantic_flags": SEMANTIC_FLAGS,
            "derivative_count": len(transform_records),
            "records": sorted(transform_records, key=lambda item: (item["stem"], item["variant_id"])),
        }
        transform_manifest_path.write_bytes(canonical_json(manifest))
        print(f"transform manifest {transform_manifest_path}", flush=True)

    if args.phase == "transform":
        return 0

    manifest = json.loads(transform_manifest_path.read_bytes())

    # Re-hash all 36 derivatives and all 36 inference inputs against the manifest before
    # inference, the same gate the frozen inputs passed through.
    for record in manifest["records"]:
        for key in ("encoded_derivative", "decoded_inference_input"):
            entry = record[key]
            actual = sha256_file(Path(entry["absolute_path"]))
            if actual != entry["sha256"]:
                raise StabilityHalt(
                    f"{key} hash mismatch for {record['stem']}/{record['variant_id']}: "
                    f"manifest {entry['sha256']}, on disk {actual}"
                )
    print(f"re-verified {len(manifest['records'])} derivatives + inference inputs", flush=True)

    t4a_path = args.t4a_manifest.resolve()
    t4a = json.loads(t4a_path.read_bytes())
    baseline, baseline_sha256 = verify_baseline(args.t3a_baseline.resolve(), t4a)
    baseline_by_stem = {
        Path(record["derivative"]["relative_path"]).stem: record for record in baseline["records"]
    }
    print(f"baseline gate passed against {PRESERVATION_TASK} frozen hash {baseline_sha256}", flush=True)

    detector = load_detector(args.detector)
    metadata = runtime_metadata(detector, args.model_path, args.detector)

    results: list[dict] = []
    total = len(manifest["records"])
    for index, record in enumerate(manifest["records"], start=1):
        print(f"[measure {index}/{total}] {record['stem']} {record['variant_id']}", flush=True)
        baseline_record = baseline_by_stem.get(record["stem"])
        if baseline_record is None:
            raise StabilityHalt(f"no {BASELINE_TASK} baseline record for {record['stem']}")

        inference = measure_one(record, detector, args.model_path)
        offset = estimate_leading_offset(
            Path(record["source_frozen_wav"]["absolute_path"]),
            Path(record["decoded_inference_input"]["absolute_path"]),
        )
        stability = align_and_measure(baseline_record, inference, offset)

        results.append(
            {
                **{key: record[key] for key in
                   ("recording_id", "panel", "person", "stem", "variant_id", "transform",
                    "commands", "source_frozen_wav", "encoded_derivative",
                    "decoded_inference_input")},
                "baseline": {
                    "task": BASELINE_TASK,
                    "total_samples": baseline_record["observed"]["total_samples"],
                    "windows_returned": baseline_record["observed"]["windows_returned"],
                    "derivative_sha256": baseline_record["derivative"]["sha256_verified"],
                },
                "inference": inference,
                "stability": stability,
            }
        )

    anomaly = anomaly_block(t4a, results)
    pooled = pooled_by_variant(results)

    # `aligned_pairs` exists to build the CSV; it duplicates the window arrays and never reaches
    # the JSON artifact.
    report_results = []
    for result in results:
        stability = {key: value for key, value in result["stability"].items()
                     if key != "aligned_pairs"}
        report_results.append({**result, "stability": stability})

    report = {
        "schema_version": SCHEMA_VERSION,
        "task": TASK,
        "artifact": "stability_analysis_report",
        "purpose": PURPOSE,
        "dataset": manifest["dataset"],
        "lineage": {
            "chain": "frozen source wav -> encoded derivative -> decoded inference input -> inference",
            "frozen_input_task": FROZEN_INPUT_TASK,
            "frozen_input_manifest_sha256": manifest["upstream"]["frozen_input_manifest_sha256"],
            "baseline_task": BASELINE_TASK,
            "baseline_path": str(args.t3a_baseline.resolve()),
            "baseline_sha256": baseline_sha256,
            "baseline_hash_authority": PRESERVATION_TASK,
            "baseline_hash_matches_preservation_manifest": True,
            "preservation_manifest_path": str(t4a_path),
            "preservation_manifest_sha256": sha256_file(t4a_path),
            "transform_manifest_path": str(transform_manifest_path),
            "transform_manifest_sha256": sha256_file(transform_manifest_path),
        },
        "measurement_definitions": {
            "mean_absolute_deviation": (
                "mean over aligned windows of |variant logit - baseline logit|"
            ),
            "max_absolute_deviation": (
                "maximum over aligned windows of |variant logit - baseline logit|, with the "
                "start_sample at which it occurred"
            ),
            "pearson_correlation": (
                "Pearson r between the baseline and variant logit arrays over aligned windows; "
                f"reported as {UNDEFINED!r} when fewer than two aligned windows exist or either "
                "array has zero variance"
            ),
            "scope": "aligned windows only; not comparable windows contribute to no statistic",
            "interpretation": (
                "None is offered. These are distances between measurement arrays, not scores."
            ),
        },
        "semantic_flags": SEMANTIC_FLAGS,
        "runtime": metadata,
        "counts": {
            "source_count": len(frozen_inputs),
            "variant_count": len(VARIANTS),
            "derivative_count": len(results),
            "variant_window_total": sum(
                result["inference"]["observed"]["windows_returned"] for result in results
            ),
            "aligned_window_total": sum(
                result["stability"]["alignment"]["aligned_window_count"] for result in results
            ),
            "not_comparable_window_total": sum(
                result["stability"]["alignment"]["not_comparable_window_count"]
                for result in results
            ),
        },
        "pooled_by_variant": pooled,
        "rec02_l_p04": anomaly,
        "records": report_results,
    }

    json_first, json_second = canonical_json(report), canonical_json(report)
    csv_first, csv_second = serialize_csv(results), serialize_csv(results)
    if json_first != json_second or csv_first != csv_second:
        raise StabilityHalt("serialization is not deterministic over one result structure")

    json_path = args.out_dir / "r11_t5a_stability.json"
    csv_path = args.out_dir / "r11_t5a_stability_windows.csv"
    json_path.write_bytes(json_first)
    csv_path.write_bytes(csv_first)

    determinism = {
        "schema_version": SCHEMA_VERSION,
        "method": "double_serialization_of_identical_in_memory_result_structure",
        "inference_runs": 1,
        "json_sha256_pass_1": sha256_bytes(json_first),
        "json_sha256_pass_2": sha256_bytes(json_second),
        "json_byte_identical": True,
        "csv_sha256_pass_1": sha256_bytes(csv_first),
        "csv_sha256_pass_2": sha256_bytes(csv_second),
        "csv_byte_identical": True,
    }
    (args.out_dir / "r11_t5a_determinism.json").write_bytes(canonical_json(determinism))

    print(f"json {json_path} sha256={determinism['json_sha256_pass_1']}", flush=True)
    print(f"csv  {csv_path} sha256={determinism['csv_sha256_pass_1']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
