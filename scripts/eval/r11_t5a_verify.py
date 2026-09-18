"""R11-T5A verification: assert the execution's claims against the artifacts and the tree.

Run on the host, after the measurement container has exited. Every check prints PASS or FAIL
with the value it actually read; the exit status is non-zero if any check failed. Nothing here
recomputes a statistic -- it verifies counts, hashes, lineage and the absence of anything this
task was forbidden to do.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/ai/reviews/R11_T5A"
DETECTOR = REPO / "apps/api/app/audio_detector.py"
RUNNER = REPO / "scripts/eval/r11_t5a_codec_stability.py"
CORPUS = Path("/home/adentechio/deepguard-corpus/external-genuine-real-turnturk")
T4A_MANIFEST = REPO / "docs/ai/reviews/R11_T4A/r11_t4a_manifest.json"
BASELINE = CORPUS / "evaluation/r11_t3a/r11_t3a_aasist_baseline.json"

EXPECTED_DERIVATIVES = 36
EXPECTED_VARIANTS = {"aac_m4a", "opus_webm", "mp3"}

results: list[tuple[bool, str, str]] = []


def check(passed: bool, name: str, detail: str = "") -> None:
    results.append((bool(passed), name, detail))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads((OUT / "r11_t5a_transform_manifest.json").read_bytes())
report = json.loads((OUT / "r11_t5a_stability.json").read_bytes())

# 1 -- derivative count is exactly 36, 12 sources x 3 variants, no more and no fewer.
check(
    manifest["derivative_count"] == EXPECTED_DERIVATIVES
    and len(manifest["records"]) == EXPECTED_DERIVATIVES
    and len(report["records"]) == EXPECTED_DERIVATIVES,
    "derivative count is exactly 36",
    f"manifest {manifest['derivative_count']}, report {len(report['records'])}",
)
variants_seen = {record["variant_id"] for record in manifest["records"]}
check(
    variants_seen == EXPECTED_VARIANTS,
    "exactly the three approved codec/container variants",
    ", ".join(sorted(variants_seen)),
)
stems = {record["stem"] for record in manifest["records"]}
check(len(stems) == 12, "12 distinct frozen sources", f"{len(stems)} stems")

# 2 -- every derivative and every inference input still hashes to what the manifest recorded.
encoded_ok = decoded_ok = 0
for record in manifest["records"]:
    for key, counter in (("encoded_derivative", "encoded"), ("decoded_inference_input", "decoded")):
        entry = record[key]
        path = Path(entry["absolute_path"].replace("/work", str(CORPUS / "r11_t5a")))
        if path.is_file() and sha256_file(path) == entry["sha256"]:
            if counter == "encoded":
                encoded_ok += 1
            else:
                decoded_ok += 1
check(encoded_ok == EXPECTED_DERIVATIVES, "36/36 derivative SHA-256 match the manifest",
      f"{encoded_ok}/36")
check(decoded_ok == EXPECTED_DERIVATIVES, "36/36 inference-input SHA-256 match the manifest",
      f"{decoded_ok}/36")

# 3 -- the exact transform metadata is on record, not summarised.
commands_ok = all(
    record["commands"]["encode"][0] == "ffmpeg"
    and record["commands"]["decode"][0] == "ffmpeg"
    and "-ar" in record["commands"]["encode"]
    and "16000" in record["commands"]["encode"]
    and "-ac" in record["commands"]["encode"]
    and record["commands"]["encode_shell"]
    and record["commands"]["decode_shell"]
    for record in manifest["records"]
)
check(commands_ok, "exact ffmpeg encode+decode argv recorded per derivative")
check(
    bool(manifest["ffmpeg"]["version_banner"]) and bool(report["runtime"]["ffmpeg"]["version_banner"]),
    "ffmpeg version recorded",
    manifest["ffmpeg"]["version_banner"],
)

# 4 -- 16 kHz mono invariance on every file the detector actually read.
invariant_ok = all(
    record["decoded_inference_input"]["probe"]["sample_rate"] == 16000
    and record["decoded_inference_input"]["probe"]["channels"] == 1
    and record["decoded_inference_input"]["probe"]["codec_name"] == "pcm_s16le"
    for record in manifest["records"]
)
check(invariant_ok, "36/36 inference inputs are 16 kHz mono PCM s16le")
observed_ok = all(
    record["inference"]["observed"]["sample_rate"] == 16000
    and record["inference"]["observed"]["channels"] == 1
    for record in report["records"]
)
check(observed_ok, "36/36 detector observations report 16 kHz mono")

# 5 -- audio_detector.py is unmodified, by the tree and by the hash the run recorded.
diff = subprocess.run(
    ["git", "-C", str(REPO), "status", "--porcelain", "--", str(DETECTOR.relative_to(REPO))],
    capture_output=True, text=True,
)
check(diff.stdout.strip() == "", "audio_detector.py has no working-tree change",
      diff.stdout.strip() or "clean")
detector_sha = sha256_file(DETECTOR)
check(
    report["runtime"]["detector_source"]["sha256"] == detector_sha,
    "detector hash recorded by the run matches the file on disk",
    detector_sha,
)
check(
    report["runtime"]["detector_source"]["entrypoint"] == "analyze_audio_authenticity",
    "the measured entrypoint is the production one",
)

# 6 -- the baseline this run compared against is the hash R11-T4A froze.
t4a = json.loads(T4A_MANIFEST.read_bytes())
frozen = {item["filename"]: item["sha256"] for item in t4a["preserved_artifacts"]}
baseline_sha = sha256_file(BASELINE)
check(
    frozen.get("r11_t3a_aasist_baseline.json") == baseline_sha == report["lineage"]["baseline_sha256"],
    "baseline input hash matches the R11-T4A frozen hash",
    baseline_sha,
)
check(
    sha256_file(CORPUS / "r11_t2a/r11_t2a_execution_manifest.json")
    == report["lineage"]["frozen_input_manifest_sha256"],
    "frozen R11-T2A input manifest hash matches",
)

# 7 -- the lineage chain is closed for all 36: source -> derivative -> inference.
lineage_ok = all(
    len(record["source_frozen_wav"]["sha256"]) == 64
    and len(record["encoded_derivative"]["sha256"]) == 64
    and len(record["decoded_inference_input"]["sha256"]) == 64
    and len(record["inference"]["inference_sha256"]) == 64
    for record in report["records"]
)
check(lineage_ok, "source -> derivative -> inference hash chain present for all 36")

# 8 -- statistics exist only where windows aligned, and nowhere else.
stats_ok = True
detail = ""
for record in report["records"]:
    alignment = record["stability"]["alignment"]
    aligned = alignment["aligned_window_count"]
    for channel, statistics in record["stability"]["statistics"].items():
        if statistics["aligned_window_count"] != aligned:
            stats_ok, detail = False, f"{record['stem']}/{record['variant_id']} {channel} count"
        if aligned == 0 and statistics["mean_absolute_deviation"] != "undefined":
            stats_ok, detail = False, f"{record['stem']}/{record['variant_id']} {channel} MAD"
        if aligned < 2 and statistics["pearson_correlation"] != "undefined":
            stats_ok, detail = False, f"{record['stem']}/{record['variant_id']} {channel} pearson"
    if aligned + alignment["not_comparable_window_count"] != alignment["variant_windows_returned"]:
        stats_ok, detail = False, f"{record['stem']}/{record['variant_id']} window accounting"
check(stats_ok, "statistics computed on aligned windows only, every window accounted for", detail)
check(
    all(
        entry["status"] == "not_comparable_due_to_alignment_shift"
        for record in report["records"]
        for entry in record["stability"]["not_comparable_windows"]
    ),
    "unaligned windows are labelled not_comparable_due_to_alignment_shift",
)

# 9 -- nothing this task was forbidden to do was done.
flags = report["semantic_flags"]
forbidden = {key: value for key, value in flags.items()
             if key != "inference_performed" and value is not False}
check(not forbidden, "no classification, threshold, FPR, accuracy, vote, verdict or risk engine",
      str(forbidden) if forbidden else "all forbidden flags are false")
check(flags["inference_performed"] is True, "inference is declared")
check(
    flags["audio_padded_or_trimmed_by_this_task"] is False
    and flags["encoder_delay_compensated_by_this_task"] is False,
    "no audio was padded, trimmed or delay-compensated by this task",
)

# A static read of the runner: the only production module it may touch is audio_detector.py.
source = RUNNER.read_text()
imports = re.findall(r"^\s*(?:from|import)\s+([\w\.]+)", source, flags=re.MULTILINE)
forbidden_imports = [
    name for name in imports
    if re.search(r"risk|ruleset|verdict|decision|sqlalchemy|psycopg|minio|app\.", name)
]
check(not forbidden_imports, "runner imports no risk engine, ruleset, DB or app package",
      ", ".join(forbidden_imports) or "none")
check(
    "analyze_audio_authenticity" in source
    and "/app/app/audio_detector.py" in source,
    "runner calls the detector directly and nothing else",
)

# 10 -- the rec02__L_p04 reference is carried, and its variant behaviour is recorded apart from it.
anomaly = report["rec02_l_p04"]
carried = anomaly["carried_baseline_reference"]["facts"]
check(
    carried is not None and carried.get("windows_returned") == 255,
    "rec02__L_p04 baseline reference (255 windows) is carried from R11-T4A",
    str(carried.get("windows_returned") if carried else None),
)
check(
    set(anomaly["observed_under_each_variant"]) == EXPECTED_VARIANTS,
    "rec02__L_p04 behaviour recorded separately under each of the 3 variants",
)

# 11 -- determinism of the artifacts.
determinism = json.loads((OUT / "r11_t5a_determinism.json").read_bytes())
check(
    determinism["json_byte_identical"] and determinism["csv_byte_identical"],
    "artifacts serialize deterministically",
)
check(
    sha256_file(OUT / "r11_t5a_stability.json") == determinism["json_sha256_pass_1"],
    "stability JSON on disk matches its recorded hash",
)
check(
    sha256_file(OUT / "r11_t5a_stability_windows.csv") == determinism["csv_sha256_pass_1"],
    "stability CSV on disk matches its recorded hash",
)

width = max(len(name) for _, name, _ in results)
failed = 0
for passed, name, detail in results:
    if not passed:
        failed += 1
    print(f"[{'PASS' if passed else 'FAIL'}] {name.ljust(width)}  {detail}")

print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
