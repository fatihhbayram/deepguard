# Detector calibration (R4-T1, R5-T3)

Measures the two risk-eligible detectors — NVIDIA's synthetic-video detector and the
EfficientNet-B7 face-manipulation detector — against one labelled corpus, derives an
operating point for each from what was measured, and writes a calibration artifact that
names everything the numbers depend on.

**It changes nothing in production.** No file under `apps/` is touched by any step here,
no threshold moves, no stored decision changes. The risk engine still classifies from the
constants in its own module, from one signal, exactly as it did before. Adopting anything
in the artifact is R4-T2's work, under its own review.

## The three steps

```bash
# 1. Build the corpus (once). Deterministic: pinned dataset revisions, evenly spaced
#    member selection, one digest over the bytes it produced.
python3 scripts/calibration/fetch_corpus.py \
    --output-dir ../deepguard-corpus/r4t1_calibration

# 2a. Score it with the face-manipulation detector, in its own torch 2.11 virtualenv
#     (see ../benchmark/README.md for the venv and the pinned weights).
PYTHONPATH=scripts ~/.venvs/deepguard-benchmark/bin/python scripts/benchmark/cli.py \
    --manifest ../deepguard-corpus/r4t1_calibration/manifest.csv \
    --model benchmark.models.face_manipulation:detect \
    --output-dir ../deepguard-corpus/runs/r4t1-face

# 2b. Score it with NVIDIA's detector, inside the api container — that is where the
#     production preparation path and the provider credentials live.
docker cp scripts/benchmark deepguard-api-1:/app/benchmark
docker cp ../deepguard-corpus/r4t1_calibration deepguard-api-1:/tmp/r4t1_calibration
docker exec -w /app deepguard-api-1 python benchmark/cli.py \
    --manifest /tmp/r4t1_calibration/manifest.csv \
    --model benchmark.models.synthetic_video:detect \
    --output-dir /tmp/r4t1-svd
docker cp deepguard-api-1:/tmp/r4t1-svd ../deepguard-corpus/runs/r4t1-svd

# 3. Derive the operating points and write the artifact.
python3 scripts/calibration/calibrate.py \
    --manifest ../deepguard-corpus/r4t1_calibration/manifest.csv \
    --sources  ../deepguard-corpus/r4t1_calibration/sources.json \
    --synthetic-video-run ../deepguard-corpus/runs/r4t1-svd/results.json \
    --face-manipulation-run ../deepguard-corpus/runs/r4t1-face/results.json \
    --output-dir docs/ai/reviews/R4_T1
```

Scoring is the R2 benchmark harness, unchanged apart from one addition: a model module may
now define `provenance()`, and whatever it returns is recorded in `results.json`. A dotted
model reference names code, not bytes, and a threshold whose weights or provider
deployment cannot be named afterwards is not traceable.

## The corpus

Three strata, so that each detector is measured both inside and outside the domain it was
built for. `genuine_face` is where every false-positive figure comes from; `face_swap` is
a face composited into otherwise authentic footage; `generated` is video that was
generated rather than edited, from audio-driven talking heads to text-to-video.

Sources are pinned by commit and members are selected deterministically, so a rebuild
reproduces the same clips. `sources.json` records the revision, archive member, byte count
and SHA-256 behind every clip, plus a `corpus_digest` over the whole set. The clips
themselves live outside the repository — third-party media does not belong in git.

## How a threshold is chosen

By one rule per band, applied to the measured scores. Nothing is entered by hand.

- `T_HIGH` is the lowest threshold at which no genuine clip in the corpus is flagged,
  placed at the midpoint between the highest genuine score and the lowest score above it.
  This is DeepGuard's adopted error policy (P7-T2 §6) — strongly avoid a false HIGH on
  legitimate media, and pay for it in detection rate — expressed as arithmetic.
- `T_LOW` is its mirror: the highest threshold below which no manipulated clip falls. Its
  reported `coverage_genuine` is what the band would be worth, and a coverage near zero
  means the corpus does not support a LOW band for that detector at all.

Both can come back as `None`. That is the honest result when the distributions overlap
completely, and it is reported as such rather than replaced by a convenient number.

The trade-off table, the AUROC per stratum and the best-separating threshold are all
reported beside the selection, so a reviewer can see the cost of the policy rather than
being shown only the point it picked. The best-separating threshold is **not** selected.

## Disagreement

The two detectors answer different questions, so their scores are never averaged,
combined, or compared as magnitudes (AGENTS.md rule 11). What the artifact records is a
joint table of their separate decisions about the same clip: agreement, each kind of
one-sided flag, and abstention. Abstention is its own state — the face detector raises
when it finds no face in a clip, and recording that as "not flagged" would turn *I never
saw a face* into *I saw no manipulation*.

## LipForensics (R5-T3)

The third detector is calibrated on its own, by `calibrate_lipforensics.py`, over the
FaceForensics++ c23 test split R5-T1 benchmarked it on rather than the R4-T1 corpus — that
is the labelled data that exists for it, and the artifact says so rather than implying a
wider corpus. **It changes nothing in production either.** The risk engine knows nothing
about LipForensics; R5-T2 stores its score as evidence and applies no threshold to it
anywhere, and adopting an operating point is the Risk Engine v3 task under its own review.

```bash
python3 scripts/calibration/calibrate_lipforensics.py \
    --manifest ../deepguard-corpus/ff++_c23_test_r3t1/manifest.csv \
    --run ../deepguard-corpus/runs/2026-09-03-lipforensics/results.json \
    --cross-device-run ../deepguard-corpus/runs/2026-09-03-lipforensics-cuda/results.json \
    --output-dir docs/ai/reviews/R5_T3
```

`T_HIGH` and `T_LOW` come from the same two rules stated above, applied unchanged. Three
things differ from the two-detector run:

- **`0.5` is measured and rejected, not inherited.** R5-T1 reported its confusion matrix
  at `0.5` — the midpoint of a sigmoid, fixed before any score existed. The artifact
  tabulates what it would have decided on this corpus beside the derived point, so the
  rejection is a measurement.
- **The corpus digest is computed here.** This split predates `fetch_corpus` and carries no
  `sources.json`, so the media is hashed clip by clip under the same `clip_id:sha256`
  construction. A manifest names clip ids; only the digest names the bytes behind them.
  Family and stratum are read from the manifest's `source` column — the source dataset's
  own layout — and an unrecognised path raises rather than becoming `unknown`.
- **The two device runs are compared.** `--run` must be the CPU run, because
  `apps/api/app/lip_forensics.py` pins `DEVICE = "cpu"`; a CUDA run over the same manifest
  is admissible only as `--cross-device-run`, which reports how far scores moved and
  whether any decision at the selected point flipped. It is one perturbation, not a second
  corpus, and the report says so.

There is no disagreement table: a joint table needs two detectors scored over one corpus,
and this corpus was scored by one. LipForensics' score is not averaged with, combined with
or compared in magnitude to either detector calibrated above (AGENTS.md rule 11).

## Tests

```bash
python3 -m pytest scripts/calibration/tests
```

Standard library only, no network, no model, no corpus: the arithmetic that decides a
threshold is checked against cases small enough to verify by hand.
