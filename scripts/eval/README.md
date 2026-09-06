# Cross-dataset robustness evaluation (R7-T5)

An offline study of how the three risk-eligible detectors behave on media the benchmark
corpora do not contain, and of what the current risk rules do with them. It answers one
question the adopted thresholds cannot answer about themselves: **R5-T3 selected LipForensics'
operating point against twenty genuine clips from one benchmark family — does it hold on
in-the-wild footage?**

**It changes nothing in production.** No file under `apps/` is written by any step here. The
rules are *imported* from `apps/api/app/risk_engine.py` and executed exactly as they ship;
thresholds, rule ids and the calibration id are read and recorded, never moved. Candidate
operating points appear in the artifacts as candidates and adopting one is a separate,
Architect-approved task.

## The five steps

```bash
# 1. Build the corpus. Pinned revisions, deterministic member selection, ffmpeg derivatives.
PYTHONPATH=scripts python3 scripts/eval/build_corpus.py \
    --output-dir ../deepguard-corpus/r7t5 \
    --private-dir ../deepguard-corpus-r7t5-private

# 2. Prove the splits are independent. Exit status 1 if anything leaked.
PYTHONPATH=scripts ~/.venvs/deepguard-lipforensics/bin/python scripts/eval/leakage_check.py \
    --corpus ../deepguard-corpus/r7t5/corpus.json \
    --output ../deepguard-corpus/r7t5/leakage.json

# 3a. Score it with LipForensics, on the GPU, with the run counts recorded.
PYTHONPATH=scripts ~/.venvs/deepguard-lipforensics-cuda/bin/python scripts/benchmark/cli.py \
    --manifest ../deepguard-corpus/r7t5/manifest.csv \
    --model eval.models.lip_counted:detect \
    --output-dir ../deepguard-corpus/runs/r7t5-lip

# 3b. Score it with EfficientNet-B7, with the face-crop counts recorded.
PYTHONPATH=scripts ~/.venvs/deepguard-benchmark/bin/python scripts/benchmark/cli.py \
    --manifest ../deepguard-corpus/r7t5/manifest.csv \
    --model eval.models.face_counted:detect \
    --output-dir ../deepguard-corpus/runs/r7t5-face

# 3c. Score the evaluation split with NVIDIA's detector, inside the api container — that is
#     where the production preparation path and the provider credentials live.
docker exec -w /app deepguard-api-1 python benchmark/cli.py \
    --manifest /tmp/r7t5/manifest_svd.csv \
    --model benchmark.models.synthetic_video:detect \
    --output-dir /tmp/r7t5-svd

# 4. Replay the current v3 rules over the scored corpus. Observation only.
PYTHONPATH=scripts python3 scripts/eval/replay.py \
    --corpus ../deepguard-corpus/r7t5/corpus.json \
    --lip-run  ../deepguard-corpus/runs/r7t5-lip/results.json \
    --face-run ../deepguard-corpus/runs/r7t5-face/results.json \
    --svd-run  ../deepguard-corpus/runs/r7t5-svd/results.json \
    --output ../deepguard-corpus/r7t5/replay.json

# 5. Compute the measurements the report is written from.
PYTHONPATH=scripts python3 scripts/eval/analyze.py \
    --corpus ../deepguard-corpus/r7t5/corpus.json \
    --lip-run  ../deepguard-corpus/runs/r7t5-lip/results.json \
    --face-run ../deepguard-corpus/runs/r7t5-face/results.json \
    --svd-run  ../deepguard-corpus/runs/r7t5-svd/results.json \
    --replay ../deepguard-corpus/r7t5/replay.json \
    --output-dir ../deepguard-corpus/r7t5/analysis
```

Scoring is the R2 benchmark harness, unchanged. `eval/models/lip_counted.py` and
`eval/models/face_counted.py` wrap the R5-T1 and R3-T1 detector modules without
re-implementing any of their arithmetic; what they add is the per-clip unit count
(`windows_scored`, `frames_scored`) that the risk rules require and that a bare score does not
carry.

## The unit of observation is a lineage

Every rate in this study is counted over distinct **recordings**, not over files. A corpus
that is one third constructed transcodes would otherwise report a false-positive rate in which
one bad recording counts nine times.

`source_lineage_id` names the recording; `clip_id` names the file. Splitting is by lineage
only, so a clip and all its derivatives always land on the same side of the boundary — the
only construction under which the evaluation split is a hold-out at all. `leakage_check.py`
proves it afterwards, twice: by identity (no `source_lineage_id` and no `sha256` on both
sides) and by content (a perceptual difference hash over sampled frames, which survives the
rescaling and recompression that separate a clip from its own transcode and would therefore
catch a shared recording the manifest failed to declare).

## Detector-specific targets

Nothing is pooled into a single "manipulated vs genuine" number. Each detector is scored
against the manipulation it was built and calibrated for; everything else it meets is reported
separately as cross-family behaviour, and the two figures are never added:

| detector | primary target | cross-family |
|---|---|---|
| NVIDIA synthetic-video | generated / synthetic video | face swaps |
| EfficientNet-B7 | face manipulation / face swap | generated video |
| LipForensics | facial-forgery mouth dynamics, where a tracked-face reading exists | generated video |

Abstentions — no trackable face, no face found — are removed from both sides of every rate and
counted in their own. Folding them into the negatives would let a detector improve its
false-positive rate by failing to run.

**A detector score is not a confidence.** None of the three providers defines one. Each score
is a model output on that detector's own scale, and the study reports it as such.

## Statistics

`stats.py` computes exact Clopper-Pearson one-sided bounds from the standard library. The one
number this study may never print is a bare `0%`: zero observed false positives in *n* draws
is a statement about *n*, and the bound is what says how small "small enough to miss" is.

The acceptance target is fixed in `acceptance_target.json`, in its own file, **before** the
evaluation split was scored — a target chosen afterwards is a description of the result. It
declares a one-sided 95% upper bound of ≤1% on genuine-media false positives, which with zero
observed events requires *n* = 299 independent genuine evaluation lineages. `analyze.py`
reports whether it was met and never re-derives it.

## Privacy and redistribution

Every corpus record carries its source, licence, permission state and a `redistributable`
flag. The two real-world regression lineages are **private**: their media stays under
`--private-dir`, outside both the repository and the corpus directory, and is referenced
rather than copied. Nothing about them travels into any artifact except a stable lineage id, a
digest, an impersonal description of the acquisition path, and the measured detector outputs.
`analyze._example` drops the clip id of a private record before it can reach an example table.

## Tests

```bash
python3 -m pytest scripts/eval/tests -q
```

They cover the parts where a mistake would be invisible in the output: that an unassigned
lineage stops the build rather than defaulting into a split, that identical bytes under two
lineage ids are still caught, that a lineage counts once however many derivatives it has, that
an abstention leaves both sides of a rate, that the confidence bounds reproduce their closed
forms, and that the replay feeds the production rules evidence it observed rather than evidence
it assumed.
