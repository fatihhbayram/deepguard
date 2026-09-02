# Detector benchmark (R2)

An offline harness for measuring a proposed detector against a labelled corpus, before
that detector is allowed to influence anything. It runs from the repository root, needs
only the standard library, and is independent of the API, the database and the
analysis-job worker.

## Run

```bash
python3 scripts/benchmark/cli.py \
    --manifest corpus/manifest.csv \
    --model mock \
    --output-dir runs/2026-09-02-mock
```

| flag | default | meaning |
|---|---|---|
| `--manifest` | required | ground-truth CSV |
| `--model` | `mock` | `mock`, or `package.module:attribute` |
| `--output-dir` | required | where `results.json` and `report.md` are written |
| `--threshold` | `0.5` | score at or above which a clip counts as manipulated |
| `--overwrite` | off | permits replacing an existing `results.json` |

Exit status is `0` for a completed run, `1` for a manifest or model that could not be
loaded. A run that could not start writes nothing.

## Manifest

CSV, one row per clip. `path` and `label` are required; `clip_id` and `audio_path` are
optional and unknown columns are ignored. Paths are resolved against the manifest's own
directory, so a corpus folder can be moved without editing it.

```csv
clip_id,path,label,audio_path
ls_1272,clips/real_ls_1272.mp4,real,
xtts_01,clips/synth_xtts_01.mp4,synthetic,clips/synth_xtts_01.wav
```

Labels are `real`, `synthetic`, `face_swap` and `audio_spoof`. Anything other than
`real` is the positive class; the families stay distinct in the per-label breakdown.

The whole manifest is validated before the first model call, and every problem in it is
reported at once with the line number that carries it.

## Models

A model is one callable taking a `dataset.Clip` and returning a float, where higher
means more likely manipulated. There is no base class to inherit and nothing to
register:

```python
# my_detector.py
def detect(clip):
    return score_of(clip.path)          # clip.audio_path is there when declared
```

```bash
PYTHONPATH=. python3 scripts/benchmark/cli.py --model my_detector:detect ...
```

`mock` is the built-in stand-in: a deterministic, label-blind score derived from the
clip id. It exists to exercise the harness and its accuracy means nothing.

### The face-manipulation candidate (R3)

`benchmark.models.face_manipulation:detect` wraps Selim Seferbekov's DFDC-winning
EfficientNet-B7, republished with immutable digests by Facetorch. It samples eight
frames, crops the face YuNet finds in each, and returns the mean probability of
manipulation. Both weights are pinned by revision and verified by SHA-256 at load, so a
run's numbers trace to exact bytes; the module docstring carries the provenance table.

The exported artifact is published for torch >=2.11,<2.12 and the wrapper refuses any
other version, so it runs from its own virtualenv rather than against whatever torch
the machine happens to carry:

```bash
python3 -m venv ~/.venvs/deepguard-benchmark
~/.venvs/deepguard-benchmark/bin/pip install \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.11.*"
~/.venvs/deepguard-benchmark/bin/pip install numpy opencv-python-headless
```

Fetch the pinned weights once. They live outside the repository — `$DEEPGUARD_FACE_MODEL_DIR`,
default `~/.cache/deepguard/face_manipulation` — because 273 MiB of weights do not
belong in git:

```bash
DIR=~/.cache/deepguard/face_manipulation
REV=4acc494f37eb63d7457166eff2acb45c5b04b9a6
mkdir -p "$DIR/facetorch-b7-$REV" "$DIR/yunet"
curl -L -o "$DIR/facetorch-b7-$REV/model-torch2.11.pt2" \
  "https://huggingface.co/tomas-gajarsky/facetorch-deepfake-efficientnet-b7/resolve/$REV/model-torch2.11.pt2"
curl -L -o "$DIR/yunet/face_detection_yunet_2023mar.onnx" \
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

```bash
~/.venvs/deepguard-benchmark/bin/python scripts/benchmark/cli.py \
    --manifest ../deepguard-corpus/ff++_c23_test_r3t1/manifest.csv \
    --model benchmark.models.face_manipulation:detect \
    --output-dir ../deepguard-corpus/runs/2026-09-02-face-b7
```

`DEEPGUARD_FACE_FRAMES` sets the number of sampled frames (default 8). A clip in which
no frame yields a face is raised as an error and excluded, not scored as genuine.

A model that raises on a clip does not end the run. That clip is recorded with its
error, excluded from the confusion matrix and from the latency figures, and counted in
the artifact — a crash is not scored as a wrong answer.

## Artifacts

`results.json` carries the run identity (model, threshold, manifest path and SHA-256,
interpreter and platform), the dataset composition, the confusion matrix, accuracy /
precision / recall / FPR / FNR, a per-label breakdown, latency and peak-RSS figures,
and one record per clip. `report.md` is the same run as a readable summary.

Undefined metrics are `null`, never `0.0`: precision over zero predicted positives is
not zero, and a benchmark that pretends otherwise has invented a measurement.

Same manifest, same model, same threshold gives the same metrics and the same records
in the same order. Timestamps and latencies are the parts that legitimately differ.

## Tests

```bash
python3 -m pytest scripts/benchmark/tests
```
