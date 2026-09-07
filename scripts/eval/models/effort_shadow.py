"""Effort scored over video clips under the protocol frozen in `scripts/eval/effort_freeze.json`.

    PYTHONPATH=scripts ~/.venvs/deepguard-effort/bin/python scripts/benchmark/cli.py \
        --manifest ../deepguard-corpus/r7t5/manifest.csv \
        --model eval.models.effort_shadow:detect \
        --threshold 0.5 \
        --output-dir ../deepguard-corpus/runs/r7t7-effort

**This is a shadow benchmark and nothing else.** Nothing under `apps/` is imported, no production
threshold is read, no risk rule is consulted, and no artifact this produces is wired into anything
that makes a decision. Effort is a candidate being measured, not a detector being deployed, and
the separation is structural rather than a matter of care: this module can only be reached through
`scripts/benchmark/cli.py`, whose entire contract is one function taking a `Clip` and returning a
float.

**The bridge from a frame detector to a video corpus is the whole risk, so it is frozen.** Effort
classifies one aligned face crop. The corpus is video. Which frames get read, whether a face is
cut out of them, how eight frame outputs become one number, and where the operating point sits are
four degrees of freedom, and choosing any of them by looking at the answer would turn a hold-out
into a training set. All four are fixed in `effort_freeze.json` ahead of the evaluation split, all
four are taken from upstream's own semantics rather than invented here, and this module implements
that file and no variation on it:

- **eight frames**, evenly spaced over the clip (`effort.yaml`: `frame_num: {'test': 8}`);
- **dlib detection and five-point alignment** on every one (upstream's README requires the face
  extractor for this checkpoint, and §2 of the freeze records why that is upstream semantics and
  not a face-shaped guess);
- **the arithmetic mean** of the per-frame fake probabilities (`get_video_metrics` in
  DeepfakeBench computes a video's prediction as `pred_sum / leng`);
- **0.5**, which for a two-class softmax is upstream's own `argmax`, and is the threshold
  `get_video_metrics` uses for video-level accuracy.

No alternative aggregation is computed anywhere in this file. Per-frame scores are persisted so a
separately approved task could study one on the calibration split, but this task reports the mean.

**A clip with no readable face abstains rather than scoring zero.** `EffortAbstention` propagates
to the harness, which records the clip as excluded and continues. That is the honest reading: the
detector did not answer. Scoring it 0.0 would be a fabricated negative, and would let the detector
buy a better false-positive rate by failing to run — the same treatment R7-T5 gave LipForensics'
untrackable clips, kept identical here so the two are comparable.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from benchmark.dataset import Clip
from eval.models import effort_upstream

# Where the pinned artifacts live. Outside the repository, like every other model weight in this
# project: the checkpoint is 1.2 GB of someone else's research release, and the corpus directory is
# already the place things that must not be committed go.
_CORPUS_MODELS = Path(
    os.environ.get(
        "EFFORT_MODEL_DIR",
        Path(__file__).resolve().parents[4] / "deepguard-corpus" / "models" / "effort",
    )
)
CHECKPOINT_PATH = Path(
    os.environ.get("EFFORT_CHECKPOINT", _CORPUS_MODELS / "effort_clip_L14_trainOn_FaceForensic.pth")
)
LANDMARK_PATH = Path(
    os.environ.get("EFFORT_LANDMARKS", _CORPUS_MODELS / "shape_predictor_81_face_landmarks.dat")
)
# The CLIP backbone the checkpoint's weights are shaped against. A HuggingFace id or a local
# directory; upstream loads `openai/clip-vit-large-patch14` and so does this.
CLIP_MODEL = os.environ.get("EFFORT_CLIP_MODEL", "openai/clip-vit-large-patch14")

# The frozen protocol, restated as constants so a reader of this file sees the same numbers as a
# reader of the freeze document. Changing one here without changing it there is the bug this
# duplication is meant to make visible; `provenance()` emits both so a run records what it used.
FRAMES_PER_CLIP = 8
OPERATING_THRESHOLD = 0.5

# The expected identity of the checkpoint, from `effort_freeze.json`. Verified at load time rather
# than trusted: a benchmark whose weights cannot be named is not evidence (R4-T1), and a benchmark
# whose weights were silently swapped is worse than none.
EXPECTED_CHECKPOINT_SHA256 = "8d86711f098d16b49c048962fc3e16a906380f7bb17b8a0e89bd545b926943ee"
EXPECTED_LANDMARK_SHA256 = "8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869"

UPSTREAM_REPOSITORY = "https://github.com/YZY-stack/Effort-AIGI-Detection"
UPSTREAM_REVISION = "96f5dea2b534d400cfd7003f053c7e93c8e16461"


class EffortAbstention(Exception):
    """The detector produced no reading for this clip, and is saying so rather than guessing."""


class EffortLoadError(Exception):
    """The pinned model could not be assembled as specified."""


# Per-clip detail, read back by `provenance()` after the run. The aggregate score alone cannot
# distinguish "eight confident frames agreed" from "one frame of eight found a face and it happened
# to score high", and that distinction turned out to be the whole story for LipForensics in R7-T5
# (§6.1, the `windows_scored` table). Recording it here means the same question can be asked of
# Effort without rerunning anything.
FRAME_DETAIL: dict[str, dict] = {}

_model = None
_face_detector = None
_landmark_predictor = None
_device = None
_load_report: dict = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load() -> None:
    """Assemble the pinned model once, verifying it is the model that was pinned.

    Ordered so the cheap refusals happen first: a missing or wrong-hashed file should fail in
    milliseconds, not after the minute of SVD work that building the backbone costs.
    """
    global _model, _face_detector, _landmark_predictor, _device, _load_report

    if _model is not None:
        return

    import dlib
    from transformers import CLIPModel

    for path, expected, what in (
        (CHECKPOINT_PATH, EXPECTED_CHECKPOINT_SHA256, "checkpoint"),
        (LANDMARK_PATH, EXPECTED_LANDMARK_SHA256, "landmark model"),
    ):
        if not path.exists():
            raise EffortLoadError(f"{what} not found at {path}")
        actual = _sha256(path)
        if actual != expected:
            raise EffortLoadError(
                f"{what} at {path} has sha256 {actual}, but the frozen protocol pins {expected}"
            )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the architecture exactly as `EffortDetector.build_backbone` does. The SVD pass here
    # only establishes parameter shapes; every value it computes is replaced by the checkpoint's.
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
    backbone = effort_upstream.apply_svd_residual_to_self_attn(
        clip_model.vision_model, r=effort_upstream.SVD_RANK
    )
    model = effort_upstream.EffortModel(backbone)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}

    # Upstream loads with `strict=False` and marks it `# FIXME`. The permissiveness is kept, because
    # tightening it would change which weights the published checkpoint is allowed to supply and
    # this task measures the model as released — but what it papered over is recorded rather than
    # discarded. A missing `head.*` or a wholesale key mismatch would mean the scores below describe
    # a partly randomly-initialised network, and `provenance()` puts that in the artifact where a
    # reader of the results can see it.
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    # `strict=False` is only defensible if someone checks what it let through, so the missing keys
    # are split by whether the forward pass would ever read them. `S_r`, `U_r` and `V_r` are the
    # frozen top-`r` singular components: they are registered because `compute_orthogonal_loss`
    # needs them during training, and `SVDResidualLinear.forward` never touches them — it uses
    # `weight_main + U_residual @ diag(S_residual) @ V_residual`. A key outside that set going
    # missing would mean part of the scoring path is randomly initialised and every score below is
    # meaningless, so that count is reported separately and is expected to be zero.
    _TRAINING_ONLY_SUFFIXES = ("S_r", "U_r", "V_r")
    missing_forward_path = [
        key for key in missing if key.rsplit(".", 1)[-1] not in _TRAINING_ONLY_SUFFIXES
    ]

    model.eval().to(_device)

    _load_report = {
        "device": str(_device),
        "clip_backbone": CLIP_MODEL,
        "checkpoint_keys": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "missing_forward_path_keys": missing_forward_path,
        "missing_forward_path_key_count": len(missing_forward_path),
        "missing_keys_are_training_only": not missing_forward_path,
        "head_loaded_from_checkpoint": not any(key.startswith("head.") for key in missing),
    }

    _face_detector = dlib.get_frontal_face_detector()
    _landmark_predictor = dlib.shape_predictor(str(LANDMARK_PATH))
    _model = model


def _sample_indices(frame_count: int) -> list[int]:
    """The frozen sampling rule: eight evenly spaced indices, deduplicated, deterministic.

    `linspace` over the closed range rather than a stride, so the last frame is always included and
    a 30-frame clip and a 3000-frame clip are sampled the same way in proportion rather than in
    absolute position. Deduplication matters only for clips shorter than eight frames, where the
    freeze says the clip contributes every frame it has.
    """
    if frame_count <= 0:
        return []
    if frame_count <= FRAMES_PER_CLIP:
        return list(range(frame_count))
    return sorted({int(index) for index in np.linspace(0, frame_count - 1, FRAMES_PER_CLIP)})


def _frame_scores(clip: Clip) -> tuple[list[float], dict]:
    """Score every sampled frame that yields a face. Returns the scores and what happened."""
    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise EffortAbstention(f"clip could not be opened for decoding: {clip.path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = _sample_indices(frame_count)

        scores: list[float] = []
        decoded = 0
        faces_found = 0

        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            read, frame = capture.read()
            if not read:
                continue
            decoded += 1

            aligned = effort_upstream.extract_aligned_face_dlib(
                _face_detector, _landmark_predictor, frame, res=effort_upstream.RESOLUTION
            )
            if aligned is None:
                continue
            faces_found += 1

            tensor = effort_upstream.preprocess_face(aligned).to(_device)
            with torch.inference_mode():
                score = _model(tensor).squeeze().item()
            scores.append(float(score))

        detail = {
            "frame_count": frame_count,
            "frames_requested": len(indices),
            "frames_decoded": decoded,
            "frames_with_face": faces_found,
        }
        return scores, detail
    finally:
        capture.release()


def detect(clip: Clip) -> float:
    """Effort's aggregated fake-probability for `clip`, on Effort's own scale.

    Not a confidence. It is the mean over sampled frames of `softmax(head(pooler_output))[:, 1]`
    from a CLIP-L14 network trained on FaceForensics++, and what that number means on media
    outside FaceForensics++ is the question this benchmark is asking rather than an assumption it
    is entitled to make.

    Raises `EffortAbstention` when no sampled frame held a detectable face, which the harness
    records as an excluded clip.
    """
    _load()

    started = time.perf_counter()
    scores, detail = _frame_scores(clip)
    detail["seconds"] = time.perf_counter() - started

    if not scores:
        detail["aggregated_score"] = None
        detail["abstained"] = True
        FRAME_DETAIL[clip.clip_id] = detail
        raise EffortAbstention(
            f"no face detected in any of {detail['frames_decoded']} decoded frames "
            f"(of {detail['frames_requested']} sampled)"
        )

    aggregated = float(sum(scores) / len(scores))

    detail["frame_scores"] = scores
    detail["aggregated_score"] = aggregated
    detail["abstained"] = False
    FRAME_DETAIL[clip.clip_id] = detail

    return aggregated


def provenance() -> dict:
    """What answered, under which frozen rules, with the per-clip frame accounting."""
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }

    return {
        "detector": "effort",
        "variant": "face-deepfake (CLIP-L14, trained on FaceForensics++)",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "checkpoint_filename": CHECKPOINT_PATH.name,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "landmark_model_sha256": EXPECTED_LANDMARK_SHA256,
        "frozen_protocol": {
            "declared_in": "scripts/eval/effort_freeze.json",
            "frames_per_clip": FRAMES_PER_CLIP,
            "frame_sampling": "evenly spaced over [0, frame_count - 1], deduplicated",
            "face_detection_and_cropping": "dlib frontal detector, 81-landmark 5-point similarity alignment, 224x224, scale 1.3",
            "frame_output": "softmax(head(pooler_output), dim=1)[:, 1]",
            "aggregation": "arithmetic mean over frames that yielded a face",
            "operating_threshold": OPERATING_THRESHOLD,
            "abstention": "clip with no face in any sampled frame produces no reading",
        },
        "load": _load_report,
        "gpu": gpu,
        "torch_version": torch.__version__,
        "frame_detail_by_clip_id": dict(sorted(FRAME_DETAIL.items())),
    }
