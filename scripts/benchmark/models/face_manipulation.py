"""Face-manipulation detector under evaluation: the DFDC-winning EfficientNet-B7.

`detect` is the callable the harness runs — one `Clip` in, one score out:

    python3 scripts/benchmark/cli.py --model benchmark.models.face_manipulation:detect ...

It is a wrapper and nothing more. It samples frames, finds the face in each, and asks
the classifier how manipulated that face looks; the mean over the sampled frames is the
clip's score. There is no registry, no base class and no configuration object, because
`--model` naming this function is already the entire contract.

**Provenance.** Both weights are pinned by revision *and* verified by digest at load,
so a run's numbers can be traced to exact bytes rather than to a mutable branch:

| role | artifact | revision | SHA-256 |
|---|---|---|---|
| classifier | `tomas-gajarsky/facetorch-deepfake-efficientnet-b7` `model-torch2.11.pt2` | `4acc494f37eb63d7457166eff2acb45c5b04b9a6` | `97b49a70…7cc9e4b7` |
| face locator | `opencv/opencv_zoo` `face_detection_yunet_2023mar.onnx` | `47534e27c9851bb1128ccc0102f1145e27f23f98` | `8f2383e4…d2552fa4` |

The classifier is Selim Seferbekov's winning entry to the Deepfake Detection Challenge
(upstream `selimsef/dfdc_deepfake_challenge` @ `89c62904`, MIT), exported to
`torch.export` by Facetorch, whose model card documents the digests reproduced above
and the preprocessing this module implements: a 380x380 RGB face crop under ImageNet
normalisation, one binary logit per face.

**Why a second model is loaded.** The classifier scores *faces*, not frames — it was
trained on crops, and feeding it whole frames measures the background instead. Locating
the face is therefore part of this detector's preprocessing rather than a separate
capability, so YuNet is pinned and verified exactly like the classifier is.

**Torch version.** The `.pt2` artifact's published support cohort is torch >=2.11,<2.12
and the loader refuses anything else, rather than silently running a model outside the
contract its digest was published under. See this package's README for the virtualenv.
"""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from benchmark.dataset import Clip

CLASSIFIER_REVISION = "4acc494f37eb63d7457166eff2acb45c5b04b9a6"
CLASSIFIER_SHA256 = "97b49a70174c0d4f72d9d510d817bdc49a907af9af0242a6a1ba934a7cc9e4b7"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# Torch cohort the published artifact is valid for, per the Facetorch model card.
TORCH_MIN, TORCH_MAX = (2, 11), (2, 12)

# Preprocessing fixed by the checkpoint's training recipe, not free parameters.
INPUT_SIZE = 380
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Crop margin around the detected box. Seferbekov's pipeline trains on face crops
# enlarged by roughly a third, which keeps hairline and jaw — the edges where a swap
# blends into the original head, and where the artefacts actually live — inside frame.
CROP_MARGIN = 1 / 3

FRAME_SAMPLES = int(os.environ.get("DEEPGUARD_FACE_FRAMES", "8"))
FACE_SCORE_THRESHOLD = 0.6

WEIGHTS_ROOT = Path(
    os.environ.get("DEEPGUARD_FACE_MODEL_DIR", "~/.cache/deepguard/face_manipulation")
).expanduser()
CLASSIFIER_PATH = WEIGHTS_ROOT / f"facetorch-b7-{CLASSIFIER_REVISION}" / "model-torch2.11.pt2"
YUNET_PATH = WEIGHTS_ROOT / "yunet" / "face_detection_yunet_2023mar.onnx"


class WeightsError(RuntimeError):
    """The pinned weights are missing, or are not the bytes that were pinned."""


def _verify(path: Path, expected_sha256: str) -> Path:
    """Fail loudly unless `path` holds exactly the bytes this module pinned.

    A revision in a docstring is a claim; a digest checked at load is the claim being
    kept. A benchmark whose weights were quietly swapped underneath it reports a number
    about a model nobody can name afterwards.
    """
    if not path.is_file():
        raise WeightsError(
            f"pinned weights not found: {path}\n"
            f"See scripts/benchmark/README.md for how to fetch them."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise WeightsError(
            f"{path} does not match its pinned digest:\n"
            f"  expected {expected_sha256}\n  found    {digest}"
        )
    return path


@lru_cache(maxsize=1)
def _classifier():
    """Load the exported B7 once, on CPU, under the torch version it was published for."""
    import torch

    major, minor = (int(part) for part in torch.__version__.split(".")[:2])
    if not TORCH_MIN <= (major, minor) < TORCH_MAX:
        raise WeightsError(
            f"torch {torch.__version__} is outside the artifact's published support "
            f"cohort (>={TORCH_MIN[0]}.{TORCH_MIN[1]},<{TORCH_MAX[0]}.{TORCH_MAX[1]}); "
            f"see scripts/benchmark/README.md"
        )
    _verify(CLASSIFIER_PATH, CLASSIFIER_SHA256)
    # No `.eval()`: an exported program has its training-mode behaviour already baked
    # in, and torch 2.11 raises rather than accept the call.
    return torch.export.load(str(CLASSIFIER_PATH)).module()


@lru_cache(maxsize=1)
def _yunet_path() -> str:
    return str(_verify(YUNET_PATH, YUNET_SHA256))


def _face_crops(clip: Clip) -> list[np.ndarray]:
    """Sample frames evenly across the clip and return one face crop from each.

    Even spacing rather than the opening seconds: a manipulated clip is not uniformly
    manipulated, and scoring only the head of the file measures the part of a video an
    editor is most likely to have left alone. Frames where no face clears the detector's
    confidence threshold contribute nothing rather than contributing a background crop.
    """
    capture = cv2.VideoCapture(str(clip.path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open media: {clip.path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < 1 or width < 1 or height < 1:
            raise ValueError(f"media declares no decodable frames: {clip.path}")

        detector = cv2.FaceDetectorYN.create(
            _yunet_path(), "", (width, height), FACE_SCORE_THRESHOLD, 0.3, 5000
        )
        wanted = np.linspace(0, frame_count - 1, FRAME_SAMPLES).astype(int)
        crops: list[np.ndarray] = []
        for index in sorted(set(int(i) for i in wanted)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue
            crop = _crop_face(frame, faces[np.argmax(faces[:, -1])][:4], width, height)
            if crop is not None:
                crops.append(crop)
        return crops
    finally:
        capture.release()


def _crop_face(
    frame: np.ndarray, box: np.ndarray, width: int, height: int
) -> np.ndarray | None:
    """Square, margin-padded RGB crop at the model's input size, or None if degenerate.

    Squared on the longer side before padding so the resize to 380x380 never stretches
    the face: an anisotropic scale is itself a geometric distortion, and this model is
    being asked to judge exactly that kind of artefact.
    """
    x, y, box_width, box_height = (float(v) for v in box)
    centre_x, centre_y = x + box_width / 2, y + box_height / 2
    half = max(box_width, box_height) * (1 + CROP_MARGIN) / 2
    left, top = max(int(centre_x - half), 0), max(int(centre_y - half), 0)
    right, bottom = min(int(centre_x + half), width), min(int(centre_y + half), height)
    if right - left < 2 or bottom - top < 2:
        return None
    crop = cv2.resize(
        frame[top:bottom, left:right],
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def provenance() -> dict:
    """Identity of the weights this run scored with, verified rather than asserted.

    `_verify` re-reads and re-digests both files here, so what the artifact records is
    what was on disk for the run — not a constant copied out of the table above. Loading
    already checked them; checking again costs one read of each and removes the case where
    a docstring and the bytes have drifted apart.
    """
    return {
        "detector": "efficientnet-b7-dfdc",
        "classifier": {
            "artifact": "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
                        "/model-torch2.11.pt2",
            "revision": CLASSIFIER_REVISION,
            "sha256": hashlib.sha256(
                _verify(CLASSIFIER_PATH, CLASSIFIER_SHA256).read_bytes()
            ).hexdigest(),
        },
        "face_locator": {
            "artifact": "opencv/opencv_zoo/face_detection_yunet_2023mar.onnx",
            "sha256": hashlib.sha256(
                _verify(YUNET_PATH, YUNET_SHA256).read_bytes()
            ).hexdigest(),
        },
        "upstream": "selimsef/dfdc_deepfake_challenge @ 89c62904 (MIT)",
        "frame_samples": FRAME_SAMPLES,
        "face_score_threshold": FACE_SCORE_THRESHOLD,
        "input_size": INPUT_SIZE,
        "crop_margin": CROP_MARGIN,
        "score_semantics": "mean sigmoid(logit) over sampled face crops",
    }


def detect(clip: Clip) -> float:
    """Probability that `clip` carries a manipulated face, in `[0, 1]`.

    The clip's score is the mean of the per-frame probabilities. Mean rather than max:
    the maximum over a sample is the single most incriminating frame, which on a genuine
    clip is a compression artefact or a bad angle, and taking it would raise the false
    positive rate in exchange for a headline recall number.

    Raises when no face is found anywhere in the sample. That is an abstention, not a
    verdict, and the harness records it as an excluded clip — calling a video genuine
    because the detector never saw a face in it would be a fabricated negative.
    """
    import torch

    crops = _face_crops(clip)
    if not crops:
        raise ValueError(
            f"no face found in any of {FRAME_SAMPLES} sampled frame(s) of {clip.path}"
        )

    batch = np.stack(crops).astype(np.float32) / 255.0
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
    with torch.inference_mode():
        logits = _classifier()(tensor)
    return float(torch.sigmoid(logits.reshape(-1)).mean())
