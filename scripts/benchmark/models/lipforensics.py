"""Face-forgery detector under evaluation: LipForensics (CVPR 2021).

`detect` is the callable the harness runs — one `Clip` in, one score out:

    python3 scripts/benchmark/cli.py --model benchmark.models.lipforensics:detect ...

It is a wrapper and nothing more. It reads short runs of consecutive frames, aligns each
frame to a mean face, crops the mouth, and asks the pretrained network how unnatural that
mouth moves; the mean logit over the sampled runs, squashed by a sigmoid, is the clip's
score. There is no registry, no base class and no configuration object, because `--model`
naming this function is already the entire contract.

**Why this detector, next to the B7 of R3.** The EfficientNet-B7 judges single face crops:
it looks for spatial artefacts frame by frame and never sees motion. LipForensics ignores
appearance almost entirely and classifies *mouth dynamics* over 25 consecutive frames, which
is what makes it a different opinion rather than a second copy of the same one — and why the
two can disagree in useful ways once R5 decides how to combine them.

**Provenance.** Weights, upstream source and the two auxiliary models are pinned by revision
*and* verified by SHA-256 at load, so a run's numbers trace to exact bytes:

| role | artifact | revision / origin | SHA-256 |
|---|---|---|---|
| forgery classifier | `lipforensics_ff.pth` | Google Drive `1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7`, linked from the upstream README | `4b7790bc…013c253d` |
| architecture + mean face | `ahaliassos/LipForensics` | `d0bf5553bfb9676f1771d590472b26a3a76de894` | per file, below |
| face detector | `s3fd-619a316812.pth` (S³FD, via face-alignment) | adrianbulat.com/downloads/python-fan | `619a3168…59b33543` |
| landmark model | `2DFAN4-11f355bf06.pth.tar` (FAN, via face-alignment) | adrianbulat.com/downloads/python-fan | `11f355bf…0d32e4aa` |

The classifier is the model Haliassos et al. trained on FaceForensics++ (Deepfakes,
FaceSwap, Face2Face, NeuralTextures) and reported Table 2 of the paper with; the network is
a ResNet-18 + MS-TCN whose definition is *the upstream source itself*, loaded from the pinned
checkout rather than retyped here, because a re-implementation that drifts by one layer loads
the same weights and quietly measures a different model.

**Why two more models are loaded.** LipForensics scores aligned mouth crops, not frames. The
alignment is 68 landmarks per frame warped onto the LRW mean face — part of this detector's
preprocessing rather than a separate capability — and the upstream README names RetinaFace
and FAN as the way to obtain them. This module uses `face-alignment`'s S³FD + 2D-FAN pair,
pinned and verified exactly like the classifier is.

**Torch and dependencies.** The checkpoint is a plain `state_dict`, so it is not bound to a
torch version the way R3's exported artifact is; the wrapper still runs from its own
virtualenv because `face-alignment`, `scikit-image` and `opencv` are not the API's
dependencies. See this package's README for the virtualenv and the fetch commands.
"""

import hashlib
import importlib.util
import json
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from skimage import transform as skimage_transform

from benchmark.dataset import Clip

UPSTREAM_REVISION = "d0bf5553bfb9676f1771d590472b26a3a76de894"
WEIGHTS_SHA256 = "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"

# The upstream files this wrapper executes or reads, relative to the pinned checkout.
SOURCE_SHA256 = {
    "models/resnet.py":
        "1d159120f159636b88291702e3e62ea5b43a8716687fb261308bc5092e3c122c",
    "models/tcn.py":
        "35f57b7b65c976ed77857f9784a94baaa15bd0bbdc94033ec2b3ef48e00c1712",
    "models/spatiotemporal_net.py":
        "eabfabbb06aa411796b10f9b1de0231bfee719fbc45ff0ab0c40f20820021d29",
    "models/configs/lrw_resnet18_mstcn.json":
        "dc9135351ee1b8458285b3bc5b8c42877f10a6403502437d312b0db81af678c9",
    "preprocessing/20words_mean_face.npy":
        "dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422",
}

SFD_SHA256 = "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"
FAN_SHA256 = "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa"

# Preprocessing fixed by the training recipe (upstream `preprocessing/crop_mouths.py` and
# the transform composed in `evaluate.py`), not free parameters.
FRAMES_PER_WINDOW = 25
STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]
MOUTH_LANDMARKS = slice(48, 68)
CROP_SIZE = 96
INPUT_SIZE = 88
SMOOTHING_WINDOW = 12
MOUTH_MEAN, MOUTH_STD = 0.421, 0.165

# How many 25-frame runs are sampled from a clip. Upstream evaluates the first 110 frames of
# each video, i.e. four consecutive runs; this samples the same number but spreads them over
# the whole clip, because a manipulation is not uniformly distributed and the opening second
# is the part an editor is most likely to have left alone.
WINDOWS = int(os.environ.get("DEEPGUARD_LIPFORENSICS_WINDOWS", "4"))

# CPU by default: cuDNN kernel selection makes GPU scores reproducible only against the same
# card, and a threshold is calibrated against the numbers a run actually produced.
DEVICE = os.environ.get("DEEPGUARD_LIPFORENSICS_DEVICE", "cpu")

MODEL_ROOT = Path(
    os.environ.get("DEEPGUARD_LIPFORENSICS_MODEL_DIR", "~/.cache/deepguard/lipforensics")
).expanduser()
SOURCE_ROOT = MODEL_ROOT / f"lipforensics-{UPSTREAM_REVISION}"
WEIGHTS_PATH = MODEL_ROOT / "weights" / "lipforensics_ff.pth"
FACE_ALIGNMENT_ROOT = MODEL_ROOT / "face-alignment"
SFD_PATH = FACE_ALIGNMENT_ROOT / "checkpoints" / "s3fd-619a316812.pth"
FAN_PATH = FACE_ALIGNMENT_ROOT / "checkpoints" / "2DFAN4-11f355bf06.pth.tar"

# The upstream package is executed under a private name so that importing a directory called
# `models` cannot shadow, or be shadowed by, anything else on the path.
UPSTREAM_PACKAGE = "_deepguard_lipforensics_upstream"


class WeightsError(RuntimeError):
    """The pinned artifacts are missing, or are not the bytes that were pinned."""


def _verify(path: Path, expected_sha256: str) -> Path:
    """Fail loudly unless `path` holds exactly the bytes this module pinned.

    A revision in a docstring is a claim; a digest checked at load is the claim being kept.
    This covers the upstream *source* as well as the weights: the architecture is executed
    from disk, so an edited `tcn.py` would change the model as surely as a swapped
    checkpoint would, and neither may happen unnoticed under a run that reports a number.
    """
    if not path.is_file():
        raise WeightsError(
            f"pinned artifact not found: {path}\n"
            f"See scripts/benchmark/README.md for how to fetch it."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise WeightsError(
            f"{path} does not match its pinned digest:\n"
            f"  expected {expected_sha256}\n  found    {digest}"
        )
    return path


def _upstream_source(relative: str) -> Path:
    """A verified file from the pinned upstream checkout."""
    return _verify(SOURCE_ROOT / relative, SOURCE_SHA256[relative])


@lru_cache(maxsize=1)
def _upstream_architecture():
    """Execute the pinned upstream `models` package and return `spatiotemporal_net`.

    The three modules are loaded in dependency order under `UPSTREAM_PACKAGE`, whose
    `__path__` is the checkout, so upstream's own relative imports resolve to the files whose
    digests were just checked and to nothing else.
    """
    package = types.ModuleType(UPSTREAM_PACKAGE)
    package.__path__ = [str(SOURCE_ROOT / "models")]
    sys.modules[UPSTREAM_PACKAGE] = package
    for name in ("resnet", "tcn", "spatiotemporal_net"):
        path = _upstream_source(f"models/{name}.py")
        spec = importlib.util.spec_from_file_location(f"{UPSTREAM_PACKAGE}.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{UPSTREAM_PACKAGE}.spatiotemporal_net"]


@lru_cache(maxsize=1)
def _classifier():
    """Load ResNet-18 + MS-TCN once, with the FF++-trained forgery weights, in eval mode.

    Built from upstream's own `Lipreading` and upstream's own config rather than through
    `get_model`, which reads that config relative to the process working directory and puts
    the model on `cuda:0` — neither of which a benchmark run should depend on.
    """
    import torch

    architecture = _upstream_architecture()
    config = json.loads(
        _upstream_source("models/configs/lrw_resnet18_mstcn.json").read_text("utf-8")
    )
    model = architecture.Lipreading(
        num_classes=1,
        relu_type=config["relu_type"],
        tcn_options={
            "num_layers": config["tcn_num_layers"],
            "kernel_size": config["tcn_kernel_size"],
            "dropout": config["tcn_dropout"],
            "dwpw": config["tcn_dwpw"],
            "width_mult": config["tcn_width_mult"],
        },
    )
    checkpoint = torch.load(
        _verify(WEIGHTS_PATH, WEIGHTS_SHA256), map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(DEVICE).eval()


@lru_cache(maxsize=1)
def _landmarker():
    """The pinned S³FD + 2D-FAN pair, reading only from the verified local cache.

    `face-alignment` fetches its weights through the torch hub cache, so hub is pointed at
    this module's own directory: the files are verified here first, and a run therefore
    never depends on adrianbulat.com being reachable or on whatever an earlier project left
    in `~/.cache/torch`.
    """
    import face_alignment
    import torch

    _verify(SFD_PATH, SFD_SHA256)
    _verify(FAN_PATH, FAN_SHA256)
    torch.hub.set_dir(str(FACE_ALIGNMENT_ROOT))
    return face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False
    )


@lru_cache(maxsize=1)
def _mean_face() -> np.ndarray:
    """The LRW mean-face landmarks every frame is aligned onto."""
    return np.load(_upstream_source("preprocessing/20words_mean_face.npy"))


def _window_starts(frame_count: int) -> list[int]:
    """First frame index of each sampled run, evenly spread and non-overlapping."""
    last_start = frame_count - FRAMES_PER_WINDOW
    if last_start < 0:
        return []
    starts = np.linspace(0, last_start, WINDOWS).astype(int)
    return sorted(set(int(start) for start in starts))


def _read_window(capture: cv2.VideoCapture, start: int, wanted: int) -> list[np.ndarray]:
    """Up to `wanted` consecutive frames from `start`, as BGR arrays."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def _landmarks(frames: list[np.ndarray]) -> list[np.ndarray] | None:
    """68 landmarks per frame, or None if any frame in the run has no face.

    All-or-nothing on purpose: the model reads a run of frames as one movement, and a run
    stitched across a frame whose alignment was guessed is a movement that never happened.
    The largest face is taken where several are found, so the same person is followed
    throughout the run.
    """
    landmarker = _landmarker()
    found = []
    for frame in frames:
        faces = landmarker.get_landmarks_from_image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        if not faces:
            return None
        found.append(max(faces, key=_landmark_area))
    return found


def _landmark_area(landmarks: np.ndarray) -> float:
    extent = landmarks.max(axis=0) - landmarks.min(axis=0)
    return float(extent[0] * extent[1])


def _mouth_crops(
    frames: list[np.ndarray], landmarks: list[np.ndarray]
) -> list[np.ndarray] | None:
    """Aligned 96x96 grayscale mouth crops, following upstream `crop_mouths.py`.

    Landmarks are averaged over a 12-frame look-ahead to take out tracker jitter, each frame
    is similarity-warped onto the mean face by five points around the eyes and nose, and the
    mouth is cut from the warped frame around the mean of landmarks 48-68. Frames past the
    last full smoothing window reuse the last transform, exactly as upstream does at the end
    of a video.
    """
    crops: list[np.ndarray] = []
    transform = None
    mean_face = _mean_face()
    for index, frame in enumerate(frames):
        if index + SMOOTHING_WINDOW <= len(frames):
            smoothed = np.mean(landmarks[index : index + SMOOTHING_WINDOW], axis=0)
            transform = skimage_transform.estimate_transform(
                "similarity", smoothed[STABLE_POINTS, :], mean_face[STABLE_POINTS, :]
            )
        if transform is None:
            return None
        warped = skimage_transform.warp(
            frame, inverse_map=transform.inverse, output_shape=STD_SIZE
        )
        warped = (warped * 255).astype(np.uint8)
        crop = _cut_patch(warped, transform(landmarks[index])[MOUTH_LANDMARKS])
        if crop is None:
            return None
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return crops


def _cut_patch(frame: np.ndarray, mouth: np.ndarray) -> np.ndarray | None:
    """A `CROP_SIZE` square around the mouth, nudged inside the frame, or None if it cannot fit.

    Upstream raises when the box sits more than five pixels outside the aligned frame. Here
    that returns None and drops the run rather than the clip: a mouth that far off frame means
    the alignment failed, and cropping the edge of the picture would feed the model something
    that is not a mouth.
    """
    half = CROP_SIZE // 2
    height, width = frame.shape[:2]
    if height < CROP_SIZE or width < CROP_SIZE:
        return None
    centre_x, centre_y = np.mean(mouth, axis=0)
    if not (-5 <= centre_x - half and centre_x + half <= width + 5):
        return None
    if not (-5 <= centre_y - half and centre_y + half <= height + 5):
        return None
    left = int(round(min(max(centre_x, half), width - half))) - half
    top = int(round(min(max(centre_y, half), height - half))) - half
    return frame[top : top + CROP_SIZE, left : left + CROP_SIZE]


def _windows(clip: Clip) -> list[np.ndarray]:
    """Every sampled run of the clip that yielded `FRAMES_PER_WINDOW` usable mouth crops."""
    capture = cv2.VideoCapture(str(clip.path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open media: {clip.path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < FRAMES_PER_WINDOW:
            raise ValueError(
                f"{clip.path} declares {frame_count} frame(s); LipForensics reads runs of "
                f"{FRAMES_PER_WINDOW}"
            )

        windows: list[np.ndarray] = []
        for start in _window_starts(frame_count):
            # The extra frames are the smoothing look-ahead: they steady the alignment of the
            # scored frames and are then discarded.
            frames = _read_window(
                capture, start, FRAMES_PER_WINDOW + SMOOTHING_WINDOW - 1
            )
            if len(frames) < FRAMES_PER_WINDOW:
                continue
            landmarks = _landmarks(frames)
            if landmarks is None:
                continue
            crops = _mouth_crops(frames, landmarks)
            if crops is None:
                continue
            windows.append(np.stack(crops[:FRAMES_PER_WINDOW]))
        return windows
    finally:
        capture.release()


def provenance() -> dict:
    """Identity of the code and weights this run scored with, verified rather than asserted.

    Every digest below is re-read from disk here, so the artifact records what was on the
    machine for the run rather than a constant copied out of the docstring. `face-alignment`
    is reported by version as well, because its landmarks are an input to every score.
    """
    import face_alignment
    import torch

    return {
        "detector": "lipforensics",
        "classifier": {
            "artifact": "lipforensics_ff.pth",
            "origin": "https://drive.google.com/file/d/"
                      "1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7 (upstream README)",
            "sha256": hashlib.sha256(
                _verify(WEIGHTS_PATH, WEIGHTS_SHA256).read_bytes()
            ).hexdigest(),
        },
        "upstream": {
            "repository": "https://github.com/ahaliassos/LipForensics",
            "revision": UPSTREAM_REVISION,
            "files": {
                relative: hashlib.sha256(
                    _upstream_source(relative).read_bytes()
                ).hexdigest()
                for relative in sorted(SOURCE_SHA256)
            },
        },
        "landmarks": {
            "library": f"face-alignment {face_alignment.__version__}",
            "face_detector": {
                "artifact": "s3fd-619a316812.pth",
                "sha256": hashlib.sha256(
                    _verify(SFD_PATH, SFD_SHA256).read_bytes()
                ).hexdigest(),
            },
            "landmark_model": {
                "artifact": "2DFAN4-11f355bf06.pth.tar",
                "sha256": hashlib.sha256(
                    _verify(FAN_PATH, FAN_SHA256).read_bytes()
                ).hexdigest(),
            },
        },
        "torch": torch.__version__,
        "device": DEVICE,
        "frames_per_window": FRAMES_PER_WINDOW,
        "windows": WINDOWS,
        "crop_size": CROP_SIZE,
        "input_size": INPUT_SIZE,
        "normalisation": {"mean": MOUTH_MEAN, "std": MOUTH_STD},
        "score_semantics": "sigmoid(mean logit over sampled 25-frame mouth runs)",
    }


def detect(clip: Clip) -> float:
    """Probability that `clip`'s mouth movement is forged, in `[0, 1]`.

    The logits of the sampled runs are averaged before the sigmoid, which is upstream's own
    video-level aggregation (`evaluate.py` averages clip logits per video). Averaging in logit
    space rather than after squashing keeps a single emphatic run from carrying the clip: the
    sigmoid saturates, and a mean of saturated probabilities is dominated by whichever run
    reached the ceiling first.

    Raises when no sampled run yields a tracked face throughout. That is an abstention, not a
    verdict, and the harness records it as an excluded clip — calling a video genuine because
    the detector never found a mouth to read would be a fabricated negative.
    """
    import torch

    windows = _windows(clip)
    if not windows:
        raise ValueError(
            f"no run of {FRAMES_PER_WINDOW} consecutive frames with a trackable face in "
            f"{clip.path}"
        )

    offset = (CROP_SIZE - INPUT_SIZE) // 2
    batch = np.stack(windows)[
        :, :, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE
    ]
    tensor = torch.from_numpy(batch).float().unsqueeze(1) / 255.0
    tensor = (tensor - MOUTH_MEAN) / MOUTH_STD
    with torch.inference_mode():
        logits = _classifier()(
            tensor.to(DEVICE), lengths=[FRAMES_PER_WINDOW] * len(windows)
        )
    return float(torch.sigmoid(logits.reshape(-1).mean()))
