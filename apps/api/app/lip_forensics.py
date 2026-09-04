"""Local mouth-dynamics evidence — LipForensics, run offline on CPU over mouth movement.

R5-T1 benchmarked and selected the pretrained model of Haliassos et al. (CVPR 2021), trained
on FaceForensics++. See `scripts/benchmark/models/lipforensics.py`, which is the wrapper that
measurement was taken through, and whose preprocessing, aggregation and pinned artifacts this
module reproduces exactly.

**Why a third detector rather than a second opinion.** The EfficientNet-B7 of R3 judges the
*appearance* of a single face crop and never sees motion. This one ignores appearance almost
entirely and classifies *mouth dynamics* over 25 consecutive frames, which is what makes it a
different question rather than a second copy of the same one. The two are never combined
(rule 11): they are separate rows, separate scores and separate scales.

**Why the signal is not called `lip_sync`.** Because it is not one, and the name would have
been a claim about the evidence rather than a label on it. Lip-sync detection asks whether a
sound track matches the mouth that appears to produce it, and needs both streams to answer;
this model is handed no audio at all. It reads the movement of the mouth in the picture and
was trained to separate forged facial motion from genuine facial motion — the upstream name
is about lips, the measurement is about forgery. A stored signal reading `lip_sync` would
have told every future reader of this database that an audio/video relationship had been
checked, and none ever was.

What this module produces is the model's own figure and nothing else:

- **one clip, one score.** Four evenly spaced runs of 25 frames are scored and the clip's
  score is `sigmoid` of the mean logit over them — upstream's own video-level aggregation, and
  the contract R5-T1 measured. Averaging in logit space rather than after squashing is part of
  that contract: the sigmoid saturates, and a mean of saturated probabilities is dominated by
  whichever run reached the ceiling first.
- **the score is not a verdict, and no threshold is applied to it here.** R5-T1 ran its
  confusion matrix at `0.5`, which is the harness default over 40 clips of one corpus and not
  an operating point anybody measured; R5-T3 rejected it by measurement and placed one of its
  own. That threshold lives in `app.risk_engine`, which since R5-T4 reads this signal as a
  calibrated decider. Nothing in this module knows what the number will be compared against,
  which is what keeps the reading and its interpretation separable.
- **no tracked face is an abstention, not a negative.** A clip in which no sampled run yields
  a face tracked through all 25 frames raises. Returning a low score for it would be a
  fabricated finding about media the classifier never saw.

**The architecture is executed, not retyped.** `models/resnet.py`, `models/tcn.py` and
`models/spatiotemporal_net.py` are loaded from the pinned upstream files this image carries,
because a re-implementation that drifts by one layer loads the same weights and quietly
measures a different model. They are executed under a private package name so that a directory
called `models` cannot shadow, or be shadowed by, anything else on the path.

**Three models are loaded because the classifier scores aligned mouth crops, not frames.** The
alignment is 68 landmarks per frame warped onto the LRW mean face; that is this detector's
preprocessing rather than a separate capability, so the S3FD detector and 2D-FAN landmark model
`face-alignment` reads are pinned and verified exactly as the classifier is.

Stateless by design, like `face_detector.py` and `audio_detector.py` and for the same reason:
no cached module, no module-level model, nothing held between calls. Every call re-verifies and
re-loads its artifacts, which is a fixed cost per job against an analysis measured in minutes,
and it keeps the weights and the landmark stack out of the worker's resident set between jobs.

Blocking CPU work, and the most expensive reading in the pipeline by a wide margin — R5-T1
measured a median of 129 s and a maximum of 856 s per clip, almost all of it landmark
detection, and this module took 235 s and peaked at 1.07 GB resident on a 15-second 854x480
clip in the worker's own container. The worker calls it synchronously, off any event loop,
and the lease in `app.worker` is what bounds a job that stops making progress; `app.limits`
says why an in-process deadline is not what guards this.

The port is faithful to what R5-T1 measured, and that is checked rather than asserted: this
module scored `Deepfakes_220_219` of the R5-T1 corpus at 0.4424368441104889, which is the
figure that benchmark recorded for the same clip to the last digit — under a different torch
(2.13 here, 2.11 there), a different Python and an uncompiled landmark network.

Deliberately shares no abstraction with `face_detector.py`, `audio_detector.py` or the provider
integrations, and must not grow one (AGENTS.md, abstraction rule).
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- Model artifacts ------------------------------------------------------------------------
#
# Every artifact pinned to one immutable origin and one exact digest, as R5-T1 pinned them. The
# Dockerfile fetches them at build time and refuses the image if the bytes do not hash to the
# digests below, and `_verify` checks them again when they are loaded — so the image carries
# exactly these bytes and the worker never downloads a model while serving a job.
#
# The forgery classifier Haliassos et al. trained on FaceForensics++ (Deepfakes, FaceSwap,
# Face2Face, NeuralTextures) and reported Table 2 of the paper with. Google Drive is where the
# upstream README links it and there is no other published origin, so the digest is doing more
# work here than it does for a Hugging Face revision: it is the whole of the artifact's
# identity.
WEIGHTS_ORIGIN = (
    "https://drive.google.com/file/d/1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7 (upstream README)"
)
WEIGHTS_FILENAME = "lipforensics_ff.pth"
WEIGHTS_SHA256 = "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"

# The upstream checkout this module executes and reads, at the commit R5-T1 pinned. Digests per
# file rather than for the repository, because these five files are what is actually loaded and
# an edited `tcn.py` would change the model as surely as a swapped checkpoint would.
UPSTREAM_REPOSITORY = "https://github.com/ahaliassos/LipForensics"
UPSTREAM_REVISION = "d0bf5553bfb9676f1771d590472b26a3a76de894"
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

# The landmark pair, published by their author as single files rather than in a repository. They
# are inputs to every score — a different landmark model is a different alignment and therefore
# a different number — so they are pinned exactly like the classifier is.
LANDMARK_ORIGIN = "https://www.adrianbulat.com/downloads/python-fan"
SFD_FILENAME = "s3fd-619a316812.pth"
SFD_SHA256 = "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"
FAN_FILENAME = "2DFAN4-11f355bf06.pth.tar"
FAN_SHA256 = "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa"

# Where the image puts them, in R5-T1's own cache layout so that a developer can point this at
# the directory that benchmark filled without editing code. Configuration, never anything a
# request can influence.
DEFAULT_MODEL_DIR = Path("/models/lip_forensics")
MODEL_DIR_ENV = "DEEPGUARD_LIPFORENSICS_MODEL_DIR"

# The upstream package is executed under a private name so that importing a directory called
# `models` cannot shadow, or be shadowed by, anything else on the path.
UPSTREAM_PACKAGE = "_deepguard_lipforensics_upstream"

# --- Runtime contract -----------------------------------------------------------------------
#
# Preprocessing fixed by the training recipe (upstream `preprocessing/crop_mouths.py` and the
# transform composed in `evaluate.py`), not free parameters. Reproduced from R5-T1 exactly: 25
# consecutive frames, each warped onto the LRW mean face by five stable points, the mouth cut at
# 96x96 and centre-cropped to 88x88 in grayscale under the recipe's own normalisation. A
# different crop or a different normalisation is a different measurement.
FRAMES_PER_WINDOW = 25
STD_SIZE = (256, 256)
STABLE_POINTS = [33, 36, 39, 42, 45]
MOUTH_LANDMARKS = slice(48, 68)
CROP_SIZE = 96
INPUT_SIZE = 88
SMOOTHING_WINDOW = 12
MOUTH_MEAN, MOUTH_STD = 0.421, 0.165

# How many 25-frame runs are sampled from the clip, evenly spread. Four is what R5-T1 measured
# at — the same number of runs upstream evaluates over its first 110 frames — so changing it
# changes what that benchmark's figures describe; it is configurable for the same reason the
# frame sample of the face classifier is, and the value used is recorded on every signal rather
# than assumed from this default.
DEFAULT_WINDOWS = 4
WINDOWS_ENV = "DEEPGUARD_LIPFORENSICS_WINDOWS"

# The device every score is produced on, and deliberately not configurable here. R5-T1 defaulted
# to CPU because cuDNN kernel selection makes GPU scores reproducible only against the same
# card, and this image carries a CPU-only torch in any case (`requirements.txt`). Recorded on
# every signal, because it is part of the identity of the number.
DEVICE = "cpu"

# Whether the landmark network is run through `torch.compile`. False, and stated as a constant
# rather than left to the library's default so that a signal can record it — see `_landmarker`,
# which says why, and `app.detection`, which writes it down beside the score.
LANDMARK_COMPILE = False

# How many per-window logits one reading may leave behind in its metadata. The sample is four by
# default and this is far above it, so nothing is truncated in practice; the cap exists so a
# deployment that raises `DEEPGUARD_LIPFORENSICS_WINDOWS` cannot grow the signal document
# without limit. `windows_scored` records how many there were either way.
MAX_PERSISTED_WINDOW_LOGITS = 32


class LipForensicsError(Exception):
    """Base class for every failure raised by this module."""


class LipForensicsModelUnavailable(LipForensicsError):
    """A pinned artifact is missing, is not the bytes that were pinned, or will not load.

    A server/deployment fault, not a fact about the media: nothing was inferred, so nothing is
    known about the video either way. Also raised when a library this detector needs is not
    installed in the image.
    """


class LipForensicsMediaError(LipForensicsError):
    """The media could not be opened, or is shorter than one run of frames.

    A clip of fewer than `FRAMES_PER_WINDOW` frames is here rather than beside the abstention
    below on purpose: nothing was sampled at all, so there is not even a run in which a face
    could have been looked for.
    """


class LipForensicsNoTrackedFace(LipForensicsError):
    """No sampled run yielded a face tracked through all of its frames.

    An abstention, and kept separate from every other failure here because it is the one that is
    an ordinary property of a video rather than a fault: footage with no person in it, a face
    that leaves frame mid-run, or an angle the landmark model cannot follow all land here. It is
    emphatically not a finding that the media is genuine — the classifier saw nothing at all.
    """


class LipForensicsInferenceError(LipForensicsError):
    """Torch failed while inferring, or returned something this module cannot read."""


@dataclass(frozen=True)
class WindowLogit:
    """One sampled run's logit, with the frame the run started at.

    `start_frame` is the position in the decoded video that DeepGuard chose to sample — its own
    slicing of the clip, not a detection the model reported. `logit` is the single raw output
    the network emitted for that run, before any sigmoid: it is what the clip's score is the
    mean of, and recording it after squashing would record something the aggregation never used.
    """

    start_frame: int
    logit: float


@dataclass(frozen=True)
class LipForensicsEvidence:
    """One clip's score, the runs behind it, and exactly what produced them.

    The provenance fields are here so a stored signal can be reproduced: a different checkpoint,
    a different upstream revision, a different landmark model or a different window count is a
    different measurement, and a reader of the database should never have to assume which one a
    stored number came from.

    `score` is the clip-level figure R5-T1's contract defines — `sigmoid` of the mean of
    `window_logits` — and it is the model's own output carrying no threshold, class or verdict.
    """

    weights_origin: str
    weights_sha256: str
    upstream_repository: str
    upstream_revision: str
    source_sha256: dict[str, str]
    landmark_library: str
    # Whether that library compiled the landmark network or ran it eagerly. Not part of the
    # artifacts' identity — it generates code for the same graph — but recorded because a
    # reader should not have to assume how a stored number was produced.
    landmark_compiled: bool
    face_detector_sha256: str
    landmark_model_sha256: str
    torch_version: str
    device: str
    frames_per_window: int
    crop_size: int
    input_size: int
    # What the sampling actually did: how many runs were asked for, how many yielded
    # `frames_per_window` frames off the decoder, and how many of those survived landmark
    # tracking and mouth cropping to reach the classifier. The three differ on ordinary media
    # and the gaps between them are what keep `score` from reading as a statement about the
    # whole clip.
    windows_requested: int
    windows_read: int
    windows_scored: int
    score: float
    window_logits: tuple[WindowLogit, ...]


def _import_cv2():
    """The OpenCV module, or a refusal that is a signal rather than a crashed job.

    Every use goes through here rather than importing at module scope, for the two reasons
    `app.face_detector` gives at the same place: the API process imports this module for its
    error types and should not pay for OpenCV to do it, and an image built without the
    dependency must produce a `FAILED` mouth-dynamics signal like any other unavailable model, not a
    `ModuleNotFoundError` that escapes `app.detection` and takes the whole analysis down with it
    (AGENTS.md, error-handling rule).
    """
    try:
        import cv2
    except ImportError as error:
        raise LipForensicsModelUnavailable("opencv is not installed") from error

    return cv2


def _import_torch():
    """The torch module, or a refusal. The same contract as `_import_cv2`, for the same reason."""
    try:
        import torch
    except ImportError as error:
        raise LipForensicsModelUnavailable("torch is not installed") from error

    return torch


def _import_skimage_transform():
    """scikit-image's transform module, or a refusal. The same contract as `_import_cv2`."""
    try:
        from skimage import transform
    except ImportError as error:
        raise LipForensicsModelUnavailable("scikit-image is not installed") from error

    return transform


def _import_face_alignment():
    """The `face_alignment` package, or a refusal. The same contract as `_import_cv2`.

    This one is the likeliest of the four to be missing, because it is installed with
    `--no-deps` (see `requirements.txt`): an image whose dependency list drifted would fail here
    with a `FAILED` signal naming the library rather than taking an analysis down.
    """
    try:
        import face_alignment
    except ImportError as error:
        raise LipForensicsModelUnavailable("face-alignment is not installed") from error

    return face_alignment


def _model_dir() -> Path:
    configured = os.getenv(MODEL_DIR_ENV, "").strip()

    return Path(configured) if configured else DEFAULT_MODEL_DIR


def windows() -> int:
    """How many runs to sample, from the environment or the benchmarked default.

    Unset and empty are the same thing and both mean the default, the way `app.limits` reads its
    bounds and for the same reason: a compose file that lists the variable without setting it
    passes the empty string through.

    Anything else has to be a positive integer. A malformed value raises rather than falling
    back, because a deployment that wrote one meant something by it and silently sampling four
    runs instead would leave an operator certain of a coverage the worker is not giving them.
    """
    configured = os.getenv(WINDOWS_ENV, "").strip()
    if not configured:
        return DEFAULT_WINDOWS

    try:
        sampled = int(configured)
    except ValueError:
        raise LipForensicsModelUnavailable(
            f"{WINDOWS_ENV} is not a whole number of windows: {configured!r}"
        ) from None

    if sampled < 1:
        raise LipForensicsModelUnavailable(
            f"{WINDOWS_ENV} must be at least 1, not {sampled}"
        )

    return sampled


def _verify(path: Path, expected_sha256: str) -> Path:
    """Fail loudly unless `path` holds exactly the bytes this module pinned.

    R5-T1's discipline, kept, and it covers the upstream *source* as well as the weights: the
    architecture is executed from disk, so an edited `tcn.py` would change the model as surely
    as a swapped checkpoint would, and neither may happen unnoticed behind a stored score.

    The Dockerfile checks the same digests at build time and this is not redundant with it — a
    bind mount, a rebuilt layer or a developer pointing `DEEPGUARD_LIPFORENSICS_MODEL_DIR` at a
    benchmark cache all put bytes in front of this process that no build ever saw.
    """
    if not path.is_file():
        raise LipForensicsModelUnavailable(f"pinned artifact is missing at '{path}'")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise LipForensicsModelUnavailable(
            f"'{path}' does not match its pinned digest: expected {expected_sha256}, "
            f"got {digest}"
        )

    return path


def _source_root(directory: Path) -> Path:
    """The pinned upstream checkout inside the model directory.

    Named by revision, which is R5-T1's layout and worth keeping: two revisions can sit side by
    side in a developer's cache without one being mistaken for the other.
    """
    return directory / f"lipforensics-{UPSTREAM_REVISION}"


def _upstream_source(directory: Path, relative: str) -> Path:
    """A verified file from the pinned upstream checkout."""
    return _verify(_source_root(directory) / relative, SOURCE_SHA256[relative])


def _upstream_architecture(directory: Path):
    """Execute the pinned upstream `models` package and return `spatiotemporal_net`.

    The three modules are loaded in dependency order under `UPSTREAM_PACKAGE`, whose `__path__`
    is the checkout, so upstream's own relative imports resolve to the files whose digests were
    just checked and to nothing else.

    Re-executed on every call rather than cached, which is this module's statelessness rather
    than an oversight: three small source files cost nothing beside the weights, and a module
    left in `sys.modules` between jobs would be exactly the retained state the docstring says
    there is none of.
    """
    package = types.ModuleType(UPSTREAM_PACKAGE)
    package.__path__ = [str(_source_root(directory) / "models")]
    sys.modules[UPSTREAM_PACKAGE] = package

    try:
        for name in ("resnet", "tcn", "spatiotemporal_net"):
            path = _upstream_source(directory, f"models/{name}.py")
            spec = importlib.util.spec_from_file_location(
                f"{UPSTREAM_PACKAGE}.{name}", path
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    except LipForensicsError:
        raise
    except Exception as error:
        raise LipForensicsModelUnavailable(
            f"the pinned upstream architecture could not be executed "
            f"({type(error).__name__}: {error})"
        ) from error

    return sys.modules[f"{UPSTREAM_PACKAGE}.spatiotemporal_net"]


def _classifier(directory: Path):
    """Load ResNet-18 + MS-TCN over the pinned bytes, with the FF++ forgery weights, in eval mode.

    Built from upstream's own `Lipreading` and upstream's own config rather than through its
    `get_model`, which reads that config relative to the process working directory and puts the
    model on `cuda:0` — neither of which a worker should depend on. R5-T1 loads it exactly this
    way, and a stored score has to have come off the same construction.

    No torch version allowlist, unlike `app.face_detector`. That module runs an *exported*
    artifact published for one cohort; this is a plain `state_dict` loaded into an architecture
    executed from source, so there is no exported program whose runtime has to match. The
    running version is recorded on every signal instead.
    """
    torch = _import_torch()
    architecture = _upstream_architecture(directory)

    config = json.loads(
        _upstream_source(directory, "models/configs/lrw_resnet18_mstcn.json").read_text(
            "utf-8"
        )
    )

    try:
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
            _verify(directory / "weights" / WEIGHTS_FILENAME, WEIGHTS_SHA256),
            map_location=DEVICE,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
    except LipForensicsError:
        raise
    except Exception as error:
        raise LipForensicsModelUnavailable(
            f"the LipForensics classifier could not be loaded "
            f"({type(error).__name__}: {error})"
        ) from error

    return model.to(DEVICE).eval(), torch.__version__


def _landmarker(directory: Path):
    """The pinned S3FD + 2D-FAN pair, reading only from the verified local directory.

    `face_alignment` fetches its weights through the torch hub cache, so hub is pointed at this
    detector's own directory: the files are verified here first, and the worker therefore never
    depends on adrianbulat.com being reachable or on what an earlier process left in
    `~/.cache/torch`.
    """
    face_alignment = _import_face_alignment()
    torch = _import_torch()

    root = directory / "face-alignment"
    _verify(root / "checkpoints" / SFD_FILENAME, SFD_SHA256)
    _verify(root / "checkpoints" / FAN_FILENAME, FAN_SHA256)

    try:
        torch.hub.set_dir(str(root))
        landmarker = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=DEVICE,
            flip_input=False,
            # Eager, not compiled. `face_alignment` defaults to wrapping the landmark network
            # in `torch.compile`, whose CPU backend shells out to a C++ compiler — which this
            # image does not carry and should not: a build toolchain in a production container
            # is a quarter of a gigabyte of attack surface to JIT one model, and the failure it
            # produced was an `InductorError` escaping mid-analysis rather than a clean
            # fallback.
            #
            # Not a departure from R5-T1's provenance. Compilation is a code-generation
            # strategy, not part of the model: the same weights, the same graph and the same
            # arithmetic, and the score this produces was checked against the figure R5-T1
            # recorded for the same clip. What the artifact identity has to pin is which
            # bytes were loaded, and `_verify` above still does that. It is recorded on every
            # signal all the same, because a reader should not have to assume it.
            compile=LANDMARK_COMPILE,
        )
    except Exception as error:
        raise LipForensicsModelUnavailable(
            f"the landmark models at '{root}' could not be loaded "
            f"({type(error).__name__}: {error})"
        ) from error

    return landmarker, face_alignment.__version__


def _window_starts(frame_count: int, sampled: int) -> list[int]:
    """First frame index of each sampled run, evenly spread and non-overlapping."""
    last_start = frame_count - FRAMES_PER_WINDOW
    if last_start < 0:
        return []

    starts = np.linspace(0, last_start, sampled).astype(int)

    return sorted({int(start) for start in starts})


def _read_window(capture, start: int, wanted: int) -> list:
    """Up to `wanted` consecutive frames from `start`, as BGR arrays."""
    cv2 = _import_cv2()

    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    return frames


def _landmark_area(landmarks: np.ndarray) -> float:
    extent = landmarks.max(axis=0) - landmarks.min(axis=0)

    return float(extent[0] * extent[1])


def _landmarks(landmarker, frames: list) -> list | None:
    """68 landmarks per frame, or None if any frame in the run has no face.

    All-or-nothing on purpose: the model reads a run of frames as one movement, and a run
    stitched across a frame whose alignment was guessed is a movement that never happened. The
    largest face is taken where several are found, so the same person is followed throughout.
    """
    cv2 = _import_cv2()

    found = []
    for frame in frames:
        faces = landmarker.get_landmarks_from_image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        if not faces:
            return None
        found.append(max(faces, key=_landmark_area))

    return found


def _cut_patch(frame: np.ndarray, mouth: np.ndarray) -> np.ndarray | None:
    """A `CROP_SIZE` square around the mouth, nudged inside the frame, or None if it cannot fit.

    Upstream raises when the box sits more than five pixels outside the aligned frame. Here that
    returns None and drops the run rather than the clip: a mouth that far off frame means the
    alignment failed, and cropping the edge of the picture would feed the model something that
    is not a mouth. R5-T1 made the same choice.
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


def _mouth_crops(mean_face: np.ndarray, frames: list, landmarks: list) -> list | None:
    """Aligned 96x96 grayscale mouth crops, following upstream `crop_mouths.py`.

    Landmarks are averaged over a 12-frame look-ahead to take out tracker jitter, each frame is
    similarity-warped onto the mean face by five points around the eyes and nose, and the mouth
    is cut from the warped frame around the mean of landmarks 48-68. Frames past the last full
    smoothing window reuse the last transform, exactly as upstream does at the end of a video.
    """
    cv2 = _import_cv2()
    skimage_transform = _import_skimage_transform()

    crops = []
    transform = None
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


def _mouth_windows(video_path: Path, sampled: int, directory: Path, landmarker):
    """Every sampled run of the clip that yielded `FRAMES_PER_WINDOW` usable mouth crops.

    Evenly spread rather than the opening seconds, which is R5-T1's one deliberate departure
    from upstream's evaluation protocol and the reason it gives: a manipulation is not uniformly
    distributed, and the first four seconds are the part an editor is most likely to have left
    alone.

    Returns the runs alongside the frame each started at and how many runs the decoder could
    supply at all, so the caller can record what the sample actually covered.
    """
    cv2 = _import_cv2()

    mean_face = np.load(_upstream_source(directory, "preprocessing/20words_mean_face.npy"))

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise LipForensicsMediaError(f"media could not be opened: '{video_path}'")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < FRAMES_PER_WINDOW:
            raise LipForensicsMediaError(
                f"media declares {frame_count} frame(s) and LipForensics reads runs of "
                f"{FRAMES_PER_WINDOW}: '{video_path}'"
            )

        found: list[tuple[int, np.ndarray]] = []
        read = 0
        for start in _window_starts(frame_count, sampled):
            # The extra frames are the smoothing look-ahead: they steady the alignment of the
            # scored frames and are then discarded.
            frames = _read_window(capture, start, FRAMES_PER_WINDOW + SMOOTHING_WINDOW - 1)
            if len(frames) < FRAMES_PER_WINDOW:
                continue

            read += 1
            landmarks = _landmarks(landmarker, frames)
            if landmarks is None:
                continue

            crops = _mouth_crops(mean_face, frames, landmarks)
            if crops is None:
                continue

            found.append((start, np.stack(crops[:FRAMES_PER_WINDOW])))

        return found, read
    finally:
        capture.release()


def analyze_lip_forensics(
    video_path: Path,
    *,
    model_dir: Path | str | None = None,
) -> LipForensicsEvidence:
    """Score the mouth movement of evenly sampled runs of one clip and report the clip's figure.

    `video_path` is the prepared artifact the rest of the pipeline reads — the derivative when
    one was transcoded, the original when it was already canonical. It is only read.

    The clip's score is `sigmoid` of the mean of the per-run logits, which is the contract R5-T1
    benchmarked and upstream's own video-level aggregation. The mean is taken in logit space and
    squashed once, never the other way around: the sigmoid saturates, so a mean of probabilities
    would be carried by whichever run reached the ceiling first.

    Raises `LipForensicsNoTrackedFace` when no sampled run yields a face tracked through all of
    its frames. That is an abstention, not a verdict — calling a video genuine because the
    detector never found a mouth to read would be a fabricated negative.

    Raises `LipForensicsModelUnavailable` when an artifact is missing, is not the pinned bytes,
    will not load, or a library it needs is absent; `LipForensicsMediaError` when the media
    cannot be read or is shorter than one run; and `LipForensicsInferenceError` when torch fails
    or answers unreadably.

    Blocking and CPU-bound, and expensive: R5-T1 measured a median of 129 s per clip, almost all
    of it in the landmark model, which is asked about 36 frames per sampled run.
    """
    torch = _import_torch()
    directory = Path(model_dir) if model_dir is not None else _model_dir()
    sampled = windows()

    # Landmarks first, and the classifier only once there is something to give it: a clip with
    # no trackable face is an ordinary outcome and is worth reaching without also paying to
    # verify and load 137 MiB of forgery weights for it.
    landmarker, landmark_library = _landmarker(directory)
    found, read = _mouth_windows(video_path, sampled, directory, landmarker)
    if not found:
        raise LipForensicsNoTrackedFace(
            f"no run of {FRAMES_PER_WINDOW} consecutive frames with a trackable face in any "
            f"of {read} run(s) read from {sampled} sampled in '{video_path}'"
        )

    classifier, torch_version = _classifier(directory)

    offset = (CROP_SIZE - INPUT_SIZE) // 2
    batch = np.stack([window for _, window in found])[
        :, :, offset : offset + INPUT_SIZE, offset : offset + INPUT_SIZE
    ]
    tensor = torch.from_numpy(batch).float().unsqueeze(1) / 255.0
    tensor = (tensor - MOUTH_MEAN) / MOUTH_STD

    try:
        with torch.inference_mode():
            logits = classifier(
                tensor.to(DEVICE), lengths=[FRAMES_PER_WINDOW] * len(found)
            )
        flattened = logits.reshape(-1)
    except Exception as error:
        raise LipForensicsInferenceError(
            f"mouth-dynamics inference failed ({type(error).__name__}: {error})"
        ) from error

    if flattened.shape != (len(found),):
        raise LipForensicsInferenceError(
            f"the classifier returned {tuple(flattened.shape)} logit(s) for "
            f"{len(found)} sampled run(s)"
        )

    values = [float(value) for value in flattened]
    if not all(np.isfinite(values)):
        raise LipForensicsInferenceError(
            f"the classifier returned non-finite logits: {values}"
        )

    # The aggregation, in the order it has to happen: mean the logits, squash once. Computed
    # through torch rather than by hand so the arithmetic is the arithmetic R5-T1 measured.
    score = float(torch.sigmoid(flattened.mean()))

    window_logits = tuple(
        WindowLogit(start_frame=start, logit=logit)
        for (start, _), logit in zip(found, values)
    )

    logger.info(
        "LipForensics scored %d run(s) of %d frames from %d run(s) read of %d sampled in %s",
        len(window_logits),
        FRAMES_PER_WINDOW,
        read,
        sampled,
        video_path,
    )

    return LipForensicsEvidence(
        weights_origin=WEIGHTS_ORIGIN,
        weights_sha256=WEIGHTS_SHA256,
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_revision=UPSTREAM_REVISION,
        source_sha256=dict(SOURCE_SHA256),
        landmark_library=f"face-alignment {landmark_library}",
        landmark_compiled=LANDMARK_COMPILE,
        face_detector_sha256=SFD_SHA256,
        landmark_model_sha256=FAN_SHA256,
        torch_version=torch_version,
        device=DEVICE,
        frames_per_window=FRAMES_PER_WINDOW,
        crop_size=CROP_SIZE,
        input_size=INPUT_SIZE,
        windows_requested=sampled,
        windows_read=read,
        windows_scored=len(window_logits),
        score=score,
        window_logits=window_logits[:MAX_PERSISTED_WINDOW_LOGITS],
    )
