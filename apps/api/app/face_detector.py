"""Local face-manipulation evidence — the DFDC-winning EfficientNet-B7, run offline on CPU.

R3-T1 benchmarked and selected Selim Seferbekov's winning entry to the Deepfake Detection
Challenge (upstream `selimsef/dfdc_deepfake_challenge` @ `89c62904`, MIT), republished with
immutable digests by Facetorch as a `torch.export` artifact. See
`scripts/benchmark/models/face_manipulation.py`, which is the wrapper that measurement was
taken through, and whose preprocessing this module reproduces exactly.

What this module produces is the model's own probability and nothing else:

- **one artifact, one score.** The classifier is asked about the face in each of eight evenly
  spaced frames and the clip's score is the mean of those probabilities. That is the contract
  R3-T1 measured, and it is preserved here unchanged — no other reduction of the frames is
  reported, because a benchmark taken over the mean does not describe the maximum.
- **the score is not a verdict.** R3-T1 ran its confusion matrix at `0.8`, which is a
  *benchmark* operating point over 40 clips of one corpus (FF++ c23). It is not a production
  threshold, nothing here compares against it, and no `Fake`/`Real` reading of the number
  exists in this codebase. Calibrating it is R4's work.
- **no face is an abstention, not a negative.** A clip in which the locator finds no face in
  any sampled frame raises. Returning a low score for it would be a fabricated finding about
  media the classifier never saw.

Two models are loaded because the classifier scores *faces*, not frames: it was trained on
crops, and feeding it whole frames measures the background instead. Locating the face is part
of this detector's preprocessing rather than a separate capability, so YuNet is pinned and
verified exactly like the classifier is.

Stateless by design, like `audio_detector.py` and for the same reason: no cached module, no
module-level model, nothing held between calls. On this host the artifact costs ~7.5 s to
load and ~1.0 s to verify against ~5.5 s of inference, which is a fixed cost per job the
worker pays once against an analysis measured in minutes — and it keeps ~600 MB out of the
worker's resident set between jobs, which matters under the container's 4 GB ceiling.

Blocking CPU work. The worker calls it synchronously, off any event loop.

Deliberately shares no abstraction with `audio_detector.py`, `nvidia_video.py` or the other
integrations, and must not grow one (AGENTS.md, abstraction rule). Its evidence is never
fused with theirs (rule 11).
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# --- Model artifacts ------------------------------------------------------------------------
#
# Both pinned to one immutable revision and one exact digest, as R3-T1 pinned them. The
# Dockerfile fetches these URLs at build time and refuses the image if the bytes do not hash to
# the digests below, and `_verify` checks them again when they are loaded — so the image
# carries exactly these bytes and the worker never downloads a model while serving a job.
CLASSIFIER_REPOSITORY = "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
CLASSIFIER_REVISION = "4acc494f37eb63d7457166eff2acb45c5b04b9a6"
CLASSIFIER_FILENAME = "model-torch2.11.pt2"
CLASSIFIER_URL = (
    f"https://huggingface.co/{CLASSIFIER_REPOSITORY}/resolve/{CLASSIFIER_REVISION}/"
    f"{CLASSIFIER_FILENAME}"
)
CLASSIFIER_SHA256 = "97b49a70174c0d4f72d9d510d817bdc49a907af9af0242a6a1ba934a7cc9e4b7"

# The face locator. `opencv/opencv_zoo` is a Git LFS repository, so the artifact is fetched
# from the media host rather than from the HTML page the plain path serves.
LOCATOR_REPOSITORY = "opencv/opencv_zoo"
LOCATOR_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
LOCATOR_FILENAME = "face_detection_yunet_2023mar.onnx"
LOCATOR_URL = (
    f"https://media.githubusercontent.com/media/{LOCATOR_REPOSITORY}/{LOCATOR_REVISION}/"
    f"models/face_detection_yunet/{LOCATOR_FILENAME}"
)
LOCATOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# Where the image puts them. Overridable so a developer can point at the cache R3-T1 filled
# without editing code; it is configuration, never anything a request can influence.
DEFAULT_MODEL_DIR = Path("/models/face_manipulation")
MODEL_DIR_ENV = "DEEPGUARD_FACE_MODEL_DIR"

# --- Runtime contract -----------------------------------------------------------------------
#
# The minor versions of torch this exported artifact may be run under here, as
# `(major, minor)`. An allowlist rather than a range, because each entry is a claim someone
# checked rather than an interval someone assumed:
#
# - `(2, 11)` is the artifact's own published support cohort, per the Facetorch model card,
#   and the version R3-T1's benchmark venv measured under;
# - `(2, 13)` is what this image ships (`requirements.txt`), and it is listed because the
#   artifact was run under both and compared: identical input gives bit-identical logits, to
#   a maximum absolute difference of exactly 0.0. The R3-T1 numbers therefore describe this
#   runtime as well as the one they were taken on.
#
# Anything else is refused rather than run. A torch upgrade that lands here has to repeat that
# comparison and add its version deliberately — silently scoring media under an unverified
# runtime would put a number behind a stored signal that no measurement covers.
VERIFIED_TORCH_VERSIONS = frozenset({(2, 11), (2, 13)})

# Preprocessing fixed by the checkpoint's training recipe, not free parameters. Reproduced
# from R3-T1 exactly: a 380x380 RGB face crop under ImageNet normalisation, one binary logit
# per face. A different crop or a different normalisation is a different measurement.
INPUT_SIZE = 380
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Crop margin around the detected box. Seferbekov's pipeline trains on face crops enlarged by
# roughly a third, which keeps hairline and jaw — the edges where a swap blends into the
# original head, and where the artefacts actually live — inside frame.
CROP_MARGIN = 1 / 3

# The locator's confidence floor, and YuNet's NMS and top-k bounds, as R3-T1 set them.
FACE_SCORE_THRESHOLD = 0.6
LOCATOR_NMS_THRESHOLD = 0.3
LOCATOR_TOP_K = 5000

# How many frames are sampled from the clip, evenly spaced. Eight is what R3-T1 measured at,
# so changing it changes what the benchmark's figures describe; it is configurable for the
# same reason the timeouts are — a deployment may need to trade latency against coverage —
# and the value used is recorded on every signal rather than assumed from this default.
DEFAULT_FRAME_SAMPLES = 8
FRAME_SAMPLES_ENV = "DEEPGUARD_FACE_FRAMES"

# How many per-frame probabilities one reading may leave behind in its metadata. The sample is
# eight by default and this is far above it, so nothing is truncated in practice; the cap
# exists so a deployment that raises `DEEPGUARD_FACE_FRAMES` cannot grow the signal document
# without limit. `frames_scored` records how many there were either way.
MAX_PERSISTED_FRAME_SCORES = 64


class FaceDetectorError(Exception):
    """Base class for every failure raised by this module."""


class FaceDetectorModelUnavailable(FaceDetectorError):
    """A pinned artifact is missing, is not the bytes that were pinned, or will not load.

    A server/deployment fault, not a fact about the media: nothing was inferred, so nothing is
    known about the video either way. Also raised when the running torch is not one of the
    versions this artifact has been verified under.
    """


class FaceDetectorMediaError(FaceDetectorError):
    """The media could not be opened, or declares nothing decodable."""


class FaceDetectorNoFaceFound(FaceDetectorError):
    """No sampled frame yielded a face, so the classifier was never asked.

    An abstention, and kept separate from every other failure here because it is the one that
    is an ordinary property of a video rather than a fault: footage of a landscape, a screen
    recording or a crowd shot too wide for the locator all land here. It is emphatically not a
    finding that the media is genuine — the classifier saw nothing at all.
    """


class FaceDetectorInferenceError(FaceDetectorError):
    """Torch failed while inferring, or returned something this module cannot read."""


@dataclass(frozen=True)
class FrameScore:
    """One sampled frame's probability, with the frame it was taken from.

    `frame_index` is the position in the decoded video that DeepGuard chose to sample — its
    own slicing of the clip, not a detection the model reported. `probability` is the sigmoid
    of the single logit the classifier emitted for the face crop taken from that frame.
    """

    frame_index: int
    probability: float


@dataclass(frozen=True)
class FaceManipulationEvidence:
    """One clip's score, the frames behind it, and exactly what produced them.

    The provenance fields are here so a stored signal can be reproduced: a different classifier
    revision, a different locator or a different frame count is a different measurement, and
    a reader of the database should never have to assume which one a stored number came from.

    `score` is the clip-level figure R3-T1's contract defines — the mean of `frame_scores` —
    and it is a probability the model emitted, carrying no threshold, class or verdict.
    """

    classifier_repository: str
    classifier_revision: str
    classifier_sha256: str
    locator_repository: str
    locator_revision: str
    locator_sha256: str
    torch_version: str
    input_size: int
    crop_margin: float
    face_score_threshold: float
    # What the sampling actually did: how many frames were asked for, how many decoded, and
    # how many of those yielded a face the classifier could be given. The three differ on
    # ordinary media and the gaps between them are what make `score` readable.
    frames_requested: int
    frames_decoded: int
    frames_scored: int
    score: float
    frame_scores: tuple[FrameScore, ...]


def _import_cv2():
    """The OpenCV module, or a refusal that is a signal rather than a crashed job.

    Every use goes through here rather than importing at module scope, for two reasons that
    both matter. The API process imports this module for its error types and should not pay
    for OpenCV to do it; and an image built without the dependency must produce a `FAILED`
    face-manipulation signal like any other unavailable model, not a `ModuleNotFoundError`
    that escapes `app.detection` and takes the whole analysis down with it (AGENTS.md,
    error-handling rule).
    """
    try:
        import cv2
    except ImportError as error:
        raise FaceDetectorModelUnavailable("opencv is not installed") from error

    return cv2


def _import_torch():
    """The torch module, or a refusal. The same contract as `_import_cv2`, for the same reason."""
    try:
        import torch
    except ImportError as error:
        raise FaceDetectorModelUnavailable("torch is not installed") from error

    return torch


def _model_dir() -> Path:
    configured = os.getenv(MODEL_DIR_ENV, "").strip()

    return Path(configured) if configured else DEFAULT_MODEL_DIR


def frame_samples() -> int:
    """How many frames to sample, from the environment or the benchmarked default.

    Unset and empty are the same thing and both mean the default, the way `app.limits` reads
    its bounds: a compose file that lists the variable without setting it passes the empty
    string through, and failing on that would refuse a perfectly ordinary deployment.

    Anything else has to be a positive integer. A malformed value raises rather than falling
    back, because a deployment that wrote one meant something by it and silently sampling
    eight frames instead would leave an operator certain of a coverage the worker is not
    giving them.
    """
    configured = os.getenv(FRAME_SAMPLES_ENV, "").strip()
    if not configured:
        return DEFAULT_FRAME_SAMPLES

    try:
        samples = int(configured)
    except ValueError:
        raise FaceDetectorModelUnavailable(
            f"{FRAME_SAMPLES_ENV} is not a whole number of frames: {configured!r}"
        ) from None

    if samples < 1:
        raise FaceDetectorModelUnavailable(
            f"{FRAME_SAMPLES_ENV} must be at least 1, not {samples}"
        )

    return samples


def _verify(path: Path, expected_sha256: str) -> Path:
    """Fail loudly unless `path` holds exactly the bytes this module pinned.

    R3-T1's discipline, kept: a revision in a docstring is a claim, and a digest checked at
    load is the claim being kept. The Dockerfile checks the same digests at build time, and
    this is not redundant with it — a bind mount, a rebuilt layer or a developer pointing
    `DEEPGUARD_FACE_MODEL_DIR` at a local cache all put bytes in front of this process that
    no build ever saw. A signal whose weights were swapped underneath it would carry a score
    about a model nobody can name afterwards.
    """
    if not path.is_file():
        raise FaceDetectorModelUnavailable(f"pinned weights are missing at '{path}'")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise FaceDetectorModelUnavailable(
            f"'{path}' does not match its pinned digest: expected {expected_sha256}, "
            f"got {digest}"
        )

    return path


def _load_classifier(path: Path):
    """Load the exported B7 over the pinned bytes, under a verified torch.

    No `.eval()`: an exported program has its training-mode behaviour already baked in, and
    torch raises rather than accept the call.
    """
    torch = _import_torch()

    try:
        cohort = tuple(int(part) for part in torch.__version__.split(".")[:2])
    except ValueError as error:
        raise FaceDetectorModelUnavailable(
            f"torch reports an unreadable version: {torch.__version__!r}"
        ) from error

    if cohort not in VERIFIED_TORCH_VERSIONS:
        verified = ", ".join(
            f"{major}.{minor}" for major, minor in sorted(VERIFIED_TORCH_VERSIONS)
        )
        raise FaceDetectorModelUnavailable(
            f"torch {torch.__version__} is not a version this artifact has been verified "
            f"under ({verified}); see app/face_detector.py"
        )

    _verify(path, CLASSIFIER_SHA256)

    try:
        return torch.export.load(str(path)).module(), torch.__version__
    except Exception as error:
        raise FaceDetectorModelUnavailable(
            f"the classifier at '{path}' could not be loaded "
            f"({type(error).__name__}: {error})"
        ) from error


def _locator(path: Path, width: int, height: int):
    """Build YuNet over the pinned bytes, sized for this video's frames.

    """
    cv2 = _import_cv2()

    _verify(path, LOCATOR_SHA256)

    try:
        return cv2.FaceDetectorYN.create(
            str(path),
            "",
            (width, height),
            FACE_SCORE_THRESHOLD,
            LOCATOR_NMS_THRESHOLD,
            LOCATOR_TOP_K,
        )
    except Exception as error:
        raise FaceDetectorModelUnavailable(
            f"the face locator at '{path}' could not be loaded "
            f"({type(error).__name__}: {error})"
        ) from error


def _crop_face(frame, box, width: int, height: int):
    """Square, margin-padded RGB crop at the model's input size, or None if degenerate.

    Squared on the longer side before padding so the resize to 380x380 never stretches the
    face: an anisotropic scale is itself a geometric distortion, and this model is being asked
    to judge exactly that kind of artefact.
    """
    cv2 = _import_cv2()

    x, y, box_width, box_height = (float(value) for value in box)
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


def _face_crops(video_path: Path, samples: int, locator_path: Path):
    """Sample frames evenly across the clip and return one face crop from each.

    Even spacing rather than the opening seconds: a manipulated clip is not uniformly
    manipulated, and scoring only the head of the file measures the part of a video an editor
    is most likely to have left alone.

    Frames where no face clears the locator's confidence threshold contribute nothing rather
    than contributing a background crop, and the count of those that did is returned alongside
    so the caller can record what the sample actually covered.
    """
    cv2 = _import_cv2()

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise FaceDetectorMediaError(f"media could not be opened: '{video_path}'")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < 1 or width < 1 or height < 1:
            raise FaceDetectorMediaError(
                f"media declares no decodable frames: '{video_path}'"
            )

        detector = _locator(locator_path, width, height)
        wanted = np.linspace(0, frame_count - 1, samples).astype(int)

        crops: list[tuple[int, object]] = []
        decoded = 0
        for index in sorted({int(value) for value in wanted}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue

            decoded += 1
            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                continue

            # The most confident face in the frame. A clip with two faces is scored on one of
            # them, which is what R3-T1 measured; ranking them is not a choice this module
            # re-opens.
            crop = _crop_face(frame, faces[np.argmax(faces[:, -1])][:4], width, height)
            if crop is not None:
                crops.append((index, crop))

        return crops, decoded
    finally:
        capture.release()


def analyze_face_manipulation(
    video_path: Path,
    *,
    model_dir: Path | str | None = None,
) -> FaceManipulationEvidence:
    """Score the face in evenly sampled frames of one clip and report the mean probability.

    `video_path` is the prepared artifact the rest of the pipeline reads — the derivative when
    one was transcoded, the original when it was already canonical. It is only read.

    The clip's score is the mean of the per-frame probabilities, which is the contract R3-T1
    benchmarked. Mean rather than max: the maximum over a sample is the single most
    incriminating frame, which on a genuine clip is a compression artefact or a bad angle, and
    taking it would raise the false positive rate in exchange for a headline recall number.

    Raises `FaceDetectorNoFaceFound` when no sampled frame yields a face. That is an
    abstention, not a verdict — calling a video genuine because the detector never saw a face
    in it would be a fabricated negative.

    Raises `FaceDetectorModelUnavailable` when an artifact is missing, is not the pinned bytes
    or will not load under the running torch, `FaceDetectorMediaError` when the media cannot be
    read, and `FaceDetectorInferenceError` when torch fails or answers unreadably.

    Blocking and CPU-bound — roughly 14 s on this host for a default eight-frame sample, of
    which about 8.5 s is loading and verifying the artifact.
    """
    directory = Path(model_dir) if model_dir is not None else _model_dir()
    samples = frame_samples()

    # Media first: a clip with no face in it is an ordinary outcome and is worth reaching
    # without paying to load 273 MB of weights for it.
    crops, decoded = _face_crops(video_path, samples, directory / LOCATOR_FILENAME)
    if not crops:
        raise FaceDetectorNoFaceFound(
            f"no face was found in any of {decoded} decoded frame(s) of "
            f"{samples} sampled from '{video_path}'"
        )

    torch = _import_torch()
    classifier, torch_version = _load_classifier(directory / CLASSIFIER_FILENAME)

    batch = np.stack([crop for _, crop in crops]).astype(np.float32) / 255.0
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))

    try:
        with torch.inference_mode():
            logits = classifier(tensor)
        probabilities = torch.sigmoid(logits.reshape(-1))
    except Exception as error:
        raise FaceDetectorInferenceError(
            f"face-manipulation inference failed ({type(error).__name__}: {error})"
        ) from error

    if probabilities.shape != (len(crops),):
        raise FaceDetectorInferenceError(
            f"the classifier returned {tuple(probabilities.shape)} probabilities for "
            f"{len(crops)} face crop(s)"
        )

    values = [float(value) for value in probabilities]
    if not all(np.isfinite(values)):
        raise FaceDetectorInferenceError(
            f"the classifier returned non-finite probabilities: {values}"
        )

    frame_scores = tuple(
        FrameScore(frame_index=index, probability=probability)
        for (index, _), probability in zip(crops, values)
    )

    logger.info(
        "EfficientNet-B7 scored %d face crop(s) from %d decoded frame(s) of %s",
        len(frame_scores),
        decoded,
        video_path,
    )

    return FaceManipulationEvidence(
        classifier_repository=CLASSIFIER_REPOSITORY,
        classifier_revision=CLASSIFIER_REVISION,
        classifier_sha256=CLASSIFIER_SHA256,
        locator_repository=LOCATOR_REPOSITORY,
        locator_revision=LOCATOR_REVISION,
        locator_sha256=LOCATOR_SHA256,
        torch_version=torch_version,
        input_size=INPUT_SIZE,
        crop_margin=CROP_MARGIN,
        face_score_threshold=FACE_SCORE_THRESHOLD,
        frames_requested=samples,
        frames_decoded=decoded,
        frames_scored=len(frame_scores),
        score=float(np.mean(values)),
        frame_scores=frame_scores[:MAX_PERSISTED_FRAME_SCORES],
    )
