"""Tests for the local EfficientNet-B7 face-manipulation evidence extractor.

None of these load the real 273 MiB classifier or run torch. Both model boundaries are faked
by putting stand-in `torch` and `cv2` modules in `sys.modules` and letting the real imports
inside the module find them, the way `test_audio_detector.py` fakes `onnxruntime`. The digest
boundary is real: the artifacts are actual files on disk with actual bytes, because whether
the pinned digest is *checked* is exactly what this module promises.

The classifier contract these fakes reproduce, as R3-T1 established it against the pinned
artifact (`tomas-gajarsky/facetorch-deepfake-efficientnet-b7` @ 4acc494f, sha256 97b49a70…):

    input   ['batch', 3, 380, 380]  float32, ImageNet-normalised RGB face crops
    output  ['batch', 1]            float32, one binary logit per face

One logit per face, through a sigmoid, meaned over the sampled frames. No class, no
threshold and no calibration — R3-T1's benchmark operating point of 0.8 is a property of
that benchmark and is deliberately absent from this module and from these tests.

Real inference over the real weights is verified out of band (R3-T2 live run), not here.
"""

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app import face_detector
from app.face_detector import (
    FaceDetectorInferenceError,
    FaceDetectorMediaError,
    FaceDetectorModelUnavailable,
    FaceDetectorNoFaceFound,
    analyze_face_manipulation,
)

CLASSIFIER = face_detector.CLASSIFIER_FILENAME
LOCATOR = face_detector.LOCATOR_FILENAME

# A frame big enough for a face box with margin to fit inside it.
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
# One face, comfortably inside the frame, with YuNet's trailing confidence column.
FACE_BOX = [200.0, 150.0, 120.0, 120.0]


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


@pytest.fixture
def model_dir(tmp_path, monkeypatch) -> Path:
    """A directory holding both artifacts, with bytes that hash to the pinned digests.

    The digests are constants of the module, so the *files* are made to match them rather
    than the other way round: a fixture that patched the expected digest would be checking
    that string comparison works, not that the pinned weights are the ones loaded.
    """
    directory = tmp_path / "models"
    directory.mkdir()

    for filename, digest_constant in (
        (CLASSIFIER, "CLASSIFIER_SHA256"),
        (LOCATOR, "LOCATOR_SHA256"),
    ):
        payload = filename.encode()
        (directory / filename).write_bytes(payload)
        monkeypatch.setattr(
            face_detector, digest_constant, hashlib.sha256(payload).hexdigest()
        )

    return directory


class _FakeCapture:
    """A stand-in for `cv2.VideoCapture`: reports a geometry, hands out frames on demand."""

    def __init__(self, *, frames=100, width=FRAME_WIDTH, height=FRAME_HEIGHT, opened=True,
                 unreadable=frozenset()):
        self._properties = {1: frames, 3: width, 4: height}
        self._opened = opened
        self._unreadable = unreadable
        self._position = 0
        self.released = False
        self.requested = []

    def isOpened(self):  # noqa: N802 — mirrors the real name
        return self._opened

    def get(self, key):
        # 7 is CAP_PROP_FRAME_COUNT, 3/4 are width/height; the fake answers only those.
        return float(self._properties.get({7: 1}.get(key, key), 0))

    def set(self, key, value):
        self._position = int(value)
        return True

    def read(self):
        self.requested.append(self._position)
        if self._position in self._unreadable:
            return False, None

        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), self._position % 256, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class _FakeLocator:
    """A stand-in for YuNet: answers with a fixed set of boxes per frame, in order."""

    def __init__(self, boxes_sequence=None):
        self._boxes_sequence = boxes_sequence
        self.calls = 0

    def detect(self, frame):
        index = self.calls
        self.calls += 1

        if self._boxes_sequence is None:
            boxes = [FACE_BOX + [0.99]]
        else:
            boxes = self._boxes_sequence[index % len(self._boxes_sequence)]

        if boxes is None:
            return 1, None

        return 1, np.asarray(boxes, dtype=np.float32)


def _install_cv2(monkeypatch, *, capture=None, locator=None, locator_error=None):
    """Put a fake `cv2` on `sys.modules` and report the capture and locator it hands out."""
    capture = capture if capture is not None else _FakeCapture()
    locator = locator if locator is not None else _FakeLocator()

    module = types.ModuleType("cv2")
    module.CAP_PROP_FRAME_COUNT = 7
    module.CAP_PROP_FRAME_WIDTH = 3
    module.CAP_PROP_FRAME_HEIGHT = 4
    module.CAP_PROP_POS_FRAMES = 1
    module.INTER_LINEAR = 1
    module.COLOR_BGR2RGB = 4
    module.VideoCapture = lambda path: capture

    def create(*args, **kwargs):
        if locator_error is not None:
            raise locator_error
        return locator

    module.FaceDetectorYN = types.SimpleNamespace(create=create)
    # Resize and colour conversion are real array operations, faked only in shape: what the
    # module is being tested on is which crop it took, not OpenCV's interpolation.
    module.resize = lambda image, size, interpolation=None: np.resize(
        image, (size[1], size[0], 3)
    )
    module.cvtColor = lambda image, code: image[:, :, ::-1]

    monkeypatch.setitem(sys.modules, "cv2", module)
    return capture, locator


class _FakeExported:
    """The loaded `torch.export` program: records its batch, answers with fixed logits."""

    def __init__(self, logits=None, error=None, output=None):
        self._logits = logits
        self._error = error
        self._output = output
        self.batches = []

    def __call__(self, tensor):
        self.batches.append(np.asarray(tensor))

        if self._error is not None:
            raise self._error
        if self._output is not None:
            return _FakeTensor(self._output)

        count = len(self.batches[-1])
        values = self._logits if self._logits is not None else [2.0] * count

        return _FakeTensor(np.asarray(values, dtype=np.float32).reshape(-1, 1))


class _FakeTensor:
    """Just enough of a tensor for `reshape`, `sigmoid`, iteration and `.shape`."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    @property
    def shape(self):
        return self._values.shape

    def reshape(self, *shape):
        return _FakeTensor(self._values.reshape(*shape))

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __array__(self, dtype=None):
        return self._values if dtype is None else self._values.astype(dtype)


def _install_torch(monkeypatch, *, exported=None, version="2.13.0+cpu", load_error=None):
    """Put a fake `torch` on `sys.modules` and report the exported program it hands out."""
    exported = exported if exported is not None else _FakeExported()

    module = types.ModuleType("torch")
    module.__version__ = version

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module.inference_mode = _InferenceMode
    module.from_numpy = lambda array: _FakeTensor(array)
    module.sigmoid = lambda tensor: _FakeTensor(
        1.0 / (1.0 + np.exp(-np.asarray(tensor)))
    )

    def load(path):
        if load_error is not None:
            raise load_error
        return types.SimpleNamespace(module=lambda: exported)

    module.export = types.SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, "torch", module)
    return exported


def _analyze(directory, video="clip.mp4"):
    return analyze_face_manipulation(Path(video), model_dir=directory)


# --- The score and what stands behind it ----------------------------------------------------


def test_the_clip_score_is_the_mean_of_the_per_frame_probabilities(model_dir, monkeypatch):
    """R3-T1's contract: one analysed clip, one raw score, and it is the mean."""
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch, exported=_FakeExported(logits=[-2.0, 0.0, 1.0, 3.0] * 2))

    evidence = _analyze(model_dir)

    expected = float(
        np.mean([_sigmoid(value) for value in [-2.0, 0.0, 1.0, 3.0] * 2])
    )
    assert evidence.score == pytest.approx(expected)
    assert [frame.probability for frame in evidence.frame_scores] == pytest.approx(
        [_sigmoid(value) for value in [-2.0, 0.0, 1.0, 3.0] * 2]
    )


def test_the_frames_are_sampled_evenly_across_the_whole_clip(model_dir, monkeypatch):
    """Not the opening seconds: an editor is most likely to have left the head alone."""
    capture, _ = _install_cv2(monkeypatch, capture=_FakeCapture(frames=100))
    _install_torch(monkeypatch)

    _analyze(model_dir)

    assert capture.requested == [0, 14, 28, 42, 56, 70, 84, 99]


def test_every_score_names_the_frame_it_came_from(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=100))
    _install_torch(monkeypatch)

    evidence = _analyze(model_dir)

    assert [frame.frame_index for frame in evidence.frame_scores] == [
        0, 14, 28, 42, 56, 70, 84, 99
    ]


def test_the_default_sample_is_the_eight_frames_r3_t1_measured(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=100))
    _install_torch(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.frames_requested == 8
    assert evidence.frames_decoded == 8
    assert evidence.frames_scored == 8


def test_the_frame_count_is_configurable_and_recorded_as_used(model_dir, monkeypatch):
    monkeypatch.setenv(face_detector.FRAME_SAMPLES_ENV, "4")
    capture, _ = _install_cv2(monkeypatch, capture=_FakeCapture(frames=100))
    _install_torch(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.frames_requested == 4
    assert len(capture.requested) == 4


@pytest.mark.parametrize("configured", ["", "   "])
def test_an_unset_frame_count_is_the_benchmarked_default(configured, monkeypatch):
    monkeypatch.setenv(face_detector.FRAME_SAMPLES_ENV, configured)

    assert face_detector.frame_samples() == face_detector.DEFAULT_FRAME_SAMPLES


@pytest.mark.parametrize("configured", ["eight", "0", "-2", "3.5"])
def test_a_malformed_frame_count_is_refused_rather_than_defaulted(configured, monkeypatch):
    monkeypatch.setenv(face_detector.FRAME_SAMPLES_ENV, configured)

    with pytest.raises(FaceDetectorModelUnavailable):
        face_detector.frame_samples()


# --- Preprocessing, which is part of the measurement ----------------------------------------


def test_the_classifier_is_fed_the_input_size_the_checkpoint_was_trained_on(
    model_dir, monkeypatch
):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    exported = _install_torch(monkeypatch)

    _analyze(model_dir)

    batch = exported.batches[0]
    assert batch.shape == (8, 3, face_detector.INPUT_SIZE, face_detector.INPUT_SIZE)
    assert batch.dtype == np.float32


def test_a_frame_with_no_face_contributes_nothing_rather_than_a_background_crop(
    model_dir, monkeypatch
):
    """The classifier scores faces. A frame with none must not be scored at all."""
    _install_cv2(
        monkeypatch,
        capture=_FakeCapture(frames=8),
        locator=_FakeLocator([[FACE_BOX + [0.99]], None] * 4),
    )
    exported = _install_torch(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.frames_decoded == 8
    assert evidence.frames_scored == 4
    assert len(exported.batches[0]) == 4


def test_a_frame_that_will_not_decode_is_skipped_and_counted_apart(model_dir, monkeypatch):
    _install_cv2(
        monkeypatch,
        capture=_FakeCapture(frames=100, unreadable={28, 70}),
    )
    _install_torch(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.frames_requested == 8
    assert evidence.frames_decoded == 6
    assert evidence.frames_scored == 6


def test_the_most_confident_face_in_a_frame_is_the_one_scored(model_dir, monkeypatch):
    """Two faces is not a reason to score both; R3-T1 scored one, ranked by confidence."""
    _install_cv2(
        monkeypatch,
        capture=_FakeCapture(frames=1),
        locator=_FakeLocator([[[10.0, 10.0, 40.0, 40.0, 0.61], FACE_BOX + [0.98]]]),
    )
    exported = _install_torch(monkeypatch)

    _analyze(model_dir)

    assert len(exported.batches[0]) == 1


def test_the_capture_is_released_even_when_the_media_is_refused(model_dir, monkeypatch):
    capture, _ = _install_cv2(monkeypatch, capture=_FakeCapture(frames=0))
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorMediaError):
        _analyze(model_dir)

    assert capture.released is True


# --- Abstention, which is not a verdict -----------------------------------------------------


def test_a_clip_with_no_face_anywhere_abstains_rather_than_scoring_it(model_dir, monkeypatch):
    """The one result that must never become a number: nothing was ever classified."""
    _install_cv2(
        monkeypatch,
        capture=_FakeCapture(frames=8),
        locator=_FakeLocator([None]),
    )
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorNoFaceFound):
        _analyze(model_dir)


def test_a_clip_with_no_face_never_loads_the_classifier_at_all(model_dir, monkeypatch):
    """Media first: 273 MiB of weights are not loaded to answer a question nobody asked."""
    _install_cv2(
        monkeypatch,
        capture=_FakeCapture(frames=8),
        locator=_FakeLocator([None]),
    )
    exported = _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorNoFaceFound):
        _analyze(model_dir)

    assert exported.batches == []


# --- Media that cannot be read --------------------------------------------------------------


def test_media_that_will_not_open_is_refused(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(opened=False))
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorMediaError):
        _analyze(model_dir)


@pytest.mark.parametrize(
    "geometry",
    [{"frames": 0}, {"width": 0}, {"height": 0}],
)
def test_media_declaring_nothing_decodable_is_refused(model_dir, monkeypatch, geometry):
    _install_cv2(monkeypatch, capture=_FakeCapture(**geometry))
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorMediaError):
        _analyze(model_dir)


# --- Provenance, which is the point ---------------------------------------------------------


def test_the_evidence_names_both_artifacts_by_revision_and_digest(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.classifier_repository == (
        "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
    )
    assert evidence.classifier_revision == "4acc494f37eb63d7457166eff2acb45c5b04b9a6"
    assert evidence.locator_repository == "opencv/opencv_zoo"
    assert evidence.locator_revision == "47534e27c9851bb1128ccc0102f1145e27f23f98"
    assert evidence.torch_version == "2.13.0+cpu"


def test_the_r3_t1_provenance_constants_are_the_ones_that_ship():
    """Restated rather than imported. If a pin is ever edited, this must fail rather than
    agree with the edit: the revision and digest are facts about the measurement R3-T1 took."""
    assert face_detector.CLASSIFIER_REVISION == "4acc494f37eb63d7457166eff2acb45c5b04b9a6"
    assert face_detector.CLASSIFIER_SHA256 == (
        "97b49a70174c0d4f72d9d510d817bdc49a907af9af0242a6a1ba934a7cc9e4b7"
    )
    assert face_detector.LOCATOR_SHA256 == (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    )
    assert face_detector.INPUT_SIZE == 380
    assert face_detector.FACE_SCORE_THRESHOLD == 0.6
    assert face_detector.DEFAULT_FRAME_SAMPLES == 8


@pytest.mark.parametrize("swapped", [CLASSIFIER, LOCATOR])
def test_weights_that_are_not_the_pinned_bytes_are_refused(model_dir, monkeypatch, swapped):
    """A digest in a docstring is a claim; a digest checked at load is the claim being kept."""
    (model_dir / swapped).write_bytes(b"something else entirely")
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorModelUnavailable, match="pinned digest"):
        _analyze(model_dir)


@pytest.mark.parametrize("missing", [CLASSIFIER, LOCATOR])
def test_weights_that_are_missing_are_refused(model_dir, monkeypatch, missing):
    (model_dir / missing).unlink()
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorModelUnavailable, match="missing"):
        _analyze(model_dir)


@pytest.mark.parametrize("version", ["2.11.0+cpu", "2.13.0+cpu"])
def test_the_verified_torch_versions_run(model_dir, monkeypatch, version):
    """2.11 is the artifact's published cohort; 2.13 is what this image ships, and the two
    were compared to bit-identical logits before 2.13 was added to the allowlist."""
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch, version=version)

    assert _analyze(model_dir).torch_version == version


@pytest.mark.parametrize("version", ["2.10.1+cpu", "2.12.0+cpu", "2.14.0+cpu", "3.0.0"])
def test_an_unverified_torch_is_refused_rather_than_run(model_dir, monkeypatch, version):
    """Scoring media under a runtime no measurement covers would put a number behind a
    stored signal that nothing describes."""
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch, version=version)

    with pytest.raises(FaceDetectorModelUnavailable, match="verified"):
        _analyze(model_dir)


# --- Failures that must stay this signal's own ----------------------------------------------


def test_a_classifier_that_will_not_load_is_a_model_failure(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch, load_error=RuntimeError("corrupt archive"))

    with pytest.raises(FaceDetectorModelUnavailable):
        _analyze(model_dir)


def test_a_locator_that_will_not_load_is_a_model_failure(model_dir, monkeypatch):
    _install_cv2(monkeypatch, locator_error=RuntimeError("bad graph"))
    _install_torch(monkeypatch)

    with pytest.raises(FaceDetectorModelUnavailable):
        _analyze(model_dir)


def test_inference_that_raises_is_an_inference_failure(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch, exported=_FakeExported(error=RuntimeError("no memory")))

    with pytest.raises(FaceDetectorInferenceError):
        _analyze(model_dir)


def test_an_output_of_the_wrong_width_is_refused(model_dir, monkeypatch):
    """A malformed output read positionally would become confident-looking evidence."""
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(
        monkeypatch,
        exported=_FakeExported(output=np.zeros((3, 1), dtype=np.float32)),
    )

    with pytest.raises(FaceDetectorInferenceError):
        _analyze(model_dir)


def test_non_finite_output_is_refused(model_dir, monkeypatch):
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(
        monkeypatch,
        exported=_FakeExported(output=np.full((8, 1), np.nan, dtype=np.float32)),
    )

    with pytest.raises(FaceDetectorInferenceError):
        _analyze(model_dir)


@pytest.mark.parametrize("absent", ["cv2", "torch"])
def test_a_missing_dependency_is_a_model_failure_not_an_escaping_import_error(
    model_dir, monkeypatch, absent
):
    """The failure that would otherwise take the whole analysis down.

    `ModuleNotFoundError` is not a `FaceDetectorError`, so an unguarded import here would
    escape `app.detection`, propagate out of `analyse` and fail the job — destroying the
    provenance, synthetic-video and audio evidence over a dependency this signal alone needs
    (AGENTS.md, error-handling rule).
    """
    _install_cv2(monkeypatch, capture=_FakeCapture(frames=8))
    _install_torch(monkeypatch)
    monkeypatch.setitem(sys.modules, absent, None)

    with pytest.raises(FaceDetectorModelUnavailable):
        _analyze(model_dir)
