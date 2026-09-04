"""Tests for the local LipForensics mouth-dynamics evidence extractor.

None of these load the real 137 MiB checkpoint or run torch. Four model boundaries are faked
by putting stand-in `torch`, `cv2`, `skimage` and `face_alignment` modules in `sys.modules`
and letting the real imports inside the module find them, the way `test_face_detector.py`
fakes `torch` and `cv2`.

Two boundaries are deliberately real, because they are what this module promises:

- **the digests.** Every artifact is an actual file on disk with actual bytes, and the
  constants are left alone — a fixture that patched the expected digest would be checking that
  string comparison works, not that the pinned artifacts are the ones loaded.
- **the architecture is executed from disk.** The stand-in `spatiotemporal_net.py` is a real
  Python file written into the fixture's checkout and run through the module's own loader, so
  the tests exercise the same path the pinned upstream source goes through — including what
  happens when one of those files is not the bytes that were pinned.

The classifier contract these fakes reproduce, as R5-T1 established it against the pinned
artifact (`ahaliassos/LipForensics` @ d0bf5553, weights sha256 4b7790bc…):

    input   [runs, 1, 25, 88, 88]  float32 grayscale mouth crops, `lengths=[25] * runs`
    output  [runs]                 float32, one raw logit per 25-frame run

One logit per run, meaned, then squashed once — never the other way round. No class, no
threshold and no calibration: R5-T1's confusion matrix was reported at the harness default of
0.5, which is a property of that run and is deliberately absent from this module and from
these tests.

Real inference over the real weights is verified out of band (R5-T2 live run), not here.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app import lip_forensics
from app.lip_forensics import (
    FRAMES_PER_WINDOW,
    SMOOTHING_WINDOW,
    LipForensicsInferenceError,
    LipForensicsMediaError,
    LipForensicsModelUnavailable,
    LipForensicsNoTrackedFace,
    analyze_lip_forensics,
)

# A frame comfortably larger than the 256x256 the alignment warps onto.
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# How many frames one sampled run consumes off the decoder: the scored 25 plus the smoothing
# look-ahead that steadies their alignment and is then discarded.
FRAMES_PER_READ = FRAMES_PER_WINDOW + SMOOTHING_WINDOW - 1

# The upstream `Lipreading` this fixture writes into the checkout. It is executed by the real
# loader, so it has to be a real module — and it records what it was given, which is how the
# input contract above is asserted.
SPATIOTEMPORAL_NET = '''
class Lipreading:
    calls = []

    def __init__(self, num_classes, relu_type, tcn_options):
        self.num_classes = num_classes
        self.relu_type = relu_type
        self.tcn_options = tcn_options
        self.state = None
        self.device = None
        self.evaluated = False

    def load_state_dict(self, state):
        self.state = state

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, tensor, lengths):
        Lipreading.calls.append((tensor, lengths))
        return _ANSWER(len(tensor))
'''


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def model_dir(tmp_path, monkeypatch) -> Path:
    """A directory holding every pinned artifact, with bytes that hash to the pinned digests.

    Laid out exactly as the image lays it out, because the layout is part of what is being
    tested: the checkout under its revision, the weights beside it, and the landmark pair in a
    torch hub directory.
    """
    directory = tmp_path / "models"
    source = directory / f"lipforensics-{lip_forensics.UPSTREAM_REVISION}"

    config = json.dumps(
        {
            "relu_type": "prelu",
            "tcn_num_layers": 4,
            "tcn_kernel_size": [3, 5, 7],
            "tcn_dropout": 0.2,
            "tcn_dwpw": False,
            "tcn_width_mult": 1,
        }
    ).encode()

    # A 68-point mean face, which is what the five stable points and the mouth slice index
    # into. The values do not matter to the fake warp; the shape does.
    mean_face_path = source / "preprocessing" / "20words_mean_face.npy"
    mean_face_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mean_face_path, np.arange(136, dtype=np.float64).reshape(68, 2))

    digests = {
        "models/resnet.py": _write(source / "models" / "resnet.py", b"# resnet\n"),
        "models/tcn.py": _write(source / "models" / "tcn.py", b"# tcn\n"),
        "models/spatiotemporal_net.py": _write(
            source / "models" / "spatiotemporal_net.py", SPATIOTEMPORAL_NET.encode()
        ),
        "models/configs/lrw_resnet18_mstcn.json": _write(
            source / "models" / "configs" / "lrw_resnet18_mstcn.json", config
        ),
        "preprocessing/20words_mean_face.npy": hashlib.sha256(
            mean_face_path.read_bytes()
        ).hexdigest(),
    }
    monkeypatch.setattr(lip_forensics, "SOURCE_SHA256", digests)

    for filename, constant in (
        (Path("weights") / lip_forensics.WEIGHTS_FILENAME, "WEIGHTS_SHA256"),
        (Path("face-alignment/checkpoints") / lip_forensics.SFD_FILENAME, "SFD_SHA256"),
        (Path("face-alignment/checkpoints") / lip_forensics.FAN_FILENAME, "FAN_SHA256"),
    ):
        digest = _write(directory / filename, filename.name.encode())
        monkeypatch.setattr(lip_forensics, constant, digest)

    return directory


class _FakeCapture:
    """A stand-in for `cv2.VideoCapture`: reports a frame count, hands out frames on demand."""

    def __init__(self, *, frames=200, opened=True, short_from=None):
        self._frames = frames
        self._opened = opened
        # The frame index from which the decoder starts refusing, so a run can be made to come
        # back short without the clip itself being short.
        self._short_from = short_from
        self._position = 0
        self.released = False
        self.starts = []

    def isOpened(self):  # noqa: N802 — mirrors the real name
        return self._opened

    def get(self, key):
        return float(self._frames if key == 7 else 0)

    def set(self, key, value):
        self._position = int(value)
        self.starts.append(self._position)
        return True

    def read(self):
        if self._short_from is not None and self._position >= self._short_from:
            return False, None

        frame = np.full(
            (FRAME_HEIGHT, FRAME_WIDTH, 3), self._position % 256, dtype=np.uint8
        )
        self._position += 1

        return True, frame

    def release(self):
        self.released = True


def _install_cv2(monkeypatch, *, capture=None):
    """Put a fake `cv2` on `sys.modules` and report the capture it hands out."""
    capture = capture if capture is not None else _FakeCapture()

    module = types.ModuleType("cv2")
    module.CAP_PROP_FRAME_COUNT = 7
    module.CAP_PROP_POS_FRAMES = 1
    module.COLOR_BGR2RGB = 4
    module.COLOR_BGR2GRAY = 6
    module.VideoCapture = lambda path: capture
    module.cvtColor = lambda image, code: (
        image[:, :, ::-1] if code == 4 else image[:, :, 0]
    )

    monkeypatch.setitem(sys.modules, "cv2", module)

    return capture


class _FakeLandmarker:
    """A stand-in for `face_alignment`: answers with 68 landmarks, or with none.

    `blind_frames` are the absolute frame positions at which it reports no face, which is how
    a run that loses its face partway through is expressed — the module's contract is that such
    a run is dropped whole rather than stitched across the gap.
    """

    def __init__(self, blind_frames=frozenset()):
        self._blind = set(blind_frames)
        self.frames_seen = 0

    def get_landmarks_from_image(self, image):
        seen = self.frames_seen
        self.frames_seen += 1

        if seen in self._blind:
            return None

        # Two faces, so the module's "follow the largest" rule is exercised on every frame.
        small = np.tile(np.array([100.0, 100.0]), (68, 1)) + np.arange(68).reshape(68, 1)
        large = np.tile(np.array([100.0, 100.0]), (68, 1)) + (
            np.arange(68).reshape(68, 1) * 3
        )

        return [small, large]


def _install_face_alignment(monkeypatch, *, landmarker=None, missing=False, error=None):
    if missing:
        monkeypatch.setitem(sys.modules, "face_alignment", None)
        monkeypatch.setattr(
            lip_forensics,
            "_import_face_alignment",
            lambda: (_ for _ in ()).throw(
                LipForensicsModelUnavailable("face-alignment is not installed")
            ),
        )
        return None

    landmarker = landmarker if landmarker is not None else _FakeLandmarker()

    module = types.ModuleType("face_alignment")
    module.__version__ = "1.5.0"
    module.LandmarksType = types.SimpleNamespace(TWO_D=2)

    def build(landmarks_type, device=None, flip_input=None, compile=None):
        if error is not None:
            raise error
        return landmarker

    module.FaceAlignment = build
    monkeypatch.setitem(sys.modules, "face_alignment", module)

    return landmarker


def _install_skimage(monkeypatch):
    """A fake similarity warp: geometry-free, but shaped exactly as the real one is."""

    class _Transform:
        inverse = object()

        def __call__(self, landmarks):
            # Landmarks inside the warped 256x256 frame, so the mouth patch always fits.
            return np.tile(np.array([128.0, 128.0]), (68, 1))

    transform_module = types.ModuleType("skimage.transform")
    transform_module.estimate_transform = lambda kind, src, dst: _Transform()
    transform_module.warp = lambda frame, inverse_map=None, output_shape=None: np.full(
        (*output_shape, 3), 0.5, dtype=np.float64
    )

    skimage_module = types.ModuleType("skimage")
    skimage_module.transform = transform_module

    monkeypatch.setitem(sys.modules, "skimage", skimage_module)
    monkeypatch.setitem(sys.modules, "skimage.transform", transform_module)


class _FakeTensor:
    """Just enough of a tensor for the arithmetic and reshaping this module does."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    @property
    def shape(self):
        return self._values.shape

    def float(self):
        return self

    def to(self, device):
        return self

    def unsqueeze(self, axis):
        return _FakeTensor(np.expand_dims(self._values, axis))

    def reshape(self, *shape):
        return _FakeTensor(self._values.reshape(*shape))

    def mean(self):
        return _FakeTensor(self._values.mean())

    def __truediv__(self, other):
        return _FakeTensor(self._values / other)

    def __sub__(self, other):
        return _FakeTensor(self._values - other)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __float__(self):
        return float(self._values)

    def __array__(self, dtype=None):
        return self._values if dtype is None else self._values.astype(dtype)


def _install_torch(
    monkeypatch, *, logits=None, answer=None, load_error=None, version="2.13.0+cpu"
):
    """Put a fake `torch` on `sys.modules`, and arrange what the classifier answers.

    The classifier itself is the upstream stand-in executed from the fixture's checkout, so the
    answer is injected into that module's namespace rather than into torch — which is exactly
    where the real answer comes from too.
    """
    module = types.ModuleType("torch")
    module.__version__ = version
    module.hub = types.SimpleNamespace(set_dir=lambda path: None)

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

    def load(path, map_location=None, weights_only=None):
        if load_error is not None:
            raise load_error
        return {"model": {"loaded": str(path)}}

    module.load = load
    monkeypatch.setitem(sys.modules, "torch", module)

    def answered(count):
        if answer is not None:
            return answer
        values = logits if logits is not None else [1.0] * count
        return _FakeTensor(np.asarray(values, dtype=np.float32))

    # The upstream module is executed fresh on every call, so the answer is installed by
    # patching the loader rather than the module object it returns.
    original = lip_forensics._upstream_architecture

    def architecture(directory):
        module = original(directory)
        module._ANSWER = answered
        module.Lipreading.calls = []
        return module

    monkeypatch.setattr(lip_forensics, "_upstream_architecture", architecture)


def _install_all(monkeypatch, *, capture=None, landmarker=None, **torch_kwargs):
    capture = _install_cv2(monkeypatch, capture=capture)
    landmarker = _install_face_alignment(monkeypatch, landmarker=landmarker)
    _install_skimage(monkeypatch)
    _install_torch(monkeypatch, **torch_kwargs)

    return capture, landmarker


def _analyze(directory, video="clip.mp4"):
    return analyze_lip_forensics(Path(video), model_dir=directory)


# --- The score and what stands behind it ----------------------------------------------------


def test_the_clip_score_is_the_sigmoid_of_the_mean_logit(model_dir, monkeypatch):
    """R5-T1's contract: mean the logits, squash once, and never the other way round."""
    _install_all(monkeypatch, logits=[-3.0, 0.0, 1.0, 4.0])

    evidence = _analyze(model_dir)

    assert evidence.score == pytest.approx(_sigmoid(np.mean([-3.0, 0.0, 1.0, 4.0])))


def test_the_score_is_not_the_mean_of_the_squashed_logits(model_dir, monkeypatch):
    """The aggregation order is the contract, and the two orders genuinely disagree.

    A mean of probabilities is dominated by whichever run saturated first, which is why
    upstream aggregates in logit space and why this module does too.
    """
    _install_all(monkeypatch, logits=[-6.0, -6.0, -6.0, 6.0])

    evidence = _analyze(model_dir)

    # Squashing first would put this clip at 0.25 — a quarter of the way up the scale on the
    # strength of one saturated run. Meaning first puts it at sigmoid(-3.0), which is 0.047.
    assert evidence.score == pytest.approx(_sigmoid(-3.0))
    assert evidence.score != pytest.approx(
        float(np.mean([_sigmoid(-6.0), _sigmoid(-6.0), _sigmoid(-6.0), _sigmoid(6.0)]))
    )


def test_the_logits_are_recorded_raw_beside_the_run_they_came_from(model_dir, monkeypatch):
    _install_all(monkeypatch, capture=_FakeCapture(frames=200), logits=[-3.0, 0.0, 1.0, 4.0])

    evidence = _analyze(model_dir)

    assert [window.logit for window in evidence.window_logits] == pytest.approx(
        [-3.0, 0.0, 1.0, 4.0]
    )
    assert [window.start_frame for window in evidence.window_logits] == [0, 58, 116, 175]


def test_the_runs_are_sampled_evenly_across_the_whole_clip(model_dir, monkeypatch):
    """Not the opening seconds: an editor is most likely to have left the head alone."""
    capture, _ = _install_all(monkeypatch, capture=_FakeCapture(frames=200))

    _analyze(model_dir)

    assert capture.starts == [0, 58, 116, 175]


def test_the_default_sample_is_the_four_runs_r5_t1_measured(model_dir, monkeypatch):
    _install_all(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.windows_requested == 4
    assert evidence.windows_scored == 4


def test_the_window_count_is_configurable_and_recorded(model_dir, monkeypatch):
    monkeypatch.setenv(lip_forensics.WINDOWS_ENV, "2")
    _install_all(monkeypatch, logits=[0.0, 2.0])

    evidence = _analyze(model_dir)

    assert evidence.windows_requested == 2
    assert evidence.windows_scored == 2


@pytest.mark.parametrize("configured", ["nonsense", "2.5", "0", "-1"])
def test_a_malformed_window_count_refuses_rather_than_falling_back(
    model_dir, monkeypatch, configured
):
    """A deployment that wrote a value meant something by it."""
    monkeypatch.setenv(lip_forensics.WINDOWS_ENV, configured)
    _install_all(monkeypatch)

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


def test_an_unset_window_count_is_the_benchmarked_default(model_dir, monkeypatch):
    monkeypatch.setenv(lip_forensics.WINDOWS_ENV, "   ")
    _install_all(monkeypatch)

    assert _analyze(model_dir).windows_requested == 4


# --- What the model is actually given -------------------------------------------------------


def test_the_model_is_given_one_batch_of_centre_cropped_runs(model_dir, monkeypatch):
    """The input contract R5-T1 measured: [runs, 1, 25, 88, 88], with a length per run."""
    _install_all(monkeypatch)
    architecture = []
    original = lip_forensics._upstream_architecture
    monkeypatch.setattr(
        lip_forensics,
        "_upstream_architecture",
        lambda directory: architecture.append(original(directory)) or architecture[-1],
    )

    _analyze(model_dir)

    tensor, lengths = architecture[-1].Lipreading.calls[-1]
    assert np.asarray(tensor).shape == (
        4,
        1,
        FRAMES_PER_WINDOW,
        lip_forensics.INPUT_SIZE,
        lip_forensics.INPUT_SIZE,
    )
    assert lengths == [FRAMES_PER_WINDOW] * 4


def test_each_run_reads_the_smoothing_look_ahead_beyond_the_frames_it_scores(
    model_dir, monkeypatch
):
    """The extra frames steady the alignment of the scored ones and are then discarded."""
    _, landmarker = _install_all(monkeypatch)

    _analyze(model_dir)

    assert landmarker.frames_seen == FRAMES_PER_READ * 4


# --- Runs that cannot be used ---------------------------------------------------------------


def test_a_run_that_loses_the_face_is_dropped_whole(model_dir, monkeypatch):
    """A run stitched across a guessed frame is a movement that never happened."""
    # Blind on one frame of the second run only.
    blind = {FRAMES_PER_READ + 3}
    _install_all(
        monkeypatch,
        landmarker=_FakeLandmarker(blind_frames=blind),
        logits=[1.0, 2.0, 3.0],
    )

    evidence = _analyze(model_dir)

    assert evidence.windows_read == 4
    assert evidence.windows_scored == 3
    assert [window.start_frame for window in evidence.window_logits] == [0, 116, 175]


def test_no_run_with_a_tracked_face_abstains_rather_than_scoring(model_dir, monkeypatch):
    """Calling a video genuine because no mouth was found would be a fabricated negative."""
    _install_all(
        monkeypatch, landmarker=_FakeLandmarker(blind_frames=set(range(10_000)))
    )

    with pytest.raises(LipForensicsNoTrackedFace):
        _analyze(model_dir)


def test_the_weights_are_not_even_loaded_when_no_run_holds_a_face(model_dir, monkeypatch):
    """The abstention is reached without paying to verify and load 137 MiB for it."""
    _install_all(
        monkeypatch,
        landmarker=_FakeLandmarker(blind_frames=set(range(10_000))),
        load_error=AssertionError("the checkpoint must not be loaded"),
    )

    with pytest.raises(LipForensicsNoTrackedFace):
        _analyze(model_dir)


def test_a_run_the_decoder_cannot_supply_is_not_counted_as_read(model_dir, monkeypatch):
    _install_all(
        monkeypatch,
        capture=_FakeCapture(frames=200, short_from=170),
        logits=[1.0, 2.0, 3.0],
    )

    evidence = _analyze(model_dir)

    assert evidence.windows_requested == 4
    assert evidence.windows_read == 3


# --- Media that cannot be read --------------------------------------------------------------


def test_media_that_cannot_be_opened_is_a_media_error(model_dir, monkeypatch):
    _install_all(monkeypatch, capture=_FakeCapture(opened=False))

    with pytest.raises(LipForensicsMediaError):
        _analyze(model_dir)


def test_a_clip_shorter_than_one_run_is_a_media_error_not_an_abstention(
    model_dir, monkeypatch
):
    """Nothing was sampled at all, so there is not even a run a face could have been in."""
    _install_all(monkeypatch, capture=_FakeCapture(frames=FRAMES_PER_WINDOW - 1))

    with pytest.raises(LipForensicsMediaError):
        _analyze(model_dir)


def test_the_capture_is_released_even_when_the_media_is_refused(model_dir, monkeypatch):
    capture, _ = _install_all(monkeypatch, capture=_FakeCapture(frames=3))

    with pytest.raises(LipForensicsMediaError):
        _analyze(model_dir)

    assert capture.released is True


# --- Provenance: the artifacts are the ones that were pinned --------------------------------


def test_swapped_weights_are_refused(model_dir, monkeypatch):
    _install_all(monkeypatch)
    (model_dir / "weights" / lip_forensics.WEIGHTS_FILENAME).write_bytes(b"other")

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


def test_an_edited_upstream_source_file_is_refused(model_dir, monkeypatch):
    """The architecture is executed from disk, so an edited layer is a different model."""
    _install_all(monkeypatch)
    source = model_dir / f"lipforensics-{lip_forensics.UPSTREAM_REVISION}"
    (source / "models" / "tcn.py").write_bytes(b"# edited\n")

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


def test_an_edited_mean_face_is_refused(model_dir, monkeypatch):
    """Every alignment is built on it, so a different mean face is a different measurement."""
    _install_all(monkeypatch)
    source = model_dir / f"lipforensics-{lip_forensics.UPSTREAM_REVISION}"
    (source / "preprocessing" / "20words_mean_face.npy").write_bytes(b"other")

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


@pytest.mark.parametrize(
    "missing",
    [
        "weights/lipforensics_ff.pth",
        "face-alignment/checkpoints/s3fd-619a316812.pth",
        "face-alignment/checkpoints/2DFAN4-11f355bf06.pth.tar",
    ],
)
def test_a_missing_artifact_is_refused(model_dir, monkeypatch, missing):
    _install_all(monkeypatch)
    (model_dir / missing).unlink()

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


def test_an_absent_library_is_a_refusal_rather_than_an_import_error(model_dir, monkeypatch):
    """An image whose dependencies drifted fails as a signal, not as a crashed analysis."""
    _install_cv2(monkeypatch)
    _install_skimage(monkeypatch)
    _install_torch(monkeypatch)
    _install_face_alignment(monkeypatch, missing=True)

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


def test_the_evidence_names_every_artifact_behind_the_score(model_dir, monkeypatch):
    _install_all(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.upstream_repository == lip_forensics.UPSTREAM_REPOSITORY
    assert evidence.upstream_revision == lip_forensics.UPSTREAM_REVISION
    assert evidence.weights_origin == lip_forensics.WEIGHTS_ORIGIN
    assert evidence.weights_sha256 == lip_forensics.WEIGHTS_SHA256
    assert evidence.face_detector_sha256 == lip_forensics.SFD_SHA256
    assert evidence.landmark_model_sha256 == lip_forensics.FAN_SHA256
    assert evidence.source_sha256 == lip_forensics.SOURCE_SHA256
    assert evidence.landmark_library == "face-alignment 1.5.0"
    assert evidence.landmark_compiled is False
    assert evidence.torch_version == "2.13.0+cpu"
    assert evidence.device == "cpu"


def test_the_preprocessing_contract_is_recorded_with_the_score(model_dir, monkeypatch):
    _install_all(monkeypatch)

    evidence = _analyze(model_dir)

    assert evidence.frames_per_window == FRAMES_PER_WINDOW
    assert evidence.crop_size == lip_forensics.CROP_SIZE
    assert evidence.input_size == lip_forensics.INPUT_SIZE


# --- Inference that answers unreadably -------------------------------------------------------


def test_a_logit_per_run_is_required(model_dir, monkeypatch):
    _install_all(monkeypatch, answer=_FakeTensor(np.zeros(2, dtype=np.float32)))

    with pytest.raises(LipForensicsInferenceError):
        _analyze(model_dir)


def test_non_finite_logits_are_refused(model_dir, monkeypatch):
    _install_all(monkeypatch, logits=[1.0, float("nan"), 2.0, 3.0])

    with pytest.raises(LipForensicsInferenceError):
        _analyze(model_dir)


def test_a_classifier_that_raises_is_an_inference_error(model_dir, monkeypatch):
    class _Exploding:
        shape = ()

        def reshape(self, *shape):
            raise RuntimeError("kernel died")

    _install_all(monkeypatch, answer=_Exploding())

    with pytest.raises(LipForensicsInferenceError):
        _analyze(model_dir)


# --- Statelessness ---------------------------------------------------------------------------


def test_nothing_is_held_between_calls(model_dir, monkeypatch):
    """The artifacts are re-verified every call, so a swap between two jobs is caught."""
    _install_all(monkeypatch)

    assert _analyze(model_dir).windows_scored == 4

    (model_dir / "weights" / lip_forensics.WEIGHTS_FILENAME).write_bytes(b"swapped")

    with pytest.raises(LipForensicsModelUnavailable):
        _analyze(model_dir)


def test_the_score_is_never_compared_against_a_threshold(model_dir, monkeypatch):
    """There is no operating point in this module: R5-T3 is where one is measured."""
    source = Path(lip_forensics.__file__).read_text("utf-8")

    assert "THRESHOLD" not in source
    assert "T_HIGH" not in source
