"""Effort scored over one prepared artifact, under the protocol frozen for R7-T7 (R7-T11).

Effort is CLIP-L14 with orthogonal subspace decomposition, trained on FaceForensics++ and
released by `YZY-stack/Effort-AIGI-Detection`. R7-T7, R7-T8 and R7-T9 measured it over three
disjoint corpora under one frozen protocol, and R7-T10 concluded what to do with the result:
**keep it as evidence and give it no decision authority at all**.

This module is that conclusion in code. It produces a reading and nothing else. It holds no
threshold, names no operating point, and reaches no risk rule — `app.risk_engine` accepts three
evidence types and Effort is not among them, so the separation is structural rather than a
filter somebody has to remember. What R7-T9 measured as a candidate operating point,
`0.9574909508228302`, appears nowhere in this package: it is an offline artifact of one
confirmation on one corpus, and adopting it as a production number would assert a measurement
nobody took (R7-T10 §5).

**Semantic preprocessing parity, with measured numerical runtime drift.** That phrase is exact
and the distinction inside it matters, so it is worth stating before anything else: what this
module reproduces is the *pipeline* the three studies measured, not their *numbers*.

Effort classifies one aligned face crop; the media here is video, and every choice bridging that
gap was fixed in `scripts/eval/effort_freeze.json` before the evaluation split was opened. Change
any of them and the three studies stop describing the thing that runs, so all of them are
preserved here unchanged:

- the same checkpoint identity, verified by digest on every load;
- the same upstream repository at the same revision;
- the same **dlib 81-landmark** detection and five-point similarity alignment to 224x224 at
  scale 1.3;
- the same deterministic **eight frames**, evenly spaced over the clip, deduplicated;
- the same **arithmetic mean** of the per-frame outputs, over the frames that yielded a face.

R7-T11 checked that against R7-T9's own run before writing a line of this module. On the fixtures
compared, the frame accounting matched the reference exactly — frames requested, decoded and
faced, including the clips where dlib found one face of eight — and the abstention behaviour
matched, including a clip where no sampled frame held a face and the detector declined to answer
in both runs.

**The scores did not match, and are not claimed to.** R7-T9 ran on Python 3.10 with
`torch 2.5.1+cu121` on a GPU; this runs on Python 3.12 with `torch 2.13.0+cpu` on the CPU. Across
the compared fixtures the absolute difference had a mean of approximately **0.0065** and a maximum
of approximately **0.028**. The drift is a property of the runtime stack — different kernels, a
different device, a different torch — and not of the pipeline, which is why the largest gaps land
on the clips whose mean is taken over one or two frames, where a single frame's perturbation moves
the aggregate furthest. It was measured rather than assumed, and it is not tuned away: correcting
it would mean altering the frozen preprocessing or the model, which is precisely what must not
happen.

Three consequences follow, and all three are load-bearing:

1. **The drift is acceptable under the evidence-only posture, and only under it.** No production
   threshold exists for Effort and no risk rule interprets its score, so there is no comparison
   for a difference of 0.028 to change the outcome of. The moment a threshold exists, that
   sentence stops being true.

2. **The runtime identity is provenance and is persisted as such.** A stored reading records the
   Python, torch, torchvision, dlib and transformers versions and the device that produced it,
   beside the checkpoint digest and the upstream revision — because on the evidence above the
   runtime is part of what determined the number, and a reading whose runtime is unknown cannot
   be compared with one taken under a different stack.

3. **Any future calibration or decisional promotion must be measured against this environment.**
   Not against the R7-T9 score distribution, which describes a runtime that is not the one
   serving traffic. R7-T10 §10.2 named this as the ordering constraint and called calibrating
   first the expensive mistake; the drift measured here is the concrete reason it applies, and it
   applies to R7-T10 §11 B4 in full.

**Stateless, like every other local checkpoint here.** No cached module, no module-level model,
nothing held between calls — the convention `app.face_detector` and `app.lip_forensics` set, for
the reason they give: the artifacts are re-verified on every call, and 3.9 GiB of CLIP weights
stay out of the worker's resident set between jobs. The benchmark module caches instead, which is
right for a run over 882 clips and wrong for a worker with a 6 GiB cap that also has to hold the
B7 and LipForensics.

**The abstention is the reading that matters most.** A clip in which no sampled frame yields a
face produces `EffortNoFaceDetected` and no score. It is not a zero, not a low reading and not a
finding that the media is genuine — the classifier was never asked. R7-T9 abstained on 10.73% of
genuine validation lineages, and the bound it measured is therefore a claim about media in which
a face is present. `app.detection` writes that distinction into the signal rather than flattening
it, and `regression-1` — the lineage the original production incident came from — abstains here
as it did in all three studies.

Nothing in this module is a claim about whether media is fake. The number is the mean over
sampled frames of `softmax(head(pooler_output))[:, 1]` from a network trained on FaceForensics++,
on that network's own scale; what it means on media outside that distribution is the question the
three studies asked, not an assumption this code is entitled to make (AGENTS.md rule 11).
"""

from __future__ import annotations

import hashlib
import os
import platform
from dataclasses import dataclass
from pathlib import Path

# The identity of the checkpoint, transcribed from `scripts/eval/effort_freeze.json` rather than
# re-derived. A different checkpoint is a different measurement, and this one is 1.2 GB of
# somebody else's research release whose first download in R7-T7 was silently truncated at 73 MB
# and would have produced numbers rather than errors — which is why the digest is verified on
# every call and not merely at build time.
UPSTREAM_REPOSITORY = "https://github.com/YZY-stack/Effort-AIGI-Detection"
UPSTREAM_REVISION = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
CHECKPOINT_FILENAME = "effort_clip_L14_trainOn_FaceForensic.pth"
CHECKPOINT_SHA256 = "8d86711f098d16b49c048962fc3e16a906380f7bb17b8a0e89bd545b926943ee"

# The 81-landmark model every alignment is built on, from the same pinned revision. Upstream's
# README requires this exact file for this checkpoint, and a different landmark model is a
# different crop and therefore a different number.
LANDMARK_FILENAME = "shape_predictor_81_face_landmarks.dat"
LANDMARK_SHA256 = "8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869"

# The CLIP backbone the checkpoint's weights are shaped against, pinned to the HuggingFace
# revision R7-T7 loaded. Vendored into the image as two files and read from disk: a model fetched
# at analysis time is a model nobody pinned (R7-T10 §13).
CLIP_REPOSITORY = "https://huggingface.co/openai/clip-vit-large-patch14"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
CLIP_DIRNAME = "clip-vit-large-patch14"
CLIP_SHA256 = {
    "config.json": "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a",
    "model.safetensors": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
}

# Where the image puts them, laid out as the benchmark's own cache is so a developer can point
# this at the directory R7-T7 filled without editing code. Configuration, never anything a
# request can influence.
DEFAULT_MODEL_DIR = Path("/models/effort")
MODEL_DIR_ENV = "DEEPGUARD_EFFORT_MODEL_DIR"

# Whether this detector runs at all. On unless a deployment turns it off, which is the opposite
# of `DEEPGUARD_SHADOW_MODE` and deliberately so: a shadow experiment is something an operator
# opts into, and this is a detector the image ships and the pipeline expects.
#
# The switch exists for one purpose — rollback. Effort is evidence and no rule reads it, so
# turning it off returns the system to the exact SVD + B7 decisional baseline immediately, with
# no ruleset to migrate and no stored row to rewrite: analyses simply stop acquiring a signal
# that nothing was consulting. Rows already written stay readable and stay non-decisional.
ENABLED_VARIABLE = "DEEPGUARD_EFFORT_ENABLED"

# The values that turn it off, spelled out rather than "anything non-empty" for the reason
# `app.shadow` spells its own set out: a compose file carrying `=false` must mean false.
DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

# --- The frozen protocol ----------------------------------------------------------------------
#
# Restated as constants so a reader of this file sees the same numbers as a reader of the freeze
# document. Changing one here without changing it there is the bug this duplication exists to make
# visible, and both are written onto every signal so a stored reading records what produced it.

# `effort.yaml` at the pinned revision: `frame_num: {'test': 8}`.
FRAMES_PER_CLIP = 8

# `get_video_metrics` in DeepfakeBench computes a video's prediction as `pred_sum / leng`. The
# arithmetic mean, over the frames that yielded a face, and no alternative aggregation is computed
# anywhere in this module.
AGGREGATION = "arithmetic mean over frames that yielded a face"

# The device every score is produced on, and deliberately not configurable — the same reason
# `app.lip_forensics` fixes it: this image carries a CPU-only torch, and kernel selection is what
# would make one card's scores differ from another's. Recorded on every signal because it is part
# of the identity of the number.
DEVICE = "cpu"

# How many per-frame outputs one reading may leave behind in its metadata. The sample is eight, so
# nothing is truncated in practice; the cap bounds the document the way `app.lip_forensics` bounds
# its own. `frames_with_face` records how many there were either way.
MAX_PERSISTED_FRAME_SCORES = 8


class EffortError(Exception):
    """Base class for every failure raised by this module."""


class EffortModelUnavailable(EffortError):
    """A pinned artifact is missing, is not the bytes that were pinned, or will not load.

    A server/deployment fault, not a fact about the media: nothing was inferred, so nothing is
    known about the video either way. Also raised when a library this detector needs is not
    installed in the image.
    """


class EffortMediaError(EffortError):
    """The media could not be opened or decoded, or reports no frames to sample."""


class EffortNoFaceDetected(EffortError):
    """No sampled frame yielded a face dlib could detect and align.

    An abstention, and kept apart from every other failure here because it is the one that is an
    ordinary property of an upload rather than a fault. It is emphatically not a finding that the
    media is genuine: the classifier saw nothing at all, and R7-T9 excluded these clips from both
    sides of every rate it reported rather than counting them as true negatives.

    Carries the frame accounting, because an abstention is a measurement of how much could be
    read and `app.detection` records that beside the reason. "Eight frames sampled, eight decoded,
    none with a face" and "one frame sampled and the decoder returned nothing" are different
    abstentions, and a stored row that could not tell them apart would be unable to say which.
    """

    def __init__(self, message: str, frames_requested: int, frames_decoded: int) -> None:
        super().__init__(message)
        self.frames_requested = frames_requested
        self.frames_decoded = frames_decoded


class EffortInferenceError(EffortError):
    """Torch failed while inferring, or returned something this module cannot read."""


@dataclass(frozen=True)
class EffortEvidence:
    """One clip's reading, the frames behind it, and exactly what produced them.

    The provenance fields are here so a stored signal can be reproduced: a different checkpoint, a
    different upstream revision, a different landmark model or a different backbone is a different
    measurement, and a reader of the database should never have to assume which one a stored
    number came from.

    `score` is the frozen protocol's clip figure — the arithmetic mean of `frame_scores` — carrying
    no threshold, class or verdict.
    """

    upstream_repository: str
    upstream_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str
    landmark_model_sha256: str
    clip_repository: str
    clip_revision: str
    clip_sha256: dict[str, str]
    # The runtime that produced the number, recorded as provenance in its own right rather than
    # as a footnote to the model's.
    #
    # R7-T11 measured why this belongs on the row. The same checkpoint, the same alignment and the
    # same eight frames under Python 3.10 with `torch 2.5.1+cu121` on a GPU and under this stack
    # on the CPU produce readings that differ by a mean of about 0.0065 and a maximum of about
    # 0.028 — semantic parity with numerical drift. The pipeline is what was frozen; the runtime
    # is what moved, so a stored reading whose runtime is unknown cannot honestly be compared with
    # one taken under a different stack, and a calibration that pooled the two would be measuring
    # across an uncontrolled variable.
    runtime: dict[str, str]
    torch_version: str
    device: str
    # What `load_state_dict(strict=False)` let through. Upstream loads permissively and marks it
    # `# FIXME`; the permissiveness is kept because tightening it would change which weights the
    # published checkpoint is allowed to supply, but what it papered over is recorded rather than
    # discarded. `S_r`, `U_r` and `V_r` are the frozen singular components the forward pass never
    # reads, so a missing key outside that set would mean part of the scoring path is randomly
    # initialised and the score means nothing — which is why that count is separate and is
    # expected to be zero.
    missing_forward_path_key_count: int
    frames_per_clip: int
    # What the sampling actually did: how many indices were asked for, how many the decoder
    # returned, and how many of those held a face dlib could align. The three differ on ordinary
    # media, and the gaps between them are what keep `score` from reading as a statement about
    # the whole clip.
    frames_requested: int
    frames_decoded: int
    frames_with_face: int
    score: float
    frame_scores: tuple[float, ...]


def is_enabled() -> bool:
    """Whether this deployment runs Effort at all. On unless explicitly switched off.

    See `ENABLED_VARIABLE`: this is the rollback switch, and it is the only thing that decides
    whether an analysis acquires an Effort signal. It cannot affect a risk decision in either
    position, because no rule reads the signal it gates.
    """
    return os.getenv(ENABLED_VARIABLE, "").strip().lower() not in DISABLED_VALUES


def _import_cv2():
    """The OpenCV module, or a refusal that is a signal rather than a crashed job.

    Every use goes through here rather than importing at module scope, for the reason
    `app.lip_forensics` gives at the same place: the API process imports this module for its error
    types and should not pay for OpenCV to do it, and an image built without the dependency must
    produce a `FAILED` signal like any other unavailable model rather than a `ModuleNotFoundError`
    that escapes `app.detection` and takes the whole analysis down with it.
    """
    try:
        import cv2
    except ImportError as error:
        raise EffortModelUnavailable("opencv is not installed") from error

    return cv2


def _import_numpy():
    """NumPy, or a refusal. The same contract as `_import_cv2`."""
    try:
        import numpy
    except ImportError as error:
        raise EffortModelUnavailable("numpy is not installed") from error

    return numpy


def _import_torch():
    """The torch module, or a refusal. The same contract as `_import_cv2`."""
    try:
        import torch
    except ImportError as error:
        raise EffortModelUnavailable("torch is not installed") from error

    return torch


def _import_dlib():
    """The dlib module, or a refusal. The same contract as `_import_cv2`.

    The likeliest of these to be missing: this image carried no dlib at all before R7-T11, having
    deliberately excluded it when `face-alignment` asked for it (`requirements.txt`). An image
    whose dependency list drifted back fails here with a `FAILED` signal naming the library rather
    than taking an analysis down.
    """
    try:
        import dlib
    except ImportError as error:
        raise EffortModelUnavailable("dlib is not installed") from error

    return dlib


def _import_clip_model():
    """`transformers.CLIPModel`, or a refusal. The same contract as `_import_cv2`."""
    try:
        from transformers import CLIPModel
    except ImportError as error:
        raise EffortModelUnavailable("transformers is not installed") from error

    return CLIPModel


def _version_of(package: str) -> str:
    """The installed version of `package`, or `"unknown"` rather than a raised import error.

    Provenance must not be able to fail a reading. A stack whose version cannot be read is worth
    recording as exactly that — the alternative is a detector that refuses to answer because it
    could not describe itself.
    """
    from importlib import metadata

    try:
        return metadata.version(package)
    except Exception:
        return "unknown"


def _model_dir() -> Path:
    configured = os.getenv(MODEL_DIR_ENV, "").strip()

    return Path(configured) if configured else DEFAULT_MODEL_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _verified(path: Path, expected: str, what: str) -> Path:
    """`path`, once its bytes are the bytes that were pinned.

    Re-executed on every call rather than trusted from the build, which is this module's
    statelessness rather than an accident: a bind mount, a rebuilt layer or a developer pointing
    `DEEPGUARD_EFFORT_MODEL_DIR` at a benchmark cache all put bytes in front of this process that
    no build ever saw. R7-T7's truncated first download is the case this exists for — 73 MB of
    1,158 MB, which loads far enough to produce plausible numbers.
    """
    if not path.exists():
        raise EffortModelUnavailable(f"{what} not found")

    actual = _sha256(path)
    if actual != expected:
        raise EffortModelUnavailable(
            f"{what} has sha256 {actual}, but the frozen protocol pins {expected}"
        )

    return path


def _load(directory: Path):
    """Assemble the pinned model, having first checked it is the model that was pinned.

    Ordered so the cheap refusals happen first: a missing or wrong-hashed file should fail in
    milliseconds rather than after the minute of SVD work that building the backbone costs.

    The architecture is built exactly as `EffortDetector.build_backbone` does — the SVD pass here
    only establishes parameter shapes, and every value it computes is replaced by the checkpoint's.
    """
    # The artifacts before the libraries, which is what "cheap refusals first" means here: hashing
    # four files costs seconds, and importing torch, transformers and dlib costs a great deal more
    # than that. It also keeps the refusals honest — a deployment missing both a library and the
    # right checkpoint should be told about the checkpoint, because that is the fault that would
    # still be there after the library was installed.
    checkpoint_path = _verified(
        directory / CHECKPOINT_FILENAME, CHECKPOINT_SHA256, "Effort checkpoint"
    )
    landmark_path = _verified(
        directory / LANDMARK_FILENAME, LANDMARK_SHA256, "Effort landmark model"
    )

    clip_dir = directory / CLIP_DIRNAME
    for filename, expected in CLIP_SHA256.items():
        _verified(clip_dir / filename, expected, f"CLIP backbone file {filename}")

    torch = _import_torch()
    CLIPModel = _import_clip_model()
    dlib = _import_dlib()

    from app import effort_upstream

    try:
        clip_model = CLIPModel.from_pretrained(clip_dir, local_files_only=True)
        backbone = effort_upstream.apply_svd_residual_to_self_attn(
            clip_model.vision_model, r=effort_upstream.SVD_RANK
        )
        model = effort_upstream.EffortModel(backbone)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.replace("module.", ""): value for key, value in state.items()}

        # `strict=False` is upstream's own, kept for the reason the module docstring gives, and
        # audited rather than trusted — see `EffortEvidence.missing_forward_path_key_count`.
        incompatible = model.load_state_dict(state, strict=False)
        training_only = ("S_r", "U_r", "V_r")
        missing_forward_path = [
            key
            for key in incompatible.missing_keys
            if key.rsplit(".", 1)[-1] not in training_only
        ]

        model.eval().to(torch.device(DEVICE))
    except EffortError:
        raise
    except Exception as error:
        raise EffortModelUnavailable("the Effort model could not be assembled") from error

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(str(landmark_path))

    # The stack this reading will have been produced by, read off the modules actually imported
    # rather than off the pin file, because the pin file says what was requested and this says
    # what answered.
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _version_of("torchvision"),
        "dlib": getattr(dlib, "__version__", "unknown"),
        "transformers": _version_of("transformers"),
    }

    return model, face_detector, landmark_predictor, len(missing_forward_path), runtime


def _sample_indices(frame_count: int) -> list[int]:
    """The frozen sampling rule: eight evenly spaced indices, deduplicated, deterministic.

    `linspace` over the closed range rather than a stride, so the last frame is always included and
    a 30-frame clip and a 3000-frame clip are sampled the same way in proportion rather than in
    absolute position. Deduplication matters only for clips shorter than eight frames, where the
    freeze says the clip contributes every frame it has.

    Transcribed from the benchmark module rather than reimplemented, because "evenly spaced" has
    more than one reasonable reading and the studies measured this one.
    """
    numpy = _import_numpy()

    if frame_count <= 0:
        return []
    if frame_count <= FRAMES_PER_CLIP:
        return list(range(frame_count))

    return sorted({int(index) for index in numpy.linspace(0, frame_count - 1, FRAMES_PER_CLIP)})


def _frame_scores(
    path: Path, model, face_detector, landmark_predictor
) -> tuple[list[float], int, int, int]:
    """Score every sampled frame that yields a face. Returns the scores and what happened."""
    cv2 = _import_cv2()
    torch = _import_torch()

    from app import effort_upstream

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise EffortMediaError("the prepared artifact could not be opened for decoding")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = _sample_indices(frame_count)
        if not indices:
            raise EffortMediaError("the prepared artifact reports no frames to sample")

        scores: list[float] = []
        decoded = 0
        faces_found = 0

        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            read, frame = capture.read()
            if not read:
                continue
            decoded += 1

            try:
                aligned = effort_upstream.extract_aligned_face_dlib(
                    face_detector,
                    landmark_predictor,
                    frame,
                    res=effort_upstream.RESOLUTION,
                )
            except Exception as error:
                raise EffortInferenceError("aligning a sampled frame failed") from error

            if aligned is None:
                continue
            faces_found += 1

            try:
                tensor = effort_upstream.preprocess_face(aligned).to(torch.device(DEVICE))
                with torch.inference_mode():
                    score = model(tensor).squeeze().item()
            except Exception as error:
                raise EffortInferenceError("scoring a sampled frame failed") from error

            scores.append(float(score))

        return scores, len(indices), decoded, faces_found
    finally:
        capture.release()


def analyze_effort(path: Path) -> EffortEvidence:
    """Effort's reading of one prepared artifact, under the frozen protocol.

    Raises rather than returning a sentinel, and the exception type is the finding: an artifact
    that is not the pinned one, media that will not decode, a clip with no detectable face, or
    torch breaking are four different facts and `app.detection` records them as four different
    rows. None of them is a score.

    Blocking CPU work — roughly 28 s to assemble the model and 9 s to score eight frames of
    ordinary 1080p footage, dlib's detector being the bulk of the second figure. Called
    synchronously from the worker, after the two cheaper local checkpoints.
    """
    torch = _import_torch()
    directory = _model_dir()

    model, face_detector, landmark_predictor, missing_forward_path, runtime = _load(directory)

    scores, requested, decoded, faces_found = _frame_scores(
        path, model, face_detector, landmark_predictor
    )

    if not scores:
        raise EffortNoFaceDetected(
            f"no face detected in any of {decoded} decoded frames "
            f"(of {requested} sampled)",
            frames_requested=requested,
            frames_decoded=decoded,
        )

    # The arithmetic mean, and nothing else. `get_video_metrics` upstream aggregates a video this
    # way and the three studies measured it this way; a median, a maximum or a trimmed mean here
    # would be a different detector wearing this one's evidence.
    score = float(sum(scores) / len(scores))

    return EffortEvidence(
        upstream_repository=UPSTREAM_REPOSITORY,
        upstream_revision=UPSTREAM_REVISION,
        checkpoint_filename=CHECKPOINT_FILENAME,
        checkpoint_sha256=CHECKPOINT_SHA256,
        landmark_model_sha256=LANDMARK_SHA256,
        clip_repository=CLIP_REPOSITORY,
        clip_revision=CLIP_REVISION,
        clip_sha256=dict(CLIP_SHA256),
        runtime=runtime,
        torch_version=torch.__version__,
        device=DEVICE,
        missing_forward_path_key_count=missing_forward_path,
        frames_per_clip=FRAMES_PER_CLIP,
        frames_requested=requested,
        frames_decoded=decoded,
        frames_with_face=faces_found,
        score=score,
        frame_scores=tuple(scores[:MAX_PERSISTED_FRAME_SCORES]),
    )
