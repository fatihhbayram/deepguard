"""Effort's model definition and face preprocessing, transcribed from upstream rather than rewritten.

Everything in this module is a transcription of code from `YZY-stack/Effort-AIGI-Detection` at
revision `96f5dea2b534d400cfd7003f053c7e93c8e16461`, specifically:

- `DeepfakeBench/training/detectors/effort_detector.py` — `SVDResidualLinear`,
  `replace_with_svd_residual`, `apply_svd_residual_to_self_attn`, and the forward path
  (`pooler_output` → `Linear(1024, 2)` → `softmax(...)[:, 1]`);
- `DeepfakeBench/training/demo.py` — `get_keypts`, `extract_aligned_face_dlib` and
  `preprocess_face`.

**Why transcribe instead of vendoring the repository.** Upstream's detector module imports the
whole DeepfakeBench training stack — a `DETECTOR` registry, a `BACKBONE` registry, a `LOSSFUNC`
registry, `loralib`, `tensorboard`, sklearn metrics and a `Trainer` — none of which participates
in a forward pass. Importing all of it to call one `nn.Module` would make the benchmark's
dependency surface much larger than the thing being benchmarked, and this harness is explicitly
the code that must not be able to reach into anything it does not need (`scripts/benchmark/cli.py`).

**Why transcribe instead of paraphrase.** The alignment in particular is a pile of magic numbers
— a canonical five-point template, a 1.3 scale margin, a similarity transform — and any of them
retyped slightly differently produces a face crop that is *almost* the one the checkpoint was
trained on. A detector fed a subtly wrong crop does not fail loudly; it returns plausible numbers
that are quietly about a different input distribution. So the arithmetic below is copied, not
reconstructed from the description, and the docstrings say where each piece came from.

The one deliberate behavioural change is marked `DEVIATION` at its site: upstream's
`extract_aligned_face_dlib` returns a two-tuple on the no-face path and a three-tuple otherwise,
which makes its own caller raise a `ValueError` when no face is found. This module returns `None`
instead, so that "no face in this frame" is a value the caller can handle rather than an
exception it has to pattern-match. Nothing about the arithmetic on the face-found path differs.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as pil_image
from skimage import transform as trans
from torchvision import transforms as T

# CLIP's normalisation constants, as declared in `effort.yaml` at the pinned revision and used by
# `preprocess_face` in `demo.py`. Restated here rather than imported so the file is self-contained.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# The input resolution the checkpoint was trained at (`effort.yaml`: `resolution: 224`).
RESOLUTION = 224

# `apply_svd_residual_to_self_attn(vision_model, r=1024-1)` in `EffortDetector.build_backbone`.
# The comment there reads "ViT-L/14 224*224: 1024-1".
SVD_RANK = 1024 - 1


class SVDResidualLinear(nn.Module):
    """A linear layer split into a fixed top-`r` singular component and a trainable residual.

    Transcribed from `effort_detector.py`. This is the whole method: the top singular subspace of
    the pretrained CLIP attention weight is frozen, and only the orthogonal remainder is trained,
    so the detector learns a manipulation cue without dismantling the representation it inherits.

    Only `forward` and the parameter shapes matter for inference. The training-time regularisers
    (`compute_orthogonal_loss`, `compute_keepsv_loss`, `compute_fn_loss`) are not transcribed —
    nothing in a shadow benchmark trains this model — but every tensor they would have touched is
    still registered, because those tensors are keys in the published checkpoint and a module that
    did not declare them would silently drop them at load time.
    """

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.weight_main = nn.Parameter(torch.Tensor(out_features, in_features))
        if init_weight is not None:
            self.weight_main.data.copy_(init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if (
            hasattr(self, "U_residual")
            and hasattr(self, "V_residual")
            and self.S_residual is not None
        ):
            residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            weight = self.weight_main + residual_weight
        else:
            weight = self.weight_main
        return F.linear(x, weight, self.bias)


def replace_with_svd_residual(module, r):
    """Decompose one `nn.Linear` into an `SVDResidualLinear`. From `effort_detector.py`.

    The SVD here is what establishes the parameter *shapes* the checkpoint expects. Its computed
    *values* are then overwritten by `load_state_dict`, so this pass costs a minute of startup and
    contributes nothing to the final weights — but skipping it would leave the module with no
    `U_residual`/`S_residual`/`V_residual` to load into.
    """
    if not isinstance(module, nn.Linear):
        return module

    in_features = module.in_features
    out_features = module.out_features
    bias = module.bias is not None

    new_module = SVDResidualLinear(
        in_features, out_features, r, bias=bias, init_weight=module.weight.data.clone()
    )
    if bias and module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)

    new_module.weight_original_fnorm = torch.norm(module.weight.data, p="fro")

    U, S, Vh = torch.linalg.svd(module.weight.data, full_matrices=False)
    r = min(r, len(S))

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    weight_main = U_r @ torch.diag(S_r) @ Vh_r
    new_module.weight_main_fnorm = torch.norm(weight_main.data, p="fro")
    new_module.weight_main.data.copy_(weight_main)

    U_residual = U[:, r:]
    S_residual = S[r:]
    Vh_residual = Vh[r:, :]

    if len(S_residual) > 0:
        new_module.S_residual = nn.Parameter(S_residual.clone())
        new_module.U_residual = nn.Parameter(U_residual.clone())
        new_module.V_residual = nn.Parameter(Vh_residual.clone())
        new_module.S_r = nn.Parameter(S_r.clone(), requires_grad=False)
        new_module.U_r = nn.Parameter(U_r.clone(), requires_grad=False)
        new_module.V_r = nn.Parameter(Vh_r.clone(), requires_grad=False)
    else:
        new_module.S_residual = None
        new_module.U_residual = None
        new_module.V_residual = None
        new_module.S_r = None
        new_module.U_r = None
        new_module.V_r = None

    return new_module


def apply_svd_residual_to_self_attn(model, r):
    """Replace every `nn.Linear` inside a `self_attn` module. From `effort_detector.py`.

    Only attention projections are decomposed; MLP blocks, embeddings and layer norms keep their
    original CLIP weights. That selectivity is the method, so it is preserved exactly.
    """
    for name, module in model.named_children():
        if "self_attn" in name:
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.Linear):
                    parent_module = module
                    sub_module_names = sub_name.split(".")
                    for module_name in sub_module_names[:-1]:
                        parent_module = getattr(parent_module, module_name)
                    setattr(
                        parent_module,
                        sub_module_names[-1],
                        replace_with_svd_residual(sub_module, r),
                    )
        else:
            apply_svd_residual_to_self_attn(module, r)
    return model


class EffortModel(nn.Module):
    """The scoring path of upstream's `EffortDetector`: backbone → head → softmax.

    `features` / `classifier` / the `softmax(pred, dim=1)[:, 1]` in `forward` are transcribed from
    `effort_detector.py`. The training-only members of that class (loss function, metric
    accumulators, `get_losses`) are omitted; they hold no weights the checkpoint carries.

    The returned score is the posterior for class 1 under a two-class cross-entropy head. Upstream
    defines no calibrated confidence semantics for it and neither does this benchmark.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(1024, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)["pooler_output"]
        pred = self.head(features)
        return torch.softmax(pred, dim=1)[:, 1]


def get_keypts(image, face, predictor):
    """The five alignment points, taken from the 81-point shape. From `demo.py`.

    Points 37 and 44 are eye corners, 30 the nose tip, 49 and 55 the mouth corners. The indices
    are upstream's and are the ones the training-time crops were built from.
    """
    shape = predictor(image, face)

    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def extract_aligned_face_dlib(face_detector, predictor, image, res=RESOLUTION, mask=None):
    """Detect the largest face and warp it onto the canonical template. From `demo.py`.

    The template, the `+= 8.0` shift for a 112-wide target, the 1.3 scale margin and the margin
    renormalisation are all upstream constants. They define the exact framing the checkpoint was
    trained under, which is why they are copied rather than restated.

    DEVIATION: upstream returns the bare two-tuple `(None, None)` when dlib finds no face, which
    makes its own three-way unpacking raise. This returns `None` so that "no face here" is an
    ordinary value; the caller treats it as an abstaining frame under the frozen protocol.
    """

    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        target_size = [112, 112]
        dst = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.0
        y_margin = target_size[1] * margin_rate / 2.0

        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))

        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        return img

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = face_detector(rgb, 1)
    if not len(faces):
        return None  # DEVIATION, see the docstring.

    face = max(faces, key=lambda rect: rect.width() * rect.height())
    landmarks = get_keypts(rgb, face, predictor)
    cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
    return cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)


_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ]
)


def preprocess_face(img_bgr: np.ndarray) -> torch.Tensor:
    """Aligned crop → normalised `1x3x224x224` tensor. From `demo.py`.

    The redundant-looking resize is upstream's and is kept: `extract_aligned_face_dlib` already
    produced 224x224, so this is a no-op on that path, but removing it would make this function
    behave differently from upstream's on any other input.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LINEAR)
    return _TRANSFORM(pil_image.fromarray(img_rgb)).unsqueeze(0)
