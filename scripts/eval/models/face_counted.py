"""EfficientNet-B7, scored exactly as R3-T1 scores it, with the face-crop count recorded.

    PYTHONPATH=scripts ~/.venvs/deepguard-benchmark/bin/python scripts/benchmark/cli.py \
        --manifest ../deepguard-corpus/r7t5/manifest.csv \
        --model eval.models.face_counted:detect \
        --output-dir ../deepguard-corpus/runs/r7t5-face

The same reasoning as `lip_counted`, applied to the other locally executed detector:
`is_usable_face` requires `frames_scored > 0`, so an offline replay of the production rules
needs to know how many face crops the stored mean was actually taken over, and the benchmark
harness records only the returned float.

`_face_crops` is intercepted and the crop count recorded; `detect` is the detector's own,
unchanged. A clip in which no face was found anywhere in the sample raises out of `detect` as it
always did, is recorded by the harness as an excluded clip, and carries a count of `0` — an
abstention, which the production rules treat as silence rather than as a finding of authentic
media.
"""

from __future__ import annotations

from benchmark.dataset import Clip
from benchmark.models import face_manipulation as _face

# clip_id -> face crops the clip's mean was taken over.
FRAMES_SCORED: dict[str, int] = {}

_original_face_crops = _face._face_crops


def _counting_face_crops(clip: Clip):
    crops = _original_face_crops(clip)
    FRAMES_SCORED[clip.clip_id] = len(crops)
    return crops


_face._face_crops = _counting_face_crops


def detect(clip: Clip) -> float:
    """The B7's face-manipulation score for `clip`, on that classifier's own scale."""
    return _face.detect(clip)


def provenance() -> dict:
    """The detector's own provenance, plus the per-clip crop counts this run observed."""
    return {
        **_face.provenance(),
        "frames_scored_by_clip_id": dict(sorted(FRAMES_SCORED.items())),
    }
