"""LipForensics, scored exactly as R5-T1 scores it, with the run count recorded alongside.

    PYTHONPATH=scripts ~/.venvs/deepguard-lipforensics-cuda/bin/python scripts/benchmark/cli.py \
        --manifest ../deepguard-corpus/r7t5/manifest.csv \
        --model eval.models.lip_counted:detect \
        --output-dir ../deepguard-corpus/runs/r7t5-lip

The risk engine does not classify a score on its own: `is_usable_lip` also requires
`windows_scored > 0`, because a mean taken over no runs and a mean taken over four are not the
same evidence and `R012` exists to say which is which. An offline replay of the production rules
therefore needs that count, and `benchmark.cli` records only the float a model returns.

**So the count is observed rather than reconstructed.** This module intercepts the one function
inside the detector that already computes the thing being counted — `_windows`, which returns
the sampled runs that held a trackable face — records how many it produced, and hands the result
straight back. The score then comes from the detector's own `detect`, unchanged: nothing here
re-implements the sampling, the logit average or the sigmoid, so a score from this module is the
R5-T1 score or it is a bug in this file rather than a second opinion.

The alternative was to assume that a successful score implies some particular count. It does
not, and a replay built on an assumed count would report the rules' behaviour on evidence the
detector never produced.

A clip in which no run held a trackable face still raises out of `detect`, and the harness still
records it as an excluded clip. The count recorded for it is `0`, which is the honest reading —
the detector abstained — and the replay treats it as the production rules treat an abstention:
silence, deciding nothing.

Nothing under `apps/` is imported, and nothing on disk is modified. The interception is a
rebinding inside this process that lasts exactly as long as the run.
"""

from __future__ import annotations

from benchmark.dataset import Clip
from benchmark.models import lipforensics as _lip

# clip_id -> runs of 25 consecutive frames that actually held a trackable face. Read back by
# `provenance` after the run and written into `results.json`, where the replay picks it up.
WINDOWS_SCORED: dict[str, int] = {}

_original_windows = _lip._windows


def _counting_windows(clip: Clip):
    windows = _original_windows(clip)
    WINDOWS_SCORED[clip.clip_id] = len(windows)
    return windows


_lip._windows = _counting_windows


def detect(clip: Clip) -> float:
    """LipForensics' mouth-dynamics score for `clip`, on LipForensics' own scale.

    Not a confidence. The provider defines no calibrated confidence semantics for this number:
    it is `sigmoid(mean window logit)` from a network trained on four FaceForensics++
    manipulation families, and what it means outside those families is the question this task
    is measuring rather than an assumption it may rely on.
    """
    return _lip.detect(clip)


def provenance() -> dict:
    """The detector's own provenance, plus the per-clip run counts this run observed."""
    return {
        **_lip.provenance(),
        "windows_scored_by_clip_id": dict(sorted(WINDOWS_SCORED.items())),
    }
