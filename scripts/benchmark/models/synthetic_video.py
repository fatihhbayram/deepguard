"""The NVIDIA synthetic-video detector, driven through DeepGuard's own preparation path.

`detect` is the callable the harness runs — one `Clip` in, one probability out:

    python3 scripts/benchmark/cli.py --model benchmark.models.synthetic_video:detect ...

**This module runs inside the `api` container and nowhere else.** It imports
`app.media`, `app.normalization` and `app.nvidia_video`, which is a deliberate exception
to the harness's isolation from the application: production almost never sends NVIDIA the
uploaded bytes — it probes the media, decides from the probe whether a derivative is owed,
and transcodes when it is (D013, D020). A calibration run over the raw source files would
measure a pipeline this service does not operate. What is imported is the *preparation and
provider* path only: no FastAPI app, no SQLAlchemy session, no worker, no storage client.
Nothing here opens a transaction, writes a row or stores an object, so a calibration pass
leaves the evidence store exactly as it found it.

    app.media.probe_media
        -> app.normalization.needs_normalization
        -> app.normalization.normalize_to_mp4   (only when the probe says one is owed)
        -> app.nvidia_video.analyze_video

**Provenance.** NVIDIA publishes no model or weights version; the NVCF function id is the
only version handle there is (D017), and it is what a calibration binds to. Every response
is checked against the configured id and a mismatch raises rather than being scored, so
`provenance()` reports the deployments that actually answered rather than the one the
environment asked for.

**The score is NVIDIA's own `probability`, untransformed.** The provider computes it as
`expit(mean(clip logits))` inside its own service (P7-T2 §1.1); this module rescales
nothing and aggregates nothing of its own.
"""

import asyncio
import os
from pathlib import Path

from app.media import probe_media
from app.normalization import needs_normalization, normalize_to_mp4
from app.nvidia_video import PREVIEW_TARGET, analyze_video

from benchmark.dataset import Clip

# What the benchmark record cannot carry and the calibration needs: which deployment
# answered, and how many clips each aggregate was computed over. Populated as the run
# proceeds and read afterwards by `provenance()`.
_OBSERVED_FUNCTION_IDS: set[str] = set()
_TOTAL_CLIPS: dict[str, int] = {}


class ProviderIdentityError(RuntimeError):
    """A deployment other than the configured one answered.

    Raised rather than scored. A threshold is only valid for the deployment it was
    measured against — P7-T2 selected one with 0.0096 of margin over observed genuine
    media — so a score from an unknown deployment is not a weaker measurement, it is a
    measurement of something else.
    """


def _prepared(clip: Clip) -> tuple[Path, bool, Path | None]:
    """Return the file production would send NVIDIA, and whether it was transcoded."""
    metadata = asyncio.run(probe_media(clip.path))
    if not needs_normalization(metadata):
        return clip.path, False, None
    derivative = asyncio.run(normalize_to_mp4(clip.path, metadata.frame_rate))
    return derivative.path, True, derivative.path


def detect(clip: Clip) -> float:
    """NVIDIA's aggregate probability that `clip` is synthetic, in `[0, 1]`.

    The derivative is removed afterwards whatever happens: a corpus of a hundred and fifty
    clips would otherwise leave a hundred and fifty transcodes in the container's temp
    directory.
    """
    target, _, temporary = _prepared(clip)
    try:
        result = asyncio.run(analyze_video(target))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    configured = os.getenv("NVIDIA_SVD_FUNCTION_ID")
    if configured and result.function_id != configured:
        raise ProviderIdentityError(
            f"expected function id {configured}, got {result.function_id}"
        )
    _OBSERVED_FUNCTION_IDS.add(result.function_id)
    _TOTAL_CLIPS[clip.clip_id] = result.total_clips
    return float(result.probability)


def provenance() -> dict:
    """Identity of what produced this run's scores, as observed rather than as configured.

    `total_clips` travels with it because the risk engine already refuses to classify an
    aggregate computed over an empty clip table (`R012`), and a calibration that never
    recorded the counts could not tell whether any sample hit that degeneracy.
    """
    return {
        "provider": "nvidia",
        "signal_type": "synthetic_video",
        "grpc_package": "nvidia.maxine.syntheticvideodetector.v1",
        "target": PREVIEW_TARGET,
        "configured_function_id": os.getenv("NVIDIA_SVD_FUNCTION_ID"),
        "observed_function_ids": sorted(_OBSERVED_FUNCTION_IDS),
        "score_semantics": "provider aggregate probability, expit(mean clip logit)",
        "preparation": "app.media.probe_media -> app.normalization -> app.nvidia_video",
        "total_clips_by_clip_id": dict(sorted(_TOTAL_CLIPS.items())),
    }
