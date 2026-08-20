"""Turning one detector's answer about one video into forensic evidence.

This is what the worker runs. It sits beside `media.py` and `normalization.py` as another
step that takes a local artifact and describes it — it opens no transaction, writes no
row, and does not know a job exists.

It lived in the upload route until P3-T2, back when detection happened on the request.
"""

import logging
from pathlib import Path

from app.db.models import (
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
    AnalysisSegment,
    AnalysisSignal,
)
from app.nvidia_video import (
    NvidiaClipResult,
    NvidiaProviderError,
    NvidiaProviderTimeout,
    analyze_video,
)

logger = logging.getLogger(__name__)

# Identity of the one detector wired in so far. NVIDIA's D009 answers exactly one
# question — whether the video is synthetic — so the pair is a constant, not a lookup.
NVIDIA_PROVIDER = "nvidia"
SYNTHETIC_VIDEO_SIGNAL = "synthetic_video"

# How many clip results one detection may leave behind. NVIDIA scores a clip every few
# frames, so a long video produces thousands and persisting all of them would let one
# provider's output grow the database without limit. The strongest evidence is kept and
# the rest is dropped; `total_clips` on the signal records how many there were, so a
# truncated set is never mistaken for the whole one (D019).
MAX_PERSISTED_SEGMENTS = 20


def strongest_clips(clips: tuple[NvidiaClipResult, ...]) -> list[AnalysisSegment]:
    """The provider's most strongly scored clips, capped at what may be persisted.

    Ranking is by NVIDIA's own logit, descending: a higher logit is the provider saying
    this clip looks more synthetic, so keeping the top of that ordering keeps the evidence
    a reader would actually want to see when the aggregate looks suspicious. The clip
    index breaks ties, so the same detection always yields the same rows.

    This is a selection, not a judgement. No threshold decides which clips are "suspicious"
    and nothing here is classified — the figures are handed on exactly as NVIDIA sent them,
    and what they mean is still nobody's call in this codebase.
    """
    ranked = sorted(clips, key=lambda clip: (-clip.logit, clip.index))

    return [
        AnalysisSegment(clip_index=clip.index, logit=clip.logit)
        for clip in ranked[:MAX_PERSISTED_SEGMENTS]
    ]


async def detect_synthetic_video(file_path: Path) -> tuple[AnalysisSignal, list[AnalysisSegment]]:
    """Ask NVIDIA about the local artifact and turn the outcome into forensic evidence.

    Every *provider* failure becomes a signal rather than an exception. A detector that
    fails is a fact about the detector, not about the media: the analysis is still real,
    its media is still stored, and the failure is recorded in its own right so the gap is
    visible rather than silent. The job that ran it finishes either way — it did the work
    it was queued to do.

    A failure on our own side is not that. `NvidiaLocalFileError` means the artifact the
    worker fetched moments ago can no longer be read — a broken machine, not a broken
    provider — and recording it as a provider signal would blame NVIDIA for our fault and
    leave a real defect looking like routine evidence. It propagates, and the job fails.

    NVIDIA's probability is written down exactly as returned, on NVIDIA's own scale.
    Interpreting it — the `risk_level` column — is not this task's, or this layer's, job,
    so it stays null. `metadata` keeps only NVIDIA's two aggregate figures; the clip table
    it also returns is timeline evidence and travels separately, as `analysis_segments`
    rows, rather than being dumped into the JSON column as unbounded provider output.

    Returns the signal and the clip evidence behind it, neither of them attached to
    anything yet. A failure has no clip evidence: the list is empty, never a placeholder
    row.
    """
    signal = AnalysisSignal(
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
    )

    try:
        result = await analyze_video(file_path)
    except NvidiaProviderTimeout as error:
        # The one failure worth telling apart at a glance: NVIDIA may still have been
        # working, so this says nothing about the video either way.
        logger.warning("NVIDIA synthetic-video detection timed out.", exc_info=True)
        signal.status = SIGNAL_STATUS_TIMEOUT
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []
    except NvidiaProviderError as error:
        # Every other provider-side outcome — rejected credentials, an unreachable
        # service, a truncated stream — is one status. Telling them apart in the schema
        # would need product requirements that do not exist yet; the server log carries
        # the detail, and the message never reaches the client.
        logger.warning("NVIDIA synthetic-video detection failed.", exc_info=True)
        signal.status = SIGNAL_STATUS_FAILED
        # The failure kind, never NVIDIA's message: that text can quote request detail.
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []

    signal.status = SIGNAL_STATUS_SUCCESS
    # Untransformed and unrounded. This is NVIDIA's number, not DeepGuard's.
    signal.score = result.probability
    signal.provider_version = result.function_id
    signal.signal_metadata = {"logit": result.logit, "total_clips": result.total_clips}

    return signal, strongest_clips(result.clips)
