"""Turning what one evidence source says about one artifact into forensic evidence.

This is what the worker runs. It sits beside `media.py` and `normalization.py` as another
step that takes a local artifact and describes it — it opens no transaction, writes no
row, and does not know a job exists.

Two sources are wired in and they answer different questions about different artifacts:
NVIDIA is asked whether the canonical derivative looks synthetic, C2PA is read off the
forensic original to see what provenance travels with the bytes as uploaded. Each becomes
its own signal row and neither is ever folded into the other (rule 11).

It lived in the upload route until P3-T2, back when detection happened on the request.
"""

import logging
from pathlib import Path

from app.c2pa_extractor import C2paEvidence, extract_c2pa_evidence
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

# Identity of the provenance signal. C2PA is not a provider that can be asked anything —
# the credentials are in the file or they are not — so the "provider" is the standard
# whose data was read, and `provider_version` records the SDK that read it.
C2PA_PROVIDER = "c2pa"
PROVENANCE_SIGNAL = "provenance"

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


def provenance_metadata(evidence: C2paEvidence) -> dict:
    """The extractor's facts, flattened for the signal's JSON column.

    Every key is present on every successful reading, including one that found no
    credentials at all, so a reader never has to tell "absent field" from "absent
    provenance": `manifest_exists` is the field that answers that, and the rest are null.

    The extractor's verbatim `manifest_json` is deliberately left out. It is unbounded
    provider output, and the column holds a small supporting document — the same reason
    NVIDIA's clip table travels as `analysis_segments` rows rather than JSON.
    """
    return {
        "manifest_exists": evidence.manifest_exists,
        "validation_state": evidence.validation_state,
        "validation_failures": list(evidence.validation_failures),
        "is_embedded": evidence.is_embedded,
        "remote_manifest_url": evidence.remote_manifest_url,
        "active_manifest_label": evidence.active_manifest_label,
        "claim_generator": evidence.claim_generator,
        "signature_issuer": evidence.signature_issuer,
        "signature_time": evidence.signature_time,
        "assertion_labels": list(evidence.assertion_labels),
    }


def extract_provenance(file_path: Path) -> AnalysisSignal:
    """Read the artifact's C2PA provenance and turn it into forensic evidence.

    Media carrying no credentials is a successful reading, not a failure: the question
    "what provenance travels with these bytes?" was answered, and the answer is "none".
    Recording that as a failed signal would hide the most common answer there is, and
    reading it as suspicion would be a verdict this layer has no business forming — most
    cameras and editors write no credentials at all.

    `score` stays null. There is no number here: provenance is a set of facts about a
    signature, not a figure on a scale, and inventing one would put it next to NVIDIA's
    probability as if the two could be compared (rule 11).

    Nothing raises. Extraction that goes wrong is recorded as a `FAILED` signal with the
    failure kind and nothing else — the same treatment a provider failure gets, for the
    same reason: one evidence source breaking must not cost the analysis the others.
    """
    signal = AnalysisSignal(provider=C2PA_PROVIDER, signal_type=PROVENANCE_SIGNAL)

    try:
        evidence = extract_c2pa_evidence(file_path)
    except Exception as error:
        # Broad on purpose. Below this line is a third-party SDK over a native library,
        # and the failure that matters to the caller is that provenance is unknown, not
        # which of the SDK's exceptions said so.
        logger.warning("C2PA extraction failed for the analysed artifact.", exc_info=True)
        signal.status = SIGNAL_STATUS_FAILED
        # The failure kind, never the message: it quotes the local artifact's path.
        signal.signal_metadata = {"error": type(error).__name__}
        return signal

    signal.status = SIGNAL_STATUS_SUCCESS
    signal.provider_version = evidence.sdk_version
    signal.signal_metadata = provenance_metadata(evidence)

    return signal


def undetectable_media(error: Exception) -> AnalysisSignal:
    """Record that NVIDIA could not be asked, because the artifact could not be prepared.

    NVIDIA takes MP4/H.264 and nothing else, so media in any other shape has to be
    transcoded before there is anything to send. When that transcode fails the provider
    was never reached — but the outcome for the evidence board is the same one a refused
    call produces: this source has no answer about this media, and the gap is recorded
    rather than left silent. Written as a real signal so an analysis that got provenance
    still keeps it, instead of the whole job being thrown away over one source (D016).

    `status` is `FAILED` even for a transcode that ran out of time. `TIMEOUT` means a
    provider that may still have been working, which says nothing about the media either
    way; ffmpeg giving up is not that, and the two are told apart by the failure kind in
    the metadata rather than by borrowing a status that would misdescribe one of them.
    """
    return AnalysisSignal(
        provider=NVIDIA_PROVIDER,
        signal_type=SYNTHETIC_VIDEO_SIGNAL,
        status=SIGNAL_STATUS_FAILED,
        # The failure kind, never the message: it quotes the local artifact's path.
        signal_metadata={"error": type(error).__name__},
    )


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
