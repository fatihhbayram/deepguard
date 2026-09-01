"""Turning what one evidence source says about one artifact into forensic evidence.

This is what the worker runs. It sits beside `media.py` and `normalization.py` as another
step that takes a local artifact and describes it — it opens no transaction, writes no
row, and does not know a job exists.

Four sources are wired in and they answer different questions about different artifacts:
NVIDIA's synthetic-video detector is asked whether the canonical derivative looks
generated, NVIDIA's Active Speaker NIM is asked which visible face is speaking when in
that same derivative, the local AASIST checkpoint is run over the audio prepared from it,
and C2PA is read off the forensic original to see what provenance travels with the bytes
as uploaded. Each becomes its own signal row and none is ever folded into another
(rule 11).

It lived in the upload route until P3-T2, back when detection happened on the request.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from app.audio_detector import (
    BONA_FIDE_LOGIT_INDEX,
    AudioAuthenticityEvidence,
    AudioDetectorError,
    analyze_audio_authenticity,
)
from app.c2pa_extractor import C2paEvidence, extract_c2pa_evidence
from app.db.models import (
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
    AnalysisSegment,
    AnalysisSignal,
)
from app.nvidia_active_speaker import (
    DiarizationSegment,
    NvidiaActiveSpeakerError,
    NvidiaActiveSpeakerResult,
    NvidiaActiveSpeakerTimeout,
    analyze_active_speaker,
)
from app.nvidia_video import (
    NvidiaClipResult,
    NvidiaProviderError,
    NvidiaProviderTimeout,
    analyze_video,
)
from app.speaker_diarization import (
    SpeakerDiarizationError,
    SpeakerTurn,
    diarize_speakers,
    prepared_audio,
)

logger = logging.getLogger(__name__)

# Identity of the one detector wired in so far. NVIDIA's D009 answers exactly one
# question — whether the video is synthetic — so the pair is a constant, not a lookup.
NVIDIA_PROVIDER = "nvidia"
SYNTHETIC_VIDEO_SIGNAL = "synthetic_video"

# The second question NVIDIA is asked, and a wholly separate signal from the first. It
# runs against the same artifact and is hosted by the same company, which is not a reason
# to record one answer where the other belongs: "does this look generated" and "who is
# speaking here" are different findings with different evidence behind them (rule 11).
ACTIVE_SPEAKER_SIGNAL = "active_speaker"

# Identity of the provenance signal. C2PA is not a provider that can be asked anything —
# the credentials are in the file or they are not — so the "provider" is the standard
# whose data was read, and `provider_version` records the SDK that read it.
C2PA_PROVIDER = "c2pa"
PROVENANCE_SIGNAL = "provenance"

# Identity of the local audio evidence. "aasist" is the checkpoint that produced it, not a
# company that was called: nothing leaves this machine to obtain it, and the exact artifact
# behind a stored signal is recorded in the signal's metadata rather than implied by the
# name. It is a fourth, wholly independent signal — it shares the prepared WAV with active
# speaker and shares nothing else, least of all a conclusion (rule 11).
AASIST_PROVIDER = "aasist"
AUDIO_AUTHENTICITY_SIGNAL = "audio_authenticity"

# How many clip results one detection may leave behind. NVIDIA scores a clip every few
# frames, so a long video produces thousands and persisting all of them would let one
# provider's output grow the database without limit. The strongest evidence is kept and
# the rest is dropped; `total_clips` on the signal records how many there were, so a
# truncated set is never mistaken for the whole one (D019).
MAX_PERSISTED_SEGMENTS = 20

# How many speaking segments one active-speaker result may leave behind, for the same
# reason and with the same consequence as the clip cap above: NVIDIA reports evidence per
# frame, and a conversation aggregates into far more rows than a clip table does.
#
# Higher than the clip cap because the two bound different things, and cut differently.
# Clips are samples of one continuous judgement about the whole video, so the strongest of
# them are representative wherever they fall. Speaking segments are a timeline, and a
# timeline is only readable as a run from the start — so this cap keeps the first fifty in
# chronological order rather than a selection from across the video. Fifty covers ordinary
# multi-speaker footage end to end while still refusing to let one provider's output grow
# the table without limit. Where the timeline stops is never silent:
# `total_speaking_segments` and `segments_truncated` on the signal both say so.
MAX_PERSISTED_SPEAKING_SEGMENTS = 50

# How many AASIST windows one audio reading may leave behind. The model consumes 4.0375 s
# at a time, so an hour of audio is 892 windows and nothing bounds how long an upload is.
#
# Cut the same way the speaking timeline is, and for a stronger reason. These windows are a
# chronological sweep of one recording, and the model publishes no threshold, no class and
# no calibration over its two logits — so there is no defensible sense in which one window
# is "the strongest" evidence, and ranking by logit would be this codebase inventing the
# operating point the checkpoint deliberately does not ship. The first fifty windows are the
# first ~3.4 minutes of audio, in order, and `total_audio_windows` plus `windows_truncated`
# on the signal say exactly where that stops.
MAX_PERSISTED_AUDIO_WINDOWS = 50


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


def undetectable_media(error: Exception, signal_type: str) -> AnalysisSignal:
    """Record that NVIDIA could not be asked, because the artifact could not be prepared.

    NVIDIA takes MP4/H.264 and nothing else, so media in any other shape has to be
    transcoded before there is anything to send. When that transcode fails the provider
    was never reached — but the outcome for the evidence board is the same one a refused
    call produces: this source has no answer about this media, and the gap is recorded
    rather than left silent. Written as a real signal so an analysis that got provenance
    still keeps it, instead of the whole job being thrown away over one source (D016).

    `signal_type` says which of NVIDIA's two questions went unanswered. Both are asked
    about the same prepared artifact, so a transcode that fails costs both of them, and
    each records that separately rather than one standing in for the other.

    A transcode that ran out of time never reaches here at all, since R1-T3.
    `NormalizationTimeout` is no longer a `NormalizationError` and is not caught beside one:
    it says the worker breached its own limit rather than that the media was refused, and it
    fails the job instead of being recorded as evidence about a video nobody looked at.
    """
    return AnalysisSignal(
        provider=NVIDIA_PROVIDER,
        signal_type=signal_type,
        status=SIGNAL_STATUS_FAILED,
        # The failure kind, never the message: it quotes the local artifact's path.
        signal_metadata={"error": type(error).__name__},
    )


def speaker_ids(turns: tuple[SpeakerTurn, ...]) -> dict[str, int]:
    """Assign each diarized voice the integer NVIDIA's proto requires, deterministically.

    pyannote names voices with strings — `SPEAKER_00`, or whatever the pipeline produced —
    and NVIDIA's `AudioSegmentInfo.speaker_id` is a `uint32`. Something has to bridge the
    two, and this is it.

    The number is assigned by order of first appearance in pyannote's chronological turns,
    starting at zero. Nothing is parsed out of the label: `SPEAKER_00` is an opaque name, not
    an index, and a pipeline that labelled a voice `Interviewer` would have nothing to parse
    at all. Reading digits out of a label would also quietly assume pyannote numbers its
    speakers from zero without gaps, which is a property of a naming convention rather than a
    guarantee.

    Deterministic for a given diarization: the same turns always produce the same mapping, so
    the same media re-analysed yields comparable speaker numbering.

    The mapping is the caller's to keep. It is the only thing that can turn NVIDIA's echoed
    integers back into the labels the model actually reported, and the labels are what gets
    persisted — the integer is a wire-format detail this codebase invented, and storing it as
    if it were the speaker's identity would record our own encoding as the provider's finding.
    """
    assigned: dict[str, int] = {}

    for turn in turns:
        if turn.speaker_id not in assigned:
            assigned[turn.speaker_id] = len(assigned)

    return assigned


def diarization_for_nvidia(
    turns: tuple[SpeakerTurn, ...], assigned: dict[str, int]
) -> list[DiarizationSegment]:
    """Restate pyannote's turns in the units NVIDIA's proto carries them in.

    Two conversions and nothing else: seconds to whole milliseconds, and label to the
    assigned integer. Turn boundaries are rounded to the nearest millisecond rather than
    truncated, because truncating would shift every boundary in one direction and NVIDIA
    matches faces to voices by comparing these times against frame timestamps.

    Order is pyannote's own, which is chronological. Overlapping turns stay overlapping:
    two people talking at once is a real observation, and resolving it here would throw away
    exactly the case active-speaker detection is most useful for.
    """
    return [
        DiarizationSegment(
            start_time_ms=round(turn.start_time * 1000),
            end_time_ms=round(turn.end_time * 1000),
            speaker_id=assigned[turn.speaker_id],
        )
        for turn in turns
    ]


@dataclass(frozen=True)
class SpeakingRun:
    """One unbroken stretch of frames in which NVIDIA saw a given face speaking.

    Frames, not seconds: this is still NVIDIA's own unit, and the conversion happens once,
    afterwards, against the rate of the video NVIDIA was actually given. Both bounds are
    inclusive — `first_frame == last_frame` is a single frame, not an empty range.
    """

    face_id: int
    diarized_speaker_id: int
    first_frame: int
    last_frame: int

    @property
    def frame_count(self) -> int:
        return self.last_frame - self.first_frame + 1


def speaking_runs(result: NvidiaActiveSpeakerResult) -> list[SpeakingRun]:
    """Collapse NVIDIA's per-frame verdicts into unbroken runs, per face and voice.

    NVIDIA answers per frame: for each tracked face in that frame, whether it is speaking.
    A run is opened when a face starts speaking and closed as soon as that stops being
    reported — either because the face is no longer speaking, because it was not reported in
    that frame at all, or because the frame numbering skipped, which means the frames either
    side of the gap are not adjacent and joining them would assert speech across footage
    NVIDIA said nothing about.

    Runs are keyed by face *and* by the voice NVIDIA matched it to, so one face reassigned to
    a different diarized speaker ends one run and begins another rather than extending a
    range whose attribution silently changed halfway.

    `is_speaking` is the only field consulted. NVIDIA has already decided it against its own
    threshold, and `face_detection_confidence` — confidence in having found a face — is not a
    second opinion about speech and is deliberately not weighed in here.
    """
    open_runs: dict[tuple[int, int], list[int]] = {}
    closed: list[SpeakingRun] = []

    def close(key: tuple[int, int]) -> None:
        first, last = open_runs.pop(key)
        closed.append(
            SpeakingRun(
                face_id=key[0], diarized_speaker_id=key[1], first_frame=first, last_frame=last
            )
        )

    for frame in result.frames:
        # Sorted so that the order runs are opened in follows from the evidence rather than
        # from set iteration order, which would vary between runs of the same input.
        speaking = sorted(
            {
                (speaker.face_id, speaker.diarized_speaker_id)
                for speaker in frame.speakers
                if speaker.is_speaking
            }
        )

        for key in list(open_runs):
            if key not in speaking or open_runs[key][1] != frame.frame_id - 1:
                close(key)

        for key in speaking:
            if key in open_runs:
                open_runs[key][1] = frame.frame_id
            else:
                open_runs[key] = [frame.frame_id, frame.frame_id]

    # Whatever is still speaking when the video ends is real evidence and ends with it.
    for key in list(open_runs):
        close(key)

    return closed


def speaking_segments(
    result: NvidiaActiveSpeakerResult,
    frame_rate: float,
    labels_by_id: dict[int, str],
) -> tuple[list[AnalysisSegment], int]:
    """Turn per-frame evidence into persistable time ranges, capped, chronological.

    `frame_rate` must be the rate of the video NVIDIA was given, because NVIDIA counts the
    frames of that video and nothing else. The worker hands over the rate the artifact was
    transcoded to hold constant, or — for media that needed no transcode — the probed rate of
    the original it sent as-is. It is positive by construction: media whose rate cannot be
    read as a positive number is rejected at upload and never reaches a job.

    A run of frames `first..last` becomes `first / rate` to `(last + 1) / rate`. The end
    bound counts the last frame's own duration, so a single speaking frame is a range one
    frame long rather than an instant of zero length.

    `labels_by_id` turns NVIDIA's echoed integer back into pyannote's own label, which is what
    is stored. A face NVIDIA matched to no voice comes back as -1 and is kept, with a null
    label: "this face was speaking and no diarized voice matched it" is an observation, and
    dropping those rows would quietly delete evidence.

    Returns the rows to persist and how many runs there were in total. Ordering is
    chronological and the cap keeps the first of them, so what is persisted is the timeline
    from the beginning of the video up to wherever the cap fell.

    Keeping the *first* rather than the longest is deliberate. A timeline read back has to be
    contiguous to mean anything: an hour of footage reduced to its fifty longest utterances
    would show gaps that look like silence but are really evidence that was dropped, and no
    reader could tell the two apart. A chronological prefix has one boundary instead, and
    `segments_truncated` on the signal names it.

    Length is also not this layer's to rank by. Ordering evidence by how much of it there is
    puts a "more speech matters more" judgement into what is supposed to be a copy of what
    NVIDIA reported, and nothing here is entitled to make it. Chronological order is the
    order the provider observed things in, and it is the only order this function imposes.
    """
    runs = speaking_runs(result)

    # Chronological, then by face and by the voice matched to it. The tie-breaks are not a
    # preference: two faces can begin speaking on the very same frame, and without a total
    # order the same result could truncate to different rows on different runs.
    ordered = sorted(
        runs, key=lambda run: (run.first_frame, run.face_id, run.diarized_speaker_id)
    )

    segments = [
        AnalysisSegment(
            start_time=run.first_frame / frame_rate,
            end_time=(run.last_frame + 1) / frame_rate,
            face_id=run.face_id,
            speaker_label=labels_by_id.get(run.diarized_speaker_id),
        )
        for run in ordered[:MAX_PERSISTED_SPEAKING_SEGMENTS]
    ]

    return segments, len(runs)


async def detect_active_speaker(
    video_path: Path, frame_rate: float, audio_path: Path
) -> tuple[AnalysisSignal, list[AnalysisSegment]]:
    """Run the whole active-speaker chain over one prepared artifact and record the outcome.

    `audio_path` is the WAV `prepared_audio` extracted from `video_path`, and it is used
    twice here: pyannote diarizes it, and the same file is streamed to NVIDIA as the separate
    audio track it wants beside the video. It belongs to the caller — since P6-T3 a third
    reader shares it — so this function only reads it and never removes it. Its coming from
    the artifact NVIDIA is given, rather than from the forensic original, is what keeps the
    two timelines the same one: diarization times are matched against the frames of this
    video, so audio from a differently-timed file would misattribute voices to faces.

    Nothing here raises for a failure of the chain itself. Diarization and NVIDIA are both
    recorded as a `FAILED` active-speaker signal, so the analysis keeps the provenance and
    synthetic-video evidence it already has. That is deliberately more forgiving than
    `detect_synthetic_video`, which lets a local file error fail the whole job: this chain
    depends on a gated model, a Hugging Face token and a second NVIDIA function, and an
    analysis must not lose its other evidence sources because one of those is not configured.

    Media with no speech in it takes the same route. pyannote returns no turns, there is
    nothing to hand NVIDIA — which requires diarization and reports every face as unmatched
    without it — and the resulting input error is recorded as the failure to produce this
    signal that it is, rather than as a fabricated empty success.

    `score` stays null on success as well as failure. Active speaker produces a timeline, not
    a figure on a scale, and a number here would sit in the same column as NVIDIA's synthetic
    probability as though the two could be compared (rule 11).
    """
    signal = AnalysisSignal(provider=NVIDIA_PROVIDER, signal_type=ACTIVE_SPEAKER_SIGNAL)

    try:
        turns = await diarize_speakers(audio_path)
        assigned = speaker_ids(turns)
        result = await analyze_active_speaker(
            video_path,
            diarization_for_nvidia(turns, assigned),
            audio_path=audio_path,
        )
    except NvidiaActiveSpeakerTimeout as error:
        # The one failure worth telling apart at a glance, and for the same reason as in
        # synthetic-video detection: NVIDIA may still have been working, so this says
        # nothing about the media either way.
        logger.warning("NVIDIA active-speaker detection timed out.", exc_info=True)
        signal.status = SIGNAL_STATUS_TIMEOUT
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []
    except (SpeakerDiarizationError, NvidiaActiveSpeakerError) as error:
        # Everything else this chain can fail with: no audio in the media, an unavailable
        # diarizer, a model that broke mid-inference, rejected NVIDIA credentials, an
        # unreachable service. One status, and the failure kind in the metadata to tell them
        # apart — never the message, which can quote the local artifact's path.
        logger.warning("NVIDIA active-speaker detection failed.", exc_info=True)
        signal.status = SIGNAL_STATUS_FAILED
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []

    segments, total = speaking_segments(
        result, frame_rate, {number: label for label, number in assigned.items()}
    )

    signal.status = SIGNAL_STATUS_SUCCESS
    signal.provider_version = result.function_id
    signal.signal_metadata = {
        # The rate the frame indices were read against. Without it the persisted times
        # cannot be checked back against NVIDIA's own frame numbering.
        "frame_rate": frame_rate,
        "total_frames": len(result.frames),
        # What the aggregation produced, against what survived the cap — so a truncated
        # timeline is never mistaken for the whole one.
        "total_speaking_segments": total,
        "segments_truncated": total > len(segments),
        # NVIDIA's own threshold, echoed back on its configuration: the value every
        # `is_speaking` behind these segments was decided against. Null when it echoed none.
        "speaker_detection_threshold": result.speaker_detection_threshold,
        # The encoding this codebase assigned pyannote's labels, kept so NVIDIA's raw
        # integers stay readable against the evidence it returned.
        "diarized_speakers": assigned,
    }

    return signal, segments


def audio_windows(
    evidence: AudioAuthenticityEvidence,
) -> tuple[list[AnalysisSegment], int]:
    """Turn AASIST's per-window output into persistable rows, capped, chronological.

    One row per window, in the order the recording was cut, carrying both raw logits in
    graph order and the sample bounds that produced them restated in seconds. The division
    by the sample rate is exact arithmetic on numbers this codebase chose — the window
    boundaries are DeepGuard's own preprocessing — and is emphatically not a timestamp the
    model published: AASIST reports no segments and no chunk-to-time mapping at all, so a
    row here says "this is the audio the graph was fed", never "the model found something
    between these two times".

    Nothing is aggregated, averaged, ranked or thresholded, and no window stands in for
    another. P6-T1 §7 measured consecutive windows of the same genuine recording crossing
    zero in both directions; the rows disagree with each other and are stored as they are.

    Returns the rows to persist and how many windows there were in total. The cap keeps the
    chronological prefix, so what is stored is the sweep from the start of the audio up to
    wherever it fell — never a selection by logit, which would smuggle in the operating
    point the checkpoint does not define.
    """
    segments = [
        AnalysisSegment(
            clip_index=window.window_index,
            # Graph order, untouched. `bona_fide_logit` is output column 1, which is the
            # column upstream reads as the bona fide score; `logit` is column 0. Both are
            # raw logits, and neither is a probability or a distance from a threshold.
            logit=window.logits[0],
            bona_fide_logit=window.logits[BONA_FIDE_LOGIT_INDEX],
            start_time=window.start_sample / evidence.sample_rate,
            end_time=window.end_sample / evidence.sample_rate,
        )
        for window in evidence.windows[:MAX_PERSISTED_AUDIO_WINDOWS]
    ]

    return segments, len(evidence.windows)


def detect_audio_authenticity(audio_path: Path) -> tuple[AnalysisSignal, list[AnalysisSegment]]:
    """Run the local checkpoint over prepared audio and record what it emitted.

    `audio_path` is the WAV `prepared_audio` extracted for this job, shared with pyannote and
    NVIDIA and owned by neither of them nor by this function. Extracting the audio a second
    time would decode the same media twice to produce the same file, and — worse — would make
    it possible for the two extractions to disagree about what was analysed.

    `score` stays null, on success as much as on failure. The model emits two raw logits per
    window with no softmax, threshold or class over them, so there is no file-level figure to
    put in that column; anything written there would be this codebase inventing a number and
    parking it beside NVIDIA's probability as though the two were comparable (rule 11). The
    evidence is the windows.

    Nothing raises. Every `AudioDetectorError` — a missing checkpoint, audio in the wrong
    shape, a decode failure, ONNX Runtime breaking — becomes a `FAILED` signal carrying the
    failure kind and nothing else, so one source going wrong costs this signal and no other.

    Blocking CPU work, so callers on an event loop reach it through `analyse_audio`.
    """
    signal = AnalysisSignal(provider=AASIST_PROVIDER, signal_type=AUDIO_AUTHENTICITY_SIGNAL)

    try:
        evidence = analyze_audio_authenticity(audio_path)
    except AudioDetectorError as error:
        # The failure kind, never the message: it can quote the local artifact's path.
        logger.warning("Local audio authenticity detection failed.", exc_info=True)
        signal.status = SIGNAL_STATUS_FAILED
        signal.signal_metadata = {"error": type(error).__name__}
        return signal, []

    segments, total = audio_windows(evidence)

    signal.status = SIGNAL_STATUS_SUCCESS
    # Which artifact produced this, exactly. A different revision is a different measurement,
    # and the digest is what lets a stored signal be checked against the model that made it.
    signal.provider_version = f"{evidence.model_repository}@{evidence.model_revision}"
    signal.signal_metadata = {
        "model_repository": evidence.model_repository,
        "model_revision": evidence.model_revision,
        "model_sha256": evidence.model_sha256,
        # The input contract every window above was produced under. The same audio cut at a
        # different offset, rate or window length is a different number (P6-T1 §7), so the
        # preparation is recorded alongside the output rather than assumed from the code.
        "sample_rate": evidence.sample_rate,
        "channels": evidence.channels,
        "window_samples": evidence.window_samples,
        "window_padding_scheme": evidence.window_padding_scheme,
        "total_samples": evidence.total_samples,
        # What the sweep produced, against what survived the cap — so a truncated set of
        # windows is never mistaken for the whole recording.
        "total_audio_windows": total,
        "persisted_audio_windows": len(segments),
        "windows_truncated": total > len(segments),
        # What the segment rows' times are. Stated in the evidence itself so a reader of the
        # database never has to assume: these are the bounds of the windows this codebase cut
        # and fed to the graph, not intervals the model reported anything about.
        "window_bounds": "deepguard_preprocessing",
        # Which of the two stored logits carries the meaning the checkpoint's own repository
        # gives it. Recorded rather than left to the column name, because the mapping is
        # upstream's fact about its model and not something this codebase decided.
        "bona_fide_logit_index": BONA_FIDE_LOGIT_INDEX,
    }

    return signal, segments


def unanalysable_audio(error: Exception) -> AnalysisSignal:
    """Record that the checkpoint could not be run, because no audio could be prepared.

    The counterpart of `undetectable_media` for the audio side, and separate from it because
    the two name different providers and are reached by different failures. Written as a real
    signal so an analysis that has provenance and a synthetic-video verdict keeps them when
    the media turns out to carry no audio at all — which is an ordinary property of an upload,
    not a fault.
    """
    return AnalysisSignal(
        provider=AASIST_PROVIDER,
        signal_type=AUDIO_AUTHENTICITY_SIGNAL,
        status=SIGNAL_STATUS_FAILED,
        # The failure kind, never the message: it can quote the local artifact's path.
        signal_metadata={"error": type(error).__name__},
    )


async def analyse_audio(
    video_path: Path, frame_rate: float
) -> tuple[
    tuple[AnalysisSignal, list[AnalysisSegment]],
    tuple[AnalysisSignal, list[AnalysisSegment]],
]:
    """Prepare the audio once, ask both audio questions of it, report both outcomes.

    Extraction is the shared step and the only one: `prepared_audio` writes a single mono
    16 kHz PCM WAV from the artifact NVIDIA is given, and pyannote, the Active Speaker NIM
    and AASIST all read that same file. Extracting it per consumer would decode the same
    media three times to produce three identical files, and would leave open the possibility
    of three readings of what was supposed to be one recording.

    Sharing the input is the whole of what these two signals share. They are asked different
    questions, answer in different units, are written as separate rows with separate statuses,
    and are never combined — an analysis that has a speaker timeline and no audio windows, or
    the reverse, records exactly that.

    So the isolation runs in both directions, and neither call can reach the other: active
    speaker records its own failures internally, and AASIST is run afterwards regardless of
    what came back. A missing Hugging Face token costs the timeline and leaves the windows; a
    missing checkpoint costs the windows and leaves the timeline.

    Audio that cannot be prepared at all is the one failure both share, because both were
    waiting on the same file. Media with no audio stream is the common case and is an
    ordinary fact about an upload, so it is recorded as two failed signals rather than a
    failed job — the provenance and synthetic-video evidence this analysis already has must
    survive it.

    Extraction that *outlived its limit* is the exception to that, since R1-T3.
    `SpeakerDiarizationTimeout` is no longer a `SpeakerDiarizationError` and so is not caught
    below: it describes this worker's condition rather than the upload's audio, and it
    propagates to fail the job. `app.speaker_diarization` states the trade that makes.

    AASIST is blocking CPU work, so it goes to a thread the way diarization does; running it
    on the event loop would stall the ffmpeg and gRPC I/O this coroutine is built around.
    """
    try:
        async with prepared_audio(video_path) as audio_path:
            speaker = await detect_active_speaker(video_path, frame_rate, audio_path)
            audio = await asyncio.to_thread(detect_audio_authenticity, audio_path)

            return speaker, audio
    except SpeakerDiarizationError as error:
        logger.warning("Preparing the audio for analysis failed.", exc_info=True)
        signal = AnalysisSignal(provider=NVIDIA_PROVIDER, signal_type=ACTIVE_SPEAKER_SIGNAL)
        signal.status = SIGNAL_STATUS_FAILED
        signal.signal_metadata = {"error": type(error).__name__}

        return (signal, []), (unanalysable_audio(error), [])


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
