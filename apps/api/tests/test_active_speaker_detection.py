"""The active-speaker chain, exercised directly against a local artifact.

`detect_active_speaker` is the third evidence source the worker runs, and the only one made
of three steps: extract the audio, diarize it locally, then ask NVIDIA which visible face
each diarized voice belongs to. All three boundaries are faked here — no ffmpeg, no pyannote,
no gRPC — because what is under test is this module's own work: the integers it assigns, the
frames it collapses into time ranges, the failures it converts into a signal, and the audio
it must not extract twice.

Persistence is not here. Nothing in this module opens a transaction or writes a row.
"""

import asyncio
from pathlib import Path

import pytest

from app import detection, nvidia_active_speaker, speaker_diarization
from app.detection import (
    ACTIVE_SPEAKER_SIGNAL,
    MAX_PERSISTED_SPEAKING_SEGMENTS,
    detect_active_speaker,
    diarization_for_nvidia,
    speaker_ids,
    speaking_runs,
    speaking_segments,
)
from app.nvidia_active_speaker import (
    DiarizationSegment,
    NvidiaActiveSpeakerFrame,
    NvidiaActiveSpeakerResult,
    NvidiaBoundingBox,
    NvidiaSpeakerObservation,
)
from app.speaker_diarization import SpeakerTurn

VIDEO = Path("/tmp/deepguard-normalized-abc.mp4")
ASD_FUNCTION_ID = "f286f937-05c4-454b-8312-fba67a2a6fa7"

# A plain 25 fps rate, so a frame is exactly 40 ms and expected times stay readable.
FRAME_RATE = 25.0


def observation(face_id, speaker_id, *, speaking, confidence=0.99):
    """One face in one frame, as NVIDIA reports it."""
    return NvidiaSpeakerObservation(
        face_id=face_id,
        diarized_speaker_id=speaker_id,
        is_speaking=speaking,
        face_detection_confidence=confidence,
        bounding_box=NvidiaBoundingBox(x=10.0, y=20.0, width=64.0, height=64.0),
    )


def frames(*specs):
    """Build a frame sequence from `(frame_id, [observation, ...])` pairs."""
    return tuple(
        NvidiaActiveSpeakerFrame(frame_id=frame_id, speakers=tuple(speakers))
        for frame_id, speakers in specs
    )


def result(*specs, threshold=0.5):
    """An NVIDIA result carrying the given frames."""
    return NvidiaActiveSpeakerResult(
        frames=frames(*specs),
        function_id=ASD_FUNCTION_ID,
        speaker_detection_threshold=threshold,
    )


def speaking_frames(face_id, speaker_id, frame_ids):
    """Frames in which exactly one face speaks, for the listed frame ids."""
    return tuple((frame_id, [observation(face_id, speaker_id, speaking=True)]) for frame_id in frame_ids)


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch, tmp_path):
    """Stand in for ffmpeg: hands back a real path without decoding anything.

    Autouse, so no test in this module can spawn a subprocess. The file is real because
    both consumers are handed the path and a test asserts they got the same one.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.sources = []
            self.path = tmp_path / "prepared.wav"
            self.removed = False

        async def extract(self, source, destination):
            self.sources.append(Path(source))
            if self.error:
                raise self.error
            Path(destination).write_bytes(b"RIFF....WAVE")

    recorder = Recorder()
    monkeypatch.setattr(speaker_diarization, "_extract_audio", recorder.extract)
    return recorder


@pytest.fixture(autouse=True)
def fake_diarization(monkeypatch):
    """Stand in for pyannote, so no test here loads torch or reaches Hugging Face."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.audio_paths = []
            self.turns = (
                SpeakerTurn(start_time=0.0, end_time=2.0, speaker_id="SPEAKER_00"),
                SpeakerTurn(start_time=2.0, end_time=4.0, speaker_id="SPEAKER_01"),
            )

        async def diarize(self, audio_path, **kwargs):
            self.audio_paths.append(Path(audio_path))
            if self.error:
                raise self.error
            return self.turns

    recorder = Recorder()
    monkeypatch.setattr(detection, "diarize_speakers", recorder.diarize)
    return recorder


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """Stand in for the ASD NIM, so no test in this module can reach NVIDIA."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.calls = []
            self.result = result(
                *speaking_frames(0, 0, range(0, 50)),
                *speaking_frames(1, 1, range(50, 100)),
            )

        async def analyze(self, video_path, diarization, *, audio_path=None, **kwargs):
            self.calls.append((Path(video_path), list(diarization), audio_path))
            # The real client refuses unusable diarization before opening a channel, and a
            # stand-in that accepted it would let this module's handling of that go
            # untested. Borrowed rather than restated so the two cannot drift apart.
            nvidia_active_speaker._validate_diarization(diarization)
            if self.error:
                raise self.error
            return self.result

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_active_speaker", recorder.analyze)
    return recorder


def run(video=VIDEO, frame_rate=FRAME_RATE):
    return asyncio.run(detect_active_speaker(video, frame_rate))


# --- mapping pyannote's labels onto NVIDIA's integers -------------------------------------


def test_labels_are_numbered_by_first_appearance():
    turns = (
        SpeakerTurn(start_time=0.0, end_time=1.0, speaker_id="SPEAKER_03"),
        SpeakerTurn(start_time=1.0, end_time=2.0, speaker_id="SPEAKER_01"),
        SpeakerTurn(start_time=2.0, end_time=3.0, speaker_id="SPEAKER_03"),
    )

    # Numbered by when the voice was first heard, never by what the label says: the first
    # speaker here is `SPEAKER_03` and it is speaker 0.
    assert speaker_ids(turns) == {"SPEAKER_03": 0, "SPEAKER_01": 1}


def test_nothing_is_parsed_out_of_the_label():
    """A pipeline that names voices rather than numbering them still maps cleanly."""
    turns = (
        SpeakerTurn(start_time=0.0, end_time=1.0, speaker_id="Interviewer"),
        SpeakerTurn(start_time=1.0, end_time=2.0, speaker_id="Guest"),
    )

    assert speaker_ids(turns) == {"Interviewer": 0, "Guest": 1}


def test_the_mapping_is_deterministic():
    """The same diarization always numbers the same way, so re-analysis is comparable."""
    turns = (
        SpeakerTurn(start_time=0.0, end_time=1.0, speaker_id="SPEAKER_01"),
        SpeakerTurn(start_time=1.0, end_time=2.0, speaker_id="SPEAKER_00"),
    )

    assert speaker_ids(turns) == speaker_ids(turns)


def test_turns_are_restated_in_milliseconds_and_integers():
    turns = (
        SpeakerTurn(start_time=0.4978125, end_time=3.1246875, speaker_id="SPEAKER_00"),
        SpeakerTurn(start_time=3.6084375, end_time=7.9059375, speaker_id="SPEAKER_01"),
    )

    segments = diarization_for_nvidia(turns, speaker_ids(turns))

    # Rounded to the nearest millisecond rather than truncated: truncating would shift
    # every boundary the same way, against frame timestamps NVIDIA matches them to.
    assert segments == [
        DiarizationSegment(start_time_ms=498, end_time_ms=3125, speaker_id=0),
        DiarizationSegment(start_time_ms=3608, end_time_ms=7906, speaker_id=1),
    ]


def test_overlapping_turns_are_handed_over_as_they_are():
    """Two voices at once is the case this detector is most useful for."""
    turns = (
        SpeakerTurn(start_time=1.0, end_time=4.0, speaker_id="SPEAKER_00"),
        SpeakerTurn(start_time=3.5, end_time=6.0, speaker_id="SPEAKER_01"),
    )

    segments = diarization_for_nvidia(turns, speaker_ids(turns))

    assert len(segments) == 2
    assert segments[0].end_time_ms > segments[1].start_time_ms


# --- aggregating frames into time ranges --------------------------------------------------


def test_contiguous_speaking_frames_become_one_range():
    segments, total = speaking_segments(
        result(*speaking_frames(0, 0, range(0, 10))), FRAME_RATE, {0: "SPEAKER_00"}
    )

    assert total == 1
    assert len(segments) == 1
    # Frames 0..9 at 25 fps: starts at the first frame, ends after the last one's own
    # duration rather than at the instant it began.
    assert segments[0].start_time == pytest.approx(0.0)
    assert segments[0].end_time == pytest.approx(0.4)


def test_a_gap_in_speaking_splits_the_range():
    segments, total = speaking_segments(
        result(
            (0, [observation(0, 0, speaking=True)]),
            (1, [observation(0, 0, speaking=True)]),
            (2, [observation(0, 0, speaking=False)]),
            (3, [observation(0, 0, speaking=True)]),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00"},
    )

    assert total == 2
    assert [(s.start_time, s.end_time) for s in segments] == [
        pytest.approx((0.0, 0.08)),
        pytest.approx((0.12, 0.16)),
    ]


def test_a_frame_the_face_is_absent_from_ends_the_range():
    """Not reported is not the same as reported silent, and neither continues a range."""
    segments, total = speaking_segments(
        result(
            (0, [observation(0, 0, speaking=True)]),
            (1, []),
            (2, [observation(0, 0, speaking=True)]),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00"},
    )

    assert total == 2


def test_a_skipped_frame_number_does_not_join_two_ranges():
    """Frames either side of a gap are not adjacent; joining them would assert speech
    across footage NVIDIA said nothing about."""
    segments, total = speaking_segments(
        result(
            (10, [observation(0, 0, speaking=True)]),
            (11, [observation(0, 0, speaking=True)]),
            (40, [observation(0, 0, speaking=True)]),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00"},
    )

    assert total == 2
    assert [(s.start_time, s.end_time) for s in segments] == [
        pytest.approx((0.4, 0.48)),
        pytest.approx((1.6, 1.64)),
    ]


def test_two_faces_speaking_at_once_are_two_ranges():
    segments, total = speaking_segments(
        result(
            (0, [observation(0, 0, speaking=True), observation(1, 1, speaking=True)]),
            (1, [observation(0, 0, speaking=True), observation(1, 1, speaking=True)]),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00", 1: "SPEAKER_01"},
    )

    assert total == 2
    assert {s.face_id for s in segments} == {0, 1}
    assert {s.speaker_label for s in segments} == {"SPEAKER_00", "SPEAKER_01"}


def test_a_face_reassigned_to_another_voice_starts_a_new_range():
    """The attribution changed halfway, so extending one range would misreport it."""
    segments, total = speaking_segments(
        result(
            (0, [observation(0, 0, speaking=True)]),
            (1, [observation(0, 1, speaking=True)]),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00", 1: "SPEAKER_01"},
    )

    assert total == 2
    assert [(s.face_id, s.speaker_label) for s in segments] == [
        (0, "SPEAKER_00"),
        (0, "SPEAKER_01"),
    ]


def test_a_single_speaking_frame_is_a_range_one_frame_long():
    segments, _ = speaking_segments(
        result((7, [observation(0, 0, speaking=True)])), FRAME_RATE, {0: "SPEAKER_00"}
    )

    assert segments[0].start_time == pytest.approx(0.28)
    assert segments[0].end_time == pytest.approx(0.32)
    assert segments[0].end_time > segments[0].start_time


def test_speech_still_running_at_the_end_of_the_video_is_kept():
    segments, total = speaking_segments(
        result(*speaking_frames(0, 0, range(0, 5))), FRAME_RATE, {0: "SPEAKER_00"}
    )

    assert total == 1
    assert segments[0].end_time == pytest.approx(0.2)


def test_frames_with_nobody_speaking_produce_nothing():
    """Silence is not an event; a range covering it would assert speech that was not there."""
    segments, total = speaking_segments(
        result(
            (0, [observation(0, 0, speaking=False)]),
            (1, []),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00"},
    )

    assert (segments, total) == ([], 0)


def test_the_actual_frame_rate_decides_the_times():
    """The same frames at a different rate are a different length of video."""
    at_25, _ = speaking_segments(
        result(*speaking_frames(0, 0, range(0, 25))), 25.0, {0: "SPEAKER_00"}
    )
    at_50, _ = speaking_segments(
        result(*speaking_frames(0, 0, range(0, 25))), 50.0, {0: "SPEAKER_00"}
    )

    assert at_25[0].end_time == pytest.approx(1.0)
    assert at_50[0].end_time == pytest.approx(0.5)


def test_a_fractional_frame_rate_is_used_as_reported():
    """29.97 fps is what NTSC footage really is; rounding it to 30 would drift the times."""
    segments, _ = speaking_segments(
        result(*speaking_frames(0, 0, range(0, 30))), 30000 / 1001, {0: "SPEAKER_00"}
    )

    assert segments[0].end_time == pytest.approx(30 * 1001 / 30000)


def test_an_unmatched_face_is_kept_with_no_label():
    """NVIDIA matched this face to no diarized voice, which is an observation, not a gap."""
    segments, total = speaking_segments(
        result(*speaking_frames(3, -1, range(0, 5))), FRAME_RATE, {0: "SPEAKER_00"}
    )

    assert total == 1
    assert segments[0].face_id == 3
    assert segments[0].speaker_label is None


def test_segments_carry_no_clip_index_or_logit():
    """Clip columns belong to the other NVIDIA signal; filling them here would invent
    provider output that does not exist in an active-speaker result."""
    segments, _ = speaking_segments(
        result(*speaking_frames(0, 0, range(0, 5))), FRAME_RATE, {0: "SPEAKER_00"}
    )

    assert segments[0].clip_index is None
    assert segments[0].logit is None


def test_face_detection_confidence_never_becomes_a_speaking_figure():
    """It is confidence in having found a face, not in that face speaking."""
    segments, _ = speaking_segments(
        result(
            (0, [observation(0, 0, speaking=True, confidence=0.42)]),
            (1, [observation(0, 0, speaking=True, confidence=0.97)]),
        ),
        FRAME_RATE,
        {0: "SPEAKER_00"},
    )

    assert len(segments) == 1
    assert segments[0].logit is None
    # Nothing on the row carries the figure, under any name.
    assert 0.42 not in vars(segments[0]).values()
    assert 0.97 not in vars(segments[0]).values()


def test_a_low_confidence_face_is_still_aggregated():
    """Only `is_speaking` is consulted, and NVIDIA already decided it against its own
    threshold. Second-guessing that here would be this layer forming a verdict."""
    segments, total = speaking_segments(
        result(*[(i, [observation(0, 0, speaking=True, confidence=0.05)]) for i in range(5)]),
        FRAME_RATE,
        {0: "SPEAKER_00"},
    )

    assert total == 1
    assert len(segments) == 1


def test_runs_are_reported_in_frames_not_seconds():
    """The intermediate stays in NVIDIA's own unit; the conversion happens once, after."""
    runs = speaking_runs(result(*speaking_frames(2, 0, range(4, 9))))

    assert len(runs) == 1
    assert (runs[0].first_frame, runs[0].last_frame) == (4, 8)
    assert runs[0].frame_count == 5


# --- bounding what gets persisted ---------------------------------------------------------


def test_evidence_is_capped_and_the_total_is_reported():
    # Every other frame speaking, so each speaking frame is its own one-frame run.
    specs = [
        (i, [observation(0, 0, speaking=i % 2 == 0)])
        for i in range(MAX_PERSISTED_SPEAKING_SEGMENTS * 4)
    ]

    segments, total = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})

    assert total == MAX_PERSISTED_SPEAKING_SEGMENTS * 2
    assert len(segments) == MAX_PERSISTED_SPEAKING_SEGMENTS
    assert total > len(segments)


def test_the_cap_keeps_the_first_segments_chronologically():
    """The timeline is kept as a run from the start, not sampled from across the video.

    A set picked from anywhere would read back with gaps that look like silence but are
    really dropped evidence, and nothing would let a reader tell the two apart.
    """
    specs = []
    for run_index in range(MAX_PERSISTED_SPEAKING_SEGMENTS + 10):
        # Two speaking frames, then a silent one, so every run is its own range.
        frame = run_index * 3
        specs.append((frame, [observation(0, 0, speaking=True)]))
        specs.append((frame + 1, [observation(0, 0, speaking=True)]))
        specs.append((frame + 2, [observation(0, 0, speaking=False)]))

    segments, total = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})

    assert total == MAX_PERSISTED_SPEAKING_SEGMENTS + 10
    assert len(segments) == MAX_PERSISTED_SPEAKING_SEGMENTS
    # Exactly the first fifty runs, in order, with nothing skipped between them.
    assert [s.start_time for s in segments] == [
        pytest.approx(index * 3 / FRAME_RATE)
        for index in range(MAX_PERSISTED_SPEAKING_SEGMENTS)
    ]


def test_a_long_late_segment_does_not_displace_earlier_ones():
    """Duration decides nothing. Ranking by it would put a "more speech matters more"
    judgement into what is meant to be a copy of what NVIDIA reported."""
    specs = []
    for run_index in range(MAX_PERSISTED_SPEAKING_SEGMENTS + 1):
        frame = run_index * 3
        specs.append((frame, [observation(0, 0, speaking=True)]))
        specs.append((frame + 1, [observation(0, 0, speaking=True)]))
        specs.append((frame + 2, [observation(0, 0, speaking=False)]))

    # One final run far longer than every other, right at the end of the video.
    last = (MAX_PERSISTED_SPEAKING_SEGMENTS + 1) * 3
    for offset in range(500):
        specs.append((last + offset, [observation(0, 0, speaking=True)]))

    segments, total = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})

    assert total == MAX_PERSISTED_SPEAKING_SEGMENTS + 2
    assert len(segments) == MAX_PERSISTED_SPEAKING_SEGMENTS
    # The long one is last in time, so it is dropped like any other overflow — it does not
    # push the opening of the conversation out of the record.
    longest = max(s.end_time - s.start_time for s in segments)
    assert longest == pytest.approx(2 / FRAME_RATE)
    assert segments[0].start_time == pytest.approx(0.0)


def test_evidence_under_the_cap_is_kept_whole():
    """Nothing is dropped when there is room for all of it."""
    specs = []
    for run_index in range(5):
        frame = run_index * 3
        specs.append((frame, [observation(0, 0, speaking=True)]))
        specs.append((frame + 1, [observation(0, 0, speaking=False)]))

    segments, total = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})

    assert total == 5
    assert len(segments) == 5


def test_persisted_segments_are_chronological():
    """It is a timeline; reading it back should not require re-sorting it."""
    specs = []
    for frame in range(0, 40):
        specs.append((frame, [observation(0, 0, speaking=frame % 3 != 0)]))

    segments, _ = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})

    assert [s.start_time for s in segments] == sorted(s.start_time for s in segments)


def test_faces_starting_on_the_same_frame_are_ordered_deterministically():
    """Chronological order alone is not a total order: two people can start together."""
    specs = [
        (0, [observation(2, 1, speaking=True), observation(1, 0, speaking=True)]),
        (1, [observation(2, 1, speaking=True), observation(1, 0, speaking=True)]),
    ]

    segments, _ = speaking_segments(
        result(*specs), FRAME_RATE, {0: "SPEAKER_00", 1: "SPEAKER_01"}
    )

    # Same start, so the face breaks the tie — and it breaks it the same way every run.
    assert [s.face_id for s in segments] == [1, 2]


def test_truncation_is_deterministic():
    specs = [
        (i, [observation(i % 3, 0, speaking=i % 2 == 0)])
        for i in range(MAX_PERSISTED_SPEAKING_SEGMENTS * 6)
    ]

    first, _ = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})
    second, _ = speaking_segments(result(*specs), FRAME_RATE, {0: "SPEAKER_00"})

    assert [(s.start_time, s.face_id) for s in first] == [
        (s.start_time, s.face_id) for s in second
    ]


# --- the chain end to end -----------------------------------------------------------------


def test_a_successful_run_is_an_independent_nvidia_signal():
    signal, segments = run()

    assert signal.provider == "nvidia"
    assert signal.signal_type == ACTIVE_SPEAKER_SIGNAL
    assert signal.status == "SUCCESS"
    # A timeline, not a figure on a scale. A number here would sit in the same column as
    # NVIDIA's synthetic probability as though the two could be compared.
    assert signal.score is None
    assert signal.risk_level is None
    assert signal.provider_version == ASD_FUNCTION_ID
    assert segments


def test_the_signal_records_how_the_frames_were_read():
    signal, segments = run()

    assert signal.signal_metadata["frame_rate"] == FRAME_RATE
    assert signal.signal_metadata["total_frames"] == 100
    assert signal.signal_metadata["speaker_detection_threshold"] == 0.5
    # The encoding this codebase assigned, kept so NVIDIA's raw integers stay readable.
    assert signal.signal_metadata["diarized_speakers"] == {"SPEAKER_00": 0, "SPEAKER_01": 1}
    assert signal.signal_metadata["total_speaking_segments"] == len(segments)
    assert signal.signal_metadata["segments_truncated"] is False


def test_truncated_evidence_says_so_on_the_signal(fake_nvidia):
    fake_nvidia.result = result(
        *[
            (i, [observation(0, 0, speaking=i % 2 == 0)])
            for i in range(MAX_PERSISTED_SPEAKING_SEGMENTS * 4)
        ]
    )

    signal, segments = run()

    # Dropped evidence is recorded rather than silently hidden.
    assert signal.signal_metadata["segments_truncated"] is True
    assert signal.signal_metadata["total_speaking_segments"] > len(segments)


def test_the_real_speaker_labels_reach_the_segments():
    _, segments = run()

    # pyannote's own strings, not the integers this codebase invented for NVIDIA's wire
    # format: the label is what the model reported, the integer is our encoding of it.
    assert {s.speaker_label for s in segments} == {"SPEAKER_00", "SPEAKER_01"}


def test_nvidia_is_given_the_video_and_the_prepared_audio(fake_nvidia, fake_audio):
    run()

    video_path, diarization, audio_path = fake_nvidia.calls[0]

    assert video_path == VIDEO
    # As a separate stream rather than demuxed from the container: the WAV already exists
    # and is the exact shape NVIDIA accepts.
    assert audio_path is not None
    assert audio_path.suffix == ".wav"
    assert diarization == [
        DiarizationSegment(start_time_ms=0, end_time_ms=2000, speaker_id=0),
        DiarizationSegment(start_time_ms=2000, end_time_ms=4000, speaker_id=1),
    ]


def test_the_audio_is_extracted_once_and_shared(fake_nvidia, fake_diarization, fake_audio):
    """The whole reason the diarization boundary was split: one decode, two consumers."""
    run()

    assert len(fake_audio.sources) == 1
    # Extracted from the artifact NVIDIA is given, so both timelines are the same one.
    assert fake_audio.sources == [VIDEO]

    _, _, nvidia_audio = fake_nvidia.calls[0]
    assert fake_diarization.audio_paths == [nvidia_audio]


def test_the_temporary_wav_is_removed_on_success(fake_nvidia):
    run()

    _, _, audio_path = fake_nvidia.calls[0]
    assert not audio_path.exists()


# --- failure, which must never cost the other evidence sources -----------------------------


def test_audio_extraction_failure_is_a_failed_signal(fake_audio):
    fake_audio.error = speaker_diarization.SpeakerDiarizationAudioError("no audio stream")

    signal, segments = run()

    assert signal.provider == "nvidia"
    assert signal.signal_type == ACTIVE_SPEAKER_SIGNAL
    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "SpeakerDiarizationAudioError"}
    assert segments == []


def test_a_missing_diarizer_is_a_failed_signal(fake_diarization):
    """No Hugging Face token is a configuration gap, and it costs this signal only."""
    fake_diarization.error = speaker_diarization.SpeakerDiarizationUnavailable(
        "HUGGINGFACE_TOKEN is not configured"
    )

    signal, segments = run()

    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "SpeakerDiarizationUnavailable"}
    assert segments == []


def test_a_broken_model_is_a_failed_signal(fake_diarization):
    fake_diarization.error = speaker_diarization.SpeakerDiarizationModelError("out of memory")

    signal, _ = run()

    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "SpeakerDiarizationModelError"}


def test_a_provider_refusal_is_a_failed_signal(fake_nvidia):
    fake_nvidia.error = nvidia_active_speaker.NvidiaActiveSpeakerAuthenticationError(
        "NVIDIA rejected the request"
    )

    signal, segments = run()

    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "NvidiaActiveSpeakerAuthenticationError"}
    assert segments == []


def test_a_provider_timeout_is_told_apart(fake_nvidia):
    fake_nvidia.error = nvidia_active_speaker.NvidiaActiveSpeakerTimeout("deadline exceeded")

    signal, _ = run()

    # NVIDIA may still have been working, which says nothing about the media either way.
    assert signal.status == "TIMEOUT"
    assert signal.signal_metadata == {"error": "NvidiaActiveSpeakerTimeout"}


def test_media_with_no_speech_produces_no_fabricated_success(fake_diarization, fake_nvidia):
    """pyannote heard nothing, so there is nothing to hand NVIDIA — which requires
    diarization and reports every face unmatched without it."""
    fake_diarization.turns = ()

    signal, segments = run()

    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "NvidiaActiveSpeakerInputError"}
    assert segments == []
    # The client refuses it before any RPC is started, so nothing was ever sent.
    assert fake_nvidia.calls[0][1] == []


def test_a_failure_never_leaks_the_local_path(fake_audio):
    fake_audio.error = speaker_diarization.SpeakerDiarizationAudioError(
        "ffmpeg failed on /tmp/deepguard-job-abc123"
    )

    signal, _ = run()

    assert "deepguard-job" not in str(signal.signal_metadata)


def test_a_failed_signal_carries_no_figures(fake_nvidia):
    fake_nvidia.error = nvidia_active_speaker.NvidiaActiveSpeakerUnavailable("unreachable")

    signal, _ = run()

    # Nothing was observed, so there is nothing to report — and the function that would
    # have identified the deployment never answered.
    assert signal.score is None
    assert signal.risk_level is None
    assert signal.provider_version is None


def test_the_temporary_wav_is_removed_after_a_failure(monkeypatch, fake_nvidia):
    created = []
    real_mkstemp = speaker_diarization.tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        descriptor, name = real_mkstemp(*args, **kwargs)
        created.append(Path(name))
        return descriptor, name

    monkeypatch.setattr(speaker_diarization.tempfile, "mkstemp", mkstemp)
    fake_nvidia.error = nvidia_active_speaker.NvidiaActiveSpeakerUnavailable("unreachable")

    run()

    assert created
    assert [path for path in created if path.exists()] == []
