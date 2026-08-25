"""The worker, against real PostgreSQL.

Claiming is the whole point of this module, and claiming is a concurrency property: two
workers reaching for the same row, and exactly one of them getting it. No fake session
can show that, so the job tests here run against the database from docker-compose and
skip when none is reachable. Only MinIO and NVIDIA are stood in for.

The loop tests at the bottom need no database — what they check is when the worker asks
for work again, not what it finds.
"""

import hashlib
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app import (
    detection,
    normalization,
    nvidia_active_speaker,
    nvidia_video,
    risk_engine,
    speaker_diarization,
    storage,
    worker,
)
from app.audio_detector import (
    AudioAuthenticityEvidence,
    AudioDetectorModelUnavailable,
    WindowEvidence,
)
from app.c2pa_extractor import C2paEvidence
from app.db.models import (
    Analysis,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import SessionLocal, engine
from app.nvidia_active_speaker import (
    NvidiaActiveSpeakerFrame,
    NvidiaActiveSpeakerResult,
    NvidiaBoundingBox,
    NvidiaSpeakerObservation,
)
from app.speaker_diarization import SpeakerTurn

VIDEO_BYTES = b"canonical-mp4-bytes"
ORIGINAL_BYTES = b"as-uploaded-original-bytes"
# What the stand-in transcoder writes. Distinct from both of the above, so a test can
# tell which artifact a detector was actually handed.
DERIVATIVE_BYTES = b"normalized-mp4-bytes"
NVIDIA_PROBABILITY = 0.8734567165374756
NVIDIA_LOGIT = 1.9142135381698608
NVIDIA_FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"

C2PA_SDK_VERSION = "0.90.14"
C2PA_SIGNATURE_ISSUER = "Test Signing Cert"

ASD_FUNCTION_ID = "f286f937-05c4-454b-8312-fba67a2a6fa7"

# The pinned AASIST artifact, and the shape of the audio it is fed. Restated rather than
# imported, so a checkpoint or a window length changed without anyone noticing shows up here
# as a failing test instead of a silently different measurement.
AASIST_REPOSITORY = "SpeechAntiSpoofingBenchmarks/AASIST"
AASIST_REVISION = "16774d458d86d2a021ae31646c1bf66a5331b53e"
AASIST_SHA256 = "130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1"
AASIST_WINDOW_SAMPLES = 64600
AASIST_SAMPLE_RATE = 16000
# The rate every queued upload below is probed at, and therefore the rate NVIDIA's frame
# indices are read against.
QUEUED_FRAME_RATE = 30.0

# Test jobs are backdated so that oldest-first claiming reaches them before anything a
# real upload left in the queue. It makes these tests deterministic against a shared
# development database without touching a single row they do not own.
BACKDATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def queue(database):
    """Creates queued uploads exactly as the route commits them, and removes them again."""
    created = []

    def enqueue(*, was_normalized=False, age=0):
        with SessionLocal() as session:
            analysis = Analysis(status="queued")
            session.add(analysis)
            session.flush()
            digest = uuid.uuid4().hex
            session.add(
                MediaFile(
                    analysis_id=analysis.id,
                    original_filename="clip.mp4",
                    content_type="video/mp4",
                    size_bytes=len(VIDEO_BYTES),
                    original_sha256=digest,
                    original_storage_key=f"originals/{digest}",
                    format_name="mov,mp4,m4a,3gp,3g2,mj2",
                    codec_name="h264",
                    width=1920,
                    height=1080,
                    duration=12.34,
                    frame_rate=30.0,
                    pix_fmt="yuv420p",
                    constant_frame_rate=True,
                    was_normalized=was_normalized,
                    # Exactly as the upload commits it: null while a derivative is owed,
                    # the original's own key when none will ever be produced.
                    derivative_storage_key=None if was_normalized else f"originals/{digest}",
                    derivative_sha256=None,
                )
            )
            job = AnalysisJob(
                analysis_id=analysis.id,
                status="queued",
                created_at=BACKDATE + timedelta(seconds=age),
            )
            session.add(job)
            session.commit()
            created.append(analysis.id)

            return analysis.id, job.id

    yield enqueue

    with SessionLocal() as session:
        for analysis_id in created:
            # Media, job, signal and segments all go through ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


@pytest.fixture
def fake_storage(monkeypatch):
    """Stand in for MinIO: hands the worker plausible bytes, and takes what it stores.

    Both directions matter now. The worker downloads the forensic original and, when the
    media needs one, uploads the derivative it transcoded — so this records uploads as
    well as fetches, and a test can check that a derivative was really stored before any
    row claimed it exists.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.upload_error = None
            self.fetched = []
            # Where the artifact was written, captured during the call: the worker
            # deletes it before returning, so looking afterwards would prove nothing.
            self.paths = []
            self.uploads = []
            self.uploaded_bytes = {}
            # Content-addressed objects are never deleted; this proves nothing tries to.
            self.removed = []

        @property
        def stored_keys(self):
            return [key for key, _ in self.uploads]

        def fget_object(self, bucket, key, file_path):
            self.fetched.append(key)
            self.paths.append(Path(file_path))
            if self.error:
                raise self.error
            # Distinguishable per artifact, so a test can tell which object was handed to
            # which evidence source rather than trusting the key it was asked for.
            Path(file_path).write_bytes(
                ORIGINAL_BYTES if key.startswith("originals/") else VIDEO_BYTES
            )

        def bucket_exists(self, bucket):
            return True

        def make_bucket(self, bucket):
            raise AssertionError("The bucket already exists.")

        def fput_object(self, bucket, key, file_path, content_type=None):
            if self.upload_error:
                raise self.upload_error
            # Captured while the file still exists: the worker deletes its temp files as
            # soon as the block that produced them ends.
            self.uploaded_bytes[key] = Path(file_path).read_bytes()
            self.uploads.append((key, content_type))

        def remove_object(self, bucket, key):
            self.removed.append(key)

    recorder = Recorder()
    monkeypatch.setattr(storage, "client", recorder)
    return recorder


@pytest.fixture(autouse=True)
def fake_ffmpeg(monkeypatch):
    """Replace only the transcode call, writing plausible derivative bytes.

    Everything around it — the decision to transcode at all, hashing, key derivation,
    storage and cleanup — is the real code under test. Autouse because a worker that
    reached the real ffmpeg would spend minutes on bytes that are not video.
    """

    class Recorder:
        def __init__(self):
            self.output = DERIVATIVE_BYTES
            self.error = None
            self.calls = []

        async def run(self, source, destination, frame_rate):
            self.calls.append((Path(source), Path(destination), frame_rate))
            if self.error:
                raise self.error
            Path(destination).write_bytes(self.output)

    recorder = Recorder()
    monkeypatch.setattr(normalization, "_run_ffmpeg", recorder.run)
    return recorder


@pytest.fixture(autouse=True)
def fake_nvidia(monkeypatch):
    """Stand in for the detector, so no test in this module can reach NVIDIA."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.clips = (
                nvidia_video.NvidiaClipResult(index=0, logit=-2.25),
                nvidia_video.NvidiaClipResult(index=8, logit=3.5),
            )
            self.analysed_bytes = []
            # Settable so a test can put the detection either side of the calibrated
            # threshold without patching the threshold itself.
            self.probability = NVIDIA_PROBABILITY
            self.function_id = NVIDIA_FUNCTION_ID
            self.total_clips = 7

        async def analyze(self, file_path, **kwargs):
            self.analysed_bytes.append(Path(file_path).read_bytes())
            if self.error:
                raise self.error
            return nvidia_video.NvidiaVideoResult(
                logit=NVIDIA_LOGIT,
                probability=self.probability,
                total_clips=self.total_clips,
                csv_data="0,-2.25\n8,3.5\n",
                function_id=self.function_id,
                clips=self.clips,
            )

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_video", recorder.analyze)
    return recorder


@pytest.fixture(autouse=True)
def fake_audio(monkeypatch):
    """Replace only the audio extraction, writing plausible WAV bytes.

    Autouse for the same reason as `fake_ffmpeg`: a worker that reached the real ffmpeg
    would spend its time demuxing bytes that are not media.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.calls = []
            self.sources_read = []

        async def extract(self, source, destination):
            self.calls.append((Path(source), Path(destination)))
            # Read during the call, because the worker deletes the artifact as soon as the
            # block that prepared it ends: which media the audio came from cannot be
            # established from a path that no longer exists.
            self.sources_read.append(Path(source).read_bytes())
            if self.error:
                raise self.error
            Path(destination).write_bytes(b"RIFF....WAVEprepared")

    recorder = Recorder()
    monkeypatch.setattr(speaker_diarization, "_extract_audio", recorder.extract)
    return recorder


@pytest.fixture(autouse=True)
def fake_diarization(monkeypatch):
    """Stand in for pyannote, so no test in this module loads torch or a gated model."""

    class Recorder:
        def __init__(self):
            self.error = None
            self.audio_paths = []
            self.turns = (
                SpeakerTurn(start_time=0.0, end_time=1.0, speaker_id="SPEAKER_00"),
                SpeakerTurn(start_time=1.0, end_time=2.0, speaker_id="SPEAKER_01"),
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
def fake_active_speaker(monkeypatch):
    """Stand in for the Active Speaker NIM, so no test here can reach NVIDIA.

    The scripted result is two faces speaking in turn: face 0 for frames 0-29 and face 1
    for frames 30-59, which at 30 fps is one second each.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.calls = []
            self.result = NvidiaActiveSpeakerResult(
                frames=tuple(
                    NvidiaActiveSpeakerFrame(
                        frame_id=frame_id,
                        speakers=(
                            NvidiaSpeakerObservation(
                                face_id=0 if frame_id < 30 else 1,
                                diarized_speaker_id=0 if frame_id < 30 else 1,
                                is_speaking=True,
                                face_detection_confidence=0.98,
                                bounding_box=NvidiaBoundingBox(
                                    x=8.0, y=16.0, width=64.0, height=64.0
                                ),
                            ),
                        ),
                    )
                    for frame_id in range(60)
                ),
                function_id=ASD_FUNCTION_ID,
                speaker_detection_threshold=0.5,
            )

        async def analyze(self, video_path, diarization, *, audio_path=None, **kwargs):
            self.calls.append(
                (
                    Path(video_path).read_bytes(),
                    list(diarization),
                    Path(audio_path) if audio_path else None,
                )
            )
            nvidia_active_speaker._validate_diarization(diarization)
            if self.error:
                raise self.error
            return self.result

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_active_speaker", recorder.analyze)
    return recorder


@pytest.fixture(autouse=True)
def fake_aasist(monkeypatch):
    """Stand in for the local checkpoint, so no test here loads onnxruntime or a model file.

    Three windows of scripted raw output, which is enough to prove the rows land in
    chronological order and keep both of the model's figures.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.audio_paths = []
            self.evidence = AudioAuthenticityEvidence(
                model_repository=AASIST_REPOSITORY,
                model_revision=AASIST_REVISION,
                model_sha256=AASIST_SHA256,
                sample_rate=AASIST_SAMPLE_RATE,
                channels=1,
                window_samples=AASIST_WINDOW_SAMPLES,
                window_padding_scheme="repeat-tile",
                total_samples=3 * AASIST_WINDOW_SAMPLES,
                windows=tuple(
                    WindowEvidence(
                        window_index=index,
                        start_sample=index * AASIST_WINDOW_SAMPLES,
                        end_sample=(index + 1) * AASIST_WINDOW_SAMPLES,
                        padded_samples=0,
                        logits=logits,
                        bona_fide_logit=logits[1],
                    )
                    # Deliberately disagreeing with each other, the way genuine speech does
                    # across consecutive windows (P6-T1 §7).
                    for index, logits in enumerate([(-1.5, 4.25), (2.0, -0.5), (-0.75, 3.0)])
                ),
            )

        def analyze(self, audio_path, **kwargs):
            self.audio_paths.append(Path(audio_path))
            if self.error:
                raise self.error
            return self.evidence

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_audio_authenticity", recorder.analyze)
    return recorder


@pytest.fixture(autouse=True)
def fake_c2pa(monkeypatch):
    """Stand in for the C2PA reader, which would reject the fake video bytes outright.

    What these tests are about is the flow around it — which artifact it is handed, and
    what becomes of its answer. Reading real credentials out of real media is covered
    against real media in `test_c2pa.py`.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.read_bytes = []
            self.evidence = C2paEvidence(
                manifest_exists=True,
                sdk_version=C2PA_SDK_VERSION,
                validation_state="Valid",
                validation_failures=("signingCredential.untrusted",),
                is_embedded=True,
                active_manifest_label="urn:c2pa:6f0e1a2b",
                claim_generator="test-camera",
                signature_issuer=C2PA_SIGNATURE_ISSUER,
                assertion_labels=("c2pa.actions.v2",),
                manifest_json='{"active_manifest": "urn:c2pa:6f0e1a2b"}',
            )

        def extract(self, file_path):
            self.read_bytes.append(Path(file_path).read_bytes())
            if self.error:
                raise self.error
            return self.evidence

    recorder = Recorder()
    monkeypatch.setattr(detection, "extract_c2pa_evidence", recorder.extract)
    return recorder


def release(claimed) -> None:
    """Put a job this test did not create back the way it found it.

    These tests share a development database with real uploads, and `claim_job` takes
    whatever is queued. Test jobs are backdated so they are always claimed first, but a
    claim that reaches past them has taken someone else's work — and leaving it
    `processing` would strand it, because nothing in P3 recovers a stalled job.
    """
    if claimed is None:
        return

    with SessionLocal() as session:
        session.query(AnalysisJob).filter(AnalysisJob.id == claimed.job_id).update(
            {"status": "queued"}
        )
        session.commit()


def read_job(job_id) -> AnalysisJob:
    """The job as another connection sees it — committed state, not session state."""
    with SessionLocal() as reader:
        return reader.query(AnalysisJob).filter(AnalysisJob.id == job_id).one()


def read_analysis(analysis_id) -> Analysis:
    with SessionLocal() as reader:
        return reader.query(Analysis).filter(Analysis.id == analysis_id).one()


def read_signals(analysis_id) -> dict[str, AnalysisSignal]:
    """The analysis's persisted signals, keyed by what each of them answers.

    Keyed by `signal_type` rather than by provider: NVIDIA now answers two different
    questions about one analysis, so the provider no longer identifies a row.
    """
    with SessionLocal() as reader:
        rows = reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).all()

    return {row.signal_type: row for row in rows}


def read_segments(signal_id) -> list[AnalysisSegment]:
    """One signal's persisted evidence, oldest row first."""
    with SessionLocal() as reader:
        return (
            reader.query(AnalysisSegment)
            .filter_by(signal_id=signal_id)
            .order_by(AnalysisSegment.created_at, AnalysisSegment.start_time)
            .all()
        )


def read_media(analysis_id) -> MediaFile:
    """The media row as another connection sees it, derivative columns included."""
    with SessionLocal() as reader:
        return reader.query(MediaFile).filter_by(analysis_id=analysis_id).one()


@pytest.mark.integration
def test_claiming_a_job_marks_it_processing_and_commits(queue):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    assert claimed.job_id == job_id
    assert claimed.analysis_id == analysis_id
    # Read on another connection: if the claim had not committed, this would still say
    # queued — and the transaction would still be holding the lock through inference.
    assert read_job(job_id).status == "processing"


@pytest.mark.integration
def test_claiming_carries_what_the_work_needs_out_of_the_transaction(queue):
    _, job_id = queue(was_normalized=True)

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    # The forensic original: the only artifact that exists yet, and the one provenance
    # must be read from.
    assert claimed.original_storage_key.startswith("originals/")
    # The decision the upload made from the probe, which this worker carries out, and the
    # rate the transcode has to hold constant. Neither can be re-derived here: the
    # decision needs `major_brand`, and no column holds it.
    assert claimed.normalization_required is True
    assert claimed.frame_rate == 30.0


def test_claiming_media_that_needs_no_derivative_says_so(queue):
    queue(was_normalized=False)

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    assert claimed.normalization_required is False


@pytest.mark.integration
def test_a_locked_job_is_skipped_rather_than_waited_for(queue):
    """The property `SKIP LOCKED` exists for, isolated.

    One session holds the row lock without committing. A second must come back with
    nothing at once — not block until the first is done, which is what a plain
    `FOR UPDATE` would do and what would serialize every worker in the fleet behind the
    slowest video.
    """
    queue()

    holder = SessionLocal()
    try:
        held = worker.claim_job(holder)
        assert held is not None

        with SessionLocal() as other:
            # If the statement ever blocks instead of skipping, this makes it fail
            # loudly rather than hang the suite forever.
            other.execute(text("SET lock_timeout = '2000ms'"))
            second = worker.claim_job(other)
            release(second)

        # Whatever else the queue happens to hold, the locked row is not what the second
        # worker came back with.
        assert second is None or second.job_id != held.job_id
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.integration
def test_racing_workers_never_claim_the_same_job_twice(queue):
    """Several workers, several jobs, started together on purpose.

    Every job must be claimed, and none of them twice. A duplicate claim would mean the
    same video analysed twice and two conflicting sets of evidence for one analysis.
    """
    expected = {queue(age=index)[1] for index in range(6)}
    start = threading.Barrier(3)
    claimed = []
    guard = threading.Lock()

    def work():
        start.wait()
        with SessionLocal() as session:
            while True:
                job = worker.claim_job(session)
                if job is None:
                    return
                if job.job_id not in expected:
                    # Reached past this test's jobs into the real queue. Put it back.
                    release(job)
                    return
                with guard:
                    claimed.append(job.job_id)

    threads = [threading.Thread(target=work) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(claimed) == sorted(set(claimed))
    assert set(claimed) == expected


@pytest.mark.integration
def test_a_processed_job_completes_with_its_evidence(queue, fake_storage):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    assert read_job(job_id).status == "completed"
    assert read_job(job_id).error_message is None
    # The analysis moves with its job: detection is the work it was waiting for.
    assert read_analysis(analysis_id).status == "completed"

    signal = read_signals(analysis_id)["synthetic_video"]
    with SessionLocal() as reader:
        segments = (
            reader.query(AnalysisSegment)
            .filter_by(signal_id=signal.id)
            .order_by(AnalysisSegment.logit.desc())
            .all()
        )

    assert signal.signal_type == "synthetic_video"
    assert signal.status == "SUCCESS"
    # NVIDIA's own number, on NVIDIA's own scale.
    assert signal.score == NVIDIA_PROBABILITY
    assert signal.risk_level is None
    assert [(s.clip_index, s.logit) for s in segments] == [(8, 3.5), (0, -2.25)]


@pytest.mark.integration
def test_the_worker_detects_the_derivative_it_produced(queue, fake_storage, fake_nvidia):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    # Only the original is ever fetched — the derivative did not exist to be fetched, and
    # once transcoded it is already on this disk, so downloading it back would be a round
    # trip to fetch bytes the worker just wrote.
    assert fake_storage.fetched == [key for key in fake_storage.fetched if key.startswith("originals/")]
    assert fake_nvidia.analysed_bytes == [DERIVATIVE_BYTES]


def test_the_worker_detects_the_original_when_no_derivative_is_needed(
    queue, fake_storage, fake_nvidia, fake_ffmpeg
):
    queue(was_normalized=False)

    with SessionLocal() as session:
        worker.process_one(session)

    # Already canonical: nothing is transcoded and the bytes as uploaded are what the
    # detector sees.
    assert fake_ffmpeg.calls == []
    assert fake_nvidia.analysed_bytes == [ORIGINAL_BYTES]


@pytest.mark.integration
def test_both_evidence_sources_are_persisted_as_independent_signals(queue, fake_storage):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert sorted(signals) == [
        "active_speaker",
        "audio_authenticity",
        "provenance",
        "synthetic_video",
    ]
    assert read_job(job_id).status == "completed"

    provenance = signals["provenance"]
    assert provenance.signal_type == "provenance"
    assert provenance.status == "SUCCESS"
    # Provenance is a set of facts about a signature, not a figure on a scale. A number
    # here would sit next to NVIDIA's probability as if the two could be compared.
    assert provenance.score is None
    assert provenance.risk_level is None
    assert provenance.provider_version == C2PA_SDK_VERSION
    assert provenance.signal_metadata["manifest_exists"] is True
    assert provenance.signal_metadata["validation_state"] == "Valid"
    assert provenance.signal_metadata["validation_failures"] == ["signingCredential.untrusted"]
    assert provenance.signal_metadata["signature_issuer"] == C2PA_SIGNATURE_ISSUER
    assert provenance.signal_metadata["assertion_labels"] == ["c2pa.actions.v2"]


@pytest.mark.integration
def test_media_carrying_no_credentials_is_still_a_successful_reading(
    queue, fake_storage, fake_c2pa
):
    analysis_id, _ = queue()
    fake_c2pa.evidence = C2paEvidence(manifest_exists=False, sdk_version=C2PA_SDK_VERSION)

    with SessionLocal() as session:
        worker.process_one(session)

    provenance = read_signals(analysis_id)["provenance"]

    # Most media carries no credentials. Recording that as a failure would hide the most
    # common answer there is behind a status that means something went wrong.
    assert provenance.status == "SUCCESS"
    assert provenance.signal_metadata["manifest_exists"] is False
    assert provenance.signal_metadata["validation_state"] is None


@pytest.mark.integration
def test_provenance_is_read_from_the_forensic_original(queue, fake_storage, fake_c2pa):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    # Normalization re-encodes the video and strips any manifest with it, so reading
    # credentials off the derivative would report every normalized upload as unsigned.
    assert fake_c2pa.read_bytes == [ORIGINAL_BYTES]
    assert [key for key in fake_storage.fetched if key.startswith("originals/")] != []


@pytest.mark.integration
def test_a_broken_provenance_reading_does_not_cost_the_analysis_its_detection(
    queue, fake_storage, fake_c2pa
):
    analysis_id, job_id = queue()
    fake_c2pa.error = RuntimeError("the native library gave up on /tmp/deepguard-job-xyz")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    # One evidence source breaking must not destroy the analysis the others succeeded on.
    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"
    assert signals["synthetic_video"].status == "SUCCESS"
    assert signals["provenance"].status == "FAILED"
    # The failure kind and nothing else: the message quotes a local artifact path.
    assert signals["provenance"].signal_metadata == {"error": "RuntimeError"}


@pytest.mark.integration
def test_the_downloaded_artifact_is_removed_afterwards(queue, fake_storage):
    queue()

    with SessionLocal() as session:
        worker.process_one(session)

    # The worker holds no media between jobs; a container that ran for a week would
    # otherwise fill its disk with every video it had ever been asked about.
    assert fake_storage.paths
    assert [path for path in fake_storage.paths if path.exists()] == []


@pytest.mark.integration
def test_a_provider_failure_still_completes_the_job(queue, fake_storage, fake_nvidia):
    analysis_id, job_id = queue()
    fake_nvidia.error = nvidia_video.NvidiaProviderTimeout("deadline exceeded")

    with SessionLocal() as session:
        worker.process_one(session)

    # NVIDIA not answering is a fact about NVIDIA, recorded as the signal's own status.
    # The job did the work it was queued to do, so it is not the job that failed.
    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"

    signal = read_signals(analysis_id)["synthetic_video"]

    assert signal.status == "TIMEOUT"
    assert signal.score is None


@pytest.mark.integration
def test_an_unreadable_artifact_fails_the_job_and_the_analysis(queue, fake_storage):
    analysis_id, job_id = queue()
    fake_storage.error = OSError("connection reset")

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    # This one is ours, not the detector's. An analysis left queued behind a job that
    # already gave up would look like work still coming.
    assert read_job(job_id).status == "failed"
    assert read_analysis(analysis_id).status == "failed"

    with SessionLocal() as reader:
        assert reader.query(AnalysisSignal).filter_by(analysis_id=analysis_id).count() == 0


@pytest.mark.integration
def test_a_failed_job_records_the_failure_kind_and_nothing_more(queue, fake_storage):
    _, job_id = queue()
    fake_storage.error = OSError("connect to minio:9000 failed for user deepguard")

    with SessionLocal() as session:
        worker.process_one(session)

    # Exception text can quote credentials, storage endpoints or SQL. The class name is
    # what an operator needs from the table; the traceback stays in the worker log.
    assert read_job(job_id).error_message == "OSError"


@pytest.mark.integration
def test_a_failed_job_does_not_stop_the_next_one(queue, fake_storage):
    _, failing = queue(age=0)
    _, following = queue(age=1)
    fake_storage.error = OSError("connection reset")

    with SessionLocal() as session:
        worker.process_one(session)
        fake_storage.error = None
        worker.process_one(session)

    # One broken job is not a broken worker.
    assert read_job(failing).status == "failed"
    assert read_job(following).status == "completed"


# The loop. What it checks is when the worker asks for work again, not what it finds, so
# nothing below needs a database.


class FakeClock:
    """Records what the loop slept for instead of actually sleeping."""

    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


def run_loop(monkeypatch, outcomes):
    """Drive `run` through a fixed sequence of `process_one` results, then stop."""
    stopping = worker.Stopping()
    remaining = list(outcomes)

    def process_one(_session):
        if not remaining:
            # Stop without a further pause, so the sequence under test is the only thing
            # the clock records.
            stopping.requested = True
            return True
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    clock = FakeClock()
    monkeypatch.setattr(worker, "process_one", process_one)
    monkeypatch.setattr(worker, "SessionLocal", lambda: _NullSession())
    worker.run(stopping, sleep=clock)

    return clock


class _NullSession:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_the_loop_does_not_pause_while_there_is_work(monkeypatch):
    clock = run_loop(monkeypatch, [True, True, True])

    # A backlog is drained as fast as it can be, not one job per poll interval.
    assert clock.slept == []


def test_an_empty_queue_is_polled_rather_than_busy_looped(monkeypatch):
    clock = run_loop(monkeypatch, [False, False])

    assert clock.slept == [worker.IDLE_POLL_SECONDS, worker.IDLE_POLL_SECONDS]


def test_a_broken_database_backs_off_instead_of_exiting(monkeypatch):
    clock = run_loop(monkeypatch, [SQLAlchemyError("connection lost"), True])

    # Restarting the container would only meet the same condition, and a crash loop is
    # harder to read in the logs than a worker saying the same thing every few seconds.
    assert clock.slept == [worker.ERROR_BACKOFF_SECONDS]


def test_the_loop_stops_when_shutdown_is_requested(monkeypatch):
    stopping = worker.Stopping()
    stopping.request()

    called = []
    monkeypatch.setattr(worker, "process_one", lambda _s: called.append(1))

    worker.run(stopping, sleep=FakeClock())

    # SIGTERM from `docker compose down`: stop asking for new work rather than being
    # killed ten seconds later, mid-job.
    assert called == []


# Normalization. It ran on the upload request until P4-F2, where a 4K HEVC file passed
# validation and was then rejected as unprocessable because a deadline meant for a
# waiting client expired mid-transcode. It happens here now, between the claim and the
# detection, and the derivative it produces is named in the database only once it exists.


@pytest.mark.integration
def test_the_transcode_reads_the_fetched_original_at_its_source_rate(
    queue, fake_storage, fake_ffmpeg
):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    assert len(fake_ffmpeg.calls) == 1
    source, destination, frame_rate = fake_ffmpeg.calls[0]
    # The forensic original as downloaded — the same file provenance was read from.
    assert source == fake_storage.paths[0]
    assert destination.suffix == ".mp4"
    # The source's own rate, from `media_files`, not a default imposed here.
    assert frame_rate == 30.0


@pytest.mark.integration
def test_the_derivative_is_hashed_and_keyed_on_its_own_bytes(queue, fake_storage):
    analysis_id, _ = queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    media = read_media(analysis_id)
    expected = hashlib.sha256(DERIVATIVE_BYTES).hexdigest()
    # A different artifact from the original, so a different identity: sharing the
    # original's hash would say the two sets of bytes were the same.
    assert media.derivative_sha256 == expected
    assert media.derivative_sha256 != media.original_sha256
    assert media.derivative_storage_key == f"derivatives/{expected}.mp4"


@pytest.mark.integration
def test_the_derivative_is_stored_as_mp4_beside_the_original(queue, fake_storage):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    # One upload, and it is the derivative: the original was already stored by the
    # request that accepted it and is only read here.
    assert len(fake_storage.uploads) == 1
    key, content_type = fake_storage.uploads[0]
    assert key.startswith("derivatives/")
    assert content_type == "video/mp4"
    assert fake_storage.uploaded_bytes[key] == DERIVATIVE_BYTES
    assert fake_storage.removed == []


@pytest.mark.integration
def test_the_derivative_is_stored_before_anything_records_it(queue, fake_storage):
    analysis_id, _ = queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    # The row names an object that provably exists, because the upload happened first.
    media = read_media(analysis_id)
    assert media.derivative_storage_key in fake_storage.uploaded_bytes


@pytest.mark.integration
def test_media_needing_no_derivative_keeps_the_key_the_upload_wrote(queue, fake_storage):
    analysis_id, _ = queue(was_normalized=False)
    before = read_media(analysis_id).derivative_storage_key

    with SessionLocal() as session:
        worker.process_one(session)

    media = read_media(analysis_id)
    # Nothing was transcoded and nothing was stored, so the column is not touched: there
    # is no second artifact, and overwriting it would say there was.
    assert media.derivative_storage_key == before == media.original_storage_key
    assert media.derivative_sha256 is None
    assert fake_storage.uploads == []


@pytest.mark.integration
def test_the_transcoded_derivative_is_removed_afterwards(queue, fake_storage, fake_ffmpeg):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    _, destination, _ = fake_ffmpeg.calls[0]
    # The derivative is in MinIO; the local copy is not the worker's to keep.
    assert not destination.exists()
    assert [path for path in fake_storage.paths if path.exists()] == []


@pytest.mark.integration
def test_media_that_cannot_be_transcoded_does_not_cost_the_analysis_its_provenance(
    queue, fake_storage, fake_ffmpeg
):
    analysis_id, job_id = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)
    # NVIDIA takes MP4 and nothing else, so a failed transcode means it was never asked.
    # That is a gap in one source's evidence, not a reason to throw the job away.
    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"
    assert signals["provenance"].status == "SUCCESS"
    assert signals["synthetic_video"].status == "FAILED"
    assert signals["synthetic_video"].score is None
    assert signals["synthetic_video"].signal_metadata == {"error": "NormalizationError"}


@pytest.mark.integration
def test_a_transcode_that_ran_out_of_time_is_told_apart_from_one_that_broke(
    queue, fake_storage, fake_ffmpeg
):
    analysis_id, _ = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationTimeout("ffmpeg timed out after 900s")

    with SessionLocal() as session:
        worker.process_one(session)

    signal = read_signals(analysis_id)["synthetic_video"]
    # `TIMEOUT` is reserved for a provider that may still have been working, which says
    # nothing about the media. ffmpeg giving up is not that; the failure kind is what
    # separates the two.
    assert signal.status == "FAILED"
    assert signal.signal_metadata == {"error": "NormalizationTimeout"}


@pytest.mark.integration
def test_a_failed_transcode_records_no_derivative(queue, fake_storage, fake_ffmpeg):
    analysis_id, _ = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    with SessionLocal() as session:
        worker.process_one(session)

    media = read_media(analysis_id)
    assert media.derivative_storage_key is None
    assert media.derivative_sha256 is None
    assert fake_storage.uploads == []


@pytest.mark.integration
def test_a_failed_transcode_leaves_nothing_on_disk(queue, fake_storage, fake_ffmpeg):
    queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    with SessionLocal() as session:
        worker.process_one(session)

    assert [path for path in fake_storage.paths if path.exists()] == []


@pytest.mark.integration
def test_a_missing_ffmpeg_binary_fails_the_job_rather_than_the_media(
    queue, fake_storage, fake_ffmpeg
):
    analysis_id, job_id = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationUnavailable("ffmpeg could not be executed")

    with SessionLocal() as session:
        worker.process_one(session)

    # A container without its media processor is a broken machine, not evidence about a
    # video. Recording it as a signal would leave a real defect looking like a routine gap.
    assert read_job(job_id).status == "failed"
    assert read_job(job_id).error_message == "NormalizationUnavailable"
    assert read_analysis(analysis_id).status == "failed"
    assert read_signals(analysis_id) == {}


@pytest.mark.integration
def test_a_derivative_that_cannot_be_stored_fails_the_job(queue, fake_storage):
    analysis_id, job_id = queue(was_normalized=True)
    fake_storage.upload_error = RuntimeError("object storage is unreachable")

    with SessionLocal() as session:
        worker.process_one(session)

    # The transcode worked; putting it where the evidence can point at it did not. That
    # is this side failing, so the job says so and no signal claims otherwise.
    assert read_job(job_id).status == "failed"
    assert read_analysis(analysis_id).status == "failed"
    assert read_signals(analysis_id) == {}


@pytest.mark.integration
def test_a_derivative_upload_failure_leaves_nothing_on_disk(
    queue, fake_storage, fake_ffmpeg
):
    queue(was_normalized=True)
    fake_storage.upload_error = RuntimeError("object storage is unreachable")

    with SessionLocal() as session:
        worker.process_one(session)

    _, destination, _ = fake_ffmpeg.calls[0]
    assert not destination.exists()
    assert [path for path in fake_storage.paths if path.exists()] == []


# Active speaker. The third evidence source, and the first to produce real time ranges.
# It runs against the same prepared artifact the synthetic-video detector is given, off
# audio extracted from that artifact once and handed to both pyannote and NVIDIA.


@pytest.mark.integration
def test_all_four_evidence_sources_are_persisted_as_independent_signals(
    queue, fake_storage
):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert sorted(signals) == [
        "active_speaker",
        "audio_authenticity",
        "provenance",
        "synthetic_video",
    ]

    speaker = signals["active_speaker"]
    # NVIDIA answers two questions about one analysis, and they are two rows: the provider
    # is the same company, the finding is not the same finding.
    assert speaker.provider == "nvidia"
    assert speaker.provider == signals["synthetic_video"].provider
    assert speaker.id != signals["synthetic_video"].id
    assert speaker.status == "SUCCESS"
    assert speaker.provider_version == ASD_FUNCTION_ID
    # A timeline, not a figure on a scale. A number here would sit in the same column as
    # NVIDIA's synthetic probability as though the two could be compared.
    assert speaker.score is None
    assert speaker.risk_level is None


@pytest.mark.integration
def test_real_speaking_times_are_persisted(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    segments = read_segments(read_signals(analysis_id)["active_speaker"].id)

    # Face 0 speaks for frames 0-29 and face 1 for frames 30-59, at the 30 fps this media
    # was probed at — so one second each, back to back, in the order they happened.
    assert [(s.start_time, s.end_time) for s in segments] == [(0.0, 1.0), (1.0, 2.0)]
    assert [s.face_id for s in segments] == [0, 1]


@pytest.mark.integration
def test_speaking_segments_carry_the_labels_the_model_reported(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    segments = read_segments(read_signals(analysis_id)["active_speaker"].id)

    # pyannote's own strings, not the integers this codebase assigned them for NVIDIA's
    # wire format: the label is the model's finding, the integer is our encoding of it.
    assert [s.speaker_label for s in segments] == ["SPEAKER_00", "SPEAKER_01"]


@pytest.mark.integration
def test_speaking_segments_invent_no_clip_evidence(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    segments = read_segments(read_signals(analysis_id)["active_speaker"].id)

    # There is no clip and no logit in an active-speaker result, so both stay null rather
    # than being filled in to make the table look uniform.
    assert [(s.clip_index, s.logit) for s in segments] == [(None, None), (None, None)]


@pytest.mark.integration
def test_clip_evidence_is_unchanged_by_the_new_signal(queue, fake_storage):
    """The synthetic-video rows still look exactly as they did before P5-T3."""
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signal = read_signals(analysis_id)["synthetic_video"]
    segments = read_segments(signal.id)

    assert signal.score == NVIDIA_PROBABILITY
    assert sorted((s.clip_index, s.logit) for s in segments) == [(0, -2.25), (8, 3.5)]
    # NVIDIA reports no times for a clip, so converting its frame index into one would
    # invent a figure it never gave.
    assert [(s.start_time, s.end_time, s.face_id, s.speaker_label) for s in segments] == [
        (None, None, None, None),
        (None, None, None, None),
    ]


@pytest.mark.integration
def test_each_signal_owns_only_its_own_evidence(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    # Three signals with evidence rows, and no row belongs to two of them. Provenance owns
    # none at all: it produces no within-media evidence.
    assert len(read_segments(signals["synthetic_video"].id)) == 2
    assert len(read_segments(signals["active_speaker"].id)) == 2
    assert len(read_segments(signals["audio_authenticity"].id)) == 3
    assert read_segments(signals["provenance"].id) == []


@pytest.mark.integration
def test_the_signal_records_the_rate_the_frames_were_read_against(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    metadata = read_signals(analysis_id)["active_speaker"].signal_metadata

    # The rate of the artifact NVIDIA was actually given. Without it the persisted times
    # cannot be checked back against NVIDIA's own frame numbering.
    assert metadata["frame_rate"] == QUEUED_FRAME_RATE
    assert metadata["total_frames"] == 60
    assert metadata["total_speaking_segments"] == 2
    assert metadata["segments_truncated"] is False
    assert metadata["diarized_speakers"] == {"SPEAKER_00": 0, "SPEAKER_01": 1}


@pytest.mark.integration
def test_nvidia_is_given_deterministically_numbered_speakers(queue, fake_storage, fake_active_speaker):
    """pyannote names voices with strings; NVIDIA's proto carries a uint32."""
    queue()

    with SessionLocal() as session:
        worker.process_one(session)

    _, diarization, _ = fake_active_speaker.calls[0]

    # Numbered by first appearance, in milliseconds, with nothing parsed out of the label.
    assert [(d.start_time_ms, d.end_time_ms, d.speaker_id) for d in diarization] == [
        (0, 1000, 0),
        (1000, 2000, 1),
    ]


@pytest.mark.integration
def test_the_prepared_artifact_is_what_both_nvidia_signals_see(
    queue, fake_storage, fake_nvidia, fake_active_speaker
):
    """One transcode, two questions. Preparing it twice would pay for it twice."""
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    video_bytes, _, _ = fake_active_speaker.calls[0]

    assert fake_nvidia.analysed_bytes == [DERIVATIVE_BYTES]
    assert video_bytes == DERIVATIVE_BYTES


@pytest.mark.integration
def test_the_audio_is_extracted_once_from_that_artifact(
    queue, fake_storage, fake_audio, fake_diarization, fake_active_speaker, fake_aasist
):
    queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    # One extraction for the whole job: doing it per consumer would decode the same media
    # three times to produce three identical files.
    assert len(fake_audio.calls) == 1
    _, destination = fake_audio.calls[0]

    # Taken from the artifact NVIDIA is given rather than the forensic original, so the
    # diarization times and the frames they are matched against share one timeline.
    assert fake_audio.sources_read == [DERIVATIVE_BYTES]
    _, _, nvidia_audio = fake_active_speaker.calls[0]
    assert fake_diarization.audio_paths == [destination]
    assert nvidia_audio == destination
    # And the local checkpoint reads that same file rather than extracting its own, so
    # every audio signal on this analysis describes one recording.
    assert fake_aasist.audio_paths == [destination]


@pytest.mark.integration
def test_the_temporary_wav_is_removed_afterwards(queue, fake_storage, fake_audio):
    queue()

    with SessionLocal() as session:
        worker.process_one(session)

    _, destination = fake_audio.calls[0]
    # A container that ran for a week would otherwise hold one WAV per analysed video.
    assert not destination.exists()


# Partial failure. One evidence source breaking must never cost the others, which is the
# whole reason each is written as its own row with its own status.


@pytest.mark.integration
def test_media_with_no_audio_keeps_the_other_two_signals(queue, fake_storage, fake_audio):
    analysis_id, job_id = queue()
    fake_audio.error = speaker_diarization.SpeakerDiarizationAudioError("no audio stream")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"
    assert signals["active_speaker"].status == "FAILED"
    assert signals["active_speaker"].signal_metadata == {
        "error": "SpeakerDiarizationAudioError"
    }
    # Both audio-dependent signals were waiting on the same file, so both record the gap —
    # each in its own row, neither standing in for the other.
    assert signals["audio_authenticity"].provider == "aasist"
    assert signals["audio_authenticity"].status == "FAILED"
    assert signals["audio_authenticity"].signal_metadata == {
        "error": "SpeakerDiarizationAudioError"
    }
    # Silent video is common, and it says nothing about whether the video is synthetic or
    # what provenance it carries.
    assert signals["synthetic_video"].status == "SUCCESS"
    assert signals["synthetic_video"].score == NVIDIA_PROBABILITY
    assert signals["provenance"].status == "SUCCESS"


@pytest.mark.integration
def test_an_unconfigured_diarizer_costs_only_its_own_signal(
    queue, fake_storage, fake_diarization
):
    """A missing Hugging Face token is a configuration gap, not a fact about the media."""
    analysis_id, job_id = queue()
    fake_diarization.error = speaker_diarization.SpeakerDiarizationUnavailable(
        "HUGGINGFACE_TOKEN is not configured"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert signals["active_speaker"].status == "FAILED"
    assert signals["synthetic_video"].status == "SUCCESS"
    assert signals["provenance"].status == "SUCCESS"
    assert read_segments(signals["active_speaker"].id) == []
    # The other source's evidence is untouched by the gap in this one.
    assert len(read_segments(signals["synthetic_video"].id)) == 2


@pytest.mark.integration
def test_an_active_speaker_refusal_does_not_touch_the_other_signals(
    queue, fake_storage, fake_active_speaker
):
    analysis_id, job_id = queue()
    fake_active_speaker.error = (
        nvidia_active_speaker.NvidiaActiveSpeakerAuthenticationError("rejected")
    )

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert signals["active_speaker"].status == "FAILED"
    assert signals["active_speaker"].score is None
    assert signals["active_speaker"].provider_version is None
    assert signals["synthetic_video"].status == "SUCCESS"


@pytest.mark.integration
def test_an_active_speaker_timeout_is_told_apart_from_a_refusal(
    queue, fake_storage, fake_active_speaker
):
    analysis_id, _ = queue()
    fake_active_speaker.error = nvidia_active_speaker.NvidiaActiveSpeakerTimeout(
        "deadline exceeded"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    # NVIDIA may still have been working, which says nothing about the media either way.
    assert read_signals(analysis_id)["active_speaker"].status == "TIMEOUT"


@pytest.mark.integration
def test_a_synthetic_video_failure_does_not_cost_the_speaker_timeline(
    queue, fake_storage, fake_nvidia
):
    """The isolation runs both ways: the other NIM failing leaves this one intact."""
    analysis_id, job_id = queue()
    fake_nvidia.error = nvidia_video.NvidiaProviderTimeout("deadline exceeded")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert signals["synthetic_video"].status == "TIMEOUT"
    assert signals["active_speaker"].status == "SUCCESS"
    assert len(read_segments(signals["active_speaker"].id)) == 2


@pytest.mark.integration
def test_media_that_cannot_be_transcoded_fails_every_artifact_signal(
    queue, fake_storage, fake_ffmpeg
):
    """All three read the prepared artifact, so none is reachable without it.

    The audio the local checkpoint reads is extracted from that artifact too, which is why
    a failed transcode reaches it as well even though it calls no provider.
    """
    analysis_id, job_id = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg exited with 1")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert signals["synthetic_video"].status == "FAILED"
    assert signals["active_speaker"].status == "FAILED"
    assert signals["audio_authenticity"].status == "FAILED"
    # Each records the gap in its own right rather than one standing in for the other.
    assert signals["active_speaker"].signal_type == "active_speaker"
    assert signals["active_speaker"].signal_metadata == {"error": "NormalizationError"}
    assert signals["audio_authenticity"].provider == "aasist"
    assert signals["audio_authenticity"].signal_metadata == {"error": "NormalizationError"}
    # And the source that had already answered keeps its evidence.
    assert signals["provenance"].status == "SUCCESS"


@pytest.mark.integration
def test_an_active_speaker_failure_leaves_no_temp_files_behind(
    queue, fake_storage, fake_audio, fake_active_speaker
):
    queue(was_normalized=True)
    fake_active_speaker.error = nvidia_active_speaker.NvidiaActiveSpeakerUnavailable(
        "unreachable"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    _, destination = fake_audio.calls[0]
    assert not destination.exists()
    assert [path for path in fake_storage.paths if path.exists()] == []


# The local audio evidence. A fourth signal, sharing the prepared WAV with active speaker
# and sharing nothing else.


@pytest.mark.integration
def test_the_audio_signal_is_persisted_without_a_file_level_score(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signal = read_signals(analysis_id)["audio_authenticity"]

    assert signal.provider == "aasist"
    assert signal.status == "SUCCESS"
    # The checkpoint publishes no softmax, threshold or class over its two logits, so there
    # is no file-level figure to store and none is invented (rule 11).
    assert signal.score is None
    assert signal.risk_level is None
    assert signal.provider_version == f"{AASIST_REPOSITORY}@{AASIST_REVISION}"


@pytest.mark.integration
def test_audio_windows_are_persisted_chronologically_with_both_logits(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    segments = read_segments(read_signals(analysis_id)["audio_authenticity"].id)

    # One row per window, in the order the recording was cut, each keeping both of the
    # model's raw outputs in graph order. Neither can be derived from the other, so
    # dropping one would throw away half of what the model said.
    assert [s.clip_index for s in segments] == [0, 1, 2]
    assert [(s.logit, s.bona_fide_logit) for s in segments] == [
        (-1.5, 4.25),
        (2.0, -0.5),
        (-0.75, 3.0),
    ]


@pytest.mark.integration
def test_audio_window_bounds_are_the_preprocessing_boundaries(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signal = read_signals(analysis_id)["audio_authenticity"]
    segments = read_segments(signal.id)

    # 64600 samples at 16 kHz is 4.0375 s. These are the bounds of the windows DeepGuard cut
    # and fed to the graph — AASIST publishes no chunk-to-time mapping and reports no
    # segments, so they are never a claim that the model located anything in that interval.
    assert [(s.start_time, s.end_time) for s in segments] == [
        (0.0, 4.0375),
        (4.0375, 8.075),
        (8.075, 12.1125),
    ]
    assert signal.signal_metadata["window_bounds"] == "deepguard_preprocessing"
    # There is no face and no voice identity in an anti-spoofing result.
    assert [(s.face_id, s.speaker_label) for s in segments] == [(None, None)] * 3


@pytest.mark.integration
def test_the_audio_signal_records_what_produced_its_windows(queue, fake_storage):
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    metadata = read_signals(analysis_id)["audio_authenticity"].signal_metadata

    # Enough to reproduce the measurement: a different checkpoint revision, sample rate or
    # window length is a different number, not a refinement of this one.
    assert metadata["model_repository"] == AASIST_REPOSITORY
    assert metadata["model_revision"] == AASIST_REVISION
    assert metadata["model_sha256"] == AASIST_SHA256
    assert metadata["sample_rate"] == AASIST_SAMPLE_RATE
    assert metadata["window_samples"] == AASIST_WINDOW_SAMPLES
    assert metadata["window_padding_scheme"] == "repeat-tile"
    assert metadata["total_audio_windows"] == 3
    assert metadata["persisted_audio_windows"] == 3
    assert metadata["windows_truncated"] is False
    # Which of the two stored logits carries the meaning upstream gives it.
    assert metadata["bona_fide_logit_index"] == 1


@pytest.mark.integration
def test_a_truncated_audio_sweep_says_so_on_the_signal(queue, fake_storage, fake_aasist):
    total = detection.MAX_PERSISTED_AUDIO_WINDOWS + 6
    fake_aasist.evidence = AudioAuthenticityEvidence(
        model_repository=AASIST_REPOSITORY,
        model_revision=AASIST_REVISION,
        model_sha256=AASIST_SHA256,
        sample_rate=AASIST_SAMPLE_RATE,
        channels=1,
        window_samples=AASIST_WINDOW_SAMPLES,
        window_padding_scheme="repeat-tile",
        total_samples=total * AASIST_WINDOW_SAMPLES,
        windows=tuple(
            WindowEvidence(
                window_index=index,
                start_sample=index * AASIST_WINDOW_SAMPLES,
                end_sample=(index + 1) * AASIST_WINDOW_SAMPLES,
                padded_samples=0,
                # The largest logits are last, so a cap that ranked by magnitude would keep
                # exactly the windows this one drops.
                logits=(0.0, 99.0) if index >= total - 3 else (0.0, 0.0),
                bona_fide_logit=99.0 if index >= total - 3 else 0.0,
            )
            for index in range(total)
        ),
    )
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signal = read_signals(analysis_id)["audio_authenticity"]
    segments = read_segments(signal.id)

    assert len(segments) == detection.MAX_PERSISTED_AUDIO_WINDOWS
    assert [s.clip_index for s in segments] == list(
        range(detection.MAX_PERSISTED_AUDIO_WINDOWS)
    )
    assert signal.signal_metadata["total_audio_windows"] == total
    assert signal.signal_metadata["windows_truncated"] is True


@pytest.mark.integration
def test_a_broken_checkpoint_costs_only_the_audio_signal(queue, fake_storage, fake_aasist):
    """A model this container never received says nothing about the media."""
    analysis_id, job_id = queue()
    fake_aasist.error = AudioDetectorModelUnavailable("checkpoint is missing")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert signals["audio_authenticity"].status == "FAILED"
    # The failure kind and nothing else: the message can quote the local artifact's path.
    assert signals["audio_authenticity"].signal_metadata == {
        "error": "AudioDetectorModelUnavailable"
    }
    assert read_segments(signals["audio_authenticity"].id) == []
    # Every other source keeps its evidence, down to the rows behind it.
    assert signals["synthetic_video"].status == "SUCCESS"
    assert signals["active_speaker"].status == "SUCCESS"
    assert signals["provenance"].status == "SUCCESS"
    assert len(read_segments(signals["synthetic_video"].id)) == 2
    assert len(read_segments(signals["active_speaker"].id)) == 2


@pytest.mark.integration
def test_an_unconfigured_diarizer_does_not_cost_the_audio_windows(
    queue, fake_storage, fake_diarization
):
    """The isolation runs both ways: the checkpoint needs no token and no network."""
    analysis_id, _ = queue()
    fake_diarization.error = speaker_diarization.SpeakerDiarizationUnavailable(
        "HUGGINGFACE_TOKEN is not configured"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert signals["active_speaker"].status == "FAILED"
    assert signals["audio_authenticity"].status == "SUCCESS"
    assert len(read_segments(signals["audio_authenticity"].id)) == 3


@pytest.mark.integration
def test_the_other_evidence_rows_are_unchanged_by_the_audio_signal(queue, fake_storage):
    """The synthetic-video and active-speaker rows still look exactly as they did."""
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)
    clips = read_segments(signals["synthetic_video"].id)
    speaking = read_segments(signals["active_speaker"].id)

    assert sorted((s.clip_index, s.logit) for s in clips) == [(0, -2.25), (8, 3.5)]
    # The column the audio evidence added stays null on every row that is not audio.
    assert {s.bona_fide_logit for s in clips + speaking} == {None}
    assert [(s.start_time, s.end_time) for s in speaking] == [(0.0, 1.0), (1.0, 2.0)]
    assert [s.speaker_label for s in speaking] == ["SPEAKER_00", "SPEAKER_01"]


# P7-T3: the classification the worker takes after persisting evidence, end to end. The
# rules themselves are exercised in `test_risk_engine.py`; what these check is that a real
# job runs them at the right moment, against the evidence it just stored, and records the
# result before it says it is done.


@pytest.mark.integration
def test_a_completed_job_carries_a_risk_decision_and_its_trace(queue, fake_storage):
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    analysis = read_analysis(analysis_id)

    assert read_job(job_id).status == "completed"
    assert analysis.status == "completed"
    # The scripted detection is 0.8735, below the calibrated threshold.
    assert analysis.risk_level == "MEDIUM"
    assert analysis.risk_rule_id == "R200"
    assert analysis.risk_rules_version == "p7-v1.0.0"
    assert analysis.risk_calibration_id == (
        "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c"
    )


@pytest.mark.integration
def test_a_detection_at_or_above_the_threshold_completes_high(
    queue, fake_storage, fake_nvidia
):
    """The provider's number is what moves the band — nothing else in the job does."""
    fake_nvidia.probability = 0.9931
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    analysis = read_analysis(analysis_id)

    assert analysis.risk_level == "HIGH"
    assert analysis.risk_rule_id == "R100"
    # And the evidence behind it is stored unchanged, on NVIDIA's own scale.
    assert read_signals(analysis_id)["synthetic_video"].score == 0.9931


@pytest.mark.integration
def test_an_uncalibrated_deployment_completes_unknown(queue, fake_storage, fake_nvidia):
    """A function id that merely contains the validated one is a different deployment.

    The job still completes and the evidence is still stored — the analysis simply says
    that no validated rule could be applied to it.
    """
    fake_nvidia.probability = 0.9999
    fake_nvidia.function_id = f"{NVIDIA_FUNCTION_ID}-preview"
    analysis_id, job_id = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    analysis = read_analysis(analysis_id)

    assert read_job(job_id).status == "completed"
    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"
    signal = read_signals(analysis_id)["synthetic_video"]
    assert signal.status == "SUCCESS"
    assert signal.provider_version == f"{NVIDIA_FUNCTION_ID}-preview"


@pytest.mark.integration
def test_a_provider_failure_completes_unknown_without_losing_the_other_evidence(
    queue, fake_storage, fake_nvidia
):
    fake_nvidia.error = nvidia_video.NvidiaProviderError("nvidia refused")
    analysis_id, _ = queue()

    with SessionLocal() as session:
        worker.process_one(session)

    analysis = read_analysis(analysis_id)
    signals = read_signals(analysis_id)

    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"
    # UNKNOWN is about the direct-risk signal alone. Everything else survives intact.
    assert signals["provenance"].status == "SUCCESS"
    assert signals["active_speaker"].status == "SUCCESS"
    assert signals["audio_authenticity"].status == "SUCCESS"


@pytest.mark.integration
def test_media_that_could_not_be_transcoded_completes_unknown(
    queue, fake_storage, fake_ffmpeg
):
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg refused the media")
    analysis_id, _ = queue(was_normalized=True)

    with SessionLocal() as session:
        worker.process_one(session)

    analysis = read_analysis(analysis_id)

    assert analysis.status == "completed"
    assert analysis.risk_level == "UNKNOWN"
    assert analysis.risk_rule_id == "R010"


@pytest.mark.integration
def test_the_engine_is_run_against_the_evidence_that_was_committed(
    queue, fake_storage, monkeypatch
):
    """Persistence happens first, and the classification reads the database, not memory.

    The engine is intercepted to look the analysis up itself: whatever it sees must already
    be committed, which is only true if the evidence transaction closed before it ran.
    """
    seen = {}

    def spy(evidence):
        with SessionLocal() as reader:
            seen["committed"] = (
                reader.query(AnalysisSignal)
                .filter_by(analysis_id=analysis_id, signal_type="synthetic_video")
                .one()
            )
        seen["evidence"] = evidence
        # The real rules, reached through the module rather than the patched name.
        return risk_engine.evaluate(evidence)

    analysis_id, _ = queue()
    monkeypatch.setattr(worker, "evaluate", spy)

    with SessionLocal() as session:
        worker.process_one(session)

    # Read on a separate connection while the engine was running: the row was already there.
    assert seen["committed"].score == NVIDIA_PROBABILITY
    assert seen["evidence"].score == NVIDIA_PROBABILITY
    assert seen["evidence"].provider_version == NVIDIA_FUNCTION_ID
    assert seen["evidence"].total_clips == 7


@pytest.mark.integration
def test_a_classification_that_breaks_fails_the_job_but_keeps_the_evidence(
    queue, fake_storage, monkeypatch
):
    """The reason persistence and classification are two transactions.

    A defect in the engine must not be swallowed as a verdict, and must not cost the
    analysis the forensic evidence that was already written down.
    """
    analysis_id, job_id = queue()

    def broken(evidence):
        raise RuntimeError("the rules are broken")

    monkeypatch.setattr(worker, "evaluate", broken)

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    analysis = read_analysis(analysis_id)
    signals = read_signals(analysis_id)

    # Failed loudly, with no fabricated classification.
    assert read_job(job_id).status == "failed"
    assert read_job(job_id).error_message == "RuntimeError"
    assert analysis.status == "failed"
    assert analysis.risk_level is None
    assert analysis.risk_rules_version is None
    # And every forensic signal the job produced is still there.
    assert set(signals) == {
        "provenance",
        "synthetic_video",
        "active_speaker",
        "audio_authenticity",
    }
    assert signals["synthetic_video"].score == NVIDIA_PROBABILITY
    assert len(read_segments(signals["synthetic_video"].id)) == 2
