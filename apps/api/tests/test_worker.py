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
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError

from app import (
    detection,
    observability,
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
from app.face_detector import (
    FaceDetectorModelUnavailable,
    FaceDetectorNoFaceFound,
    FaceManipulationEvidence,
    FrameScore,
)
from app.api.analyses import active_analyses
from app.auth import generate_api_key
from app.db.models import (
    Analysis,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    ApiKey,
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

# The pinned face-manipulation artifacts, restated for the same reason the AASIST ones are:
# a classifier or locator revision changed without anyone noticing must show up here as a
# failing test rather than as a silently different measurement.
FACETORCH_REPOSITORY = "tomas-gajarsky/facetorch-deepfake-efficientnet-b7"
FACETORCH_REVISION = "4acc494f37eb63d7457166eff2acb45c5b04b9a6"
FACETORCH_SHA256 = "97b49a70174c0d4f72d9d510d817bdc49a907af9af0242a6a1ba934a7cc9e4b7"
YUNET_REPOSITORY = "opencv/opencv_zoo"
YUNET_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FACETORCH_CHECKPOINT = f"{FACETORCH_REPOSITORY}@{FACETORCH_REVISION}"

# The classifier's figure for the scripted clip. Above `T_HIGH` on purpose: the analyses
# below must still classify on NVIDIA's score alone, and a face score that would be HIGH if
# anything read it is what makes that demonstration worth anything.
FACE_SCORE = 0.9931

# The operating point R3-T1 ran its confusion matrix at. Named here only so the tests can
# assert it never reaches production evidence or a decision: it is a property of that
# benchmark, not a threshold this pipeline holds, and calibrating one is R4's work.
R3_T1_BENCHMARK_THRESHOLD = 0.8
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

    def enqueue(*, was_normalized=False, age=0, request_id=None):
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
                # The correlation id the API writes with the job (R1-T4). Null by default,
                # which is what a job queued before the column existed carries and what
                # most of this module is about — the work runs identically either way.
                request_id=request_id,
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
def fake_face_detector(monkeypatch):
    """Stand in for the local EfficientNet-B7, so no test here loads torch or 273 MiB of weights.

    One clip, one score — R3-T1's contract — over a scripted eight-frame sample where two
    frames yielded no face, which is the ordinary shape of a real reading and keeps the three
    frame counts from being interchangeable in the assertions below.
    """

    class Recorder:
        def __init__(self):
            self.error = None
            self.video_paths = []
            self.analysed_bytes = []
            self.evidence = FaceManipulationEvidence(
                classifier_repository=FACETORCH_REPOSITORY,
                classifier_revision=FACETORCH_REVISION,
                classifier_sha256=FACETORCH_SHA256,
                locator_repository=YUNET_REPOSITORY,
                locator_revision=YUNET_REVISION,
                locator_sha256=YUNET_SHA256,
                torch_version="2.13.0+cpu",
                input_size=380,
                crop_margin=1 / 3,
                face_score_threshold=0.6,
                frames_requested=8,
                frames_decoded=8,
                frames_scored=6,
                score=FACE_SCORE,
                frame_scores=tuple(
                    FrameScore(frame_index=index, probability=probability)
                    for index, probability in enumerate([0.91, 0.88, 0.95, 0.72, 0.99, 0.83])
                ),
            )

        def analyze(self, video_path, **kwargs):
            # The bytes, not only the path: the artifact is a temp file that has been
            # cleaned up by the time any assertion below runs.
            self.video_paths.append(Path(video_path))
            self.analysed_bytes.append(Path(video_path).read_bytes())
            if self.error:
                raise self.error
            return self.evidence

    recorder = Recorder()
    monkeypatch.setattr(detection, "analyze_face_manipulation", recorder.analyze)
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

    `claim_job` takes whatever is queued. Test jobs are backdated so they are always
    claimed first, but a claim that reaches past them has taken work this test does not
    own, and putting it straight back is cheaper than waiting for its lease to run out.

    The lease goes back with the status. A `queued` job carrying a deadline would be a row
    no worker holds and recovery would step over, since it only ever looks at `processing`.
    """
    if claimed is None:
        return

    with SessionLocal() as session:
        session.query(AnalysisJob).filter(AnalysisJob.id == claimed.job_id).update(
            {"status": "queued", "lease_expires_at": None}
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


@pytest.mark.integration
def test_claiming_carries_the_request_that_asked_for_the_analysis(queue):
    """The other end of the correlation the API started (R1-T4).

    The id was written onto this row by a request that finished minutes ago in another
    process. Reading it back inside the claim is what lets this worker log the analysis under
    the same id, and it is read here rather than in a second statement because the claim
    already has the row in hand.
    """
    _, job_id = queue(request_id="web-4f21ab")

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    assert claimed.job_id == job_id
    assert claimed.request_id == "web-4f21ab"


@pytest.mark.integration
def test_a_job_queued_before_correlation_existed_is_claimed_all_the_same(queue):
    """No id is not an error, and nothing is invented to stand in for one.

    Every job queued before R1-T4 carries null here, and a worker that made an id up for one
    would write a trace that correlates to nothing anywhere — worse than an absent field,
    because it looks like a trace.
    """
    queue(request_id=None)

    with SessionLocal() as session:
        claimed = worker.claim_job(session)

    assert claimed.request_id is None


@pytest.mark.integration
def test_the_whole_job_runs_under_the_requests_id(queue, fake_storage, monkeypatch):
    """Every line the worker writes about this job reports the request that submitted it.

    Asserted from inside the work rather than off a formatted log line, because what has to
    hold is that the binding is in force *while the job runs* — which is what any logger,
    including SQLAlchemy's and a library's, reads when it emits a record. Checking a
    formatted string would only prove the formatter.
    """
    bound = []
    real = worker.extract_provenance
    monkeypatch.setattr(
        worker,
        "extract_provenance",
        lambda path: (bound.append(observability.current_request_id()), real(path))[1],
    )

    queue(request_id="web-4f21ab")

    with SessionLocal() as session:
        worker.process_one(session)

    assert bound == ["web-4f21ab"]
    # And it is not still bound afterwards: this process runs job after job in one thread,
    # so a binding left standing would attribute the next job to this request.
    assert observability.current_request_id() is None


@pytest.mark.integration
def test_a_job_with_no_recorded_request_binds_nothing(queue, fake_storage, monkeypatch):
    bound = []
    real = worker.extract_provenance
    monkeypatch.setattr(
        worker,
        "extract_provenance",
        lambda path: (bound.append(observability.current_request_id()), real(path))[1],
    )

    queue(request_id=None)

    with SessionLocal() as session:
        worker.process_one(session)

    assert bound == [None]


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
        "face_manipulation",
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
def test_a_transcode_that_ran_out_of_time_fails_the_job(queue, fake_storage, fake_ffmpeg):
    analysis_id, job_id = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationTimeout("ffmpeg timed out after 900s")

    with SessionLocal() as session:
        worker.process_one(session)

    # R1-T3. A transcode that ran out of time is a statement about this worker — a machine
    # under load, or a limit set too tight for it — and not about the video, so it is not
    # written down as evidence the way a transcode ffmpeg *refused* is. The job is closed as
    # failed and the capacity it was holding is released.
    assert read_job(job_id).status == "failed"
    assert read_analysis(analysis_id).status == "failed"

    signals = read_signals(analysis_id)
    # The provenance was read off the forensic original before the transcode was attempted,
    # and it is a complete reading in its own right. It survives.
    assert set(signals) == {"provenance"}
    assert signals["provenance"].status == "SUCCESS"
    # Nothing stands in for the three signals that were never reached. A `FAILED` row would
    # be a finding — "this source was asked and had no answer" — and none of them was asked.
    assert "synthetic_video" not in signals


@pytest.mark.integration
def test_a_timed_out_transcode_records_only_the_failure_kind(
    queue, fake_storage, fake_ffmpeg
):
    _, job_id = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationTimeout(
        "ffmpeg timed out after 900s on /tmp/deepguard-job-abc123"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    # The class name and nothing else. A timeout message can quote the local artifact's
    # path, and `error_message` is a column a customer-facing surface may read.
    error_message = read_job(job_id).error_message
    assert error_message == "NormalizationTimeout"
    assert "deepguard-job" not in error_message


@pytest.mark.integration
def test_audio_extraction_that_ran_out_of_time_fails_the_job(
    queue, fake_storage, fake_audio
):
    analysis_id, job_id = queue()
    fake_audio.error = speaker_diarization.SpeakerDiarizationTimeout(
        "Audio extraction timed out after 300s"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    # The audio side of the same rule: the job is failed, because it did not finish.
    assert read_job(job_id).status == "failed"
    assert read_job(job_id).error_message == "SpeakerDiarizationTimeout"
    assert read_analysis(analysis_id).status == "failed"

    signals = read_signals(analysis_id)
    # But the readings taken before the extraction was attempted are complete and independent
    # of it, so they are kept rather than discarded with the job. Only the two signals the
    # timeout actually prevented are absent.
    #
    # The face-manipulation reading is among the survivors because it runs before the audio
    # chain and needs nothing from it — which is the whole reason it was placed there.
    assert set(signals) == {"provenance", "synthetic_video", "face_manipulation"}
    assert "active_speaker" not in signals
    assert "audio_authenticity" not in signals


@pytest.mark.integration
def test_evidence_produced_before_a_timeout_survives_it(
    queue, fake_storage, fake_audio, fake_nvidia
):
    analysis_id, job_id = queue()
    fake_audio.error = speaker_diarization.SpeakerDiarizationTimeout(
        "Audio extraction timed out after 300s"
    )

    with SessionLocal() as session:
        worker.process_one(session)

    # The regression this test exists for: NVIDIA answered, and the answer is a complete,
    # self-contained reading of this media (rule 11). A later timeout in the unrelated audio
    # chain does not make it less true, and the job failing must not take it down with it.
    signal = read_signals(analysis_id)["synthetic_video"]
    assert signal.status == "SUCCESS"
    assert signal.score == NVIDIA_PROBABILITY
    assert signal.provider_version == NVIDIA_FUNCTION_ID
    assert signal.signal_metadata["logit"] == NVIDIA_LOGIT

    # Its clip evidence comes with it. Segments hang off their signal's id, so a partial
    # write that dropped them would leave a scored signal with nothing behind it.
    assert read_segments(signal.id)

    # And the analysis is still failed and still unclassified. Keeping the evidence must not
    # turn a job that did not finish into one that did, and no rule was run over a partial
    # evidence set: a null risk level is the absence of a conclusion, not `UNKNOWN`, which
    # is a conclusion an explicit rule reached.
    analysis = read_analysis(analysis_id)
    assert analysis.status == "failed"
    assert analysis.risk_level is None
    assert analysis.risk_rule_id is None
    assert read_job(job_id).error_message == "SpeakerDiarizationTimeout"


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
def test_all_five_evidence_sources_are_persisted_as_independent_signals(
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
        "face_manipulation",
        "provenance",
        "synthetic_video",
    ]

    face = signals["face_manipulation"]
    # A fifth row, not a field on somebody else's. It reads the same artifact NVIDIA's
    # synthetic-video detector does and shares nothing else with it: a different provider, a
    # different question, a different unit, its own status and its own id.
    assert face.provider == "efficientnet-b7"
    assert face.id != signals["synthetic_video"].id
    assert face.status == "SUCCESS"
    assert face.provider_version == FACETORCH_CHECKPOINT
    # The model's own probability, stored untransformed.
    assert face.score == FACE_SCORE
    # And no verdict beside it. Risk is a decision about the analysis under a named ruleset,
    # never a label per provider.
    assert face.risk_level is None
    # It owns no segments: R3-T1's contract is one clip to one score, and the per-frame
    # probabilities the mean was taken over are metadata that documents the score rather
    # than a timeline of detections.
    assert read_segments(face.id) == []
    assert face.signal_metadata["classifier_revision"] == FACETORCH_REVISION
    assert face.signal_metadata["classifier_sha256"] == FACETORCH_SHA256
    assert face.signal_metadata["locator_revision"] == YUNET_REVISION
    assert face.signal_metadata["frames_requested"] == 8
    assert face.signal_metadata["frames_decoded"] == 8
    assert face.signal_metadata["frames_scored"] == 6
    assert len(face.signal_metadata["frame_scores"]) == 6
    # The R3-T1 benchmark operating point is nowhere in the stored evidence, under any name.
    # It was a property of that measurement over 40 clips of one corpus, and persisting it
    # beside the score is how it would quietly become the production threshold this task
    # exists to not have. `face_score_threshold` is the locator's confidence floor — how
    # sure YuNet has to be that it found a face — and is unrelated to the classifier's scale.
    assert R3_T1_BENCHMARK_THRESHOLD not in face.signal_metadata.values()
    assert face.signal_metadata["face_score_threshold"] == 0.6

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
def test_a_clip_with_no_face_is_a_failed_signal_and_not_a_score(
    queue, fake_storage, fake_face_detector
):
    """The abstention, persisted as one.

    A clip the locator found no face in is an ordinary property of an upload — a landscape,
    a screen recording, a wide crowd shot. The classifier was never asked, so the row carries
    no number: a low score here would be a fabricated finding about media nothing looked at,
    and it would read as evidence of authenticity.
    """
    analysis_id, job_id = queue()
    fake_face_detector.error = FaceDetectorNoFaceFound("no face in any sampled frame")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)
    face = signals["face_manipulation"]

    assert face.status == "FAILED"
    assert face.score is None
    assert face.provider_version is None
    # The failure kind and nothing else — never the message, which quotes the local path.
    assert face.signal_metadata == {"error": "FaceDetectorNoFaceFound"}

    # And the job still finished, with every other reading intact.
    assert read_job(job_id).status == "completed"
    assert signals["synthetic_video"].status == "SUCCESS"
    assert signals["provenance"].status == "SUCCESS"
    assert signals["audio_authenticity"].status == "SUCCESS"


@pytest.mark.integration
def test_a_broken_face_classifier_costs_only_the_face_signal(
    queue, fake_storage, fake_face_detector
):
    """One detector failing must never destroy the analysis (AGENTS.md, error-handling rule)."""
    analysis_id, job_id = queue()
    fake_face_detector.error = FaceDetectorModelUnavailable("weights are missing")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert signals["face_manipulation"].status == "FAILED"
    assert signals["face_manipulation"].signal_metadata == {
        "error": "FaceDetectorModelUnavailable"
    }
    assert signals["synthetic_video"].status == "SUCCESS"
    assert signals["synthetic_video"].score == NVIDIA_PROBABILITY
    assert signals["audio_authenticity"].status == "SUCCESS"
    assert signals["provenance"].status == "SUCCESS"


@pytest.mark.integration
def test_the_face_classifier_reads_the_same_artifact_the_video_detector_does(
    queue, fake_storage, fake_face_detector, fake_nvidia
):
    """One preparation serves them both, and neither transcodes a second copy for itself."""
    queue()

    with SessionLocal() as session:
        worker.process_one(session)

    assert fake_face_detector.analysed_bytes == [ORIGINAL_BYTES]
    # The same artifact NVIDIA was handed, not a second copy prepared for this detector.
    assert fake_face_detector.analysed_bytes == fake_nvidia.analysed_bytes


@pytest.mark.integration
def test_the_face_score_does_not_move_the_risk_decision(queue, fake_storage, fake_nvidia):
    """The R3 constraint, at the level the decision is actually taken.

    NVIDIA's score is held at a value that bands MEDIUM while the face classifier reports
    0.9931 — comfortably above `T_HIGH`, the boundary the calibrated signal is banded on. The
    analysis must still be MEDIUM by `R200`: the face score is raw evidence in R3, the engine
    reads one provider and one signal type, and calibrating this one is R4's work.
    """
    analysis_id, _ = queue()
    fake_nvidia.probability = 0.4646

    with SessionLocal() as session:
        worker.process_one(session)

    analysis = read_analysis(analysis_id)
    signals = read_signals(analysis_id)

    assert signals["face_manipulation"].score == FACE_SCORE
    assert signals["face_manipulation"].score > 0.98
    assert analysis.risk_level == "MEDIUM"
    assert analysis.risk_rule_id == "R200"


@pytest.mark.integration
def test_media_that_cannot_be_transcoded_leaves_no_face_signal_at_all(
    queue, fake_storage, fake_ffmpeg, fake_face_detector
):
    """A detector that was never invoked leaves no row — not a `FAILED` one.

    The two states must not be merged. A `FAILED` face-manipulation signal is a finding:
    the classifier ran and could not produce evidence, which is what a missing checkpoint,
    an unverified torch or a clip with no face in it each record. Nothing of the sort
    happened here — the transcode raised before the call that runs the classifier was ever
    reached — so writing a failure would be recording a finding nobody made, and would put
    this detector's name on a gap it had no part in.

    The other three artifact-dependent signals *are* written as `FAILED` on this path, and
    that asymmetry is the point rather than an inconsistency: each of those sources was
    genuinely asked about this media and had no answer, because NVIDIA takes MP4 and
    nothing else and the audio is demuxed from the same derivative that never existed.
    """
    analysis_id, job_id = queue(was_normalized=True)
    fake_ffmpeg.error = normalization.NormalizationError("ffmpeg refused the source")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    # The row is absent, and the classifier is what was never invoked to produce it.
    assert "face_manipulation" not in signals
    assert fake_face_detector.video_paths == []

    # The sources that *were* asked keep their failures, and the provenance read off the
    # forensic original before any of this survives untouched.
    assert signals["synthetic_video"].status == "FAILED"
    assert signals["active_speaker"].status == "FAILED"
    assert signals["audio_authenticity"].status == "FAILED"
    assert signals["provenance"].status == "SUCCESS"


@pytest.mark.integration
def test_a_failed_face_signal_always_means_the_classifier_ran(
    queue, fake_storage, fake_face_detector
):
    """The other half of the rule above, so the two cannot drift into meaning one thing.

    Same media, same job, and this time the classifier is reached and fails. Here a row is
    exactly what must appear — the reading was attempted and produced no evidence — which is
    what makes its absence above a statement about reachability rather than about failure.
    """
    analysis_id, job_id = queue()
    fake_face_detector.error = FaceDetectorModelUnavailable("weights are missing")

    with SessionLocal() as session:
        worker.process_one(session)

    signals = read_signals(analysis_id)

    assert read_job(job_id).status == "completed"
    assert fake_face_detector.video_paths != []
    assert signals["face_manipulation"].status == "FAILED"
    assert signals["face_manipulation"].score is None
    assert signals["face_manipulation"].signal_metadata == {
        "error": "FaceDetectorModelUnavailable"
    }


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
        "face_manipulation",
    }
    assert signals["synthetic_video"].score == NVIDIA_PROBABILITY
    assert len(read_segments(signals["synthetic_video"].id)) == 2


# --- stale job recovery (P9-F1) -----------------------------------------------------------
#
# A worker that dies mid-job used to leave it `processing` forever. Nothing went back for it,
# the analysis stayed `queued`, and after P9 that permanently consumed one of an API key's
# five concurrency slots — five crashes and a customer was locked out of the public API with
# no way to clear it.
#
# The fix is a lease, and the tests below are in two halves for a reason. Recovery is easy to
# write and easy to get subtly wrong in two ways: failing a job whose worker is alive and
# merely slow, and letting that worker come back and undo the recovery. Both halves are here.


@pytest.fixture
def api_key(database):
    """One real API key, so a recovered slot can be counted rather than asserted about.

    Deliberately not part of `queue`: it is torn down after it, because `analyses.api_key_id`
    is `ON DELETE RESTRICT` and a key removed while an analysis still pointed at it would
    fail. A test that wants both asks for this one first, and pytest unwinds in reverse.
    """
    generated = generate_api_key()
    with SessionLocal() as session:
        key = ApiKey(name="stale-recovery", key_hash=generated.key_hash)
        session.add(key)
        session.commit()
        key_id = key.id

    yield key_id

    with SessionLocal() as session:
        session.query(ApiKey).filter(ApiKey.id == key_id).delete()
        session.commit()


def own(analysis_id, key_id) -> None:
    """Attribute an analysis to an API key, as a public submission would."""
    with SessionLocal() as session:
        session.query(Analysis).filter(Analysis.id == analysis_id).update(
            {"api_key_id": key_id}
        )
        session.commit()


def expire_lease(job_id, seconds_ago=1) -> None:
    """Put a claimed job's deadline in the past, as a dead worker's would drift.

    Written relative to `now()` in the database rather than to a Python timestamp, for the
    same reason the worker writes it that way: this is the clock the comparison uses.
    """
    with SessionLocal() as session:
        session.execute(
            AnalysisJob.__table__.update()
            .where(AnalysisJob.id == job_id)
            .values(lease_expires_at=func.now() - timedelta(seconds=seconds_ago))
        )
        session.commit()


def recover() -> int:
    with SessionLocal() as session:
        return worker.recover_stale_jobs(session)


def claim_one():
    """Claim whatever this test just queued, on a session that is then let go."""
    with SessionLocal() as session:
        return worker.claim_job(session)


@pytest.mark.integration
def test_claiming_a_job_starts_its_lease(queue):
    """A claim without a deadline is the bug this whole mechanism exists to close.

    `NULL < now()` is null, so a `processing` row with no lease is one recovery can never
    reach — exactly the job that most needs reaching.
    """
    _, job_id = queue()

    claimed = claim_one()

    assert claimed.job_id == job_id
    job = read_job(job_id)
    assert job.status == "processing"
    assert job.lease_expires_at is not None
    # Ahead of the clock, which is what "leased" means. Compared against the database's own
    # `now()`, not this process's.
    assert job.lease_expires_at > datetime.now(timezone.utc)


@pytest.mark.integration
def test_a_queued_job_carries_no_lease(queue):
    """Only a claimed job is anybody's. A deadline on an unclaimed row would say otherwise."""
    _, job_id = queue()

    assert read_job(job_id).lease_expires_at is None


@pytest.mark.integration
def test_an_expired_lease_fails_the_job_and_its_analysis(queue):
    analysis_id, job_id = queue()
    claim_one()
    expire_lease(job_id)

    assert recover() >= 1

    job = read_job(job_id)
    analysis = read_analysis(analysis_id)
    assert job.status == "failed"
    assert job.error_message == worker.STALE_LEASE_ERROR
    # The parent analysis moves with it. Left `queued`, it would look like work still
    # coming and would go on holding its API key's slot — the whole point of this task.
    assert analysis.status == "failed"


@pytest.mark.integration
def test_a_recovered_analysis_records_no_classification(queue):
    """Nothing classified this analysis, and null says so.

    `UNKNOWN` would be a conclusion an explicit rule reached. A worker that vanished reached
    no conclusion at all, and inventing a risk level for it would put a fabricated verdict
    in a forensic record.
    """
    analysis_id, job_id = queue()
    claim_one()
    expire_lease(job_id)

    recover()

    analysis = read_analysis(analysis_id)
    assert analysis.risk_level is None
    assert analysis.risk_rules_version is None
    assert analysis.risk_rule_id is None
    assert analysis.risk_calibration_id is None


@pytest.mark.integration
def test_a_recovered_job_keeps_no_lease(queue):
    """A terminal row carrying a deadline would claim a worker is still running it."""
    _, job_id = queue()
    claim_one()
    expire_lease(job_id)

    recover()

    assert read_job(job_id).lease_expires_at is None


@pytest.mark.integration
def test_recovery_releases_the_api_key_concurrency_slot(api_key, queue):
    """The objective, stated as the public API sees it.

    A crashed worker's analysis stays `queued`, and `active_analyses` counts exactly the
    `queued` ones — so before recovery this key has a slot consumed by work nobody is doing,
    and after it the key is free again. This is the check that would have caught P9's
    reported limitation.
    """
    analysis_id, job_id = queue()
    own(analysis_id, api_key)
    claim_one()
    expire_lease(job_id)

    with SessionLocal() as reader:
        assert active_analyses(reader, api_key) == 1

    recover()

    with SessionLocal() as reader:
        assert active_analyses(reader, api_key) == 0


@pytest.mark.integration
def test_a_live_lease_is_left_alone(queue):
    """The failure mode that would be worse than the bug.

    A four-minute video is not a crashed worker. If recovery could not tell them apart it
    would fail real analyses in flight, and a customer would rather have a slot held than a
    result thrown away.
    """
    analysis_id, job_id = queue()
    claim_one()

    recover()

    assert read_job(job_id).status == "processing"
    assert read_analysis(analysis_id).status == "queued"


@pytest.mark.integration
def test_staleness_is_not_read_off_updated_at(queue):
    """The explicit constraint, as a test.

    `updated_at` is ancient here and the lease is live — which is exactly the shape of a
    real long job, because the middle of an analysis writes nothing for minutes at a time.
    An age-based rule would fail this job. The lease does not, because it is a promise about
    the future rather than a record of the past.
    """
    _, job_id = queue()
    claim_one()

    with SessionLocal() as session:
        # Straight past the ORM's `onupdate`, which would otherwise refresh the very column
        # being backdated.
        session.execute(
            text("UPDATE analysis_jobs SET updated_at = now() - interval '1 day' WHERE id = :id"),
            {"id": job_id},
        )
        session.commit()

    job = read_job(job_id)
    assert job.updated_at < datetime.now(timezone.utc) - timedelta(hours=1)

    recover()

    assert read_job(job_id).status == "processing"


@pytest.mark.integration
def test_a_queued_job_is_never_recovered(queue):
    """Recovery only ever looks at `processing`. A queued job is waiting, not abandoned."""
    analysis_id, job_id = queue()

    recover()

    assert read_job(job_id).status == "queued"
    assert read_analysis(analysis_id).status == "queued"


@pytest.mark.integration
def test_a_finished_job_is_never_recovered(queue, fake_storage):
    """A completed job has no lease to expire, and recovery must not reopen it."""
    analysis_id, job_id = queue()
    with SessionLocal() as session:
        assert worker.process_one(session) is True

    assert read_job(job_id).status == "completed"
    # The lease ends with the job, so there is nothing left for recovery to match on.
    assert read_job(job_id).lease_expires_at is None

    recover()

    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"


@pytest.mark.integration
def test_the_heartbeat_pushes_the_deadline_forward(queue):
    """What keeps a legitimately long analysis alive."""
    _, job_id = queue()
    claim_one()
    expire_lease(job_id)

    with SessionLocal() as session:
        assert worker.renew_lease(session, job_id) is True

    assert read_job(job_id).lease_expires_at > datetime.now(timezone.utc)
    # And a renewed job is no longer stale, which is the point of renewing it.
    recover()
    assert read_job(job_id).status == "processing"


@pytest.mark.integration
def test_the_heartbeat_cannot_revive_a_recovered_job(queue):
    """A worker that comes back must not take its job off the recovery list.

    If renewal were unconditional, a paused container waking up would push the deadline
    forward on a job already failed and start heartbeating a corpse — leaving a `failed`
    row that looks leased and, worse, telling the worker it still owned the job.
    """
    _, job_id = queue()
    claim_one()
    expire_lease(job_id)
    recover()

    with SessionLocal() as session:
        assert worker.renew_lease(session, job_id) is False

    job = read_job(job_id)
    assert job.status == "failed"
    assert job.lease_expires_at is None


@pytest.mark.integration
def test_a_recovered_job_cannot_be_completed_by_its_old_worker(queue, fake_storage):
    """The resurrection case, and the one that makes recovery mean anything.

    The worker did all the work and is about to publish a verdict when it discovers it was
    declared dead. It must not overwrite `failed` with `completed`: the analysis has already
    been reported failed and its concurrency slot already handed back, and taking either of
    those back after the fact would make every recovery provisional.
    """
    analysis_id, job_id = queue()
    claimed = claim_one()
    expire_lease(job_id)
    recover()

    with SessionLocal() as session:
        # Exactly what the old worker would do next, on evidence it really produced.
        with worker.fetched_artifact(claimed.original_storage_key) as original:
            provenance = detection.extract_provenance(original)
            evidence = worker.analyse(claimed, original)
        decision = worker.complete_job(session, claimed, evidence, provenance)

    # It is told it lost, rather than silently succeeding.
    assert decision is None

    job = read_job(job_id)
    analysis = read_analysis(analysis_id)
    assert job.status == "failed"
    assert job.error_message == worker.STALE_LEASE_ERROR
    assert analysis.status == "failed"
    # No verdict was published on an analysis this worker no longer owned.
    assert analysis.risk_level is None


@pytest.mark.integration
def test_a_recovered_job_keeps_the_evidence_its_old_worker_committed(queue, fake_storage):
    """Losing the job costs the verdict, not the forensic record.

    The signals are independent evidence of what was genuinely observed about the media
    (AGENTS.md rule 11). Deleting real findings because a scheduling event overtook the
    worker that produced them would destroy evidence to tidy up.
    """
    analysis_id, job_id = queue()
    claimed = claim_one()
    expire_lease(job_id)
    recover()

    with SessionLocal() as session:
        with worker.fetched_artifact(claimed.original_storage_key) as original:
            provenance = detection.extract_provenance(original)
            evidence = worker.analyse(claimed, original)
        worker.complete_job(session, claimed, evidence, provenance)

    assert "synthetic_video" in read_signals(analysis_id)


@pytest.mark.integration
def test_a_recovered_job_is_not_relabelled_by_its_old_worker(queue):
    """A worker that crashes *after* being recovered must not rewrite the reason.

    Both outcomes are `failed`, so nothing is at stake but the explanation — and
    `StaleWorkerLease` is what actually happened, while whatever the dying worker tripped
    over on its way out is a symptom of it.
    """
    _, job_id = queue()
    claimed = claim_one()
    expire_lease(job_id)
    recover()

    with SessionLocal() as session:
        worker.fail_job(session, claimed, RuntimeError("storage went away"))

    assert read_job(job_id).error_message == worker.STALE_LEASE_ERROR


@pytest.mark.integration
def test_racing_workers_recover_a_job_exactly_once(queue):
    """Several workers, one stale job, all reaching for it together.

    Recovery has no `SKIP LOCKED`: the second worker blocks on the first's row lock and then
    re-checks the `WHERE` clause against the committed row, where the status is no longer
    `processing`. Exactly one of them may report having recovered it — two would mean two
    workers each believing they had freed a slot, and a count that could be double-released.
    """
    # Clear anything an earlier test left stale, so the counts below are about this job.
    recover()

    _, job_id = queue()
    claim_one()
    expire_lease(job_id)

    start = threading.Barrier(4)
    counts = []
    guard = threading.Lock()

    def work():
        start.wait(timeout=10)
        with SessionLocal() as session:
            recovered = worker.recover_stale_jobs(session)
        with guard:
            counts.append(recovered)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sum(counts) == 1
    assert sorted(counts) == [0, 0, 0, 1]
    assert read_job(job_id).status == "failed"


@pytest.mark.integration
def test_the_loop_recovers_stale_work_before_claiming_new_work(queue, fake_storage):
    """Recovery rides the existing poll, and runs first.

    Running it first is what lets one pass both release a slot and use it. It also means no
    scheduler, no cron and no second process — the loop was already going to ask the
    database for work.
    """
    stale_analysis, stale_job = queue(age=0)
    claim_one()
    expire_lease(stale_job)

    fresh_analysis, fresh_job = queue(age=1)

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    # The abandoned job was failed on the way past...
    assert read_job(stale_job).status == "failed"
    assert read_analysis(stale_analysis).status == "failed"
    # ...and the same pass went on to do real work.
    assert read_job(fresh_job).status == "completed"
    assert read_analysis(fresh_analysis).status == "completed"


@pytest.mark.integration
def test_a_job_outliving_its_lease_is_kept_alive_by_its_heartbeat(queue, fake_storage, monkeypatch):
    """End to end: an analysis longer than the lease still completes.

    The lease is shortened to a second and the heartbeat to well under it, then the work is
    made to take several times the lease. Without renewal the job would be recovered out
    from under itself and finish `failed`; with it, the deadline is pushed forward
    throughout and the job completes normally.
    """
    monkeypatch.setattr(worker, "LEASE_SECONDS", 1)
    monkeypatch.setattr(worker, "HEARTBEAT_SECONDS", 0.2)

    analysis_id, job_id = queue()
    deadlines = []
    real_analyse = worker.analyse

    def slow_analyse(claimed, original):
        # Long enough to outlive several leases, and sampled while it runs so the renewal
        # is observed rather than inferred from the outcome.
        for _ in range(15):
            time.sleep(0.1)
            deadlines.append(read_job(job_id).lease_expires_at)
        # Recovery is running on this job's own poll interval in production; here it is
        # invoked directly, mid-job, which is the harshest version of the same question.
        recover()
        return real_analyse(claimed, original)

    monkeypatch.setattr(worker, "analyse", slow_analyse)

    with SessionLocal() as session:
        assert worker.process_one(session) is True

    # The work took far longer than one lease...
    assert len(deadlines) == 15
    # ...the deadline moved while it ran...
    assert max(deadlines) > min(deadlines)
    # ...and a recovery pass that ran mid-job left it alone.
    assert read_job(job_id).status == "completed"
    assert read_analysis(analysis_id).status == "completed"
