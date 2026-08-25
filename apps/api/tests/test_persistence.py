"""Persistence proof against real PostgreSQL.

The rest of the suite fakes the session, which proves what the route does but not that
the schema exists or that PostgreSQL accepts and returns these values. This module runs
against the database from docker-compose: it needs the Alembic migration to have been
applied (`alembic upgrade head`) and skips when no database is reachable.
"""

import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.api.analyses import DASHBOARD_SEGMENTS, RECENT_ANALYSES_LIMIT
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_PROCESSING,
    JOB_STATUS_QUEUED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    Analysis,
    AnalysisJob,
    AnalysisSegment,
    AnalysisSignal,
    MediaFile,
)
from app.db.session import SessionLocal, engine
from app.main import app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def session(database):
    """A real session whose rows are removed again, so the database is left as found."""
    created = []

    with SessionLocal() as session:
        yield session, created

        session.rollback()
        for analysis_id in created:
            # The media row goes with it through the foreign key's ON DELETE CASCADE.
            session.query(Analysis).filter(Analysis.id == analysis_id).delete()
        session.commit()


def media_file(analysis_id: uuid.UUID, **overrides) -> MediaFile:
    """A media row shaped exactly like the one the upload pipeline persists."""
    values = {
        "analysis_id": analysis_id,
        "original_filename": "clip.mov",
        "content_type": "video/quicktime",
        "size_bytes": 4096,
        "original_sha256": hashlib.sha256(b"original").hexdigest(),
        "original_storage_key": f"originals/{hashlib.sha256(b'original').hexdigest()}",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "duration": 12.34,
        "frame_rate": 30000 / 1001,
        "pix_fmt": "yuv420p",
        "constant_frame_rate": True,
        "was_normalized": True,
        "derivative_storage_key": f"derivatives/{hashlib.sha256(b'derivative').hexdigest()}.mp4",
        "derivative_sha256": hashlib.sha256(b"derivative").hexdigest(),
    }

    return MediaFile(**{**values, **overrides})


def test_migration_created_the_analysis_schema(database):
    inspector = inspect(database)

    assert {
        "analyses",
        "media_files",
        "analysis_jobs",
        "analysis_signals",
        "analysis_segments",
    } <= set(inspector.get_table_names())
    # The risk columns are the decision trace P7-T3 added, and they are the whole of it:
    # no score, no threshold, no clip count is copied here from the evidence rows.
    assert {column["name"] for column in inspector.get_columns("analyses")} == {
        "id",
        "status",
        "created_at",
        "risk_level",
        "risk_rules_version",
        "risk_calibration_id",
        "risk_rule_id",
    }
    assert {column["name"] for column in inspector.get_columns("media_files")} == {
        "id",
        "analysis_id",
        "original_filename",
        "content_type",
        "size_bytes",
        "original_sha256",
        "original_storage_key",
        "format_name",
        "codec_name",
        "width",
        "height",
        "duration",
        "frame_rate",
        "pix_fmt",
        "constant_frame_rate",
        "was_normalized",
        "derivative_storage_key",
        "derivative_sha256",
    }
    assert {column["name"] for column in inspector.get_columns("analysis_jobs")} == {
        "id",
        "analysis_id",
        "status",
        "error_message",
        "created_at",
        "updated_at",
    }
    assert {column["name"] for column in inspector.get_columns("analysis_signals")} == {
        "id",
        "analysis_id",
        "provider",
        "signal_type",
        "score",
        "risk_level",
        "provider_version",
        "status",
        "metadata",
        "created_at",
    }
    # Evidence rows carry what their source reports and nothing else: no score and no
    # risk_level anywhere. A clip logit is not a probability (D019), and neither is an
    # audio one.
    assert {column["name"] for column in inspector.get_columns("analysis_segments")} == {
        "id",
        "signal_id",
        # Clip evidence, from the synthetic-video detector. Audio window evidence reuses
        # `clip_index` for the window's place in the sequence and fills both logits.
        "clip_index",
        "logit",
        "bona_fide_logit",
        # Temporal evidence, from Active Speaker Detection, and the preprocessing bounds of
        # an audio window. A row fills the group its source has facts for; none is required,
        # because no source reports another's facts.
        "start_time",
        "end_time",
        "face_id",
        "speaker_label",
        "created_at",
    }


def test_neither_kind_of_segment_evidence_is_required(database):
    """Clip columns stopped being mandatory when a second source began writing rows.

    They were `NOT NULL` while clip evidence was the only evidence there was. An
    active-speaker row has no clip and no logit, and requiring them would have forced a
    placeholder figure into a column that is supposed to hold provider output.
    """
    columns = {
        column["name"]: column for column in inspect(database).get_columns("analysis_segments")
    }

    assert columns["clip_index"]["nullable"] is True
    assert columns["logit"]["nullable"] is True
    assert columns["start_time"]["nullable"] is True
    assert columns["end_time"]["nullable"] is True
    assert columns["face_id"]["nullable"] is True
    assert columns["speaker_label"]["nullable"] is True
    # Only AASIST emits a second figure for one scored unit; every other source leaves it
    # empty rather than repeating the first.
    assert columns["bona_fide_logit"]["nullable"] is True


def test_analysis_and_media_round_trip_through_postgresql(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    original = media_file(analysis.id)
    db.add(original)
    db.commit()

    assert isinstance(analysis.id, uuid.UUID)
    # Read back through a second session, so the assertions cannot be served from the
    # identity map of the session that wrote them.
    with SessionLocal() as reader:
        stored_analysis = reader.get(Analysis, analysis.id)
        stored_media = (
            reader.query(MediaFile).filter(MediaFile.analysis_id == analysis.id).one()
        )

        assert stored_analysis.status == ANALYSIS_STATUS_COMPLETED
        # The timestamp comes from the database default, not the application.
        assert stored_analysis.created_at is not None
        assert stored_analysis.created_at.tzinfo is not None

        assert stored_media.analysis_id == analysis.id
        assert stored_media.original_filename == "clip.mov"
        assert stored_media.content_type == "video/quicktime"
        assert stored_media.size_bytes == 4096
        assert stored_media.original_sha256 == original.original_sha256
        assert stored_media.original_storage_key == original.original_storage_key
        assert stored_media.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
        assert stored_media.codec_name == "h264"
        assert stored_media.width == 1920
        assert stored_media.height == 1080
        assert stored_media.duration == pytest.approx(12.34)
        # A fractional rate must survive the round trip, not be rounded to 30.
        assert stored_media.frame_rate == pytest.approx(30000 / 1001, rel=1e-12)
        assert stored_media.pix_fmt == "yuv420p"
        assert stored_media.constant_frame_rate is True
        assert stored_media.was_normalized is True
        assert stored_media.derivative_storage_key == original.derivative_storage_key
        assert stored_media.derivative_sha256 == original.derivative_sha256


def test_a_media_row_may_be_committed_before_its_derivative_exists(session):
    """The schema has to be able to say "a derivative is owed but does not exist yet".

    Since P4-F2 the transcode runs in the worker, so an upload that needs a derivative
    commits before there is one. A NOT NULL column would have forced a placeholder key
    naming an object nobody ever stored.
    """
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(
        media_file(
            analysis.id,
            was_normalized=True,
            derivative_storage_key=None,
            derivative_sha256=None,
        )
    )
    db.commit()

    with SessionLocal() as reader:
        stored = reader.query(MediaFile).filter_by(analysis_id=analysis.id).one()
        assert stored.was_normalized is True
        assert stored.derivative_storage_key is None
        assert stored.derivative_sha256 is None


def test_media_without_a_derivative_persists_null(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    key = f"originals/{hashlib.sha256(b'canonical').hexdigest()}"
    db.add(
        media_file(
            analysis.id,
            was_normalized=False,
            derivative_storage_key=key,
            derivative_sha256=None,
        )
    )
    db.commit()

    with SessionLocal() as reader:
        stored = reader.query(MediaFile).filter(MediaFile.analysis_id == analysis.id).one()

        assert stored.was_normalized is False
        assert stored.derivative_sha256 is None


def test_the_same_media_can_be_analysed_more_than_once(session):
    db, created = session
    first = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    second = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add_all([first, second])
    db.flush()
    created.extend([first.id, second.id])

    # Identical bytes resolve to the same content-addressed keys and hashes; the two
    # analyses of them stay independently identifiable.
    db.add_all([media_file(first.id), media_file(second.id)])
    db.commit()

    assert first.id != second.id
    with SessionLocal() as reader:
        stored = reader.query(MediaFile).filter(
            MediaFile.original_storage_key == media_file(first.id).original_storage_key
        )
        assert stored.count() == 2


def test_persisted_analyses_can_be_read_back_through_the_listing_endpoint(session):
    """The dashboard's read path, end to end against real PostgreSQL.

    No session is overridden here: the request opens its own connection to the same
    database, so this proves a committed row is readable by a later request rather than
    only visible inside the writing session.
    """
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id, original_filename="listed.mov"))
    db.commit()

    with TestClient(app) as client:
        response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    body = response.json()
    assert len(body) <= RECENT_ANALYSES_LIMIT

    listed = next(row for row in body if row["id"] == str(analysis.id))
    assert listed["status"] == ANALYSIS_STATUS_COMPLETED
    assert listed["original_filename"] == "listed.mov"
    assert listed["declared_content_type"] == "video/quicktime"
    assert listed["size_bytes"] == 4096
    assert listed["original_sha256"] == hashlib.sha256(b"original").hexdigest()
    assert listed["was_normalized"] is True
    # The probed facts survive the round trip through real columns, so the container and
    # codec evidence the dashboard renders is the evidence the database holds.
    assert listed["media"] == {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "duration": pytest.approx(12.34),
        "frame_rate": pytest.approx(30000 / 1001, rel=1e-12),
        "pix_fmt": "yuv420p",
        "constant_frame_rate": True,
    }
    # The timestamp survives as a real value from the database default.
    assert listed["created_at"] is not None


def test_the_listing_returns_the_most_recent_analysis_first(session):
    db, created = session
    older = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(older)
    db.flush()
    db.add(media_file(older.id))
    db.commit()

    newer = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(newer)
    db.flush()
    db.add(media_file(newer.id))
    db.commit()
    created.extend([older.id, newer.id])

    with TestClient(app) as client:
        body = client.get("/api/v1/analyses").json()

    ids = [row["id"] for row in body]
    assert ids.index(str(newer.id)) < ids.index(str(older.id))


def queued_analysis(db, created) -> Analysis:
    """An analysis and its media, committed exactly as the upload route commits them."""
    analysis = Analysis(status=ANALYSIS_STATUS_QUEUED)
    db.add(analysis)
    db.flush()
    db.add(media_file(analysis.id))
    created.append(analysis.id)

    return analysis


def test_a_queued_job_round_trips_through_postgresql(session):
    db, created = session
    analysis = queued_analysis(db, created)
    db.add(AnalysisJob(analysis_id=analysis.id, status=JOB_STATUS_QUEUED))
    db.commit()

    with SessionLocal() as reader:
        stored = reader.query(AnalysisJob).filter_by(analysis_id=analysis.id).one()

        assert stored.status == JOB_STATUS_QUEUED
        # Nothing has failed, so there is nothing to say about a failure.
        assert stored.error_message is None
        # Both stamps are the database's own, not the application's.
        assert stored.created_at is not None
        assert stored.updated_at is not None


def test_an_analysis_can_only_be_queued_once(session):
    db, created = session
    analysis = queued_analysis(db, created)
    db.add(AnalysisJob(analysis_id=analysis.id, status=JOB_STATUS_QUEUED))
    db.commit()

    db.add(AnalysisJob(analysis_id=analysis.id, status=JOB_STATUS_QUEUED))

    # Enforced by the database, not merely assumed by the code that writes it: two jobs
    # for one analysis would make "has this been detected?" two different answers.
    with pytest.raises(IntegrityError):
        db.commit()


def test_a_job_cannot_name_an_analysis_that_does_not_exist(session):
    db, _ = session
    db.add(AnalysisJob(analysis_id=uuid.uuid4(), status=JOB_STATUS_QUEUED))

    # Queued work pointing at nothing would be a runner's job to fail on forever.
    with pytest.raises(IntegrityError):
        db.commit()


def test_a_failed_job_persists_why_it_failed(session):
    db, created = session
    analysis = queued_analysis(db, created)
    db.add(
        AnalysisJob(
            analysis_id=analysis.id,
            status="failed",
            error_message="NVIDIA rejected the request (UNAUTHENTICATED)",
        )
    )
    db.commit()

    with SessionLocal() as reader:
        stored = reader.query(AnalysisJob).filter_by(analysis_id=analysis.id).one()

        # Diagnostic text of unbounded length, kept as written rather than truncated.
        assert stored.error_message == "NVIDIA rejected the request (UNAUTHENTICATED)"


def test_moving_a_job_on_touches_its_updated_at(session):
    db, created = session
    analysis = queued_analysis(db, created)
    job = AnalysisJob(analysis_id=analysis.id, status=JOB_STATUS_QUEUED)
    db.add(job)
    db.commit()
    queued_at = job.updated_at

    job.status = JOB_STATUS_PROCESSING
    db.commit()

    with SessionLocal() as reader:
        stored = reader.query(AnalysisJob).filter_by(analysis_id=analysis.id).one()

        # How a job stuck in one state is eventually spotted: the age of that state has
        # to be readable, so the stamp cannot stay at the time the row was created.
        assert stored.status == JOB_STATUS_PROCESSING
        assert stored.updated_at > queued_at
        assert stored.created_at < stored.updated_at


def test_deleting_an_analysis_takes_its_job_with_it(session):
    db, created = session
    analysis = queued_analysis(db, created)
    db.add(AnalysisJob(analysis_id=analysis.id, status=JOB_STATUS_QUEUED))
    db.commit()

    db.query(Analysis).filter(Analysis.id == analysis.id).delete()
    db.commit()

    with SessionLocal() as reader:
        assert reader.query(AnalysisJob).filter_by(analysis_id=analysis.id).count() == 0


NVIDIA_PROBABILITY = 0.8734567165374756


def nvidia_signal(analysis_id: uuid.UUID, **overrides) -> AnalysisSignal:
    """A signal row shaped exactly like the one a successful detection persists."""
    values = {
        "analysis_id": analysis_id,
        "provider": "nvidia",
        "signal_type": "synthetic_video",
        "score": NVIDIA_PROBABILITY,
        "provider_version": "847b6e53-0133-452d-ab85-d7acf3ace723",
        "status": SIGNAL_STATUS_SUCCESS,
        "signal_metadata": {"logit": 1.9142135381698608, "total_clips": 7},
    }

    return AnalysisSignal(**{**values, **overrides})


def test_a_detector_signal_round_trips_through_postgresql(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.add(nvidia_signal(analysis.id))
    db.commit()

    with SessionLocal() as reader:
        stored = (
            reader.query(AnalysisSignal)
            .filter(AnalysisSignal.analysis_id == analysis.id)
            .one()
        )

        assert stored.provider == "nvidia"
        assert stored.signal_type == "synthetic_video"
        # Full double precision, not the float4 that would silently truncate it.
        assert stored.score == NVIDIA_PROBABILITY
        assert stored.provider_version == "847b6e53-0133-452d-ab85-d7acf3ace723"
        assert stored.status == SIGNAL_STATUS_SUCCESS
        # JSONB gives the document back as the Python types that went in.
        assert stored.signal_metadata == {"logit": 1.9142135381698608, "total_clips": 7}
        assert stored.created_at is not None
        assert stored.created_at.tzinfo is not None


def test_a_persisted_signal_carries_no_risk_level(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.add(nvidia_signal(analysis.id))
    db.commit()

    with SessionLocal() as reader:
        stored = (
            reader.query(AnalysisSignal)
            .filter(AnalysisSignal.analysis_id == analysis.id)
            .one()
        )
        # The column exists for the phase that owns risk logic. Nothing fills it yet.
        assert stored.risk_level is None


def test_a_failed_signal_persists_without_a_score(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.add(
        nvidia_signal(
            analysis.id,
            score=None,
            provider_version=None,
            status=SIGNAL_STATUS_FAILED,
            signal_metadata={"error": "NvidiaAuthenticationError"},
        )
    )
    db.commit()

    with SessionLocal() as reader:
        stored = (
            reader.query(AnalysisSignal)
            .filter(AnalysisSignal.analysis_id == analysis.id)
            .one()
        )

        assert stored.status == SIGNAL_STATUS_FAILED
        # A detector that never answered has no number, and the schema allows that.
        assert stored.score is None
        assert stored.signal_metadata == {"error": "NvidiaAuthenticationError"}


def test_deleting_an_analysis_takes_its_signals_with_it(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    db.add(media_file(analysis.id))
    db.add(nvidia_signal(analysis.id))
    db.commit()

    db.query(Analysis).filter(Analysis.id == analysis.id).delete()
    db.commit()

    with SessionLocal() as reader:
        assert (
            reader.query(AnalysisSignal)
            .filter(AnalysisSignal.analysis_id == analysis.id)
            .count()
            == 0
        )


def test_a_persisted_signal_reaches_the_listing_endpoint(session):
    """The dashboard's evidence read path, end to end against real PostgreSQL.

    The fake-session suite proves the query's shape; only a real database proves the
    outer join returns the stored figures, and returns nothing where none were stored.
    """
    db, created = session
    with_signal = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    without_signal = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add_all([with_signal, without_signal])
    db.flush()
    created.extend([with_signal.id, without_signal.id])
    db.add_all([media_file(with_signal.id), media_file(without_signal.id)])
    db.add(nvidia_signal(with_signal.id))
    db.commit()

    with TestClient(app) as client:
        body = client.get("/api/v1/analyses").json()

    listed = next(row for row in body if row["id"] == str(with_signal.id))["synthetic_video"]
    assert listed["provider"] == "nvidia"
    assert listed["signal_type"] == "synthetic_video"
    assert listed["status"] == SIGNAL_STATUS_SUCCESS
    # Full double precision survives the round trip through the join.
    assert listed["score"] == NVIDIA_PROBABILITY
    assert listed["provider_version"] == "847b6e53-0133-452d-ab85-d7acf3ace723"
    assert listed["logit"] == 1.9142135381698608
    assert listed["total_clips"] == 7

    # An analysis with no signal still lists, and claims no evidence it does not have.
    other = next(row for row in body if row["id"] == str(without_signal.id))
    assert other["synthetic_video"] is None


def test_a_failed_signal_reaches_the_listing_without_a_score(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.add(
        nvidia_signal(
            analysis.id,
            score=None,
            provider_version=None,
            status=SIGNAL_STATUS_FAILED,
            signal_metadata={"error": "NvidiaAuthenticationError"},
        )
    )
    db.commit()

    with TestClient(app) as client:
        response = client.get("/api/v1/analyses")

    listed = next(row for row in response.json() if row["id"] == str(analysis.id))
    assert listed["synthetic_video"]["status"] == SIGNAL_STATUS_FAILED
    assert listed["synthetic_video"]["score"] is None
    # The provider's failure detail stays in the database and the server log.
    assert "NvidiaAuthenticationError" not in response.text


def nvidia_segment(signal_id: uuid.UUID, clip_index: int, logit: float) -> AnalysisSegment:
    """A clip evidence row shaped exactly like the one a detection persists."""
    return AnalysisSegment(signal_id=signal_id, clip_index=clip_index, logit=logit)


def test_clip_evidence_round_trips_through_postgresql(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    signal = nvidia_signal(analysis.id)
    db.add(signal)
    db.flush()
    db.add_all(
        [
            nvidia_segment(signal.id, 8, 3.5),
            nvidia_segment(signal.id, 0, -2.25),
        ]
    )
    db.commit()

    with SessionLocal() as reader:
        stored = (
            reader.query(AnalysisSegment)
            .filter(AnalysisSegment.signal_id == signal.id)
            .order_by(AnalysisSegment.logit.desc())
            .all()
        )

        assert [(row.clip_index, row.logit) for row in stored] == [(8, 3.5), (0, -2.25)]
        # Full double precision, negative logits included, straight back out again.
        assert stored[1].logit == -2.25
        assert stored[0].created_at is not None
        assert stored[0].created_at.tzinfo is not None


def test_a_wide_frame_index_survives_postgresql(session):
    """NVIDIA's clip index is a uint32, which overflows a 32-bit integer column."""
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    signal = nvidia_signal(analysis.id)
    db.add(signal)
    db.flush()
    db.add(nvidia_segment(signal.id, 4294967295, 0.5))
    db.commit()

    with SessionLocal() as reader:
        stored = (
            reader.query(AnalysisSegment)
            .filter(AnalysisSegment.signal_id == signal.id)
            .one()
        )

        assert stored.clip_index == 4294967295


def test_deleting_an_analysis_takes_its_clip_evidence_with_it(session):
    """The cascade has to reach through the signal to the evidence beneath it."""
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    db.add(media_file(analysis.id))
    signal = nvidia_signal(analysis.id)
    db.add(signal)
    db.flush()
    signal_id = signal.id
    db.add(nvidia_segment(signal_id, 8, 3.5))
    db.commit()

    db.query(Analysis).filter(Analysis.id == analysis.id).delete()
    db.commit()

    with SessionLocal() as reader:
        assert (
            reader.query(AnalysisSegment)
            .filter(AnalysisSegment.signal_id == signal_id)
            .count()
            == 0
        )


def test_clip_evidence_reaches_the_listing_endpoint(session):
    """The dashboard's segment read path, end to end against real PostgreSQL.

    The fake-session suite proves the second query's shape; only a real database proves
    the grouping hands each analysis its own evidence, strongest clip first.
    """
    db, created = session
    with_clips = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    without_clips = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add_all([with_clips, without_clips])
    db.flush()
    created.extend([with_clips.id, without_clips.id])
    db.add_all([media_file(with_clips.id), media_file(without_clips.id)])
    signal = nvidia_signal(with_clips.id)
    db.add_all([signal, nvidia_signal(without_clips.id)])
    db.flush()
    db.add_all(
        [
            nvidia_segment(signal.id, 0, -2.25),
            nvidia_segment(signal.id, 8, 3.5),
            nvidia_segment(signal.id, 4, 1.75),
        ]
    )
    db.commit()

    with TestClient(app) as client:
        body = client.get("/api/v1/analyses").json()

    listed = next(row for row in body if row["id"] == str(with_clips.id))["synthetic_video"]
    assert listed["segments"] == [
        {"clip_index": 8, "logit": 3.5},
        {"clip_index": 4, "logit": 1.75},
        {"clip_index": 0, "logit": -2.25},
    ]

    # A signal with no stored evidence claims none.
    other = next(row for row in body if row["id"] == str(without_clips.id))
    assert other["synthetic_video"]["segments"] == []


def test_the_listing_returns_at_most_the_display_count_of_clips(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    signal = nvidia_signal(analysis.id)
    db.add(signal)
    db.flush()
    db.add_all(
        [
            nvidia_segment(signal.id, index, float(index))
            for index in range(DASHBOARD_SEGMENTS + 3)
        ]
    )
    db.commit()

    with TestClient(app) as client:
        body = client.get("/api/v1/analyses").json()

    segments = next(row for row in body if row["id"] == str(analysis.id))["synthetic_video"][
        "segments"
    ]
    assert len(segments) == DASHBOARD_SEGMENTS
    # Trimmed from the weak end: the strongest clips are the ones that survive.
    assert [segment["clip_index"] for segment in segments] == list(
        range(DASHBOARD_SEGMENTS + 2, 2, -1)
    )


# Audio window evidence. The third source to write into `analysis_segments`, and the first
# whose rows carry two figures for one scored unit.


def aasist_signal(analysis_id: uuid.UUID, **overrides) -> AnalysisSignal:
    """A signal row shaped exactly like the one a successful audio reading persists."""
    values = {
        "analysis_id": analysis_id,
        "provider": "aasist",
        "signal_type": "audio_authenticity",
        # The checkpoint publishes no calibration over its logits, so there is no
        # file-level figure and the column stays empty.
        "score": None,
        "provider_version": (
            "SpeechAntiSpoofingBenchmarks/AASIST@16774d458d86d2a021ae31646c1bf66a5331b53e"
        ),
        "status": SIGNAL_STATUS_SUCCESS,
        "signal_metadata": {
            "sample_rate": 16000,
            "window_samples": 64600,
            "total_audio_windows": 2,
            "persisted_audio_windows": 2,
            "windows_truncated": False,
            "window_bounds": "deepguard_preprocessing",
            "bona_fide_logit_index": 1,
        },
    }

    return AnalysisSignal(**{**values, **overrides})


def audio_window(signal_id: uuid.UUID, index: int, logits, bounds) -> AnalysisSegment:
    """An audio window row shaped exactly like the one an audio reading persists."""
    return AnalysisSegment(
        signal_id=signal_id,
        clip_index=index,
        logit=logits[0],
        bona_fide_logit=logits[1],
        start_time=bounds[0],
        end_time=bounds[1],
    )


def test_audio_window_evidence_round_trips_through_postgresql(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    signal = aasist_signal(analysis.id)
    db.add(signal)
    db.flush()
    db.add_all(
        [
            audio_window(signal.id, 0, (-1.5, 4.25), (0.0, 4.0375)),
            audio_window(signal.id, 1, (2.0, -0.5), (4.0375, 8.075)),
        ]
    )
    db.commit()

    with SessionLocal() as reader:
        stored = (
            reader.query(AnalysisSegment)
            .filter(AnalysisSegment.signal_id == signal.id)
            .order_by(AnalysisSegment.clip_index)
            .all()
        )

    # Both of the model's raw outputs survive, in graph order and at full double precision.
    # Neither can be derived from the other, so a schema that held one would have lost half
    # of what the model said.
    assert [(row.logit, row.bona_fide_logit) for row in stored] == [
        (-1.5, 4.25),
        (2.0, -0.5),
    ]
    # 64600 samples at 16 kHz, exactly, in the order the recording was cut.
    assert [(row.start_time, row.end_time) for row in stored] == [
        (0.0, 4.0375),
        (4.0375, 8.075),
    ]
    assert [row.clip_index for row in stored] == [0, 1]
    # No face and no voice identity in an anti-spoofing result.
    assert [(row.face_id, row.speaker_label) for row in stored] == [(None, None)] * 2


def test_an_audio_signal_persists_without_a_score(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.add(aasist_signal(analysis.id))
    db.commit()

    with SessionLocal() as reader:
        stored = reader.query(AnalysisSignal).filter_by(analysis_id=analysis.id).one()

    assert stored.provider == "aasist"
    assert stored.signal_type == "audio_authenticity"
    assert stored.status == SIGNAL_STATUS_SUCCESS
    assert stored.score is None
    assert stored.risk_level is None
    assert stored.signal_metadata["window_bounds"] == "deepguard_preprocessing"


def test_the_audio_signal_is_independent_of_the_other_evidence(session):
    """Four sources, four rows, one analysis — and none of them waits on another."""
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    detection_signal = nvidia_signal(analysis.id)
    audio_signal = aasist_signal(analysis.id, status=SIGNAL_STATUS_FAILED, score=None)
    db.add_all([detection_signal, audio_signal])
    db.flush()
    db.add_all(
        [
            nvidia_segment(detection_signal.id, 8, 3.5),
            nvidia_segment(detection_signal.id, 0, -2.25),
        ]
    )
    db.commit()

    with SessionLocal() as reader:
        clips = (
            reader.query(AnalysisSegment)
            .filter(AnalysisSegment.signal_id == detection_signal.id)
            .all()
        )
        audio_rows = (
            reader.query(AnalysisSegment)
            .filter(AnalysisSegment.signal_id == audio_signal.id)
            .all()
        )

    # A failed audio reading owns no rows, and the clip evidence beside it is untouched —
    # including the column the audio source added, which stays null on rows that are not
    # audio rather than repeating a figure NVIDIA never gave.
    assert audio_rows == []
    assert sorted((row.clip_index, row.logit) for row in clips) == [(0, -2.25), (8, 3.5)]
    assert {row.bona_fide_logit for row in clips} == {None}


def test_deleting_an_analysis_takes_its_audio_evidence_with_it(session):
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    signal = aasist_signal(analysis.id)
    db.add(signal)
    db.flush()
    db.add(audio_window(signal.id, 0, (-1.5, 4.25), (0.0, 4.0375)))
    db.commit()

    db.query(Analysis).filter(Analysis.id == analysis.id).delete()
    db.commit()

    with SessionLocal() as reader:
        assert reader.query(AnalysisSegment).filter_by(signal_id=signal.id).count() == 0


def test_the_audio_signal_reaches_the_listing_beside_the_others(session):
    """Four signals on one analysis, still one row.

    Every join in the listing names one provider and one signal type, so the fourth signal
    can neither multiply the analysis's rows nor land in another source's fields.
    """
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.add_all([nvidia_signal(analysis.id), aasist_signal(analysis.id)])
    db.commit()

    with TestClient(app) as client:
        body = client.get("/api/v1/analyses").json()

    listed = [row for row in body if row["id"] == str(analysis.id)]
    assert len(listed) == 1
    assert listed[0]["synthetic_video"]["provider"] == "nvidia"

    audio = listed[0]["audio_authenticity"]
    assert audio["provider"] == "aasist"
    assert audio["signal_type"] == "audio_authenticity"
    assert audio["status"] == SIGNAL_STATUS_SUCCESS
    # The reading owns no windows here, and the listing says so rather than borrowing the
    # clip evidence hanging off the signal beside it.
    assert audio["windows"] == []


def test_audio_windows_round_trip_out_through_the_listing_endpoint(session):
    """What the worker stored is what the dashboard is handed, through real PostgreSQL."""
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    signal = aasist_signal(analysis.id)
    db.add(signal)
    db.flush()
    db.add_all(
        [
            audio_window(signal.id, 0, (-1.5, 4.25), (0.0, 4.0375)),
            audio_window(signal.id, 1, (2.0, -0.5), (4.0375, 8.075)),
        ]
    )
    db.commit()

    with TestClient(app) as client:
        body = client.get("/api/v1/analyses").json()

    windows = [row for row in body if row["id"] == str(analysis.id)][0][
        "audio_authenticity"
    ]["windows"]

    # Both raw outputs at full double precision, in the order the audio was cut.
    assert [(window["logit"], window["bona_fide_logit"]) for window in windows] == [
        (-1.5, 4.25),
        (2.0, -0.5),
    ]
    assert [(window["start_time"], window["end_time"]) for window in windows] == [
        (0.0, 4.0375),
        (4.0375, 8.075),
    ]
    assert [window["clip_index"] for window in windows] == [0, 1]


def test_audio_windows_do_not_leak_into_the_clip_evidence(session):
    """Both kinds of row live in one table and both fill `clip_index`.

    The evidence queries are keyed by signal id, so nothing but the parent signal decides
    which rows a source is handed — but the two sources are now close enough in shape that
    it is worth proving rather than assuming.
    """
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    detection_signal = nvidia_signal(analysis.id)
    audio_signal = aasist_signal(analysis.id)
    db.add_all([detection_signal, audio_signal])
    db.flush()
    db.add_all(
        [
            nvidia_segment(detection_signal.id, 8, 3.5),
            audio_window(audio_signal.id, 0, (-1.5, 4.25), (0.0, 4.0375)),
        ]
    )
    db.commit()

    with TestClient(app) as client:
        listed = [
            row
            for row in client.get("/api/v1/analyses").json()
            if row["id"] == str(analysis.id)
        ][0]

    assert listed["synthetic_video"]["segments"] == [{"clip_index": 8, "logit": 3.5}]
    assert len(listed["audio_authenticity"]["windows"]) == 1
    assert listed["audio_authenticity"]["windows"][0]["clip_index"] == 0


# The single-analysis endpoint the report route reads. The fake-session suite proves the
# query's shape; only a real database proves the narrowed select returns that analysis's own
# evidence and nobody else's.


def test_one_analysis_is_read_back_through_the_detail_endpoint(session):
    db, created = session
    wanted = Analysis(
        status=ANALYSIS_STATUS_COMPLETED,
        risk_level="HIGH",
        risk_rules_version="p7-v1.0.0",
        risk_rule_id="R100",
        risk_calibration_id=(
            "3e362e8edfe253437234e3c291230a2921a6344555ab0861ee5871c53d20949c"
        ),
    )
    db.add(wanted)
    db.flush()
    created.append(wanted.id)
    db.add(media_file(wanted.id))
    db.add(nvidia_signal(wanted.id))
    db.commit()

    with TestClient(app) as client:
        body = client.get(f"/api/v1/analyses/{wanted.id}").json()

    assert body["id"] == str(wanted.id)
    # The decision comes off the analysis row, whole.
    assert body["risk_level"] == "HIGH"
    assert body["risk_rules_version"] == "p7-v1.0.0"
    assert body["risk_rule_id"] == "R100"
    assert body["risk_calibration_id"].startswith("3e362e8e")
    # And the evidence behind it is the provider's own figures, unchanged.
    assert body["synthetic_video"]["score"] == NVIDIA_PROBABILITY
    assert body["synthetic_video"]["status"] == SIGNAL_STATUS_SUCCESS


def test_the_detail_endpoint_returns_only_the_requested_analysis_evidence(session):
    """Two analyses, each with a signal. Neither may be shown the other's."""
    db, created = session
    wanted = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    other = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add_all([wanted, other])
    db.flush()
    created.extend([wanted.id, other.id])
    db.add_all([media_file(wanted.id), media_file(other.id)])
    db.add(nvidia_signal(wanted.id, score=0.25))
    db.add(nvidia_signal(other.id, score=0.75))
    db.commit()

    with TestClient(app) as client:
        body = client.get(f"/api/v1/analyses/{wanted.id}").json()

    assert body["id"] == str(wanted.id)
    assert body["synthetic_video"]["score"] == 0.25


def test_a_pre_p7_analysis_keeps_null_risk_through_the_detail_endpoint(session):
    """Never backfilled to `UNKNOWN`: the report has to be able to say nothing was decided."""
    db, created = session
    analysis = Analysis(status=ANALYSIS_STATUS_COMPLETED)
    db.add(analysis)
    db.flush()
    created.append(analysis.id)
    db.add(media_file(analysis.id))
    db.commit()

    with TestClient(app) as client:
        body = client.get(f"/api/v1/analyses/{analysis.id}").json()

    assert body["risk_level"] is None
    assert body["risk_rules_version"] is None
    assert body["risk_rule_id"] is None
    assert body["risk_calibration_id"] is None


def test_an_analysis_that_does_not_exist_is_a_404(session):
    with TestClient(app) as client:
        response = client.get(f"/api/v1/analyses/{uuid.uuid4()}")

    assert response.status_code == 404
