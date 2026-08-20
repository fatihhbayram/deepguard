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
from sqlalchemy.exc import SQLAlchemyError

from app.api.analyses import RECENT_ANALYSES_LIMIT
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    Analysis,
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

    assert {"analyses", "media_files", "analysis_signals"} <= set(inspector.get_table_names())
    assert {column["name"] for column in inspector.get_columns("analyses")} == {
        "id",
        "status",
        "created_at",
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
