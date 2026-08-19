"""Persistence proof against real PostgreSQL.

The rest of the suite fakes the session, which proves what the route does but not that
the schema exists or that PostgreSQL accepts and returns these values. This module runs
against the database from docker-compose: it needs the Alembic migration to have been
applied (`alembic upgrade head`) and skips when no database is reachable.
"""

import hashlib
import uuid

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import ANALYSIS_STATUS_COMPLETED, Analysis, MediaFile
from app.db.session import SessionLocal, engine

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

    assert {"analyses", "media_files"} <= set(inspector.get_table_names())
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
