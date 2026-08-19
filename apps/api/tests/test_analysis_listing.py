"""Unit coverage for the dashboard listing endpoint.

The session is faked here so the suite needs no live database: what is under test is the
query this route builds and how it maps rows onto the response. The query is asserted by
compiling it, because a fake driver cannot prove ordering or a limit.
`tests/test_persistence.py` covers the same endpoint against real PostgreSQL.
"""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.api.analyses import RECENT_ANALYSES_LIMIT
from app.db.session import get_session
from app.main import app

CREATED_AT = datetime(2026, 8, 19, 18, 8, 1, tzinfo=timezone.utc)

# Exactly the fields the dashboard renders. Anything else appearing here would be a leak
# of storage internals or of a phase that has not happened yet.
EXPECTED_FIELDS = {
    "id",
    "status",
    "created_at",
    "original_filename",
    "declared_content_type",
    "size_bytes",
    "original_sha256",
    "was_normalized",
}


def listing_row(**overrides):
    """A row shaped like the one the select emits, with column names, not model names."""
    values = {
        "id": uuid.uuid4(),
        "status": "completed",
        "created_at": CREATED_AT,
        "original_filename": "clip.mp4",
        # The column is `content_type`; the response renames it to `declared_content_type`.
        "content_type": "video/mp4",
        "size_bytes": 13054,
        "original_sha256": "a" * 64,
        "was_normalized": False,
    }

    return SimpleNamespace(**{**values, **overrides})


class FakeSession:
    """Stand-in for a SQLAlchemy session that records the statement it was given."""

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.execute_error = None
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        if self.execute_error is not None:
            raise self.execute_error

        return SimpleNamespace(all=lambda: self.rows)


@pytest.fixture
def fake_session():
    session = FakeSession()
    app.dependency_overrides[get_session] = lambda: session
    yield session
    app.dependency_overrides.clear()


@pytest.fixture
def client(fake_session):
    with TestClient(app) as test_client:
        yield test_client


def compiled(session) -> str:
    """The single statement the route issued, as literal SQL."""
    statement = session.statements[0]

    return str(statement.compile(compile_kwargs={"literal_binds": True}))


def test_empty_database_returns_an_empty_list(client, fake_session):
    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    assert response.json() == []


def test_persisted_analysis_is_returned_with_the_dashboard_fields(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": str(row.id),
            "status": "completed",
            "created_at": "2026-08-19T18:08:01Z",
            "original_filename": "clip.mp4",
            "declared_content_type": "video/mp4",
            "size_bytes": 13054,
            "original_sha256": "a" * 64,
            "was_normalized": False,
        }
    ]


def test_listing_exposes_no_field_beyond_the_dashboard_set(client, fake_session):
    fake_session.rows = [listing_row()]

    response = client.get("/api/v1/analyses")

    # Storage keys, ffprobe geometry and derivative identity are not part of this view.
    assert set(response.json()[0]) == EXPECTED_FIELDS


def test_declared_content_type_reports_the_declared_mime(client, fake_session):
    fake_session.rows = [listing_row(content_type="video/quicktime")]

    response = client.get("/api/v1/analyses")

    assert response.json()[0]["declared_content_type"] == "video/quicktime"


def test_a_missing_filename_is_returned_as_null(client, fake_session):
    fake_session.rows = [listing_row(original_filename=None)]

    response = client.get("/api/v1/analyses")

    assert response.json()[0]["original_filename"] is None


def test_normalized_analysis_reports_it(client, fake_session):
    fake_session.rows = [listing_row(was_normalized=True)]

    response = client.get("/api/v1/analyses")

    assert response.json()[0]["was_normalized"] is True


def test_every_persisted_analysis_is_returned(client, fake_session):
    fake_session.rows = [
        listing_row(created_at=CREATED_AT - timedelta(minutes=offset)) for offset in range(3)
    ]

    response = client.get("/api/v1/analyses")

    assert len(response.json()) == 3


def test_the_query_joins_media_onto_the_analysis(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # An analysis and its media are written in one transaction, so an inner join cannot
    # hide a row.
    assert "JOIN media_files ON media_files.analysis_id = analyses.id" in sql
    assert "LEFT" not in sql


def test_the_query_returns_the_most_recent_analyses_first(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # The id breaks ties, because two analyses committed together share `created_at`.
    assert "ORDER BY analyses.created_at DESC, analyses.id DESC" in sql


def test_the_query_applies_the_fixed_limit(client, fake_session):
    client.get("/api/v1/analyses")

    assert f"LIMIT {RECENT_ANALYSES_LIMIT}" in compiled(fake_session)


def test_the_query_selects_only_the_columns_the_listing_needs(client, fake_session):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    for absent in ("original_storage_key", "derivative_storage_key", "derivative_sha256"):
        assert absent not in sql


def test_database_failure_returns_a_controlled_503(client, fake_session):
    fake_session.execute_error = OperationalError("SELECT", None, Exception("connection lost"))

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    # No statement, connection string or driver detail reaches the client.
    assert response.json() == {"detail": "analyses are temporarily unavailable"}
