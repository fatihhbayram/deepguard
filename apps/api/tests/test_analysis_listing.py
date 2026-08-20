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
    "synthetic_video",
}

# The signal object the dashboard receives: the provider's own identity, state and
# figures. `risk_level` is not among them — no phase owns risk yet — and neither is the
# raw metadata document, which carries diagnostic detail on a failure.
EXPECTED_SIGNAL_FIELDS = {
    "provider",
    "signal_type",
    "status",
    "score",
    "provider_version",
    "logit",
    "total_clips",
}

PROBABILITY = 0.7929722666740417
FUNCTION_ID = "847b6e53-0133-452d-ab85-d7acf3ace723"


def listing_row(**overrides):
    """A row shaped like the one the select emits, with column names, not model names.

    The default carries a successful NVIDIA signal, since that is what an analysis run
    through the current pipeline has.
    """
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
        "signal_provider": "nvidia",
        "signal_type": "synthetic_video",
        "signal_status": "SUCCESS",
        "signal_score": PROBABILITY,
        "signal_provider_version": FUNCTION_ID,
        "signal_metadata": {"logit": 1.9142135381698608, "total_clips": 7},
    }

    return SimpleNamespace(**{**values, **overrides})


def unsignalled_row(**overrides):
    """A row for an analysis the outer join found no NVIDIA signal for."""
    return listing_row(
        signal_provider=None,
        signal_type=None,
        signal_status=None,
        signal_score=None,
        signal_provider_version=None,
        signal_metadata=None,
        **overrides,
    )


def failed_signal_row(status: str, **overrides):
    """A row whose detection did not produce a number, as the pipeline persists it."""
    return listing_row(
        signal_status=status,
        signal_score=None,
        signal_provider_version=None,
        signal_metadata={"error": "NvidiaProviderTimeout"},
        **overrides,
    )


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
            "synthetic_video": {
                "provider": "nvidia",
                "signal_type": "synthetic_video",
                "status": "SUCCESS",
                "score": PROBABILITY,
                "provider_version": FUNCTION_ID,
                "logit": 1.9142135381698608,
                "total_clips": 7,
            },
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
    assert "LEFT OUTER JOIN media_files" not in sql


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


def test_the_signal_is_read_in_the_same_statement_as_the_listing(client, fake_session):
    fake_session.rows = [listing_row(), listing_row()]

    client.get("/api/v1/analyses")

    # One statement for the whole page: a query per analysis is the N+1 this must not be.
    assert len(fake_session.statements) == 1
    sql = compiled(fake_session)
    assert "LEFT OUTER JOIN analysis_signals" in sql


def test_the_signal_join_is_restricted_to_the_nvidia_synthetic_video_signal(
    client, fake_session
):
    client.get("/api/v1/analyses")

    sql = compiled(fake_session)
    # Narrowing the join keeps a second provider's future signal from multiplying rows.
    assert "analysis_signals.provider = 'nvidia'" in sql
    assert "analysis_signals.signal_type = 'synthetic_video'" in sql


def test_the_query_selects_no_signal_column_the_dashboard_must_not_show(client, fake_session):
    client.get("/api/v1/analyses")

    # Risk classification does not exist yet, and reading it would imply it does.
    assert "analysis_signals.risk_level" not in compiled(fake_session)


def test_a_successful_signal_is_exposed_with_the_provider_figures(client, fake_session):
    fake_session.rows = [listing_row()]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert set(signal) == EXPECTED_SIGNAL_FIELDS
    assert signal["provider"] == "nvidia"
    assert signal["signal_type"] == "synthetic_video"
    assert signal["status"] == "SUCCESS"
    # Untransformed and unrounded: NVIDIA's number, on NVIDIA's scale.
    assert signal["score"] == PROBABILITY
    assert signal["provider_version"] == FUNCTION_ID
    assert signal["total_clips"] == 7


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_a_failed_signal_is_exposed_with_a_null_score(client, fake_session, status):
    fake_session.rows = [failed_signal_row(status)]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["status"] == status
    # Never 0.0: the detector produced no number, and zero would be a fabricated answer.
    assert signal["score"] is None
    assert signal["logit"] is None
    assert signal["total_clips"] is None


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT"])
def test_a_failed_signal_never_exposes_the_provider_error(client, fake_session, status):
    fake_session.rows = [failed_signal_row(status)]

    body = client.get("/api/v1/analyses").text

    # The stored metadata holds the exception class name; it is diagnostic, not evidence.
    assert "NvidiaProviderTimeout" not in body
    assert "error" not in body


def test_an_analysis_without_a_signal_is_listed_with_none(client, fake_session):
    fake_session.rows = [unsignalled_row()]

    row = client.get("/api/v1/analyses").json()[0]

    assert row["synthetic_video"] is None
    # The analysis itself still lists in full.
    assert row["original_filename"] == "clip.mp4"


def test_unusable_metadata_does_not_break_the_listing(client, fake_session):
    fake_session.rows = [
        listing_row(signal_metadata=None),
        listing_row(signal_metadata={"total_clips": "seven"}),
        listing_row(signal_metadata={"logit": True, "total_clips": True}),
    ]

    response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    # A figure that is missing or of the wrong type is reported as absent, not guessed at.
    for row in response.json():
        assert row["synthetic_video"]["logit"] is None
        assert row["synthetic_video"]["total_clips"] is None
        assert row["synthetic_video"]["score"] == PROBABILITY


def test_a_whole_logit_survives_the_json_round_trip(client, fake_session):
    fake_session.rows = [listing_row(signal_metadata={"logit": 2, "total_clips": 7})]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["logit"] == 2.0


def test_database_failure_returns_a_controlled_503(client, fake_session):
    fake_session.execute_error = OperationalError("SELECT", None, Exception("connection lost"))

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    # No statement, connection string or driver detail reaches the client.
    assert response.json() == {"detail": "analyses are temporarily unavailable"}
