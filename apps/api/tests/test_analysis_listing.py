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

from app.api.analyses import DASHBOARD_SEGMENTS, RECENT_ANALYSES_LIMIT
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
    "segments",
}

# One clip of provider evidence: the frame index NVIDIA scored and the logit it gave it.
# No time range and no probability, because NVIDIA reports neither per clip.
EXPECTED_SEGMENT_FIELDS = {"clip_index", "logit"}

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
        "signal_id": uuid.uuid4(),
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
        signal_id=None,
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


def segment_row(signal_id, clip_index: int, logit: float):
    """A row shaped like the one the segment select emits."""
    return SimpleNamespace(signal_id=signal_id, clip_index=clip_index, logit=logit)


class FakeSession:
    """Stand-in for a SQLAlchemy session that records the statements it was given.

    The route issues two: the listing, then the clip evidence for the signals it found.
    They are answered from separate row sets, in that order.
    """

    def __init__(self, rows=(), segment_rows=()):
        self.rows = list(rows)
        self.segment_rows = list(segment_rows)
        self.execute_error = None
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        if self.execute_error is not None:
            raise self.execute_error

        rows = self.rows if len(self.statements) == 1 else self.segment_rows

        return SimpleNamespace(all=lambda: rows)


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


def compiled(session, index: int = 0) -> str:
    """One statement the route issued, as literal SQL. The listing is the first."""
    statement = session.statements[index]

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
                "segments": [],
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

    # The signal rides along on the listing itself rather than being fetched per analysis.
    assert "LEFT OUTER JOIN analysis_signals" in compiled(fake_session)


def test_the_page_costs_the_same_number_of_queries_however_many_analyses_it_holds(
    client, fake_session
):
    """The N+1 guard: two statements for one analysis, and two for twenty."""
    fake_session.rows = [listing_row()]
    client.get("/api/v1/analyses")
    assert len(fake_session.statements) == 2

    fake_session.statements.clear()
    fake_session.rows = [listing_row() for _ in range(20)]
    client.get("/api/v1/analyses")

    assert len(fake_session.statements) == 2


def test_no_segment_query_is_issued_when_no_analysis_carries_a_signal(client, fake_session):
    fake_session.rows = [unsignalled_row(), unsignalled_row()]

    client.get("/api/v1/analyses")

    # Nothing to look up, so the second statement is not worth its round trip.
    assert len(fake_session.statements) == 1


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


def test_segment_evidence_is_exposed_on_the_signal(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.segment_rows = [
        segment_row(row.signal_id, 45, 2.5),
        segment_row(row.signal_id, 12, 1.25),
    ]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["segments"] == [
        {"clip_index": 45, "logit": 2.5},
        {"clip_index": 12, "logit": 1.25},
    ]


def test_a_segment_exposes_no_field_beyond_what_the_provider_reported(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.segment_rows = [segment_row(row.signal_id, 45, 2.5)]

    segment = client.get("/api/v1/analyses").json()[0]["synthetic_video"]["segments"][0]

    # No start_time, no end_time, no score, no risk_level: NVIDIA reports a frame index
    # and a logit for a clip, and anything else here would have been invented.
    assert set(segment) == EXPECTED_SEGMENT_FIELDS


def test_segment_logits_are_exposed_untransformed(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    # A raw model logit: negative, and on no bounded scale.
    fake_session.segment_rows = [segment_row(row.signal_id, 3, -1.4362945556640625)]

    segment = client.get("/api/v1/analyses").json()[0]["synthetic_video"]["segments"][0]

    assert segment["logit"] == -1.4362945556640625


def test_segments_are_capped_at_the_display_count(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]
    fake_session.segment_rows = [
        segment_row(row.signal_id, index, 3.0 - index) for index in range(DASHBOARD_SEGMENTS + 4)
    ]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert len(signal["segments"]) == DASHBOARD_SEGMENTS
    # The strongest survive the cap; the weakest are the ones dropped.
    assert [segment["clip_index"] for segment in signal["segments"]] == list(
        range(DASHBOARD_SEGMENTS)
    )


def test_each_analysis_receives_only_its_own_segments(client, fake_session):
    first, second = listing_row(), listing_row()
    fake_session.rows = [first, second]
    fake_session.segment_rows = [
        segment_row(first.signal_id, 1, 2.0),
        segment_row(second.signal_id, 9, 1.0),
        segment_row(second.signal_id, 8, 0.5),
    ]

    body = client.get("/api/v1/analyses").json()

    assert body[0]["synthetic_video"]["segments"] == [{"clip_index": 1, "logit": 2.0}]
    assert [s["clip_index"] for s in body[1]["synthetic_video"]["segments"]] == [9, 8]


def test_a_signal_with_no_stored_segments_reports_an_empty_list(client, fake_session):
    fake_session.rows = [listing_row()]
    fake_session.segment_rows = []

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    # Empty, not null: the signal exists and simply carries no clip evidence.
    assert signal["segments"] == []


def test_a_failed_signal_carries_no_segments(client, fake_session):
    fake_session.rows = [failed_signal_row("FAILED")]

    signal = client.get("/api/v1/analyses").json()[0]["synthetic_video"]

    assert signal["segments"] == []


def test_the_segment_query_asks_only_for_the_listed_signals(client, fake_session):
    row = listing_row()
    fake_session.rows = [row, unsignalled_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=1)
    assert "analysis_segments.signal_id IN" in sql
    # Rendered without its dashes by the literal bind.
    assert row.signal_id.hex in sql
    # The analysis carrying no signal contributes no id to look up.
    assert sql.count("'") == 2


def test_the_segment_query_orders_the_strongest_clip_first(client, fake_session):
    fake_session.rows = [listing_row()]

    client.get("/api/v1/analyses")

    sql = compiled(fake_session, index=1)
    # The clip index breaks ties, so the same evidence always reads back in one order.
    assert (
        "ORDER BY analysis_segments.signal_id, analysis_segments.logit DESC, "
        "analysis_segments.clip_index" in sql
    )


def test_a_segment_query_failure_returns_the_same_controlled_503(client, fake_session):
    fake_session.rows = [listing_row()]

    def fail_on_the_segment_query(statement):
        fake_session.statements.append(statement)
        if len(fake_session.statements) == 1:
            return SimpleNamespace(all=lambda: fake_session.rows)
        raise OperationalError("SELECT", None, Exception("connection lost"))

    fake_session.execute = fail_on_the_segment_query

    response = client.get("/api/v1/analyses")

    assert response.status_code == 503
    assert response.json() == {"detail": "analyses are temporarily unavailable"}
