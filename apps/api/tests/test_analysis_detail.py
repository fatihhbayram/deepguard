"""Unit coverage for the single-analysis endpoint the report route reads.

The session is faked here for the same reason it is in `test_analysis_listing.py`: what is
under test is the query this route builds and how it maps a row onto the response. The row
builders are imported from that module rather than copied — the two endpoints read the same
columns through the same select, so a second copy of a forty-column row would drift from the
query the moment a signal gained a field, and these tests would keep passing while saying
nothing. `tests/test_persistence.py` covers this endpoint against real PostgreSQL.
"""

import uuid

import pytest
from sqlalchemy.exc import OperationalError

from tests.test_analysis_listing import (  # noqa: F401 — fixtures used by name
    CALIBRATION_ID,
    EXPECTED_FIELDS,
    RULE_CALIBRATED_HIGH,
    RULE_INDETERMINATE_BAND,
    RULE_UNVALIDATED_PROVIDER,
    RULES_VERSION,
    admin,
    client,
    compiled,
    fake_session,
    listing_row,
    unsignalled_row,
)

MISSING_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def detail(client, analysis_id):
    return client.get(f"/api/v1/analyses/{analysis_id}")


# Identity. The report is about one analysis, so the first thing that must hold is that the
# row it renders is the row that was asked for.


def test_the_requested_analysis_is_returned(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]

    response = detail(client, row.id)

    assert response.status_code == 200
    assert response.json()["id"] == str(row.id)


def test_the_query_is_narrowed_to_the_requested_id(client, fake_session):
    """The filter is in the statement, not applied to a listing after the fact.

    Reading every recent analysis and picking one out in Python would make the report
    silently depend on its analysis still being among the most recent.
    """
    row = listing_row()
    fake_session.rows = [row]

    detail(client, row.id)

    sql = compiled(fake_session)
    assert "analyses.id = " in sql
    # Rendered without hyphens by the compiler's literal binding.
    assert row.id.hex in sql
    # The listing's narrowing has no business here: this route is not a filtered listing.
    assert "LIMIT" not in sql.upper()


def test_the_detail_response_carries_the_same_fields_as_the_listing(client, fake_session):
    """One model serves both readers, so the report can never be shown less than the
    dashboard already shows about the same analysis."""
    row = listing_row()
    fake_session.rows = [row]

    assert set(detail(client, row.id).json()) == EXPECTED_FIELDS


# Absence. A well-formed id that names nothing is a different fact from a malformed one, and
# from a database that could not be reached.


def test_an_unknown_analysis_is_a_404(client, fake_session):
    fake_session.rows = []

    response = detail(client, MISSING_ID)

    assert response.status_code == 404
    assert response.json()["detail"] == "analysis not found"


def test_a_malformed_id_is_rejected_before_any_statement_runs(client, fake_session):
    """Validation answers this one, so a junk path segment never reaches the database."""
    response = client.get("/api/v1/analyses/not-a-uuid")

    assert response.status_code == 422
    assert fake_session.statements == []


def test_a_database_failure_is_not_reported_as_a_missing_analysis(client, fake_session):
    """503, not 404. An unreachable database has not established that the row is absent."""
    fake_session.execute_error = OperationalError("SELECT 1", {}, Exception("gone"))

    response = detail(client, MISSING_ID)

    assert response.status_code == 503


# The risk decision. Read off the analysis row, exactly as the listing reads it, and for the
# same reason: the report states what was decided, it does not decide.


@pytest.mark.parametrize(
    ("level", "rule"),
    [
        ("HIGH", RULE_CALIBRATED_HIGH),
        ("MEDIUM", RULE_INDETERMINATE_BAND),
        ("UNKNOWN", RULE_UNVALIDATED_PROVIDER),
    ],
)
def test_each_decision_is_returned_with_its_whole_trace(client, fake_session, level, rule):
    row = listing_row(risk_level=level, risk_rule_id=rule)
    fake_session.rows = [row]

    analysis = detail(client, row.id).json()

    assert analysis["risk_level"] == level
    assert analysis["risk_rule_id"] == rule
    assert analysis["risk_rules_version"] == RULES_VERSION
    assert analysis["risk_calibration_id"] == CALIBRATION_ID


def test_an_analysis_with_no_decision_reports_null_rather_than_unknown(client, fake_session):
    """Null is not `UNKNOWN`, and the report has to be able to tell them apart."""
    row = listing_row(
        status="queued",
        risk_level=None,
        risk_rules_version=None,
        risk_rule_id=None,
        risk_calibration_id=None,
    )
    fake_session.rows = [row]

    analysis = detail(client, row.id).json()

    assert analysis["risk_level"] is None
    assert analysis["risk_rules_version"] is None
    assert analysis["risk_rule_id"] is None
    assert analysis["risk_calibration_id"] is None


def test_the_stored_decision_is_reported_even_when_the_score_beside_it_disagrees(
    client, fake_session
):
    """The invariant the report rests on: it renders the decision, it does not derive one.

    The score here is above `T_HIGH`, so a route that classified from the signal would answer
    `HIGH`. The stored decision says `MEDIUM`, and a forensic report that silently upgraded
    it would be asserting something no ruleset ever concluded.
    """
    row = listing_row(signal_score=0.999)
    fake_session.rows = [row]

    analysis = detail(client, row.id).json()

    assert analysis["risk_level"] == "MEDIUM"
    assert analysis["risk_rule_id"] == RULE_INDETERMINATE_BAND


def test_no_per_detector_risk_column_is_read(client, fake_session):
    """`analysis_signals` has a `risk_level` of its own. It is not the product decision."""
    row = listing_row()
    fake_session.rows = [row]

    detail(client, row.id)

    sql = compiled(fake_session)
    assert "analyses.risk_level" in sql
    for signal_table in (
        "analysis_signals",
        "analysis_signals_1",
        "analysis_signals_2",
        "analysis_signals_3",
    ):
        assert f"{signal_table}.risk_level" not in sql


# The four signals stay four independent facts. The report has a section per source, and a
# collapsed state there would be a claim the evidence does not make.


def test_the_four_signals_are_returned_independently(client, fake_session):
    row = listing_row()
    fake_session.rows = [row]

    analysis = detail(client, row.id).json()

    assert analysis["synthetic_video"]["status"] == "SUCCESS"
    assert analysis["provenance"]["status"] == "SUCCESS"
    assert analysis["active_speaker"]["status"] == "SUCCESS"
    assert analysis["audio_authenticity"]["status"] == "SUCCESS"


def test_a_failed_signal_is_distinguishable_from_an_absent_one(client, fake_session):
    """FAILED and absent are different forensic facts and must not arrive the same way.

    A detector that ran and failed produced a record of failing. A signal that is not there
    at all was never run — for an analysis stored before that source was wired in, there is
    nothing to report rather than a failure to report.
    """
    row = listing_row(
        signal_status="FAILED",
        signal_score=None,
        active_speaker_provider=None,
        active_speaker_signal_type=None,
        active_speaker_status=None,
        active_speaker_id=None,
    )
    fake_session.rows = [row]

    analysis = detail(client, row.id).json()

    assert analysis["synthetic_video"]["status"] == "FAILED"
    assert analysis["synthetic_video"]["score"] is None
    assert analysis["active_speaker"] is None


def test_an_analysis_with_no_signals_at_all_returns_none_for_each(client, fake_session):
    row = unsignalled_row()
    fake_session.rows = [row]

    analysis = detail(client, row.id).json()

    for signal in ("synthetic_video", "provenance", "active_speaker", "audio_authenticity"):
        assert analysis[signal] is None


# Query budget. One analysis costs what the listing costs, and no more.


def test_one_analysis_costs_four_statements(client, fake_session):
    """The same budget as the whole listing: the shape of the read does not change because
    one row comes back, and nothing is fetched per signal."""
    row = listing_row()
    fake_session.rows = [row]

    detail(client, row.id)

    assert len(fake_session.statements) == 4


def test_an_analysis_with_no_signals_reads_no_evidence_tables(client, fake_session):
    """Nothing to look evidence up for, so the three evidence statements are not issued."""
    row = unsignalled_row()
    fake_session.rows = [row]

    detail(client, row.id)

    assert len(fake_session.statements) == 1


# Dimensions. A rotated phone video has two different picture sizes — the one encoded in the
# original and the one a detector was handed — and the report is only truthful if it can
# tell them apart. What is under test here is that the endpoint carries both through
# untransformed, and that a row which never recorded the second says so rather than
# substituting the first.


def test_both_dimension_pairs_are_reported_separately(client, fake_session):
    """The rotated phone video: encoded landscape, analysed portrait.

    These are the figures the worker measured off the derivative, and the endpoint states
    them as such beside the original's own. Nothing here multiplies out `display_rotation`
    to reach the second pair — the pair is stored, and the rotation only explains it.
    """
    fake_session.rows = [
        listing_row(
            width=1920,
            height=1080,
            display_rotation=90,
            analyzed_width=1080,
            analyzed_height=1920,
        )
    ]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert (media["original_width"], media["original_height"]) == (1920, 1080)
    assert (media["analyzed_width"], media["analyzed_height"]) == (1080, 1920)
    assert media["display_rotation"] == 90


def test_unrotated_media_reports_the_same_pair_twice(client, fake_session):
    """The overwhelmingly common case, and the one that must not regress.

    Landscape video that needed no turning is analysed at the size it was encoded at. The
    two pairs agreeing is a measured result here, not a shortcut: the second pair was still
    probed off the artifact.
    """
    fake_session.rows = [
        listing_row(
            width=1920,
            height=1080,
            display_rotation=0,
            analyzed_width=1920,
            analyzed_height=1080,
        )
    ]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert (media["original_width"], media["original_height"]) == (1920, 1080)
    assert (media["analyzed_width"], media["analyzed_height"]) == (1920, 1080)


def test_natively_portrait_media_is_not_treated_as_rotated(client, fake_session):
    """A physically portrait encode carries no rotation and needs none."""
    fake_session.rows = [
        listing_row(
            width=1080,
            height=1920,
            display_rotation=0,
            analyzed_width=1080,
            analyzed_height=1920,
        )
    ]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert (media["original_width"], media["original_height"]) == (1080, 1920)
    assert (media["analyzed_width"], media["analyzed_height"]) == (1080, 1920)
    assert media["display_rotation"] == 0


def test_a_historical_row_reports_null_rather_than_the_encoded_size(client, fake_session):
    """The whole point of the nullable columns.

    An analysis from before the derivative was measured has no analysed geometry, and there
    is no honest value for it — the artifact it was detected against may or may not have
    been the shape the original was encoded at. Filling the gap with `original_width` would
    assert exactly the thing this pair was added to stop, so it stays null and the reader
    falls back to reporting the encoded size *as* the encoded size.
    """
    fake_session.rows = [
        listing_row(
            width=1920,
            height=1080,
            display_rotation=None,
            analyzed_width=None,
            analyzed_height=None,
        )
    ]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert media["analyzed_width"] is None
    assert media["analyzed_height"] is None
    # Null, not `0`: nobody probed this row for rotation, which is a different fact from a
    # probe that found none.
    assert media["display_rotation"] is None
    # The encoded pair is untouched and still reported — the gap costs the analysed figures
    # and nothing else.
    assert (media["original_width"], media["original_height"]) == (1920, 1080)


def test_the_dimension_columns_ride_the_same_statement(client, fake_session):
    """No extra query for the analysed pair: it is on the joined media row already."""
    fake_session.rows = [listing_row()]

    detail(client, fake_session.rows[0].id)

    sql = compiled(fake_session)
    for column in ("media_files.analyzed_width", "media_files.analyzed_height"):
        assert column in sql


# The compatibility aliases. `width`/`height` were the only dimension fields this object had
# before the two pairs were separated, and they always meant the original's coded size. They
# are kept meaning exactly that, so a reader written against the old shape keeps getting the
# figures it always got.


def test_the_legacy_names_alias_the_original_pair(client, fake_session):
    fake_session.rows = [listing_row(width=1920, height=1080)]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert (media["width"], media["height"]) == (1920, 1080)
    assert (media["original_width"], media["original_height"]) == (1920, 1080)


def test_the_legacy_names_never_follow_the_analysed_pair(client, fake_session):
    """The guardrail, on the row where the two pairs actually disagree.

    A rotated phone video is the whole reason these fields were split. A reader asking for
    `width` is asking the old question — what does the submitted file encode — and answering
    it with the analysed geometry would reintroduce the exact misreporting this task removed,
    in the one place nobody would think to look for it.
    """
    fake_session.rows = [
        listing_row(
            width=1920,
            height=1080,
            display_rotation=90,
            analyzed_width=1080,
            analyzed_height=1920,
        )
    ]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert (media["width"], media["height"]) == (1920, 1080)
    assert (media["analyzed_width"], media["analyzed_height"]) == (1080, 1920)
    # Stated as an inequality too, so a change that made the aliases track the analysed pair
    # fails here rather than passing on figures that happen to match.
    assert media["width"] != media["analyzed_width"]
    assert media["height"] != media["analyzed_height"]


def test_the_legacy_names_survive_a_row_with_no_analysed_pair(client, fake_session):
    """A historical row loses its analysed figures, never its encoded ones."""
    fake_session.rows = [
        listing_row(width=1920, height=1080, analyzed_width=None, analyzed_height=None)
    ]

    media = detail(client, fake_session.rows[0].id).json()["media"]

    assert (media["width"], media["height"]) == (1920, 1080)
    assert media["analyzed_width"] is None
