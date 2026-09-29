"""R14-T4: detector health on the operational summary — per deployment, with failure and missing.

Written to the conventions of `test_admin_analytics.py` and reusing its helpers. Every assertion
against the live route is a **difference** across a seed or is made on a provider name unique to
this run, for the reason that file's docstring gives: the payload aggregates every row in the
deployment. The pure functions the route is built from are asserted absolutely.

What is asserted:

*Grouping.* One provider under two versions and under a null version is three entries, and the
null stays null — no placeholder string is invented for it.

*Arithmetic.* `failure_rate` is `(FAILED + TIMEOUT) / (SUCCESS + FAILED + TIMEOUT)`: 10/2/3 is
exactly 5/15. Null when nothing was run.

*No double count.* A completed analysis with a `FAILED` row for an expected detector moves that
detector's `FAILED` and leaves its `MISSING` where it was.

*Missing, and only when terminal.* A completed `r9-v5.0.0` analysis with no row for an expected
detector moves `MISSING`; a queued or failed analysis, a legacy-ruleset analysis, and a row from
another deployment of the same detector do not. An evidence-only detector never carries the key.

*Missing is windowed on completion.* An analysis submitted eight days ago and completed today is
counted; one completed eight days ago is not.

*Separation.* The evaluation code is not touched by this route.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.api.admin_analytics import (
    SIGNAL_MISSING,
    SIGNAL_STATUSES,
    DetectorHealthMetric,
    detector_health,
    expected_signals,
)
from app.db.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_QUEUED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_QUEUED,
    SIGNAL_STATUS_FAILED,
    SIGNAL_STATUS_SUCCESS,
    SIGNAL_STATUS_TIMEOUT,
    AnalysisJob,
)
from app.risk_trace import RULESET_V5
from tests.test_admin_analytics import (  # noqa: F401
    LEGACY_LAST,
    V5,
    administrator,
    database,
    detector,
    make_analysis,
    make_signal,
    plain_http_environment,
    read,
    session,
    unique_provider,
)
from tests.test_operational_summary_taxonomy import (
    PAYLOAD as TAXONOMY_PAYLOAD,
    WEB_ANALYTICS_PAGE,
    parse_with_node,
    requires_node,
    requires_web,
)

pytestmark = pytest.mark.integration

# The two decision-eligible detectors of `r9-v5.0.0`, and its evidence-only one, by the
# deployment the ruleset froze. Providers and signal types are written out rather than read off
# the module under test, so a ruleset that stopped expecting one fails here.
V5_SIGNALS = {signal.provider: signal for signal in RULESET_V5.signals}
NVIDIA = V5_SIGNALS["nvidia"]
FACE = V5_SIGNALS["efficientnet-b7"]
LIP = V5_SIGNALS["lipforensics"]


def no_rows() -> dict[tuple[str, str | None], int]:
    return {}


# --- the contract, on the pure functions ----------------------------------------------------


def test_only_the_v5_decision_eligible_detectors_are_expected():
    """The coverage model starts at `r9-v5.0.0`, and only its two decision-eligible detectors are
    expected. The mouth-dynamics detector is evidence-only and nothing before v5 expects
    anything."""
    assert sorted(expected_signals()) == sorted(
        [
            (V5, "nvidia", "synthetic_video", NVIDIA.provider_version),
            (V5, "efficientnet-b7", "face_manipulation", FACE.provider_version),
        ]
    )


def test_one_provider_under_three_versions_is_three_entries():
    health = detector_health(
        [
            ("p", "v1", SIGNAL_STATUS_SUCCESS, 2),
            ("p", "v2", SIGNAL_STATUS_FAILED, 1),
            ("p", None, SIGNAL_STATUS_TIMEOUT, 4),
        ],
        no_rows(),
    )

    assert [(m.provider, m.provider_version) for m in health] == [
        ("p", None),
        ("p", "v1"),
        ("p", "v2"),
    ]
    assert health[0].status_counts[SIGNAL_STATUS_TIMEOUT] == 4
    assert health[1].status_counts[SIGNAL_STATUS_SUCCESS] == 2
    assert health[2].status_counts[SIGNAL_STATUS_FAILED] == 1


def test_failure_rate_is_failed_and_timeout_over_every_run():
    [metric] = detector_health(
        [
            ("p", "v", SIGNAL_STATUS_SUCCESS, 10),
            ("p", "v", SIGNAL_STATUS_FAILED, 2),
            ("p", "v", SIGNAL_STATUS_TIMEOUT, 3),
        ],
        no_rows(),
    )

    assert metric.failure_rate == 5 / 15


def test_missing_is_outside_the_rate_and_a_deployment_with_no_runs_has_no_rate():
    """A deployment listed only because its expected signals went missing ran nothing, so there
    is no rate — null, never zero — and `MISSING` enters neither half of it."""
    missing_only, with_runs = sorted(
        detector_health(
            [("q", "v", SIGNAL_STATUS_SUCCESS, 1)],
            {("p", "v"): 4, ("q", "v"): 9},
        ),
        key=lambda metric: metric.provider,
    )

    assert missing_only.failure_rate is None
    assert missing_only.status_counts == {
        **{status: 0 for status in SIGNAL_STATUSES},
        SIGNAL_MISSING: 4,
    }
    assert with_runs.failure_rate == 0
    assert with_runs.status_counts[SIGNAL_MISSING] == 9


def test_an_expected_deployment_with_nothing_missing_reads_zero_and_an_unexpected_one_has_no_key():
    health = detector_health(
        [("expected", "v", SIGNAL_STATUS_SUCCESS, 1), ("other", "v", SIGNAL_STATUS_SUCCESS, 1)],
        {("expected", "v"): 0, ("quiet", "v"): 0},
    )

    by_provider = {metric.provider: metric for metric in health}
    assert set(by_provider) == {"expected", "other"}
    assert by_provider["expected"].status_counts[SIGNAL_MISSING] == 0
    assert SIGNAL_MISSING not in by_provider["other"].status_counts


def test_an_unrecognised_status_is_carried_and_left_out_of_the_rate():
    [metric] = detector_health(
        [("p", "v", SIGNAL_STATUS_SUCCESS, 1), ("p", "v", "UNAVAILABLE", 5)], no_rows()
    )

    assert metric.status_counts["UNAVAILABLE"] == 5
    assert metric.failure_rate == 0


def test_the_version_is_declared_nullable_on_the_wire():
    schema = DetectorHealthMetric.model_json_schema()["properties"]
    assert {"type": "null"} in schema["provider_version"]["anyOf"]
    assert {"type": "null"} in schema["failure_rate"]["anyOf"]


# --- the live route ---------------------------------------------------------------------------


def health_entry(payload: dict, provider: str, version: str | None) -> dict | None:
    for entry in payload["detectors_health"]:
        if entry["provider"] == provider and entry["provider_version"] == version:
            return entry
    return None


def missing(payload: dict, signal) -> int:
    """`MISSING` on an expected deployment, zero while it is not yet listed."""
    counts = detector(payload, signal.provider, signal.provider_version)
    return 0 if counts is None else counts[SIGNAL_MISSING]


def status(payload: dict, signal, name: str) -> int:
    counts = detector(payload, signal.provider, signal.provider_version)
    return 0 if counts is None else counts[name]


def eight_days_ago() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=8)


def closed_analysis(
    session,
    *,
    status: str = ANALYSIS_STATUS_COMPLETED,
    rules_version: str | None = V5,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
):
    """An analysis and its one job, in matching states, as `app.worker._set_status` leaves them.

    `completed_at` is the job's `updated_at`, which is the completion time the missing count is
    windowed on. Left unset it is the column's `now()` default, as a real completion's is.
    """
    job_status = {
        ANALYSIS_STATUS_COMPLETED: JOB_STATUS_COMPLETED,
        ANALYSIS_STATUS_FAILED: JOB_STATUS_FAILED,
        ANALYSIS_STATUS_QUEUED: JOB_STATUS_QUEUED,
    }[status]
    analysis = make_analysis(
        session, status=status, rules_version=rules_version, created_at=created_at
    )

    db, _, _ = session
    job = AnalysisJob(analysis_id=analysis.id, status=job_status)
    if created_at is not None:
        job.created_at = created_at
    if completed_at is not None:
        job.updated_at = completed_at
    db.add(job)
    db.commit()

    return analysis


def expected_row(session, analysis, signal, *, run: str = SIGNAL_STATUS_SUCCESS) -> None:
    make_signal(
        session,
        analysis,
        provider=signal.provider,
        signal_type=signal.signal_type,
        provider_version=signal.provider_version,
        status=run,
    )


def test_the_route_returns_each_version_of_a_provider_separately_and_null_as_null(session):
    provider = unique_provider()
    for version in ("v1", "v2", None):
        make_signal(
            session, make_analysis(session), provider=provider, provider_version=version
        )

    payload = read(administrator(session))

    listed = [
        entry["provider_version"]
        for entry in payload["detectors_health"]
        if entry["provider"] == provider
    ]
    assert sorted(listed, key=lambda version: version or "") == [None, "v1", "v2"]
    assert "unknown" not in listed


def test_the_route_computes_ten_two_three_as_one_third(session):
    provider = unique_provider()
    analysis = make_analysis(session)
    for run, times in (
        (SIGNAL_STATUS_SUCCESS, 10),
        (SIGNAL_STATUS_FAILED, 2),
        (SIGNAL_STATUS_TIMEOUT, 3),
    ):
        for _ in range(times):
            make_signal(session, analysis, provider=provider, provider_version="v", status=run)

    entry = health_entry(read(administrator(session)), provider, "v")

    assert entry["failure_rate"] == pytest.approx(0.333, abs=5e-4)
    assert entry["failure_rate"] == 5 / 15
    assert SIGNAL_MISSING not in entry["status_counts"]


def test_a_failed_row_is_a_failure_and_not_also_missing(session):
    client = administrator(session)
    before = read(client)

    analysis = closed_analysis(session)
    expected_row(session, analysis, NVIDIA, run=SIGNAL_STATUS_FAILED)
    expected_row(session, analysis, FACE, run=SIGNAL_STATUS_TIMEOUT)

    after = read(client)

    assert status(after, NVIDIA, SIGNAL_STATUS_FAILED) == status(
        before, NVIDIA, SIGNAL_STATUS_FAILED
    ) + 1
    assert status(after, FACE, SIGNAL_STATUS_TIMEOUT) == status(
        before, FACE, SIGNAL_STATUS_TIMEOUT
    ) + 1
    assert missing(after, NVIDIA) == missing(before, NVIDIA)
    assert missing(after, FACE) == missing(before, FACE)


def test_a_completed_analysis_without_an_expected_row_is_missing_it(session):
    client = administrator(session)
    before = read(client)

    analysis = closed_analysis(session)
    expected_row(session, analysis, FACE)

    after = read(client)

    assert missing(after, NVIDIA) == missing(before, NVIDIA) + 1
    assert missing(after, FACE) == missing(before, FACE)


def test_in_flight_failed_and_legacy_analyses_are_never_missing_anything(session):
    """No row for either expected detector on any of them, and nothing moves: a queued analysis
    has not run yet, a failed one stopped before its detectors could, and a legacy ruleset
    predates the coverage model and expected nothing."""
    client = administrator(session)
    before = read(client)

    closed_analysis(session, status=ANALYSIS_STATUS_QUEUED, rules_version=None)
    closed_analysis(session, status=ANALYSIS_STATUS_QUEUED)
    closed_analysis(session, status=ANALYSIS_STATUS_FAILED)
    closed_analysis(session, rules_version=LEGACY_LAST)

    after = read(client)

    assert missing(after, NVIDIA) == missing(before, NVIDIA)
    assert missing(after, FACE) == missing(before, FACE)


def test_an_analysis_submitted_long_ago_and_completed_today_is_missing_its_signal_today(session):
    """The regression the window anchor exists for: submitted eight days ago, stuck in the queue,
    completed today without an expected row. That is this week's missing signal, and windowing
    on `Analysis.created_at` would have dropped it."""
    client = administrator(session)
    before = read(client)

    analysis = closed_analysis(session, created_at=eight_days_ago())
    expected_row(session, analysis, FACE)

    after = read(client)

    assert missing(after, NVIDIA) == missing(before, NVIDIA) + 1


def test_an_analysis_completed_before_the_window_is_not_missing_anything_now(session):
    client = administrator(session)
    before = read(client)

    closed_analysis(session, created_at=eight_days_ago(), completed_at=eight_days_ago())

    after = read(client)

    assert missing(after, NVIDIA) == missing(before, NVIDIA)
    assert missing(after, FACE) == missing(before, FACE)


def test_a_row_from_another_deployment_is_present_not_missing(session):
    """The expected detector answered, from a deployment the ruleset was not calibrated on. That
    row is counted under its own version and is not a missing signal."""
    other_version = f"{NVIDIA.provider_version}-other"
    client = administrator(session)
    before = read(client)

    analysis = closed_analysis(session)
    make_signal(
        session,
        analysis,
        provider=NVIDIA.provider,
        signal_type=NVIDIA.signal_type,
        provider_version=other_version,
    )
    expected_row(session, analysis, FACE)

    after = read(client)

    assert missing(after, NVIDIA) == missing(before, NVIDIA)
    assert SIGNAL_MISSING not in detector(after, NVIDIA.provider, other_version)


def test_an_evidence_only_detector_never_carries_a_missing_count(session):
    analysis = closed_analysis(session)
    expected_row(session, analysis, LIP)

    counts = detector(read(administrator(session)), LIP.provider, LIP.provider_version)

    assert SIGNAL_MISSING not in counts


# --- the web half of the contract -----------------------------------------------------------


HEALTH = [
    {
        "provider": "nvidia",
        "provider_version": None,
        "status_counts": {"SUCCESS": 10, "FAILED": 2, "TIMEOUT": 3, "MISSING": 1},
        "failure_rate": 5 / 15,
    },
    {
        "provider": "idle",
        "provider_version": "v",
        "status_counts": {"SUCCESS": 0, "FAILED": 0, "TIMEOUT": 0, "MISSING": 4},
        "failure_rate": None,
    },
]


def with_health(entries) -> dict:
    return {**TAXONOMY_PAYLOAD, "detectors_health": entries}


WEB_CASES = {
    "full": with_health(HEALTH),
    "rate_above_one": with_health([{**HEALTH[0], "failure_rate": 1.5}]),
    "numeric_version": with_health([{**HEALTH[0], "provider_version": 3}]),
    "missing_version_key": with_health(
        [{k: v for k, v in HEALTH[0].items() if k != "provider_version"}]
    ),
    "old_map_contract": {
        **{k: v for k, v in TAXONOMY_PAYLOAD.items() if k != "detectors_health"},
        "detectors": {"nvidia": {"SUCCESS": 1}},
    },
}


@requires_node
def test_the_web_parses_detector_health_as_the_api_sent_it():
    parsed = parse_with_node(WEB_CASES)

    assert parsed["full"]["detectors_health"] == HEALTH
    for case in ("rate_above_one", "numeric_version", "missing_version_key", "old_map_contract"):
        assert parsed[case] is None, case


@requires_web
def test_the_detector_section_carries_its_disclaimer_and_computes_nothing():
    source = WEB_ANALYTICS_PAGE.read_text(encoding="utf-8")
    table = source.split("function Detectors(", 1)[1].split("\nfunction ", 1)[0]
    flat = " ".join(source.split())

    assert "These are operational execution errors, not model evaluation errors" in flat
    assert "not a false-positive or error rate of any detector" in flat
    assert "not a model error or false-positive rate" in flat
    assert "rateText(detector.failure_rate)" in table
    assert '"unknown"' not in table
    # The page reads a count only to draw it, and divides nothing to produce the rate.
    assert table.count("status_counts[") == 1
    assert 'status_counts[status] ?? "—"' in table
    assert "reduce(" not in table
