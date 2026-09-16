"""The two-axis execution state, from the projection to the page that renders it (R10-T3).

R10-T2 split execution into a Fast Decision Path and Deep Evidence Enrichment and derived
the two axes from the rows each leaves behind. This module is about the next question: what
a client is told, and what a client is allowed to work out for itself.

The answer to the second is *nothing*. The API projects `decision_state`,
`aggregate_enrichment_state` and the per-component states, and those projections are the
authority. A frontend cannot reach them by looking at the evidence it was sent: an absent
detector panel is a detector that failed, or abstained, or was never asked, and those are
three different facts about what is known. The last section of this module is a guard over
the report's own sources that proves it does not try.

Four claims, in order:

- the product mode a submission asks for reaches the enrichment trigger and nothing else;
- the projection is correct over every permutation that matters, legacy rows included;
- a `DECIDED` analysis is served in full while its enrichment is still running, and the
  verdict it carries does not move as that enrichment progresses;
- the report renders states rather than computing them, and renders `abstained` as the
  success it is.
"""

import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import enrichment
from app.api.analyses import product_mode
from app.db.models import (
    ENRICHMENT_TASK_STATUS_ABSTAINED,
    ENRICHMENT_TASK_STATUS_COMPLETED,
    ENRICHMENT_TASK_STATUS_FAILED,
    ENRICHMENT_TASK_STATUS_NOT_REQUESTED,
    ENRICHMENT_TASK_STATUS_PROCESSING,
    ENRICHMENT_TASK_STATUS_QUEUED,
    PRODUCT_MODE_DEEP_ANALYSIS,
    PRODUCT_MODE_QUICK_SCAN,
    AnalysisEnrichmentTask,
)
from app.db.session import SessionLocal, engine
from app.main import app

from tests.test_analysis_listing import (  # noqa: F401 — fixtures used by name
    admin,
    client,
    fake_session,
    listing_row,
)

LIP_COMPONENT = ("lipforensics", "lip_forensics")
AASIST_COMPONENT = ("aasist", "audio_authenticity")


def component_row(analysis_id, component, state):
    """A row shaped like the one `component_states` reads, with column names."""
    from types import SimpleNamespace

    return SimpleNamespace(
        analysis_id=analysis_id,
        provider=component[0],
        signal_type=component[1],
        status=state,
    )


def detail(client, analysis_id):
    return client.get(f"/api/v1/analyses/{analysis_id}")


# --------------------------------------------------------------------------------------
# Product modes: an execution choice that cannot reach a decision
# --------------------------------------------------------------------------------------


def test_an_absent_mode_is_no_mode_rather_than_a_default():
    """The whole of the backward-compatibility story, in one assertion.

    A client that has never heard of modes sends nothing and gets null, and null falls
    through to the deployment's own policy below — which is exactly what every submission
    got before this parameter existed.
    """
    assert product_mode(None) is None
    # What an HTML form posts for a control nobody touched. Read as absent, not as invalid:
    # refusing it would refuse submissions from the dashboard's own form.
    assert product_mode("") is None
    assert product_mode("   ") is None


def test_the_two_modes_are_accepted_and_normalised():
    assert product_mode("quick_scan") == PRODUCT_MODE_QUICK_SCAN
    assert product_mode("deep_analysis") == PRODUCT_MODE_DEEP_ANALYSIS
    assert product_mode("  Deep_Analysis  ") == PRODUCT_MODE_DEEP_ANALYSIS


def test_an_unknown_mode_is_refused_rather_than_defaulted():
    """Fails fast, and does not silently hand the caller the other mode.

    A caller that sent `thorough` meant something by it. Quietly giving them a quick scan
    would answer a question they did not ask, and they would have no way to find out.
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refusal:
        product_mode("thorough")

    assert refusal.value.status_code == 422
    assert "thorough" in refusal.value.detail


def test_deep_analysis_queues_and_quick_scan_defers(monkeypatch):
    """The only thing a mode decides: the status the component rows are written in."""
    monkeypatch.delenv(enrichment.ENRICHMENT_POLICY_VARIABLE, raising=False)

    assert (
        enrichment.initial_status(PRODUCT_MODE_DEEP_ANALYSIS)
        == ENRICHMENT_TASK_STATUS_QUEUED
    )
    assert (
        enrichment.initial_status(PRODUCT_MODE_QUICK_SCAN)
        == ENRICHMENT_TASK_STATUS_NOT_REQUESTED
    )


def test_a_named_mode_overrides_the_deployment_policy_in_both_directions(monkeypatch):
    """A mode is a decision this submission made; a policy is the default it overrides."""
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, enrichment.POLICY_DEFERRED)
    assert (
        enrichment.initial_status(PRODUCT_MODE_DEEP_ANALYSIS)
        == ENRICHMENT_TASK_STATUS_QUEUED
    )

    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, enrichment.POLICY_IMMEDIATE)
    assert (
        enrichment.initial_status(PRODUCT_MODE_QUICK_SCAN)
        == ENRICHMENT_TASK_STATUS_NOT_REQUESTED
    )


def test_no_mode_falls_through_to_the_deployment_policy(monkeypatch):
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, enrichment.POLICY_IMMEDIATE)
    assert enrichment.initial_status(None) == ENRICHMENT_TASK_STATUS_QUEUED

    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, enrichment.POLICY_DEFERRED)
    assert enrichment.initial_status(None) == ENRICHMENT_TASK_STATUS_NOT_REQUESTED


def test_an_unrecognised_stored_mode_is_read_as_no_mode(monkeypatch):
    """Fail-open, in the direction that produces a report with all of its evidence.

    Nothing can write such a value — the route refuses it and the check constraint refuses
    it again — but the reading is stated rather than left to whichever branch happens to
    catch it.
    """
    monkeypatch.setenv(enrichment.ENRICHMENT_POLICY_VARIABLE, enrichment.POLICY_IMMEDIATE)

    assert enrichment.initial_status("thorough") == ENRICHMENT_TASK_STATUS_QUEUED


def test_the_mode_is_read_nowhere_between_the_claim_and_the_verdict():
    """Why "the two modes decide identically" is a property of the code's shape.

    The worker carries the mode out of the claim and hands it to `enrichment.enqueue`, which
    runs after the verdict is published. If it were read anywhere else in that file, it
    could reach a detector, a threshold or the engine. Asserted over the source because what
    is being claimed is an absence, and an absence cannot be demonstrated by running one
    path through the file.
    """
    source = (Path(__file__).resolve().parents[1] / "app" / "worker.py").read_text(
        encoding="utf-8"
    )

    uses = re.findall(r"^.*enrichment_mode.*$", source, flags=re.MULTILINE)
    # The dataclass field, its comment block, the claim that reads it off the job row, and
    # the one argument it is passed as. Every line is either a declaration or the handover;
    # none of them is a branch.
    assert uses, "the worker no longer carries the mode at all"
    assert not any(
        line.strip().startswith(("if ", "elif ", "while ")) for line in uses
    ), uses


# --------------------------------------------------------------------------------------
# The projection: derived by the API, over every permutation that matters
# --------------------------------------------------------------------------------------


def test_a_decided_analysis_still_being_enriched_projects_both_axes(client, fake_session):
    """`DECIDED` beside `ENRICHMENT_PROCESSING` — the state R10 exists to create."""
    row = listing_row(status="completed", risk_level="MANIPULATION_DETECTED")
    fake_session.rows = [row]
    fake_session.enrichment_rows = [
        component_row(row.id, LIP_COMPONENT, ENRICHMENT_TASK_STATUS_PROCESSING),
        component_row(row.id, AASIST_COMPONENT, ENRICHMENT_TASK_STATUS_QUEUED),
    ]

    payload = detail(client, row.id).json()

    assert payload["decision_state"] == enrichment.DECIDED
    assert payload["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PROCESSING


def test_the_report_opens_while_enrichment_is_still_running(client, fake_session):
    """Enrichment state never gates report availability (§5.3).

    The verdict, its trace and every evidence field are served in full, with a 200, while
    supplementary detectors are still running. A report withheld until enrichment finished
    would give back exactly the latency R10 was built to remove.
    """
    row = listing_row(status="completed", risk_level="MANIPULATION_DETECTED")
    fake_session.rows = [row]
    fake_session.enrichment_rows = [
        component_row(row.id, LIP_COMPONENT, ENRICHMENT_TASK_STATUS_PROCESSING),
    ]

    response = detail(client, row.id)

    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_level"] == "MANIPULATION_DETECTED"
    assert payload["risk_rule_id"] is not None
    assert payload["risk_rules_version"] is not None
    assert payload["risk_calibration_id"] is not None
    assert payload["synthetic_video"] is not None


def test_the_verdict_payload_is_identical_across_every_enrichment_state(
    client, fake_session
):
    """The decision does not move as the enrichment axis does (invariant I3, read side).

    The same analysis row is served under each of the six live enrichment states and under
    the legacy projection, and the five decision fields and the trace are compared
    byte-for-byte across all of them. This is the API-side half of "the verdict DOM stays
    static"; the frontend half is the guard at the bottom of this module, which proves the
    verdict block is not handed an enrichment state at all.
    """
    row = listing_row(status="completed", risk_level="MANIPULATION_DETECTED")
    decisions = {}

    for states in (
        (),  # legacy: no task rows at all
        (ENRICHMENT_TASK_STATUS_NOT_REQUESTED, ENRICHMENT_TASK_STATUS_NOT_REQUESTED),
        (ENRICHMENT_TASK_STATUS_QUEUED, ENRICHMENT_TASK_STATUS_QUEUED),
        (ENRICHMENT_TASK_STATUS_PROCESSING, ENRICHMENT_TASK_STATUS_QUEUED),
        (ENRICHMENT_TASK_STATUS_COMPLETED, ENRICHMENT_TASK_STATUS_ABSTAINED),
        (ENRICHMENT_TASK_STATUS_COMPLETED, ENRICHMENT_TASK_STATUS_FAILED),
        (ENRICHMENT_TASK_STATUS_FAILED, ENRICHMENT_TASK_STATUS_FAILED),
    ):
        # The fake session answers by statement position, so each pass starts from a clean
        # count as a separate request would.
        fake_session.statements.clear()
        fake_session.rows = [row]
        fake_session.enrichment_rows = [
            component_row(row.id, component, state)
            for component, state in zip((LIP_COMPONENT, AASIST_COMPONENT), states)
        ]

        payload = detail(client, row.id).json()
        decisions[states] = {
            field: payload[field]
            for field in (
                "status",
                "decision_state",
                "risk_level",
                "risk_rules_version",
                "risk_rule_id",
                "risk_calibration_id",
                "risk_trace",
            )
        }

    verdicts = list(decisions.values())
    assert all(verdict == verdicts[0] for verdict in verdicts), decisions


def test_the_per_component_states_name_the_components_by_their_signal_pair(
    client, fake_session
):
    """`ENRICHMENT_PARTIAL` has to be traceable to which component did what."""
    row = listing_row(status="completed", risk_level="INCONCLUSIVE")
    fake_session.rows = [row]
    fake_session.enrichment_rows = [
        component_row(row.id, LIP_COMPONENT, ENRICHMENT_TASK_STATUS_FAILED),
        component_row(row.id, AASIST_COMPONENT, ENRICHMENT_TASK_STATUS_COMPLETED),
    ]

    payload = detail(client, row.id).json()

    assert payload["aggregate_enrichment_state"] == enrichment.ENRICHMENT_PARTIAL
    assert payload["per_component_state"] == [
        {
            "provider": "lipforensics",
            "signal_type": "lip_forensics",
            "state": "failed",
        },
        {
            "provider": "aasist",
            "signal_type": "audio_authenticity",
            "state": "completed",
        },
    ]


def test_an_abstention_reaches_the_client_as_an_abstention(client, fake_session):
    """Not translated into a failure at the boundary, where no renderer could recover it.

    A detector that found no trackable face or no audio stream answered the question it was
    asked. The aggregate says so too: an analysis whose every component abstained is
    complete, not partially enriched — there was nothing more to get.
    """
    row = listing_row(status="completed", risk_level="INCONCLUSIVE")
    fake_session.rows = [row]
    fake_session.enrichment_rows = [
        component_row(row.id, LIP_COMPONENT, ENRICHMENT_TASK_STATUS_ABSTAINED),
        component_row(row.id, AASIST_COMPONENT, ENRICHMENT_TASK_STATUS_ABSTAINED),
    ]

    payload = detail(client, row.id).json()

    assert payload["aggregate_enrichment_state"] == enrichment.ENRICHMENT_COMPLETE
    assert [component["state"] for component in payload["per_component_state"]] == [
        "abstained",
        "abstained",
    ]


def test_a_legacy_analysis_projects_as_single_stage_and_not_as_complete(
    client, fake_session
):
    """§8.1: a pre-R10 row says the full chain ran in one stage and nothing more.

    `ENRICHMENT_COMPLETE` would assert that every Deep Evidence component terminated *and*
    succeeded, which no legacy row can support — a pre-R10 analysis could carry a failed
    evidence-only signal and still reach `completed`.
    """
    row = listing_row(status="completed", risk_level="MEDIUM")
    fake_session.rows = [row]
    fake_session.enrichment_rows = []

    payload = detail(client, row.id).json()

    assert payload["decision_state"] == enrichment.DECIDED
    assert payload["aggregate_enrichment_state"] == enrichment.LEGACY_SINGLE_STAGE
    assert payload["per_component_state"] == []


def test_an_undecided_analysis_is_given_no_enrichment_reading(client, fake_session):
    """A state beside a decision that never happened invites reading the two as one."""
    queued = listing_row(
        status="queued",
        risk_level=None,
        risk_rules_version=None,
        risk_rule_id=None,
        risk_calibration_id=None,
    )
    fake_session.rows = [queued]

    payload = detail(client, queued.id).json()

    assert payload["decision_state"] == enrichment.DECISION_PENDING
    assert payload["aggregate_enrichment_state"] == enrichment.ENRICHMENT_NOT_APPLICABLE


def test_a_failed_analysis_projects_as_a_failed_decision(client, fake_session):
    failed = listing_row(
        status="failed",
        risk_level=None,
        risk_rules_version=None,
        risk_rule_id=None,
        risk_calibration_id=None,
    )
    fake_session.rows = [failed]

    payload = detail(client, failed.id).json()

    assert payload["decision_state"] == enrichment.DECISION_FAILED
    assert payload["aggregate_enrichment_state"] == enrichment.ENRICHMENT_NOT_APPLICABLE


def test_the_projection_is_read_from_the_task_rows_and_not_from_the_signals(
    client, fake_session
):
    """The reason the API can answer this and a renderer cannot.

    The row carries a full set of successful signals and no task rows at all. Anything
    deriving the state from the evidence would call that fully enriched; the projection
    calls it legacy, because that is what the execution record says.
    """
    row = listing_row(status="completed", risk_level="MEDIUM")
    fake_session.rows = [row]
    fake_session.enrichment_rows = []

    assert (
        detail(client, row.id).json()["aggregate_enrichment_state"]
        == enrichment.LEGACY_SINGLE_STAGE
    )


def test_the_states_are_projected_per_analysis_across_a_listing(client, fake_session):
    """One statement for the whole page, and no two analyses sharing an answer."""
    enriching = listing_row(status="completed", risk_level="MANIPULATION_DETECTED")
    legacy = listing_row(status="completed", risk_level="MEDIUM")
    fake_session.rows = [enriching, legacy]
    fake_session.enrichment_rows = [
        component_row(enriching.id, LIP_COMPONENT, ENRICHMENT_TASK_STATUS_PROCESSING),
    ]

    payload = {entry["id"]: entry for entry in client.get("/api/v1/analyses").json()}

    assert (
        payload[str(enriching.id)]["aggregate_enrichment_state"]
        == enrichment.ENRICHMENT_PROCESSING
    )
    assert (
        payload[str(legacy.id)]["aggregate_enrichment_state"]
        == enrichment.LEGACY_SINGLE_STAGE
    )


# --------------------------------------------------------------------------------------
# The report's side of the contract: it renders states, it does not compute them
# --------------------------------------------------------------------------------------
#
# Asserted from the sources rather than by rendering them, which is how this repository
# already checks properties of the web application from the backend suite — see the R7-T6
# and R9-T5 blocks in `tests/test_risk_trace.py`, and `tests/test_shadow_mode.py` before
# them. Rendering the component would check React. What is being checked here is mostly an
# *absence*, and an absence cannot be demonstrated by rendering one state and finding the
# page acceptable: the defect this guards against is a line of arithmetic that only fires on
# the combination nobody thought to render.

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
WEB_ANALYSIS = WEB_ROOT / "app" / "analysis.ts"
WEB_REPORT = WEB_ROOT / "app" / "app" / "report" / "[id]" / "page.tsx"

requires_web = pytest.mark.skipif(
    not WEB_ROOT.exists(), reason="the web application is not present"
)

# Every state name the API can put on either axis. The report may not contain one: it looks
# each state up in a table keyed by the API's own string, so a state name written into the
# renderer could only be there to branch on.
AGGREGATE_STATES = (
    "ENRICHMENT_NOT_REQUESTED",
    "ENRICHMENT_PENDING",
    "ENRICHMENT_PROCESSING",
    "ENRICHMENT_COMPLETE",
    "ENRICHMENT_PARTIAL",
    "ENRICHMENT_FAILED",
    "LEGACY_SINGLE_STAGE",
    "ENRICHMENT_NOT_APPLICABLE",
)

COMPONENT_STATES = (
    "not_requested",
    "queued",
    "processing",
    "completed",
    "abstained",
    "failed",
)

# What deriving a state out of evidence looks like in TypeScript. Counting rows, testing
# whether some or every component did something, folding a list into a summary, or comparing
# two lengths — the operations `ENRICHMENT_PARTIAL` would have to be assembled from.
DERIVATION_OPERATORS = (
    ".filter(",
    ".some(",
    ".every(",
    ".reduce(",
    ".includes(",
    ".find(",
    ".count",
    "++",
    "+=",
)


def _code_only(source: str) -> str:
    """The source with its comments removed.

    These guards are about what the code *does*, and every one of them is stated as the
    absence of something. A prose paragraph explaining why the report must not assemble
    `ENRICHMENT_PARTIAL` would otherwise fail the test that proves it does not — and the
    comments saying so are worth keeping, so the test learns to read code instead.
    """
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)

    return "\n".join(
        line for line in without_blocks.splitlines() if not line.strip().startswith("//")
    )


def _function_body(source: str, signature: str) -> str:
    """One function's body, from its signature to the next top-level declaration."""
    body = source.split(signature, 1)[1]

    for boundary in ("\nfunction ", "\nexport ", "\nconst "):
        body = body.split(boundary, 1)[0]

    return body


@requires_web
def test_the_report_names_no_enrichment_state_anywhere():
    """The guard, in one line: the renderer does not hold the vocabulary it renders.

    Every state the API can report is looked up in a table keyed by the API's own string.
    A state name written into the report could only be there to branch on, and a branch on
    a state is one edit away from a *derivation* of one — the report deciding for itself
    that two failures and a success make a partial enrichment.
    """
    report = _code_only(WEB_REPORT.read_text(encoding="utf-8"))

    for state in AGGREGATE_STATES:
        assert state not in report, state

    for state in COMPONENT_STATES:
        # As a quoted literal. The words themselves appear in ordinary prose on this page —
        # "completed", "failed" — and it is the comparison value that is forbidden.
        assert f'"{state}"' not in report, state
        assert f"'{state}'" not in report, state


@requires_web
def test_the_reports_enrichment_section_counts_nothing():
    """No arithmetic over the components, in the one component that is handed them."""
    body = _function_body(
        _code_only(WEB_REPORT.read_text(encoding="utf-8")),
        "function DeepEvidenceSection(",
    )

    for operator in DERIVATION_OPERATORS:
        assert operator not in body, operator

    # `.map` and a single `.length > 0` are what a list renderer is: one draws the entries,
    # the other decides whether the list is drawn at all. Neither reads a state, and that is
    # the property — the length is compared against zero and against nothing else.
    assert ".map(" in body
    assert body.count(".length") == 1
    assert ".length > 0" in body


@requires_web
def test_the_report_asks_the_shared_vocabulary_for_every_word_it_prints():
    """Each state reaches a sentence through a lookup, and through nothing else."""
    body = _function_body(
        WEB_REPORT.read_text(encoding="utf-8"),
        "function DeepEvidenceSection(",
    )

    assert "deepEvidenceWording(state)" in body
    assert "componentStateText(component.state)" in body
    assert "componentDetectorName(component)" in body


@requires_web
def test_the_verdict_is_never_handed_an_enrichment_state():
    """Why the verdict DOM cannot move while enrichment runs.

    The summary that states the verdict takes the persisted trace and nothing else, and the
    helper that picks the verdict out of that trace reads the same. Neither has an
    enrichment state in scope, so no enrichment transition can reach a character of what
    they render — which is a stronger claim than "the wording happens to be the same", and
    it is the one R10-T3 needs.
    """
    report = WEB_REPORT.read_text(encoding="utf-8")

    assert "<OperationalSummary trace={analysis.risk_trace} />" in report

    for signature in ("function OperationalSummary(", "function v5Verdict("):
        body = _function_body(_code_only(report), signature)
        for identifier in (
            "enrichment",
            "Enrichment",
            "decision_state",
            "per_component_state",
        ):
            assert identifier not in body, (signature, identifier)


@requires_web
def test_nothing_gates_the_report_on_an_execution_state():
    """A decided analysis opens, whatever its enrichment is doing (§5.3).

    The page's own guards are the three it always had — unauthenticated, missing, and a
    read that failed — and the verdict is rendered after them with no fourth condition in
    between. An enrichment state cannot gate this page because the page never reads one
    before it draws the verdict.
    """
    body = _function_body(
        _code_only(WEB_REPORT.read_text(encoding="utf-8")),
        "export default async function Report(",
    )
    before_verdict = body.split("<OperationalSummary", 1)[0]

    assert "result.unauthenticated" in before_verdict
    assert "result.missing" in before_verdict
    assert "aggregate_enrichment_state" not in before_verdict
    assert "decision_state" not in before_verdict


@requires_web
def test_the_wording_table_covers_every_state_the_api_can_report():
    """A state with no entry would render as nothing, silently.

    The vocabulary is asserted against `app.enrichment`'s own constants rather than against
    a list this test keeps, so a seventh enrichment state added to the API fails here on the
    day it is added instead of on the day somebody opens a report carrying it.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")

    api_states = {
        enrichment.ENRICHMENT_NOT_REQUESTED,
        enrichment.ENRICHMENT_PENDING,
        enrichment.ENRICHMENT_PROCESSING,
        enrichment.ENRICHMENT_COMPLETE,
        enrichment.ENRICHMENT_PARTIAL,
        enrichment.ENRICHMENT_FAILED,
        enrichment.LEGACY_SINGLE_STAGE,
        enrichment.ENRICHMENT_NOT_APPLICABLE,
    }

    for state in api_states:
        assert f'export const {state} = "{state}";' in source, state

    table = source.split("DEEP_EVIDENCE_WORDING: Record<", 1)[1].split("\n};", 1)[0]
    assert {
        key for key in re.findall(r"^  \[([A-Z_]+)\]:", table, flags=re.MULTILINE)
    } == api_states


@requires_web
def test_a_legacy_analysis_draws_no_enrichment_section_at_all():
    """No phantom state on a report from before any of this existed (§8.1).

    `LEGACY_SINGLE_STAGE` maps to no wording, so the section is absent rather than present
    and hedged. A "processing" indicator would describe work that will never run, and an
    "all supplementary evidence succeeded" line would be a claim no legacy row can support.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    table = source.split("DEEP_EVIDENCE_WORDING: Record<", 1)[1].split("\n};", 1)[0]

    assert "[LEGACY_SINGLE_STAGE]: null," in table
    assert "[ENRICHMENT_NOT_APPLICABLE]: null," in table

    # And nothing in the table claims a legacy report's evidence succeeded.
    assert "LEGACY_SINGLE_STAGE" not in _code_only(
        WEB_REPORT.read_text(encoding="utf-8")
    )


@requires_web
def test_an_abstention_is_not_worded_as_a_failure():
    """§5.2, at the only place a reader meets it.

    The detector was asked, it answered, and its answer is evidence. The two labels must be
    different sentences, and the abstention's must not contain the vocabulary of a broken
    detector — which is checked word by word rather than by comparing the two, because a
    label reading "failed to apply" would differ from the failure label and still be wrong.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    table = source.split("COMPONENT_STATE_LABELS: Record<", 1)[1].split("\n};", 1)[0]

    abstained = re.search(r"\[COMPONENT_ABSTAINED\]: \"([^\"]+)\"", table).group(1)
    failed = re.search(r"\[COMPONENT_FAILED\]: \"([^\"]+)\"", table).group(1)

    assert abstained != failed
    assert "Not applicable" in abstained
    assert "abstained" in abstained
    for word in ("fail", "Fail", "error", "Error", "unavailable", "broke"):
        assert word not in abstained, word

    # And the one that is a failure still says so, so the distinction is legible in both
    # directions rather than by one of them going quiet.
    assert "failed" in failed


@requires_web
def test_a_component_nobody_asked_is_not_a_component_that_failed():
    """The other distinction the per-component states exist to carry."""
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    table = source.split("COMPONENT_STATE_LABELS: Record<", 1)[1].split("\n};", 1)[0]

    not_requested = re.search(
        r"\[COMPONENT_NOT_REQUESTED\]: \"([^\"]+)\"", table
    ).group(1)

    assert not_requested == "Not requested"


@requires_web
def test_the_enrichment_section_is_printed_rather_than_hidden():
    """A PDF says what was true when it was rendered, and says that it does (§7.5).

    A report printed during enrichment is a truthful document about a decided analysis with
    evidence outstanding — not a draft. What makes it truthful is that the state it was
    rendered under is on the paper, so this section is never dropped from print, and it
    carries the sentence that dates it.
    """
    body = _function_body(
        WEB_REPORT.read_text(encoding="utf-8"),
        "function DeepEvidenceSection(",
    )

    assert "print:hidden" not in body
    assert "This states what had run when this page was rendered." in body
