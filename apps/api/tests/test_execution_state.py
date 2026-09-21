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

from app import enrichment, risk_engine
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
    assert "componentDetectorName(component)" in body

    # The component sentence takes the state and what the record holds for that component,
    # and the second argument is a call into the same shared vocabulary rather than a test
    # the renderer performs. Asserted as the two calls it is: a renderer that inlined the
    # reading check would have the signal vocabulary in it, which is the thing this module
    # exists to keep out.
    assert "componentStateText(" in body
    assert "component.state," in body
    assert "componentReadingPresence(analysis, component)" in body


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
    assert "abstention" in abstained
    for word in ("fail", "Fail", "error", "Error", "unavailable", "broke"):
        assert word not in abstained, word

    # And the one that is a failure still says so, so the distinction is legible in both
    # directions rather than by one of them going quiet.
    assert "Failed" in failed

    # M1 strengthens this rather than relaxing it. Both successes on the enrichment axis now
    # open with the same word, so the family a component belongs to is legible from the
    # sentence alone: an abstention reads as a kind of completion, which is what the API
    # means by writing it, and not merely as "not the failure sentence".
    completed = re.search(r"\[COMPONENT_COMPLETED\]: \"([^\"]+)\"", table).group(1)

    assert completed.startswith("Completed")
    assert abstained.startswith("Completed")
    assert not failed.startswith("Completed")


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


# ---------------------------------------------------------------------------------------
# M1 — presentation state against reading presence
#
# The failure these close was found by running a quick scan beside a deep analysis and
# reading both reports. A report can hold a final stored verdict while enrichment is still
# mid-flight, and in that window the component sentences were the only thing standing between
# a reader and the wrong conclusion: a detector that had not answered *yet* was worded exactly
# like one that had finished and found nothing.
#
# These are source guards for the same reason every other web guard in this module is. The
# repository runs no JavaScript test runner, and the property being protected is a property of
# the wording tables — which sentence a state and a reading resolve to — rather than of a
# rendering. What the tables hold is what the reader gets.
# ---------------------------------------------------------------------------------------

# Wording a reader could take as "this detector is finished and there is nothing". Every one
# of these is a true sentence about a terminal component and a false one about a running one.
TERMINAL_ABSENCE_WORDING = (
    "no stored signal",
    "no evidence",
    "nothing recorded",
    "no reading recorded",
    "not recorded",
    "no result",
)


def _reading_labels(source: str) -> str:
    """The reading-aware overlay table, which is the only place a reading changes a word."""
    return source.split("COMPONENT_STATE_READING_LABELS: Record<", 1)[1].split(
        "\n};", 1
    )[0]


@requires_web
def test_a_processing_component_with_no_reading_yet_is_never_worded_as_terminal():
    """The mandatory one: `DECIDED` + `ENRICHMENT_PROCESSING` + nothing on the record.

    A detector that is still running has produced no reading *yet*, and that is a statement
    about a job rather than about the media. The sentence has to carry the "yet" — without it
    a reader meeting an empty panel under a final assessment concludes the detector looked and
    found nothing, which is the one reading this state must not permit.
    """
    table = _reading_labels(WEB_ANALYSIS.read_text(encoding="utf-8"))
    processing = re.search(
        r"\[COMPONENT_PROCESSING\]: \{\s*\[READING_ABSENT\]: \"([^\"]+)\"", table
    ).group(1)

    assert processing == "Processing — no reading available yet"
    assert "yet" in processing

    for wording in TERMINAL_ABSENCE_WORDING:
        assert wording not in processing.lower(), wording


@requires_web
def test_a_completed_component_with_no_reading_is_not_normalised_into_a_terminal_zero():
    """The opposite error, and it is not fixed by making the sentence harmless.

    Under the API's contract `completed` means the detector produced a reading: one that was
    asked and found nothing to score is written `abstained`, and one that broke is written
    `failed`. So `completed` with no successful signal on the payload is an inconsistency
    between the task row and the signal row, and the sentence says that the reading is not in
    this record rather than implying the detector terminated with a zero result. No backend
    state is invented for it, here or in the web build.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    completed = re.search(
        r"\[COMPONENT_COMPLETED\]: \{\s*\[READING_ABSENT\]: \"([^\"]+)\"",
        _reading_labels(source),
    ).group(1)

    assert completed == "Completed — reading not found in this record"

    # It describes the record, not the media, and it is not one of the sentences that would
    # read as "this detector looked and there was nothing".
    for wording in ("no evidence", "nothing recorded", "no stored signal"):
        assert wording not in completed.lower(), wording

    # And no state was invented to carry it. The overlay picks a sentence for a state the API
    # already reports; it does not add a seventh component state to the vocabulary.
    assert "completed_no_signal" not in source
    for state in COMPONENT_STATES:
        assert f'export const COMPONENT_{state.upper()}' in source or state in (
            "not_requested",
            "queued",
            "processing",
            "completed",
            "abstained",
            "failed",
        )


@requires_web
def test_the_reading_overlay_covers_only_the_two_states_a_reading_can_change():
    """A reading chooses between sentences a state permits; it never chooses a state.

    Four of the six states say the same thing whatever is on the record — a queued,
    not-requested, abstained or failed component's sentence follows from its state — so only
    `processing` and `completed` are overridable. An overlay entry on any of the other four
    would be the record contradicting the API about what a detector did.
    """
    table = _reading_labels(WEB_ANALYSIS.read_text(encoding="utf-8"))
    overridden = set(re.findall(r"^  \[(COMPONENT_[A-Z_]+)\]:", table, flags=re.MULTILINE))

    assert overridden == {"COMPONENT_PROCESSING", "COMPONENT_COMPLETED"}

    # A component with a reading on the record is never overridden: the state's own sentence
    # is already the one that says a reading was recorded, and an entry here could only
    # contradict it.
    assert "[READING_PRESENT]" not in table

    # `processing` is overridden for the absent reading alone. Its base sentence is
    # execution-state-only, so an unobservable component still running is already described
    # exactly and needs no entry.
    processing = table.split("[COMPONENT_PROCESSING]:", 1)[1].split("},", 1)[0]
    assert "[READING_ABSENT]" in processing
    assert "[READING_UNOBSERVABLE]" not in processing


@requires_web
def test_the_four_states_a_reading_cannot_change_keep_their_own_sentences():
    """The rest of the matrix, asserted as the table it is."""
    table = WEB_ANALYSIS.read_text(encoding="utf-8").split(
        "COMPONENT_STATE_LABELS: Record<", 1
    )[1].split("\n};", 1)[0]

    labels = dict(re.findall(r"\[(COMPONENT_[A-Z_]+)\]: \"([^\"]+)\"", table))

    assert labels["COMPONENT_NOT_REQUESTED"] == "Not requested"
    assert labels["COMPONENT_QUEUED"] == "Pending"
    assert labels["COMPONENT_ABSTAINED"] == "Completed with abstention"
    assert labels["COMPONENT_FAILED"] == "Failed"

    # The two that a reading can change still have a sentence of their own, which is what a
    # component with a reading — and an unobservable one — resolves to.
    assert labels["COMPONENT_PROCESSING"] == "Processing"
    assert labels["COMPONENT_COMPLETED"] == "Completed with reading"

    # The vocabulary is the API's six and no more. `pending` and `unavailable` are not
    # component states in this contract — `ENRICHMENT_PENDING` is an *aggregate* state and
    # nothing projects a component as unavailable — so neither is given an entry here. A
    # seventh state reaching an older build is reported as uninterpretable rather than guessed.
    assert set(labels) == {f"COMPONENT_{state.upper()}" for state in COMPONENT_STATES}


@requires_web
def test_reading_presence_is_read_from_the_signals_status_and_not_from_its_contents():
    """A terminal zero-result is a reading, and is not scored as a missing one.

    A successful active-speaker signal with no segments and a successful audio-authenticity
    signal with no windows are detectors that ran and answered — "nobody was speaking", "no
    windows were stored". Deciding presence on the contents would turn both into "no reading",
    which is the §5.2 distinction collapsing at the last step in the chain that keeps it.
    """
    body = _function_body(
        _code_only(WEB_ANALYSIS.read_text(encoding="utf-8")),
        "export function componentReadingPresence(",
    )

    assert "signal.status === SIGNAL_STATUS_SUCCESS" in body

    # Nothing in it looks past the status at what the signal carries.
    for contents in (
        "segments",
        "windows",
        "score",
        "total_speaking_segments",
        ".length",
    ):
        assert contents not in body, contents

    # And it derives no aggregate on the way. This is one component at a time, by construction.
    for operator in DERIVATION_OPERATORS:
        assert operator not in body, operator


@requires_web
def test_a_component_this_build_cannot_see_is_not_reported_as_having_no_reading():
    """`face_forgery` is the case a boolean would have had to lie about.

    Four components are Deep Evidence under r9-v5.0.0 and the API projects a signal field for
    only three of them: `face_forgery` is EFFORT's own signal type and `AnalysisSummary`
    carries no field for it. Answering "no reading" there would report this renderer's blind
    spot as a fact about the analysis, and routing it to `face_manipulation` would report
    EfficientNet-B7's score as EFFORT's — a different model on a different scale. So the
    answer is a third one, and its sentence is the state's own.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    body = _function_body(_code_only(source), "function componentSignal(")

    # Every signal type it does answer for is a field the API actually projects, and the pair
    # it must never join is not joined.
    assert "analysis.lip_forensics" in body
    assert "analysis.active_speaker" in body
    assert "analysis.audio_authenticity" in body
    assert "face_forgery" not in body
    assert "face_manipulation" not in body
    assert "return undefined;" in body

    # The membership this is asserted against is the API's own, so a fourth projected signal —
    # or a fifth component — fails here on the day it lands rather than on the day a reader
    # meets a sentence about it.
    components = {
        signal_type
        for _, signal_type in enrichment.evidence_only_components(
            risk_engine.RULES_VERSION_V5
        )
    }
    assert components == {
        "lip_forensics",
        "face_forgery",
        "active_speaker",
        "audio_authenticity",
    }

    assert (
        'export const READING_UNOBSERVABLE: ComponentReadingPresence = '
        '"READING_UNOBSERVABLE";' in source
    )


@requires_web
def test_a_snapshot_notice_marks_the_report_and_not_the_decision():
    """§4 — the report may be intermediate while the decision is already final.

    The notice exists so a reader knows the panels below are still changing. Its whole
    difficulty is that it must not be readable as a hedge on the assessment, which was taken,
    persisted, and cannot be reached by any detector still running under this ruleset.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    notice = source.split("ENRICHMENT_SNAPSHOT_NOTICE =", 1)[1].split(";", 1)[0]

    assert "snapshot" in notice.lower()
    assert "processing" in notice.lower()
    assert "may not have produced a reading yet" in notice

    # It says the assessment is final, in those words, rather than leaving it to be inferred
    # from the absence of a hedge.
    assert "final" in notice
    assert "not provisional" in notice

    # And carries none of the vocabulary that would qualify the decision.
    for hedge in ("draft", "preliminary", "subject to change", "may change", "provisional "):
        assert hedge not in notice.lower(), hedge


@requires_web
def test_only_a_running_enrichment_draws_the_snapshot_notice():
    """One state earns it, and the renderer is not the thing that knows which.

    The comparison lives in the shared vocabulary beside every other sentence this page
    prints, so the report holds no enrichment state name to branch on — the property the
    guard at the top of this section protects, and the reason the notice could not simply be
    written as a conditional in the page.
    """
    body = _function_body(
        _code_only(WEB_ANALYSIS.read_text(encoding="utf-8")),
        "export function enrichmentSnapshotNotice(",
    )

    assert "state === ENRICHMENT_PROCESSING" in body

    for state in AGGREGATE_STATES:
        if state != "ENRICHMENT_PROCESSING":
            assert state not in body, state

    report = _code_only(WEB_REPORT.read_text(encoding="utf-8"))
    assert "enrichmentSnapshotNotice(state)" in report
    assert "<EnrichmentSnapshotNotice analysis={analysis} />" in report

    # Drawn under the assessment and above the first detector panel, so it is read before the
    # empty panels it is about rather than after them.
    assert report.index("<EnrichmentSnapshotNotice") > report.index("<OperationalSummary")
    assert report.index("<EnrichmentSnapshotNotice") < report.index(
        "<DeepEvidenceSection"
    )

    # And never hidden from print, for the same reason the enrichment section is not: a PDF
    # taken mid-enrichment is truthful only if what was outstanding is on the paper.
    notice_body = _function_body(
        WEB_REPORT.read_text(encoding="utf-8"), "function EnrichmentSnapshotNotice("
    )
    assert "print:hidden" not in notice_body


@requires_web
def test_the_footer_does_not_contradict_a_manipulation_assessment():
    """§5 — the closing sentence had to stop denying what the page above it says.

    It used to read that nothing in the document states the media is genuine or manipulated,
    on a report that can correctly carry "Manipulation detected." A reader who noticed had no
    way to tell which of the two to believe. The replacement keeps all three things that were
    actually true — no binary Fake/Real determination, a calibrated manipulation assessment
    under a named ruleset, provenance separate from it — and denies none of them.
    """
    # Read as code, like every other guard here. The comment above the sentence quotes the
    # wording it replaced — which is worth keeping, and is exactly what this test forbids.
    report = _code_only(WEB_REPORT.read_text(encoding="utf-8"))
    footer = report.split("<footer", 1)[1]

    # The contradiction itself, gone in both of the shapes it could come back in.
    assert "genuine or manipulated" not in footer
    assert "is genuine or manipulated" not in report

    assert "binary Fake/Real authenticity determination" in footer
    assert "calibrated manipulation evidence" in footer
    assert "stated ruleset" in footer

    # Provenance is stated as separate and as non-determining, which is the invariant the old
    # sentence carried by accident and the new one has to carry on purpose.
    assert "Provenance is reported separately" in footer
    assert "does not determine" in footer


@requires_web
def test_the_footer_makes_no_authenticity_claim_on_an_inconclusive_report():
    """The other direction: the replacement must not have bought consistency with a claim.

    The same footer is printed under `INCONCLUSIVE` and under `MANIPULATION_DETECTED` — it is
    not branched on the verdict — so it may not assert anything about the media in either
    direction. "Where applicable" is what carries that: a report with no calibrated
    manipulation evidence has nothing for the clause to apply to, and the sentence still reads
    true.
    """
    report = _code_only(WEB_REPORT.read_text(encoding="utf-8"))
    footer = report.split("<footer", 1)[1]

    assert "Where applicable" in footer

    # No word in it says the media is authentic, genuine, real, unaltered or clean.
    for claim in ("is genuine", "is authentic", "is real", "is unaltered", "verified as"):
        assert claim not in footer, claim

    # And it is one sentence for every report: nothing in the footer reads the verdict.
    for identifier in ("risk_level", "risk_trace", "v5Verdict", "classificationLabel"):
        assert identifier not in footer, identifier


@requires_web
def test_reading_presence_is_three_valued_in_its_name_and_in_its_type():
    """The helper is not a predicate, and nothing about it may suggest it is.

    It answers present, absent, or not-observable-by-this-build, and the third is truthy. A
    name in the `hasX` shape invites `if (hasX(...))`, which would read "this component has a
    reading" about the one case where this build cannot tell — silently reinstating the
    boolean lie the third answer exists to avoid. So the name states the three-valued thing it
    returns, and the type spells the three answers out rather than deriving them from a
    `string`-typed constant, which would widen to `string` and type-check every misuse.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")

    assert "export function componentReadingPresence(" in source
    assert "componentHasReading" not in source

    # The union is written out, so `tsc` refuses a fourth answer and a bare string.
    union = source.split("export type ComponentReadingPresence =", 1)[1].split(";", 1)[0]
    assert set(re.findall(r'"([A-Z_]+)"', union)) == {
        "READING_PRESENT",
        "READING_ABSENT",
        "READING_UNOBSERVABLE",
    }
    assert "typeof" not in union

    # And the sentence lookup takes the presence as a required argument. A default would let a
    # caller omit it and get the unobservable wording for a component this build can see.
    signature = source.split("export function componentStateText(", 1)[1].split(
        ")", 1
    )[0]
    assert "ComponentReadingPresence" in signature
    assert "=" not in signature

    # No caller treats the answer as a boolean.
    for surface in (source, WEB_REPORT.read_text(encoding="utf-8")):
        code = _code_only(surface)
        assert "if (componentReadingPresence" not in code
        assert "!componentReadingPresence" not in code


@requires_web
def test_an_empty_panel_does_not_settle_the_evidence_while_its_detector_runs():
    """§3, in the place a reader actually looks for a reading.

    The per-component section says what has run, but a reader who wants the mouth-dynamics
    reading scrolls to the mouth-dynamics panel. That panel used to say, of a detector still
    in flight, that "nothing recorded one, so there is no evidence from this source either
    way" — a terminal statement about the evidence, printed under a final assessment, about a
    detector that had not answered. This was found by rendering the real report and is the
    reason the wording now depends on what the API says is still owed.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")

    terminal = source.split("SIGNAL_ABSENT_TERMINAL =", 1)[1].split(";", 1)[0]
    running = source.split("SIGNAL_ABSENT_STILL_TO_ANSWER =", 1)[1].split(";", 1)[0]

    assert terminal != running

    # The sentence for a detector still working says so, and says the absence is not a
    # finding — neither clause can be read as "this source produced nothing".
    assert "has not produced a reading yet" in running
    assert "not a finding that there is no evidence" in running
    for wording in ("nothing recorded one", "either way"):
        assert wording not in running, wording

    # And the terminal sentence is unchanged for the cases it was always right about.
    assert "nothing recorded one" in terminal

    # Only the two non-terminal component states reach it, read from the API's vocabulary
    # rather than from a list this module keeps.
    still_owed = source.split("COMPONENT_STATES_STILL_TO_ANSWER: readonly string[] = [", 1)[
        1
    ].split("]", 1)[0]
    assert "COMPONENT_QUEUED" in still_owed
    assert "COMPONENT_PROCESSING" in still_owed
    for terminal_state in (
        "COMPONENT_COMPLETED",
        "COMPONENT_ABSTAINED",
        "COMPONENT_FAILED",
        "COMPONENT_NOT_REQUESTED",
    ):
        assert terminal_state not in still_owed, terminal_state

    # The three enrichment panels ask the question; the three decisional ones do not, because
    # nothing is ever owed for them once a verdict exists.
    report = _code_only(WEB_REPORT.read_text(encoding="utf-8"))
    for signal_type in ("active_speaker", "audio_authenticity", "lip_forensics"):
        assert f'signalType="{signal_type}"' in report, signal_type
    for decisional in ("synthetic-video", "provenance", "face-manipulation"):
        assert f'<NoSignal what="{decisional}" analysis={{analysis}} />' in report, decisional

    # The panel renders the sentence and does not choose it: no component state is named here.
    panel = _function_body(report, "function NoSignal(")
    assert "absentSignalDetail(analysis, signalType)" in panel
    for state in COMPONENT_STATES:
        assert f'"{state}"' not in panel, state


@requires_web
def test_an_unobservable_component_is_never_presented_as_a_verified_reading():
    """A component this build cannot see is not vouched for (M1 review fix).

    `completed` on the API's side establishes that the component's task finished. It does not
    establish that the reading it wrote is exposed in `AnalysisSummary`, and for
    `face_forgery` it is not — the API projects no field for EFFORT's signal, so this layer
    has nothing to check.

    That leaves exactly one honest sentence. "Completed — reading not found in this record"
    would report this renderer's blind spot as a defect in the analysis. "Completed with
    reading" is the worse error in the other direction: it is the presentation layer vouching
    for evidence it cannot see, which is indistinguishable to a reader from a reading that was
    actually verified. So the wording retreats to the execution state alone, and this guard
    pins that pair so no later edit can quietly upgrade it again.
    """
    source = WEB_ANALYSIS.read_text(encoding="utf-8")
    table = _reading_labels(source)

    completed = table.split("[COMPONENT_COMPLETED]:", 1)[1].split("},", 1)[0]
    unobservable = re.search(
        r"\[READING_UNOBSERVABLE\]: \"([^\"]+)\"", completed
    ).group(1)

    assert unobservable == "Completed"

    # It claims nothing about a reading, in either direction.
    for claim in ("with reading", "no reading", "not found", "recorded", "evidence"):
        assert claim not in unobservable.lower(), claim

    # And it is a different sentence from both of the other two `completed` outcomes, so the
    # three are legible as three rather than two.
    base = dict(
        re.findall(
            r"\[(COMPONENT_[A-Z_]+)\]: \"([^\"]+)\"",
            source.split("COMPONENT_STATE_LABELS: Record<", 1)[1].split("\n};", 1)[0],
        )
    )
    present = base["COMPONENT_COMPLETED"]
    absent = re.search(r"\[READING_ABSENT\]: \"([^\"]+)\"", completed).group(1)

    assert len({present, absent, unobservable}) == 3
    assert present == "Completed with reading"
    assert absent == "Completed — reading not found in this record"

    # The component that makes this case real is still the one with no field on the payload,
    # so if a later API projects EFFORT's signal this guard's premise fails here first.
    signal_map = _function_body(_code_only(source), "function componentSignal(")
    assert "face_forgery" not in signal_map
