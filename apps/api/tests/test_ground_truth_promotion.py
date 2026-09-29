"""R14-T5: promoting a case into the verified dataset — governance, then Ground Truth, then review.

The workflow writes three records through three unchanged endpoints, in order, from the web
route `/admin/promote-ground-truth`. Its rules live in `apps/web/app/admin/promotion.ts`, free of
`fetch`, and this file executes them in node against stubs — the order and the stop-on-failure
rule are behaviour, and a source-text assertion would pass just as happily with two steps
swapped. What is held:

* the steps run governance → Ground Truth → review, and a failure stops everything after it
  (governance failing means the Ground Truth PUT and the review PUT are never called);
* a failure after a save is reported as what was saved, what failed and what never ran;
* a user's claimed label prefills the Ground Truth label only when it names one exactly —
  `OTHER`, null and anything unknown leave it blank — and never chooses a source class;
* a source class other than `UNKNOWN` needs an attestation and stated evidence;
* the review selector retains `NEEDS_FOLLOW_UP` by default, the PUT keeps the stored assessment
  and note, and a review that changed after the form was opened is not written;
* the governance statement is whole, with "not recorded" as null;
* the media choice is every hash of the analysis (`media_sha256s`), with no default among
  several, and the form is seeded from the chosen file's records;
* no backend contract moved: `SourceClass` is the R12 four, and no endpoint takes Ground Truth,
  governance and review together.
"""

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from shutil import which
from typing import get_args

import pytest

from app.api.admin_analyses import ReviewChange
from app.dataset_governance import MediaGovernanceContract
from app.ground_truth import GroundTruthContract, SourceClass
from app.main import app

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
WEB_ADMIN = WEB_ROOT / "app" / "admin"
PROMOTION = WEB_ADMIN / "promotion.ts"
REVIEWS = WEB_ADMIN / "reviews.ts"
GROUND_TRUTH_WEB = WEB_ADMIN / "ground-truth.ts"
ROUTE = WEB_ADMIN / "promote-ground-truth" / "route.ts"
PAGE = WEB_ADMIN / "analyses" / "[id]" / "page.tsx"

NODE = "node"
TYPESCRIPT = WEB_ROOT / "node_modules" / "typescript"

requires_web = pytest.mark.skipif(
    not WEB_ROOT.exists(), reason="the web application is not present"
)
requires_node = pytest.mark.skipif(
    not (PROMOTION.exists() and TYPESCRIPT.exists() and which(NODE) is not None),
    reason="node and the web application's TypeScript are needed to run the promotion rules",
)

SHA = "a" * 64

# Every scenario runs in one node process. `promotion.ts` is transpiled and loaded with
# `./reviews` resolved to the real, transpiled `reviews.ts` (its own imports stubbed: only its
# constants are reached), so the review statuses are the ones the page uses.
DRIVER = r"""
const fs = require("fs");
const ts = require(process.argv[1]);
const stub = new Proxy({}, { get: () => () => undefined });

function load(path, modules) {
  const compiled = ts.transpileModule(fs.readFileSync(path, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module_ = { exports: {} };
  new Function("exports", "module", "require", compiled)(
    module_.exports, module_, (name) => modules[name] ?? stub
  );
  return module_.exports;
}

const reviews = load(process.argv[3], {});
const p = load(process.argv[2], { "./reviews": reviews });
const SHA = process.argv[4];

const review = (status) => ({
  analysis_id: "x", status, analyst_assessment: "DISAGREES_WITH_AUTOMATED_ASSESSMENT",
  note: "keep <b>me</b>", reviewer_id: null, reviewer_email_snapshot: null,
  created_at: null, updated_at: null,
});

async function run(failAt, from) {
  const calls = [];
  const step = (name) => async () => {
    calls.push(name);
    return name === failAt ? { ok: false, error: `${name} refused` } : { ok: true };
  };
  const outcome = await p.runPromotion(
    { governance: step("governance"), ground_truth: step("ground_truth"), review: step("review") },
    from,
  );
  return { calls, outcome };
}

async function review_(target, expected, current) {
  const calls = [];
  const result = await p.setReviewStatus(target, expected, {
    readReview: async () => { calls.push("read"); return { ok: true, review: review(current) }; },
    putReview: async (body) => { calls.push(["put", body]); return { ok: true }; },
  });
  return { result, calls };
}

const form = (values) => (name) => (name in values ? values[name] : null);

(async () => {
  const out = {};

  out.order = {
    all_ok: await run(null),
    governance_fails: await run("governance"),
    ground_truth_fails: await run("ground_truth"),
    review_fails: await run("review"),
    review_retry: await run(null, "review"),
  };

  out.prefill = Object.fromEntries(
    ["GENUINE", "AI_GENERATED", "FACE_SWAP", "OTHER", "BOGUS", "toString", "__proto__"]
      .map((claim) => [claim, p.prefilledGroundTruthLabel(claim)])
  );
  out.prefill_null = p.prefilledGroundTruthLabel(null);

  out.provenance = {
    none: p.provenanceProblem("", false, ""),
    unknown: p.provenanceProblem("UNKNOWN", false, ""),
    owner_unattested: p.provenanceProblem("OWNER_KNOWN", false, "the owner confirmed by phone"),
    external_no_notes: p.provenanceProblem("EXTERNAL_VERIFIED", true, "   "),
    controlled_ok: p.provenanceProblem("CONTROLLED_TEST", true, "generated in-house, run 12"),
    external_ok: p.provenanceProblem("EXTERNAL_VERIFIED", true, "AFP fact-check 2026-09-01"),
  };

  out.review_default = Object.fromEntries(
    ["NEEDS_FOLLOW_UP", "REVIEWED", "UNREVIEWED", "SOMETHING"].map((s) => [s, p.defaultReviewTarget(s)])
  );
  out.review_targets = p.REVIEW_TARGETS;
  out.review = {
    retain_follow_up: await review_("NEEDS_FOLLOW_UP", "NEEDS_FOLLOW_UP", "NEEDS_FOLLOW_UP"),
    close_follow_up_chosen: await review_("REVIEWED", "NEEDS_FOLLOW_UP", "NEEDS_FOLLOW_UP"),
    changed_since_opened: await review_("REVIEWED", "UNREVIEWED", "NEEDS_FOLLOW_UP"),
    no_target: await review_("", "UNREVIEWED", "UNREVIEWED"),
    invented_status: await review_("RESOLVED", "UNREVIEWED", "UNREVIEWED"),
  };

  out.governance = {
    missing_lineage: p.governanceStatement(form({ dataset_split: "TEST" })),
    missing_split: p.governanceStatement(form({ source_lineage_id: "lin-1" })),
    minimal: p.governanceStatement(form({
      source_lineage_id: "lin-1", dataset_split: "TEST", license: "", redistributable: "",
      private: "", transformations: "",
    })),
    full: p.governanceStatement(form({
      source_lineage_id: "lin-1", dataset_split: "HOLDOUT", recording_identity: "rec-7",
      derived_from_sha256: SHA, generation_pipeline: "veo 3", license: "CC-BY-4.0",
      permission_status: "granted", redistributable: "true", private: "false",
      stratum_primary: "news", source: "newsroom upload", acquisition_type: "upload",
      benchmark_family: "veo", transformations: "reencode\r\n\r\nresize\n",
    })),
    bad_boolean: p.governanceStatement(form({
      source_lineage_id: "lin-1", dataset_split: "TEST", private: "yes",
    })),
  };

  const partial = { saved: ["governance", "ground_truth"], failed: "review", error: "Review: nope", notAttempted: [] };
  const params = p.outcomeParams(partial);
  out.outcome = {
    params,
    parsed: p.parseOutcome(form(params)),
    partial_is_review_only: p.isReviewOnlyFailure(p.parseOutcome(form(params))),
    refusal: p.parseOutcome(form(p.refusalParams("Choose what the media is."))),
    forged: p.parseOutcome(form({ promo_saved: "governance,verdict", promo_failed: "risk_level" })),
    none: p.parseOutcome(form({})),
  };

  const B = "b".repeat(64);
  out.candidates = {
    one: p.promotionCandidates([SHA]),
    none: p.promotionCandidates([]),
    many: p.promotionCandidates([B, SHA, B]),
  };
  out.media = {
    single_unrequested: p.selectedPromotionMedia([SHA], null),
    single_foreign_request: p.selectedPromotionMedia([SHA], B),
    many_unrequested: p.selectedPromotionMedia([SHA, B], null),
    many_requested: p.selectedPromotionMedia([SHA, B], B),
    many_foreign_request: p.selectedPromotionMedia([SHA, B], "c".repeat(64)),
    none: p.selectedPromotionMedia([], null),
  };

  console.log(JSON.stringify(out));
})().catch((error) => { console.error(error); process.exit(1); });
"""


@lru_cache(maxsize=1)
def _run() -> dict:
    result = subprocess.run(
        [NODE, "-e", DRIVER, "--", str(TYPESCRIPT), str(PROMOTION), str(REVIEWS), SHA],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"node exited {result.returncode}:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


# --------------------------------------------------------------------------------------------
# Order and failure safety
# --------------------------------------------------------------------------------------------


@requires_node
def test_the_three_steps_run_governance_then_ground_truth_then_review():
    run = _run()["order"]["all_ok"]

    assert run["calls"] == ["governance", "ground_truth", "review"]
    assert run["outcome"] == {
        "saved": ["governance", "ground_truth", "review"],
        "failed": None,
        "error": None,
        "notAttempted": [],
    }


@requires_node
def test_a_failed_governance_put_calls_neither_ground_truth_nor_review():
    run = _run()["order"]["governance_fails"]

    assert run["calls"] == ["governance"]
    assert run["outcome"]["saved"] == []
    assert run["outcome"]["failed"] == "governance"
    assert run["outcome"]["notAttempted"] == ["ground_truth", "review"]


@requires_node
def test_a_failed_ground_truth_put_never_touches_the_review():
    run = _run()["order"]["ground_truth_fails"]

    assert run["calls"] == ["governance", "ground_truth"]
    assert run["outcome"]["saved"] == ["governance"]
    assert run["outcome"]["failed"] == "ground_truth"
    assert run["outcome"]["notAttempted"] == ["review"]


@requires_node
def test_a_failed_review_after_both_saves_is_a_partial_success_offering_a_review_retry():
    run = _run()["order"]["review_fails"]
    outcome = _run()["outcome"]

    assert run["calls"] == ["governance", "ground_truth", "review"]
    assert run["outcome"]["saved"] == ["governance", "ground_truth"]
    assert run["outcome"]["failed"] == "review"

    # Through the query string and back, as the redirect carries it.
    assert outcome["parsed"] == {
        "saved": ["governance", "ground_truth"],
        "failed": "review",
        "error": "Review: nope",
    }
    assert outcome["partial_is_review_only"] is True


@requires_node
def test_the_review_retry_runs_the_review_step_alone():
    assert _run()["order"]["review_retry"]["calls"] == ["review"]


@requires_node
def test_the_outcome_names_only_real_steps_and_a_refusal_saved_nothing():
    outcome = _run()["outcome"]

    assert outcome["forged"] == {"saved": ["governance"], "failed": None, "error": None}
    assert outcome["refusal"] == {
        "saved": [],
        "failed": None,
        "error": "Choose what the media is.",
    }
    assert outcome["none"] is None


# --------------------------------------------------------------------------------------------
# Prefill and provenance
# --------------------------------------------------------------------------------------------


@requires_node
def test_a_claim_prefills_the_label_only_when_it_names_one_exactly():
    prefill = _run()["prefill"]

    assert prefill["GENUINE"] == "GENUINE"
    assert prefill["AI_GENERATED"] == "AI_GENERATED"
    assert prefill["FACE_SWAP"] == "FACE_SWAP"
    # A user's "other" is not OTHER_MANIPULATION; unknown words and prototype names map to nothing.
    for claim in ("OTHER", "BOGUS", "toString", "__proto__"):
        assert prefill[claim] == "", claim
    assert _run()["prefill_null"] == ""


@requires_node
def test_every_prefilled_label_is_a_ground_truth_label():
    from app.ground_truth import GroundTruthLabel

    labels = set(get_args(GroundTruthLabel))
    assert {v for v in _run()["prefill"].values() if v} <= labels


@requires_node
def test_a_source_class_must_be_chosen_and_only_unknown_needs_no_evidence():
    provenance = _run()["provenance"]

    assert provenance["none"] is not None
    assert provenance["unknown"] is None
    assert provenance["owner_unattested"] is not None
    assert provenance["external_no_notes"] is not None
    assert provenance["controlled_ok"] is None
    assert provenance["external_ok"] is None


@requires_web
def test_the_promotion_form_preselects_neither_the_source_class_nor_the_media():
    form = PAGE.read_text(encoding="utf-8").split("function PromotionForm(", 1)[1].split(
        "\nfunction ", 1
    )[0]

    assert '<select name="source_class" required defaultValue="" className={inputClass}>' in form
    assert '<input type="radio" name="sha256" value={sha256} required />' in form
    assert "defaultChecked" not in form and "checked=" not in form
    # The label's only prefill is the mapped claim.
    assert (
        'defaultValue={feedback === null ? "" : prefilledGroundTruthLabel(feedback.claimed_label)}'
        in form
    )


@requires_web
def test_feedback_is_chosen_per_record_not_first_in_the_list():
    page = PAGE.read_text(encoding="utf-8")

    assert 'promotionPath(id, { promote: "feedback", feedback: entry.user_id })' in page
    assert "entries.find((entry) => entry.user_id === feedbackUserId)" in page
    assert "entries[0]" not in page


# --------------------------------------------------------------------------------------------
# Review state
# --------------------------------------------------------------------------------------------


@requires_node
def test_the_review_selector_retains_needs_follow_up_by_default():
    default = _run()["review_default"]

    assert default["NEEDS_FOLLOW_UP"] == "NEEDS_FOLLOW_UP"
    assert default["REVIEWED"] == "REVIEWED"
    assert default["UNREVIEWED"] == ""
    assert default["SOMETHING"] == ""
    assert _run()["review_targets"] == ["REVIEWED", "NEEDS_FOLLOW_UP"]


@requires_node
def test_retaining_follow_up_keeps_it_and_keeps_the_assessment_and_note():
    run = _run()["review"]["retain_follow_up"]

    assert run["result"] == {"ok": True}
    assert run["calls"] == [
        "read",
        [
            "put",
            {
                "status": "NEEDS_FOLLOW_UP",
                "analyst_assessment": "DISAGREES_WITH_AUTOMATED_ASSESSMENT",
                "note": "keep <b>me</b>",
            },
        ],
    ]


@requires_node
def test_follow_up_is_closed_only_when_the_analyst_chose_it():
    chosen = _run()["review"]["close_follow_up_chosen"]
    assert chosen["calls"][1][1]["status"] == "REVIEWED"

    # Opened while unreviewed, flagged for follow-up by someone else since: nothing is written.
    stale = _run()["review"]["changed_since_opened"]
    assert stale["result"]["ok"] is False
    assert "NEEDS_FOLLOW_UP" in stale["result"]["error"]
    assert stale["calls"] == ["read"]


@requires_node
def test_no_status_and_an_invented_status_write_nothing():
    for case in ("no_target", "invented_status"):
        run = _run()["review"][case]
        assert run["result"]["ok"] is False, case
        assert run["calls"] == [], case


# --------------------------------------------------------------------------------------------
# Governance statement
# --------------------------------------------------------------------------------------------


@requires_node
def test_lineage_and_split_are_required():
    governance = _run()["governance"]

    assert governance["missing_lineage"]["ok"] is False
    assert governance["missing_split"]["ok"] is False
    assert governance["bad_boolean"]["ok"] is False


@requires_node
def test_the_statement_is_whole_and_unrecorded_is_null():
    minimal = _run()["governance"]["minimal"]["statement"]
    full = _run()["governance"]["full"]["statement"]

    # Every field the API contract declares, no more: a PUT is the whole record.
    assert set(minimal) == set(MediaGovernanceContract.model_fields)
    assert set(full) == set(MediaGovernanceContract.model_fields)

    assert {k: v for k, v in minimal.items() if v is not None} == {
        "source_lineage_id": "lin-1",
        "dataset_split": "TEST",
    }
    assert full["redistributable"] is True and full["private"] is False
    assert full["transformations"] == ["reencode", "resize"]
    # And the API accepts it as it is.
    MediaGovernanceContract(**full)


# --------------------------------------------------------------------------------------------
# Media and route
# --------------------------------------------------------------------------------------------


@requires_node
def test_the_candidates_are_every_media_hash_of_the_analysis():
    assert _run()["candidates"] == {
        "one": [SHA],
        "none": [],
        # Every media hash, not only the original: deduplicated and sorted.
        "many": [SHA, "b" * 64],
    }


@requires_node
def test_with_several_media_the_form_opens_only_on_a_chosen_one():
    """Nothing is defaulted among several files: the form is seeded from one file's records, and
    saving another file's seeded governance would overwrite it. One file is the only one."""
    media = _run()["media"]

    assert media["single_unrequested"] == SHA
    assert media["single_foreign_request"] == SHA
    assert media["many_unrequested"] is None
    assert media["many_requested"] == "b" * 64
    assert media["many_foreign_request"] is None
    assert media["none"] is None


@requires_web
def test_the_page_builds_the_choice_from_media_sha256s_and_seeds_from_the_chosen_file():
    page = PAGE.read_text(encoding="utf-8")

    assert "candidates={promotionCandidates(analysisResult.analysis.media_sha256s)}" in page
    assert "promotionCandidates(result.analysis.media_sha256s)" in page
    # Governance and the form's Ground Truth are read for the chosen media, not the original.
    assert "sha256 === null ? null : fetchDatasetGovernance(sha256)" in page
    assert ": fetchGroundTruth(sha256)," in page
    # With several files, each is a link that reopens the panel on it; the form's radio holds
    # only the chosen file and is still unchecked.
    assert "href={promotionPath(analysisId, { ...panel, media: sha256 })}" in page
    assert "candidates={[media]}" in page


@requires_web
def test_the_route_calls_the_three_existing_endpoints_through_the_ordered_runner():
    route = ROUTE.read_text(encoding="utf-8")

    assert "runPromotion(steps" in route
    assert "put(datasetGovernanceUrl(sha256)" in route
    assert "put(\n          groundTruthUrl(sha256)" in route
    assert "put(reviewUrl(analysisId)" in route
    # No API path spelled in code here (the docstring names them): only the three builders the
    # rest of the surface uses.
    code = route.split("*/", 1)[1]
    assert "/api/v1" not in code
    # The chosen hash must be one of this analysis's media.
    assert "promotionCandidates(analysis.analysis.media_sha256s).includes(sha256)" in route


# --------------------------------------------------------------------------------------------
# Backend contracts unchanged
# --------------------------------------------------------------------------------------------


def test_source_class_is_still_the_r12_four():
    assert get_args(SourceClass) == (
        "OWNER_KNOWN",
        "CONTROLLED_TEST",
        "EXTERNAL_VERIFIED",
        "UNKNOWN",
    )


@requires_web
def test_the_web_offers_exactly_the_api_source_classes():
    source = GROUND_TRUTH_WEB.read_text(encoding="utf-8")
    block = source.split("GROUND_TRUTH_SOURCE_CLASS_LABELS: Record<string, string> = {", 1)[1]
    block = block.split("};", 1)[0]
    keys = [line.strip().split(":", 1)[0] for line in block.strip().splitlines()]

    assert tuple(keys) == get_args(SourceClass)


def test_the_three_request_contracts_do_not_carry_each_others_fields():
    assert set(ReviewChange.model_fields) == {"status", "analyst_assessment", "note"}
    assert set(GroundTruthContract.model_fields) == {"source_class", "label", "notes"}
    governance = set(MediaGovernanceContract.model_fields)
    assert not governance & {"label", "source_class", "status", "analysis_id", "note", "notes"}


def test_no_endpoint_takes_more_than_one_of_the_three_records():
    # Read from the OpenAPI document: the routers are mounted, so `app.routes` does not list them.
    paths = app.openapi()["paths"]
    puts = {path for path, methods in paths.items() if "put" in methods}

    assert "/api/v1/admin/dataset-governance/{sha256}" in puts
    assert "/api/v1/admin/ground-truth/{sha256}" in puts
    assert "/api/v1/admin/analyses/{analysis_id}/review" in puts
    assert not [path for path in paths if "promot" in path]

    # Each of the three takes one record's contract as its body, and nothing else.
    schemas = {
        "/api/v1/admin/dataset-governance/{sha256}": "MediaGovernanceContract",
        "/api/v1/admin/ground-truth/{sha256}": "GroundTruthContract",
        "/api/v1/admin/analyses/{analysis_id}/review": "ReviewChange",
    }
    for path, schema in schemas.items():
        body = paths[path]["put"]["requestBody"]["content"]["application/json"]["schema"]
        assert body == {"$ref": f"#/components/schemas/{schema}"}, path
