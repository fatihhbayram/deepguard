/**
 * Promoting one analysis's media into the verified dataset (R14-T5): the rules and the order,
 * with no network of its own.
 *
 * The workflow writes three records through three unchanged endpoints — dataset governance, Ground
 * Truth, the review — and no endpoint was added that takes more than one of them. What this module
 * holds is everything about that sequence that can be decided without a request: what a user's
 * claim may prefill, what a source class needs before it may be stated, how a form becomes a
 * governance statement, what the review PUT carries, and the order the three writes run in. The
 * route (`promote-ground-truth/route.ts`) supplies the requests. Keeping the decisions here, free
 * of `fetch`, is what lets `tests/test_ground_truth_promotion.py` run them against stubs.
 *
 * **User feedback is context, never provenance.** A user's claimed label may prefill the label
 * select when it names a Ground Truth label exactly, and nothing else. It never chooses a source
 * class: the claim is the analysis owner's statement, and a person reading it has not thereby
 * established how the media was made. `OWNER_KNOWN`, `CONTROLLED_TEST` and `EXTERNAL_VERIFIED`
 * are accepted only with an explicit attestation and the evidence stated in the notes.
 *
 * **The order is governance, then Ground Truth, then the review, and each step runs only if the
 * one before it succeeded.** Governance first so a label never reaches the evaluation pipeline
 * without the lineage, split and licence it is exported under; Ground Truth before the review so
 * a case is never closed on a label that was not stored. A step that fails stops the sequence,
 * and the outcome names what was saved, what failed and what was never attempted.
 */

import { REVIEW_STATUS_NEEDS_FOLLOW_UP, REVIEW_STATUS_REVIEWED } from "./reviews";
import type { AnalysisReview } from "./reviews";

/* ------------------------------------------------------------------ *
 * Ground Truth: prefill and provenance
 * ------------------------------------------------------------------ */

/**
 * A user's claimed label → the Ground Truth label it names exactly.
 *
 * `OTHER` is deliberately absent. A user's "other" may be audio manipulation, a manipulation
 * nobody has a word for, or not a manipulation at all; `OTHER_MANIPULATION` would be a guess.
 */
const GROUND_TRUTH_LABEL_BY_CLAIM: Record<string, string> = {
  GENUINE: "GENUINE",
  AI_GENERATED: "AI_GENERATED",
  FACE_SWAP: "FACE_SWAP",
};

/** The label the select opens on: the claim's exact counterpart, or nothing. */
export function prefilledGroundTruthLabel(claimedLabel: string | null): string {
  if (claimedLabel === null) {
    return "";
  }

  return Object.prototype.hasOwnProperty.call(GROUND_TRUTH_LABEL_BY_CLAIM, claimedLabel)
    ? GROUND_TRUTH_LABEL_BY_CLAIM[claimedLabel]
    : "";
}

/**
 * The source class that states no provenance. The only one that needs no attestation, and still
 * never preselected: the analyst chooses every source class, this one included.
 */
export const SOURCE_CLASS_UNKNOWN = "UNKNOWN";

/** Why the stated provenance may not be forwarded, or null when it may. */
export function provenanceProblem(
  sourceClass: string,
  attested: boolean,
  notes: string,
): string | null {
  if (sourceClass === "") {
    return "Choose how the label is known.";
  }

  if (sourceClass === SOURCE_CLASS_UNKNOWN) {
    return null;
  }

  if (!attested || notes.trim() === "") {
    return (
      "A source class other than Unknown needs evidence independent of the user's feedback: " +
      "confirm the attestation and state the evidence in the notes, or choose Unknown."
    );
  }

  return null;
}

/* ------------------------------------------------------------------ *
 * Dataset governance: form → statement
 * ------------------------------------------------------------------ */

// Optional one-line fields. An empty input is "not recorded" and is sent as null; anything else
// is forwarded exactly as typed, and the API is what refuses a malformed identifier.
export const GOVERNANCE_OPTIONAL_TEXT_FIELDS = [
  "recording_identity",
  "derived_from_sha256",
  "generation_pipeline",
  "license",
  "permission_status",
  "stratum_primary",
  "source",
  "acquisition_type",
  "benchmark_family",
] as const;

export const GOVERNANCE_BOOLEAN_FIELDS = ["redistributable", "private"] as const;

// The select value for an unrecorded boolean. Unknown is NULL, never a default of false.
export const BOOLEAN_NOT_RECORDED = "";

export type GovernanceStatement = {
  source_lineage_id: string;
  dataset_split: string;
  recording_identity: string | null;
  transformations: string[] | null;
  derived_from_sha256: string | null;
  generation_pipeline: string | null;
  license: string | null;
  permission_status: string | null;
  redistributable: boolean | null;
  private: boolean | null;
  stratum_primary: string | null;
  source: string | null;
  acquisition_type: string | null;
  benchmark_family: string | null;
};

/**
 * The whole governance statement out of the form, or why there is none.
 *
 * Whole, because the PUT is: a field left out would be cleared. Transformations are one per
 * line, in order; blank lines are dropped and an all-blank box is null.
 */
export function governanceStatement(
  field: (name: string) => string | null,
): { ok: true; statement: GovernanceStatement } | { ok: false; problem: string } {
  const lineage = field("source_lineage_id") ?? "";
  const split = field("dataset_split") ?? "";

  if (lineage === "" || split === "") {
    return {
      ok: false,
      problem: "State the source lineage and the dataset split before promoting.",
    };
  }

  const optional = (name: string): string | null => {
    const value = field(name);
    return value === null || value === "" ? null : value;
  };

  const booleans: Record<string, boolean | null> = {};
  for (const name of GOVERNANCE_BOOLEAN_FIELDS) {
    const value = field(name) ?? BOOLEAN_NOT_RECORDED;
    if (value === BOOLEAN_NOT_RECORDED) {
      booleans[name] = null;
    } else if (value === "true" || value === "false") {
      booleans[name] = value === "true";
    } else {
      return { ok: false, problem: `${name} must be yes, no or not recorded.` };
    }
  }

  const steps = (field("transformations") ?? "")
    .split(/\r?\n/)
    .filter((step) => step !== "");

  return {
    ok: true,
    statement: {
      source_lineage_id: lineage,
      dataset_split: split,
      recording_identity: optional("recording_identity"),
      transformations: steps.length === 0 ? null : steps,
      derived_from_sha256: optional("derived_from_sha256"),
      generation_pipeline: optional("generation_pipeline"),
      license: optional("license"),
      permission_status: optional("permission_status"),
      redistributable: booleans.redistributable,
      private: booleans.private,
      stratum_primary: optional("stratum_primary"),
      source: optional("source"),
      acquisition_type: optional("acquisition_type"),
      benchmark_family: optional("benchmark_family"),
    },
  };
}

/* ------------------------------------------------------------------ *
 * Media and review
 * ------------------------------------------------------------------ */

/**
 * The hashes this analysis may be promoted under: every media hash the analysis read reports.
 *
 * Offered as a choice with nothing preselected — even when there is one — and the route refuses
 * any hash not in this list, so the analyst states which bytes the record is about every time.
 * A copy, deduplicated and sorted, so the list is the same whatever order the API sent.
 */
export function promotionCandidates(mediaSha256s: string[]): string[] {
  return Array.from(new Set(mediaSha256s)).sort();
}

/**
 * The media the promotion form is opened on, whose stored governance and Ground Truth it is
 * seeded from — or null while the analyst still has to choose one.
 *
 * A requested hash counts only if it is one of the candidates. With one candidate that one is
 * used; with several and no valid request there is no default, because seeding the form from the
 * wrong file's governance and saving it would overwrite that file's whole record.
 */
export function selectedPromotionMedia(candidates: string[], requested: string | null): string | null {
  if (requested !== null && candidates.includes(requested)) {
    return requested;
  }

  return candidates.length === 1 ? candidates[0] : null;
}

/** The two workflow states a promotion may leave the review in. */
export const REVIEW_TARGETS = [REVIEW_STATUS_REVIEWED, REVIEW_STATUS_NEEDS_FOLLOW_UP];

/**
 * The review status the selector opens on: the current one when it is a real state, so leaving
 * the control alone retains it — `NEEDS_FOLLOW_UP` is never turned into `REVIEWED` by default.
 * An unreviewed case opens on nothing and the analyst has to choose.
 */
export function defaultReviewTarget(currentStatus: string): string {
  return REVIEW_TARGETS.includes(currentStatus) ? currentStatus : "";
}

/**
 * The review PUT body: the chosen status, with the stored assessment and note carried unchanged.
 *
 * The PUT is the whole review, and an absent assessment means "none recorded"; a body of the
 * status alone would erase what the reviewer wrote.
 */
export function reviewChange(target: string, current: AnalysisReview) {
  return {
    status: target,
    analyst_assessment: current.analyst_assessment,
    note: current.note,
  };
}

export type StepResult = { ok: true } | { ok: false; error: string };

/**
 * Set the review to the chosen status, only if it still is what the analyst saw.
 *
 * `expectedStatus` is the status the form was rendered with. If another administrator changed
 * the review since — to `NEEDS_FOLLOW_UP`, say — nothing is written and the analyst is told,
 * rather than their choice silently replacing a state they never saw.
 */
export async function setReviewStatus(
  target: string,
  expectedStatus: string,
  deps: {
    readReview: () => Promise<{ ok: true; review: AnalysisReview } | { ok: false; error: string }>;
    putReview: (body: ReturnType<typeof reviewChange>) => Promise<StepResult>;
  },
): Promise<StepResult> {
  if (!REVIEW_TARGETS.includes(target)) {
    return { ok: false, error: "Choose the review status to leave this case in." };
  }

  const current = await deps.readReview();
  if (!current.ok) {
    return current;
  }

  if (current.review.status !== expectedStatus) {
    return {
      ok: false,
      error: `The review changed to ${current.review.status} after this form was opened; it was not updated. Check it and choose again.`,
    };
  }

  return deps.putReview(reviewChange(target, current.review));
}

/* ------------------------------------------------------------------ *
 * The ordered sequence
 * ------------------------------------------------------------------ */

export const PROMOTION_STEPS = ["governance", "ground_truth", "review"] as const;
export type PromotionStep = (typeof PROMOTION_STEPS)[number];

export type PromotionOutcome = {
  saved: PromotionStep[];
  failed: PromotionStep | null;
  error: string | null;
  notAttempted: PromotionStep[];
};

/**
 * Run the steps in order from `from`, stopping at the first that fails.
 *
 * `from` is `"governance"` for a promotion and `"review"` for the retry of a review that failed
 * after the other two were saved. A step before `from` is neither run nor reported.
 */
export async function runPromotion(
  steps: Record<PromotionStep, () => Promise<StepResult>>,
  from: PromotionStep = "governance",
): Promise<PromotionOutcome> {
  const order = PROMOTION_STEPS.slice(PROMOTION_STEPS.indexOf(from));
  const saved: PromotionStep[] = [];

  for (const [index, step] of order.entries()) {
    const result = await steps[step]();
    if (!result.ok) {
      return { saved, failed: step, error: result.error, notAttempted: order.slice(index + 1) };
    }
    saved.push(step);
  }

  return { saved, failed: null, error: null, notAttempted: [] };
}

/* ------------------------------------------------------------------ *
 * The outcome in the query string
 * ------------------------------------------------------------------ */

// Enough of the API's message to be useful, and bounded. Rendered as text by React.
export const MAX_PROMOTION_ERROR_LENGTH = 200;

/** The outcome as query parameters for the redirect back to the analysis. */
export function outcomeParams(outcome: PromotionOutcome): Record<string, string> {
  const params: Record<string, string> = { promo_saved: outcome.saved.join(",") };

  if (outcome.failed !== null) {
    params.promo_failed = outcome.failed;
  }
  if (outcome.error !== null) {
    params.promo_error = outcome.error.slice(0, MAX_PROMOTION_ERROR_LENGTH);
  }

  return params;
}

/** A refusal before any step ran: nothing saved, nothing failed, one sentence. */
export function refusalParams(problem: string): Record<string, string> {
  return { promo_saved: "", promo_error: problem.slice(0, MAX_PROMOTION_ERROR_LENGTH) };
}

export type ParsedOutcome = {
  saved: PromotionStep[];
  failed: PromotionStep | null;
  error: string | null;
};

function isStep(value: string): value is PromotionStep {
  return (PROMOTION_STEPS as readonly string[]).includes(value);
}

/**
 * The outcome read back out of the query string, or null when there is none.
 *
 * A step name that is not one of the three is dropped rather than rendered: the page states what
 * was saved, and must not state a save of something it cannot name.
 */
export function parseOutcome(
  query: (name: string) => string | null,
): ParsedOutcome | null {
  const saved = query("promo_saved");
  if (saved === null) {
    return null;
  }

  const failed = query("promo_failed");

  return {
    saved: saved.split(",").filter(isStep),
    failed: failed !== null && isStep(failed) ? failed : null,
    error: query("promo_error"),
  };
}

/** Whether the outcome is the partial success of a review that failed after the rest was saved. */
export function isReviewOnlyFailure(outcome: ParsedOutcome): boolean {
  return outcome.failed === "review";
}
