/**
 * The administrative surface's half of the review API: reading the human review of an analysis.
 *
 * The same split `jobs.ts`, `users.ts`, `analytics.ts` and `audit.ts` make, made again for the
 * same reason — one record, one reader, one module. What is shared with them is the plumbing
 * (`apiUrl`, the session headers, the request id), and that is imported rather than copied.
 *
 * **Nothing here is a forensic value and nothing here can become one.** The review is an
 * opinion attached to an analysis; the risk classification, the ruleset, the calibration and
 * the signal rows come from `../analysis` and are read by the page separately. The two are
 * fetched by two functions from two modules against two endpoints, which is what keeps them
 * from being merged into one object that a renderer could present as a single answer.
 *
 * **This module writes nothing.** The mutation is a plain HTML form posting to
 * `/admin/update-review`, the same shape the account controls use and for the same reason: the
 * API serves no CORS headers, so the browser cannot call it directly, and the alternative is a
 * client component whose only job is to call `fetch` with a verb.
 *
 * **The note is a string and is only ever rendered as text.** There is no parsing here, no
 * Markdown, no HTML, and the page above must not introduce any — a `dangerouslySetInnerHTML`
 * anywhere near this value would turn a note one administrator typed into script running in
 * another administrator's browser. The API stores it verbatim precisely because nothing
 * renders it, and that invariant lives in the reader, not in the column.
 *
 * **The address on a review is history, not a directory lookup.** `reviewer_email_snapshot` was
 * frozen when the review was last written and the API never resolves it against `users`. So it
 * can legitimately name an address no account currently has — the same property `audit.ts`
 * documents, and the same feature.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

/**
 * The API path for one analysis's review, in one place.
 *
 * The page is at `/admin/analyses/<id>`; this is the internal address behind it, and the two
 * are unrelated strings that happen to share a word. Exported because the route handler that
 * forwards the mutation needs the same address and must not spell it a second time.
 */
export function reviewUrl(analysisId: string): string {
  return `/api/v1/admin/analyses/${encodeURIComponent(analysisId)}/review`;
}

// How long the API is given to answer. One indexed read of at most one row, held to the same
// short bound as the job listing and the audit log.
export const REVIEW_TIMEOUT_MS = 5000;

/**
 * The two statuses an administrator may store, as the API spells them.
 *
 * They must match `REVIEW_STATUSES` in `apps/api/app/db/models.py`, which is where the value is
 * validated and where the database's check constraint enforces it. This is the same restating
 * across languages that `SESSION_COOKIE_NAME` is, and for the same reason — there is no way to
 * derive one from the other.
 *
 * **Neither is a verdict, and no third value belongs here.** `REVIEWED` means somebody looked;
 * `NEEDS_FOLLOW_UP` means somebody looked and wants it looked at again. Whether the media is
 * genuine is `risk_level`, decided by the risk engine under a named ruleset — adding a
 * `"FAKE"` to this array would not make the API accept one, but it would put the word on a
 * control, which is most of the way to somebody making it work.
 */
export const REVIEW_STATUS_REVIEWED = "REVIEWED";
export const REVIEW_STATUS_NEEDS_FOLLOW_UP = "NEEDS_FOLLOW_UP";

/**
 * The state an analysis nobody has reviewed is in.
 *
 * Not a value the API stores — it is the absence of a review row — but it is a value the API
 * *sends*, and this screen prints the word, so the spelling is named rather than typed into the
 * markup. It is deliberately not in the two constants above: it is not a status a control may
 * offer, because there is no route that un-reviews an analysis.
 */
export const REVIEW_STATUS_UNREVIEWED = "UNREVIEWED";

// The longest note the API will accept, matching `MAX_REVIEW_NOTE_LENGTH` in the models
// module. Restated here so the textarea can carry a `maxLength` that agrees with the validator
// — a control that let somebody type 1,400 characters and then lost the last 400 to a 422 is a
// control that wastes the one thing a reviewer actually wrote.
export const MAX_REVIEW_NOTE_LENGTH = 1000;

/** Every status the API may report, including the one that is the absence of a row. */
export const REVIEW_STATUS_LABELS: Record<string, string> = {
  [REVIEW_STATUS_UNREVIEWED]: "Not reviewed",
  [REVIEW_STATUS_REVIEWED]: "Reviewed",
  [REVIEW_STATUS_NEEDS_FOLLOW_UP]: "Needs follow-up",
};

/**
 * One review as `/api/v1/admin/analyses/{id}/review` reports it.
 *
 * Mirrors `AnalysisReviewState` in `app/api/admin_analyses.py`. The four nullable fields are
 * null together: they belong to the review row, and an unreviewed analysis has none. A null
 * `reviewer_id` therefore means "nobody has reviewed this", never "somebody did and we lost
 * who".
 *
 * `status` is a plain string and not a union of the three constants above. The API is the
 * authority on what it may send, and a closed type here would make a value this file was not
 * written to expect fail to parse — hiding the one review a reader most needs to see behind an
 * error that says the review could not be read.
 *
 * The timestamps stay strings. They are the API's own values, and the workspace shows a
 * timestamp as stored rather than reformatted into a local rendering the record does not hold —
 * the convention the case log, the account table and the audit log all follow.
 */
export type AnalysisReview = {
  analysis_id: string;
  status: string;
  note: string;
  reviewer_id: string | null;
  reviewer_email_snapshot: string | null;
  created_at: string | null;
  updated_at: string | null;
};

/**
 * What a read of one review produced.
 *
 * The same shape `AuditResult` has, with one field added: `missing` separates "this analysis
 * does not exist" from every other failure, because it is the only one the page answers with a
 * 404 rather than a sentence beside the content.
 *
 * There is no `unreviewed` variant, deliberately. An analysis nobody has looked at is a
 * successful read of a review whose status is `UNREVIEWED` — the API says so with a 200 — and
 * making it a separate case here would put a branch in every caller for the state that the
 * entire history of the deployment is in.
 */
export type ReviewResult =
  | { ok: true; review: AnalysisReview }
  | { ok: false; missing: boolean; unauthenticated: boolean; error: string };

/** A field that is either a string or absent, or `undefined` for anything that is neither. */
function nullableString(value: unknown): string | null | undefined {
  if (value === null || typeof value === "string") {
    return value ?? null;
  }

  return undefined;
}

/** One review out of the payload, or null for anything that is not one. */
export function parseReview(payload: unknown): AnalysisReview | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    analysis_id,
    status,
    note,
    reviewer_id,
    reviewer_email_snapshot,
    created_at,
    updated_at,
  } = payload as Record<string, unknown>;

  // The four nullable fields go through one helper rather than four repetitions of the same
  // two-branch test, because a `typeof x !== "string" && x !== null` written four times is four
  // places for the null half to be forgotten.
  const reviewerId = nullableString(reviewer_id);
  const reviewerEmail = nullableString(reviewer_email_snapshot);
  const createdAt = nullableString(created_at);
  const updatedAt = nullableString(updated_at);

  if (
    typeof analysis_id !== "string" ||
    typeof status !== "string" ||
    typeof note !== "string" ||
    reviewerId === undefined ||
    reviewerEmail === undefined ||
    createdAt === undefined ||
    updatedAt === undefined
  ) {
    return null;
  }

  return {
    analysis_id,
    status,
    note,
    reviewer_id: reviewerId,
    reviewer_email_snapshot: reviewerEmail,
    created_at: createdAt,
    updated_at: updatedAt,
  };
}

/**
 * Where one analysis stands in the review workflow.
 *
 * A 404 here means the analysis does not exist, never that it has not been reviewed — the API
 * answers the second with a 200 carrying `UNREVIEWED`, which is what lets this screen render
 * the entire history of the deployment without a special case.
 *
 * A 403 is reported rather than treated as a sign-in problem, for the reason `fetchAccounts`
 * gives: the session is fine and the role is not, and sending the reader to sign in would send
 * them round a loop that cannot fix it.
 */
export async function fetchReview(analysisId: string): Promise<ReviewResult> {
  try {
    const response = await fetch(`${apiUrl()}${reviewUrl(analysisId)}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(REVIEW_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        missing: false,
        unauthenticated: true,
        error: "Sign in to read this review.",
      };
    }

    if (response.status === 403) {
      return {
        ok: false,
        missing: false,
        unauthenticated: false,
        error: "This account is not an administrator.",
      };
    }

    if (response.status === 404) {
      return {
        ok: false,
        missing: true,
        unauthenticated: false,
        error: "No analysis was found with this id.",
      };
    }

    // The API's own failure detail is server-side context, not something to surface here, so
    // every other unsuccessful status becomes one generic message.
    if (!response.ok) {
      return {
        ok: false,
        missing: false,
        unauthenticated: false,
        error: "The review is temporarily unavailable.",
      };
    }

    const review = parseReview(await response.json().catch(() => null));
    if (review === null) {
      return {
        ok: false,
        missing: false,
        unauthenticated: false,
        error: "The review could not be read.",
      };
    }

    return { ok: true, review };
  } catch {
    // A timeout or an unreachable API. Indistinguishable from here and equally unactionable by
    // the reader, so they get the one sentence that is true of both.
    return {
      ok: false,
      missing: false,
      unauthenticated: false,
      error: "The review is temporarily unavailable.",
    };
  }
}
