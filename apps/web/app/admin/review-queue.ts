/**
 * The administrative review queue (R14-T3): which analyses an administrator may want to open
 * next, filtered on facts the API already stores.
 *
 * **Being in the queue says nothing about the media.** The API orders by `created_at` and
 * nothing else, and every flag on an item names the stored fact it matched — feedback somebody
 * gave, a detector that did not answer, a review nobody has closed. None of it is a verdict, a
 * priority or a label, and this module reads it as such.
 *
 * Feedback arrives as counts per assessment rather than as a record: several accounts can give
 * feedback on one analysis, and the queue never picks one of them to show.
 *
 * The payload is parsed rather than cast, the convention `jobs.ts` states: an item that is not
 * the expected shape fails the whole read.
 */

import { apiUrl, V5_VERDICTS } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";
import { FEEDBACK_ASSESSMENT_LABELS } from "../user-feedback";

export const ADMIN_REVIEW_QUEUE_URL = "/api/v1/admin/analyses/queue";

const REVIEW_QUEUE_TIMEOUT_MS = 5000;

// One page, as the API defaults it.
export const REVIEW_QUEUE_PAGE_SIZE = 50;

// What the `decision` filter accepts: the v5 verdicts and `UNDECIDED`, the keys the operational
// summary's `decisions` map carries. Legacy risk levels are not decisions and are not offered.
export const QUEUE_DECISIONS = [...V5_VERDICTS, "UNDECIDED"] as const;

export const QUEUE_FEEDBACK_ASSESSMENTS = Object.keys(FEEDBACK_ASSESSMENT_LABELS);

/** The filters, as the page's form and the API's query string both spell them. */
export type ReviewQueueFilters = {
  feedback_assessment: string | null;
  decision: string | null;
  signal_error: boolean;
  unreviewed_only: boolean;
  offset: number;
};

/** One item as `ReviewQueueItem` in `app/api/admin_analyses.py` reports it. */
export type ReviewQueueItem = {
  analysis_id: string;
  created_at: string;
  status: string;
  media_sha256s: string[];
  risk_rules_version: string | null;
  decision: string | null;
  recorded_risk_level: string | null;
  unrecognised_risk_state: string | null;
  review_status: string;
  analyst_assessment: string | null;
  feedback_counts: Record<string, number>;
  has_disagree_feedback: boolean;
  signal_error_counts: Record<string, number>;
  has_signal_errors: boolean;
};

export type ReviewQueuePage = {
  items: ReviewQueueItem[];
  total: number;
  limit: number;
  offset: number;
};

export type ReviewQueueResult =
  | { ok: true; page: ReviewQueuePage }
  | { ok: false; unauthenticated: boolean; error: string };

/**
 * The filters out of the page's search parameters. A value outside the vocabulary is dropped
 * rather than forwarded, so a hand-edited address widens the queue instead of failing it.
 */
export function parseFilters(
  params: Record<string, string | string[] | undefined>,
): ReviewQueueFilters {
  const one = (name: string): string | null => {
    const value = params[name];
    return typeof value === "string" ? value : null;
  };

  const feedback = one("feedback_assessment");
  const decision = one("decision");
  const offset = Number.parseInt(one("offset") ?? "0", 10);

  return {
    feedback_assessment:
      feedback !== null && QUEUE_FEEDBACK_ASSESSMENTS.includes(feedback) ? feedback : null,
    decision:
      decision !== null && (QUEUE_DECISIONS as readonly string[]).includes(decision)
        ? decision
        : null,
    signal_error: one("signal_error") === "true",
    // On unless explicitly turned off, as the API defaults it.
    unreviewed_only: one("unreviewed_only") !== "false",
    offset: Number.isFinite(offset) && offset > 0 ? offset : 0,
  };
}

/** The query string for these filters, carrying only what differs from the API's defaults. */
export function filterQuery(filters: ReviewQueueFilters): string {
  const query = new URLSearchParams();

  if (filters.feedback_assessment !== null) {
    query.set("feedback_assessment", filters.feedback_assessment);
  }
  if (filters.decision !== null) {
    query.set("decision", filters.decision);
  }
  if (filters.signal_error) {
    query.set("signal_error", "true");
  }
  if (!filters.unreviewed_only) {
    query.set("unreviewed_only", "false");
  }
  if (filters.offset > 0) {
    query.set("offset", String(filters.offset));
  }

  return query.toString();
}

function nullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function parseCounts(value: unknown): Record<string, number> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }

  const counts: Record<string, number> = {};
  for (const [key, count] of Object.entries(value)) {
    if (typeof count !== "number" || !Number.isInteger(count) || count < 0) {
      return null;
    }
    counts[key] = count;
  }

  return counts;
}

export function parseItem(payload: unknown): ReviewQueueItem | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const item = payload as Record<string, unknown>;
  const feedbackCounts = parseCounts(item.feedback_counts);
  const signalErrorCounts = parseCounts(item.signal_error_counts);

  if (
    typeof item.analysis_id !== "string" ||
    typeof item.created_at !== "string" ||
    typeof item.status !== "string" ||
    !Array.isArray(item.media_sha256s) ||
    !item.media_sha256s.every((sha) => typeof sha === "string") ||
    !nullableString(item.risk_rules_version) ||
    !nullableString(item.decision) ||
    !nullableString(item.recorded_risk_level) ||
    !nullableString(item.unrecognised_risk_state) ||
    typeof item.review_status !== "string" ||
    !nullableString(item.analyst_assessment) ||
    feedbackCounts === null ||
    signalErrorCounts === null ||
    typeof item.has_disagree_feedback !== "boolean" ||
    typeof item.has_signal_errors !== "boolean"
  ) {
    return null;
  }

  return {
    analysis_id: item.analysis_id,
    created_at: item.created_at,
    status: item.status,
    media_sha256s: item.media_sha256s as string[],
    risk_rules_version: item.risk_rules_version,
    decision: item.decision,
    recorded_risk_level: item.recorded_risk_level,
    unrecognised_risk_state: item.unrecognised_risk_state,
    review_status: item.review_status,
    analyst_assessment: item.analyst_assessment,
    feedback_counts: feedbackCounts,
    has_disagree_feedback: item.has_disagree_feedback,
    signal_error_counts: signalErrorCounts,
    has_signal_errors: item.has_signal_errors,
  };
}

export function parsePage(payload: unknown): ReviewQueuePage | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { items, total, limit, offset } = payload as Record<string, unknown>;

  if (
    !Array.isArray(items) ||
    typeof total !== "number" ||
    typeof limit !== "number" ||
    typeof offset !== "number"
  ) {
    return null;
  }

  const parsed = items.map(parseItem);
  if (parsed.some((item) => item === null)) {
    return null;
  }

  return { items: parsed as ReviewQueueItem[], total, limit, offset };
}

export async function fetchReviewQueue(
  filters: ReviewQueueFilters,
): Promise<ReviewQueueResult> {
  const unavailable = {
    ok: false as const,
    unauthenticated: false,
    error: "The review queue is temporarily unavailable.",
  };

  const query = filterQuery(filters);

  try {
    const response = await fetch(
      `${apiUrl()}${ADMIN_REVIEW_QUEUE_URL}?${query ? `${query}&` : ""}limit=${REVIEW_QUEUE_PAGE_SIZE}`,
      {
        cache: "no-store",
        headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
        signal: AbortSignal.timeout(REVIEW_QUEUE_TIMEOUT_MS),
      },
    );

    if (response.status === 401) {
      return { ok: false, unauthenticated: true, error: "Sign in to read the review queue." };
    }

    if (response.status === 403) {
      return {
        ok: false,
        unauthenticated: false,
        error: "This account is not an administrator.",
      };
    }

    if (!response.ok) {
      return unavailable;
    }

    const page = parsePage(await response.json().catch(() => null));
    if (page === null) {
      return { ok: false, unauthenticated: false, error: "The review queue could not be read." };
    }

    return { ok: true, page };
  } catch {
    return unavailable;
  }
}
