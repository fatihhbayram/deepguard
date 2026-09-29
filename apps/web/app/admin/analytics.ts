/**
 * The administrative surface's half of the analytics API: reading the seven-day summary.
 *
 * The same split `jobs.ts` and `users.ts` make, made again for the same reason — one record,
 * one reader, one module. What is shared with them is the plumbing (`apiUrl`, the session
 * headers, the request id), and that is imported rather than copied.
 *
 * What is different here is that there is no record. Every value on this payload is a count
 * the database produced, and **nothing in this module or the page above it derives a second
 * number from them.** No percentage, no failure rate, no "healthiest provider", no total
 * assembled by summing a map. That is not squeamishness about arithmetic: a number computed
 * in the browser is a number with no definition anywhere in the system, and on a screen whose
 * subject is detector health an invented ratio is indistinguishable from a measured one. The
 * API decides what the numbers mean; this decides where they are drawn.
 *
 * The one exception is drawing itself — a bar has to be some fraction of its track's width, so
 * the page divides a count by the largest count beside it. That is a length, it is scoped to
 * the row it renders, and it never appears as text. It is stated here so the distinction stays
 * explicit rather than becoming precedent for a real ratio later.
 *
 * The two ratios on the payload, the feedback `disagreement_rate` (R14-T2) and each detector
 * deployment's `failure_rate` (R14-T4), are the API's: they are parsed and drawn as they arrive,
 * and nothing here divides to produce them.
 *
 * The payload is parsed rather than cast, the convention `users.ts` states in full: a body that
 * is not the expected shape fails the whole read, and the page says the summary could not be
 * read instead of drawing a dashboard with holes in it. That matters more on this screen than
 * on the tables next door — a missing row in a listing is visibly missing, whereas a count that
 * silently read as zero looks exactly like a quiet week.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

// The API path this module reads, in one place. The page is at `/admin/analytics`; this is the
// internal address behind it, and the two are unrelated strings.
export const ADMIN_ANALYTICS_URL = "/api/v1/admin/analytics";

// How long the API is given to answer. Six aggregate scans of a week's rows rather than the
// single indexed read the job listing makes, so a little more room than the 5s next door —
// still short enough that a wedged database fails the page rather than hanging it.
export const ANALYTICS_TIMEOUT_MS = 8000;

/**
 * The seven-day summary as `/api/v1/admin/analytics` reports it.
 *
 * Mirrors `AnalyticsWindow` in `app/api/admin_analytics.py`. The distributions are open maps
 * rather than named fields, and deliberately: the API guarantees the schema's vocabulary is
 * present and also carries through any status it did not recognise, and a closed type here
 * would drop exactly that value — the unrecognised one, which is the single most interesting
 * thing that can appear on this screen.
 *
 * The risk column arrives as three maps the API has already placed (R13-T3): `decisions` holds
 * the `r9-v5.0.0` verdicts and `UNDECIDED`, `recordedRiskLevels` the levels a pinned earlier
 * ruleset wrote, and `unrecognised` every stored pairing of ruleset and value that is neither.
 * The API placed each row by the ruleset version stored beside it, which this module never
 * sees; nothing here re-sorts, re-labels or second-guesses a key between the three.
 */
export type AdminAnalytics = {
  window: string;
  analyses_total: number;
  analyses_by_status: Record<string, number>;
  jobs_by_status: Record<string, number>;
  decisions: Record<string, number>;
  recordedRiskLevels: Record<string, number>;
  unrecognised: Record<string, number>;
  acquisition: Record<string, number>;
  detectors_health: DetectorHealth[];
  feedback_total: number;
  feedback_by_assessment: Record<string, number>;
  feedback_by_claimed_label: Record<string, number>;
  feedback_without_claimed_label: number;
  feedback_by_decision: Record<string, FeedbackBucket>;
  feedback_by_recorded_risk_level: Record<string, FeedbackBucket>;
  feedback_by_unrecognised_risk_state: Record<string, FeedbackBucket>;
};

/**
 * How one detector deployment's runs ended in the window (R14-T4), as the API counted them.
 *
 * `provider_version` is null when the rows were persisted without one — a real group, kept as
 * null, never replaced with a made-up name. `status_counts` holds `SUCCESS`, `FAILED`,
 * `TIMEOUT`, any status the API carried through, and `MISSING` only on a deployment a ruleset
 * expects. `failure_rate` is the API's `(FAILED + TIMEOUT) / (SUCCESS + FAILED + TIMEOUT)`,
 * null when nothing was run, and never recomputed here. It is an operational execution rate,
 * not a model error or false-positive rate.
 */
export type DetectorHealth = {
  provider: string;
  provider_version: string | null;
  status_counts: Record<string, number>;
  failure_rate: number | null;
};

/**
 * The user feedback given about one verdict (R14-T2), as the API counted it.
 *
 * `assessments` holds every other count on the bucket — `AGREE`, `DISAGREE`, `UNSURE`, and any
 * assessment the API carried through that this file has never heard of. `disagreement_rate` is
 * the API's number, never recomputed here: `DISAGREE / total_feedback`, null when there is no
 * feedback. It is unverified user opinion, not a false-positive or error rate.
 */
export type FeedbackBucket = {
  total_feedback: number;
  disagreement_rate: number | null;
  assessments: Record<string, number>;
};

/**
 * What a read of the summary produced.
 *
 * The same shape `JobsResult` has, and read the same way: the unauthenticated case is its own
 * field because it is the one outcome the page can act on, and everything else it can only
 * report.
 */
export type AnalyticsResult =
  | { ok: true; analytics: AdminAnalytics }
  | { ok: false; unauthenticated: boolean; error: string };

/**
 * A count, or undefined for anything that is not one.
 *
 * Strict about what a count may be. A non-integer or a negative would not be a count this API
 * can produce — every one of them is a `COUNT(*)` — so either means the contract has changed
 * and the read should fail loudly rather than render a dashboard built on it.
 */
function count(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    return undefined;
  }

  return value;
}

/** One distribution out of the payload, or null for anything that is not one. */
export function parseCounts(payload: unknown): Record<string, number> | null {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return null;
  }

  const counts: Record<string, number> = {};
  for (const [key, value] of Object.entries(payload)) {
    const parsed = count(value);
    // One bad entry fails the whole map rather than being skipped. A distribution missing a
    // category it could not parse would still render — as a smaller total, with no sign that
    // anything had been dropped — and that is the failure this convention exists to prevent.
    if (parsed === undefined) {
      return null;
    }
    counts[key] = parsed;
  }

  return counts;
}

/**
 * The detector health list, or null for anything that is not one.
 *
 * The rate is held to its range rather than merely typed — null, or a number from 0 to 1 —
 * because a rate outside it means the API's definition has changed. Whether it is null is the
 * API's call: deciding that here would take a sum of the counts, which is the arithmetic this
 * module does not do.
 */
function parseDetectorHealth(payload: unknown): DetectorHealth[] | null {
  if (!Array.isArray(payload)) {
    return null;
  }

  const health: DetectorHealth[] = [];
  for (const entry of payload) {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      return null;
    }

    const { provider, provider_version, status_counts, failure_rate } = entry as Record<
      string,
      unknown
    >;
    const counts = parseCounts(status_counts);

    if (
      typeof provider !== "string" ||
      (provider_version !== null && typeof provider_version !== "string") ||
      counts === null ||
      (failure_rate !== null && !isRate(failure_rate))
    ) {
      return null;
    }

    health.push({
      provider,
      provider_version,
      status_counts: counts,
      failure_rate: failure_rate as number | null,
    });
  }

  return health;
}

/**
 * One feedback bucket, or null for anything that is not one.
 *
 * The rate is held to the contract rather than merely typed: null exactly when there is no
 * feedback, otherwise a number from 0 to 1. A rate beside a zero total, or a null beside a
 * non-zero one, means the API's definition of it has changed, and the read fails.
 */
function parseFeedbackBucket(payload: unknown): FeedbackBucket | null {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return null;
  }

  const { total_feedback, disagreement_rate, ...rest } = payload as Record<string, unknown>;
  const total = count(total_feedback);
  const assessments = parseCounts(rest);

  if (total === undefined || assessments === null) {
    return null;
  }

  if (total === 0 ? disagreement_rate !== null : !isRate(disagreement_rate)) {
    return null;
  }

  return {
    total_feedback: total,
    disagreement_rate: disagreement_rate as number | null,
    assessments,
  };
}

function isRate(value: unknown): boolean {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}

/** A map of verdicts to their feedback buckets. */
function parseFeedbackBuckets(payload: unknown): Record<string, FeedbackBucket> | null {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return null;
  }

  const buckets: Record<string, FeedbackBucket> = {};
  for (const [key, value] of Object.entries(payload)) {
    const bucket = parseFeedbackBucket(value);
    if (bucket === null) {
      return null;
    }
    buckets[key] = bucket;
  }

  return buckets;
}

/** The summary out of the payload, or null for anything that is not one. */
export function parseAnalytics(payload: unknown): AdminAnalytics | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    window,
    analyses_total,
    analyses_by_status,
    jobs_by_status,
    decisions,
    recorded_risk_levels,
    unrecognised,
    acquisition,
    detectors_health,
    feedback_total,
    feedback_by_assessment,
    feedback_by_claimed_label,
    feedback_without_claimed_label,
    feedback_by_decision,
    feedback_by_recorded_risk_level,
    feedback_by_unrecognised_risk_state,
  } = payload as Record<string, unknown>;

  const total = count(analyses_total);
  const analyses = parseCounts(analyses_by_status);
  const jobs = parseCounts(jobs_by_status);
  const decided = parseCounts(decisions);
  const recorded = parseCounts(recorded_risk_levels);
  const unplaced = parseCounts(unrecognised);
  const acquired = parseCounts(acquisition);
  const providers = parseDetectorHealth(detectors_health);
  const feedbackTotal = count(feedback_total);
  const byAssessment = parseCounts(feedback_by_assessment);
  const byClaimedLabel = parseCounts(feedback_by_claimed_label);
  const withoutClaimedLabel = count(feedback_without_claimed_label);
  const byDecision = parseFeedbackBuckets(feedback_by_decision);
  const byRecordedLevel = parseFeedbackBuckets(feedback_by_recorded_risk_level);
  const byUnrecognised = parseFeedbackBuckets(feedback_by_unrecognised_risk_state);

  if (
    feedbackTotal === undefined ||
    byAssessment === null ||
    byClaimedLabel === null ||
    withoutClaimedLabel === undefined ||
    byDecision === null ||
    byRecordedLevel === null ||
    byUnrecognised === null
  ) {
    return null;
  }

  if (
    typeof window !== "string" ||
    total === undefined ||
    analyses === null ||
    jobs === null ||
    decided === null ||
    recorded === null ||
    unplaced === null ||
    acquired === null ||
    providers === null
  ) {
    return null;
  }

  return {
    window,
    analyses_total: total,
    analyses_by_status: analyses,
    jobs_by_status: jobs,
    decisions: decided,
    recordedRiskLevels: recorded,
    unrecognised: unplaced,
    acquisition: acquired,
    detectors_health: providers,
    feedback_total: feedbackTotal,
    feedback_by_assessment: byAssessment,
    feedback_by_claimed_label: byClaimedLabel,
    feedback_without_claimed_label: withoutClaimedLabel,
    feedback_by_decision: byDecision,
    feedback_by_recorded_risk_level: byRecordedLevel,
    feedback_by_unrecognised_risk_state: byUnrecognised,
  };
}

/**
 * The deployment's last seven days, as the API counted them.
 *
 * The window is not named here and not computed here. It arrives on the payload as the API's
 * own label, because the boundary these counts were taken against is a `now()` PostgreSQL
 * evaluated — a "last 7 days" caption written in this file would be this process's claim about
 * a window it had no part in choosing, and would go on reading "7 days" after the API's own
 * changed.
 *
 * A 403 is reported rather than treated as a sign-in problem, for the reason `fetchAccounts`
 * gives: the session is fine and the role is not, and sending the reader to sign in would send
 * them round a loop that cannot fix it.
 */
export async function fetchAnalytics(): Promise<AnalyticsResult> {
  try {
    const response = await fetch(`${apiUrl()}${ADMIN_ANALYTICS_URL}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(ANALYTICS_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        unauthenticated: true,
        error: "Sign in to read the operational summary.",
      };
    }

    if (response.status === 403) {
      return {
        ok: false,
        unauthenticated: false,
        error: "This account is not an administrator.",
      };
    }

    // The API's own failure detail is server-side context, not something to surface here, so
    // every other unsuccessful status becomes one generic message.
    if (!response.ok) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The operational summary is temporarily unavailable.",
      };
    }

    const analytics = parseAnalytics(await response.json().catch(() => null));
    if (analytics === null) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The operational summary could not be read.",
      };
    }

    return { ok: true, analytics };
  } catch {
    // A timeout or an unreachable API. Indistinguishable from here and equally unactionable
    // by the reader, so they get the one sentence that is true of both.
    return {
      ok: false,
      unauthenticated: false,
      error: "The operational summary is temporarily unavailable.",
    };
  }
}
