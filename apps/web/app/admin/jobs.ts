/**
 * The administrative surface's half of the job API: reading the detection queue.
 *
 * The same split `users.ts` makes, made again for the same reason — one record, one reader,
 * one module. What is shared with it is the plumbing (`apiUrl`, the session headers, the
 * request id), and that is imported rather than copied.
 *
 * Nothing here decides anything, and one thing in particular it does not decide: whether a job
 * is stale. That is `is_stale` on the payload, computed by the API against the database clock
 * in the same statement that read the row — see `stale_lease` in `app/api/admin_jobs.py`. A
 * `new Date(lease_expires_at) < new Date()` written here would be the browser's clock judging
 * the worker's lease, and the two disagree by however far the reader's machine is out. So the
 * boolean is carried across and the deadline is carried across beside it as a fact to read,
 * never as an input to a comparison.
 *
 * The payload is parsed rather than cast, the convention `users.ts` states in full: a row that
 * is not the expected shape fails the whole read, and the page says the listing could not be
 * read instead of drawing a table with holes in it.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

// The API path this module reads, in one place. The page is at `/admin/jobs`; this is the
// internal address behind it, and the two are unrelated strings.
export const ADMIN_JOBS_URL = "/api/v1/admin/jobs";

// How long the API is given to answer. One ordered read of a capped number of rows, held to
// the same short bound as the account listing beside it.
export const JOBS_TIMEOUT_MS = 5000;

/**
 * One job as `/api/v1/admin/jobs` reports it.
 *
 * Nine fields, which is the whole of what the API returns — see `AdminJob` in
 * `app/api/admin_jobs.py`. Eight of them are columns on the job row; `is_stale` is derived
 * server-side and has no column behind it.
 *
 * The three timestamps stay strings. They are the API's own values and the workspace shows a
 * timestamp as stored rather than reformatted into a local rendering the record does not hold
 * — and here that convention is load-bearing rather than stylistic: a `Date` built from
 * `lease_expires_at` in this process is exactly the thing the module comment says must not
 * decide anything.
 */
export type AdminJob = {
  id: string;
  analysis_id: string;
  status: string;
  request_id: string | null;
  lease_expires_at: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  is_stale: boolean;
};

/**
 * What a read of the job list produced.
 *
 * The same shape `AccountsResult` has, and read the same way: the unauthenticated case is its
 * own field because it is the one outcome the page can act on, and everything else it can only
 * report.
 */
export type JobsResult =
  | { ok: true; jobs: AdminJob[] }
  | { ok: false; unauthenticated: boolean; error: string };

/** A field that is either a string or absent, or `undefined` for anything that is neither. */
function nullableString(value: unknown): string | null | undefined {
  if (value === null || typeof value === "string") {
    return value ?? null;
  }

  return undefined;
}

/** One job out of the payload, or null for anything that is not one. */
export function parseJob(payload: unknown): AdminJob | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    id,
    analysis_id,
    status,
    request_id,
    lease_expires_at,
    error_message,
    created_at,
    updated_at,
    is_stale,
  } = payload as Record<string, unknown>;

  // The four nullable fields are read through one helper rather than four repetitions of the
  // same two-branch test, because a `typeof x !== "string" && x !== null` written four times
  // is four places for the null half to be forgotten — and the field it would be forgotten on
  // is `error_message`, which is null on every job that has not failed.
  const requestId = nullableString(request_id);
  const leaseExpiresAt = nullableString(lease_expires_at);
  const errorMessage = nullableString(error_message);

  if (
    typeof id !== "string" ||
    typeof analysis_id !== "string" ||
    typeof status !== "string" ||
    typeof created_at !== "string" ||
    typeof updated_at !== "string" ||
    // Strictly a boolean. A missing `is_stale` read as falsy would silently stop labelling
    // stale jobs — the one thing this listing exists to show — and would do it without any
    // sign that the contract had changed.
    typeof is_stale !== "boolean" ||
    requestId === undefined ||
    leaseExpiresAt === undefined ||
    errorMessage === undefined
  ) {
    return null;
  }

  return {
    id,
    analysis_id,
    status,
    request_id: requestId,
    lease_expires_at: leaseExpiresAt,
    error_message: errorMessage,
    created_at,
    updated_at,
    is_stale,
  };
}

/**
 * The most recent jobs, as the API orders them.
 *
 * The order is not re-established here. The API sorts newest first and breaks ties on the id;
 * sorting again in this process would replace that with whatever this file thought was
 * sensible, and the two would drift.
 *
 * A 403 is reported rather than treated as a sign-in problem, for the reason `fetchAccounts`
 * gives: the session is fine and the role is not, and sending the reader to sign in would send
 * them round a loop that cannot fix it.
 */
export async function fetchJobs(): Promise<JobsResult> {
  try {
    const response = await fetch(`${apiUrl()}${ADMIN_JOBS_URL}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(JOBS_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        unauthenticated: true,
        error: "Sign in to inspect detection jobs.",
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
        error: "The job list is temporarily unavailable.",
      };
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The job list could not be read.",
      };
    }

    const jobs = payload.map(parseJob);
    if (jobs.some((job) => job === null)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The job list could not be read.",
      };
    }

    return { ok: true, jobs: jobs as AdminJob[] };
  } catch {
    // A timeout or an unreachable API. Indistinguishable from here and equally unactionable
    // by the reader, so they get the one sentence that is true of both.
    return {
      ok: false,
      unauthenticated: false,
      error: "The job list is temporarily unavailable.",
    };
  }
}
