/**
 * The administrative surface's half of the audit API: reading what administrators have done.
 *
 * The same split `jobs.ts`, `users.ts` and `analytics.ts` make, made again for the same reason
 * — one record, one reader, one module. What is shared with them is the plumbing (`apiUrl`,
 * the session headers, the request id), and that is imported rather than copied.
 *
 * **Nothing here writes, and there is no function in this file that could.** The audit surface
 * is a single GET; the events are written by the API, inside the transaction that makes the
 * change they record. A `deleteEvent` or `clearLog` helper would have no endpoint behind it,
 * and the reason that endpoint does not exist is in `app/api/admin_audit.py`: a record the
 * recorded party can erase is not evidence of anything.
 *
 * **The addresses on these rows are history, not directory lookups.** `actor_email_snapshot`
 * and `target_email_snapshot` were frozen when the change happened. This module does not
 * resolve them against anything, and the page above it must not either — an address here
 * answers "who was this, then", and the day somebody renames an account it will legitimately
 * disagree with what the account listing says. That is the feature.
 *
 * `changes` is carried across as the arbitrary object it is. Its keys are field names the API
 * chose and its values are whatever those fields hold — a string for `role`, a boolean for
 * `is_active`, and something else entirely the day a third field becomes auditable. So it is
 * validated as "an object whose values are old/new pairs" rather than against a fixed set of
 * fields: a closed type here would drop the one entry that a reader most needs to see, namely
 * the one this file was not written to expect.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

// The API path this module reads, in one place. The page is at `/admin/audit`; this is the
// internal address behind it, and the two are unrelated strings.
export const ADMIN_AUDIT_URL = "/api/v1/admin/audit";

// How long the API is given to answer. One ordered read of a capped number of rows, held to
// the same short bound as the job listing.
export const AUDIT_TIMEOUT_MS = 5000;

/**
 * One recorded field change: what it was, and what it became.
 *
 * `unknown` on both sides rather than `string`. The API writes whatever the column held, and
 * `is_active` holds booleans — typing this as a string would be a lie that `JSON.parse` would
 * never catch and that would surface as `false` rendering as nothing at all.
 */
export type FieldChange = { old: unknown; new: unknown };

/**
 * One event as `/api/v1/admin/audit` reports it.
 *
 * Mirrors `AdminAuditEntry` in `app/api/admin_audit.py`. The two snapshots are nullable
 * because the columns are: a null means the address was not recorded, not that the account had
 * none.
 *
 * `created_at` stays a string. It is the API's own value, and the workspace shows a timestamp
 * as stored rather than reformatted into a local rendering the record does not hold — the
 * convention the case log, the account table and the job listing all follow.
 */
export type AdminAuditEntry = {
  id: string;
  actor_id: string;
  actor_email_snapshot: string | null;
  action: string;
  target_type: string;
  target_id: string;
  target_email_snapshot: string | null;
  changes: Record<string, FieldChange>;
  request_id: string | null;
  created_at: string;
};

/**
 * What a read of the audit log produced.
 *
 * The same shape `JobsResult` has, and read the same way: the unauthenticated case is its own
 * field because it is the one outcome the page can act on, and everything else it can only
 * report.
 */
export type AuditResult =
  | { ok: true; events: AdminAuditEntry[] }
  | { ok: false; unauthenticated: boolean; error: string };

/** A field that is either a string or absent, or `undefined` for anything that is neither. */
function nullableString(value: unknown): string | null | undefined {
  if (value === null || typeof value === "string") {
    return value ?? null;
  }

  return undefined;
}

/**
 * The `changes` payload, or null for anything that is not one.
 *
 * Every entry must be an object carrying both an `old` and a `new` key. The keys are required
 * but their values are not constrained: `null` is a legitimate old value, and so is `false`,
 * so the test is for the key's presence rather than for its contents being truthy — the
 * difference between `"old" in entry` and `entry.old !== undefined` is the whole correctness
 * of this function.
 */
export function parseChanges(payload: unknown): Record<string, FieldChange> | null {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    return null;
  }

  const changes: Record<string, FieldChange> = {};
  for (const [field, entry] of Object.entries(payload)) {
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      return null;
    }
    if (!("old" in entry) || !("new" in entry)) {
      return null;
    }
    changes[field] = { old: entry.old, new: entry.new };
  }

  return changes;
}

/** One event out of the payload, or null for anything that is not one. */
export function parseEvent(payload: unknown): AdminAuditEntry | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const {
    id,
    actor_id,
    actor_email_snapshot,
    action,
    target_type,
    target_id,
    target_email_snapshot,
    changes,
    request_id,
    created_at,
  } = payload as Record<string, unknown>;

  // The three nullable fields go through one helper rather than three repetitions of the same
  // two-branch test, because a `typeof x !== "string" && x !== null` written three times is
  // three places for the null half to be forgotten.
  const actorEmail = nullableString(actor_email_snapshot);
  const targetEmail = nullableString(target_email_snapshot);
  const requestId = nullableString(request_id);
  const parsedChanges = parseChanges(changes);

  if (
    typeof id !== "string" ||
    typeof actor_id !== "string" ||
    typeof action !== "string" ||
    typeof target_type !== "string" ||
    typeof target_id !== "string" ||
    typeof created_at !== "string" ||
    parsedChanges === null ||
    actorEmail === undefined ||
    targetEmail === undefined ||
    requestId === undefined
  ) {
    return null;
  }

  return {
    id,
    actor_id,
    actor_email_snapshot: actorEmail,
    action,
    target_type,
    target_id,
    target_email_snapshot: targetEmail,
    changes: parsedChanges,
    request_id: requestId,
    created_at,
  };
}

/**
 * The most recent administrative changes, as the API orders them.
 *
 * The order is not re-established here. The API sorts newest first and breaks ties on the id;
 * sorting again in this process would replace that with whatever this file thought was
 * sensible, and the two would drift.
 *
 * A 403 is reported rather than treated as a sign-in problem, for the reason `fetchAccounts`
 * gives: the session is fine and the role is not, and sending the reader to sign in would send
 * them round a loop that cannot fix it.
 */
export async function fetchAuditEvents(): Promise<AuditResult> {
  try {
    const response = await fetch(`${apiUrl()}${ADMIN_AUDIT_URL}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(AUDIT_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        unauthenticated: true,
        error: "Sign in to read the audit log.",
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
        error: "The audit log is temporarily unavailable.",
      };
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The audit log could not be read.",
      };
    }

    const events = payload.map(parseEvent);
    if (events.some((event) => event === null)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The audit log could not be read.",
      };
    }

    return { ok: true, events: events as AdminAuditEntry[] };
  } catch {
    // A timeout or an unreachable API. Indistinguishable from here and equally unactionable
    // by the reader, so they get the one sentence that is true of both.
    return {
      ok: false,
      unauthenticated: false,
      error: "The audit log is temporarily unavailable.",
    };
  }
}
