/**
 * The administrative surface's half of the account API: reading the list, and naming a change.
 *
 * The same split `analysis.ts` makes for the workspace, made again here rather than extended
 * there. `analysis.ts` is 2,000 lines about one record and two pages read it; accounts are a
 * different record read by one page, and folding them in would mean the report route pulling
 * an administrative parser it has no use for into its bundle. What is shared is the plumbing
 * — `apiUrl`, the session headers, the request id — and that is imported rather than copied.
 *
 * Nothing here decides anything. It does not filter the listing by role, does not hide an
 * account from an administrator, and does not check whether a change is allowed: every one of
 * those is the API's, in `app/api/admin_users.py`, and a second opinion formed in this process
 * could only ever disagree with the one that is actually enforced. What these functions do is
 * carry the API's answer — including its refusals — to something that can render it.
 *
 * The payload is parsed rather than cast. `await response.json()` is `any` as far as
 * TypeScript is concerned, and an `as AdminAccount[]` on it would be this file asserting a
 * shape it has not looked at; a field that arrived missing or as the wrong type would then
 * surface as a blank cell or a crash in a component, a long way from the response that caused
 * it. So a row that is not the expected shape fails the whole read, and the page says the
 * listing could not be read instead of drawing a table with holes in it.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

// The API path this module reads, in one place. The page links to `/admin`; this is the
// internal address behind it, and the two are unrelated strings that would otherwise both be
// spelled out wherever they are used.
export const ADMIN_USERS_URL = "/api/v1/admin/users";

// How long the API is given to answer. One unordered read of a small table, so it is held to
// the same short bound as the workspace's own listing rather than to the generous ones the
// submission paths use for a download.
export const ACCOUNTS_TIMEOUT_MS = 5000;

/**
 * An account as `/api/v1/admin/users` reports it.
 *
 * Five fields, which is the whole of what the API returns — see `AdminUser` in
 * `app/api/admin_users.py`. There is no password hash here and there is none there; this type
 * is the second statement of that contract and `parseAccount` is what enforces it, in the
 * sense that anything else the API ever sent would simply not be carried across.
 *
 * `created_at` stays a string. It is the API's own timestamp and the workspace shows one as
 * stored rather than reformatted into a local rendering the record does not hold; parsing it
 * into a `Date` here would be this file deciding a presentation the page has not asked for.
 */
export type AdminAccount = {
  id: string;
  email: string;
  role: string;
  is_active: boolean;
  created_at: string;
};

/**
 * What a read of the account list produced.
 *
 * The unauthenticated case is its own field rather than another error string, because it is
 * the one outcome the page can act on: a session the API would not accept sends the reader to
 * sign in, and everything else it can only report. This is the shape `fetchAnalyses` uses, and
 * the page below reads it the same way the workspace reads that one.
 */
export type AccountsResult =
  | { ok: true; accounts: AdminAccount[] }
  | { ok: false; unauthenticated: boolean; error: string };

/** One account out of the payload, or null for anything that is not one. */
export function parseAccount(payload: unknown): AdminAccount | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { id, email, role, is_active, created_at } = payload as Record<string, unknown>;

  if (
    typeof id !== "string" ||
    typeof email !== "string" ||
    typeof role !== "string" ||
    typeof is_active !== "boolean" ||
    typeof created_at !== "string"
  ) {
    return null;
  }

  return { id, email, role, is_active, created_at };
}

/**
 * Every account in the system, as the API orders them.
 *
 * The order is not re-established here. The API sorts by creation time and breaks ties on the
 * id precisely so the table does not reshuffle itself when somebody's role changes; sorting
 * again in this process would replace that with whatever this file thought was sensible, and
 * the two would drift.
 *
 * A 403 is reported rather than treated as a sign-in problem. It means the session is fine and
 * the role is not — a reader who reached this page without being an administrator, which the
 * layout guard normally prevents — and sending them to sign in would send them round a loop
 * that cannot fix it.
 */
export async function fetchAccounts(): Promise<AccountsResult> {
  try {
    const response = await fetch(`${apiUrl()}${ADMIN_USERS_URL}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(ACCOUNTS_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return {
        ok: false,
        unauthenticated: true,
        error: "Sign in to manage accounts.",
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
        error: "The account list is temporarily unavailable.",
      };
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The account list could not be read.",
      };
    }

    const accounts = payload.map(parseAccount);
    if (accounts.some((account) => account === null)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The account list could not be read.",
      };
    }

    return { ok: true, accounts: accounts as AdminAccount[] };
  } catch {
    // A timeout or an unreachable API. Indistinguishable from here and equally unactionable
    // by the reader, so they get the one sentence that is true of both.
    return {
      ok: false,
      unauthenticated: false,
      error: "The account list is temporarily unavailable.",
    };
  }
}
