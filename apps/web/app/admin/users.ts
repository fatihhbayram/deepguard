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
 *
 * R8-T8 added the rest of the lifecycle: reading one account, creating one, and resetting a
 * password. Three things about that are worth stating here rather than at each function.
 *
 * **A password passes through this module and is kept nowhere.** `createAccount` and
 * `resetPassword` take one as an argument, put it in one request body, and hold no copy. It is
 * never logged — the route handlers beside this file log an account id and never the object the
 * password is on — never put in a URL, and never returned: the API answers a creation with an
 * `AdminAccount`, which has no password field, and a reset with `204` and no body at all.
 *
 * **Nothing here returns a secret, which is why none of this needs client-side JavaScript.**
 * `createApiKey` in `api-keys.ts` is the one function in this application that does, and the
 * long note there explains why that flow alone has a client component: a secret cannot travel
 * back in a query string. These mutations carry a secret only *inwards*, in a POST body, so the
 * plain-form-and-redirect pattern every other administrative control uses works unchanged.
 *
 * **There is no delete.** Not because one was forgotten: `media_files.uploader_id` references
 * `users.id` with `ON DELETE RESTRICT` and the audit log keeps actor ids, so removing an account
 * would either fail on a constraint or destroy the history that makes an analysis attributable.
 * Deactivation — `updateAccount` with `is_active: false` — is the removal this system has, and
 * the detail page says so in those words rather than offering a "Delete" that means something
 * else.
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

// And a write. A single insert, or a single locked update, or a hash-and-two-statements
// transaction — none of which does work a read does not, except the Argon2id hashing the two
// password paths pay. That is deliberately expensive and deliberately bounded, and it is well
// inside this budget at the library's default parameters.
export const ACCOUNTS_MUTATION_TIMEOUT_MS = 5000;

// Enough of the API's message to be useful, and bounded. It is DeepGuard's own client-facing
// text — "An account with that email already exists." and the like — and it is rendered as text
// by React, never as markup.
const MAX_ERROR_LENGTH = 200;

/**
 * The shortest password this application will accept, restated from `app.web_auth`.
 *
 * The API is the authority and refuses anything shorter with a 422, whatever this file thinks.
 * What this constant is for is the form: a `minLength` on the input and a sentence under it, so
 * an operator setting somebody's password learns the floor while they are typing rather than
 * after a round trip. It must match `MINIMUM_PASSWORD_LENGTH` in `apps/api/app/web_auth.py` —
 * the same restating across languages `USER_ROLE_ADMIN` is, and for the same reason: there is
 * no way to derive one from the other.
 *
 * A floor that drifted *below* the API's would produce a form that accepts what the API
 * refuses, which is a confusing round trip and nothing worse. One that drifted above would
 * refuse passwords the API would take, in the browser, where it cannot be appealed.
 */
export const MINIMUM_PASSWORD_LENGTH = 12;

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

/**
 * What a read of one account produced.
 *
 * `missing` is its own field beside `unauthenticated`, the shape `ReviewResult` uses: an id that
 * names no account is a 404 the page turns into Next's own not-found rendering, and it has to be
 * distinguishable from "the API could not be reached", which is a sentence to show on a page
 * that still exists.
 */
export type AccountResult =
  | { ok: true; account: AdminAccount }
  | { ok: false; missing: boolean; unauthenticated: boolean; error: string };

/**
 * What a mutation produced.
 *
 * `unauthenticated` is its own field for the reason it is on the reads: it is the one outcome
 * the caller can act on, by sending the reader to sign in. Everything else — the 403 a
 * non-administrator gets, the 409 a duplicate address gets, the 400 a broken invariant gets and
 * the 422 an unusable field gets — is a sentence to show beside the form.
 *
 * The successful case carries no payload. `createAccount` has an account to hand back and does;
 * `updateAccount` and `resetPassword` are consumed by route handlers that answer with a redirect
 * and have nothing to render, so neither asks the API to say more than that it worked.
 */
export type MutationResult =
  | { ok: true }
  | { ok: false; unauthenticated: boolean; error: string };

export type CreateAccountResult =
  | { ok: true; account: AdminAccount }
  | { ok: false; unauthenticated: boolean; error: string };

/** What an account is being asked to become. Every field optional, as the API's PATCH is. */
export type AccountChange = {
  email?: string;
  role?: string;
  is_active?: boolean;
};

/** What a new account is being created as. `role` and `is_active` default at the API. */
export type NewAccount = {
  email: string;
  password: string;
  role?: string;
  is_active?: boolean;
};

/** What the API said went wrong, or a generic statement when it said nothing usable. */
async function failureText(response: Response, fallback: string): Promise<string> {
  const payload = await response.json().catch(() => null);

  if (typeof payload !== "object" || payload === null) {
    return fallback;
  }

  const { detail } = payload as Record<string, unknown>;

  // A 422 from pydantic reports `detail` as a list of per-field objects rather than a string.
  // Those name internal field paths and are not a sentence anybody should be shown, so only a
  // string detail is passed on; the list becomes the caller's own fallback.
  return typeof detail === "string" && detail.length > 0
    ? detail.slice(0, MAX_ERROR_LENGTH)
    : fallback;
}

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

/**
 * One account, for the screen that shows a single account.
 *
 * A 404 is reported as `missing` rather than as another error sentence, because it is the one
 * failure the page answers with a different rendering — Next's `notFound()` — instead of a
 * message beside a table that would have nothing in it.
 *
 * The id is interpolated after `encodeURIComponent`, which is what keeps a value out of the URL
 * on a route segment that arrived from a path. The API validates it as a UUID and answers 422
 * for anything else, so nothing here checks the shape — a second validation in this process
 * would be one that could disagree.
 */
export async function fetchAccount(accountId: string): Promise<AccountResult> {
  let response: Response;

  try {
    response = await fetch(
      `${apiUrl()}${ADMIN_USERS_URL}/${encodeURIComponent(accountId)}`,
      {
        cache: "no-store",
        headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
        signal: AbortSignal.timeout(ACCOUNTS_TIMEOUT_MS),
      },
    );
  } catch {
    return {
      ok: false,
      missing: false,
      unauthenticated: false,
      error: "The account is temporarily unavailable.",
    };
  }

  if (response.status === 401) {
    return {
      ok: false,
      missing: false,
      unauthenticated: true,
      error: "Sign in to manage accounts.",
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
      error: "No account has that id.",
    };
  }

  if (!response.ok) {
    return {
      ok: false,
      missing: false,
      unauthenticated: false,
      error: "The account is temporarily unavailable.",
    };
  }

  const account = parseAccount(await response.json().catch(() => null));

  if (account === null) {
    return {
      ok: false,
      missing: false,
      unauthenticated: false,
      error: "The account could not be read.",
    };
  }

  return { ok: true, account };
}

/**
 * Put a new account on file, with a password the administrator chose.
 *
 * **The password is in the body and nowhere else.** It is not logged by this function or by the
 * handler that calls it, it is not put in the URL, and it is not returned — the API answers with
 * an `AdminAccount`, which has no password field, so there is no shape in which the credential
 * could come back even by accident.
 *
 * The address is not normalized here and not checked for a duplicate here. Both are the API's:
 * `normalize_email` runs in its validator and the unique index is what actually enforces
 * uniqueness, and a copy of either rule in this process would be one that could disagree with
 * the one that holds. What comes back on a collision is the API's own 409 sentence.
 *
 * The `origin` argument is the browser's own `Origin`, already checked against this server's
 * host by the route handler that calls this. It is forwarded because the API's
 * `require_same_origin` demands one and a server-to-server call carries none of its own; see
 * `forwardedOrigin` in `session.ts` for why fabricating one here would defeat the check rather
 * than satisfy it.
 */
export async function createAccount(
  account: NewAccount,
  origin: Record<string, string>,
): Promise<CreateAccountResult> {
  let response: Response;

  try {
    response = await fetch(`${apiUrl()}${ADMIN_USERS_URL}`, {
      method: "POST",
      headers: {
        ...(await sessionHeaders()),
        ...origin,
        ...(await requestIdHeaders()),
        "content-type": "application/json",
      },
      body: JSON.stringify(account),
      signal: AbortSignal.timeout(ACCOUNTS_MUTATION_TIMEOUT_MS),
    });
  } catch {
    return {
      ok: false,
      unauthenticated: false,
      error: "The API could not be reached.",
    };
  }

  if (response.status === 401) {
    return { ok: false, unauthenticated: true, error: "Sign in to create an account." };
  }

  if (!response.ok) {
    return {
      ok: false,
      unauthenticated: false,
      // Covers the 403 a non-administrator gets, the 409 a duplicate address gets and the 422 an
      // unusable field gets. The API's own sentence is the useful one when it sent a string.
      error: await failureText(response, "The account could not be created."),
    };
  }

  const created = parseAccount(await response.json().catch(() => null));

  if (created === null) {
    // The account may well exist — a parse failure here says the response was not the expected
    // shape, not that the write failed — so the message does not claim it does not. The listing
    // is what settles it, and it is one reload away.
    return {
      ok: false,
      unauthenticated: false,
      error: "The account was created but could not be read back. Check the list.",
    };
  }

  return { ok: true, account: created };
}

/**
 * Change an account's address, role, activation, or any combination of them.
 *
 * Only the fields the caller actually names travel. The API's PATCH treats an absent field as
 * "leave it alone", and sending the untouched ones as `null` would turn every submission into a
 * change to everything — restoring whatever this page last rendered over a value another
 * administrator may have just altered.
 *
 * Nothing here decides whether the change is allowed. Not the self-modification rule, not the
 * last-administrator invariant, not the set of roles, and not whether the address is free: all
 * of them are the API's, in `app/api/admin_users.py`, and a copy of any of them here would be a
 * second rule that could drift from the one actually enforced.
 */
export async function updateAccount(
  accountId: string,
  change: AccountChange,
  origin: Record<string, string>,
): Promise<MutationResult> {
  let response: Response;

  try {
    response = await fetch(
      `${apiUrl()}${ADMIN_USERS_URL}/${encodeURIComponent(accountId)}`,
      {
        method: "PATCH",
        headers: {
          ...(await sessionHeaders()),
          ...origin,
          ...(await requestIdHeaders()),
          "content-type": "application/json",
        },
        body: JSON.stringify(change),
        signal: AbortSignal.timeout(ACCOUNTS_MUTATION_TIMEOUT_MS),
      },
    );
  } catch {
    return {
      ok: false,
      unauthenticated: false,
      error: "The API could not be reached.",
    };
  }

  if (response.status === 401) {
    return { ok: false, unauthenticated: true, error: "Sign in to manage accounts." };
  }

  if (!response.ok) {
    return {
      ok: false,
      unauthenticated: false,
      error: await failureText(response, "The change was refused."),
    };
  }

  return { ok: true };
}

/**
 * Replace an account's password, which also ends every session it was holding open.
 *
 * **The plaintext goes out in the body and is kept nowhere in this process.** Not in a log line,
 * not in the URL, and not in the result — this returns `{ ok: true }` and no payload, because
 * there is nothing a successful reset produces that anybody should be shown twice. The API
 * answers 204 with no body for the same reason.
 *
 * The session revocation is not requested separately and could not be: it is part of the same
 * transaction as the new hash on the API side, which is what keeps a reset from leaving the old
 * sessions alive. There is no argument here to turn it off, and there should not be one.
 *
 * An administrator resetting their own password ends their own session too, so the handler that
 * calls this has to expect the next request to be unauthenticated. That is correct rather than a
 * bug — see `reset_password` in the API — and the handler says how it is presented.
 */
export async function resetPassword(
  accountId: string,
  password: string,
  origin: Record<string, string>,
): Promise<MutationResult> {
  let response: Response;

  try {
    response = await fetch(
      `${apiUrl()}${ADMIN_USERS_URL}/${encodeURIComponent(accountId)}/password`,
      {
        method: "POST",
        headers: {
          ...(await sessionHeaders()),
          ...origin,
          ...(await requestIdHeaders()),
          "content-type": "application/json",
        },
        body: JSON.stringify({ password }),
        signal: AbortSignal.timeout(ACCOUNTS_MUTATION_TIMEOUT_MS),
      },
    );
  } catch {
    return {
      ok: false,
      unauthenticated: false,
      error: "The API could not be reached.",
    };
  }

  if (response.status === 401) {
    return { ok: false, unauthenticated: true, error: "Sign in to manage accounts." };
  }

  if (!response.ok) {
    return {
      ok: false,
      unauthenticated: false,
      error: await failureText(response, "The password could not be reset."),
    };
  }

  return { ok: true };
}
