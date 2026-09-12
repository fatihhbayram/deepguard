/**
 * The administrative surface's half of the API-key API: reading the keys, issuing one, ending one.
 *
 * The same split `users.ts`, `jobs.ts`, `analytics.ts` and `audit.ts` make, made again for the
 * same reason — one record, one reader, one module. What is shared with them is the plumbing
 * (`apiUrl`, the session headers, the request id), and that is imported rather than copied.
 *
 * Every function here runs on the server. Not one of them is reachable from the browser, and
 * that is structural rather than a convention: the API serves no CORS headers, so a `fetch`
 * from a page would be blocked, and the session cookie these calls authenticate with is
 * `httpOnly` and unreadable by script in the first place. The browser talks to the two route
 * handlers beside this file; those call these.
 *
 * **`createApiKey` is the only function in this application that returns a secret, and this
 * module is where the rules about it are stated.**
 *
 * - The value exists in this process for the length of one request and is handed straight back
 *   to the route handler, which puts it in one JSON response and keeps no copy.
 * - It is never logged. `logInfo` is called on the way through the route handler with the key's
 *   id and nothing else, which is deliberately not the same object.
 * - It is never put in a URL. That is the reason the creation flow is the one control on this
 *   surface with client-side JavaScript behind it: every other administrative mutation answers
 *   with a `303` carrying its outcome in the query string, and a secret sent that way would be
 *   written into browser history, into this server's access log, and into the `Referer` of the
 *   next request the page makes. A secret in a query string is a secret that has leaked.
 * - It is never persisted anywhere by this application — no cookie, no `localStorage`, no
 *   session store. The component that shows it holds it in React state, which is gone on the
 *   next navigation and is not recoverable afterwards. The API cannot reissue it either; there
 *   is only the digest on the row.
 *
 * Nothing here decides anything. It does not filter revoked keys out of the listing, does not
 * check whether a name is acceptable, and does not decide whether a revocation is allowed:
 * every one of those is the API's, in `app/api/admin_api_keys.py`, and a second opinion formed
 * in this process could only ever disagree with the one that is actually enforced.
 *
 * The payload is parsed rather than cast, the convention `users.ts` sets and for the reason it
 * gives: `await response.json()` is `any`, and an `as AdminApiKey[]` would be this file
 * asserting a shape it has not looked at.
 */

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import { sessionHeaders } from "../session";

// The API path this module reads and writes, in one place. The page is at `/admin/api-keys`;
// this is the internal address behind it, and the two are unrelated strings that happen to
// read alike.
export const ADMIN_API_KEYS_URL = "/api/v1/admin/api-keys";

// How long the API is given to answer a read. One ordered read of a small table, held to the
// same short bound as the account and job listings.
export const API_KEYS_TIMEOUT_MS = 5000;

// And a write. A single insert or a single locked update, so the same bound: neither mutation
// does any work a read does not.
export const API_KEYS_MUTATION_TIMEOUT_MS = 5000;

// Enough of the API's message to be useful, and bounded. It is DeepGuard's own client-facing
// text — "name must not be blank" and the like — and it is rendered as text by React, never as
// markup.
const MAX_ERROR_LENGTH = 200;

/**
 * A key as `/api/v1/admin/api-keys` reports it.
 *
 * Five fields, which is the whole of what the API returns — see `AdminApiKey` in
 * `app/api/admin_api_keys.py`. There is no `key_hash` here and there is none there; this type
 * is the second statement of that contract and `parseApiKey` is what enforces it, in the sense
 * that anything else the API ever sent would simply not be carried across.
 *
 * `created_at` and `last_used_at` stay strings. They are the API's own values, and this surface
 * shows a timestamp as stored rather than reformatted into a local rendering the record does
 * not hold — the convention every table here follows.
 */
export type AdminApiKey = {
  id: string;
  name: string;
  is_active: boolean;
  created_at: string;
  last_used_at: string | null;
};

/**
 * The creation response: a key, plus the one copy of its secret that will ever exist.
 *
 * A separate type from `AdminApiKey` rather than an optional field on it, mirroring the API's
 * own split. As a distinct type, the listing cannot accidentally be typed as carrying a
 * plaintext, and every place that handles one of these is a place a reader can see is handling
 * a secret.
 */
export type CreatedApiKey = AdminApiKey & { plaintext: string };

/** What a read of the key list produced. The shape `AccountsResult` has, read the same way. */
export type ApiKeysResult =
  | { ok: true; keys: AdminApiKey[] }
  | { ok: false; unauthenticated: boolean; error: string };

/**
 * What a mutation produced.
 *
 * `unauthenticated` is its own field for the reason it is on the read results: it is the one
 * outcome the caller can act on, by sending the reader to sign in. Everything else — including
 * the 403 a non-administrator gets and the 422 an unusable name gets — is a sentence to show.
 */
export type CreateResult =
  | { ok: true; key: CreatedApiKey }
  | { ok: false; unauthenticated: boolean; error: string };

export type RevokeResult =
  | { ok: true; key: AdminApiKey }
  | { ok: false; unauthenticated: boolean; error: string };

/** A field that is either a string or absent, or `undefined` for anything that is neither. */
function nullableString(value: unknown): string | null | undefined {
  if (value === null || typeof value === "string") {
    return value ?? null;
  }

  return undefined;
}

/** One key out of the payload, or null for anything that is not one. */
export function parseApiKey(payload: unknown): AdminApiKey | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { id, name, is_active, created_at, last_used_at } = payload as Record<
    string,
    unknown
  >;

  const lastUsed = nullableString(last_used_at);

  if (
    typeof id !== "string" ||
    typeof name !== "string" ||
    typeof is_active !== "boolean" ||
    typeof created_at !== "string" ||
    lastUsed === undefined
  ) {
    return null;
  }

  return { id, name, is_active, created_at, last_used_at: lastUsed };
}

/**
 * A creation response out of the payload, or null for anything that is not one.
 *
 * The plaintext is required, and a response without one fails the parse rather than being
 * carried across as an empty string. A blank secret rendered under "copy this now, you will
 * not see it again" would be the worst possible outcome of this screen: the operator would
 * believe they had been shown a key, and the real one would be unrecoverable.
 */
export function parseCreatedApiKey(payload: unknown): CreatedApiKey | null {
  const key = parseApiKey(payload);

  if (key === null) {
    return null;
  }

  const { plaintext } = payload as Record<string, unknown>;

  if (typeof plaintext !== "string" || plaintext.length === 0) {
    return null;
  }

  return { ...key, plaintext };
}

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

/**
 * Every key ever issued, as the API orders them.
 *
 * The order is not re-established here. The API sorts newest first and breaks ties on the id;
 * sorting again in this process would replace that with whatever this file thought was
 * sensible, and the two would drift.
 *
 * Revoked keys are included, because the API includes them. Filtering them out here would hide
 * exactly the rows somebody opened this screen to check.
 */
export async function fetchApiKeys(): Promise<ApiKeysResult> {
  try {
    const response = await fetch(`${apiUrl()}${ADMIN_API_KEYS_URL}`, {
      cache: "no-store",
      headers: { ...(await sessionHeaders()), ...(await requestIdHeaders()) },
      signal: AbortSignal.timeout(API_KEYS_TIMEOUT_MS),
    });

    if (response.status === 401) {
      return { ok: false, unauthenticated: true, error: "Sign in to manage API keys." };
    }

    if (response.status === 403) {
      return {
        ok: false,
        unauthenticated: false,
        error: "This account is not an administrator.",
      };
    }

    if (!response.ok) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The API key list is temporarily unavailable.",
      };
    }

    const payload = await response.json().catch(() => null);
    if (!Array.isArray(payload)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The API key list could not be read.",
      };
    }

    const keys = payload.map(parseApiKey);
    if (keys.some((key) => key === null)) {
      return {
        ok: false,
        unauthenticated: false,
        error: "The API key list could not be read.",
      };
    }

    return { ok: true, keys: keys as AdminApiKey[] };
  } catch {
    // A timeout or an unreachable API. Indistinguishable from here and equally unactionable by
    // the reader, so they get the one sentence that is true of both.
    return {
      ok: false,
      unauthenticated: false,
      error: "The API key list is temporarily unavailable.",
    };
  }
}

/**
 * Issue a key, and carry back the one copy of its secret.
 *
 * The `origin` argument is the browser's own `Origin`, already checked against this server's
 * host by the route handler that calls this. It is forwarded because the API's
 * `require_same_origin` demands one and a server-to-server call carries none of its own; see
 * `forwardedOrigin` in `session.ts` for why fabricating one here would defeat the check rather
 * than satisfy it.
 *
 * The returned object holds the plaintext. Its lifetime is the caller's, and every caller in
 * this repository puts it in exactly one response body and keeps nothing.
 */
export async function createApiKey(
  name: string,
  origin: Record<string, string>,
): Promise<CreateResult> {
  let response: Response;

  try {
    response = await fetch(`${apiUrl()}${ADMIN_API_KEYS_URL}`, {
      method: "POST",
      headers: {
        ...(await sessionHeaders()),
        ...origin,
        ...(await requestIdHeaders()),
        "content-type": "application/json",
      },
      body: JSON.stringify({ name }),
      signal: AbortSignal.timeout(API_KEYS_MUTATION_TIMEOUT_MS),
    });
  } catch {
    return {
      ok: false,
      unauthenticated: false,
      error: "The API could not be reached.",
    };
  }

  if (response.status === 401) {
    return { ok: false, unauthenticated: true, error: "Sign in to issue an API key." };
  }

  if (!response.ok) {
    return {
      ok: false,
      unauthenticated: false,
      // Covers the 403 a non-administrator gets and the 422 an unusable name gets. The API's
      // own sentence is the useful one when it sent a string.
      error: await failureText(response, "The key could not be issued."),
    };
  }

  const key = parseCreatedApiKey(await response.json().catch(() => null));

  if (key === null) {
    // The key may well have been created — a parse failure here says the response was not the
    // expected shape, not that the write failed — so the message does not claim it was not.
    // The listing is what settles it, and it is one reload away.
    return {
      ok: false,
      unauthenticated: false,
      error: "The key was issued but could not be read back. Check the list below.",
    };
  }

  return { ok: true, key };
}

/**
 * End a key's access.
 *
 * The id is interpolated into the path after `encodeURIComponent`, which is what keeps a value
 * from the form out of the structure of the URL. The API validates it as a UUID and answers
 * 422 for anything else, so nothing here checks the shape — a second validation in this
 * process would be one that could disagree.
 *
 * Revoking a key that is already revoked is a 200 from the API and therefore an `ok` here. It
 * is not an error and the page does not present it as one: the caller asked for a state and
 * that is the state.
 */
export async function revokeApiKey(
  keyId: string,
  origin: Record<string, string>,
): Promise<RevokeResult> {
  let response: Response;

  try {
    response = await fetch(
      `${apiUrl()}${ADMIN_API_KEYS_URL}/${encodeURIComponent(keyId)}/revoke`,
      {
        method: "POST",
        headers: {
          ...(await sessionHeaders()),
          ...origin,
          ...(await requestIdHeaders()),
        },
        signal: AbortSignal.timeout(API_KEYS_MUTATION_TIMEOUT_MS),
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
    return { ok: false, unauthenticated: true, error: "Sign in to revoke an API key." };
  }

  if (!response.ok) {
    return {
      ok: false,
      unauthenticated: false,
      error: await failureText(response, "The key could not be revoked."),
    };
  }

  const key = parseApiKey(await response.json().catch(() => null));

  if (key === null) {
    return {
      ok: false,
      unauthenticated: false,
      error: "The key was revoked but could not be read back. Check the list below.",
    };
  }

  return { ok: true, key };
}
