/**
 * The web application's half of the session the API issues.
 *
 * The browser never talks to the API directly — it has not since the dashboard was built,
 * because the API serves no CORS headers — so every authenticated request goes out from this
 * server, and the session cookie the browser holds for *this* origin has to be restated on
 * that server-to-server call. That restating is the whole of what this file does, plus the
 * one check that keeps a browser mutation from being driven by a page on another site.
 *
 * The token itself is never read by anything here. It arrives as an opaque `HttpOnly` cookie
 * this server cannot interpret and does not try to: which account it belongs to, whether it
 * has expired and what it may reach are all questions only the API answers, and asking them
 * here would be a second opinion that could disagree with the one that matters. Nothing in
 * the web application decides authorization — see `app/api/analyses.py`.
 */

import { cookies } from "next/headers";

// The cookie the API sets, named here because this server has to find it in the jar to
// forward it. It must match `SESSION_COOKIE_NAME` in `app/web_auth.py`; there is no way to
// derive one from the other across the two languages, so the pairing is stated in both.
export const SESSION_COOKIE_NAME = "deepguard_session";

// Where an unauthenticated reader is sent, and where a sign-out ends.
export const LOGIN_PATH = "/login";

// How long the API is given to answer "who is this session". It is one indexed lookup, so
// it is held to the same short bound as the other reads the dashboard does on render.
export const SESSION_TIMEOUT_MS = 5000;

/** Who the session cookie authenticates, as `/auth/me` reports it. */
export type SessionUser = {
  id: string;
  email: string;
  role: string;
};

/**
 * The session cookie restated as a request header for the API call, or nothing.
 *
 * Only this one cookie is forwarded, by name. Passing the whole jar on would send the API
 * every unrelated cookie the browser happens to hold for this origin, which is more than the
 * API has any use for and more than it should be told.
 */
export async function sessionHeaders(): Promise<Record<string, string>> {
  const session = (await cookies()).get(SESSION_COOKIE_NAME);

  return session ? { cookie: `${SESSION_COOKIE_NAME}=${session.value}` } : {};
}

/** The user object the API returned, or null for anything that is not one. */
export function parseSessionUser(payload: unknown): SessionUser | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const { id, email, role } = payload as Record<string, unknown>;
  if (typeof id !== "string" || typeof email !== "string" || typeof role !== "string") {
    return null;
  }

  return { id, email, role };
}

/**
 * Whether a browser mutation came from a page on this same origin.
 *
 * The minimum CSRF protection this task calls for, and deliberately not a token scheme: the
 * session cookie is already `SameSite=Lax`, which keeps a browser from attaching it to a
 * cross-site POST at all, and this is the second, independent check behind it. A framework
 * with a per-form token would add a store, a rotation policy and a failure mode of its own
 * to a boundary two header comparisons already close.
 *
 * A missing `Origin` is a refusal, not a pass. Every browser sends one on a form POST, so
 * the only requests this turns away are the ones not made by a browser form — which is
 * exactly the shape a forged submission has. Reading it as "no opinion" and letting the
 * request through would leave a check that any attacker can skip by omitting a header.
 *
 * `Origin` is compared against the `Host` the request arrived on rather than against a
 * configured address. The container listens on one port and is published on another, so a
 * hardcoded origin would be wrong in every deployment but the one it was written for; the
 * host the browser actually asked for is the thing the browser's own origin has to match.
 */
/**
 * The browser's `Origin`, restated for the server-to-server call.
 *
 * The API enforces its own origin check on every dashboard mutation, and it has to: this
 * server is not the only thing that can reach the API from a browser — cookies are not
 * scoped by port, so a page on another local origin can post straight at it with the
 * session attached, and a check that lived only here would be one the attacker simply walks
 * around. But the call this server makes carries no origin of its own, so what the browser
 * sent is forwarded verbatim and validated there against the deployment's configured web
 * origin.
 *
 * Only ever called after `isSameOrigin` has already accepted the header, so what is
 * forwarded is a value this server has itself checked against its own host — not an
 * arbitrary string a caller chose. That ordering is the whole safety of forwarding it at
 * all, and it is why this returns nothing when the header is absent rather than inventing
 * one: a fabricated `Origin` from here would defeat the API's check on this server's behalf.
 */
export function forwardedOrigin(request: Request): Record<string, string> {
  const origin = request.headers.get("origin");

  return origin ? { origin } : {};
}

export function isSameOrigin(request: Request): boolean {
  const origin = request.headers.get("origin");
  const host = request.headers.get("host");

  if (origin === null || host === null) {
    return false;
  }

  try {
    return new URL(origin).host === host;
  } catch {
    // A malformed `Origin` is not this origin.
    return false;
  }
}
