/**
 * Issue an API key, and hand the secret to the page that asked for it — once.
 *
 * This forwards to the API for the same reason `/submit`, `/session` and `/admin/update-user`
 * do: the API serves no CORS headers, and opening it to a browser origin would be a security
 * decision made to save a hop.
 *
 * **It answers with JSON rather than a redirect, and that is the whole reason it is not shaped
 * like `/admin/update-user`.** Every other administrative mutation is a plain HTML form that
 * answers `303` with the outcome in the query string, which needs no JavaScript on the client
 * at all — the pattern this codebase prefers and uses everywhere else. It cannot be used here.
 * The thing being carried back is a credential, and a credential in a query string is written
 * into browser history, into this server's access log, into the `Referer` header of the next
 * request the page makes, and into any bookmark or shared link. There is no way to put a
 * secret in a URL safely, so it travels in a response body and the page that receives it holds
 * it in memory.
 *
 * That is what buys the one client component on this surface. The trade is stated here rather
 * than left implicit, because "why does this one screen need JavaScript" is exactly the
 * question the next person to read it will have.
 *
 * **The secret passes through this handler and is kept nowhere.** It is not logged — the log
 * line below names the key's id and its label, never the object the plaintext is on — it is
 * not put in a cookie, and the response is marked `no-store` so no cache holds it. It exists
 * in this process for the length of one request.
 *
 * **This handler is not behind `admin/layout.tsx`.** A layout wraps the pages beneath it and
 * not the route handlers, so the guard that keeps a non-administrator from *seeing*
 * `/admin/api-keys` does not keep one from posting here. That is not a hole being tolerated:
 * the API is where this is decided, `require_admin` refuses the forwarded POST with a 403, and
 * the refusal is relayed below. It is stated because the opposite belief — that being under
 * `/admin` makes a route privileged — is the kind that gets a check left out of the next
 * handler added here, and the next handler here would be one that mints credentials.
 */

import { NextResponse } from "next/server";

import { logError, logInfo } from "../../observability";
import { LOGIN_PATH, forwardedOrigin, isSameOrigin } from "../../session";
import { createApiKey } from "../api-keys";

/** The longest label this handler will forward. The API is the authority; see below. */
const MAX_NAME_LENGTH = 255;

/**
 * A JSON answer that nothing may cache or store.
 *
 * `no-store` on every response from this route, not only the successful one. The successful
 * body carries a credential and must not sit in a disk cache; the failures carry nothing
 * sensitive but would be equally wrong to serve from one, since this is a mutation whose
 * answer is only ever about the request that just happened.
 */
function json(body: unknown, status: number): NextResponse {
  return NextResponse.json(body, {
    status,
    headers: { "cache-control": "no-store" },
  });
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read. The session cookie is `SameSite=Lax` and would not be
  // attached to a cross-site post in the first place; this is the independent check in front of
  // that, and what a forged request would be driving here is the minting of a credential for
  // the external API.
  if (!isSameOrigin(request)) {
    return json({ error: "This request did not come from the dashboard." }, 403);
  }

  const payload = await request.json().catch(() => null);

  if (typeof payload !== "object" || payload === null) {
    return json({ error: "The request could not be read." }, 400);
  }

  const { name } = payload as Record<string, unknown>;

  // Two checks, and deliberately only two. A blank name and an over-long one are refused here
  // because they are refusals this server can make without asking — and because the second one
  // keeps a 4KB label out of a request that would only be rejected at the other end anyway.
  // What is *not* duplicated here is any rule about what a name may contain: the API validates
  // that against its own column, and a second rule in this process would be one that could
  // disagree with the one actually enforced.
  if (typeof name !== "string" || name.trim().length === 0) {
    return json({ error: "Give the key a name." }, 400);
  }

  if (name.length > MAX_NAME_LENGTH) {
    return json({ error: `A name may be at most ${MAX_NAME_LENGTH} characters.` }, 400);
  }

  const result = await createApiKey(name, forwardedOrigin(request));

  if (!result.ok) {
    if (result.unauthenticated) {
      // The session expired, was revoked, or was never there. Reported as a status the page can
      // act on rather than as a redirect: this is a `fetch`, and a 303 would be followed
      // transparently, handing the component the sign-in page's HTML as though it were the
      // answer to its request.
      return json({ error: result.error, signIn: LOGIN_PATH }, 401);
    }

    await logError("An API key could not be issued.", { reason: result.error });

    return json({ error: result.error }, 502);
  }

  // The id and the label, and nothing else. `result.key` carries the plaintext and is never
  // handed to the logger — this is the line where getting that wrong would be easiest and
  // least visible, which is why the fields are named individually rather than spread.
  await logInfo("Issued an API key.", {
    api_key_id: result.key.id,
    name: result.key.name,
  });

  // The one response in this application that carries a secret. It goes to the component that
  // asked for it, which shows it once and holds it in memory until the reader leaves the page.
  return json(result.key, 201);
}
