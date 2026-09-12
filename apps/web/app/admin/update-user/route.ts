/**
 * The administrative surface's one mutation: change an account's role or its activation.
 *
 * A plain HTML form posts here and this forwards a PATCH to the API, for the same reason
 * `/submit` and `/session` exist rather than the browser calling the API directly — the API
 * serves no CORS headers, and opening it to a browser origin would be a security decision
 * made to save a hop. No JavaScript is involved on the client at all, which is why the
 * outcome travels back in the query string: there is nothing on the page to receive a
 * response body.
 *
 * A POST, from a form, carrying the method the API actually wants in the body of the request
 * rather than on it. A browser form can only issue GET or POST, and the alternative to
 * translating here would be a client component whose only job is to call `fetch` with a verb
 * — a script dependency bought for the sake of a header, on the one page where the thing
 * being changed is who holds administrative access.
 *
 * **This handler is not behind `admin/layout.tsx`.** A layout wraps the pages beneath it and
 * not the route handlers, so the guard that keeps a non-administrator from *seeing* `/admin`
 * does not keep one from posting here. That is not a hole being tolerated: the API is where
 * this is decided, `require_admin` refuses the forwarded PATCH with a 403, and the refusal is
 * relayed below. It is stated because the opposite belief — that being under `/admin` makes a
 * route privileged — is the kind that gets a check left out of the next handler added here.
 *
 * Nothing in this file decides whether a change is allowed. Not the self-modification rule,
 * not the last-administrator invariant, not the set of roles: all three are the API's, in
 * `app/api/admin_users.py`, and a copy of any of them here would be a second rule that could
 * drift from the one actually enforced. What this does is carry the form to the API and the
 * API's answer back to the page.
 */

import { NextResponse } from "next/server";

import { apiUrl } from "../../analysis";
import { logError, logInfo, requestIdHeaders } from "../../observability";
import {
  ADMIN_PATH,
  LOGIN_PATH,
  forwardedOrigin,
  isSameOrigin,
  sessionHeaders,
} from "../../session";
import { ADMIN_USERS_URL } from "../users";

// How long the API is given to answer. A single indexed update behind a row lock, so it is
// held to the short bound the reads are rather than the generous ones the upload paths need.
const UPDATE_TIMEOUT_MS = 5000;

// Enough of the API's message to be useful, and bounded. It is DeepGuard's own client-facing
// text — "An administrator cannot change their own role or activation." and the like — and it
// is rendered as text by React, never as markup.
const MAX_ERROR_LENGTH = 200;

/**
 * Back to `/admin`, with the outcome of this change attached.
 *
 * A relative `Location`, resolved by the browser against the address it actually asked for.
 * Building an absolute URL out of `request.url` looks more careful and is wrong here: the
 * container listens on 3000 and is published on another port, so the absolute form would send
 * the operator to a port nothing is listening on outside Docker.
 *
 * 303, so the browser follows it with a GET. A 307 would re-post the form, and a reload would
 * then apply the same change a second time.
 */
function back(params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${ADMIN_PATH}?${query}` },
  });
}

/** What the API said went wrong, or a generic statement when it said nothing usable. */
async function failureText(response: Response): Promise<string> {
  const payload = await response.json().catch(() => null);
  const detail =
    typeof payload === "object" && payload !== null
      ? (payload as Record<string, unknown>).detail
      : null;

  return typeof detail === "string" && detail.length > 0
    ? detail.slice(0, MAX_ERROR_LENGTH)
    : `The change was refused (HTTP ${response.status}).`;
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read. The session cookie is `SameSite=Lax` and would not be
  // attached to a cross-site post in the first place; this is the independent check in front
  // of that, and it matters more here than on a submission because what a forged form would
  // be driving is the granting of administrative access.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return back({ error: "The change could not be read." });
  }

  const userId = (form.get("user_id") ?? "").toString().trim();
  const role = form.get("role");
  const isActive = form.get("is_active");

  if (!userId) {
    return back({ error: "The change named no account." });
  }

  // Only the fields the form actually carried. The two controls are separate submissions —
  // the role control sends a role, the activation control sends an activation — and sending
  // the absent one as `null` would turn each into a change to both, restoring whatever the
  // page happened to have rendered last over a value somebody else may have just altered.
  const change: Record<string, unknown> = {};
  if (typeof role === "string") {
    // Passed through exactly as the control sent it, without being checked against a list of
    // roles held here. The API validates it against its own constants and answers 422 for
    // anything else; a second list in this process would be one that could disagree.
    change.role = role;
  }
  if (typeof isActive === "string") {
    // A checkbox would send nothing when off, so the control is a button carrying the value
    // it means. `"true"` is the only spelling that activates, and everything else deactivates
    // — the direction that fails towards less access when a form arrives malformed.
    change.is_active = isActive === "true";
  }

  const forwarded = {
    ...(await sessionHeaders()),
    ...forwardedOrigin(request),
    ...(await requestIdHeaders()),
    "content-type": "application/json",
  };

  let response: Response;
  try {
    response = await fetch(`${apiUrl()}${ADMIN_USERS_URL}/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: forwarded,
      body: JSON.stringify(change),
      signal: AbortSignal.timeout(UPDATE_TIMEOUT_MS),
    });
  } catch (error) {
    // The underlying message can name internal hosts, so it is not passed on to the browser —
    // but it is exactly what an operator reading the logs needs, and this server's log is not
    // the browser.
    const reason = error instanceof Error ? error.message : String(error);
    await logError("The API could not be reached for an account change.", { reason });

    return back({ error: "The API could not be reached." });
  }

  // The session expired, was revoked, or was never there. Sending the operator to sign in is
  // the only useful answer; reporting it beside the table would leave them retrying a change
  // that cannot succeed until they do.
  if (response.status === 401) {
    return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
  }

  if (!response.ok) {
    // Includes the 403 a non-administrator gets and the 400 a change that would break a system
    // invariant gets. Both are shown beside the table rather than turned into a redirect: the
    // operator is on the right page and the sentence is the whole of what they need.
    return back({ error: await failureText(response) });
  }

  // Which account, and by nothing more than its id. The email is personal data and does not
  // need to be in a log line to make it actionable, and the API has already written the
  // authoritative line naming both the administrator and the values it stored.
  await logInfo("Forwarded an account change to the API.", { account_id: userId });

  return back({ updated: userId });
}
