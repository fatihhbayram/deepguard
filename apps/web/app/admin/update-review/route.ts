/**
 * The review form's half of the review API: carrying a status and a note to the API's PUT.
 *
 * The same shape `/admin/update-user` has, for the same reasons stated there — the API serves
 * no CORS headers, a browser form can only issue GET or POST, and there is no JavaScript on the
 * page to receive a response body, so the outcome travels back in the query string.
 *
 * **This handler is not behind `admin/layout.tsx`.** A layout wraps the pages beneath it and
 * not the route handlers, so the guard that keeps a non-administrator from *seeing* `/admin`
 * does not keep one from posting here. The API is where this is decided: `require_admin`
 * refuses the forwarded PUT with a 403 and the refusal is relayed below. It is restated because
 * the opposite belief — that being under `/admin` makes a route privileged — is the kind that
 * gets a check left out of the next handler added here.
 *
 * **Nothing in this file decides what a review may say.** Not the two statuses, not the length
 * of a note, not what counts as plain text: all three are the API's, in
 * `app/api/admin_analyses.py`, and a copy of any of them here would be a second rule that could
 * drift from the one actually enforced. The note is forwarded exactly as typed — not trimmed,
 * not escaped, not rewritten — because the API normalizes it once and silently altering
 * somebody's review on the way past would make the stored record disagree with what they wrote.
 *
 * **This route cannot touch a forensic value.** It forwards two fields to an endpoint whose
 * request model forbids extras, so a body carrying `risk_level` is a 422 rather than a change;
 * and the endpoint behind it writes only the review row and its audit event. There is no
 * address in this file that names the analyses API at all.
 */

import { NextResponse } from "next/server";

import { apiUrl } from "../../analysis";
import { logError, logInfo, requestIdHeaders } from "../../observability";
import {
  LOGIN_PATH,
  adminAnalysisPath,
  forwardedOrigin,
  isSameOrigin,
  sessionHeaders,
} from "../../session";
import { reviewUrl } from "../reviews";

// How long the API is given to answer. One upsert of a single row behind a row lock, so it is
// held to the short bound the reads are rather than the generous ones the upload paths need.
const UPDATE_TIMEOUT_MS = 5000;

// Enough of the API's message to be useful, and bounded. It is DeepGuard's own client-facing
// text — "note must be at most 1000 characters" and the like — and it is rendered as text by
// React, never as markup.
const MAX_ERROR_LENGTH = 200;

/**
 * Back to the analysis's review screen, with the outcome of this change attached.
 *
 * A relative `Location`, resolved by the browser against the address it actually asked for.
 * Building an absolute URL out of `request.url` looks more careful and is wrong here: the
 * container listens on 3000 and is published on another port, so the absolute form would send
 * the operator to a port nothing is listening on outside Docker.
 *
 * 303, so the browser follows it with a GET. A 307 would re-post the form, and a reload would
 * then apply the same review a second time — which the API would treat as a no-op, but a
 * reload that silently re-sends a note is not a thing to rely on being harmless.
 */
function back(analysisId: string, params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${adminAnalysisPath(analysisId)}?${query}` },
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
    : `The review was refused (HTTP ${response.status}).`;
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read. The session cookie is `SameSite=Lax` and would not be
  // attached to a cross-site post in the first place; this is the independent check in front of
  // that, made by the server that owns the data.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return new NextResponse(null, { status: 400 });
  }

  const analysisId = (form.get("analysis_id") ?? "").toString().trim();
  const status = form.get("status");
  const note = form.get("note");

  // No analysis means nowhere to redirect back to, so this is a bare 400 rather than the
  // outcome-in-the-query-string answer every other failure below gets. A form that arrives
  // without it did not come from the page.
  if (!analysisId) {
    return new NextResponse(null, { status: 400 });
  }

  const change = {
    // Passed through exactly as the control sent it, without being checked against a list of
    // statuses held here. The API validates it against its own constants — and against the
    // database's check constraint behind them — and answers 422 for anything else; a second
    // list in this process would be one that could disagree, and the thing it would disagree
    // about is whether a word that reads as a verdict may be stored.
    status: typeof status === "string" ? status : "",
    // A textarea always submits, even when empty, so an absent field here means the form was
    // not the page's. Empty is a legitimate value — "reviewed, nothing to add" — and is sent as
    // the empty string rather than omitted, which is the same thing the API's default produces.
    note: typeof note === "string" ? note : "",
  };

  const forwarded = {
    ...(await sessionHeaders()),
    ...forwardedOrigin(request),
    ...(await requestIdHeaders()),
    "content-type": "application/json",
  };

  let response: Response;
  try {
    response = await fetch(`${apiUrl()}${reviewUrl(analysisId)}`, {
      method: "PUT",
      headers: forwarded,
      body: JSON.stringify(change),
      signal: AbortSignal.timeout(UPDATE_TIMEOUT_MS),
    });
  } catch (error) {
    // The underlying message can name internal hosts, so it is not passed on to the browser —
    // but it is exactly what an operator reading the logs needs, and this server's log is not
    // the browser.
    const reason = error instanceof Error ? error.message : String(error);
    await logError("The API could not be reached for a review.", { reason });

    return back(analysisId, { error: "The API could not be reached." });
  }

  // The session expired, was revoked, or was never there. Sending the operator to sign in is
  // the only useful answer; reporting it beside the form would leave them retrying a change
  // that cannot succeed until they do.
  if (response.status === 401) {
    return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
  }

  if (!response.ok) {
    // Includes the 403 a non-administrator gets, the 404 for an analysis that does not exist,
    // and the 422 for a status outside the taxonomy or a note that is too long. All are shown
    // beside the form rather than turned into a redirect elsewhere: the operator is on the
    // right page and the sentence is the whole of what they need.
    return back(analysisId, { error: await failureText(response) });
  }

  // Which analysis, and by nothing more than its id. The note is free prose a person wrote
  // about a case, and a log line is read by more eyes and kept in more places than the screen
  // it came from — the API has already written the authoritative line naming the administrator
  // and the status they stored.
  await logInfo("Forwarded a review to the API.", { analysis_id: analysisId });

  return back(analysisId, { saved: analysisId });
}
