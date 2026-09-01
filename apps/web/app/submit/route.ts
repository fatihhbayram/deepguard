/**
 * The dashboard's submission endpoint: takes the form, hands it to the API, comes back.
 *
 * The dashboard submits through this handler rather than from the browser to the API
 * directly, for a plain reason: the API serves no CORS headers, because nothing outside the
 * server has ever needed to call it. A browser POST to another origin would be blocked, and
 * opening the API up to one would be a security decision made to save a hop. This runs on
 * the server, alongside the reads the page already does, and talks to the API on the Docker
 * network exactly as they do.
 *
 * The page it serves is a plain HTML form, so no JavaScript is involved on the client at
 * all: the browser posts here, this posts to the API, and the answer is a redirect back to
 * the dashboard carrying what happened. That keeps the submission working the same way the
 * rest of this server-rendered dashboard does, and it is why the outcome travels in the
 * query string rather than in a response body nobody would render.
 *
 * Since R1-T2 the submission is authenticated. The browser's session cookie is forwarded to
 * the API, which resolves the account from it and records that account as the analysis's
 * owner. Nothing about the owner travels in the form: this handler could not name one if it
 * wanted to, which is what stops a signed-in person from filing an analysis under somebody
 * else's account by editing a hidden field.
 *
 * One deliberate limitation: `request.formData()` reads the whole upload into memory before
 * forwarding it. The API caps a file at 100 MiB, so that is the ceiling here too. Streaming
 * it through would be better and is not what this task is: the dashboard is an internal
 * tool with one operator, not the ingestion path a customer uses — that is the public API,
 * which the browser never goes through.
 */

import { NextResponse } from "next/server";

import { apiUrl } from "../analysis";
import { logError, logInfo, requestIdHeaders } from "../observability";
import { LOGIN_PATH, forwardedOrigin, isSameOrigin, sessionHeaders } from "../session";

// How long the API is given to answer. A URL submission waits for the download, which is a
// real network fetch of up to 100 MiB, so this is generous where the dashboard's read
// timeouts are not — cutting it off early would abandon a download the API is still doing
// and leave the operator with no idea whether the analysis was queued.
const SUBMIT_TIMEOUT_MS = 300_000;

// Enough of the API's message to be useful, and bounded. It is DeepGuard's own client-facing
// text — the API keeps extractor, socket and storage detail in its logs — and it is rendered
// as text by React, never as markup.
const MAX_ERROR_LENGTH = 200;

/**
 * The dashboard, with the outcome of this submission attached.
 *
 * A relative `Location`, which the browser resolves against the address it actually asked
 * for. Building an absolute URL out of `request.url` looks more careful and is wrong here:
 * the container listens on 3000 and is published on another port, so the absolute form
 * sends the operator to a port nothing is listening on outside Docker.
 *
 * 303, so the browser follows it with a GET. A 307 would re-post the form, and a reload
 * would then submit the same media a second time.
 */
function back(params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, { status: 303, headers: { Location: `/?${query}` } });
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
    : `The API refused the submission (HTTP ${response.status}).`;
}

export async function POST(request: Request): Promise<NextResponse> {
  // A submission driven from another site is refused before the upload is read, so a forged
  // form cannot make this server spend a 100 MiB read on it. The `SameSite=Lax` cookie means
  // such a request would arrive without a session and be refused by the API anyway; this is
  // the independent check in front of that.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return back({ error: "The submission could not be read." });
  }

  const url = (form.get("url") ?? "").toString().trim();
  const file = form.get("file");
  // An empty file input still arrives as a part, with no name and no bytes.
  const hasFile = file instanceof File && file.size > 0;

  // The URL wins when both are filled in, and the form says so. Guessing between them, or
  // submitting both as two analyses, would be the page deciding something the operator did
  // not ask for.
  if (!url && !hasFile) {
    return back({ error: "Choose a file or paste a URL first." });
  }

  const target = url ? `${apiUrl()}/api/v1/analyses/url` : `${apiUrl()}/api/v1/analyses`;
  const body = url ? JSON.stringify({ url }) : new FormData();
  if (!url && body instanceof FormData) {
    body.append("file", file as File, (file as File).name);
  }

  // The session cookie and the browser's origin, restated for the server-to-server call.
  // The cookie is what the API resolves the owner of this analysis from, so a submission
  // without it is refused there rather than quietly committed to nobody; the origin is what
  // the API's own CSRF check reads, and it has already been accepted against this server's
  // host a few lines above.
  //
  // The request id travels with them (R1-T4), and this is the call it matters most on: the
  // API writes it onto the queued job, so the analysis the worker runs minutes from now is
  // still findable from the line logged just below.
  const forwarded = {
    ...(await sessionHeaders()),
    ...forwardedOrigin(request),
    ...(await requestIdHeaders()),
  };

  await logInfo("Forwarding a dashboard submission to the API.", {
    submission: url ? "url" : "file",
  });

  let response: Response;
  try {
    response = await fetch(target, {
      method: "POST",
      headers: url
        ? { ...forwarded, "content-type": "application/json" }
        : forwarded,
      body,
      signal: AbortSignal.timeout(SUBMIT_TIMEOUT_MS),
    });
  } catch (error) {
    // A timeout or a connection failure. The underlying message can name internal hosts,
    // so it is not passed on to the browser — but it is exactly what an operator reading
    // the logs needs, and this server's log is not the browser.
    await logError("The API could not be reached for a dashboard submission.", {
      reason: error instanceof Error ? error.message : String(error),
    });

    return back({ error: "The API could not be reached." });
  }

  // The session expired, was revoked, or was never there. Sending the operator to sign in is
  // the only useful answer; reporting it beside the form as a refused submission would leave
  // them retrying an upload that cannot succeed until they do.
  if (response.status === 401) {
    return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
  }

  if (!response.ok) {
    return back({ error: await failureText(response) });
  }

  const payload = await response.json().catch(() => null);
  const id =
    typeof payload === "object" && payload !== null
      ? (payload as Record<string, unknown>).id
      : null;

  // Accepted either way; the id is what the operator can follow. A 202 whose body could not
  // be read is still a queued analysis, and saying otherwise would be worse than saying less.
  //
  // Logged with both identities, which is what makes the trace usable from either end: the
  // request id joins this line to the API's and to the worker's, and the analysis id is what
  // the person on the dashboard is looking at.
  await logInfo("The API queued a dashboard submission.", {
    analysis_id: typeof id === "string" ? id : null,
  });

  return typeof id === "string" ? back({ submitted: id }) : back({ submitted: "" });
}
