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
 * One deliberate limitation: `request.formData()` reads the whole upload into memory before
 * forwarding it. The API caps a file at 100 MiB, so that is the ceiling here too. Streaming
 * it through would be better and is not what this task is: the dashboard is an internal
 * tool with one operator, not the ingestion path a customer uses — that is the public API,
 * which the browser never goes through.
 */

import { NextResponse } from "next/server";

import { API_URL } from "../analysis";

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

  const target = url ? `${API_URL}/api/v1/analyses/url` : `${API_URL}/api/v1/analyses`;
  const body = url ? JSON.stringify({ url }) : new FormData();
  if (!url && body instanceof FormData) {
    body.append("file", file as File, (file as File).name);
  }

  let response: Response;
  try {
    response = await fetch(target, {
      method: "POST",
      headers: url ? { "content-type": "application/json" } : undefined,
      body,
      signal: AbortSignal.timeout(SUBMIT_TIMEOUT_MS),
    });
  } catch {
    // A timeout or a connection failure. The underlying message can name internal hosts,
    // so it is not passed on.
    return back({ error: "The API could not be reached." });
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
  return typeof id === "string" ? back({ submitted: id }) : back({ submitted: "" });
}
