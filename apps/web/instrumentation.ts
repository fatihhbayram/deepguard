/**
 * Where a server-side crash in this application becomes a log line (R1-T4).
 *
 * Next.js catches an uncaught exception thrown while rendering a page or running a route
 * handler, shows the reader the error boundary in `app/error.tsx`, and calls this hook with
 * the error it caught. Without the hook the exception is still printed by the framework, but
 * as the framework's own text: no request id, no JSON, nothing a log pipeline can group with
 * the API and worker lines belonging to the same request.
 *
 * `onRequestError` is the whole of the file. There is no `register`, because registering an
 * observability agent is exactly what R1-T4 excludes — no Datadog, no Sentry, no OTel
 * exporter. What the host ingests is stdout and stderr, and this writes to them.
 *
 * The error itself may not be the object that was thrown: React re-creates errors it catches
 * during a Server Component render. Its `digest` is the stable identifier for the real one
 * and is what the reader is shown in the boundary, so it is recorded here beside the message
 * — that pairing is what turns "something went wrong (digest 1a2b3c)" on somebody's screen
 * into the stack trace in this log.
 */

import type { Instrumentation } from "next";

import { REQUEST_ID_HEADER, acceptedRequestId, logError } from "./app/observability";

/**
 * The request id as the failed request carried it, or null if it carried none.
 *
 * The header bag is Node's, where every lookup is optionally `undefined` — a header that
 * was never sent and one whose value is empty are the same absence here, and both end as
 * null.
 */
function requestIdOf(headers: Record<string, string | string[] | undefined>): string | null {
  const value = headers[REQUEST_ID_HEADER];

  // Read out of the request that failed rather than out of the current context: this hook
  // is called back outside the render, where `headers()` is not available. A header that
  // arrived more than once is not one this server sent, so it is treated as absent.
  return typeof value === "string" ? acceptedRequestId(value) : null;
}

export const onRequestError: Instrumentation.onRequestError = async (
  error,
  request,
  context,
) => {
  const forwarded = requestIdOf(request.headers);

  await logError("Unhandled server error.", {
    // Only when the failed request actually carried one. Passing an explicit null would
    // override the id `logError` can still find for itself when this hook happens to run
    // inside the request's own context, and would report "not correlated" for a line that is.
    ...(forwarded === null ? {} : { request_id: forwarded }),
    path: request.path,
    method: request.method,
    route_path: context.routePath,
    route_type: context.routeType,
    message: error instanceof Error ? error.message : String(error),
    digest:
      typeof error === "object" && error !== null && "digest" in error
        ? String(error.digest)
        : null,
    // The stack, when there is one. It names this application's own files and nothing a
    // browser is shown — the reader gets the boundary below, and this is the half an
    // operator needs to act on it.
    stack: error instanceof Error ? error.stack : null,
  });
};
