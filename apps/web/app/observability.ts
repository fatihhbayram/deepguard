/**
 * The web application's end of the request id, and the one place this server logs from.
 *
 * A browser request that submits media touches three processes: this server renders and
 * forwards it, the API accepts and queues it, and the worker analyses it minutes later.
 * Until R1-T4 those three wrote logs that had nothing in common, so "what happened to the
 * upload at 14:02" was a question answered by joining timestamps by eye. The id minted here
 * is what joins them instead: it travels to the API in `X-Request-ID`, the API binds it to
 * every line it logs, and it is written onto the queued job so the worker can bind it too.
 *
 * This is the boundary, which is why the id is minted *here* rather than in the API. The
 * browser sends none, and a value the API invented would already be one hop late — the
 * redirect this server writes, and any failure before the call is even made, would sit
 * outside the trace. An incoming `X-Request-ID` is honoured instead of replaced, so a
 * reverse proxy or a caller that already runs a trace keeps its own id.
 *
 * There is no logging library here and no reporting SDK. `console` writes to stdout and
 * stderr, which is where a container's logs come from; formatting them as JSON in production
 * is the whole of what an aggregator needs, and it is four lines rather than a dependency.
 */

import { cache } from "react";
import { headers } from "next/headers";

// The header the id travels in. Must match `REQUEST_ID_HEADER` in `app/observability.py`;
// there is no way to derive one from the other across the two languages, so the pairing is
// stated in both — as the session cookie's name already is.
export const REQUEST_ID_HEADER = "x-request-id";

// What an id arriving from outside may look like. The same narrow rule the API applies, and
// for the same reason: this value is written into log lines, so a newline in it could forge
// a record and an unbounded one could fill a disk. Anything else is replaced by a fresh id
// rather than trimmed — a mangled id would still claim to correlate these lines with
// somebody else's.
const REQUEST_ID_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;

/** The caller's id if it is one this server may repeat, otherwise nothing. */
export function acceptedRequestId(value: string | null): string | null {
  return value !== null && REQUEST_ID_PATTERN.test(value) ? value : null;
}

/**
 * The id for the browser request being served, minted once and reused.
 *
 * `cache` is what makes "once" true. A page render calls this from every fetch it makes —
 * the session, the listing, the health probe — and without memoization each of those would
 * mint an id of its own, leaving one page view scattered across three unrelated traces.
 * React's cache is per-request, so two renders never share one either.
 */
export const requestId = cache(async (): Promise<string> => {
  const incoming = acceptedRequestId((await headers()).get(REQUEST_ID_HEADER));

  return incoming ?? crypto.randomUUID();
});

/** The id restated as the header the API reads it from. */
export async function requestIdHeaders(): Promise<Record<string, string>> {
  return { [REQUEST_ID_HEADER]: await requestId() };
}

// Structured output in production, readable text in development. A development terminal
// gets a sentence; a deployment gets one JSON object per line for whatever ingests its
// stdout.
//
// Decided from `DEEPGUARD_ENV`, the same variable the API reads (`app/observability.py`) and
// the same closed list of development environments, so one setting decides the shape for the
// whole stack rather than each process having an opinion. Deliberately **not** `NODE_ENV`,
// which looks like the obvious choice and is the wrong signal here: this application is
// always a production Next.js build, in local development too, so `NODE_ENV` says
// "production" on a laptop and could never tell the two deployments apart.
//
// Unset means production, matching the API's fail-secure direction: a host nobody configured
// emits the machine-readable shape rather than quietly dropping structure a pipeline expects.
const DEVELOPMENT_ENVIRONMENTS = new Set(["development", "test"]);
const STRUCTURED = !DEVELOPMENT_ENVIRONMENTS.has(
  (process.env.DEEPGUARD_ENV ?? "").trim().toLowerCase(),
);

type LogFields = Record<string, unknown>;

function write(level: "info" | "error", message: string, fields: LogFields): void {
  const line = STRUCTURED
    ? JSON.stringify({
        timestamp: new Date().toISOString(),
        level: level.toUpperCase(),
        logger: "web",
        message,
        ...fields,
      })
    : `${new Date().toISOString()} ${level.toUpperCase()} web ${message} ${JSON.stringify(
        fields,
      )}`;

  // `console.error` for errors so they land on stderr, which is what a host that separates
  // the two streams expects. Everything else goes to stdout.
  if (level === "error") {
    console.error(line);
  } else {
    console.log(line);
  }
}

/**
 * The id of the request being served, or null where there is no request to read one from.
 *
 * `headers()` refuses outside a request scope, and the one caller that logs from outside one
 * is `instrumentation.ts` — which has the failed request's own headers and passes the id in
 * explicitly. Null rather than a fresh id, because an id nothing else ever saw is not a
 * trace: an absent field says "not correlated", while a minted one would say the opposite
 * and be wrong.
 */
async function boundRequestId(): Promise<string | null> {
  try {
    return await requestId();
  } catch {
    return null;
  }
}

/** Record something this server did, under the id of the request that caused it. */
export async function logInfo(message: string, fields: LogFields = {}): Promise<void> {
  write("info", message, { request_id: await boundRequestId(), ...fields });
}

/**
 * Record something that went wrong, under the id of the request that caused it.
 *
 * A `request_id` passed in `fields` wins over the bound one, which is what lets the error
 * hook report the id of the request that actually failed rather than of the context it is
 * called back in.
 */
export async function logError(message: string, fields: LogFields = {}): Promise<void> {
  write("error", message, { request_id: await boundRequestId(), ...fields });
}
