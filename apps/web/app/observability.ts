/**
 * The web application's end of the request id, and the one place this server logs from.
 *
 * A browser request that submits media touches three processes: this server renders and
 * forwards it, the API accepts and queues it, and the worker analyses it minutes later.
 * Until R1-T4 those three wrote logs that had nothing in common, so "what happened to the
 * upload at 14:02" was a question answered by joining timestamps by eye. The id this server
 * stamps on the request is what joins them instead: it travels to the API in `X-Request-ID`,
 * the API binds it to every line it logs, and it is written onto the queued job so the
 * worker can bind it too.
 *
 * This server is the boundary, which is why the id starts *here* rather than in the API. The
 * browser sends none, and a value the API invented would already be one hop late — the
 * redirect this server writes, and any failure before the call is even made, would sit
 * outside the trace. It is `middleware.ts` that puts it on the request; what this module
 * does is read it back and write it into log lines.
 *
 * There is no logging library here and no reporting SDK. `console` writes to stdout and
 * stderr, which is where a container's logs come from; formatting them as JSON in production
 * is the whole of what an aggregator needs, and it is four lines rather than a dependency.
 */

import { headers } from "next/headers";

// The header name and the rule for what may travel in it, defined in `app/request-id.ts`
// because `middleware.ts` needs them too and cannot import this module. Re-exported so the
// callers that have always taken them from here — `instrumentation.ts` among them — go on
// doing so.
import { REQUEST_ID_HEADER, acceptedRequestId } from "./request-id";

export { REQUEST_ID_HEADER, acceptedRequestId };

/**
 * The id for the browser request being served.
 *
 * A read, not a decision. `middleware.ts` has already put the id on this request's headers
 * before anything here runs, so every call within one request — three fetches in a render,
 * or the forwarded header and both log lines in `POST /submit` — reads the one value that
 * was written there, with nothing to memoize and no way for two readers to disagree.
 *
 * The mint below is unreachable while the middleware's matcher covers every path that runs
 * application code, and it stays here rather than throwing because this is the code path
 * that logs: a fresh id makes one uncorrelated line, and an exception raised inside a logger
 * would lose the line the caller was trying to write.
 */
export async function requestId(): Promise<string> {
  return (await headers()).get(REQUEST_ID_HEADER) ?? crypto.randomUUID();
}

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
