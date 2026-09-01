/**
 * Where the request id becomes a property of the request itself (R1-FINAL-QA).
 *
 * Every process in this stack agrees about one id per browser request, and this is the point
 * at which that id starts existing. It runs before the page render or the route handler that
 * serves the request, stamps `x-request-id` onto the incoming headers, and lets the request
 * through; from there `requestId()` in `app/observability.ts` is a plain read of a header
 * that is always present, and every reader of it — three fetches in one render, or the
 * forwarded header and two log lines in `POST /submit` — necessarily gets the same answer.
 *
 * The first attempt at this memoized the minted id inside `observability.ts` instead, first
 * with React's `cache` and then keyed on the object `headers()` returns. Both are the same
 * mistake in two shapes: they make correlation depend on how the framework happens to scope
 * a value, which `cache` does not do inside a Route Handler at all, and which nothing
 * guarantees for the header object's identity either. The id is not a memoization concern.
 * It belongs to the request, so it is written onto the request.
 *
 * An id the caller already sent is kept rather than replaced, which is what lets a reverse
 * proxy or an upstream service that already runs a trace keep its own. Anything that fails
 * `acceptedRequestId` is replaced outright rather than trimmed, for the reason that function
 * gives: a mangled id would still claim to correlate these lines with somebody else's.
 */

import { NextResponse, type NextRequest } from "next/server";

import { REQUEST_ID_HEADER, acceptedRequestId } from "./app/request-id";

export function middleware(request: NextRequest): NextResponse {
  const incoming = acceptedRequestId(request.headers.get(REQUEST_ID_HEADER));

  // A copy, because the incoming header bag is read-only. `NextResponse.next({ request })`
  // is what makes the copy the one the render or the route handler goes on to read.
  const headers = new Headers(request.headers);
  headers.set(REQUEST_ID_HEADER, incoming ?? crypto.randomUUID());

  return NextResponse.next({ request: { headers } });
}

// Everything this application serves, and nothing it does not. The three exclusions are
// static files and the icon: they are served without running any application code, so there
// is no log line for an id to appear on and no fetch for it to travel with.
export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
