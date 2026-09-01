/**
 * The vocabulary of the request id, and nothing else (R1-FINAL-QA).
 *
 * These two live apart from `observability.ts` for one reason: `middleware.ts` needs them,
 * and `observability.ts` imports `next/headers`, which belongs to the request-scoped server
 * runtime the middleware runs *before*. Restating the header name and the pattern in the
 * middleware instead would leave two definitions of what a valid id is, in the same
 * language, free to drift — the thing the comment in `observability.ts` about stating the
 * pairing in both languages is careful to avoid within one.
 *
 * Both are re-exported by `observability.ts`, so nothing that already imported them from
 * there had to change.
 */

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
