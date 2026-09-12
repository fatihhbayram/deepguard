/**
 * A machine value, or an em dash where the record holds nothing.
 *
 * Three screens had this as a private `Value` — the job queue, the audit log and the case review
 * — which is exactly the count at which the Rule of Three says to extract and not one screen
 * before it. What it encodes is the surface's most load-bearing typographic rule: a string a
 * reader has to be able to compare character by character is set in the figure typeface, and one
 * the record does not hold is drawn as an absence rather than as an empty space that could be a
 * rendering fault.
 *
 * The dash itself is `ABSENT` from `../../analysis` rather than an em dash typed here. It is the
 * same absence the evidence report draws, and two spellings of one character is exactly the kind
 * of drift this extraction exists to end.
 *
 * `break-all` is not set here. The case review wanted it for full UUIDs in a narrow definition
 * list and the two tables did not, so it stays a class the caller adds — a value that breaks
 * mid-token inside a table cell is harder to read than one that is clipped with its full text in
 * a `title`.
 */

import { ABSENT } from "../../analysis";

export function AdminValue({
  children,
  className = "",
}: {
  children: string | null;
  className?: string;
}) {
  return children === null ? (
    <span className="text-muted">{ABSENT}</span>
  ) : (
    <span className={`font-mono text-[11px] text-muted ${className}`}>{children}</span>
  );
}
