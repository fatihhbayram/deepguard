"use client";

/**
 * What a reader sees when a page in this application throws, and what the browser records
 * about it (R1-T4).
 *
 * Next.js renders this in place of the segment that crashed. Before it existed the reader
 * got the framework's own error screen — in development a stack trace, in production a bare
 * "Application error" with no way back — and nothing was written anywhere a person
 * supporting them could look.
 *
 * Two things are deliberately separate here. The **digest** is shown, because it is the one
 * value that ties what is on this screen to the stack trace `instrumentation.ts` wrote on
 * the server: it is an opaque hash, not a message, so showing it discloses nothing. The
 * **error itself** is only logged to the browser console, because in production its message
 * is already redacted by Next.js and in development it is developer detail that has no place
 * in a forensics product's interface.
 *
 * `retry` re-renders the segment. This dashboard's failures are mostly a render that could
 * not reach the API, so trying again is a real remedy rather than a decorative button.
 */

import { useEffect } from "react";

export default function ErrorPage({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    // The client half of the same event `onRequestError` records on the server. There is no
    // reporting endpoint to send it to and R1-T4 adds none: the console is where a browser's
    // errors belong, and the digest above is what connects this to the server's own log.
    console.error("Unhandled client error.", { digest: error.digest, error });
  }, [error]);

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-4 p-8">
      <h1 className="text-xl font-semibold">Something went wrong</h1>
      <p className="text-sm opacity-80">
        This page could not be rendered. Nothing about any analysis has been changed.
      </p>
      {error.digest && (
        <p className="font-mono text-xs opacity-60">Reference: {error.digest}</p>
      )}
      <div>
        <button
          type="button"
          onClick={() => retry()}
          className="rounded border px-3 py-1.5 text-sm"
        >
          Try again
        </button>
      </div>
    </main>
  );
}
