"use client";

/**
 * The boundary for a failure in the root layout itself, where `app/error.tsx` cannot reach.
 *
 * `error.tsx` renders *inside* the root layout, so an exception thrown by that layout leaves
 * nothing to render it. This replaces the whole document instead — which is why it carries
 * its own `<html>` and `<body>`, and why the fonts and theme class the layout normally
 * applies are absent from it.
 *
 * Rare enough that it is deliberately plain: no dashboard chrome, no navigation, nothing
 * that could throw a second time on the way to reporting the first. The digest is shown for
 * the same reason it is in `error.tsx` — it is what ties this screen to the server's log.
 *
 * The words follow the reader's stored language (R16-T2). Reading it cannot throw: the store
 * catches every failure of `localStorage` and answers English, so the language is not a second
 * way for this screen to fail.
 */

import { useEffect } from "react";

import { useLocale } from "./i18n/client";
import { translate } from "./i18n/core";

export default function GlobalError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  const [locale] = useLocale();

  useEffect(() => {
    console.error("Unhandled root layout error.", { digest: error.digest, error });
  }, [error]);

  return (
    <html lang={locale}>
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, padding: "2rem" }}>
        <h1 style={{ fontSize: "1.25rem" }}>{translate(locale, "error.title")}</h1>
        <p style={{ fontSize: "0.875rem" }}>{translate(locale, "error.application")}</p>
        {error.digest && (
          <p style={{ fontSize: "0.75rem", opacity: 0.6 }}>
            {translate(locale, "error.reference")} {error.digest}
          </p>
        )}
        <button type="button" onClick={() => retry()}>
          {translate(locale, "error.retry")}
        </button>
      </body>
    </html>
  );
}
