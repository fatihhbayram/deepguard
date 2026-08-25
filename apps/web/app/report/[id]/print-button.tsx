"use client";

/**
 * Opens the browser's own print dialog.
 *
 * The only client-side code on the report, kept in its own module so the report route stays a
 * Server Component. It is a convenience and never a dependency: the page is plain HTML and
 * prints correctly from the browser's own menu with JavaScript disabled, so nothing about the
 * document requires this button to work.
 */
export function PrintButton() {
  return (
    <button
      type="button"
      onClick={() => window.print()}
      className="rounded border border-black/20 px-2 py-1 text-sm dark:border-white/25"
    >
      Print / Save as PDF
    </button>
  );
}
