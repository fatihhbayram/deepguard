/**
 * The outcome of the last change, or the reason there is nothing on the screen.
 *
 * The most-copied thing in this surface: seven pages carried a private `Alert`, and the comments
 * at the top of each of them counted the copies as they went — `jobs/page.tsx` marked the third
 * as the point the Rule of Three says to extract and every screen after it carried the number
 * forward rather than resetting it. The count stopped at seven. This is the extraction.
 *
 * The copies had already diverged in a way worth recording: `analyses/[id]/page.tsx` drew its
 * success state in emerald while the other six drew it in accent. Accent wins here, because a
 * successful save is the system speaking — emerald in this palette means a human affirmed
 * something, which is what the review badge uses it for. The review page loses one green box and
 * keeps the one that carries meaning.
 *
 * `role="status"` rather than `role="alert"`, including on the error tone. These are rendered
 * into a page the browser has just navigated to, not injected into one the reader is already
 * looking at; `alert` is assertive and would interrupt a screen reader mid-sentence for a
 * message that is already at the top of the document it is about to read.
 */

const TONES = {
  error: { field: "border-rose-500/40 bg-rose-500/10 text-rose-200", dot: "bg-rose-400" },
  success: { field: "border-accent/40 bg-accent/10 text-bone", dot: "bg-accent" },
} as const;

export function AdminAlert({
  tone,
  children,
}: {
  tone: "error" | "success";
  children: React.ReactNode;
}) {
  const styles = TONES[tone];

  return (
    <p
      role="status"
      className={`flex items-start gap-3 rounded-md border px-4 py-3 text-[13px] leading-relaxed ${styles.field}`}
    >
      <span aria-hidden className={`mt-1.5 size-1.5 shrink-0 rounded-full ${styles.dot}`} />
      <span>{children}</span>
    </p>
  );
}
