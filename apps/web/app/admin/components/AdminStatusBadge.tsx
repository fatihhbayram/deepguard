/**
 * A state, as a chip.
 *
 * Six screens grew one of these each — the account's `Activity`, the key's `Status`, the job's
 * `Status` and `Stale`, the review's `StatusBadge`, the audit log's action chip — and between
 * them they used four different paddings and three different radii for the same object. The
 * tones are named here so that a state's colour is a decision this file makes once rather than a
 * Tailwind string typed at a call site.
 *
 * **The tone is not the word.** This component colours what it is given and never translates it:
 * `FAILED` stays `FAILED`, `REVOKED` stays `REVOKED`, `UNREVIEWED` stays `UNREVIEWED`. A status
 * the application does not recognise is passed through with the `muted` tone, which is the
 * honest treatment of a word nobody here understands — the alternative is a blank chip that
 * hides exactly the case a reader needs to see.
 *
 * The five tones, and there is no sixth:
 *
 * - `neutral`  — a normal, working state. Active, completed.
 * - `muted`    — a state with no opinion in it. Queued, processing, unreviewed.
 * - `warning`  — attention, not failure. A stale lease, a case needing follow-up.
 * - `negative` — ended or failed. Deactivated, revoked, failed.
 * - `positive` — an affirmative human act. A review somebody signed.
 *
 * `dot` draws the 6px marker the account and key chips carry. It is decoration over a word that
 * is already there, so it is `aria-hidden`, and it is off by default: the job and review chips
 * never had one, and adding it everywhere would be this component changing six screens rather
 * than unifying them.
 */

const TONES = {
  neutral: { field: "bg-chip text-bone", dot: "bg-accent" },
  muted: { field: "bg-chip text-muted", dot: "bg-muted" },
  warning: { field: "bg-amber-500/10 text-amber-200", dot: "bg-amber-400" },
  negative: { field: "bg-rose-500/10 text-rose-200", dot: "bg-rose-400" },
  positive: { field: "bg-emerald-500/10 text-emerald-200", dot: "bg-emerald-400" },
} as const;

export type AdminStatusTone = keyof typeof TONES;

export function AdminStatusBadge({
  tone,
  dot = false,
  mono = false,
  title,
  children,
}: {
  tone: AdminStatusTone;
  dot?: boolean;
  mono?: boolean;
  title?: string;
  children: React.ReactNode;
}) {
  const styles = TONES[tone];

  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-sm px-2 py-1 text-[11px] font-medium tracking-[0.08em] uppercase ${
        mono ? "font-mono" : ""
      } ${styles.field}`}
    >
      {dot && <span aria-hidden className={`size-1.5 shrink-0 rounded-full ${styles.dot}`} />}
      {children}
    </span>
  );
}
