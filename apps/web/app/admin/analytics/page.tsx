/**
 * The deployment's last seven days, as an operator reads it (R8-T4).
 *
 * Counts and nothing else. Where `/admin/jobs` answers "what is this piece of work doing", this
 * answers "what has this deployment been doing lately" — and it answers it without showing a
 * single record, which is what lets it be a whole-deployment view at all.
 *
 * **Every number on this page arrives computed.** PostgreSQL grouped the rows, the API named
 * the buckets, and this file draws them. There is no arithmetic here beyond the width of a bar:
 * no percentage, no failure rate, no total summed from a map, no provider ranked against
 * another. The reasoning is at the top of `../analytics.ts` and it is the load-bearing
 * constraint on this screen — an invented ratio on a page about detector health is
 * indistinguishable, to the person reading it, from a measured one.
 *
 * **No charting library.** The bars below are a div inside a div with a percentage width, which
 * is the whole of what these distributions need; each is labelled with its own count in text,
 * so the length is decoration over a number the reader can already see. A dependency that draws
 * an axis would be a dependency added for four bar charts.
 *
 * **Absent is not zero, in the detector table.** A provider is listed because it persisted at
 * least one signal this week. One that was never invoked is not listed at zero, because a row
 * of zeros in a health table reads as "asked and never answered" — the opposite of what it
 * would mean. The page says so in prose rather than leaving the reader to infer it, since an
 * absence is the one thing a table cannot show.
 *
 * Read-only in the strict sense: no form, no route handler behind it, nothing that mutates.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import {
  ADMIN_ANALYTICS_PATH,
  ADMIN_JOBS_PATH,
  ADMIN_PATH,
  LOGIN_PATH,
  WORKSPACE_PATH,
} from "../../session";
import { AdminAnalytics, fetchAnalytics } from "../analytics";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * `Legend`, `Heading` and `Alert` restated a fourth time. `app/admin/jobs/page.tsx` notes that
 * the third copy is where the Rule of Three says to extract, and that extracting means editing
 * pages the task at hand is not about — which is as true here as it was there. The count is
 * four now. Carried forward deliberately so it stays visible rather than being quietly reset.
 */

/** The small accented label above a section heading. */
function Legend({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] font-medium tracking-[0.16em] text-accent uppercase">
      {children}
    </p>
  );
}

/** A section heading. Tight, semibold, at the scale the rest of the application sets it. */
function Heading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mt-2.5 text-2xl font-semibold tracking-[-0.02em] text-bone sm:text-[28px]">
      {children}
    </h2>
  );
}

/** Why the summary is not on the screen, stated in the page's own voice. */
function Alert({ children }: { children: React.ReactNode }) {
  return (
    <p
      role="status"
      className="flex items-start gap-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-[13px] leading-relaxed text-rose-200"
    >
      <span aria-hidden className="mt-1.5 size-1.5 shrink-0 rounded-full bg-rose-400" />
      <span>{children}</span>
    </p>
  );
}

/* ------------------------------------------------------------------ *
 * Distributions
 * ------------------------------------------------------------------ */

/**
 * One distribution, as labelled rows with a bar behind each count.
 *
 * The bar's width is the only number this file computes, and it is a length rather than a
 * reading: a count over the largest count in the same distribution, scoped to this card, never
 * rendered as text. Against the largest rather than the total on purpose — a share of the total
 * is a percentage, and a percentage is a claim about the population that the API has not made.
 * This is a comparison between the bars next to each other and nothing more.
 *
 * The count itself is always written out, so a distribution where one category dwarfs the rest
 * stays readable when every other bar has collapsed to a sliver.
 *
 * Keys are drawn in the order the API sent them, which is the order `admin_analytics.py` seeds
 * its categories in — the schema's own vocabulary first, and anything it did not recognise
 * after. Re-sorting here by size would move a row between reads and make the card hard to scan;
 * re-sorting alphabetically would bury an unrecognised status in the middle of the list.
 */
function Distribution({
  title,
  note,
  counts,
}: {
  title: string;
  note: string;
  counts: Record<string, number>;
}) {
  const entries = Object.entries(counts);
  // The divisor for the bars. Never zero: a week in which every category is zero would make
  // every bar 0/0, so the floor of 1 keeps them all at zero width instead of NaN.
  const largest = Math.max(1, ...entries.map(([, value]) => value));

  return (
    <section className="rounded-lg border border-line bg-ink-2 p-5">
      <h3 className="text-[13px] font-medium tracking-[0.08em] text-bone uppercase">{title}</h3>
      <p className="mt-1.5 text-[12px] leading-relaxed text-muted">{note}</p>

      <dl className="mt-4 space-y-2.5">
        {entries.map(([category, value]) => (
          <div key={category}>
            <div className="flex items-baseline justify-between gap-4">
              <dt className="truncate font-mono text-[11px] tracking-[0.06em] text-muted uppercase">
                {category}
              </dt>
              <dd className="font-mono text-[13px] text-bone tabular-nums">{value}</dd>
            </div>
            {/* Decoration over the number above, and marked as such: the reading is the
                figure in the `dd`, and a screen reader that announced the track as well would
                be announcing the same count twice. */}
            <div aria-hidden className="mt-1 h-1 overflow-hidden rounded-full bg-chip">
              <div
                className="h-full rounded-full bg-accent/70"
                style={{ width: `${(value / largest) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </dl>
    </section>
  );
}

/* ------------------------------------------------------------------ *
 * Detector health
 * ------------------------------------------------------------------ */

/**
 * Provider health: how each detector's runs ended this week.
 *
 * A real table, because this is the one place on the page with two dimensions — provider down,
 * status across — and a grid of divs would leave a screen reader with no way to associate a
 * number with either.
 *
 * The status columns are taken from the providers themselves rather than from a list written
 * here. The API seeds every provider with the schema's whole status vocabulary and carries
 * through anything else it found, so the union of the keys is exactly the set of outcomes this
 * deployment actually recorded — including one this file has never heard of, which is precisely
 * the column that must not be dropped.
 *
 * Nothing is ranked, totalled or scored. There is no "health %" column: that would be a
 * fraction with a denominator this page chose, on the subject the constraint is strictest
 * about.
 */
function Detectors({ detectors }: { detectors: Record<string, Record<string, number>> }) {
  const providers = Object.keys(detectors);
  const statuses = [
    ...new Set(providers.flatMap((provider) => Object.keys(detectors[provider]))),
  ];

  if (providers.length === 0) {
    return (
      <p className="text-[13px] text-muted">
        No detector recorded a result in the last seven days.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-ink-2">
      <table className="w-full min-w-[420px] border-collapse text-left">
        <thead>
          <tr className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
            <th scope="col" className="px-5 py-3 font-medium">
              Provider
            </th>
            {statuses.map((status) => (
              <th key={status} scope="col" className="px-5 py-3 text-right font-medium">
                {status}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {providers.map((provider) => (
            <tr key={provider} className="border-t border-hair">
              <th
                scope="row"
                className="px-5 py-3 font-mono text-[13px] font-normal text-bone"
              >
                {provider}
              </th>
              {statuses.map((status) => (
                <td
                  key={status}
                  className="px-5 py-3 text-right font-mono text-[13px] text-muted tabular-nums"
                >
                  {/* A provider that recorded a status another provider did not holds no key
                      for it. Shown as an em dash rather than a zero, because "this detector
                      never produces that outcome" and "it produced it zero times this week"
                      are different facts and the table should not flatten them. */}
                  {detectors[provider][status] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

/** The four distributions and the detector table, once the summary has been read. */
function Summary({ analytics }: { analytics: AdminAnalytics }) {
  return (
    <>
      <section className="rounded-lg border border-line bg-ink-2 p-5">
        <h3 className="text-[13px] font-medium tracking-[0.08em] text-muted uppercase">
          Analyses submitted
        </h3>
        {/* The API's own total, not a sum of the status map. The two agree today; if a status
            outside the schema's vocabulary ever appeared, a sum written here would be the one
            that was wrong. */}
        <p className="mt-2 font-mono text-4xl text-bone tabular-nums">
          {analytics.analyses_total}
        </p>
        <p className="mt-1.5 text-[12px] text-muted">
          In the last seven days, across every account in this deployment.
        </p>
      </section>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Distribution
          title="Analyses by status"
          note="Where each submission got to."
          counts={analytics.analyses_by_status}
        />
        <Distribution
          title="Detection jobs by status"
          note="The queue behind those submissions. Counted separately from the analyses above rather than assumed to match them."
          counts={analytics.jobs_by_status}
        />
        <Distribution
          title="Risk levels recorded"
          note="As persisted on the analysis. UNDECIDED means no decision has been written yet — it is not the same as UNKNOWN, which is a decision the risk engine reached."
          counts={analytics.risk_distribution}
        />
        <Distribution
          title="How the media arrived"
          note="By acquisition method. 'unrecorded' is media stored before this was kept, not a third way in."
          counts={analytics.acquisition}
        />
      </div>

      <section className="mt-8">
        <h3 className="text-[13px] font-medium tracking-[0.08em] text-bone uppercase">
          Detector health
        </h3>
        <p className="mt-1.5 max-w-[74ch] text-[12px] leading-relaxed text-muted">
          How often each provider&rsquo;s runs ended in each state, over the same seven days.
          These are operational counts, not forensic ones: a FAILED run means the provider could
          not answer, never that the media is fake. A provider is listed only if it recorded at
          least one result — one that appears nowhere below was not asked, which is not the same
          as having failed. No scores are shown, averaged or compared.
        </p>
        <div className="mt-4">
          <Detectors detectors={analytics.detectors} />
        </div>
      </section>
    </>
  );
}

export default async function AdminAnalyticsPage() {
  const [user, result] = await Promise.all([fetchSession(), fetchAnalytics()]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and this read. It is not the
  // access control: the API refused the summary before this line.
  if (user === null || (!result.ok && result.unauthenticated)) {
    redirect(LOGIN_PATH);
  }

  return (
    <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 py-14 sm:px-8">
      <Legend>Administration</Legend>
      <Heading>Operational summary</Heading>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed text-muted">
        What this deployment has processed over the last seven days. Every figure is a count of
        stored rows, aggregated by the database at the moment this page was requested; nothing
        here is averaged, scored or ranked. The window is fixed and there is no date filter.
      </p>

      <div className="mt-6">
        {!result.ok ? (
          <Alert>{result.error}</Alert>
        ) : (
          // No empty branch. A week with nothing in it renders as zeros, which is a reading —
          // "nothing was submitted" — and swapping it for a placeholder sentence would throw
          // away the distinction between a quiet week and a page that failed to load.
          <Summary analytics={result.analytics} />
        )}
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-5">
        <Link href={ADMIN_PATH} className="text-sm text-muted underline hover:text-bone">
          ← Accounts
        </Link>
        <Link href={ADMIN_JOBS_PATH} className="text-sm text-muted underline hover:text-bone">
          Detection jobs
        </Link>
        <Link href={WORKSPACE_PATH} className="text-sm text-muted underline hover:text-bone">
          Back to workspace
        </Link>
        {/* A reload of this page, and deliberately a link rather than a button: the counts move
            on their own and there is no JavaScript here to notice when they do. */}
        <Link
          href={ADMIN_ANALYTICS_PATH}
          className="text-sm text-muted underline hover:text-bone"
        >
          Refresh
        </Link>
      </div>
    </main>
  );
}
