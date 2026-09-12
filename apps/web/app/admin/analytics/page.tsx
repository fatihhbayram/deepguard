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
 * **The headline figures are a KPI strip, not a card each.** The first version of this screen
 * gave `analyses_total` a full-width plate with a 36px numeral and three hundred pixels of empty
 * ground beside it, which spent the most valuable band on the page — the one directly under the
 * title — on one number. The strip below states three, on one row, at the density the rest of
 * the console uses. Every one of them is a figure the API computed: the total it reports, and
 * two direct reads of keys `admin_analytics.py` guarantees it seeds. Nothing is summed, divided
 * or ranked here, which is the constraint the whole screen is built on.
 *
 * The two failure counts also appear in the distributions below. That repetition is deliberate
 * and is the difference between a headline and a breakdown: an operator opening this page is
 * looking for whether anything failed, and making them find it inside a four-row list is making
 * them work for the one number they came for.
 *
 * Read-only in the strict sense: no form, no route handler behind it, nothing that mutates.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import { ANALYSIS_STATUS_FAILED } from "../../analysis";
import { ADMIN_ANALYTICS_PATH, LOGIN_PATH } from "../../session";
import { AdminAlert } from "../components/AdminAlert";
import { AdminPageHeader } from "../components/AdminPageHeader";
import { AdminSection } from "../components/AdminSection";
import { AdminAnalytics, fetchAnalytics } from "../analytics";
import { JOB_STATUS_FAILED } from "../jobs";

/*
 * The private `Legend`, `Heading` and `Alert` this file carried — the fourth copy of three
 * components, as the comment that stood here counted them — are `AdminPageHeader` and
 * `AdminAlert` in `../components` since R8-T9.
 */

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
    // Not an `AdminSection`: that component's heading is a `text-[15px]` card title, and these
    // four cards are a set of small labelled figures rather than four sections of the page. The
    // plate is the same — `rounded-md border border-line bg-ink-2` — which is the whole of what
    // the consolidation was about. The heading is deliberately quieter than a section title:
    // these are the secondary half of the page and should not read at the same weight as the
    // strip above them.
    <section className="rounded-md border border-line bg-ink-2 p-4">
      <h3 className="text-[11px] font-medium tracking-[0.12em] text-muted uppercase">{title}</h3>
      <p className="mt-1 text-[12px] leading-relaxed text-muted/80">{note}</p>

      {/* One line per category: the label, its track, and its count, on a three-column grid so
          the bars all start and end at the same two x-positions and the figures align on their
          right edge. Two lines per category — label above, full-width track below — spent 44px
          on a single number; this spends 26px.

          The magnitude is a track in its own column rather than a wash behind the whole row.
          A partial fill spanning the row read as a selected or highlighted row rather than as a
          quantity, and a full one read as the row being picked out — which is a meaning this
          card does not have and cannot afford on a screen about detector health. */}
      <dl className="mt-3">
        {entries.map(([category, value]) => (
          <div
            key={category}
            className="grid grid-cols-[minmax(0,1fr)_72px_2.75rem] items-center gap-3 py-1"
          >
            <dt className="truncate font-mono text-[11px] tracking-[0.06em] text-muted uppercase">
              {category}
            </dt>
            {/* Decoration over the number beside it, and marked as such: the reading is the
                figure in the `dd`, and a screen reader that announced the track as well would be
                announcing the same count twice. */}
            <div aria-hidden className="h-1 overflow-hidden rounded-full bg-plate">
              <div
                className="h-full rounded-full bg-accent/70"
                style={{ width: `${(value / largest) * 100}%` }}
              />
            </div>
            <dd className="text-right font-mono text-[12px] text-bone tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

/* ------------------------------------------------------------------ *
 * The headline strip
 * ------------------------------------------------------------------ */

/**
 * One headline figure.
 *
 * `primary` is the only hierarchy this component has: the total sets the scale for the page and
 * the two counts beside it are read against it, so it is a step larger and in `bone` while they
 * sit at the size every other figure in the console is set at.
 */
function Kpi({
  label,
  value,
  note,
  primary = false,
}: {
  label: string;
  value: number;
  note: string;
  primary?: boolean;
}) {
  return (
    <div className="bg-ink-2 px-4 py-3.5">
      <dt className="text-[10px] font-medium tracking-[0.16em] text-muted uppercase">{label}</dt>
      <dd
        className={`mt-1.5 font-mono tabular-nums ${
          primary ? "text-[26px] leading-none text-bone" : "text-[20px] leading-none text-bone"
        }`}
      >
        {value}
      </dd>
      <p className="mt-1.5 text-[11px] leading-relaxed text-muted/80">{note}</p>
    </div>
  );
}

/**
 * The three figures an operator opens this page for, on one row.
 *
 * `gap-px` over a `line` ground draws the dividers: three plates separated by a hairline read as
 * one instrument, where three bordered cards read as three unrelated panels. The strip is a `dl`
 * because each tile is a label and its value.
 *
 * **Every figure is the API's.** `analyses_total` is the total it reports; the two failure counts
 * are direct reads of keys `admin_analytics.py` seeds from the schema's own vocabulary, so they
 * are present on every payload and are not defaulted here. Nothing on this row is summed from a
 * map or divided by anything.
 *
 * Analyses and jobs are counted separately, and the notes say so: a submission can be recorded
 * failed without its job being, and the reverse. Collapsing them into one "failures" figure would
 * be this page inventing a number the API never produced.
 */
function Headline({ analytics }: { analytics: AdminAnalytics }) {
  return (
    <dl className="grid gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-3">
      <Kpi
        primary
        label="Analyses submitted"
        value={analytics.analyses_total}
        note="Across every account, last seven days."
      />
      <Kpi
        label="Analyses failed"
        value={analytics.analyses_by_status[ANALYSIS_STATUS_FAILED]}
        note="Submissions the pipeline could not complete."
      />
      <Kpi
        label="Jobs failed"
        value={analytics.jobs_by_status[JOB_STATUS_FAILED]}
        note="Counted separately from the analyses beside them."
      />
    </dl>
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
    <AdminSection bleed scroll>
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
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

/** The four distributions and the detector table, once the summary has been read. */
function Summary({ analytics }: { analytics: AdminAnalytics }) {
  return (
    <>
      {/* The API's own total and two of its own counts. Nothing here is a sum of a map: the two
          agree today, and if a status outside the schema's vocabulary ever appeared, a sum
          written here would be the one that was wrong. */}
      <Headline analytics={analytics} />

      {/* `items-start` so a card with three rows stops at three rows. Stretched to match its
          neighbour it grew a band of empty plate at the bottom, which read as a panel that had
          failed to load the rest of itself. */}
      <div className="mt-4 grid items-start gap-4 sm:grid-cols-2">
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

      <section className="mt-6">
        <h3 className="text-[11px] font-medium tracking-[0.12em] text-bone uppercase">
          Detector health
        </h3>
        <p className="mt-1.5 max-w-[74ch] text-[12px] leading-relaxed text-muted">
          How often each provider&rsquo;s runs ended in each state, over the same seven days.
          These are operational counts, not forensic ones: a FAILED run means the provider could
          not answer, never that the media is fake. A provider is listed only if it recorded at
          least one result — one that appears nowhere below was not asked, which is not the same
          as having failed. No scores are shown, averaged or compared.
        </p>
        <div className="mt-3">
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
    <>
      <AdminPageHeader
        title="Operational summary"
        description="What this deployment has processed over the last seven days. Every figure is a count of stored rows, aggregated by the database at the moment this page was requested; nothing here is averaged, scored or ranked. The window is fixed and there is no date filter."
        actions={
          // A reload of this page, and deliberately a link rather than a button: the counts move
          // on their own and there is no JavaScript here to notice when they do. The cross-links
          // that used to sit beside it are the sidebar now.
          <Link
            href={ADMIN_ANALYTICS_PATH}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            Refresh
          </Link>
        }
      />

      <div className="mt-5">
        {!result.ok ? (
          <AdminAlert tone="error">{result.error}</AdminAlert>
        ) : (
          // No empty branch. A week with nothing in it renders as zeros, which is a reading —
          // "nothing was submitted" — and swapping it for a placeholder sentence would throw
          // away the distinction between a quiet week and a page that failed to load.
          <Summary analytics={result.analytics} />
        )}
      </div>
    </>
  );
}
