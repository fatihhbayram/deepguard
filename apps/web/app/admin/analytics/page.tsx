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
 * **Detector health is per deployment, with the API's failure rate (R14-T4).** One row per
 * provider and version, a null version shown as such, the API's `failure_rate` formatted and
 * never computed, and `MISSING` — expected signals no row was written for — beside the run
 * statuses. The section says in its own text that these are operational execution failures, not
 * model error or false-positive rates.
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
 * **User feedback is its own section, last, under a rule (R14-T2).** It carries the one ratio on
 * the page — the API's user disagreement rate, formatted here and never computed — and it is
 * unverified opinion, so it is kept apart from everything the pipeline itself recorded and says
 * in its own text that it is neither Ground Truth nor an error rate.
 *
 * Read-only in the strict sense: no form, no route handler behind it, nothing that mutates.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import { ANALYSIS_STATUS_FAILED } from "../../analysis";
import { type CanonicalDomain } from "../../i18n/core";
import { CanonicalLabel, MessageTitle, T } from "../../i18n/client";
import { ADMIN_ANALYTICS_PATH, LOGIN_PATH } from "../../session";
import { AdminAlert } from "../components/AdminAlert";
import { AdminPageHeader } from "../components/AdminPageHeader";
import { AdminSection } from "../components/AdminSection";
import { AdminAnalytics, DetectorHealth, FeedbackBucket, fetchAnalytics } from "../analytics";
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
  domain,
}: {
  title: React.ReactNode;
  note: React.ReactNode;
  counts: Record<string, number>;
  /** The API vocabulary the categories are in, when this build names it; otherwise raw. */
  domain?: CanonicalDomain;
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
            {/* `lang="en"`: a category drawn as the API spelled it — a verdict, a level, a
                ruleset/value pair — is cased by English rules; a category this build names is a
                leaf that names its own language. */}
            <dt
              lang="en"
              className="truncate font-mono text-[11px] tracking-[0.06em] text-muted uppercase"
            >
              {domain === undefined ? category : <CanonicalLabel domain={domain} value={category} />}
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
  label: React.ReactNode;
  value: number;
  note: React.ReactNode;
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
        label={<T k="admin.analytics.kpi.analysesSubmitted" />}
        value={analytics.analyses_total}
        note={<T k="admin.analytics.kpi.analysesSubmittedNote" />}
      />
      <Kpi
        label={<T k="admin.analytics.kpi.analysesFailed" />}
        value={analytics.analyses_by_status[ANALYSIS_STATUS_FAILED]}
        note={<T k="admin.analytics.kpi.analysesFailedNote" />}
      />
      <Kpi
        label={<T k="admin.analytics.kpi.jobsFailed" />}
        value={analytics.jobs_by_status[JOB_STATUS_FAILED]}
        note={<T k="admin.analytics.kpi.jobsFailedNote" />}
      />
    </dl>
  );
}

/* ------------------------------------------------------------------ *
 * Detector health
 * ------------------------------------------------------------------ */

/**
 * Detector health: how each detector deployment's runs ended this week (R14-T4).
 *
 * A real table, because this is the one place on the page with two dimensions — deployment
 * down, status across — and a grid of divs would leave a screen reader with no way to associate
 * a number with either.
 *
 * The status columns are taken from the rows themselves rather than from a list written here.
 * The API seeds every deployment with the schema's whole status vocabulary and carries through
 * anything else it found, so the union of the keys is exactly the set of outcomes this
 * deployment actually recorded — including one this file has never heard of, which is precisely
 * the column that must not be dropped. `MISSING` is among them only where a ruleset expects the
 * deployment, and elsewhere reads as a dash.
 *
 * The failure rate is the API's number, formatted by `rateText` and never re-derived from the
 * counts beside it. Nothing is ranked, totalled or scored.
 */
function Detectors({ detectors }: { detectors: DetectorHealth[] }) {
  const statuses = [
    ...new Set(detectors.flatMap((detector) => Object.keys(detector.status_counts))),
  ];

  if (detectors.length === 0) {
    return (
      <p className="text-[13px] text-muted">
        <T k="admin.analytics.detectors.empty" />
      </p>
    );
  }

  return (
    <AdminSection bleed scroll>
      <table className="w-full min-w-[560px] border-collapse text-left">
        <thead>
          <tr className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.analytics.detectors.provider" />
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.analytics.detectors.version" />
            </th>
            {statuses.map((status) => (
              <th key={status} lang="en" scope="col" className="px-5 py-3 text-right font-medium">
                <CanonicalLabel domain="signalStatus" value={status} />
              </th>
            ))}
            <th scope="col" className="px-5 py-3 text-right font-medium">
              {/* The formula keeps the API's status names in English, which is what the column
                  headings beside it say in English; the hover text around it is translated. */}
              <MessageTitle k="admin.analytics.detectors.failureRateTitle">
                <T k="admin.analytics.detectors.failureRate" />
              </MessageTitle>
            </th>
          </tr>
        </thead>
        <tbody>
          {detectors.map((detector) => (
            <tr
              key={JSON.stringify([detector.provider, detector.provider_version])}
              className="border-t border-hair"
            >
              <th
                scope="row"
                className="px-5 py-3 font-mono text-[13px] font-normal text-bone"
              >
                {detector.provider}
              </th>
              <td className="max-w-[28ch] px-5 py-3 font-mono text-[12px] break-all text-muted">
                {/* Null is what the row holds: no version was persisted. Shown as that, in
                    italics, rather than as a name that reads like a deployment. */}
                {detector.provider_version === null ? (
                  <span className="italic">null</span>
                ) : (
                  detector.provider_version
                )}
              </td>
              {statuses.map((status) => (
                <td
                  key={status}
                  className="px-5 py-3 text-right font-mono text-[13px] text-muted tabular-nums"
                >
                  {/* A deployment that recorded a status another did not holds no key for it.
                      Shown as an em dash rather than a zero, because "this detector never
                      produces that outcome" and "it produced it zero times this week" are
                      different facts and the table should not flatten them. */}
                  {detector.status_counts[status] ?? "—"}
                </td>
              ))}
              <td className="px-5 py-3 text-right font-mono text-[13px] text-muted tabular-nums">
                {rateText(detector.failure_rate)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * User feedback activity (R14-T2)
 * ------------------------------------------------------------------ */

/**
 * The API's disagreement rate, as text.
 *
 * Formatted and nothing more — three decimals of the number the API sent, never re-derived from
 * the counts beside it and never multiplied into a percentage. Null is a bucket nobody gave
 * feedback on, drawn as a dash: there is no rate, which is not a rate of zero.
 */
function rateText(rate: number | null): string {
  return rate === null ? "—" : rate.toFixed(3);
}

/**
 * Feedback by the verdict on the analysis it is about, one table per list the API placed it in.
 *
 * The assessment columns are the union of the buckets' own keys, for the reason the detector
 * table takes its columns from the providers: the API seeds the vocabulary and carries through
 * anything else, and a column this file has never heard of must still be drawn.
 */
function FeedbackTable({
  title,
  buckets,
}: {
  title: React.ReactNode;
  buckets: Record<string, FeedbackBucket>;
}) {
  const verdicts = Object.keys(buckets);
  const assessments = [
    ...new Set(verdicts.flatMap((verdict) => Object.keys(buckets[verdict].assessments))),
  ];

  return (
    <AdminSection bleed scroll>
      <table className="w-full min-w-[560px] border-collapse text-left">
        <caption className="px-5 pt-3 text-left text-[11px] font-medium tracking-[0.12em] text-muted uppercase">
          {title}
        </caption>
        <thead>
          <tr className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.analytics.feedback.verdictShown" />
            </th>
            <th scope="col" className="px-5 py-3 text-right font-medium">
              <T k="admin.analytics.feedback.feedback" />
            </th>
            {assessments.map((assessment) => (
              <th key={assessment} lang="en" scope="col" className="px-5 py-3 text-right font-medium">
                <CanonicalLabel domain="feedbackAssessmentCode" value={assessment} />
              </th>
            ))}
            <th scope="col" className="px-5 py-3 text-right font-medium">
              <T k="admin.analytics.feedback.disagreementRate" />
            </th>
          </tr>
        </thead>
        <tbody>
          {verdicts.map((verdict) => (
            <tr key={verdict} className="border-t border-hair">
              <th
                scope="row"
                className="px-5 py-3 font-mono text-[12px] font-normal text-bone"
              >
                {verdict}
              </th>
              <td className="px-5 py-3 text-right font-mono text-[13px] text-muted tabular-nums">
                {buckets[verdict].total_feedback}
              </td>
              {assessments.map((assessment) => (
                <td
                  key={assessment}
                  className="px-5 py-3 text-right font-mono text-[13px] text-muted tabular-nums"
                >
                  {buckets[verdict].assessments[assessment] ?? "—"}
                </td>
              ))}
              <td className="px-5 py-3 text-right font-mono text-[13px] text-muted tabular-nums">
                {rateText(buckets[verdict].disagreement_rate)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </AdminSection>
  );
}

/**
 * What the owners of analyses said about the results they were shown.
 *
 * **Its own section, last on the page, under a rule.** Everything above is what the pipeline
 * did; this is what people thought of it, and it is kept visibly apart so a disagreement rate is
 * never read beside a decision count as though it scored it. Nothing here is Ground Truth, a
 * Human Review or an evaluation metric, and the two disclaimers say so in the section itself
 * rather than in a tooltip a reader can miss.
 */
function FeedbackActivity({ analytics }: { analytics: AdminAnalytics }) {
  return (
    <section
      aria-labelledby="user-feedback-activity"
      className="mt-10 border-t border-line pt-6"
    >
      <h3
        id="user-feedback-activity"
        className="text-[11px] font-medium tracking-[0.12em] text-bone uppercase"
      >
        <T k="admin.analytics.feedback.title" />
      </h3>
      <div className="mt-2 max-w-[74ch] space-y-1.5 text-[12px] leading-relaxed text-muted">
        <p>
          <T k="admin.analytics.feedback.state" />
        </p>
        <p>
          <T k="admin.analytics.feedback.unverified" />
        </p>
      </div>

      <dl className="mt-4 grid gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-2">
        <Kpi
          label={<T k="admin.analytics.kpi.feedbackInWindow" />}
          value={analytics.feedback_total}
          note={<T k="admin.analytics.kpi.feedbackInWindowNote" />}
        />
        <Kpi
          label={<T k="admin.analytics.kpi.withoutClaim" />}
          value={analytics.feedback_without_claimed_label}
          note={<T k="admin.analytics.kpi.withoutClaimNote" />}
        />
      </dl>

      <div className="mt-4 grid items-start gap-4 sm:grid-cols-2">
        <Distribution
          title={<T k="admin.analytics.dist.byAssessment" />}
          note={<T k="admin.analytics.dist.byAssessmentNote" />}
          counts={analytics.feedback_by_assessment}
          domain="feedbackAssessmentCode"
        />
        <Distribution
          title={<T k="admin.analytics.dist.claimedLabel" />}
          note={<T k="admin.analytics.dist.claimedLabelNote" />}
          counts={analytics.feedback_by_claimed_label}
          domain="claimedLabelCode"
        />
      </div>

      <p className="mt-6 max-w-[74ch] text-[12px] leading-relaxed text-muted">
        <T k="admin.analytics.feedback.byVerdict" />
      </p>
      <div className="mt-3 space-y-4">
        <FeedbackTable
          title={<T k="admin.analytics.feedback.decisions" values={{ ruleset: "r9-v5.0.0" }} />}
          buckets={analytics.feedback_by_decision}
        />
        <FeedbackTable
          title={
            <T
              k="admin.analytics.feedback.levels"
              values={{ first: "p7-v1.0.0", last: "r7-v4.0.0" }}
            />
          }
          buckets={analytics.feedback_by_recorded_risk_level}
        />
        {Object.keys(analytics.feedback_by_unrecognised_risk_state).length > 0 && (
          <FeedbackTable
            title={<T k="admin.analytics.dist.unrecognised" />}
            buckets={analytics.feedback_by_unrecognised_risk_state}
          />
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

/** The distributions and the detector table, once the summary has been read. */
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
          title={<T k="admin.analytics.dist.analyses" />}
          note={<T k="admin.analytics.dist.analysesNote" />}
          counts={analytics.analyses_by_status}
          domain="analysisStatus"
        />
        <Distribution
          title={<T k="admin.analytics.dist.jobs" />}
          note={<T k="admin.analytics.dist.jobsNote" />}
          counts={analytics.jobs_by_status}
          domain="jobStatus"
        />
        {/* The risk column holds two vocabularies, drawn as two lists so a legacy level never
            sits in the list of verdicts (R13-T2). The API places every row by the ruleset
            version stored with it (R13-T3); this page draws its three maps as they arrive and
            translates nothing across. */}
        <Distribution
          title={<T k="admin.analytics.dist.decisions" />}
          note={<T k="admin.analytics.dist.decisionsNote" />}
          counts={analytics.decisions}
        />
        <Distribution
          title={<T k="admin.analytics.dist.levels" />}
          note={<T k="admin.analytics.dist.levelsNote" />}
          counts={analytics.recordedRiskLevels}
        />
        <Distribution
          title={<T k="admin.analytics.dist.acquisition" />}
          note={<T k="admin.analytics.dist.acquisitionNote" />}
          counts={analytics.acquisition}
          domain="acquisitionMethod"
        />
        {/* Only when the week holds a stored value that does not belong to the ruleset stamped
            on it. Carried through under its ruleset and value, as the API carries it, rather
            than dropped or filed into a list it does not belong to. */}
        {Object.keys(analytics.unrecognised).length > 0 && (
          <Distribution
            title={<T k="admin.analytics.dist.unrecognised" />}
            note={<T k="admin.analytics.dist.unrecognisedNote" />}
            counts={analytics.unrecognised}
          />
        )}
      </div>

      <section className="mt-6">
        <h3 className="text-[11px] font-medium tracking-[0.12em] text-bone uppercase">
          <T k="admin.analytics.detectors.title" />
        </h3>
        <p className="mt-1.5 max-w-[74ch] text-[12px] leading-relaxed text-muted">
          <T k="admin.analytics.detectors.description" />
        </p>
        <div className="mt-3">
          <Detectors detectors={analytics.detectors_health} />
        </div>
      </section>

      <FeedbackActivity analytics={analytics} />
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
        title={<T k="admin.analytics.title" />}
        description={<T k="admin.analytics.description" />}
        actions={
          // A reload of this page, and deliberately a link rather than a button: the counts move
          // on their own and there is no JavaScript here to notice when they do. The cross-links
          // that used to sit beside it are the sidebar now.
          <Link
            href={ADMIN_ANALYTICS_PATH}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            <T k="admin.common.refresh" />
          </Link>
        }
      />

      <div className="mt-5">
        {!result.ok ? (
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={result.error} />
          </AdminAlert>
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
