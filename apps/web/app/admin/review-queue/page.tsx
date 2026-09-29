/**
 * The review queue, as an administrator works through it (R14-T3).
 *
 * Analyses filtered on stored operational facts — owner feedback, the recorded decision, a
 * detector that did not answer, whether a review has closed it — newest first, each linking to
 * the review screen for that analysis. The filters are a plain GET form, so a filtered queue is
 * an address that can be reloaded or shared.
 *
 * **Queue membership is not a finding.** Nothing on this page calls an analysis a false
 * positive, an error or a priority. The columns restate what the API stored, in its own
 * spelling: a v5 decision, a legacy risk level or an unrecognised pairing, never one read as the
 * other. Feedback is shown as counts, because several accounts can give it on one analysis.
 *
 * Read-only: no mutation, no route handler. Reviewing happens on the analysis's own screen.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import { ADMIN_REVIEW_QUEUE_PATH, adminAnalysisPath, LOGIN_PATH } from "../../session";
import { AdminAlert } from "../components/AdminAlert";
import { AdminPageHeader } from "../components/AdminPageHeader";
import { AdminSection } from "../components/AdminSection";
import { AdminStatusBadge } from "../components/AdminStatusBadge";
import { AdminValue } from "../components/AdminValue";
import {
  QUEUE_DECISIONS,
  QUEUE_FEEDBACK_ASSESSMENTS,
  ReviewQueueFilters,
  ReviewQueueItem,
  fetchReviewQueue,
  filterQuery,
  parseFilters,
} from "../review-queue";

const GRID = "grid grid-cols-[minmax(0,2fr)_minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)] items-start gap-6";

const LABEL = "text-[11px] tracking-[0.08em] text-muted uppercase";

const CONTROL =
  "mt-1 block w-full rounded-md border border-line bg-plate px-2 py-1.5 text-[13px] text-bone";

/* ------------------------------------------------------------------ *
 * Filters
 * ------------------------------------------------------------------ */

function Select({
  name,
  label,
  value,
  options,
}: {
  name: string;
  label: string;
  value: string;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="block min-w-0">
      <span className={LABEL}>{label}</span>
      <select name={name} defaultValue={value} className={CONTROL}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function Filters({ filters }: { filters: ReviewQueueFilters }) {
  return (
    // A GET to this page: the filters become the address. The offset is not carried, so a
    // changed filter starts from the first page.
    <form method="get" action={ADMIN_REVIEW_QUEUE_PATH}>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Select
          name="feedback_assessment"
          label="Owner feedback"
          value={filters.feedback_assessment ?? ""}
          options={[
            { value: "", label: "Any" },
            ...QUEUE_FEEDBACK_ASSESSMENTS.map((assessment) => ({
              value: assessment,
              label: `Has ${assessment}`,
            })),
          ]}
        />
        <Select
          name="decision"
          label="Decision (r9-v5.0.0)"
          value={filters.decision ?? ""}
          options={[
            { value: "", label: "Any" },
            ...QUEUE_DECISIONS.map((decision) => ({ value: decision, label: decision })),
          ]}
        />
        <Select
          name="signal_error"
          label="Detector signals"
          value={filters.signal_error ? "true" : ""}
          options={[
            { value: "", label: "Any" },
            { value: "true", label: "Has FAILED or TIMEOUT" },
          ]}
        />
        <Select
          name="unreviewed_only"
          label="Review state"
          value={filters.unreviewed_only ? "" : "false"}
          options={[
            { value: "", label: "Not closed (unreviewed or follow-up)" },
            { value: "false", label: "All, including REVIEWED" },
          ]}
        />
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button
          type="submit"
          className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-bone transition-colors duration-150 hover:border-rule"
        >
          Apply filters
        </button>
        <Link
          href={ADMIN_REVIEW_QUEUE_PATH}
          className="text-[12px] text-muted underline decoration-hair underline-offset-2 hover:text-bone"
        >
          Reset
        </Link>
      </div>
    </form>
  );
}

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/** The stored verdict, in whichever of the three vocabularies `risk_bucket` placed it. */
function Verdict({ item }: { item: ReviewQueueItem }) {
  const [kind, value] =
    item.decision !== null
      ? ["Decision", item.decision]
      : item.recorded_risk_level !== null
        ? ["Recorded risk level", item.recorded_risk_level]
        : ["Unrecognised", item.unrecognised_risk_state];

  return (
    <div className="space-y-1">
      <div className={LABEL}>{kind}</div>
      <div className="font-mono text-[13px] break-all text-bone">{value}</div>
      <AdminValue>{item.risk_rules_version}</AdminValue>
    </div>
  );
}

function Counts({ counts }: { counts: Record<string, number> }) {
  return (
    <div className="space-y-0.5 font-mono text-[12px] text-muted">
      {Object.entries(counts).map(([key, count]) => (
        <div key={key} className={count > 0 ? "text-bone" : undefined}>
          {key} {count}
        </div>
      ))}
    </div>
  );
}

function QueueRow({ item }: { item: ReviewQueueItem }) {
  return (
    <div className={`${GRID} border-t border-hair px-5 py-4`}>
      <div className="min-w-0 space-y-1">
        <div className="truncate font-mono text-[13px] text-bone" title={item.analysis_id}>
          <Link
            href={adminAnalysisPath(item.analysis_id)}
            className="underline decoration-hair underline-offset-2 transition-colors duration-150 hover:decoration-rule"
          >
            {item.analysis_id}
          </Link>
        </div>
        <div>
          <span className={LABEL}>Created </span>
          <AdminValue>{item.created_at}</AdminValue>
        </div>
        {item.media_sha256s.length === 0 ? (
          <AdminValue>{null}</AdminValue>
        ) : (
          item.media_sha256s.map((sha) => (
            <div key={sha} className="truncate" title={sha}>
              <AdminValue>{`sha256 ${sha}`}</AdminValue>
            </div>
          ))
        )}
      </div>

      <Verdict item={item} />

      <div className="flex flex-wrap items-center gap-2">
        <AdminStatusBadge tone="muted">{item.review_status}</AdminStatusBadge>
        {item.analyst_assessment !== null && <AdminValue>{item.analyst_assessment}</AdminValue>}
      </div>

      <Counts counts={item.feedback_counts} />

      <Counts counts={item.signal_error_counts} />
    </div>
  );
}

function QueueHeader() {
  return (
    <div className={`${GRID} px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase`}>
      <div>Analysis</div>
      <div>Stored verdict</div>
      <div>Review</div>
      <div>Owner feedback</div>
      <div>Signal errors</div>
    </div>
  );
}

function Pager({
  filters,
  total,
  limit,
  shown,
}: {
  filters: ReviewQueueFilters;
  total: number;
  limit: number;
  shown: number;
}) {
  const previous = Math.max(filters.offset - limit, 0);
  const next = filters.offset + limit;
  const link = (offset: number) => {
    const query = filterQuery({ ...filters, offset });
    return query ? `${ADMIN_REVIEW_QUEUE_PATH}?${query}` : ADMIN_REVIEW_QUEUE_PATH;
  };
  const style =
    "rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone";

  return (
    <div className="mt-4 flex items-center justify-between gap-4">
      <p className="text-[12px] text-muted">
        {shown === 0
          ? `0 of ${total}`
          : `${filters.offset + 1}–${filters.offset + shown} of ${total}`}
      </p>
      <div className="flex gap-2">
        {filters.offset > 0 && (
          <Link href={link(previous)} className={style}>
            Newer
          </Link>
        )}
        {next < total && (
          <Link href={link(next)} className={style}>
            Older
          </Link>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default async function AdminReviewQueue({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const filters = parseFilters(await searchParams);
  const [user, result] = await Promise.all([fetchSession(), fetchReviewQueue(filters)]);

  // A session that expired between the layout's guard and this read. Not the access control:
  // the API refused the queue before this line.
  if (user === null || (!result.ok && result.unauthenticated)) {
    redirect(LOGIN_PATH);
  }

  return (
    <>
      <AdminPageHeader
        title="Review queue"
        description="Analyses matching the filters below, newest first. Being listed here is not a finding: it means only that the stored facts matched — owner feedback, the recorded decision, a detector that did not answer, or a review not yet closed. Nothing here is ranked or scored, and nothing here changes a result."
      />

      <div className="mt-5 space-y-5">
        <AdminSection>
          <Filters filters={filters} />
        </AdminSection>

        {!result.ok ? (
          <AdminAlert tone="error">{result.error}</AdminAlert>
        ) : result.page.items.length === 0 ? (
          <AdminSection>
            <p className="text-[13px] text-muted">No analysis matches these filters.</p>
          </AdminSection>
        ) : (
          <div>
            <AdminSection bleed scroll>
              <div className="min-w-[1040px]">
                <QueueHeader />
                {result.page.items.map((item) => (
                  <QueueRow key={item.analysis_id} item={item} />
                ))}
              </div>
            </AdminSection>
            <Pager
              filters={filters}
              total={result.page.total}
              limit={result.page.limit}
              shown={result.page.items.length}
            />
          </div>
        )}
      </div>
    </>
  );
}
