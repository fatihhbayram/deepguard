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
import { CanonicalLabel, MessageOption, T } from "../../i18n/client";
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
  children,
}: {
  name: string;
  label: React.ReactNode;
  value: string;
  children: React.ReactNode;
}) {
  // The options are leaves that name themselves in the reader's language (`<MessageOption>`,
  // `<CanonicalOption>`); every `value` they post is the API's own spelling, untouched.
  return (
    <label className="block min-w-0">
      <span className={LABEL}>{label}</span>
      <select name={name} defaultValue={value} className={CONTROL}>
        {children}
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
          label={<T k="admin.queue.filter.feedback" />}
          value={filters.feedback_assessment ?? ""}
        >
          <MessageOption value="" k="admin.queue.filter.any" />
          {QUEUE_FEEDBACK_ASSESSMENTS.map((assessment) => (
            <MessageOption
              key={assessment}
              value={assessment}
              k="admin.queue.filter.has"
              values={{ assessment }}
              domains={{ assessment: "feedbackAssessmentCode" }}
            />
          ))}
        </Select>
        <Select
          name="decision"
          label={<T k="admin.queue.filter.decision" values={{ ruleset: "r9-v5.0.0" }} />}
          value={filters.decision ?? ""}
        >
          <MessageOption value="" k="admin.queue.filter.any" />
          {/* A verdict is the ruleset's own word and is drawn raw in both languages, as the
              report draws it. */}
          {QUEUE_DECISIONS.map((decision) => (
            <option key={decision} value={decision}>
              {decision}
            </option>
          ))}
        </Select>
        <Select
          name="signal_error"
          label={<T k="admin.queue.filter.signals" />}
          value={filters.signal_error ? "true" : ""}
        >
          <MessageOption value="" k="admin.queue.filter.any" />
          <MessageOption value="true" k="admin.queue.filter.signalError" />
        </Select>
        <Select
          name="unreviewed_only"
          label={<T k="admin.queue.filter.reviewState" />}
          value={filters.unreviewed_only ? "" : "false"}
        >
          <MessageOption value="" k="admin.queue.filter.notClosed" />
          <MessageOption value="false" k="admin.queue.filter.all" />
        </Select>
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button
          type="submit"
          className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-bone transition-colors duration-150 hover:border-rule"
        >
          <T k="admin.queue.applyFilters" />
        </button>
        <Link
          href={ADMIN_REVIEW_QUEUE_PATH}
          className="text-[12px] text-muted underline decoration-hair underline-offset-2 hover:text-bone"
        >
          <T k="admin.queue.reset" />
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
      ? (["admin.queue.decision", item.decision] as const)
      : item.recorded_risk_level !== null
        ? (["admin.queue.recordedRiskLevel", item.recorded_risk_level] as const)
        : (["admin.queue.unrecognised", item.unrecognised_risk_state] as const);

  return (
    <div className="space-y-1">
      <div className={LABEL}>
        <T k={kind} />
      </div>
      <div className="font-mono text-[13px] break-all text-bone">{value}</div>
      <AdminValue>{item.risk_rules_version}</AdminValue>
    </div>
  );
}

function Counts({
  counts,
  domain,
}: {
  counts: Record<string, number>;
  domain: "feedbackAssessmentCode" | "signalStatus";
}) {
  // Each key named in the reader's language when this build knows it, and drawn exactly as the
  // API spelled it when it does not; the count beside it is the API's own figure.
  return (
    <div className="space-y-0.5 font-mono text-[12px] text-muted">
      {Object.entries(counts).map(([key, count]) => (
        <div key={key} className={count > 0 ? "text-bone" : undefined}>
          <CanonicalLabel domain={domain} value={key} /> {count}
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
          <span className={LABEL}>
            <T k="admin.common.created" />{" "}
          </span>
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
        <AdminStatusBadge tone="muted">
          <CanonicalLabel domain="reviewStatusCode" value={item.review_status} />
        </AdminStatusBadge>
        {item.analyst_assessment !== null && (
          <AdminValue>
            <CanonicalLabel domain="analystAssessmentCode" value={item.analyst_assessment} />
          </AdminValue>
        )}
      </div>

      <Counts counts={item.feedback_counts} domain="feedbackAssessmentCode" />

      <Counts counts={item.signal_error_counts} domain="signalStatus" />
    </div>
  );
}

function QueueHeader() {
  return (
    <div className={`${GRID} px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase`}>
      <div>
        <T k="admin.queue.column.analysis" />
      </div>
      <div>
        <T k="admin.queue.column.verdict" />
      </div>
      <div>
        <T k="admin.queue.column.review" />
      </div>
      <div>
        <T k="admin.queue.column.feedback" />
      </div>
      <div>
        <T k="admin.queue.column.signalErrors" />
      </div>
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
        <T
          k="admin.queue.range"
          values={{
            range: shown === 0 ? "0" : `${filters.offset + 1}–${filters.offset + shown}`,
            total,
          }}
        />
      </p>
      <div className="flex gap-2">
        {filters.offset > 0 && (
          <Link href={link(previous)} className={style}>
            <T k="admin.queue.newer" />
          </Link>
        )}
        {next < total && (
          <Link href={link(next)} className={style}>
            <T k="admin.queue.older" />
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
        title={<T k="admin.queue.title" />}
        description={<T k="admin.queue.description" />}
      />

      <div className="mt-5 space-y-5">
        <AdminSection>
          <Filters filters={filters} />
        </AdminSection>

        {!result.ok ? (
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={result.error} />
          </AdminAlert>
        ) : result.page.items.length === 0 ? (
          <AdminSection>
            <p className="text-[13px] text-muted">
              <T k="admin.queue.empty" />
            </p>
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
