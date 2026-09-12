/**
 * The detection queue, as an operator reads it (R8-T3).
 *
 * One table, no controls. An administrator arrives here because a report did not appear, and
 * what answers that is the job row: what state the work is in, when it last moved, and — if it
 * failed — the sentence the worker wrote when it did. Everything on this page is one of those.
 *
 * **There is no retry button, and its absence is the design rather than an unfinished part of
 * it.** Re-running a job that already reached `persist_evidence` would write a second signal
 * per provider over the first one's, which is duplicate forensic record on the table whose
 * whole purpose is that a stored signal is what the detector actually said. A control that
 * cannot be honoured safely is not drawn disabled here — it is not drawn. The reasoning is in
 * `app/api/admin_jobs.py`, where the missing endpoint is.
 *
 * **Stale is the API's word, not this page's.** `is_stale` arrives computed against the
 * database clock — the same clock the worker leases against — and this file renders it. It
 * never compares `lease_expires_at` to the reader's own clock: a browser an hour out would
 * either invent dead workers or hide one, and the deadline is shown here as a fact to read
 * rather than as an input to a comparison.
 *
 * The page is read-only in the strict sense: no form, no route handler behind it, nothing that
 * mutates. Recovering a stale job remains the worker's `recover_stale_jobs`, which fails it on
 * its next pass; what this screen does is let somebody see that before the pass happens.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import { ADMIN_JOBS_PATH, adminAnalysisPath, LOGIN_PATH } from "../../session";
import { AdminAlert } from "../components/AdminAlert";
import { AdminPageHeader } from "../components/AdminPageHeader";
import { AdminSection } from "../components/AdminSection";
import { AdminStatusBadge } from "../components/AdminStatusBadge";
import { AdminValue } from "../components/AdminValue";
import { AdminJob, JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, fetchJobs } from "../jobs";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * The comment that used to stand here counted `Legend`, `Heading` and `Alert` to three and said
 * that was the point the Rule of Three calls for an extraction. R8-T9 made it. They are
 * `AdminPageHeader` and `AdminAlert` in `../components` now, along with the chip and the `Value`
 * this file also had a private copy of.
 */

/**
 * The state of one job, as a chip.
 *
 * The four the column can hold and no fifth. `queued`, `processing`, `completed` and `failed`
 * are what `app/db/models.py` writes, and anything else that arrived would be a status this
 * application does not have — so it is rendered as it came rather than mapped to a guess, in
 * the neutral treatment, which is the honest way to show a word nobody here understands.
 */
function Status({ status }: { status: string }) {
  return (
    <AdminStatusBadge
      tone={
        status === JOB_STATUS_FAILED
          ? "negative"
          : status === JOB_STATUS_COMPLETED
            ? "neutral"
            : "muted"
      }
    >
      {/* The API's own spelling, never title-cased or softened. */}
      {status}
    </AdminStatusBadge>
  );
}

/**
 * The stale marker, drawn only when the API says so.
 *
 * Rendered off `is_stale` and nothing else — no local clock, no comparison. The title says
 * what the flag means, because "stale" on its own reads like a judgement about the analysis
 * rather than about the worker that was holding it.
 */
function Stale() {
  return (
    <AdminStatusBadge
      tone="warning"
      dot
      title="The worker's claim on this job expired while it was still running. Recovery will fail it on its next pass."
    >
      Stale
    </AdminStatusBadge>
  );
}

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/** One job: what it is, what state it is in, and — when it failed — why. */
function JobRow({ job }: { job: AdminJob }) {
  return (
    <div className="border-t border-hair px-5 py-4">
      <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1.4fr)] items-start gap-6">
        <div className="min-w-0">
          {/* The analysis first and the job second: the analysis id is what an operator has in
              front of them from the workspace, the report or a customer's message, and the job
              id is the thing they only ever learn here. */}
          {/* A link since R8-T7: the analysis id is the handle an operator already has, and
              the review screen is the only page that is about one analysis. The row itself
              stays a row — nothing else in this table is clickable, and the job id below is
              deliberately still plain text, because there is no page about a job. */}
          <div className="truncate font-mono text-[13px] text-bone" title={job.analysis_id}>
            <Link
              href={adminAnalysisPath(job.analysis_id)}
              className="underline decoration-hair underline-offset-2 transition-colors duration-150 hover:decoration-rule"
            >
              {job.analysis_id}
            </Link>
          </div>
          <div className="mt-1 truncate" title={job.id}>
            <AdminValue>{`job ${job.id}`}</AdminValue>
          </div>
          {/* The correlation id the API bound to the request that queued this work, so one
              grep covers the browser request and the analysis minutes later. Null on every job
              queued before the column existed, and shown as absent rather than invented. */}
          <div className="mt-1 truncate">
            <AdminValue>{job.request_id === null ? null : `request ${job.request_id}`}</AdminValue>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Status status={job.status} />
          {/* Strictly the server's boolean. */}
          {job.is_stale && <Stale />}
        </div>

        {/* The API's own timestamps, shown as stored rather than reformatted into a local
            rendering the record does not hold — the convention the case log and the account
            table already follow, and the one that keeps the lease out of a clock comparison. */}
        <div className="space-y-1">
          <div>
            <span className="text-[11px] tracking-[0.08em] text-muted uppercase">Created </span>
            <AdminValue>{job.created_at}</AdminValue>
          </div>
          <div>
            <span className="text-[11px] tracking-[0.08em] text-muted uppercase">Updated </span>
            <AdminValue>{job.updated_at}</AdminValue>
          </div>
          <div>
            <span className="text-[11px] tracking-[0.08em] text-muted uppercase">Lease </span>
            <AdminValue>{job.lease_expires_at}</AdminValue>
          </div>
        </div>
      </div>

      {/* The worker's own diagnostic text, on its own line because it is prose of unbounded
          length and a table cell would either clip it or wreck the column widths. Rendered
          only when there is one: a job that has not failed holds null, and an empty box under
          every row would read as a failure with nothing to say. */}
      {job.error_message !== null && (
        <pre className="mt-3 overflow-x-auto rounded-md border border-rose-500/30 bg-rose-500/5 px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-rose-200">
          {job.error_message}
        </pre>
      )}
    </div>
  );
}

/** The column headings. Always drawn, and always in columns — the table scrolls below the width
 *  its columns need rather than restacking. See `docs/ui-guidance.md`. */
function JobHeader() {
  return (
    <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1.4fr)] gap-6 px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
      <div>Analysis</div>
      <div>State</div>
      <div>Timestamps</div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default async function AdminJobs() {
  const [user, result] = await Promise.all([fetchSession(), fetchJobs()]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and this read. It is not the
  // access control: the API refused the listing before this line.
  if (user === null || (!result.ok && result.unauthenticated)) {
    redirect(LOGIN_PATH);
  }

  return (
    <>
      <AdminPageHeader
        title="Detection jobs"
        description="The most recent detection work in this deployment, newest first. A job marked stale was claimed by a worker whose lease has since expired; recovery fails those on its next pass and nothing here changes them. Failures carry the message the worker recorded."
        actions={
          // A reload of this page, and deliberately a link rather than a button: the queue moves
          // on its own and there is no JavaScript here to notice when it does. The cross-links
          // that used to sit beside it are the sidebar now.
          <Link
            href={ADMIN_JOBS_PATH}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            Refresh
          </Link>
        }
      />

      <div className="mt-5">
        {!result.ok ? (
          <AdminAlert tone="error">{result.error}</AdminAlert>
        ) : result.jobs.length === 0 ? (
          <AdminSection>
            <p className="text-[13px] text-muted">
              No detection work has been submitted yet.
            </p>
          </AdminSection>
        ) : (
          <AdminSection bleed scroll>
            {/* Wide enough for the three columns and the ids they carry; below that the wrapper
                scrolls rather than the row restacking. */}
            <div className="min-w-[880px]">
              <JobHeader />
              {result.jobs.map((job) => (
                <JobRow key={job.id} job={job} />
              ))}
            </div>
          </AdminSection>
        )}
      </div>
    </>
  );
}
