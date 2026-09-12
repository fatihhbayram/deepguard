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
import { ADMIN_JOBS_PATH, ADMIN_PATH, LOGIN_PATH, WORKSPACE_PATH } from "../../session";
import { AdminJob, fetchJobs } from "../jobs";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * `Legend`, `Heading` and `Alert` restated again, as `app/admin/page.tsx` restates them from
 * the workspace. This is the third screen to carry them, which is the point at which the Rule
 * of Three says to extract — and extracting means editing two pages that this task is not
 * about. Noted here deliberately so the next person to want them has the count in front of
 * them rather than a fourth copy to make.
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

/** Why the listing is not on the screen, stated in the page's own voice. */
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

/**
 * The state of one job, as a chip.
 *
 * The four the column can hold and no fifth. `queued`, `processing`, `completed` and `failed`
 * are what `app/db/models.py` writes, and anything else that arrived would be a status this
 * application does not have — so it is rendered as it came rather than mapped to a guess, in
 * the neutral treatment, which is the honest way to show a word nobody here understands.
 */
function Status({ status }: { status: string }) {
  const tone =
    status === "failed"
      ? "bg-rose-500/10 text-rose-200"
      : status === "completed"
        ? "bg-chip text-bone"
        : "bg-chip text-muted";

  return (
    <span
      className={`inline-flex items-center rounded-sm px-2 py-1 text-[11px] font-medium tracking-[0.08em] uppercase ${tone}`}
    >
      {status}
    </span>
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
    <span
      title="The worker's claim on this job expired while it was still running. Recovery will fail it on its next pass."
      className="inline-flex items-center gap-1.5 rounded-sm bg-amber-500/10 px-2 py-1 text-[11px] font-medium tracking-[0.08em] text-amber-200 uppercase"
    >
      <span aria-hidden className="size-1.5 rounded-full bg-amber-400" />
      Stale
    </span>
  );
}

/** A machine value, or an em dash where the record holds nothing. */
function Value({ children }: { children: string | null }) {
  return children === null ? (
    <span className="text-muted">—</span>
  ) : (
    <span className="font-mono text-[11px] text-muted">{children}</span>
  );
}

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/** One job: what it is, what state it is in, and — when it failed — why. */
function JobRow({ job }: { job: AdminJob }) {
  return (
    <div className="border-t border-hair px-5 py-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1.4fr)] sm:items-start sm:gap-6">
        <div className="min-w-0">
          {/* The analysis first and the job second: the analysis id is what an operator has in
              front of them from the workspace, the report or a customer's message, and the job
              id is the thing they only ever learn here. */}
          <div className="truncate font-mono text-[13px] text-bone" title={job.analysis_id}>
            {job.analysis_id}
          </div>
          <div className="mt-1 truncate" title={job.id}>
            <Value>{`job ${job.id}`}</Value>
          </div>
          {/* The correlation id the API bound to the request that queued this work, so one
              grep covers the browser request and the analysis minutes later. Null on every job
              queued before the column existed, and shown as absent rather than invented. */}
          <div className="mt-1 truncate">
            <Value>{job.request_id === null ? null : `request ${job.request_id}`}</Value>
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
            <Value>{job.created_at}</Value>
          </div>
          <div>
            <span className="text-[11px] tracking-[0.08em] text-muted uppercase">Updated </span>
            <Value>{job.updated_at}</Value>
          </div>
          <div>
            <span className="text-[11px] tracking-[0.08em] text-muted uppercase">Lease </span>
            <Value>{job.lease_expires_at}</Value>
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

/** The column headings, on the layouts wide enough to have columns. */
function JobHeader() {
  return (
    <div className="hidden grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1.4fr)] gap-6 px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase sm:grid">
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
    <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 py-14 sm:px-8">
      <Legend>Administration</Legend>
      <Heading>Detection jobs</Heading>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed text-muted">
        The most recent detection work in this deployment, newest first. A job marked stale was
        claimed by a worker whose lease has since expired; recovery fails those on its next pass
        and nothing here changes them. Failures carry the message the worker recorded.
      </p>

      <div className="mt-6">
        {!result.ok ? (
          <Alert>{result.error}</Alert>
        ) : result.jobs.length === 0 ? (
          <p className="text-[13px] text-muted">No detection work has been submitted yet.</p>
        ) : (
          <div className="overflow-hidden rounded-lg border border-line bg-ink-2">
            <JobHeader />
            {result.jobs.map((job) => (
              <JobRow key={job.id} job={job} />
            ))}
          </div>
        )}
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-5">
        <Link href={ADMIN_PATH} className="text-sm text-muted underline hover:text-bone">
          ← Accounts
        </Link>
        <Link href={WORKSPACE_PATH} className="text-sm text-muted underline hover:text-bone">
          Back to workspace
        </Link>
        {/* A reload of this page, and deliberately a link rather than a button: the queue moves
            on its own and there is no JavaScript here to notice when it does. */}
        <Link href={ADMIN_JOBS_PATH} className="text-sm text-muted underline hover:text-bone">
          Refresh
        </Link>
      </div>
    </main>
  );
}
