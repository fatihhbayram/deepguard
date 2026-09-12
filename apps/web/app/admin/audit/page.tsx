/**
 * What administrators have done to other people's accounts (R8-T5).
 *
 * One table, no controls, and the absence of controls is stricter here than on any other
 * screen in this surface. The detection queue has no retry button because a retry cannot be
 * honoured safely; this page has no delete button because a log its own subjects can edit is
 * not a log. There is no endpoint behind such a control either — see `app/api/admin_audit.py`
 * — so the omission is structural rather than a decision this file makes.
 *
 * **The addresses shown are snapshots, and the page says so.** `actor_email_snapshot` and
 * `target_email_snapshot` were frozen when the change happened, and the API never resolves
 * them against the accounts table. So a row can legitimately name an address that no account
 * currently has — after a rename, it will. That reads as a bug unless the screen explains it,
 * so the screen explains it, in prose, above the table.
 *
 * **The diff is rendered from the payload's structure, never re-derived.** `changes` holds
 * only the fields that actually moved; this page turns each into `old → new` and nothing more.
 * It does not decide whether a change was an escalation, does not colour a demotion
 * differently from a promotion, and does not summarise two field changes into a sentence. Each
 * of those would be this file's interpretation of an event it did not witness.
 *
 * Read-only in the strict sense: no form, no route handler behind it, nothing that mutates.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import {
  ADMIN_AUDIT_PATH,
  ADMIN_JOBS_PATH,
  ADMIN_PATH,
  LOGIN_PATH,
  WORKSPACE_PATH,
} from "../../session";
import { AdminAuditEntry, FieldChange, fetchAuditEvents } from "../audit";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * `Legend`, `Heading` and `Alert` restated a fifth time. `app/admin/jobs/page.tsx` noted the
 * third copy as the point the Rule of Three says to extract, and `analytics/page.tsx` carried
 * the count forward to four. It is five. The extraction is now clearly owed and is still a
 * change to four pages that this task is not about — carried forward again so the number stays
 * in front of whoever picks it up rather than resetting quietly at each new screen.
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

/** Why the log is not on the screen, stated in the page's own voice. */
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

/** A machine value, or an em dash where the record holds nothing. */
function Value({ children }: { children: string | null }) {
  return children === null ? (
    <span className="text-muted">—</span>
  ) : (
    <span className="font-mono text-[11px] text-muted">{children}</span>
  );
}

/* ------------------------------------------------------------------ *
 * The diff
 * ------------------------------------------------------------------ */

/**
 * One recorded value, as text.
 *
 * `old` and `new` are `unknown` because the API writes whatever the column held — a string for
 * `role`, a boolean for `is_active`, something else the day a third field becomes auditable.
 * `String()` rather than an interpolation into JSX, and deliberately: React renders `false` and
 * `null` as nothing at all, so `{change.old}` would draw a deactivation as an empty box where
 * the word `true` should be. Every value is turned into text here, including the falsy ones.
 */
function recorded(value: unknown): string {
  return value === null ? "null" : String(value);
}

/**
 * The fields one event changed, as `field: old → new`.
 *
 * Rendered straight from the payload's own structure. The fields present are the fields that
 * moved — the API records nothing else — so there is no filtering to do here and no "unchanged"
 * case to draw.
 *
 * A definition list rather than a sentence: the entries are pairs, a screen reader should be
 * able to associate each value with its field, and an event that changed two things should
 * read as two rows instead of a clause somebody has to parse.
 */
function Diff({ changes }: { changes: Record<string, FieldChange> }) {
  const entries = Object.entries(changes);

  if (entries.length === 0) {
    // The API does not write an event with an empty diff — that is the no-op case, and it
    // writes nothing at all. Drawn anyway rather than assumed away: an empty cell here would
    // otherwise be indistinguishable from a rendering bug.
    return <span className="text-muted">—</span>;
  }

  return (
    <dl className="space-y-1">
      {entries.map(([field, change]) => (
        <div key={field} className="flex flex-wrap items-baseline gap-x-2">
          <dt className="font-mono text-[11px] tracking-[0.06em] text-muted uppercase">
            {field}
          </dt>
          <dd className="font-mono text-[12px] text-bone">
            <span className="text-muted">{recorded(change.old)}</span>
            <span aria-hidden className="px-1.5 text-muted">
              →
            </span>
            {/* The arrow is decorative; the reading order of the two values already carries
                "from this, to this", and a screen reader announcing "right arrow" between them
                adds noise rather than meaning. */}
            <span className="sr-only">to </span>
            {recorded(change.new)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/** One recorded change: who did it, to whom, what moved, and when. */
function EventRow({ event }: { event: AdminAuditEntry }) {
  return (
    <tr className="border-t border-hair align-top">
      <td className="px-5 py-4">
        {/* The address as it read at the time, and the id beside it. The id is what stays
            true: it is what the database can be queried by, and it does not change when
            somebody renames themselves. */}
        <div className="truncate text-[13px] text-bone" title={event.actor_email_snapshot ?? ""}>
          {event.actor_email_snapshot ?? <span className="text-muted">not recorded</span>}
        </div>
        <div className="mt-1 truncate" title={event.actor_id}>
          <Value>{event.actor_id}</Value>
        </div>
      </td>

      <td className="px-5 py-4">
        <div
          className="truncate text-[13px] text-bone"
          title={event.target_email_snapshot ?? ""}
        >
          {event.target_email_snapshot ?? <span className="text-muted">not recorded</span>}
        </div>
        <div className="mt-1 truncate" title={event.target_id}>
          {/* The target's type is carried across as stored. It is `USER` on every row today;
              it is shown because the day it is not, this column is what says so. */}
          <Value>{`${event.target_type.toLowerCase()} ${event.target_id}`}</Value>
        </div>
      </td>

      <td className="px-5 py-4">
        <span className="inline-flex items-center rounded-sm bg-chip px-2 py-1 font-mono text-[11px] tracking-[0.08em] text-bone uppercase">
          {event.action}
        </span>
      </td>

      <td className="px-5 py-4">
        <Diff changes={event.changes} />
      </td>

      <td className="px-5 py-4">
        {/* The API's own timestamp, shown as stored rather than reformatted into a local
            rendering the record does not hold — the convention every table in this surface
            follows. */}
        <Value>{event.created_at}</Value>
        <div className="mt-1 truncate">
          {/* The correlation id of the request that made the change, so one grep covers this
              row and the application's log lines around it. Null on an event written outside a
              request, and shown as absent rather than invented. */}
          <Value>{event.request_id === null ? null : `request ${event.request_id}`}</Value>
        </div>
      </td>
    </tr>
  );
}

/** The log as a table, or the sentence that says there is nothing in it yet. */
function AuditTable({ events }: { events: AdminAuditEntry[] }) {
  if (events.length === 0) {
    return (
      <p className="text-[13px] text-muted">
        No administrative changes have been recorded yet. The log begins where it was
        introduced; changes made before that were not recorded and are not reconstructed here.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-ink-2">
      <table className="w-full min-w-[900px] border-collapse text-left">
        <thead>
          <tr className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
            <th scope="col" className="px-5 py-3 font-medium">
              Administrator
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              Account changed
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              Action
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              What changed
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              When
            </th>
          </tr>
        </thead>
        <tbody>
          {events.map((event) => (
            <EventRow key={event.id} event={event} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

export default async function AdminAudit() {
  const [user, result] = await Promise.all([fetchSession(), fetchAuditEvents()]);

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
      <Heading>Audit log</Heading>
      <p className="mt-3 max-w-[74ch] text-[15px] leading-relaxed text-muted">
        Every change an administrator has made to another account, newest first. Each entry is
        written in the same transaction as the change itself, so a change that was refused or
        rolled back leaves no entry, and a request that altered nothing leaves none either. The
        log is append-only: nothing in this application can edit or remove an entry.
      </p>
      <p className="mt-3 max-w-[74ch] text-[13px] leading-relaxed text-muted">
        The addresses below are recorded as they read at the moment of the change, not looked
        up now — so an entry may name an address an account no longer uses. That is deliberate:
        an entry is a statement about what happened, and renaming an account does not change
        what it did.
      </p>

      <div className="mt-6">
        {!result.ok ? <Alert>{result.error}</Alert> : <AuditTable events={result.events} />}
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
        {/* A reload of this page, and deliberately a link rather than a button: the log grows
            on its own and there is no JavaScript here to notice when it does. */}
        <Link href={ADMIN_AUDIT_PATH} className="text-sm text-muted underline hover:text-bone">
          Refresh
        </Link>
      </div>
    </main>
  );
}
