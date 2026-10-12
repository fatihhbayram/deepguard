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
import { CanonicalLabel, T } from "../../i18n/client";
import { ADMIN_AUDIT_PATH, LOGIN_PATH } from "../../session";
import { AdminAlert } from "../components/AdminAlert";
import { AdminPageHeader } from "../components/AdminPageHeader";
import { AdminSection } from "../components/AdminSection";
import { AdminStatusBadge } from "../components/AdminStatusBadge";
import { AdminValue } from "../components/AdminValue";
import { AdminAuditEntry, FieldChange, fetchAuditEvents } from "../audit";

/*
 * The private `Legend`, `Heading`, `Alert` and `Value` this file carried — the fifth copy of the
 * first three, as the comment that stood here counted them — are `AdminPageHeader`, `AdminAlert`
 * and `AdminValue` in `../components` since R8-T9.
 */

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
          {/* The field's name as the API wrote it — a column name, never translated, and marked
              English so a Turkish page cannot case `is_active` into `İS_ACTİVE`. */}
          <dt lang="en" className="font-mono text-[11px] tracking-[0.06em] text-muted uppercase">
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
            <span className="sr-only">
              <T k="admin.audit.diffTo" />{" "}
            </span>
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
          {event.actor_email_snapshot ?? (
            <span className="text-muted">
              <T k="admin.common.notRecorded" />
            </span>
          )}
        </div>
        <div className="mt-1 truncate" title={event.actor_id}>
          <AdminValue>{event.actor_id}</AdminValue>
        </div>
      </td>

      <td className="px-5 py-4">
        <div
          className="truncate text-[13px] text-bone"
          title={event.target_email_snapshot ?? ""}
        >
          {event.target_email_snapshot ?? (
            <span className="text-muted">
              <T k="admin.common.notRecorded" />
            </span>
          )}
        </div>
        <div className="mt-1 truncate" title={event.target_id}>
          {/* The target's type is carried across as stored, named in the reader's language when
              this build knows it and drawn exactly as the API spelled it when it does not. It is
              shown because the day a row names a type nobody expected, this column says so. */}
          <AdminValue>
            <CanonicalLabel domain="auditTarget" value={event.target_type} /> {event.target_id}
          </AdminValue>
        </div>
      </td>

      <td className="px-5 py-4">
        {/* A known action is named in the reader's language and keeps the API's spelling as
            `data-canonical`; an action this build does not know is drawn exactly as recorded,
            in the figure typeface, because that is the row a reader most needs to see as it is. */}
        <AdminStatusBadge tone="neutral" mono>
          <CanonicalLabel domain="auditAction" value={event.action} />
        </AdminStatusBadge>
      </td>

      <td className="px-5 py-4">
        <Diff changes={event.changes} />
      </td>

      <td className="px-5 py-4">
        {/* The API's own timestamp, shown as stored rather than reformatted into a local
            rendering the record does not hold — the convention every table in this surface
            follows. */}
        <AdminValue>{event.created_at}</AdminValue>
        <div className="mt-1 truncate">
          {/* The correlation id of the request that made the change, so one grep covers this
              row and the application's log lines around it. Null on an event written outside a
              request, and shown as absent rather than invented. */}
          <AdminValue>
            {event.request_id === null ? null : (
              <T k="admin.common.requestId" values={{ id: event.request_id }} />
            )}
          </AdminValue>
        </div>
      </td>
    </tr>
  );
}

/** The log as a table, or the sentence that says there is nothing in it yet. */
function AuditTable({ events }: { events: AdminAuditEntry[] }) {
  if (events.length === 0) {
    return (
      <AdminSection>
        <p className="text-[13px] text-muted">
          <T k="admin.audit.empty" />
        </p>
      </AdminSection>
    );
  }

  return (
    <AdminSection bleed scroll>
      <table className="w-full min-w-[900px] border-collapse text-left">
        <thead>
          <tr className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.audit.column.administrator" />
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.audit.column.accountChanged" />
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.audit.column.action" />
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.audit.column.whatChanged" />
            </th>
            <th scope="col" className="px-5 py-3 font-medium">
              <T k="admin.audit.column.when" />
            </th>
          </tr>
        </thead>
        <tbody>
          {events.map((event) => (
            <EventRow key={event.id} event={event} />
          ))}
        </tbody>
      </table>
    </AdminSection>
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
    <>
      <AdminPageHeader
        title={<T k="admin.audit.title" />}
        description={
          <>
            <p>
              <T k="admin.audit.intro" />
            </p>
            <p>
              <T k="admin.audit.snapshots" />
            </p>
          </>
        }
        actions={
          // A reload of this page, and deliberately a link rather than a button: the log grows
          // on its own and there is no JavaScript here to notice when it does. The cross-links
          // that used to sit beside it are the sidebar now.
          <Link
            href={ADMIN_AUDIT_PATH}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            <T k="admin.common.refresh" />
          </Link>
        }
      />

      <div className="mt-5">
        {!result.ok ? (
          // The fetcher's sentence, named in the reader's language when it is one of this
          // application's own; anything else is drawn as it arrived.
          <AdminAlert tone="error">
            <CanonicalLabel domain="webMessage" value={result.error} />
          </AdminAlert>
        ) : (
          <AuditTable events={result.events} />
        )}
      </div>
    </>
  );
}
