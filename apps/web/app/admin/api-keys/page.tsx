/**
 * The credentials B2B callers authenticate the public API with: what exists, and what still works.
 *
 * The fifth administrative screen and the only one about the *other* surface. Everything else
 * under `/admin` is about this deployment's own accounts, jobs and history; this is about
 * `/api/public/v1` — who has been given a way in, and how to take it away.
 *
 * **Two controls, deliberately built differently.**
 *
 * Issuing a key is a client component (`create-key.tsx`). It is the only scripting on this
 * surface and it exists for one reason: the response carries a secret, and a secret cannot be
 * carried in the query string a `303` would redirect through. The reasoning is written out
 * there and in `create-api-key/route.ts`.
 *
 * Revoking one is a plain HTML form posting to `/admin/revoke-api-key`, exactly like the
 * account controls on `/admin`. Nothing comes back that cannot go in a URL, so nothing here
 * needs JavaScript — the destructive control is the one that keeps working when the script does
 * not, which is the right way round.
 *
 * **Revocation is confirmed in the page rather than in a dialog.** Clicking `Revoke` links to
 * this same page with `?revoking=<id>`, which draws the confirmation in place of that row's
 * control; confirming posts the form, cancelling is a link back. A `window.confirm` would need
 * the script, and a modal would need considerably more of it — for a confirmation step that a
 * second click on a rendered page already provides. It also means the confirmation names the
 * key it is about, which a generic dialog cannot.
 *
 * **There is no reactivation control, and there is no rename.** Not an omission: the API offers
 * neither. A revoked credential that could come back is exactly what an operator revoking one
 * is relying on being impossible, and a rename would edit a label the audit log already froze.
 * A key is issued, and it is revoked, and after that it is history — which is why revoked keys
 * stay in this table rather than disappearing from it.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../../analysis";
import { ADMIN_API_KEYS_PATH, LOGIN_PATH } from "../../session";
import { AdminAlert } from "../components/AdminAlert";
import { AdminPageHeader } from "../components/AdminPageHeader";
import { AdminSection } from "../components/AdminSection";
import { AdminStatusBadge } from "../components/AdminStatusBadge";
import { AdminApiKey, fetchApiKeys } from "../api-keys";
import { CreateKey } from "./create-key";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * The private `Legend`, `Heading` and `Alert` this file carried are `AdminPageHeader` and
 * `AdminAlert` in `../components` since R8-T9. The comment that stood here said they were
 * restated rather than imported because exporting them from one page module would make one
 * screen a component library for the next; the extraction answers that by giving them a module
 * of their own, which is what nobody had written yet.
 */

/**
 * Whether a key still authenticates, as a chip.
 *
 * Two states and no third, for the reason the account table's chip has two: `is_active` is what
 * `require_api_key` filters on and the only thing this application means by a key working.
 * There is no expired and no suspended, because there is no such column — a badge inventing
 * that vocabulary would promise a lifecycle the system does not have.
 */
function Status({ active }: { active: boolean }) {
  return (
    <AdminStatusBadge tone={active ? "neutral" : "negative"} dot>
      {active ? "Active" : "Revoked"}
    </AdminStatusBadge>
  );
}

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/**
 * The revocation control, in whichever of its two states this render is.
 *
 * Unconfirmed it is a link, which is what makes the first click free of consequence — a link is
 * a GET, and the only thing it changes is which row this page draws a confirmation on.
 * Confirmed it is a form, and the POST behind it is the only request that revokes anything.
 *
 * Nothing is drawn for a key that is already revoked. Not a disabled button: there is no state
 * the page could be in where it would become usable, and a disabled control invites the reader
 * to work out how to enable it.
 */
function RevokeControl({ apiKey, confirming }: { apiKey: AdminApiKey; confirming: boolean }) {
  if (!apiKey.is_active) {
    return <span className="text-[12px] text-muted">Revoked</span>;
  }

  if (!confirming) {
    return (
      <Link
        href={`${ADMIN_API_KEYS_PATH}?revoking=${encodeURIComponent(apiKey.id)}`}
        className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        Revoke
      </Link>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      {/* Named, so the confirmation is about this key rather than about "the selected one". */}
      <span className="text-[12px] text-muted">
        Revoke <span className="text-bone">{apiKey.name}</span>? This cannot be undone.
      </span>
      <form action="/admin/revoke-api-key" method="post">
        <input type="hidden" name="key_id" value={apiKey.id} />
        <button
          type="submit"
          className="rounded-md border border-rose-500/40 px-3 py-1.5 text-[12px] font-medium text-rose-200 transition-colors duration-150 hover:border-rose-400"
        >
          Confirm
        </button>
      </form>
      <Link
        href={ADMIN_API_KEYS_PATH}
        className="text-[12px] text-muted underline hover:text-bone"
      >
        Cancel
      </Link>
    </div>
  );
}

/** One key: who it is for, whether it works, when it was issued and last used. */
function KeyRow({ apiKey, confirming }: { apiKey: AdminApiKey; confirming: boolean }) {
  return (
    <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1.2fr)_minmax(0,auto)] items-center gap-6 border-t border-hair px-5 py-4">
      <div className="min-w-0">
        <div className="truncate text-[13px] text-bone" title={apiKey.name}>
          {apiKey.name}
        </div>
        {/* The id, monospaced: it is a machine value the reader has to be able to check
            character by character, and it is what the audit log names this key by. */}
        <div className="mt-1 truncate font-mono text-[11px] text-muted" title={apiKey.id}>
          {apiKey.id}
        </div>
      </div>

      <div>
        <Status active={apiKey.is_active} />
      </div>

      <div className="min-w-0">
        {/* The API's own timestamps, shown as stored rather than reformatted into a local
            rendering the record does not hold — the convention every table here follows. */}
        <div className="truncate font-mono text-[11px] text-bone">{apiKey.created_at}</div>
        <div className="mt-1 truncate font-mono text-[11px] text-muted">
          {/* Null until the key has authenticated something. Nothing writes this column yet,
              so it reads "never used" on every row today; it is shown because the day it is
              written, this is the field that says whether a key is safe to revoke. */}
          {apiKey.last_used_at ?? "never used"}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2">
        <RevokeControl apiKey={apiKey} confirming={confirming} />
      </div>
    </div>
  );
}

/** The column headings. Always drawn, and always in columns — the table scrolls below the width
 *  its columns need rather than restacking. See `docs/ui-guidance.md`. */
function KeyHeader() {
  return (
    <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1.2fr)_minmax(0,auto)] gap-6 px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
      <div>Key</div>
      <div>Status</div>
      <div>Issued / last used</div>
      <div className="text-right">Manage</div>
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

/** One query-string value, or null. A repeated parameter is not an outcome. */
function singleParam(value: string | string[] | undefined): string | null {
  return typeof value === "string" ? value : null;
}

export default async function ApiKeys({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const [user, result] = await Promise.all([fetchSession(), fetchApiKeys()]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and this read. It is not the access
  // control: the API refused the listing before this line, and what the reader gets here is
  // somewhere useful to go rather than an empty page they cannot explain.
  if (user === null || (!result.ok && result.unauthenticated)) {
    redirect(LOGIN_PATH);
  }

  const error = singleParam(params.error);
  const revoked = singleParam(params.revoked);
  const revoking = singleParam(params.revoking);

  return (
    <>
      <AdminPageHeader
        title="API keys"
        description="The credentials customers authenticate the public API with. A key is shown in full once, when it is issued, and never again — DeepGuard stores only a hash of it. Revoking a key ends its access immediately and leaves the analyses it submitted on file."
        actions={
          // A reload of this page, and deliberately a link rather than a button: the table is a
          // view of state another administrator can change, and the only script on this screen
          // is the issuing control. The cross-link that used to sit beside it is the sidebar now.
          <Link
            href={ADMIN_API_KEYS_PATH}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            Refresh
          </Link>
        }
      />

      {/* The outcome of the last revocation, read out of the query string because that control
          is a plain form and the answer arrives as a redirect. Nothing secret is ever in these
          parameters; the issued key comes back in a response body instead, which is what the
          component below exists for. */}
      {error && (
        <div className="mt-5">
          <AdminAlert tone="error">{error}</AdminAlert>
        </div>
      )}
      {revoked !== null && !error && (
        <div className="mt-5">
          <AdminAlert tone="success">
            Key revoked
            {revoked ? <span className="font-mono"> · {revoked.slice(0, 8)}</span> : null}. It no
            longer authenticates the public API.
          </AdminAlert>
        </div>
      )}

      <div className="mt-5 space-y-5">
        <AdminSection
          title="Issue a key"
          description="The name is a label for this table and the audit log. It is not part of the credential and nothing authenticates by it."
        >
          <div className="mt-4">
            <CreateKey signInPath={LOGIN_PATH} />
          </div>
        </AdminSection>

        {!result.ok ? (
          <AdminAlert tone="error">{result.error}</AdminAlert>
        ) : result.keys.length === 0 ? (
          <AdminSection>
            <p className="text-center text-[13px] text-muted">
              No API keys have been issued. The public API cannot be reached without one.
            </p>
          </AdminSection>
        ) : (
          <AdminSection bleed scroll>
            {/* Wide enough for the four columns and the confirmation sentence the revoke
                control expands into; below that the wrapper scrolls rather than the row
                restacking. */}
            <div className="min-w-[900px]">
              <KeyHeader />
              {result.keys.map((apiKey) => (
                <KeyRow
                  key={apiKey.id}
                  apiKey={apiKey}
                  // The one row, if any, whose revocation is being confirmed on this render.
                  confirming={apiKey.id === revoking}
                />
              ))}
            </div>
          </AdminSection>
        )}
      </div>
    </>
  );
}
