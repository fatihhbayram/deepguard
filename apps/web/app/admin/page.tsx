/**
 * The administrative workspace: who has an account here, and what each of them may do.
 *
 * R8-T1 created this route to prove one thing — that a privileged surface can be entered by an
 * administrator and by nobody else — and deliberately put nothing in it. This is the first
 * thing operated from here.
 *
 * Every account in the system is listed, deactivated ones included: they are the accounts
 * somebody comes to this page to reactivate, and a table that filtered them would make that
 * impossible from the only screen that offers it. Two controls sit on each row — the role, and
 * whether the account can sign in — and each is a plain HTML form posting to
 * `/admin/update-user`. No client component and no JavaScript, which is what the rest of this
 * application does and what the redirect-and-read-the-query-string feedback below is the cost
 * of.
 *
 * **The controls on the reader's own row are not rendered.** An administrator may not change
 * their own role or activation — it is the one change nobody can undo for them, because after
 * it they can no longer reach the route that would put it back — and the API refuses it with a
 * 400. What this page does is not draw a control for a change that would be refused. That is
 * presentation, not enforcement: the rule is the API's, it is enforced on every request, and a
 * reader who posted the form by hand would be refused there. Drawing a disabled control and
 * relying on it would be a security boundary in a browser; drawing nothing is simply not
 * offering an operator a button that cannot work.
 *
 * The last-active-administrator invariant has no counterpart here, and that is deliberate.
 * Whether a demotion would empty the role is a `COUNT` over the accounts table at the moment
 * the change is applied, and a page rendered seconds earlier cannot know it — a control hidden
 * on this page's arithmetic would be hidden on a stale answer. The API decides, and its
 * refusal is shown above the table like any other.
 *
 * **This route stays the single source of truth for the account list (R8-T8).** There is no
 * `/admin/users` listing and there must not be one: two screens answering "who has an account
 * here" would be two tables to keep in step and two places an operator could read a stale one.
 * What R8-T8 added is a creation form at the top of this page and a link from each row to
 * `/admin/users/[id]`, which is about one account and shows what a row cannot — the address in
 * full, the password reset, and the deactivation policy stated in words.
 *
 * **The creation form is a `<details>` disclosure, not a modal.** A modal needs JavaScript to
 * open, to trap focus and to close, and this surface has none by design; a disclosure is a
 * browser primitive that does all three. It is collapsed by default so the page still opens on
 * the table, which is what an operator came for.
 *
 * **Since R8-T9 this page owns no chrome.** The rail, the page gutter and the `<main>` belong to
 * `admin/layout.tsx`, and the strip of cross-links that used to sit at the foot of the screen is
 * the sidebar now. What is left here is the one action that is genuinely about this page — a
 * reload — and the table itself.
 *
 * **The password field is a plain `<input type="password">` in a form that POSTs.** That is
 * safe in a way the equivalent for an API key would not be: the secret travels inwards in a
 * request body and nothing comes back but a status, so no credential is ever put in the query
 * string this page reads its feedback out of. `/admin/create-user` states the same trade from
 * the other side.
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../analysis";
import {
  ADMIN_PATH,
  LOGIN_PATH,
  USER_ROLE_ADMIN,
  USER_ROLE_USER,
  adminAccountPath,
} from "../session";
import { AdminAlert } from "./components/AdminAlert";
import { AdminPageHeader } from "./components/AdminPageHeader";
import { AdminSection } from "./components/AdminSection";
import { AdminStatusBadge } from "./components/AdminStatusBadge";
import { MINIMUM_PASSWORD_LENGTH, AdminAccount, fetchAccounts } from "./users";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * The private `Legend`, `Heading`, `Alert` and `Activity` this file used to carry are gone.
 * They were the first of the seven copies whose count every later screen carried forward in a
 * comment; R8-T9 extracted them to `./components`, and this page now draws the same header, the
 * same alert and the same chip as the other six.
 */

/**
 * Whether an account can sign in, as a chip.
 *
 * Two states and no third. `is_active` is a boolean on the account and the only thing this
 * application means by an account being usable — there is no pending, no invited and no
 * suspended, because there is no such column and inventing the vocabulary in a badge would
 * promise a lifecycle the system does not have.
 */
function Activity({ active }: { active: boolean }) {
  return (
    <AdminStatusBadge tone={active ? "neutral" : "negative"} dot>
      {active ? "Active" : "Deactivated"}
    </AdminStatusBadge>
  );
}

/* ------------------------------------------------------------------ *
 * Controls
 * ------------------------------------------------------------------ */

/**
 * The role control: a select and the button that submits it.
 *
 * The two options are `USER_ROLE_USER` and `USER_ROLE_ADMIN` read from `../session`, not two
 * strings written into the markup. The API validates what arrives against its own constants
 * and refuses anything else, so a `"Admin"` typed here would not silently strip somebody's
 * access — but it would be a control that produces a 422 on every use, and the constants are
 * there precisely so the two ends spell the column the same way.
 *
 * `defaultValue` rather than `value`: this is an uncontrolled input in a server-rendered form,
 * and a `value` with no `onChange` is a field React will not let the reader alter.
 *
 * The submit button is always enabled, including when the select still shows the current role.
 * Submitting a role an account already has is a no-op the API applies and returns unchanged,
 * which is a better outcome than a button whose disabled state would need JavaScript to track.
 */
function RoleControl({ account }: { account: AdminAccount }) {
  return (
    <form action="/admin/update-user" method="post" className="flex items-center gap-2">
      <input type="hidden" name="user_id" value={account.id} />
      <label className="sr-only" htmlFor={`role-${account.id}`}>
        Role for {account.email}
      </label>
      <select
        id={`role-${account.id}`}
        name="role"
        defaultValue={account.role}
        className="rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule"
      >
        <option value={USER_ROLE_USER}>User</option>
        <option value={USER_ROLE_ADMIN}>Admin</option>
      </select>
      <button
        type="submit"
        className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        Apply
      </button>
    </form>
  );
}

/**
 * The activation control: one button that means the opposite of the current state.
 *
 * A button carrying a hidden value rather than a checkbox, because an unchecked checkbox sends
 * nothing at all — the form would arrive with no `is_active` field and the change would be a
 * no-op in exactly the case the operator meant to deactivate somebody. The value is the state
 * being asked for, so what the form says and what the button is labelled cannot drift apart.
 */
function ActivityControl({ account }: { account: AdminAccount }) {
  return (
    <form action="/admin/update-user" method="post">
      <input type="hidden" name="user_id" value={account.id} />
      <input type="hidden" name="is_active" value={account.is_active ? "false" : "true"} />
      <button
        type="submit"
        className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
      >
        {account.is_active ? "Deactivate" : "Reactivate"}
      </button>
    </form>
  );
}

/**
 * The creation form, collapsed behind a disclosure at the top of the page.
 *
 * A `<details>` rather than a modal for the reason stated at the top of this file: a modal needs
 * JavaScript and this surface has none. Collapsed by default so the page still opens on the
 * table, which is what an operator came here for.
 *
 * **The password is typed here and travels in the POST body, which is why this is safe as a
 * plain form.** Nothing comes back but a status, so the redirect that follows carries an id and
 * never a credential. That is the whole difference between this control and the API key one,
 * which cannot redirect at all.
 *
 * `minLength` on the password field is a courtesy, not a check. The API enforces
 * `MINIMUM_PASSWORD_LENGTH` and refuses anything shorter whatever the browser did; this is so an
 * operator learns the floor while typing rather than after a round trip. `required` is the same
 * kind of thing — the handler refuses an empty field too, and a reader with JavaScript disabled
 * or a browser that ignores the attribute gets the same refusal, just later.
 *
 * `autoComplete="new-password"` so the browser does not offer the *administrator's* own saved
 * credential for a field that sets somebody else's, and does not offer to remember what is typed
 * here as though it were theirs.
 *
 * There is no activation control. An account is created able to sign in, because an account
 * created deactivated is one somebody has to remember to come back and switch on — and the
 * detail page can deactivate it a moment later if that is really what was wanted.
 */
function CreateAccount() {
  return (
    // Styled as an `AdminSection` rather than wrapped in one: a disclosure needs `<details>` and
    // `<summary>` to be the elements that open it, and a section component that rendered those
    // would be a section component with a browser primitive inside it. The plate is the same —
    // `rounded-md border border-line bg-ink-2` — which is what the consolidation was about.
    <details className="overflow-hidden rounded-md border border-line bg-ink-2">
      <summary className="cursor-pointer px-5 py-4 text-[13px] font-medium text-bone select-none">
        Create an account
      </summary>

      <form
        action="/admin/create-user"
        method="post"
        className="border-t border-hair px-5 py-5"
      >
        <p className="max-w-[68ch] text-[13px] leading-relaxed text-muted">
          The password is set here and told to the person by whoever creates the account. This
          deployment sends no mail, so there is no invitation and no reset link — an account that
          needs a new password gets one from its own page.
        </p>

        <div className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,2fr)_minmax(0,2fr)_minmax(0,1fr)]">
          <div>
            <label
              htmlFor="new-account-email"
              className="block text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
            >
              Email
            </label>
            <input
              id="new-account-email"
              name="email"
              type="email"
              required
              autoComplete="off"
              className="mt-1.5 w-full rounded-md border border-line bg-ink px-2.5 py-1.5 font-mono text-[12px] text-bone transition-colors duration-150 hover:border-rule"
            />
          </div>

          <div>
            <label
              htmlFor="new-account-password"
              className="block text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
            >
              Password
            </label>
            <input
              id="new-account-password"
              name="password"
              type="password"
              required
              minLength={MINIMUM_PASSWORD_LENGTH}
              autoComplete="new-password"
              className="mt-1.5 w-full rounded-md border border-line bg-ink px-2.5 py-1.5 font-mono text-[12px] text-bone transition-colors duration-150 hover:border-rule"
            />
            <p className="mt-1.5 text-[11px] text-muted">
              At least {MINIMUM_PASSWORD_LENGTH} characters.
            </p>
          </div>

          <div>
            <label
              htmlFor="new-account-role"
              className="block text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
            >
              Role
            </label>
            <select
              id="new-account-role"
              name="role"
              defaultValue={USER_ROLE_USER}
              className="mt-1.5 w-full rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule"
            >
              <option value={USER_ROLE_USER}>User</option>
              <option value={USER_ROLE_ADMIN}>Admin</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="mt-4 rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          Create account
        </button>
      </form>
    </details>
  );
}

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/** One account: who it is, what it may do, and — unless it is the reader's — the two controls. */
function AccountRow({ account, self }: { account: AdminAccount; self: boolean }) {
  return (
    <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,auto)] items-center gap-6 border-t border-hair px-5 py-4">
      <div className="min-w-0">
        {/* Monospace: an address is a machine value the reader has to be able to check
            character by character, which is the same reason the URL field on the workspace is
            set this way and its label is not.

            It is the link to the account's own page rather than a separate "view" control in
            the last column, because the address is what identifies the row — the thing an
            operator is already pointing at when they want to see more of it. */}
        <Link
          href={adminAccountPath(account.id)}
          className="block truncate font-mono text-[13px] text-bone underline-offset-2 hover:underline"
          title={account.email}
        >
          {account.email}
        </Link>
        {/* The API's own timestamp, shown as stored rather than reformatted into a local
            rendering the record does not hold — the convention the case log already follows. */}
        <div className="mt-1 font-mono text-[11px] text-muted">{account.created_at}</div>
      </div>

      <div className="text-[13px] text-bone">
        {account.role === USER_ROLE_ADMIN ? "Admin" : "User"}
      </div>

      <div>
        <Activity active={account.is_active} />
      </div>

      <div className="flex flex-wrap items-center justify-end gap-2">
        {self ? (
          // Not a disabled control and not a greyed-out button: a sentence saying why there is
          // nothing here. A disabled control invites the reader to work out how to enable it,
          // and this one never can be — the rule is about who they are, not about what state
          // the page is in.
          //
          // The link beside it is the one thing a reader may do to their own account from here.
          // The detail page draws the password reset even for one's own account, because unlike
          // a role change that is not a change nobody can undo.
          <>
            <span className="text-[12px] text-muted">
              Your own account — another administrator must change it.
            </span>
            <Link
              href={adminAccountPath(account.id)}
              className="text-[12px] text-muted underline hover:text-bone"
            >
              Open
            </Link>
          </>
        ) : (
          <>
            <RoleControl account={account} />
            <ActivityControl account={account} />
          </>
        )}
      </div>
    </div>
  );
}

/**
 * The column headings.
 *
 * Always drawn, and always in columns. Until R8-T9 the whole table collapsed to one column below
 * `sm` and these headings were hidden, each row restating its own labels — which meant the
 * column alignment an operator is scanning for disappeared on exactly the screen where they are
 * least able to hold the shape of the table in their head. It scrolls now. See
 * `docs/ui-guidance.md`.
 */
function AccountHeader() {
  return (
    <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,auto)] gap-6 px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
      <div>Account</div>
      <div>Role</div>
      <div>Status</div>
      <div className="text-right">Change</div>
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

export default async function Admin({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const [user, result] = await Promise.all([fetchSession(), fetchAccounts()]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and this read — plus the narrowing
  // that lets everything below read `user` as present. It is not the access control: the API
  // refused the listing before this line, and what the reader gets here is somewhere useful to
  // go rather than an empty page they cannot explain.
  if (user === null || (!result.ok && result.unauthenticated)) {
    redirect(LOGIN_PATH);
  }

  const error = singleParam(params.error);
  const updated = singleParam(params.updated);
  const createdAccount = singleParam(params.created);

  return (
    <>
      <AdminPageHeader
        title="Accounts"
        description="Every account in this deployment, including the ones that can no longer sign in. Changing a role grants or withdraws access to this surface; deactivating an account ends its ability to sign in and leaves the analyses it submitted on file."
        actions={
          // A reload of this page, and deliberately a link rather than a button: the table is a
          // view of state another administrator can change, and there is no JavaScript here to
          // notice when they do. The cross-links that used to sit beside it are the sidebar now.
          <Link
            href={ADMIN_PATH}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            Refresh
          </Link>
        }
      />

      {/* The outcome of the last change, read out of the query string because the controls are
          plain forms and the answer arrives as a redirect. The error text is the API's own
          client-facing sentence — which rule the change broke — rendered as text. */}
      {error && (
        <div className="mt-5">
          <AdminAlert tone="error">{error}</AdminAlert>
        </div>
      )}
      {updated !== null && !error && (
        <div className="mt-5">
          <AdminAlert tone="success">
            Account updated
            {updated ? <span className="font-mono"> · {updated.slice(0, 8)}</span> : null}.
          </AdminAlert>
        </div>
      )}
      {createdAccount !== null && !error && (
        <div className="mt-5">
          <AdminAlert tone="success">
            Account created. The password is not recoverable from anywhere — tell the person what
            it is, or set a new one from{" "}
            <Link
              href={adminAccountPath(createdAccount)}
              className="underline hover:text-bone"
            >
              their account page
            </Link>
            .
          </AdminAlert>
        </div>
      )}

      <div className="mt-5 space-y-5">
        <CreateAccount />

        {!result.ok ? (
          <AdminAlert tone="error">{result.error}</AdminAlert>
        ) : (
          <AdminSection bleed scroll>
            {/* Wide enough for the four columns to keep their proportions; below that the
                wrapper scrolls rather than the row restacking. */}
            <div className="min-w-[860px]">
              <AccountHeader />
              {result.accounts.map((account) => (
                <AccountRow
                  key={account.id}
                  account={account}
                  // Compared by id, which is what the session resolved to. Comparing emails
                  // would compare two normalizations, and the one rendered here could be stale.
                  self={account.id === user.id}
                />
              ))}
            </div>
          </AdminSection>
        )}
      </div>
    </>
  );
}
