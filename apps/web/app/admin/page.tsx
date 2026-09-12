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
 */

import Link from "next/link";
import { redirect } from "next/navigation";

import { fetchSession } from "../analysis";
import {
  ADMIN_PATH,
  LOGIN_PATH,
  USER_ROLE_ADMIN,
  USER_ROLE_USER,
  WORKSPACE_PATH,
} from "../session";
import { AdminAccount, fetchAccounts } from "./users";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * The workspace's `Legend`, `Heading` and `Alert` restated rather than imported.
 *
 * They live inside `app/app/page.tsx` as local functions and are not exported. Exporting them
 * from a page module to reach them from here would make one route's file a component library
 * for another's — a dependency between two screens that have no other reason to know about
 * each other, and one that makes a change to the workspace's heading a change to this page.
 * Three small elements are cheaper restated than coupled. If a fourth screen wants them, that
 * is the moment they move to a shared module, not before.
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

/** The outcome of the last change, stated in the page's own voice. */
function Alert({
  tone,
  children,
}: {
  tone: "error" | "success";
  children: React.ReactNode;
}) {
  const styles =
    tone === "error"
      ? { field: "border-rose-500/40 bg-rose-500/10 text-rose-200", dot: "bg-rose-400" }
      : { field: "border-accent/40 bg-accent/10 text-bone", dot: "bg-accent" };

  return (
    <p
      role="status"
      className={`flex items-start gap-3 rounded-md border px-4 py-3 text-[13px] leading-relaxed ${styles.field}`}
    >
      <span aria-hidden className={`mt-1.5 size-1.5 shrink-0 rounded-full ${styles.dot}`} />
      <span>{children}</span>
    </p>
  );
}

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
    <span
      className={`inline-flex items-center gap-1.5 rounded-sm px-2 py-1 text-[11px] font-medium tracking-[0.08em] uppercase ${
        active ? "bg-chip text-bone" : "bg-rose-500/10 text-rose-200"
      }`}
    >
      <span
        aria-hidden
        className={`size-1.5 rounded-full ${active ? "bg-accent" : "bg-rose-400"}`}
      />
      {active ? "Active" : "Deactivated"}
    </span>
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

/* ------------------------------------------------------------------ *
 * The table
 * ------------------------------------------------------------------ */

/** One account: who it is, what it may do, and — unless it is the reader's — the two controls. */
function AccountRow({ account, self }: { account: AdminAccount; self: boolean }) {
  return (
    <div className="grid grid-cols-1 gap-3 border-t border-hair px-5 py-4 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,auto)] sm:items-center sm:gap-6">
      <div className="min-w-0">
        {/* Monospace: an address is a machine value the reader has to be able to check
            character by character, which is the same reason the URL field on the workspace is
            set this way and its label is not. */}
        <div className="truncate font-mono text-[13px] text-bone" title={account.email}>
          {account.email}
        </div>
        {/* The API's own timestamp, shown as stored rather than reformatted into a local
            rendering the record does not hold — the convention the case log already follows. */}
        <div className="mt-1 font-mono text-[11px] text-muted">{account.created_at}</div>
      </div>

      <div>
        <span className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase sm:hidden">
          Role
        </span>
        <div className="text-[13px] text-bone sm:mt-0">
          {account.role === USER_ROLE_ADMIN ? "Admin" : "User"}
        </div>
      </div>

      <div>
        <Activity active={account.is_active} />
      </div>

      <div className="flex flex-wrap items-center gap-2 sm:justify-end">
        {self ? (
          // Not a disabled control and not a greyed-out button: a sentence saying why there is
          // nothing here. A disabled control invites the reader to work out how to enable it,
          // and this one never can be — the rule is about who they are, not about what state
          // the page is in.
          <span className="text-[12px] text-muted">
            Your own account — another administrator must change it.
          </span>
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

/** The column headings, on the layouts wide enough to have columns. */
function AccountHeader() {
  return (
    <div className="hidden grid-cols-[minmax(0,2fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,auto)] gap-6 px-5 py-3 text-[11px] font-medium tracking-[0.08em] text-muted uppercase sm:grid">
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

  return (
    <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 py-14 sm:px-8">
      <Legend>Administration</Legend>
      <Heading>Accounts</Heading>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed text-muted">
        Every account in this deployment, including the ones that can no longer sign in.
        Changing a role grants or withdraws access to this page itself; deactivating an account
        ends its ability to sign in and leaves the analyses it submitted on file.
      </p>

      {/* The outcome of the last change, read out of the query string because the controls are
          plain forms and the answer arrives as a redirect. The error text is the API's own
          client-facing sentence — which rule the change broke — rendered as text. */}
      {error && (
        <div className="mt-6">
          <Alert tone="error">{error}</Alert>
        </div>
      )}
      {updated !== null && !error && (
        <div className="mt-6">
          <Alert tone="success">
            Account updated
            {updated ? <span className="font-mono"> · {updated.slice(0, 8)}</span> : null}.
          </Alert>
        </div>
      )}

      <div className="mt-6">
        {!result.ok ? (
          <Alert tone="error">{result.error}</Alert>
        ) : (
          <div className="overflow-hidden rounded-lg border border-line bg-ink-2">
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
        )}
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-5">
        <Link
          href={WORKSPACE_PATH}
          className="text-sm text-muted underline hover:text-bone"
        >
          ← Back to workspace
        </Link>
        {/* A reload of this page, and deliberately a link rather than a button: the table is
            a view of state another administrator can change, and there is no JavaScript here
            to notice when they do. */}
        <Link href={ADMIN_PATH} className="text-sm text-muted underline hover:text-bone">
          Refresh
        </Link>
      </div>
    </main>
  );
}
