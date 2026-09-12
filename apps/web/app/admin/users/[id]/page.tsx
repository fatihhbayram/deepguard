/**
 * One account: what it is, what it may do, and the three things an administrator can change.
 *
 * Reached from a row on `/admin`, which stays the single source of truth for the account list —
 * there is no `/admin/users` listing and there must not be one. What this screen adds over a row
 * is everything a row has no space for: the address in full and editable, the password reset,
 * and the reason there is no delete, written out rather than implied by its absence.
 *
 * **The removal action on this page is "Deactivate", and it is not a euphemism for a delete
 * that was too hard to write.** `media_files.uploader_id` references `users.id` with
 * `ON DELETE RESTRICT`, and the audit log and the review table both keep a user's id so that who
 * did what survives. Removing an account would therefore fail on a constraint for anybody who
 * has ever uploaded anything, and destroy attribution for everybody else. Deactivation ends
 * every way in — `authenticate` folds `is_active` into the sign-in lookup and `session_user`
 * folds it into session resolution — while leaving the forensic record intact. The card below
 * says that in those words, because an operator who cannot find a Delete button will otherwise
 * assume one is missing.
 *
 * **Three separate forms, not one.** The account form changes the address, the role and the
 * activation; the deactivation button is its own form so it is one click rather than a change to
 * a select and a save; the reset is its own form because it posts somewhere else entirely. Each
 * submits only what it carries, which is what lets the API treat an absent field as "leave it
 * alone" — a single form restating every value would write back whatever this page last rendered
 * over a change another administrator may have made a second ago.
 *
 * **The reader's own account gets no controls.** An administrator may not change their own role
 * or activation — it is the one change nobody can undo for them — and the API refuses it with a
 * 400. This page does not draw a control for a change that would be refused. That is
 * presentation and not enforcement: the rule is the API's, it holds on every request, and a
 * reader who posted the form by hand would still be refused. The password reset *is* drawn on
 * their own account, because that one is allowed — see the card.
 *
 * A Server Component with no client-side code at all. The controls are plain HTML forms, the
 * same as the account list's, so the page works with JavaScript disabled and the outcome of a
 * save arrives as a redirect carrying it in the query string. The password typed into the reset
 * form travels inwards in a POST body and nothing comes back but a status, so no credential is
 * ever put in the query string this page reads its feedback out of.
 */

import Link from "next/link";
import { notFound, redirect } from "next/navigation";

import { fetchSession } from "../../../analysis";
import { LOGIN_PATH, USER_ROLE_ADMIN, USER_ROLE_USER, adminAccountPath } from "../../../session";
import { AdminAlert } from "../../components/AdminAlert";
import { AdminPageHeader } from "../../components/AdminPageHeader";
import { AdminSection } from "../../components/AdminSection";
import { AdminStatusBadge } from "../../components/AdminStatusBadge";
import { AdminAccount, MINIMUM_PASSWORD_LENGTH, fetchAccount } from "../../users";

/* ------------------------------------------------------------------ *
 * Primitives
 * ------------------------------------------------------------------ */

/*
 * The private `Legend`, `Heading`, `Alert` and `Activity` this file carried — the seventh copy of
 * the first three, which is where the count the screens had been passing along came to rest —
 * are `AdminPageHeader`, `AdminAlert` and `AdminStatusBadge` in `../../components` since R8-T9.
 *
 * `Fact` stays local. It is on two screens, this one and the case review, and two is not three.
 * Recorded rather than left implicit, which is the habit that got the extraction above made.
 */

/** One labelled fact, as a definition-list pair. */
function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase">
        {label}
      </dt>
      <dd className="mt-1 text-[13px] text-bone">{children}</dd>
    </div>
  );
}

/** Whether an account can sign in, as a chip. Two states and no third, because `is_active` is a
 *  boolean and there is no pending, invited or suspended column to name. */
function Activity({ active }: { active: boolean }) {
  return (
    <AdminStatusBadge tone={active ? "neutral" : "negative"} dot>
      {active ? "Active" : "Deactivated"}
    </AdminStatusBadge>
  );
}

/** The class every text input and select on this page carries. Written once. */
const FIELD =
  "mt-1.5 w-full rounded-md border border-line bg-ink px-2.5 py-1.5 text-[12px] text-bone transition-colors duration-150 hover:border-rule";

/** And every submit button. */
const BUTTON =
  "rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone";

/* ------------------------------------------------------------------ *
 * The cards
 * ------------------------------------------------------------------ */

/** What the account is, as stored. No control on this card — the ones below do the changing. */
function Identity({ account }: { account: AdminAccount }) {
  return (
    <AdminSection title="Account">
      <dl className="mt-4 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Fact label="Email">
          {/* Monospace: an address is a machine value the reader has to be able to check
              character by character, the same reason the list sets it this way. */}
          <span className="font-mono text-[12px] break-all">{account.email}</span>
        </Fact>
        <Fact label="Identifier">
          <span className="font-mono text-[11px] break-all text-muted">{account.id}</span>
        </Fact>
        <Fact label="Role">{account.role === USER_ROLE_ADMIN ? "Admin" : "User"}</Fact>
        <Fact label="Status">
          <Activity active={account.is_active} />
        </Fact>
        <Fact label="Created">
          {/* The API's own timestamp, shown as stored rather than reformatted into a local
              rendering the record does not hold — the convention every table here follows. */}
          <span className="font-mono text-[11px] text-muted">{account.created_at}</span>
        </Fact>
      </dl>
    </AdminSection>
  );
}

/**
 * The address, the role and the activation, in one form that submits all three.
 *
 * All three travel together here, unlike on the list where each control is its own submission.
 * That is safe because this form is rendered from a read of *this* account taken on the same
 * request: the values in the fields are the values the API just reported, so a save that carries
 * them all writes back what is already there for the ones nobody touched. The list's rows are
 * rendered from a listing that is shown alongside many other accounts and is reloaded far less
 * often, which is why its controls stay separate.
 *
 * `defaultValue` rather than `value`: these are uncontrolled inputs in a server-rendered form,
 * and a `value` with no `onChange` is a field React will not let the reader alter.
 *
 * `return_to` is the token `/admin/update-user` reads to decide where to send the browser back.
 * It is not a path and cannot become one — see that handler.
 */
function AccountForm({ account }: { account: AdminAccount }) {
  return (
    <AdminSection
      title="Change this account"
      description="Changing the address renames the account: the old one stops working immediately and the new one starts. It does not sign the person out — a session is bound to the account, not to the address. Changing the role grants or withdraws access to this administrative surface."
    >
      <form
        action="/admin/update-user"
        method="post"
        className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]"
      >
        <input type="hidden" name="user_id" value={account.id} />
        <input type="hidden" name="return_to" value="detail" />

        <div>
          <label
            htmlFor="account-email"
            className="block text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
          >
            Email
          </label>
          <input
            id="account-email"
            name="email"
            type="email"
            required
            defaultValue={account.email}
            autoComplete="off"
            className={`${FIELD} font-mono`}
          />
        </div>

        <div>
          <label
            htmlFor="account-role"
            className="block text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
          >
            Role
          </label>
          <select
            id="account-role"
            name="role"
            defaultValue={account.role}
            className={FIELD}
          >
            <option value={USER_ROLE_USER}>User</option>
            <option value={USER_ROLE_ADMIN}>Admin</option>
          </select>
        </div>

        <div className="sm:col-span-2">
          <button type="submit" className={BUTTON}>
            Save changes
          </button>
        </div>
      </form>
    </AdminSection>
  );
}

/**
 * Deactivation, presented as what it is: this system's removal, and a reversible one.
 *
 * A button carrying a hidden value rather than a checkbox, because an unchecked checkbox sends
 * nothing at all — the form would arrive with no `is_active` field and the change would be a
 * no-op in exactly the case the operator meant to deactivate somebody. The value is the state
 * being asked for, so what the form says and what the button is labelled cannot drift apart.
 *
 * The prose is the point of this card. It is the only place in the application that says why
 * there is no Delete, and an operator who does not read it will go looking for one.
 */
function Lifecycle({ account }: { account: AdminAccount }) {
  return (
    <AdminSection
      title="Ending access"
      description="Deactivating is how an account is removed here, and there is no delete. An account that has uploaded media is referenced by that media, and the audit log and any case reviews keep its identifier so that who did what stays answerable — deleting the row would either fail outright or erase that history. A deactivated account cannot sign in and its existing sessions stop working, while everything it submitted remains attributable. It can be reactivated from this page at any time."
    >
      <form action="/admin/update-user" method="post" className="mt-4">
        <input type="hidden" name="user_id" value={account.id} />
        <input type="hidden" name="return_to" value="detail" />
        <input type="hidden" name="is_active" value={account.is_active ? "false" : "true"} />
        <button type="submit" className={BUTTON}>
          {account.is_active ? "Deactivate account" : "Reactivate account"}
        </button>
      </form>
    </AdminSection>
  );
}

/**
 * The password reset.
 *
 * Drawn on every account including the reader's own, unlike the role and activation controls.
 * The self-modification rule exists for the changes nobody can undo for them, and this is not
 * one: the operator chose the new password and can sign in with it immediately. What it does do
 * is end their session along with all the others, so the card says so — an operator bounced to
 * a login screen with no warning will believe something broke.
 *
 * `minLength` is a courtesy and not a check: the API enforces `MINIMUM_PASSWORD_LENGTH` whatever
 * the browser did. `autoComplete="new-password"` so the browser neither offers the
 * administrator's own saved credential for a field that sets somebody else's nor offers to
 * remember what is typed here as though it were theirs.
 */
function PasswordReset({ account, self }: { account: AdminAccount; self: boolean }) {
  return (
    <AdminSection
      title="Reset the password"
      description={
        <p>
          Setting a password here replaces the old one and immediately ends every session this
          account has open, so anyone signed in as it is signed out. The password is stored only
          as a hash and cannot be read back from anywhere — tell the person what it is, because
          nothing in this deployment sends mail.
          {self
            ? " This is your own account: saving will sign you out, and you will need to sign back in with the password you set here."
            : ""}
        </p>
      }
    >
      <form
        action="/admin/reset-password"
        method="post"
        className="mt-4 flex flex-wrap items-end gap-4"
      >
        <input type="hidden" name="user_id" value={account.id} />

        <div className="min-w-[16rem] flex-1">
          <label
            htmlFor="account-password"
            className="block text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
          >
            New password
          </label>
          <input
            id="account-password"
            name="password"
            type="password"
            required
            minLength={MINIMUM_PASSWORD_LENGTH}
            autoComplete="new-password"
            className={`${FIELD} font-mono`}
          />
          <p className="mt-1.5 text-[11px] text-muted">
            At least {MINIMUM_PASSWORD_LENGTH} characters.
          </p>
        </div>

        <button type="submit" className={`${BUTTON} mb-7`}>
          Reset password
        </button>
      </form>
    </AdminSection>
  );
}

/* ------------------------------------------------------------------ *
 * Page
 * ------------------------------------------------------------------ */

/** One query-string value, or null. A repeated parameter is not an outcome. */
function singleParam(value: string | string[] | undefined): string | null {
  return typeof value === "string" ? value : null;
}

export default async function AdminAccountDetail({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { id } = await params;
  const query = await searchParams;

  const [user, result] = await Promise.all([fetchSession(), fetchAccount(id)]);

  // The session the API would not accept. `admin/layout.tsx` has already turned away a reader
  // with no session and one whose role is not administrator, so what is left for this is the
  // narrow case of a session that expired between the guard and this read — plus, since R8-T8,
  // the case an administrator produces deliberately by resetting their own password, which
  // revokes the session they are holding. It is not the access control: the API refused the read
  // before this line, and what the reader gets here is somewhere useful to go.
  if (user === null || (!result.ok && result.unauthenticated)) {
    redirect(LOGIN_PATH);
  }

  // An id that names no account. Next's own not-found rendering rather than a sentence on an
  // otherwise empty page, because there is no account for the rest of this screen to be about.
  if (!result.ok && result.missing) {
    notFound();
  }

  const error = singleParam(query.error);
  const updated = singleParam(query.updated);
  const reset = singleParam(query.reset);

  return (
    <>
      <AdminPageHeader
        title="Account"
        description="One account, and everything an administrator can change about it. The list of every account in this deployment is on the accounts page; this screen is about this one."
        actions={
          // A reload of this page, and deliberately a link rather than a button: the account is
          // state another administrator can change, and there is no JavaScript here to notice
          // when they do. The link back to the account list is the sidebar now.
          <Link
            href={adminAccountPath(id)}
            className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
          >
            Refresh
          </Link>
        }
      />

      {/* The outcome of the last change, read out of the query string because the controls are
          plain forms and the answer arrives as a redirect. The error text is the API's own
          client-facing sentence — which rule the change broke, or that the address is taken —
          rendered as text. */}
      {error && (
        <div className="mt-5">
          <AdminAlert tone="error">{error}</AdminAlert>
        </div>
      )}
      {reset !== null && !error && (
        <div className="mt-5">
          <AdminAlert tone="success">
            Password reset. Every session this account had open has been ended.
          </AdminAlert>
        </div>
      )}
      {updated !== null && !error && reset === null && (
        <div className="mt-5">
          <AdminAlert tone="success">Account updated.</AdminAlert>
        </div>
      )}

      <div className="mt-5 space-y-5">
        {!result.ok ? (
          <AdminAlert tone="error">{result.error}</AdminAlert>
        ) : (
          <>
            <Identity account={result.account} />

            {/* Compared by id, which is what the session resolved to. Comparing addresses would
                compare two normalizations, and the one rendered here could be stale. */}
            {result.account.id === user.id ? (
              // Not a disabled control and not a greyed-out button: a sentence saying why there
              // is nothing here. A disabled control invites the reader to work out how to enable
              // it, and this one never can be — the rule is about who they are.
              <AdminSection
                title="Change this account"
                description="This is your own account. Its address, role and activation must be changed by another administrator — an administrator who demoted or deactivated themselves could no longer reach this page to put it back. You can still reset your own password below."
              />
            ) : (
              <>
                <AccountForm account={result.account} />
                <Lifecycle account={result.account} />
              </>
            )}

            <PasswordReset
              account={result.account}
              self={result.account.id === user.id}
            />
          </>
        )}
      </div>
    </>
  );
}
