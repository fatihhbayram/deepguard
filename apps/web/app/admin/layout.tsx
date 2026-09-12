/**
 * The door to the administrative workspace (R8-T1).
 *
 * Two refusals, in the order the answers arrive. No session at all is sent to sign in, which
 * is the same answer `/app` gives and has to be: a reader with no session cannot be told
 * whether this address exists for anyone. A session that is not an administrator's is sent
 * to `/app` — somewhere they are entitled to be, rather than a dead end — and told nothing
 * about what is behind this address.
 *
 * The role is compared against `USER_ROLE_ADMIN`, the one spelling of it this application
 * has, rather than a `"admin"` written here. A magic string in a guard is a guard that stops
 * working the day the API changes the word and says nothing when it does; the comparison is
 * also case-sensitive, and the API spells the role in capitals.
 *
 * Since R8-T3 it also carries the surface's own navigation, which is the one piece of markup
 * it owns. There are five administrative screens now — the accounts, the detection queue, the
 * operational summary, the audit log and the API keys — and a link to each from a place that is
 * on all of them beats each page linking to the others, which is what it was doing while there
 * were two. The last of them is the only one about the public API rather than about this
 * application, and it is in the same strip because it is the same audience: one operator, one
 * surface, one place to find everything they administer.
 * It is a strip of links and not a shell: the pages below still own their own `main`, their own
 * heading and their own ground.
 *
 * Like the workspace guard, this is a signpost rather than a lock. The privileged data an
 * administrator can read is privileged in the API — `app/web_auth.py` makes the same check
 * on every request that actually returns any of it — and nothing behind this layout is
 * reachable because this layout let a reader through. That is what makes a role check in
 * React acceptable here at all: it decides what to draw, not what may be read.
 */

import { redirect } from "next/navigation";

import Link from "next/link";

import { fetchSession } from "../analysis";
import {
  ADMIN_ANALYTICS_PATH,
  ADMIN_API_KEYS_PATH,
  ADMIN_AUDIT_PATH,
  ADMIN_JOBS_PATH,
  ADMIN_PATH,
  LOGIN_PATH,
  USER_ROLE_ADMIN,
  WORKSPACE_PATH,
} from "../session";

// This subtree is rendered per request, always.
//
// Not a preference. `fetchSession()` reads the session cookie, and reading a cookie is what
// normally tells Next this route cannot be prerendered — but that signal is an exception
// thrown from `cookies()`, and `fetchSession` wraps its whole body in a `try/catch` so that
// an unreachable API reads as "no session" rather than a crash. The catch swallows the
// bailout along with it. A segment whose page takes no other dynamic input therefore looks
// static to the build, and the build resolves the guard once, with no cookie jar, and freezes
// the answer: the first version of `/admin` was prerendered as a permanent 307 to the sign-in
// page, which is what every reader would have got, administrator or not.
//
// So the route states outright what the swallowed exception can no longer say. It is stated
// on the layout because segment config applies to everything beneath it, and because the
// reason is the guard — not any one page below it.
export const dynamic = "force-dynamic";


export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await fetchSession();

  if (user === null) {
    redirect(LOGIN_PATH);
  }

  if (user.role !== USER_ROLE_ADMIN) {
    redirect(WORKSPACE_PATH);
  }

  return (
    <>
      {/* Plain links, and no marking of which one the reader is on. Knowing that needs the
          current path, which a server component does not have — `usePathname` would make this
          guard a client component, and the guard is the reason this file exists. The pages
          below are titled, which is what tells a reader where they are. */}
      <nav
        aria-label="Administration"
        className="flex items-center gap-5 border-b border-hair px-4 py-3 sm:px-8"
      >
        <Link href={ADMIN_PATH} className="text-sm text-muted hover:text-bone">
          Accounts
        </Link>
        <Link href={ADMIN_JOBS_PATH} className="text-sm text-muted hover:text-bone">
          Jobs
        </Link>
        <Link href={ADMIN_ANALYTICS_PATH} className="text-sm text-muted hover:text-bone">
          Analytics
        </Link>
        <Link href={ADMIN_AUDIT_PATH} className="text-sm text-muted hover:text-bone">
          Audit
        </Link>
        <Link href={ADMIN_API_KEYS_PATH} className="text-sm text-muted hover:text-bone">
          API keys
        </Link>
      </nav>
      {children}
    </>
  );
}
