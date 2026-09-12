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
 * Like the workspace guard, this is a signpost rather than a lock. The privileged data an
 * administrator can read is privileged in the API — `app/web_auth.py` makes the same check
 * on every request that actually returns any of it — and nothing behind this layout is
 * reachable because this layout let a reader through. That is what makes a role check in
 * React acceptable here at all: it decides what to draw, not what may be read.
 */

import { redirect } from "next/navigation";

import { fetchSession } from "../analysis";
import { LOGIN_PATH, USER_ROLE_ADMIN, WORKSPACE_PATH } from "../session";

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

  return children;
}
