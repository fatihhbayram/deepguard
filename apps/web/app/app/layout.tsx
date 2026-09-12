/**
 * The door to the authenticated workspace (R8-T1).
 *
 * Everything under `/app` — the dashboard and every report — renders inside this layout, and
 * a layout is a server component that runs before any page below it fetches anything. That
 * ordering is the whole point of putting the check here: `redirect()` thrown from a layout
 * aborts the render of the nested page before its data calls are made, so one guard covers a
 * route subtree instead of every page in it remembering to guard itself. Before this the
 * check lived in each page, which is one page away from being forgotten.
 *
 * It is a signpost, not the lock. What a session may read is decided by the API and by
 * nothing here: `fetchSession()` asks `/auth/me` and believes the answer, every listing this
 * subtree renders is narrowed by the API's own `WHERE` clause, and a reader who got past
 * this guard with a session the API later refuses still sees nothing that is not theirs. A
 * web layout that decided authorization would be a second opinion in a place anyone can edit.
 *
 * The cost is one extra `/auth/me` per workspace render, and it is a real second call rather
 * than a free one: Next memoizes identical `fetch`es within a render pass, but `fetchSession`
 * passes an `AbortSignal` for its timeout, and a signal opts that request out of memoization.
 * Two indexed lookups on a page that already makes three calls is the price of the guard
 * being in one place; collapsing it back to one would mean memoizing `fetchSession` itself.
 *
 * No shell markup of its own. The dashboard and the report are two different surfaces — one
 * is an instrument panel on the dark ground, the other a printable document on a light one —
 * and a wrapper here would have to be neutral enough to sit behind both, which is a wrapper
 * that adds nothing. The root layout already owns the page frame.
 */

import { redirect } from "next/navigation";

import { fetchSession } from "../analysis";
import { LOGIN_PATH } from "../session";

// This subtree is rendered per request, always.
//
// Not a preference. `fetchSession()` reads the session cookie, and reading a cookie is what
// normally tells Next this route cannot be prerendered — but that signal is an exception
// thrown from `cookies()`, and `fetchSession` wraps its whole body in a `try/catch` so that
// an unreachable API reads as "no session" rather than a crash. The catch swallows the
// bailout along with it. A segment whose page takes no other dynamic input therefore looks
// static to the build, and the build resolves the guard once, with no cookie jar, and freezes
// the answer: the first version of the admin shell was prerendered as a permanent 307 to the sign-in
// page, which is what every reader would have got, administrator or not.
//
// So the route states outright what the swallowed exception can no longer say. It is stated
// on the layout because segment config applies to everything beneath it, and because the
// reason is the guard — not any one page below it.
export const dynamic = "force-dynamic";


export default async function WorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await fetchSession();

  if (user === null) {
    redirect(LOGIN_PATH);
  }

  return children;
}
