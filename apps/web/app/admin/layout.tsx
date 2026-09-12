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
 * Since R8-T9 it also owns the surface's shell: the navigation rail, the mobile drawer that
 * holds it on a narrow screen, and the `<main>` every page below renders into. That is a change
 * of ownership rather than an addition. Until R8-T9 the layout carried a horizontal strip of
 * five links and each page owned its own `main`, its own gutter and its own footer of
 * cross-links back to the other screens — seven pages each deciding a page's padding, and seven
 * places for that decision to drift, which it had. The pages return their content now.
 *
 * The strip is gone for the reason `components/AdminSidebar.tsx` states: five links across the
 * top could list the screens but could never group them, and by R8-T8 it was presenting the
 * account table, the detection queue, a counts page, an append-only log and the public API's
 * credentials as five of a kind.
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
import { AdminMobileNav } from "./components/AdminMobileNav";
import { AdminSidebar } from "./components/AdminSidebar";

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
    <div className="flex min-h-0 flex-1">
      {/* The rail, on the widths that have room for it beside a table. `lg` rather than `sm`:
          the widest table in this surface is the audit log at 900px, and a 240px rail taken out
          of a tablet's 768 leaves that table scrolling when the viewport could have shown it. */}
      <div className="hidden w-60 shrink-0 border-r border-line lg:block">
        {/* Sticky and full-height, so the rail stays put while a long log scrolls past it. */}
        <div className="sticky top-0 h-dvh overflow-y-auto">
          <AdminSidebar />
        </div>
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Below `lg` the same rail is rendered into the drawer. It is rendered here, on the
            server, and handed to the client component as children — see `AdminMobileNav`, which
            holds one boolean and knows nothing about what it is holding open. */}
        <div className="lg:hidden">
          <AdminMobileNav>
            <AdminSidebar />
          </AdminMobileNav>
        </div>

        {/* The one `main` on this surface. Every page below returns its content into it, which
            is what makes the page gutter a single decision instead of seven. */}
        <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 py-8 sm:px-6 lg:px-8">
          {children}
        </main>
      </div>
    </div>
  );
}
