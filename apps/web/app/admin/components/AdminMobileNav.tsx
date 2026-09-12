"use client";

/**
 * The drawer that holds the rail on a narrow screen. One boolean, and deliberately nothing else.
 *
 * This is the second client component in the whole application — `api-keys/create-key.tsx` is the
 * other — and it is written to the same test that one is: a client component exists only where
 * the server genuinely cannot do the job. A drawer has an open state and the server has none, so
 * the open state is here.
 *
 * **What is not here is the point of the file.** No route table, no `usePathname`, no session, no
 * role. The rail is rendered on the server by `AdminSidebar` and handed in as `children`, so the
 * bundle this ships to a browser knows that something can open and close and nothing at all about
 * what is inside it. Moving the links in here would put a second copy of the console's
 * information architecture in the one place it cannot be checked against the routes.
 *
 * Three ways out, none of which needs an effect:
 *
 * - the close button in the drawer, which takes focus when the drawer opens;
 * - the backdrop, which is a real `<button>` so it is reachable by keyboard and announced;
 * - Escape, handled by a `keydown` on the drawer itself. That works because focus is inside the
 *   drawer the moment it opens, so there is no document-level listener to attach and tear down.
 *
 * A click on any link inside closes it too, by delegation. Next keeps this layout mounted across
 * a client-side navigation, so without that the drawer would still be sitting open over the page
 * it just navigated to. Delegation rather than a callback threaded through the rail, because a
 * callback would mean the server component knew this component existed.
 */

import { useState } from "react";

export function AdminMobileNav({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <div className="flex items-center gap-3 border-b border-hair px-4 py-3">
        <button
          type="button"
          aria-expanded={open}
          aria-controls="admin-drawer"
          onClick={() => setOpen(true)}
          className="rounded-md border border-line px-2.5 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          <span aria-hidden className="mr-2">
            ☰
          </span>
          Menu
        </button>
        <span className="text-[11px] font-medium tracking-[0.16em] text-muted uppercase">
          Administration
        </span>
      </div>

      {/* Rendered only while open. A drawer translated off-screen would leave its links in the
          tab order of every page on this surface, and there is nothing to animate here anyway —
          the console does not animate. */}
      {open && (
        <div
          id="admin-drawer"
          className="fixed inset-0 z-40 flex"
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              setOpen(false);
            }
          }}
          onClick={(event) => {
            // Delegation: any link inside the rail closes the drawer behind it. `closest`
            // rather than a check on the target itself, because the link's own children are
            // what a click actually lands on.
            if ((event.target as HTMLElement).closest("a") !== null) {
              setOpen(false);
            }
          }}
        >
          <div className="flex w-72 max-w-[85vw] flex-col border-r border-line bg-ink">
            <div className="flex justify-end px-4 pt-4">
              <button
                type="button"
                autoFocus
                onClick={() => setOpen(false)}
                className="rounded-md border border-line px-2.5 py-1.5 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
              >
                Close
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
          </div>

          {/* A button rather than a div, so closing by tapping outside is also closing by
              tabbing to it and pressing Enter. */}
          <button
            type="button"
            aria-label="Close navigation"
            onClick={() => setOpen(false)}
            className="flex-1 bg-ink/80"
          />
        </div>
      )}
    </>
  );
}
