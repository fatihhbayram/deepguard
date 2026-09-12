/**
 * The console's navigation, as a rail.
 *
 * It replaces the horizontal strip `admin/layout.tsx` carried since R8-T3. Five links across the
 * top of the page was the right shape while there were two screens and an acceptable one while
 * there were five; what it could never do is group them, and by R8-T8 the strip was listing the
 * account table, the detection queue, a counts page, an append-only log and the public API's
 * credentials as though those were five of a kind.
 *
 * **The groups are what an operator is doing, not what the API is.** Operations is the day's
 * work — who has an account, what the queue is doing, what the week looked like. Governance is
 * the record of what was done and by whom. Access is the other surface's credentials, which is a
 * different audience's way in and the only thing here that is not about this application.
 *
 * **Two routes are deliberately absent**: `/admin/analyses/[id]` and `/admin/users/[id]`. Neither
 * has a listing to land on — the account list *is* `/admin`, and there is no reviews page, only
 * the review of a particular analysis — so the way into both is a row that already names the
 * record. A sidebar entry would have to invent a destination.
 *
 * **The current item is marked, and `AdminNavLink` is how.** The rail first shipped without a
 * marker on the grounds that `usePathname` needs a client component; what that missed is that an
 * item comparing the path against *its own* `href` leaks nothing, because that href is already
 * in the rendered HTML. The table below — which items exist, how they group, what they are
 * called — stays here, on the server. See `AdminNavLink`.
 *
 * A server component, rendered twice per request: once into the desktop rail and once into the
 * drawer's children. That is a second render of a list of links, not a second fetch — this
 * component reads nothing.
 */

import Link from "next/link";

import { AdminNavLink } from "./AdminNavLink";
import {
  ADMIN_ANALYTICS_PATH,
  ADMIN_API_KEYS_PATH,
  ADMIN_AUDIT_PATH,
  ADMIN_JOBS_PATH,
  ADMIN_PATH,
  WORKSPACE_PATH,
} from "../../session";

/*
 * The route table, stated once. Every address is a constant from `../../session` rather than a
 * string written here, for the reason that module gives: a literal `"/admin/jobs"` in a
 * navigation component and another in the page it points at are two copies of one address.
 */
const GROUPS: {
  heading: string;
  items: { href: string; label: string; prefix?: string }[];
}[] = [
  {
    heading: "Operations",
    items: [
      // `/admin/users/[id]` has no entry of its own — the account list is `/admin` — so Accounts
      // owns it and stays marked while a reader is on one account. See `AdminNavLink`.
      { href: ADMIN_PATH, label: "Accounts", prefix: "/admin/users" },
      { href: ADMIN_JOBS_PATH, label: "Detection jobs", prefix: ADMIN_JOBS_PATH },
      { href: ADMIN_ANALYTICS_PATH, label: "Operational summary", prefix: ADMIN_ANALYTICS_PATH },
    ],
  },
  {
    heading: "Governance",
    items: [{ href: ADMIN_AUDIT_PATH, label: "Audit log", prefix: ADMIN_AUDIT_PATH }],
  },
  {
    heading: "Access",
    items: [{ href: ADMIN_API_KEYS_PATH, label: "API keys", prefix: ADMIN_API_KEYS_PATH }],
  },
];

export function AdminSidebar() {
  return (
    <nav aria-label="Administration" className="flex h-full flex-col px-3 py-4">
      <div className="flex items-center gap-2.5 px-3 pb-4">
        <span aria-hidden className="size-1.5 rounded-full bg-accent" />
        {/* The product's name in the UI typeface, matching the workspace's own bar: it is a
            name, not a value the pipeline produced. `Console` under it because this rail is
            only ever drawn on the administrative surface, and an operator arriving from the
            workspace should be able to tell which of the two they are in. */}
        <span className="text-[13px] font-semibold tracking-[0.14em] text-bone uppercase">
          InspectRoot
        </span>
      </div>

      <div className="flex-1 space-y-4 border-t border-hair pt-4">
        {GROUPS.map((group) => (
          <div key={group.heading}>
            {/* Compact and muted: a group heading is a filing label, not a second level of
                navigation, and at `text-[10px]` it separates the items without competing with
                them for the reader's eye. */}
            <p className="px-3 pb-1 text-[10px] font-medium tracking-[0.16em] text-muted/80 uppercase">
              {group.heading}
            </p>
            <ul className="space-y-px">
              {group.items.map((item) => (
                <li key={item.href}>
                  <AdminNavLink href={item.href} prefix={item.prefix}>
                    {item.label}
                  </AdminNavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      {/* The bridge out. Pinned to the foot of the rail rather than filed under a group,
          because it is the one link here that leaves this surface — and the counterpart to the
          `Admin console` link the workspace header draws for administrators only. Between them
          an operator can move either way without the address bar. */}
      <div className="mt-4 border-t border-hair pt-3">
        <Link
          href={WORKSPACE_PATH}
          className="block rounded-md px-3 py-1.5 text-[13px] text-muted transition-colors duration-150 hover:bg-plate hover:text-bone"
        >
          <span aria-hidden>⟵ </span>Back to workspace
        </Link>
      </div>
    </nav>
  );
}
