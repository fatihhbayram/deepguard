"use client";

/**
 * One item in the navigation rail, and the only thing on this surface that knows which screen
 * the reader is on.
 *
 * The rail shipped without an active marker. The reason recorded at the time was that knowing
 * the current path needs `usePathname`, which needs a client component, and that the split with
 * `AdminMobileNav` existed precisely so the route table never crossed into the browser. The
 * first half of that is still true; the second half was too broad a conclusion. **This component
 * is told its own address and nothing else.** The `href` it compares against is already in the
 * HTML it renders — it is the `<a href>` — so nothing crosses into the client that was not
 * already there. What stays on the server is the table: which items exist, how they group, and
 * what they are called, all of it still in `AdminSidebar`.
 *
 * `prefix` is the second address an item owns, for the screens that have no rail entry of their
 * own. `/admin/users/[id]` is part of Accounts and `/admin` is the account list, so Accounts
 * carries `prefix="/admin/users"` and lights up for both. It is separate from `href` rather than
 * derived from it because `/admin` is a prefix of every address in this surface — matching on it
 * would mark Accounts active on all five screens.
 *
 * The treatment is a raised ground and a two-pixel accent rule down the left edge, not an
 * outlined button. An outline in the rail reads as a control the reader can press for some
 * further effect; what this has to say is quieter — you are here — so it is said with the
 * plate the rest of the console already uses for a raised surface, a foreground lifted from
 * `muted` to `bone`, and the accent used as an indicator rather than as a fill.
 *
 * The ground is `plate` and not `chip`. `chip` is the badge ground — it is the brightest of the
 * three raised tokens, and behind a full-width rail item it stopped reading as a surface and
 * started reading as a filled button, which is the thing this comment says it is not. `plate` is
 * a step back from it; the accent rule and the `bone` foreground carry the state, and the ground
 * only has to lift the row off the rail.
 *
 * `aria-current="page"` carries the same fact to a reader who is not looking at the colour. It
 * is the attribute for this and not `aria-selected`, which belongs to widgets rather than to
 * navigation.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";

export function AdminNavLink({
  href,
  prefix,
  children,
}: {
  href: string;
  prefix?: string;
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const active =
    pathname === href ||
    (prefix !== undefined && (pathname === prefix || pathname.startsWith(`${prefix}/`)));

  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`relative block rounded-md py-1.5 pl-3 pr-2 text-[13px] transition-colors duration-150 ${
        active ? "bg-plate font-medium text-bone" : "text-muted hover:bg-plate/60 hover:text-bone"
      }`}
    >
      {active && (
        <span
          aria-hidden
          className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent"
        />
      )}
      {children}
    </Link>
  );
}
