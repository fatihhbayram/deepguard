/**
 * The administrative workspace, which so far is only the fact that it exists (R8-T1).
 *
 * This route was created to prove one thing — that a privileged surface can be entered by an
 * administrator and by nobody else — and it deliberately does no more than that. The
 * operations an administrator will eventually run from here (accounts, audit, retries) are
 * later work, and a page that sketched them now would be a set of controls that do nothing,
 * which is worse than an empty room with a locked door.
 */

import Link from "next/link";

import { WORKSPACE_PATH } from "../session";

export default function Admin() {
  return (
    <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 py-14 sm:px-8">
      <p className="text-[11px] font-medium tracking-[0.16em] text-accent uppercase">
        Administration
      </p>
      <h1 className="mt-2.5 text-2xl font-semibold tracking-[-0.02em] text-bone">
        Admin workspace
      </h1>
      <p className="mt-3 max-w-[68ch] text-[15px] leading-relaxed text-muted">
        Reachable only by an administrator. Nothing is operated from here yet — the
        privileged tooling will be built onto this surface.
      </p>

      <Link
        href={WORKSPACE_PATH}
        className="mt-8 inline-block text-sm text-muted underline hover:text-bone"
      >
        ← Back to workspace
      </Link>
    </main>
  );
}
