/**
 * The public surface (R8-T1).
 *
 * Until this task `/` was the dashboard, and every visitor to the address was bounced to the
 * sign-in page. The dashboard now lives at `/app` behind a layout guard, which leaves this
 * address free to be the one thing in the product that anybody may load: it fetches nothing,
 * reads no cookie and names no account, so there is no session for it to be wrong about.
 *
 * Minimal on purpose. What belongs on a marketing page — what the product establishes, what
 * it does not, who it is for — is writing that has not been done yet, and a page of invented
 * claims about a forensic tool is the one kind of placeholder this product cannot afford. So
 * this states the name, one true sentence, and the way in.
 */

import Link from "next/link";

import { LOGIN_PATH, WORKSPACE_PATH } from "./session";

export default function Landing() {
  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-4 py-20 sm:px-8">
      <div className="flex items-center gap-2.5">
        <span aria-hidden className="size-1.5 rounded-full bg-accent" />
        {/* A name, set in the UI typeface — as it is in the dashboard header and above the
            sign-in form. The figure typeface is reserved for values the pipeline produced. */}
        <h1 className="text-[14px] font-semibold tracking-[0.14em] text-bone uppercase">
          InspectRoot
        </h1>
      </div>

      <h2 className="mt-10 max-w-[20ch] text-4xl font-semibold tracking-[-0.03em] text-balance text-bone">
        Media forensics, signal by signal.
      </h2>
      <p className="mt-5 max-w-[62ch] text-[15px] leading-relaxed text-muted">
        Submit a file or a URL and read what each detector established on its own — provenance,
        synthetic video, face manipulation, voice — kept as separate evidence rather than
        averaged into one number that would mean nothing.
      </p>

      <div className="mt-10 flex flex-wrap items-center gap-4">
        <Link
          href={LOGIN_PATH}
          className="rounded-md bg-accent px-5 py-2.5 text-[13px] font-semibold text-ink transition-[opacity,transform] duration-150 hover:opacity-90 active:translate-y-px"
        >
          Sign in
        </Link>
        {/* For a reader who already has a session: the guard on `/app` sends them to sign in
            if they do not, so this link is safe to show to everyone. */}
        <Link
          href={WORKSPACE_PATH}
          className="text-[13px] text-muted underline hover:text-bone"
        >
          Go to workspace
        </Link>
      </div>

      <p className="mt-14 max-w-[62ch] text-[13px] leading-relaxed text-muted">
        Accounts are created by an administrator. Analyses are visible to the account that
        submitted them.
      </p>
    </main>
  );
}
