/**
 * Sign in. Two fields, one button, one refusal.
 *
 * A plain HTML form posting to `/session`, like the ingest form on the dashboard and for the
 * same reason: signing in needs no JavaScript, the API serves no CORS headers for a browser to
 * post across, and a form is what works without either. The only script on this page is the
 * language switch and the words it repaints (R16-T2); with it disabled the page reads in
 * English and the form still signs in. The handler on the other side talks to
 * the API from the server and relays the cookie the API sets, so the internal API address is
 * never in anything the browser can read.
 *
 * Every failure is one failure. A wrong password, an address with no account and a deactivated
 * account produce the same sentence here, because the API deliberately answers all three
 * identically — a page that said "no such account" would republish, in the one place anyone can
 * reach without credentials, exactly what that uniform 401 exists to withhold. The message is
 * also the same for an API that could not be reached: this page has no way to tell a reader
 * anything useful about that either, and a second, distinguishable message would be a way to
 * probe from outside which of the two happened.
 */

import { LanguageSelector, T } from "../i18n/client";

/** One query-string value, or null. A repeated parameter is not an outcome. */
function singleParam(value: string | string[] | undefined): string | null {
  return typeof value === "string" ? value : null;
}

export default async function Login({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  // Only the presence of the parameter is read. Its value is never rendered: it arrives in
  // a URL anyone can hand to anyone, and echoing it would make this page print whatever a
  // link says it should.
  const failed = singleParam(params.error) !== null;

  return (
    <main className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center px-4 py-16">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span aria-hidden className="size-1.5 rounded-full bg-accent" />
          {/* A name, not a machine value — set in the UI typeface, as it is in the header of
              the dashboard this page leads to. */}
          <h1 className="text-[14px] font-semibold tracking-[0.14em] text-bone uppercase">
            InspectRoot
          </h1>
        </div>
        <LanguageSelector variant="bar" />
      </div>

      <p className="mt-8 text-[11px] font-medium tracking-[0.16em] text-accent uppercase">
        <T k="auth.signIn" />
      </p>
      <h2 className="mt-2.5 text-2xl font-semibold tracking-[-0.02em] text-bone">
        <T k="login.title" />
      </h2>
      <p className="mt-3 text-[15px] leading-relaxed text-muted">
        <T k="login.intro" />
      </p>

      <form action="/session" method="post" className="mt-8 flex flex-col gap-5">
        <label className="flex flex-col gap-2">
          <span className="text-[11px] font-medium tracking-[0.1em] text-muted uppercase">
            <T k="login.email" />
          </span>
          <input
            type="email"
            name="email"
            required
            autoComplete="email"
            autoFocus
            className="w-full rounded-md border border-line bg-ink px-3 py-2.5 font-mono text-[13px] text-bone transition-colors duration-150 placeholder:text-muted hover:border-rule focus:border-accent"
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="text-[11px] font-medium tracking-[0.1em] text-muted uppercase">
            <T k="login.password" />
          </span>
          <input
            type="password"
            name="password"
            required
            autoComplete="current-password"
            className="w-full rounded-md border border-line bg-ink px-3 py-2.5 font-mono text-[13px] text-bone transition-colors duration-150 hover:border-rule focus:border-accent"
          />
        </label>

        <button
          type="submit"
          className="mt-1 rounded-md bg-accent px-5 py-2.5 text-[13px] font-semibold text-ink transition-[opacity,transform] duration-150 hover:opacity-90 active:translate-y-px"
        >
          <T k="auth.signIn" />
        </button>
      </form>

      {failed && (
        <p
          role="status"
          className="mt-6 flex items-start gap-3 rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-[13px] leading-relaxed text-rose-200"
        >
          <span aria-hidden className="mt-1.5 size-1.5 shrink-0 rounded-full bg-rose-400" />
          {/* The single failure. Not "invalid password", not "unknown user" — see above. */}
          <span>
            <T k="login.failed" />
          </span>
        </p>
      )}
    </main>
  );
}
