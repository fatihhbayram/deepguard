/**
 * Sign in. Two fields, one button, one refusal.
 *
 * A plain HTML form posting to `/session`, like the ingest form on the dashboard and for the
 * same reason: this page ships no JavaScript, the API serves no CORS headers for a browser to
 * post across, and a form is what works without either. The handler on the other side talks to
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

// The single failure. Not "invalid password", not "unknown user" — see above.
const SIGN_IN_FAILED = "Sign in failed. Check your email and password and try again.";

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
      <div className="flex items-center gap-3">
        <span aria-hidden className="size-1.5 bg-accent" />
        <h1 className="font-mono text-[13px] tracking-[0.24em] text-bone">INSPECTROOT</h1>
      </div>

      <p className="mt-8 font-mono text-[11px] tracking-[0.18em] text-accent">— SIGN IN</p>
      <h2 className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-bone">
        Authenticate
      </h2>
      <p className="mt-4 text-[15px] leading-relaxed text-muted">
        Analyses are visible to the account that submitted them. Accounts are created by an
        administrator.
      </p>

      <form action="/session" method="post" className="mt-8 flex flex-col gap-5">
        <label className="flex flex-col gap-2">
          <span className="font-mono text-[10px] tracking-[0.18em] text-muted">EMAIL</span>
          <input
            type="email"
            name="email"
            required
            autoComplete="email"
            autoFocus
            className="w-full border border-line bg-ink px-3 py-2.5 font-mono text-[13px] text-bone transition-colors duration-150 placeholder:text-muted hover:border-rule focus:border-accent focus:outline-none"
          />
        </label>

        <label className="flex flex-col gap-2">
          <span className="font-mono text-[10px] tracking-[0.18em] text-muted">PASSWORD</span>
          <input
            type="password"
            name="password"
            required
            autoComplete="current-password"
            className="w-full border border-line bg-ink px-3 py-2.5 font-mono text-[13px] text-bone transition-colors duration-150 hover:border-rule focus:border-accent focus:outline-none"
          />
        </label>

        <button
          type="submit"
          className="mt-1 bg-accent px-6 py-3 font-mono text-[11px] tracking-[0.18em] text-ink transition-[opacity,transform] duration-150 hover:opacity-90 active:translate-y-px"
        >
          SIGN IN
        </button>
      </form>

      {failed && (
        <p
          role="status"
          className="mt-6 flex items-start gap-3 border border-rose-500/40 bg-rose-500/10 px-4 py-3 font-mono text-[11px] leading-relaxed text-rose-200"
        >
          <span aria-hidden className="mt-1.5 size-1.5 shrink-0 bg-rose-400" />
          <span>{SIGN_IN_FAILED}</span>
        </p>
      )}
    </main>
  );
}
