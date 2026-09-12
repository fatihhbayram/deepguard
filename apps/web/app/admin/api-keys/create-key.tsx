"use client";

/**
 * Issuing a key, and showing the secret the one time it can be shown.
 *
 * **The only client component on the administrative surface, and the only one in this
 * application that is not a convenience.** `PrintButton` on the report is scripting the page
 * could do without; this is not. The reason is stated in `create-api-key/route.ts` and is worth
 * repeating where the decision is visible: every other administrative mutation is a plain form
 * that answers `303` with its outcome in the query string, and the outcome here is a
 * credential. A secret in a URL is written into browser history, into the server's access log,
 * into the `Referer` of the next request, and into any link the operator copies. There is no
 * safe way to put it there, so it comes back in a response body and this component holds it in
 * memory.
 *
 * **Held in memory, and nowhere else.** The plaintext lives in React state for as long as this
 * page is open. It is not written to `localStorage`, not to `sessionStorage`, not to a cookie,
 * and not into the URL — there is no `router.replace` carrying it and no hash. A reload, a
 * navigation, or the dismiss button below ends it permanently, because the API keeps only the
 * digest and cannot reissue the value. That is why the panel says what it says.
 *
 * The revocation control on the page beside this is deliberately *not* a client component: it
 * is an ordinary form post, and it keeps working with scripting disabled. The destructive half
 * of this screen does not depend on JavaScript; only the half that must carry a secret does.
 */

import { useRouter } from "next/navigation";
import { useState } from "react";

import type { CreatedApiKey } from "../api-keys";

// Type-only, and it has to stay that way. `../api-keys` reads the session cookie through
// `next/headers`, which does not exist in a browser; a value import would pull that module
// into this bundle. `import type` is erased at compile time, so what is shared here is the
// shape of the API's response and none of its code.

// The address this posts to. The route handler forwards to the API, which serves no CORS
// headers and could not be called from here directly.
const CREATE_URL = "/admin/create-api-key";

// What the API's column holds. Stated on the input as `maxLength` so the browser stops the
// operator at the limit rather than letting them write a label that gets refused on submit.
// The API validates the same bound and is the authority; this is the courtesy in front of it.
const MAX_NAME_LENGTH = 255;

/** The sentence shown when the fetch itself failed, rather than the API refusing something. */
const UNREACHABLE = "The dashboard could not be reached. The key was not issued.";

/**
 * The secret, shown once, with the warning that makes that fact actionable.
 *
 * Deliberately loud. The consequence of a reader skimming past this is a credential that has to
 * be revoked and reissued, which is a customer-visible event — so the panel states the
 * irreversibility before it shows the value, rather than in a footnote under it.
 *
 * The value sits in a `readOnly` input rather than a `<code>` block. An input is selectable
 * with a single click on every platform, works with a keyboard, and — unlike a text node —
 * lets the copy control below fall back to `select()` where the clipboard API is unavailable.
 * `readOnly` and not `disabled`: a disabled field cannot be selected or read by a screen
 * reader, which would defeat both.
 */
function Secret({ value, onDismiss }: { value: string; onDismiss: () => void }) {
  const [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");

  async function copy() {
    try {
      // Only available in a secure context. A deployment reached over plain HTTP has no
      // `navigator.clipboard` at all, which is why the failure below is a real state and not a
      // defensive nicety — the operator is told to copy it by hand rather than left believing
      // a silent button worked.
      await navigator.clipboard.writeText(value);
      setCopied("done");
    } catch {
      setCopied("failed");
    }
  }

  return (
    <div
      role="alert"
      className="mt-6 rounded-lg border border-accent/40 bg-accent/10 p-5"
    >
      <p className="text-[13px] font-medium text-bone">
        Copy this key now. It will not be shown again.
      </p>
      <p className="mt-1.5 max-w-[68ch] text-[13px] leading-relaxed text-muted">
        DeepGuard stores only a hash of it and cannot recover the value. If it is lost, the only
        remedy is to revoke this key and issue another.
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <label className="sr-only" htmlFor="issued-key">
          The issued API key
        </label>
        <input
          id="issued-key"
          readOnly
          value={value}
          // Selected on focus so a click puts the whole key on the clipboard with the
          // operator's own copy shortcut, whatever the button below is able to do.
          onFocus={(event) => event.currentTarget.select()}
          className="min-w-0 flex-1 rounded-md border border-line bg-ink px-3 py-2 font-mono text-[12px] text-bone"
        />
        <button
          type="button"
          onClick={copy}
          className="cursor-pointer rounded-md border border-line px-3 py-2 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          {copied === "done" ? "Copied" : "Copy"}
        </button>
        <button
          type="button"
          // The only way to clear the value deliberately, and it is worded as what it does. A
          // reload does the same thing by accident, which is precisely the risk the warning
          // above is about.
          onClick={onDismiss}
          className="cursor-pointer rounded-md border border-line px-3 py-2 text-[12px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone"
        >
          I have copied it
        </button>
      </div>

      {copied === "failed" && (
        <p className="mt-3 text-[12px] text-muted">
          This browser would not let the page write to the clipboard. Select the key above and
          copy it yourself.
        </p>
      )}
    </div>
  );
}

/**
 * The name field, the submit, and whatever came back.
 *
 * One key at a time: issuing a second replaces the first in the panel above, which is why the
 * form is disabled while a request is in flight and why the panel has an explicit dismissal.
 * The alternative — a list of issued secrets accumulating down the page — would keep every one
 * of them on screen for as long as the tab is open.
 *
 * `signInPath` is passed in rather than imported. It lives in `session.ts`, which reads cookies
 * through `next/headers` — a value import would pull that server-only module into this bundle,
 * and the same reason the response type above is imported with `import type`. The page already
 * has the constant; handing it down is cheaper than restating the literal here, where a second
 * copy could drift from the one every other redirect uses.
 */
export function CreateKey({ signInPath }: { signInPath: string }) {
  const router = useRouter();
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<CreatedApiKey | null>(null);

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (pending) {
      return;
    }

    setPending(true);
    setError(null);

    try {
      const response = await fetch(CREATE_URL, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }),
      });

      const payload = await response.json().catch(() => null);

      if (response.status === 401) {
        // The session went away. A navigation the component performs itself, because the fetch
        // would have followed a `303` transparently and handed this code the sign-in page's
        // HTML as though it were an answer.
        router.push(signInPath);

        return;
      }

      if (!response.ok) {
        const detail =
          typeof payload === "object" && payload !== null
            ? (payload as Record<string, unknown>).error
            : null;

        setError(typeof detail === "string" ? detail : "The key could not be issued.");

        return;
      }

      // The one place the secret enters this component's state. It is not logged, not stored,
      // and not put in the URL; `router.refresh()` below re-renders the server component with
      // the new row in it and does not carry the value anywhere.
      setCreated(payload as CreatedApiKey);
      setName("");
      router.refresh();
    } catch {
      setError(UNREACHABLE);
    } finally {
      setPending(false);
    }
  }

  return (
    <div>
      <form onSubmit={submit} className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1">
          <label
            htmlFor="key-name"
            className="text-[11px] font-medium tracking-[0.08em] text-muted uppercase"
          >
            Name
          </label>
          <input
            id="key-name"
            name="name"
            required
            maxLength={MAX_NAME_LENGTH}
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Who is this key for?"
            disabled={pending}
            className="mt-1.5 w-full rounded-md border border-line bg-ink px-3 py-2 text-[13px] text-bone transition-colors duration-150 hover:border-rule focus:border-rule focus:outline-none disabled:opacity-60"
          />
        </div>
        <button
          type="submit"
          disabled={pending}
          className="cursor-pointer rounded-md border border-line px-4 py-2 text-[13px] font-medium text-muted transition-colors duration-150 hover:border-rule hover:text-bone disabled:cursor-default disabled:opacity-60"
        >
          {pending ? "Issuing…" : "Issue key"}
        </button>
      </form>

      {/* The one control on this surface that does not work without scripting, said plainly
          rather than left as a button that silently does nothing. Revocation below is a plain
          form and is unaffected. */}
      <noscript>
        <p className="mt-3 text-[12px] text-muted">
          Issuing a key needs JavaScript: the secret is returned in the response and cannot be
          carried safely in a page reload.
        </p>
      </noscript>

      {error && (
        <p role="status" className="mt-3 text-[13px] text-rose-200">
          {error}
        </p>
      )}

      {created && (
        <Secret value={created.plaintext} onDismiss={() => setCreated(null)} />
      )}
    </div>
  );
}
