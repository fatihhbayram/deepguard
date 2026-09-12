/**
 * Put a new account on file, with a password the administrator chose.
 *
 * A plain HTML form on `/admin` posts here and this forwards a POST to the API, for the same
 * reason `/submit`, `/session` and `/admin/update-user` do: the API serves no CORS headers, and
 * opening it to a browser origin would be a security decision made to save a hop.
 *
 * **It is shaped like `/admin/update-user` and not like `/admin/create-api-key`, and the
 * difference is which way the secret travels.** Issuing an API key has to hand a credential
 * *back* to the browser, which cannot be done in a query string — that is why it answers JSON
 * and why it is the one flow on this surface with a client component behind it. Creating an
 * account carries a credential *inwards*, in a POST body, and hands back nothing but an id. So
 * this answers `303` with the outcome in the query string and needs no JavaScript on the client
 * at all, which is what the rest of this application does.
 *
 * **The password passes through this handler and is kept nowhere.** It is read off the form,
 * put in one request body, and dropped. It is not logged — the line below names the new
 * account's id and never the object the password is on — not put in a cookie, and not in the
 * redirect: the query string carries an id and an error sentence, never a field of the form.
 *
 * **This handler is not behind `admin/layout.tsx`.** A layout wraps the pages beneath it and
 * not the route handlers, so the guard that keeps a non-administrator from *seeing* `/admin`
 * does not keep one from posting here. That is not a hole being tolerated: the API is where this
 * is decided, `require_admin` refuses the forwarded POST with a 403, and the refusal is relayed
 * below. It is stated because the opposite belief — that being under `/admin` makes a route
 * privileged — is the kind that gets a check left out of the next handler added here, and this
 * one creates accounts whose role is on the body.
 *
 * Nothing here decides whether the creation is allowed. Not the address's shape, not whether it
 * is already taken, not the set of roles: all of them are the API's, in
 * `app/api/admin_users.py`, and a copy of any of them in this process would be a second rule
 * that could drift from the one actually enforced.
 */

import { NextResponse } from "next/server";

import { logError, logInfo } from "../../observability";
import {
  ADMIN_PATH,
  LOGIN_PATH,
  USER_ROLE_ADMIN,
  forwardedOrigin,
  isSameOrigin,
} from "../../session";
import { NewAccount, createAccount } from "../users";

/**
 * Back to `/admin`, with the outcome of this creation attached.
 *
 * A relative `Location` and a 303, for the two reasons `/admin/update-user` gives: an absolute
 * URL built from `request.url` would name the container's own port, and a 307 would re-post the
 * form on a reload — which here would mean a second account, or a 409.
 */
function back(params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${ADMIN_PATH}?${query}` },
  });
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read. The session cookie is `SameSite=Lax` and would not be
  // attached to a cross-site post in the first place; this is the independent check in front of
  // that, and what a forged form would be driving here is the creation of an account — with a
  // password the forger chose and, since `role` is on the body, possibly an administrative one.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return back({ error: "The account could not be read." });
  }

  const email = (form.get("email") ?? "").toString();
  const password = (form.get("password") ?? "").toString();
  const role = form.get("role");

  // Two checks, and deliberately only two — the same two `/admin/create-api-key` makes and for
  // the same reason. A missing address and a missing password are refusals this server can make
  // without asking, and making them here keeps an empty form out of a request that would only be
  // rejected at the other end. What is *not* duplicated here is any rule about what an address
  // may look like or how long a password must be: the API decides both, and a second opinion in
  // this process could only ever disagree with the one that holds.
  if (email.trim().length === 0) {
    return back({ error: "Give the account an email address." });
  }

  if (password.length === 0) {
    return back({ error: "Give the account a password." });
  }

  const account: NewAccount = { email, password };

  // The role travels only when the form named it, so an absent control means the API's own
  // default — an ordinary user. Passed through as sent, without being checked against a list of
  // roles held here.
  if (typeof role === "string") {
    account.role = role;
  }

  const result = await createAccount(account, forwardedOrigin(request));

  if (!result.ok) {
    if (result.unauthenticated) {
      // The session expired, was revoked, or was never there. Sending the operator to sign in is
      // the only useful answer; reporting it beside the form would leave them retrying a
      // creation that cannot succeed until they do.
      return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
    }

    // `result.error` is a sentence, never a field of the form — there is no path by which the
    // password could reach a log line here.
    await logError("An account could not be created.", { reason: result.error });

    return back({ error: result.error });
  }

  // The id and the role. Not the address, which is personal data the API's own log line and the
  // audit row already carry, and above all not `password`, which is the local it would be
  // easiest to include by spreading the form into this call.
  await logInfo("Created an account.", {
    account_id: result.account.id,
    is_admin: result.account.role === USER_ROLE_ADMIN,
  });

  return back({ created: result.account.id });
}
