/**
 * Replace an account's password — which also ends every session that account was holding open.
 *
 * A plain HTML form on the account detail page posts here and this forwards a POST to the API,
 * for the reason `/admin/update-user` and `/admin/create-user` do: the API serves no CORS
 * headers, and opening it to a browser origin would be a security decision made to save a hop.
 *
 * **The password travels inwards only, which is what lets this be a plain form.** It is read off
 * the body, put in one request, and dropped. Nothing comes back but a status — the API answers
 * 204 with no body — so the outcome can go in a query string without a secret going with it.
 * That is the whole difference between this handler and `/admin/create-api-key`, which has to
 * hand a credential *back* and therefore cannot redirect at all.
 *
 * It is not logged. The line below names the account by id and never the form, and the redirect
 * carries an id or an error sentence and never a field of it.
 *
 * **Resetting one's own password signs one out.** The revocation the API performs covers every
 * open session for the target account, the caller's included, and that is the point of it rather
 * than an oversight: a credential replaced because it may have been compromised is not replaced
 * at all if the sessions opened with it keep working. The redirect below therefore lands on a
 * page whose own session read will fail, and `/admin/users/[id]` sends the reader to sign in —
 * with the new password, which they just chose. The detail page says so above the form, because
 * an operator who is bounced to a login screen with no warning will believe something broke.
 *
 * **This handler is not behind `admin/layout.tsx`.** A layout wraps the pages beneath it and not
 * the route handlers, so the guard that keeps a non-administrator from *seeing* the detail page
 * does not keep one from posting here. The API is where this is decided — `require_admin`
 * refuses the forwarded POST with a 403 and the refusal is relayed below — and it is stated
 * because this is the route where believing otherwise would be worst: a handler that trusted its
 * location would be an unauthenticated way to set anybody's password.
 */

import { NextResponse } from "next/server";

import { logError, logInfo } from "../../observability";
import {
  ADMIN_PATH,
  LOGIN_PATH,
  adminAccountPath,
  forwardedOrigin,
  isSameOrigin,
} from "../../session";
import { resetPassword } from "../users";

/**
 * Back to the account's own page, with the outcome of the reset attached.
 *
 * The path is built from the id by `adminAccountPath` rather than taken from the form. A handler
 * that redirected to an address out of a request body would be an open redirect on a route
 * posted to with a session cookie attached.
 *
 * A relative `Location` and a 303, for the two reasons `/admin/update-user` gives: an absolute
 * URL built from `request.url` would name the container's own port, and a 307 would re-post the
 * form on a reload — which here would mean setting the password a second time and revoking the
 * sessions the first one opened.
 */
function back(accountId: string, params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);
  const path = accountId ? adminAccountPath(accountId) : ADMIN_PATH;

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${path}?${query}` },
  });
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read. The session cookie is `SameSite=Lax` and would not be
  // attached to a cross-site post in the first place; this is the independent check in front of
  // that, and this is the request in the whole surface a forged form would most want to drive: a
  // password an attacker chose is an account an attacker can sign into.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return back("", { error: "The reset could not be read." });
  }

  const userId = (form.get("user_id") ?? "").toString().trim();
  const password = (form.get("password") ?? "").toString();

  if (!userId) {
    return back("", { error: "The reset named no account." });
  }

  // The only check this server makes. An empty password is a refusal it can make without asking;
  // how long a password must actually be is the API's `MINIMUM_PASSWORD_LENGTH`, and a second
  // floor here would be one that could disagree with the one that holds.
  if (password.length === 0) {
    return back(userId, { error: "Give the account a new password." });
  }

  const result = await resetPassword(userId, password, forwardedOrigin(request));

  if (!result.ok) {
    if (result.unauthenticated) {
      return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
    }

    // A sentence, never a field of the form.
    await logError("A password could not be reset.", { account_id: userId });

    return back(userId, { error: result.error });
  }

  // The account by id and nothing else. Never `password`, and never the form — this is the line
  // in this file where including either would be easiest and least visible.
  await logInfo("Reset an account password.", { account_id: userId });

  return back(userId, { reset: userId });
}
