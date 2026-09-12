/**
 * Change an account's address, its role, or its activation.
 *
 * A plain HTML form posts here and this forwards a PATCH to the API, for the same reason
 * `/submit` and `/session` exist rather than the browser calling the API directly — the API
 * serves no CORS headers, and opening it to a browser origin would be a security decision
 * made to save a hop. No JavaScript is involved on the client at all, which is why the
 * outcome travels back in the query string: there is nothing on the page to receive a
 * response body.
 *
 * A POST, from a form, carrying the method the API actually wants in the body of the request
 * rather than on it. A browser form can only issue GET or POST, and the alternative to
 * translating here would be a client component whose only job is to call `fetch` with a verb
 * — a script dependency bought for the sake of a header, on the one page where the thing
 * being changed is who holds administrative access.
 *
 * **This handler is not behind `admin/layout.tsx`.** A layout wraps the pages beneath it and
 * not the route handlers, so the guard that keeps a non-administrator from *seeing* `/admin`
 * does not keep one from posting here. That is not a hole being tolerated: the API is where
 * this is decided, `require_admin` refuses the forwarded PATCH with a 403, and the refusal is
 * relayed below. It is stated because the opposite belief — that being under `/admin` makes a
 * route privileged — is the kind that gets a check left out of the next handler added here.
 *
 * Nothing in this file decides whether a change is allowed. Not the self-modification rule,
 * not the last-administrator invariant, not the set of roles, and since R8-T8 not whether an
 * address is free either: all of them are the API's, in `app/api/admin_users.py`, and a copy of
 * any of them here would be a second rule that could drift from the one actually enforced. What
 * this does is carry the form to the API and the API's answer back to the page.
 *
 * **Two pages post here, and the form says which one to go back to.** The accounts list has the
 * role and activation controls on every row; the account detail page (R8-T8) has a form that can
 * also change the address. A handler that always redirected to `ADMIN_PATH` would bounce an
 * operator off the detail page every time they saved, so the form carries a `return_to` field.
 * It is not a URL and cannot become one: the only value with any effect is the literal string
 * `"detail"`, which sends the browser to this account's own detail page built from the id by
 * `adminAccountPath`. Anything else, including an absent field, means the list. A handler that
 * redirected to a path out of the form body would be an open redirect on a route that is posted
 * to with a session cookie attached.
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
import { AccountChange, updateAccount } from "../users";

// The one `return_to` value that means anything, and the reason it is compared rather than
// used. See the note above: this is a token the handler interprets, never a path it follows.
const RETURN_TO_DETAIL = "detail";

/**
 * Back to whichever page submitted this, with the outcome attached.
 *
 * A relative `Location`, resolved by the browser against the address it actually asked for.
 * Building an absolute URL out of `request.url` looks more careful and is wrong here: the
 * container listens on 3000 and is published on another port, so the absolute form would send
 * the operator to a port nothing is listening on outside Docker.
 *
 * 303, so the browser follows it with a GET. A 307 would re-post the form, and a reload would
 * then apply the same change a second time.
 */
function back(path: string, params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${path}?${query}` },
  });
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read. The session cookie is `SameSite=Lax` and would not be
  // attached to a cross-site post in the first place; this is the independent check in front
  // of that, and it matters more here than on a submission because what a forged form would
  // be driving is the granting of administrative access.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return back(ADMIN_PATH, { error: "The change could not be read." });
  }

  const userId = (form.get("user_id") ?? "").toString().trim();
  const email = form.get("email");
  const role = form.get("role");
  const isActive = form.get("is_active");

  if (!userId) {
    return back(ADMIN_PATH, { error: "The change named no account." });
  }

  // Where to send the browser afterwards, decided here and not taken from the form. The field
  // selects between two addresses this file knows; it never supplies one.
  const destination =
    form.get("return_to") === RETURN_TO_DETAIL ? adminAccountPath(userId) : ADMIN_PATH;

  // Only the fields the form actually carried. The controls are separate submissions — the role
  // control sends a role, the activation control sends an activation, the detail page's form
  // sends an address — and sending the absent ones as `null` would turn each into a change to
  // all of them, restoring whatever the page happened to have rendered last over a value
  // somebody else may have just altered.
  const change: AccountChange = {};
  if (typeof email === "string") {
    // Passed through as the field sent it, without being normalized or checked for a duplicate
    // here. The API normalizes in its validator and the unique index is what enforces
    // uniqueness; a second answer to either in this process would be one that could disagree.
    change.email = email;
  }
  if (typeof role === "string") {
    // Passed through exactly as the control sent it, without being checked against a list of
    // roles held here. The API validates it against its own constants and answers 422 for
    // anything else; a second list in this process would be one that could disagree.
    change.role = role;
  }
  if (typeof isActive === "string") {
    // A checkbox would send nothing when off, so the control is a button carrying the value
    // it means. `"true"` is the only spelling that activates, and everything else deactivates
    // — the direction that fails towards less access when a form arrives malformed.
    change.is_active = isActive === "true";
  }

  const result = await updateAccount(userId, change, forwardedOrigin(request));

  if (!result.ok) {
    if (result.unauthenticated) {
      // The session expired, was revoked, or was never there. Sending the operator to sign in is
      // the only useful answer; reporting it beside the table would leave them retrying a change
      // that cannot succeed until they do.
      return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
    }

    // Includes the 403 a non-administrator gets, the 400 a change that would break a system
    // invariant gets and the 409 a taken address gets. All are shown beside the form rather than
    // turned into a redirect elsewhere: the operator is on the right page and the sentence is
    // the whole of what they need. The underlying reason is also logged, because a failure to
    // reach the API at all is an operational fact this server's log should carry.
    await logError("An account change was refused.", { account_id: userId });

    return back(destination, { error: result.error });
  }

  // Which account, and by nothing more than its id. The email is personal data and does not
  // need to be in a log line to make it actionable, and the API has already written the
  // authoritative line naming both the administrator and the values it stored.
  await logInfo("Forwarded an account change to the API.", { account_id: userId });

  return back(destination, { updated: userId });
}
