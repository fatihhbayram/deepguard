/**
 * End an API key's access.
 *
 * A plain HTML form posts here and this forwards a POST to the API — the same shape
 * `/admin/update-user` has, and deliberately the same. Nothing about a revocation needs
 * JavaScript: there is no secret coming back, so the outcome can travel in the query string
 * the way every other administrative mutation's does, and the control that drives it is an
 * ordinary submit button that works with scripting disabled.
 *
 * That is the difference between this handler and `create-api-key/route.ts` next door, and it
 * is worth being explicit about: the creation route answers JSON *only* because a credential
 * may never be put in a URL. Revocation has no such constraint and therefore gets no such
 * exception — the destructive half of this screen is the half that keeps working when the
 * script does not.
 *
 * **Revoking an already-revoked key is a success, not an error.** The API treats it as a no-op
 * and returns 200 without writing anything; this relays that as an ordinary confirmation. An
 * operator who clicks twice, or reloads, has not done anything wrong, and telling them
 * otherwise would suggest the key's state is somehow in doubt when it is precisely not.
 *
 * **This handler is not behind `admin/layout.tsx`** — a layout wraps pages and not route
 * handlers. The API is where this is decided: `require_admin` refuses the forwarded POST with
 * a 403 and the refusal is shown beside the table. Stated for the reason its sibling states it.
 *
 * Nothing in this file decides whether a revocation is allowed, and there is no reactivation
 * counterpart to it — the API offers none, on purpose, because a revoked credential coming
 * back is the one thing an operator revoking one is relying on being impossible.
 */

import { NextResponse } from "next/server";

import { logError, logInfo } from "../../observability";
import {
  ADMIN_API_KEYS_PATH,
  LOGIN_PATH,
  forwardedOrigin,
  isSameOrigin,
} from "../../session";
import { revokeApiKey } from "../api-keys";

/**
 * Back to `/admin/api-keys`, with the outcome of this revocation attached.
 *
 * A relative `Location`, resolved by the browser against the address it actually asked for.
 * Building an absolute URL out of `request.url` looks more careful and is wrong here: the
 * container listens on 3000 and is published on another port, so the absolute form would send
 * the operator to a port nothing is listening on outside Docker.
 *
 * 303, so the browser follows it with a GET. A 307 would re-post the form, and a reload would
 * then apply the same revocation again — harmless, since the API makes the second one a no-op,
 * but it would still be this server asking for a write nobody requested.
 *
 * Nothing secret is ever put in these parameters. What goes here is a key id and an error
 * sentence; the plaintext belongs to the creation flow, which for exactly this reason does not
 * redirect at all.
 */
function back(params: Record<string, string>): NextResponse {
  const query = new URLSearchParams(params);

  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${ADMIN_API_KEYS_PATH}?${query}` },
  });
}

export async function POST(request: Request): Promise<NextResponse> {
  // Refused before the body is read, for the reason its sibling gives. What a forged request
  // would be driving here is the withdrawal of a customer's access.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return back({ error: "The revocation could not be read." });
  }

  const keyId = (form.get("key_id") ?? "").toString().trim();

  if (!keyId) {
    return back({ error: "The revocation named no key." });
  }

  const result = await revokeApiKey(keyId, forwardedOrigin(request));

  if (!result.ok) {
    if (result.unauthenticated) {
      // The session expired, was revoked, or was never there. Sending the operator to sign in
      // is the only useful answer; reporting it beside the table would leave them retrying a
      // revocation that cannot succeed until they do.
      return new NextResponse(null, { status: 303, headers: { Location: LOGIN_PATH } });
    }

    await logError("An API key could not be revoked.", {
      api_key_id: keyId,
      reason: result.error,
    });

    // Includes the 403 a non-administrator gets and the 404 an id that names nothing gets.
    // Both are shown beside the table: the operator is on the right page and the sentence is
    // the whole of what they need.
    return back({ error: result.error });
  }

  // Which key, by id. The API has already written the authoritative line naming the
  // administrator who did it, and the audit event beside it is the durable record.
  await logInfo("Forwarded an API key revocation to the API.", { api_key_id: keyId });

  return back({ revoked: result.key.id });
}
