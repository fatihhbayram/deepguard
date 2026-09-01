/**
 * Sign-in: takes the login form, hands the credentials to the API, relays the cookie back.
 *
 * The browser posts here rather than to the API for the reason every other write in this
 * application does — the API serves no CORS headers, and opening it up to a browser origin
 * would be a security decision made to save a hop. Running the call from the server also
 * keeps the internal API address out of anything the browser can read, and it is what lets
 * the session cookie be set on *this* origin, which is the origin the dashboard is served
 * from and therefore the only one a cookie is any use on.
 *
 * The cookie is relayed exactly as the API wrote it, flags included. Rebuilding it here
 * would mean this file deciding `HttpOnly`, `SameSite`, `Secure` and the lifetime a second
 * time, and the copy that drifted would be the one that quietly stopped protecting anything
 * — the API owns those, and `app/web_auth.py` says why for each.
 *
 * Nothing about the credentials is kept. They are read out of the form, sent once, and go
 * out of scope; they are never logged, never put in a URL and never echoed back to the page.
 */

import { NextResponse } from "next/server";

import { API_URL } from "../analysis";
import { requestIdHeaders } from "../observability";
import { LOGIN_PATH, isSameOrigin } from "../session";

// How long the API is given to answer a sign-in. Argon2id is deliberately slow — that is
// what makes a stolen hash expensive to attack — so this is generous next to the read
// timeouts elsewhere, and still bounded.
const LOGIN_TIMEOUT_MS = 15_000;

/**
 * Back to the login page, marked as failed.
 *
 * A relative `Location` resolved by the browser against the address it asked for, as in
 * `/submit`: the container listens on 3000 and is published on another port, so an absolute
 * URL built from `request.url` would send the reader to a port nothing serves.
 *
 * 303, so the browser follows it with a GET and a reload does not re-post the password.
 *
 * The parameter carries no detail — not the address, not which of the failures it was.
 * `login/page.tsx` reads only that it is there.
 */
function failed(): NextResponse {
  return new NextResponse(null, {
    status: 303,
    headers: { Location: `${LOGIN_PATH}?error=1` },
  });
}

export async function POST(request: Request): Promise<NextResponse> {
  // Cross-origin submissions are refused before the credentials are even read. The session
  // cookie is `SameSite=Lax` and would not be attached to a cross-site post in the first
  // place; this is the independent second check, and it is a flat refusal rather than a
  // redirect because nothing about it is a state the login page should render.
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return failed();
  }

  const email = (form.get("email") ?? "").toString();
  const password = (form.get("password") ?? "").toString();

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/v1/auth/login`, {
      method: "POST",
      // The request id, and nothing else added: a sign-in carries no session yet, so this
      // is the only thing joining this call to the API's line about it.
      headers: { ...(await requestIdHeaders()), "content-type": "application/json" },
      body: JSON.stringify({ email, password }),
      signal: AbortSignal.timeout(LOGIN_TIMEOUT_MS),
    });
  } catch {
    // An unreachable API is the same answer as a refused sign-in. It has to be: a
    // distinguishable response here would let an unauthenticated caller tell the two apart,
    // and the reader can do nothing different about either.
    return failed();
  }

  if (!response.ok) {
    return failed();
  }

  const signedIn = new NextResponse(null, { status: 303, headers: { Location: "/" } });

  // Every `Set-Cookie` the API sent, verbatim and in order. `getSetCookie` rather than
  // `get("set-cookie")`, which would fold multiple cookies into one comma-joined string and
  // hand the browser a header it cannot parse.
  for (const cookie of response.headers.getSetCookie()) {
    signedIn.headers.append("set-cookie", cookie);
  }

  return signedIn;
}
