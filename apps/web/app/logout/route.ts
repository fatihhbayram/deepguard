/**
 * Sign-out: revokes the session at the API and clears the cookie the browser holds.
 *
 * Both halves matter and neither is enough alone. Clearing only the cookie would leave a
 * live row in `auth_sessions` that anyone holding a copy of the token could keep using;
 * revoking only the row would leave the browser sending a dead cookie on every request. The
 * API's `/auth/logout` does the first and answers with the header that does the second, and
 * this handler relays that header exactly as `/session` relays the one from sign-in.
 *
 * A POST, from a form, because signing out is a state change. As a link it would be a GET
 * that any prefetch, crawler or `<img>` on a page could fire, and the reader would find
 * themselves signed out by something they never clicked.
 */

import { NextResponse } from "next/server";

import { apiUrl } from "../analysis";
import { requestIdHeaders } from "../observability";
import {
  LOGIN_PATH,
  SESSION_COOKIE_NAME,
  forwardedOrigin,
  isSameOrigin,
  sessionHeaders,
} from "../session";

// Short. Revoking a session is one indexed update, and a sign-out that hangs is worse than
// one that falls back to clearing the cookie.
const LOGOUT_TIMEOUT_MS = 5000;

export async function POST(request: Request): Promise<NextResponse> {
  if (!isSameOrigin(request)) {
    return new NextResponse(null, { status: 403 });
  }

  const signedOut = new NextResponse(null, {
    status: 303,
    headers: { Location: LOGIN_PATH },
  });

  let response: Response | null = null;
  try {
    response = await fetch(`${apiUrl()}/api/v1/auth/logout`, {
      method: "POST",
      // The origin travels with the cookie: the API refuses this mutation too if it did not
      // come from the web application, and this call carries no origin of its own.
      headers: {
        ...(await sessionHeaders()),
        ...forwardedOrigin(request),
        ...(await requestIdHeaders()),
      },
      signal: AbortSignal.timeout(LOGOUT_TIMEOUT_MS),
    });
  } catch {
    // Left null, and handled below. The API being unreachable must not end with the reader
    // still signed in on this browser.
    response = null;
  }

  const relayed = response ? response.headers.getSetCookie() : [];
  for (const cookie of relayed) {
    signedOut.headers.append("set-cookie", cookie);
  }

  if (relayed.length === 0) {
    // The API did not answer, or answered without clearing anything. The row may well still
    // be live — nothing here can revoke it — but the cookie can still be dropped from this
    // browser, and a sign-out that visibly did nothing would be worse than one that did half.
    //
    // `secure` follows the scheme the request actually arrived on: a browser refuses to
    // overwrite a `Secure` cookie over plain HTTP, and hardcoding either value would make
    // this deletion silently miss in one of the two deployments.
    signedOut.cookies.set(SESSION_COOKIE_NAME, "", {
      path: "/",
      maxAge: 0,
      httpOnly: true,
      sameSite: "lax",
      secure: new URL(request.url).protocol === "https:",
    });
  }

  return signedOut;
}
