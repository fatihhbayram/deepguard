"""The three endpoints a browser signs in, signs out and identifies itself through.

Mounted under `/api/v1`, alongside the other internal routes and deliberately not on the
public B2B surface. `/api/public/v1` authenticates by API key and only by API key; nothing
here is reachable from it, and a session cookie presented to it is simply not looked at.

The routes are thin on purpose. Normalization, hashing, session rotation and the cookie
flags all live in `app.web_auth`, so what happens on sign-in is described in one place
rather than half here and half there — and the bootstrap script reaches the same functions
these routes do.

The plaintext session token appears in exactly one place in this module: the value handed to
`set_session_cookie`. It is not returned in a body, not echoed in a header of our own, and
not logged.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_session
from app.web_auth import (
    authenticate,
    clear_session_cookie,
    invalid_credentials,
    require_same_origin,
    require_user,
    revoke_session,
    session_token,
    set_session_cookie,
    start_session,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class Credentials(BaseModel):
    """What a sign-in presents.

    `str` rather than pydantic's `EmailStr`, which would pull in a validator dependency to
    reject addresses the lookup already fails to find. A syntactically invalid address is
    simply an address with no account, and it gets the same 401 as any other — which is
    better than a 422 that tells an unauthenticated caller their guess was not even
    well-formed.

    The password is bounded because it is hashed, and Argon2 will faithfully spend time on
    however many megabytes it is handed; an unbounded field would make an unauthenticated
    request able to choose the server's workload.
    """

    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class AuthenticatedUser(BaseModel):
    """Who the caller is signed in as.

    Three fields. No password hash, no session id, no timestamps — this answers "who am I
    and what may I do", and anything further would be a wider contract than the frontend
    that will consume it needs.
    """

    id: uuid.UUID
    email: str
    role: str


def authenticated_user(user: User) -> AuthenticatedUser:
    """The stored account, narrowed to what a caller is told about itself."""
    return AuthenticatedUser(id=user.id, email=user.email, role=user.role)


@router.post("/login", response_model=AuthenticatedUser)
def login(
    credentials: Credentials,
    response: Response,
    session: Session = Depends(get_session),
) -> AuthenticatedUser:
    """Sign in, and set the session cookie that authenticates subsequent requests.

    A successful sign-in revokes whatever session the account already had — see
    `start_session`. That happens in the same transaction as the new session's insert, so
    the account never briefly holds two.

    Every failure is one 401 with one body. An address with no account, a wrong password and
    a deactivated account are indistinguishable here, and the Argon2 verification runs even
    when no account matched so they are indistinguishable by timing too.

    The token goes into the cookie and nowhere else. The response body identifies the user;
    it does not carry the credential, which would put a `HttpOnly` token straight back where
    script can read it.
    """
    user = authenticate(session, credentials.email, credentials.password)

    if user is None:
        # No address, no reason, no hash. A log line naming the attempted address on every
        # failed sign-in would accumulate a list of guessed addresses in the log.
        logger.info("A sign-in attempt failed.")
        raise invalid_credentials()

    set_session_cookie(response, start_session(session, user))

    return authenticated_user(user)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    # Signing out is a state change — it revokes a row — so it carries the same origin check
    # as the dashboard's other mutations. Being signed out by a page you merely visited is a
    # small harm next to having an analysis filed in your name, but it is still something
    # done to you by a site that has no business doing it.
    dependencies=[Depends(require_same_origin)],
)
def logout(
    response: Response,
    session: Session = Depends(get_session),
    presented: str | None = Depends(session_token),
) -> None:
    """End the caller's session and clear the cookie.

    Behind `require_same_origin` but deliberately not behind `require_user`. The two are
    different questions and only the first is worth asking here: where the request came from
    is a property of the request, while demanding a *valid* session to end one would strand
    exactly the browsers that most need to sign out.

    Signing out is not a privileged action, and
    demanding a valid session for it would mean a browser holding an expired or already
    revoked cookie could not get rid of it — it would be told 401 and keep the cookie. So
    this always answers 204: the row is revoked if there was one to revoke, and the cookie
    is cleared either way.

    That uniformity is also why it tells the caller nothing about what it found. A sign-out
    that answered differently for a real session than for a made-up token would let anyone
    test whether a token they hold is live.
    """
    if presented is not None:
        revoke_session(session, presented)

    clear_session_cookie(response)


@router.get("/me", response_model=AuthenticatedUser)
def me(user: User = Depends(require_user)) -> AuthenticatedUser:
    """Who the session cookie authenticates, or 401 if it authenticates nobody.

    The frontend's way of asking whether it is signed in and as whom, without having to
    parse a cookie it deliberately cannot read. An absent, unknown, expired or revoked
    session all answer the same 401.
    """
    return authenticated_user(user)
