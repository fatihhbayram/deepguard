"""Password and session authentication for people signing into the web application.

Separate from `app.auth`, which authenticates B2B callers by API key, and deliberately so.
These are two credential families with two different threat models and two different
surfaces: a browser presents a cookie to the internal API, a customer's server presents a
bearer token to `/api/public/v1`. Nothing here is mounted on the public router and nothing
here reads an `Authorization` header, so a web cookie cannot authenticate a public endpoint
— not because a check rejects it, but because no code path on that surface ever looks at a
cookie. `tests/test_web_auth.py` asserts that rather than leaving it to be believed.

Two hashes appear below and the difference between them is the whole design. A password is
human-chosen and therefore guessable, so it goes through Argon2id — memory-hard and slow on
purpose, paid once per sign-in. A session token is 256 bits out of the system CSPRNG, so
there is nothing to guess and it goes through SHA-256, paid on every authenticated request.
Using the slow hash for the token would buy no security and charge for it on the hot path;
using the fast one for the password would be a real weakness.

The plaintext session token leaves this module exactly twice — as the return value of
`start_session` and in the `Set-Cookie` header `set_session_cookie` writes — and is never
persisted and never logged. What is stored is its digest, which is enough to recognise the
token the browser sends back and useless to anyone who reads the table.

Every sign-in failure is the same failure. An address with no account, a wrong password and
a deactivated account all produce an identical 401, and the verification cost is paid even
when no account was found — otherwise the response time alone would answer the question the
uniform body is refusing to.
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, InvalidHashError
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.db.models import USER_ROLE_ADMIN, AuthSession, User
from app.db.session import get_session

logger = logging.getLogger(__name__)

# Argon2id at the library's own defaults, which track RFC 9106's recommended parameters and
# are revised by upstream as hardware moves. Pinning our own numbers here would mean
# freezing 2026's idea of expensive into the codebase and having to remember to revisit it;
# the encoded hash carries the parameters it was made with, so an existing password keeps
# verifying when those defaults change.
PASSWORD_HASHER = PasswordHasher()

# A hash of nothing, verified against when no account matched, purely so that the failing
# path costs what the succeeding one costs. Without it a request for an unknown address
# would return in microseconds while a wrong password took the full Argon2 time, and an
# attacker could enumerate which addresses have accounts with a stopwatch — defeating the
# identical response body a few lines below.
#
# Built from fresh randomness at import: nothing can ever verify against it, and no fixed
# string that looks like a credential is committed to the repository.
_UNMATCHED_PASSWORD_HASH = PASSWORD_HASHER.hash(secrets.token_urlsafe(32))

# The cookie the browser carries a session in. Prefixed so it is recognisable in a browser's
# storage inspector as this application's.
SESSION_COOKIE_NAME = "deepguard_session"

# Bytes of entropy behind a session token. 32 bytes is 256 bits, the same floor the API keys
# use and the reason the stored digest can be a plain SHA-256: there is no dictionary to
# attack, so there is nothing for a slow hash to slow down.
SESSION_ENTROPY_BYTES = 32

# How long a sign-in lasts. Long enough to cover a working day without a re-login in the
# middle of it, short enough that a session left behind on a shared machine stops working
# the same day. There is no refresh and no sliding window: R1-T1 has no requirement for one,
# and an expiry that renews itself on every request is not really an expiry.
SESSION_LIFETIME = timedelta(hours=12)

# Which deployment this is. `Secure` cookies are refused by browsers over plain HTTP, so a
# hardcoded `Secure=True` would make local development and this suite unable to sign in at
# all, while a hardcoded `False` would put a production session token on the wire in clear.
# The environment decides — and it decides in the fail-secure direction.
#
# `Secure` is dropped only for an environment that explicitly names itself as one of the
# plain-HTTP ones below. Everything else — the variable unset, empty, misspelled, or set to
# a name nobody taught this list about — gets a `Secure` cookie. The asymmetry is the point:
# an operator who forgets the variable on a production host would otherwise ship session
# tokens in clear and see nothing wrong, because an insecure cookie works perfectly. The
# opposite mistake, a developer who forgets it on localhost, cannot sign in and finds out in
# seconds. Production safety must not depend on somebody remembering to ask for it.
#
# `docker-compose.yml` sets `development` explicitly for exactly this reason: local HTTP is
# the case that has to opt in.
ENVIRONMENT_VARIABLE = "DEEPGUARD_ENV"

# The deployments served over plain HTTP, where a `Secure` cookie would be discarded by the
# browser and sign-in would simply not work. A closed list, and the only way to get an
# insecure cookie.
INSECURE_COOKIE_ENVIRONMENTS = frozenset({"development", "test"})


def cookie_is_secure() -> bool:
    """Whether the session cookie is marked `Secure`, read fresh on every response.

    Read at call time rather than captured at import so a test can set the variable around
    a request. It is one `os.getenv` on a path that is already writing a hashed session to
    PostgreSQL.

    Note the direction of the test: this asks whether the environment is a known insecure
    one, not whether it is production. Written the other way round, every value that is not
    literally `production` — including the unset variable — would silently disable the flag.
    """
    environment = os.getenv(ENVIRONMENT_VARIABLE, "").strip().lower()

    return environment not in INSECURE_COOKIE_ENVIRONMENTS


def normalize_email(email: str) -> str:
    """The single normalization every address goes through, on creation and on login.

    Case-folded and trimmed. It has to be one function used by both paths: if creation
    stored `Alice@example.com` verbatim and login looked up `alice@example.com`, the account
    would exist and be unreachable — and if only login normalized, two accounts could differ
    by case and one sign-in would have two answers. The unique index on `users.email` is
    over this same normalized form, which is what makes the second case impossible rather
    than merely unlikely.

    Only case and surrounding whitespace. Not dot-stripping, not plus-address removal —
    those are one provider's delivery rules, not part of an address's identity, and applying
    them would merge accounts that belong to different people.
    """
    return email.strip().lower()


def hash_password(password: str) -> str:
    """The Argon2id hash stored for a password, salt and parameters included."""
    return PASSWORD_HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Whether the password matches the stored hash, without ever raising for a mismatch.

    argon2-cffi signals a wrong password by exception, which reads as an error at the call
    site when it is an ordinary expected outcome. A boolean is what the caller needs, and it
    keeps the failing path from being distinguishable from the succeeding one by anything
    other than its return value.

    A stored value Argon2 cannot parse is `False` too, not a crash. That is a corrupted or
    hand-edited row, and the correct answer to "does this password match" is still no.
    """
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except (Argon2Error, InvalidHashError):
        return False


def generate_session_token() -> str:
    """Mint a session token: 256 bits from the system CSPRNG, and nothing else.

    Opaque on purpose. It encodes no user id, no expiry and no signature, so there is
    nothing in it for a holder to read or for a forger to reconstruct — the row it hashes to
    is the entire meaning of the token, and revoking that row revokes the session
    immediately. A signed token carrying its own claims would stay valid after sign-out
    until it expired on its own.

    `secrets`, not `random`: this is a credential.
    """
    return secrets.token_urlsafe(SESSION_ENTROPY_BYTES)


def hash_session_token(token: str) -> str:
    """The digest stored for a session token, and the only form of it that is persisted."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def invalid_credentials() -> HTTPException:
    """The single 401 every failed sign-in gets, identical in every detail.

    One body for "no such address", "wrong password" and "this account is deactivated". A
    caller who could tell them apart could enumerate which addresses have accounts here, and
    learn that an account they hold a stale password for still exists.
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )


def unauthenticated() -> HTTPException:
    """The 401 for a request carrying no usable session.

    No cookie, an unknown token, an expired one, a revoked one and a session whose account
    has since been deactivated are all this same response. The distinctions are real
    server-side and none of them is the caller's business: the answer to every one of them
    is "sign in again".
    """
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
    )


def forbidden() -> HTTPException:
    """The 403 an authenticated user gets for an action their role does not cover.

    Deliberately not a 401. The caller is authenticated, and answering with 401 would tell a
    signed-in user their session had gone bad and send them back to a login they do not need
    — the request failed on authorization, and the status says which.
    """
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Insufficient permissions",
    )


def authenticate(session: Session, email: str, password: str) -> User | None:
    """The account for these credentials, or None — for any reason at all.

    Activity is part of the lookup rather than a check afterwards, so a deactivated account
    simply matches nothing and cannot be distinguished from one that never existed.

    The Argon2 verification runs either way. When no row matched it runs against
    `_UNMATCHED_PASSWORD_HASH` and can only fail; the point is not the answer but the time
    it takes to arrive, which is what stops the response clock from revealing whether the
    address has an account.
    """
    stored = session.execute(
        select(User).where(
            User.email == normalize_email(email), User.is_active.is_(True)
        )
    ).scalar_one_or_none()

    password_hash = stored.password_hash if stored else _UNMATCHED_PASSWORD_HASH

    if not verify_password(password_hash, password):
        return None

    return stored


def start_session(session: Session, user: User) -> str:
    """Open a web session for this user and return its token — the one plaintext copy.

    Signing in revokes whatever session the account already had. One active web session per
    account is a decision, not a limitation of the schema: it means a person who suspects
    their session was taken can end it by signing in again, and it keeps a forgotten session
    on a machine they no longer have from outliving them. It is also why revocation and
    insertion happen in one transaction — a crash between them must not leave an account
    with two live sessions or with none.

    The token is returned rather than written anywhere the caller could read it back later.
    Once this value is dropped, only its digest exists.
    """
    now = datetime.now(timezone.utc)

    session.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user.id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=now)
    )

    token = generate_session_token()
    session.add(
        AuthSession(
            user_id=user.id,
            token_hash=hash_session_token(token),
            expires_at=now + SESSION_LIFETIME,
        )
    )
    session.commit()

    # The user, not the token. Which account signed in is an operational fact worth having
    # in a log; the credential that came out of it is not, and this is the line where it
    # would be easiest to add it by accident.
    logger.info("Opened a web session for user %s.", user.id)

    return token


def revoke_session(session: Session, token: str) -> None:
    """End the session this token names, if it is still open.

    Idempotent and silent about what it found. Signing out twice, or signing out with a
    token that expired an hour ago, is not an error and the caller is told nothing either
    way — a sign-out that reported whether the session had been real would be an oracle for
    testing stolen tokens.
    """
    session.execute(
        update(AuthSession)
        .where(
            AuthSession.token_hash == hash_session_token(token),
            AuthSession.revoked_at.is_(None),
        )
        .values(revoked_at=datetime.now(timezone.utc))
    )
    session.commit()


def session_user(session: Session, token: str) -> User | None:
    """The account this session token authenticates, or None.

    Every condition is in the one query: the token matches, the session was not revoked, it
    has not expired, and the account is still active. Fetching the session and then checking
    those in Python would work and would put four separate chances to forget one between the
    row and the decision — as a `WHERE` clause, a session that fails any of them is never
    read at all.

    Expiry is compared against the database's own clock rather than this process's, so a
    server whose time has drifted cannot extend or shorten a session that another process
    would judge differently.

    Looked up by digest, so no plaintext token appears in the statement or its parameters.
    """
    return session.execute(
        select(User)
        .join(AuthSession, AuthSession.user_id == User.id)
        .where(
            AuthSession.token_hash == hash_session_token(token),
            AuthSession.revoked_at.is_(None),
            AuthSession.expires_at > func.now(),
            User.is_active.is_(True),
        )
    ).scalar_one_or_none()


def session_token(request: Request) -> str | None:
    """The session token out of the request's cookies, or None if it is not there."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return token or None


def set_session_cookie(response: Response, token: str) -> None:
    """Hand the browser its session, with every flag that keeps it there.

    `HttpOnly`, so script on the page cannot read the token — the difference between an XSS
    that defaces a page and one that walks off with a session.

    `SameSite=Lax`, so the cookie is not attached to cross-site POSTs. That is most of what
    CSRF protection is for on this surface, and it is why R1-T1 ships no CSRF tokens: the
    cookie flag covers the case, and a token scheme with nothing extra to protect would be
    machinery for its own sake.

    `Secure` only in production, because a browser will not store a `Secure` cookie sent
    over plain HTTP and localhost is plain HTTP — see `cookie_is_secure`.

    An explicit `max_age` rather than a session cookie the browser decides the lifetime of.
    It is set from the same `SESSION_LIFETIME` the stored row expires on, so the copy in the
    browser and the copy in PostgreSQL stop being valid together. The server's row is still
    the authority; the cookie's expiry only saves a round trip.

    `path="/"` so one cookie covers the whole internal API rather than only the auth routes.
    """
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=int(SESSION_LIFETIME.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=cookie_is_secure(),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Remove the session cookie, with the flags it was set with.

    A browser matches a deletion to the cookie by name, path and domain, so the path here
    has to be the one `set_session_cookie` used or the old cookie survives the sign-out.
    """
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        samesite="lax",
        secure=cookie_is_secure(),
        path="/",
    )


def require_user(
    request: Request,
    session: Session = Depends(get_session),
) -> User:
    """Authenticate a request by its session cookie, or refuse it with 401.

    The dependency internal routes use to demand a signed-in person. It reads a cookie and
    nothing else — no `Authorization` header, no query parameter — so it can only ever
    authenticate a browser session.
    """
    presented = session_token(request)
    if presented is None:
        raise unauthenticated()

    user = session_user(session, presented)
    if user is None:
        raise unauthenticated()

    return user


def require_admin(user: User = Depends(require_user)) -> User:
    """Demand an administrator, on top of `require_user`.

    Layered on the dependency above rather than reimplementing the lookup, so there is one
    place a session is resolved. An unauthenticated caller therefore gets 401 from
    `require_user` before this function runs, and only an authenticated non-administrator
    reaches the 403 — which is the distinction the two statuses are supposed to carry.
    """
    if user.role != USER_ROLE_ADMIN:
        logger.info("User %s was refused an administrative action.", user.id)
        raise forbidden()

    return user
