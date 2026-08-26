"""API key authentication for the public API.

Three things live here: how a key is minted, how a presented key is turned into the digest
stored in `api_keys`, and the dependency a public endpoint uses to demand one. No endpoint
uses the dependency yet — P9-T1 is the credential, not the API it will guard.

The plaintext key leaves this module exactly once, in the return value of `generate_api_key`,
and is never written anywhere: not to the database, not to a log line, not into an error
body. Everything downstream works from the digest.

Every rejection is the same rejection. A caller who sends no header, a malformed one, a key
that was never issued and a key that has been deactivated all get an identical 401 — same
status, same body, same headers. Distinguishing them would answer questions an unauthenticated
caller has no business asking: whether a guessed key exists at all, and whether an old key
still names a real customer. That uniformity is a security property, so it is asserted in the
tests rather than left to look after itself.
"""

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ApiKey
from app.db.session import get_session

# What every issued key starts with, so a leaked string is recognisable as a DeepGuard
# credential on sight — in a paste, a log, or a repository scanner's ruleset. `live` marks
# the environment the key is good for; nothing yet issues any other kind.
API_KEY_PREFIX = "dg_live_"

# Bytes of entropy behind the secret half. 32 bytes is 256 bits, which is the floor this
# task sets and is not negotiable downward: it is what makes guessing a key hopeless and
# therefore what lets the stored digest be a plain SHA-256.
API_KEY_ENTROPY_BYTES = 32

# The scheme the key is presented under: `Authorization: Bearer <key>`. Compared
# case-insensitively, because RFC 7235 makes the scheme token case-insensitive and clients
# do vary.
BEARER_SCHEME = "bearer"


@dataclass(frozen=True)
class GeneratedApiKey:
    """A freshly minted key: the one copy of the plaintext, and what to store instead.

    The two are returned together because they are only ever useful together — the caller
    persists `key_hash` and hands `plaintext` to its owner. Once this object is discarded
    the plaintext is unrecoverable, which is the point.
    """

    plaintext: str
    key_hash: str


@dataclass(frozen=True)
class ApiKeyPrincipal:
    """Who a request authenticated as.

    Deliberately small: the key's identity and its human-readable name, and no secret. It
    carries no permissions because P9-T1 issues no permissions — every valid key is equal,
    and a scope model belongs to the phase that has something to scope.
    """

    id: uuid.UUID
    name: str


def hash_api_key(key: str) -> str:
    """The digest stored for a key — SHA-256 of the *full* string, prefix included.

    Hashing the whole presented value rather than the secret half means verification never
    has to split the string, so a caller cannot reach a stored row by presenting the same
    secret under a different prefix.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_api_key() -> GeneratedApiKey:
    """Mint a new key: `dg_live_` followed by 256 bits from the system CSPRNG.

    `secrets.token_urlsafe` rather than `random` — this is a credential, and the difference
    between the two modules is the difference between unguessable and merely unpredictable-
    looking.
    """
    plaintext = API_KEY_PREFIX + secrets.token_urlsafe(API_KEY_ENTROPY_BYTES)
    return GeneratedApiKey(plaintext=plaintext, key_hash=hash_api_key(plaintext))


def unauthenticated() -> HTTPException:
    """The single 401 every failure path raises, identical in every detail."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def bearer_token(request: Request) -> str | None:
    """The credential out of `Authorization: Bearer <key>`, or None if it is not there.

    Anything that is not exactly one `Bearer` scheme followed by one non-empty token is
    None. The caller turns that into the same 401 an unknown key gets, so a malformed
    header is not a distinguishable outcome.
    """
    header = request.headers.get("Authorization")
    if not header:
        return None

    scheme, _, token = header.partition(" ")
    if scheme.lower() != BEARER_SCHEME:
        return None

    token = token.strip()
    return token or None


def require_api_key(
    request: Request,
    session: Session = Depends(get_session),
) -> ApiKeyPrincipal:
    """Authenticate a request by its API key, or refuse it.

    The presented key is hashed and looked up by digest — one indexed lookup, no scan, and
    no plaintext anywhere in the query. Activity is part of the lookup rather than a
    second check afterwards, so a deactivated key simply matches nothing.

    The digest is compared again with `hmac.compare_digest` after the lookup. PostgreSQL
    has already decided equality by then, so this settles nothing about *which* row was
    found; it is there because the row was fetched by a value derived from attacker-supplied
    input, and a constant-time confirmation of that is cheap insurance against the index
    lookup being made to leak by timing.
    """
    presented = bearer_token(request)
    if presented is None:
        raise unauthenticated()

    key_hash = hash_api_key(presented)

    api_key = session.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
    ).scalar_one_or_none()

    if api_key is None or not hmac.compare_digest(api_key.key_hash, key_hash):
        raise unauthenticated()

    return ApiKeyPrincipal(id=api_key.id, name=api_key.name)
