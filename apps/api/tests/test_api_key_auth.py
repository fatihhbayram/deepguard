"""API key generation, hashing and the 401 the public API refuses callers with.

The dependency is mounted on a throwaway app defined here rather than on a product route.
P9-T1 adds no public endpoints, and inventing one just so the tests have something to call
would ship an endpoint the phase did not ask for.

Most of this needs no database: a fake session is enough to say what the lookup found. The
one thing that does need PostgreSQL is that no column anywhere holds the plaintext, and
that test lives at the bottom behind the integration marker.
"""

import hashlib
import uuid

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.auth import (
    API_KEY_ENTROPY_BYTES,
    API_KEY_PREFIX,
    ApiKeyPrincipal,
    generate_api_key,
    hash_api_key,
    require_api_key,
)
from app.db.models import ApiKey
from app.db.session import SessionLocal, engine, get_session
from app.main import app as production_app

# The one 401 body every rejection produces. Written out rather than read back from the
# first response, so a change to it has to be made here deliberately.
UNAUTHORIZED_BODY = {"detail": "Invalid or missing API key"}


class FakeResult:
    """What `session.execute(...)` hands back: the row the query matched, or nothing."""

    def __init__(self, row):
        self.row = row

    def scalar_one_or_none(self):
        return self.row


class FakeSession:
    """A session that answers every query with one preloaded row, and records the query.

    The recorded statement is how the tests check that the *lookup* filters on activity,
    rather than trusting a fake that was told to return nothing.
    """

    def __init__(self, row=None) -> None:
        self.row = row
        self.statements = []

    def execute(self, statement):
        self.statements.append(statement)
        return FakeResult(self.row)


@pytest.fixture
def app():
    """A minimal app whose single route exists only to be guarded."""
    test_app = FastAPI()

    @test_app.get("/guarded")
    def guarded(principal: ApiKeyPrincipal = Depends(require_api_key)) -> dict:
        return {"id": str(principal.id), "name": principal.name}

    yield test_app
    test_app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def use_session(app: FastAPI, session: FakeSession) -> None:
    app.dependency_overrides[get_session] = lambda: session


def stored_key(**overrides) -> ApiKey:
    """An `api_keys` row as the database would return it."""
    values = {
        "id": uuid.uuid4(),
        "name": "acme-production",
        "key_hash": hash_api_key("dg_live_whatever"),
        "is_active": True,
        "last_used_at": None,
    }
    values.update(overrides)
    return ApiKey(**values)


def assert_generic_unauthorized(response) -> None:
    """The uniform refusal: same status, same body, same challenge, every time."""
    assert response.status_code == 401
    assert response.json() == UNAUTHORIZED_BODY
    assert response.headers["WWW-Authenticate"] == "Bearer"


# --- generation and hashing -------------------------------------------------------------


def test_generated_key_carries_the_live_prefix():
    generated = generate_api_key()

    assert generated.plaintext.startswith(API_KEY_PREFIX)


def test_generated_key_has_at_least_256_bits_of_entropy():
    """The secret half must be a full `token_urlsafe(32)`, not a truncated one.

    Base64url carries 6 bits per character, so 256 bits cannot be expressed in fewer than
    43 of them. Asserting the length is how a future edit that shortens the token — the
    cheap way to make keys prettier — fails here instead of quietly weakening every key.
    """
    secret = generate_api_key().plaintext.removeprefix(API_KEY_PREFIX)

    assert len(secret) * 6 >= API_KEY_ENTROPY_BYTES * 8


def test_generated_keys_are_unique():
    keys = {generate_api_key().plaintext for _ in range(100)}

    assert len(keys) == 100


def test_hash_is_sha256_of_the_full_key_including_the_prefix():
    generated = generate_api_key()

    assert generated.key_hash == hashlib.sha256(
        generated.plaintext.encode("utf-8")
    ).hexdigest()
    assert len(generated.key_hash) == 64


def test_hash_does_not_contain_the_plaintext():
    generated = generate_api_key()

    assert generated.plaintext not in generated.key_hash
    assert generated.plaintext.removeprefix(API_KEY_PREFIX) not in generated.key_hash


def test_hashing_is_deterministic_and_key_specific():
    first = generate_api_key()
    second = generate_api_key()

    assert hash_api_key(first.plaintext) == first.key_hash
    assert first.key_hash != second.key_hash


# --- authenticating -----------------------------------------------------------------------


def test_valid_active_key_returns_the_principal(app, client):
    generated = generate_api_key()
    key = stored_key(key_hash=generated.key_hash)
    use_session(app, FakeSession(row=key))

    response = client.get(
        "/guarded", headers={"Authorization": f"Bearer {generated.plaintext}"}
    )

    assert response.status_code == 200
    assert response.json() == {"id": str(key.id), "name": key.name}


def test_lookup_never_sends_the_plaintext_to_the_database(app, client):
    generated = generate_api_key()
    session = FakeSession(row=stored_key(key_hash=generated.key_hash))
    use_session(app, session)

    client.get("/guarded", headers={"Authorization": f"Bearer {generated.plaintext}"})

    compiled = session.statements[0].compile()
    rendered = str(compiled) + repr(compiled.params)
    assert generated.plaintext not in rendered
    assert generated.key_hash in repr(compiled.params)


def test_lookup_filters_on_activity(app, client):
    """A deactivated key must fail to match, not match and then be rejected.

    The compiled predicate is read, not merely the column name: `is_active` appears in the
    statement the moment the column is selected at all, so a substring check would keep
    passing after the filter itself was dropped. What has to be there is the comparison.

    This says what the *query* asks for. That an inactive row is then genuinely refused is
    a property of PostgreSQL answering it, and is asserted against a real database in
    `test_an_inactive_key_is_refused_by_the_real_lookup` — no fake can establish it, since
    a fake decides for itself what to return.
    """
    generated = generate_api_key()
    session = FakeSession(row=stored_key(key_hash=generated.key_hash))
    use_session(app, session)

    client.get("/guarded", headers={"Authorization": f"Bearer {generated.plaintext}"})

    where = str(session.statements[0].whereclause)
    assert "api_keys.is_active IS true" in where


def test_bearer_scheme_is_case_insensitive(app, client):
    generated = generate_api_key()
    use_session(app, FakeSession(row=stored_key(key_hash=generated.key_hash)))

    response = client.get(
        "/guarded", headers={"Authorization": f"bEaReR {generated.plaintext}"}
    )

    assert response.status_code == 200


# --- every refusal is the same refusal -----------------------------------------------------


def test_missing_header_is_rejected(app, client):
    use_session(app, FakeSession(row=None))

    assert_generic_unauthorized(client.get("/guarded"))


@pytest.mark.parametrize(
    "header",
    [
        "",
        "dg_live_abcdef",
        "Bearer",
        "Bearer ",
        "Basic dg_live_abcdef",
        "Token dg_live_abcdef",
        "Bearer  ",
    ],
    ids=[
        "empty",
        "no-scheme",
        "scheme-only",
        "scheme-and-space",
        "basic-scheme",
        "token-scheme",
        "whitespace-token",
    ],
)
def test_malformed_header_is_rejected(app, client, header):
    use_session(app, FakeSession(row=None))

    assert_generic_unauthorized(
        client.get("/guarded", headers={"Authorization": header})
    )


def test_unknown_key_is_rejected(app, client):
    use_session(app, FakeSession(row=None))

    assert_generic_unauthorized(
        client.get(
            "/guarded",
            headers={"Authorization": f"Bearer {generate_api_key().plaintext}"},
        )
    )


def test_key_hash_presented_as_the_key_is_rejected(app, client):
    """Knowing the stored digest must not be enough to authenticate with it."""
    generated = generate_api_key()
    use_session(app, FakeSession(row=None))

    assert_generic_unauthorized(
        client.get("/guarded", headers={"Authorization": f"Bearer {generated.key_hash}"})
    )


def test_all_rejections_are_byte_identical(app, client):
    """Missing, malformed and unknown must be indistinguishable to the caller.

    Compared as whole responses rather than four separate assertions on the same
    constants, because the property under test is that they cannot be told apart.

    A deactivated key is the fourth member of that set and cannot be staged here: the fake
    session returns whatever it was handed regardless of the filter, so an "inactive" case
    written against it would be the unknown-key case under another name. It is compared
    against the same refusal in `test_an_inactive_key_is_refused_by_the_real_lookup`, where
    PostgreSQL is the one excluding the row.
    """
    use_session(app, FakeSession(row=None))
    unknown = generate_api_key().plaintext

    responses = [
        client.get("/guarded"),
        client.get("/guarded", headers={"Authorization": "Bearer"}),
        client.get("/guarded", headers={"Authorization": f"Basic {unknown}"}),
        client.get("/guarded", headers={"Authorization": f"Bearer {unknown}"}),
    ]

    signatures = {
        (r.status_code, r.text, r.headers.get("WWW-Authenticate")) for r in responses
    }
    assert len(signatures) == 1


def test_rejection_never_echoes_the_presented_key(app, client):
    use_session(app, FakeSession(row=None))
    presented = generate_api_key().plaintext

    response = client.get("/guarded", headers={"Authorization": f"Bearer {presented}"})

    assert presented not in response.text
    assert hash_api_key(presented) not in response.text


# --- existing auth and existing routes are untouched ----------------------------------------


def walk_routes(routes):
    """Every route in the app, including those behind an included router.

    `include_router` does not splice the child routes into `app.routes`; it appends one
    wrapper object that holds them. Iterating `app.routes` alone therefore reaches only
    `/health`, and a check written that way would pass by never looking at the analyses
    routes at all.

    The wrapper is unwrapped through `effective_route_contexts()`, not through
    `original_router`. The latter hands back the child router's own routes, whose
    `dependant` does not carry dependencies attached at `include_router(...)` time — so a
    gate applied to the whole router would be invisible there. The effective contexts are
    what the app actually dispatches with, which is what a security check has to read.
    """
    for route in routes:
        if hasattr(route, "dependant"):
            yield route
        contexts = getattr(route, "effective_route_contexts", None)
        if contexts is not None:
            yield from contexts()


def test_existing_routes_do_not_require_an_api_key():
    """The product app gained a credential, not a gate.

    `/health` and the analyses routes answered unauthenticated before P9-T1 and still do;
    a dependency accidentally mounted app-wide would show up here as a 401.
    """
    def requires_api_key(dependant) -> bool:
        if dependant.call is require_api_key:
            return True
        return any(requires_api_key(child) for child in dependant.dependencies)

    inspected = list(walk_routes(production_app.routes))
    paths = {route.path for route in inspected}

    # The routes this test exists to cover. Asserted explicitly so that a future FastAPI
    # release changing how included routers are stored breaks the test loudly instead of
    # narrowing it back down to `/health`.
    assert {"/health", "/api/v1/analyses", "/api/v1/analyses/{analysis_id}"} <= paths

    guarded = [
        route.path for route in inspected if requires_api_key(route.dependant)
    ]

    assert guarded == []


def test_health_still_answers_without_credentials():
    class OkSession:
        def execute(self, statement):
            return None

    production_app.dependency_overrides[get_session] = lambda: OkSession()
    try:
        with TestClient(production_app) as client:
            response = client.get("/health")
    finally:
        production_app.dependency_overrides.clear()

    assert response.status_code == 200


# --- against real PostgreSQL ---------------------------------------------------------------

integration = pytest.mark.integration


@pytest.fixture(scope="module")
def database():
    """The live engine, or a skip when this environment has no PostgreSQL."""
    try:
        with engine.connect():
            pass
    except SQLAlchemyError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error.__class__.__name__}")

    return engine


@pytest.fixture
def session(database):
    """A real session whose key rows are removed again."""
    created = []

    with SessionLocal() as session:
        yield session, created

        session.rollback()
        for key_id in created:
            session.query(ApiKey).filter(ApiKey.id == key_id).delete()
        session.commit()


@integration
def test_migration_created_the_table_with_a_unique_hash_index(database):
    inspector = inspect(database)

    assert "api_keys" in inspector.get_table_names()

    columns = {column["name"] for column in inspector.get_columns("api_keys")}
    assert columns == {
        "id",
        "name",
        "key_hash",
        "created_at",
        "is_active",
        "last_used_at",
    }

    unique_hash = [
        index
        for index in inspector.get_indexes("api_keys")
        if index["column_names"] == ["key_hash"] and index["unique"]
    ]
    assert unique_hash, "key_hash must carry a unique index"


@integration
def test_a_persisted_key_stores_no_plaintext_anywhere(session):
    """The whole row, read back column by column, must not contain the key.

    Read as raw text rather than through the model, so a column added later that happens
    to capture the plaintext fails this test instead of slipping past a fixed list of
    fields.
    """
    db, created = session
    generated = generate_api_key()

    key = ApiKey(name="plaintext-check", key_hash=generated.key_hash)
    db.add(key)
    db.commit()
    created.append(key.id)

    row = db.execute(
        text("SELECT api_keys::text AS whole_row FROM api_keys WHERE id = :id"),
        {"id": key.id},
    ).scalar_one()

    assert generated.key_hash in row
    assert generated.plaintext not in row
    assert generated.plaintext.removeprefix(API_KEY_PREFIX) not in row


@integration
def test_a_persisted_key_defaults_to_active_and_unused(session):
    db, created = session

    key = ApiKey(name="defaults-check", key_hash=generate_api_key().key_hash)
    db.add(key)
    db.commit()
    created.append(key.id)
    db.refresh(key)

    assert key.is_active is True
    assert key.last_used_at is None
    assert key.created_at is not None


@integration
def test_authenticating_does_not_touch_last_used_at(session):
    """P9-T1 explicitly does not record usage; this is what would catch it starting to."""
    db, created = session
    generated = generate_api_key()

    key = ApiKey(name="last-used-check", key_hash=generated.key_hash)
    db.add(key)
    db.commit()
    created.append(key.id)

    app = FastAPI()

    @app.get("/guarded")
    def guarded(principal: ApiKeyPrincipal = Depends(require_api_key)) -> dict:
        return {"name": principal.name}

    app.dependency_overrides[get_session] = lambda: db

    with TestClient(app) as client:
        response = client.get(
            "/guarded", headers={"Authorization": f"Bearer {generated.plaintext}"}
        )

    assert response.status_code == 200

    db.expire_all()
    assert db.get(ApiKey, key.id).last_used_at is None


@integration
def test_two_keys_cannot_share_a_hash(session):
    from sqlalchemy.exc import IntegrityError

    db, created = session
    generated = generate_api_key()

    first = ApiKey(name="first", key_hash=generated.key_hash)
    db.add(first)
    db.commit()
    created.append(first.id)

    db.add(ApiKey(name="second", key_hash=generated.key_hash))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


@integration
def test_an_inactive_key_is_refused_by_the_real_lookup(session):
    """A key that exists but was deactivated must be refused, and refused identically.

    The only test in this file where the inactive case is real: the row is written with
    `is_active = false`, and it is PostgreSQL applying the dependency's filter that
    excludes it. The unknown key is sent through the same app in the same state so the two
    refusals can be compared as whole responses — if a deactivated key were ever to fail
    differently from one that was never issued, that difference would tell an
    unauthenticated caller that the key it holds names a real customer.
    """
    db, created = session
    generated = generate_api_key()

    key = ApiKey(name="deactivated", key_hash=generated.key_hash, is_active=False)
    db.add(key)
    db.commit()
    created.append(key.id)

    app = FastAPI()

    @app.get("/guarded")
    def guarded(principal: ApiKeyPrincipal = Depends(require_api_key)) -> dict:
        return {"name": principal.name}

    app.dependency_overrides[get_session] = lambda: db

    with TestClient(app) as client:
        response = client.get(
            "/guarded", headers={"Authorization": f"Bearer {generated.plaintext}"}
        )
        never_issued = client.get(
            "/guarded",
            headers={"Authorization": f"Bearer {generate_api_key().plaintext}"},
        )

    assert_generic_unauthorized(response)

    def signature(refusal):
        return (
            refusal.status_code,
            refusal.text,
            refusal.headers.get("WWW-Authenticate"),
            refusal.headers.get("content-type"),
        )

    assert signature(response) == signature(never_issued)


@integration
def test_analyses_listing_still_answers_without_credentials(session):
    """The structural check says no analyses route depends on `require_api_key`; this
    calls one without an `Authorization` header and confirms it is not a 401 in practice.
    """
    db, _ = session

    production_app.dependency_overrides[get_session] = lambda: db
    try:
        with TestClient(production_app) as client:
            response = client.get("/api/v1/analyses")
    finally:
        production_app.dependency_overrides.clear()

    assert response.status_code == 200
