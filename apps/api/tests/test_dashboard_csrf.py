"""The origin check in front of the dashboard's mutations, and its edges.

A `SameSite=Lax` cookie already keeps a browser from attaching a session to a cross-site
POST, so this check is the second, independent boundary behind that one — and it is the one
enforced by the server that owns the data. That distinction is the whole reason it exists at
the API rather than only in the web application: cookies are not scoped by port, so a page on
any origin the same browser can reach may post straight at this API with the session
attached, going round the Next.js handler entirely. A check that lived only in front of the
proxy would be a check an attacker simply does not use.

Three properties are what this file is for:

1. A mutation from an origin this deployment does not accept is refused — and refused with
   nothing persisted, fetched or stored on the way.
2. A mutation with no `Origin` at all is refused too. Every browser sends one on a POST, so
   absence is not a browser; treating it as "no opinion" would leave a check anyone can skip
   by omitting a header.
3. The check is on the dashboard's mutations and nowhere else: not on the reads, and not on
   the public API, which authenticates by a header no browser attaches on its own.

The session is faked and the user dependency is overridden, because none of that is what is
under test here — what is under test is a header comparison in front of routes that would
otherwise succeed. `tests/test_dashboard_authorization.py` runs the same routes against real
accounts and real rows.
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from app import media, storage
from app.auth import generate_api_key
from app.db.models import USER_ROLE_USER, Analysis, ApiKey, User
from app.db.session import get_session
from app.main import app
from app.web_auth import (
    WEB_ORIGIN_VARIABLE,
    allowed_origins,
    normalize_origin,
    require_user,
)
from tests.conftest import DASHBOARD_ORIGIN

UPLOAD_URL = "/api/v1/analyses"
URL_SUBMISSION_URL = "/api/v1/analyses/url"
LOGOUT_URL = "/api/v1/auth/logout"
PUBLIC_UPLOAD_URL = "/api/public/v1/analyses"

# An origin that is not this deployment's. It is a well-formed one on purpose: the check must
# turn away a valid origin that is merely the wrong one, which is exactly what a forged
# submission carries.
FOREIGN_ORIGIN = "https://evil.example"

CROSS_ORIGIN_BODY = {"detail": "Cross-origin request refused"}


# --- the allowlist itself ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "http://dashboard.test",
        "http://Dashboard.Test",
        "http://dashboard.test/",
        "  http://dashboard.test  ",
    ],
    ids=["plain", "mixed-case", "trailing-slash", "padded"],
)
def test_an_origin_is_normalized_before_it_is_compared(raw):
    """Case and a trailing slash are spelling, not identity.

    A browser sends the plain form; an operator writing the allowlist will sooner or later
    paste one of the others, and an allowlist that failed to match would refuse every real
    submission while looking configured.
    """
    assert normalize_origin(raw) == "http://dashboard.test"


@pytest.mark.parametrize(
    "other",
    ["https://dashboard.test", "http://dashboard.test:3000", "http://other.test"],
    ids=["scheme", "port", "host"],
)
def test_the_scheme_host_and_port_are_all_part_of_the_origin(other):
    """None of the three is normalized away. Each names a different origin, and a browser
    treats them as different — an allowlist that ignored the port would accept a page served
    by anything else on the same host."""
    assert normalize_origin(other) != normalize_origin(DASHBOARD_ORIGIN)


def test_several_origins_may_be_configured(monkeypatch):
    monkeypatch.setenv(
        WEB_ORIGIN_VARIABLE, f"{DASHBOARD_ORIGIN}, https://inspectroot.example.com/"
    )

    assert allowed_origins() == {DASHBOARD_ORIGIN, "https://inspectroot.example.com"}


@pytest.mark.parametrize("configured", ["", "   ", " , "], ids=["empty", "blank", "commas"])
def test_an_unconfigured_deployment_allows_no_origin(monkeypatch, configured):
    """The fail-secure default, asserted rather than assumed.

    An empty allowlist refuses every dashboard mutation. The opposite default — nothing
    configured means everything accepted — is the one an operator would never notice,
    because a protection that is off looks exactly like one that is on until somebody uses
    it against them.
    """
    monkeypatch.setenv(WEB_ORIGIN_VARIABLE, configured)

    assert allowed_origins() == frozenset()


def test_an_unset_variable_allows_no_origin(monkeypatch):
    monkeypatch.delenv(WEB_ORIGIN_VARIABLE, raising=False)

    assert allowed_origins() == frozenset()


# --- the routes -------------------------------------------------------------------------


class RecordingSession:
    """A session that would let any of these routes succeed, and records what reached it.

    Deliberately permissive: the point of every route test below is that nothing was
    persisted, and a session that could not have persisted anything would prove nothing.
    """

    def __init__(self) -> None:
        self.added = []
        self.commits = 0

    def execute(self, statement):
        class Result:
            def scalar_one_or_none(self):
                return None

            def scalar_one(self):
                return 0

            def all(self):
                return []

        return Result()

    def add(self, instance):
        self.added.append(instance)

    def flush(self):
        for instance in self.added:
            if getattr(instance, "id", None) is None:
                instance.id = uuid.uuid4()

    def commit(self):
        self.flush()
        self.commits += 1

    def rollback(self):
        pass


class FakeMinio:
    def __init__(self) -> None:
        self.uploads = []

    def bucket_exists(self, bucket):
        return True

    def make_bucket(self, bucket):
        pass

    def fput_object(self, bucket, key, file_path, content_type=None):
        self.uploads.append((bucket, key))


@pytest.fixture
def object_store(monkeypatch):
    fake = FakeMinio()
    monkeypatch.setattr(storage, "client", fake)

    return fake


@pytest.fixture(autouse=True)
def fake_ffprobe(monkeypatch):
    probe = json.dumps(
        {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "12.34",
                    "pix_fmt": "yuv420p",
                    "avg_frame_rate": "30/1",
                    "r_frame_rate": "30/1",
                }
            ],
            "format": {
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                "duration": "12.34",
                "tags": {"major_brand": "mp42"},
            },
        }
    )

    async def run(path):
        return probe

    monkeypatch.setattr(media, "_run_ffprobe", run)


@pytest.fixture
def downloads(monkeypatch):
    """Records whether the URL route ever got as far as fetching anything."""
    from app import downloader

    fetched = []

    def download(url):
        fetched.append(url)
        raise AssertionError("the download must not be reached")

    monkeypatch.setattr(downloader, "download", download)

    return fetched


@pytest.fixture
def dashboard_session():
    session = RecordingSession()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[require_user] = lambda: User(
        id=uuid.uuid4(),
        email="operator@example.com",
        password_hash="unused",
        role=USER_ROLE_USER,
        is_active=True,
    )

    yield session

    app.dependency_overrides.clear()


@pytest.fixture
def client(dashboard_session):
    with TestClient(app) as test_client:
        yield test_client


def origin_header(origin: str | None) -> dict:
    return {} if origin is None else {"Origin": origin}


def upload(client, origin: str | None):
    return client.post(
        UPLOAD_URL,
        files={"file": ("clip.mp4", b"pretend-this-is-video", "video/mp4")},
        headers=origin_header(origin),
    )


def submit_url(client, origin: str | None):
    return client.post(
        URL_SUBMISSION_URL,
        json={"url": "https://videos.example.com/clip"},
        headers=origin_header(origin),
    )


def sign_out(client, origin: str | None):
    return client.post(LOGOUT_URL, headers=origin_header(origin))


REFUSED_ORIGINS = pytest.mark.parametrize(
    "origin",
    [None, FOREIGN_ORIGIN, "http://dashboard.test.evil.example", "not-an-origin", ""],
    ids=["absent", "foreign", "lookalike", "malformed", "empty"],
)


@REFUSED_ORIGINS
def test_an_upload_from_an_unaccepted_origin_is_refused(
    client, dashboard_session, object_store, origin
):
    """Refused, and refused without keeping anything.

    The status alone would not be enough. What makes this a CSRF defence rather than a
    cosmetic 403 is that the forged request leaves no analysis behind and no object in
    storage — a route that stored the upload and then refused it would have done the
    attacker's work and merely declined to say so.
    """
    response = upload(client, origin)

    assert response.status_code == 403
    assert response.json() == CROSS_ORIGIN_BODY
    assert dashboard_session.added == []
    assert dashboard_session.commits == 0
    assert object_store.uploads == []


@REFUSED_ORIGINS
def test_a_url_submission_from_an_unaccepted_origin_is_refused(
    client, dashboard_session, downloads, origin
):
    """And nothing is fetched. This route makes the server issue a request of its own, so a
    forged submission that got as far as the downloader would be an outbound fetch chosen by
    whoever wrote the page — the SSRF guard's job, and not a burden it should be handed by a
    request that was never legitimate."""
    response = submit_url(client, origin)

    assert response.status_code == 403
    assert response.json() == CROSS_ORIGIN_BODY
    assert downloads == []
    assert dashboard_session.added == []


@REFUSED_ORIGINS
def test_a_sign_out_from_an_unaccepted_origin_is_refused(client, origin):
    """Signing somebody out is a smaller harm than filing an analysis in their name, and it
    is still something a page they merely visited has no business doing to them."""
    response = sign_out(client, origin)

    assert response.status_code == 403
    assert response.json() == CROSS_ORIGIN_BODY


def test_the_dashboards_own_origin_is_accepted(client, object_store):
    """The other half: the check must let the real dashboard through, or it is just an
    outage with a security-shaped explanation."""
    assert upload(client, DASHBOARD_ORIGIN).status_code == 202
    assert sign_out(client, DASHBOARD_ORIGIN).status_code == 204


def test_the_accepted_origin_is_matched_case_insensitively(client, object_store):
    assert upload(client, DASHBOARD_ORIGIN.upper()).status_code == 202


def test_an_unconfigured_deployment_refuses_its_own_dashboard(
    client, object_store, monkeypatch
):
    """With no allowlist configured, even the right origin is refused.

    The visible consequence of failing secure, stated as a test so nobody quietly "fixes" it
    by treating an empty allowlist as permission. A deployment in this state has submissions
    that do not work, which an operator finds in seconds; the alternative is one that accepts
    them from anywhere, which nobody finds at all.
    """
    monkeypatch.delenv(WEB_ORIGIN_VARIABLE, raising=False)

    assert upload(client, DASHBOARD_ORIGIN).status_code == 403


# --- where the check deliberately is not ------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/api/v1/analyses", "/api/v1/analyses/00000000-0000-0000-0000-000000000000"],
    ids=["listing", "detail"],
)
def test_the_reads_do_not_require_an_origin(client, path):
    """A GET changes nothing, and cannot be read cross-origin anyway — no CORS headers are
    served. Requiring the header here would make an origin a condition of *looking* at the
    dashboard, and would break every server-side render the web application does."""
    assert client.get(path).status_code in {200, 404}


def test_signing_in_does_not_require_an_origin(client):
    """Login is not one of the mutations this task put behind the check.

    It carries no session to abuse — there is nothing to forge on behalf of a user who is
    not signed in yet — and the credentials still have to be right. The web application's
    own same-origin check stands in front of it, which is where a login-CSRF defence belongs
    for now; putting one here as well would be R1-T2 widening its own scope.
    """
    response = client.post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "x"}
    )

    assert response.status_code == 401


def test_the_public_api_is_untouched_by_the_origin_check(object_store, monkeypatch):
    """A customer's server sends no `Origin`, and must keep working.

    The public surface authenticates by a header a browser will not attach on its own, so
    there is no cross-site request for an origin check to prevent — and a check here would
    break every integration on the day it shipped.
    """
    session = RecordingSession()
    generated = generate_api_key()
    session.api_key = ApiKey(
        id=uuid.uuid4(), name="acme", key_hash=generated.key_hash, is_active=True
    )

    # The public route authenticates by looking a key up, so the fake has to return one.
    def execute(statement):
        key = session.api_key

        class Result:
            def scalar_one_or_none(self):
                return key

            def scalar_one(self):
                return 0

        return Result()

    session.execute = execute
    app.dependency_overrides[get_session] = lambda: session

    try:
        with TestClient(app) as client:
            response = client.post(
                PUBLIC_UPLOAD_URL,
                files={"file": ("clip.mp4", b"pretend-this-is-video", "video/mp4")},
                headers={"Authorization": f"Bearer {generated.plaintext}"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    persisted = next(row for row in session.added if isinstance(row, Analysis))
    assert persisted.api_key_id == session.api_key.id
    assert persisted.owner_id is None
