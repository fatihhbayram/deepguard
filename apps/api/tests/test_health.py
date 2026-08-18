import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_session
from app.main import app


class FakeSession:
    """Minimal stand-in for a SQLAlchemy session, so tests need no live database."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def execute(self, statement):
        if self.error is not None:
            raise self.error
        return None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def override_session(session: FakeSession) -> None:
    app.dependency_overrides[get_session] = lambda: session


def test_health_returns_ok_when_database_is_reachable(client):
    override_session(FakeSession())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_returns_degraded_when_database_is_unavailable(client):
    override_session(
        FakeSession(OperationalError("SELECT 1", None, Exception("connection refused")))
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}
