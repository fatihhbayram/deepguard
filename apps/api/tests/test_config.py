"""The startup gate on required credentials (R1-T5).

Two things are worth proving here and they are different from each other.

The unit tests below are about `app.config` itself: what counts as configured, what the
refusal says, and that `DATABASE_URL` stands in for the three PostgreSQL variables the way
`app.db.session` actually resolves them.

The last test is the one that matters operationally, and it is a subprocess. The gate lives
in `app/__init__.py` and fires at package import, which is a thing that has already happened
by the time any test in this suite runs — asserting on it in-process would only ever be
asserting that this environment is configured, which it is. So a fresh interpreter is started
with a cleared environment and asked to import the application, and what is checked is that it
exits non-zero saying which variable is missing. That is exactly what uvicorn does in the
container, and it is the only way to see it from here.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import config

# Where `app/` lives, so the subprocess below can import it without an installed package.
API_ROOT = Path(__file__).resolve().parent.parent


def test_is_set_rejects_absent_empty_and_whitespace(monkeypatch):
    """Only an actual value counts. Empty is what a compose pass-through produces."""
    monkeypatch.delenv("EXAMPLE_SECRET", raising=False)
    assert not config.is_set("EXAMPLE_SECRET")

    monkeypatch.setenv("EXAMPLE_SECRET", "")
    assert not config.is_set("EXAMPLE_SECRET")

    monkeypatch.setenv("EXAMPLE_SECRET", "   ")
    assert not config.is_set("EXAMPLE_SECRET")

    monkeypatch.setenv("EXAMPLE_SECRET", "value")
    assert config.is_set("EXAMPLE_SECRET")


def test_require_returns_the_value_unchanged(monkeypatch):
    """Surrounding whitespace is part of a password and is not trimmed away."""
    monkeypatch.setenv("EXAMPLE_SECRET", " pa ss ")

    assert config.require("EXAMPLE_SECRET") == " pa ss "


def test_require_names_the_variable_it_refused(monkeypatch):
    """The refusal has to be actionable: the message is all the operator gets."""
    monkeypatch.delenv("EXAMPLE_SECRET", raising=False)

    with pytest.raises(config.MissingConfiguration) as refusal:
        config.require("EXAMPLE_SECRET")

    assert "EXAMPLE_SECRET" in str(refusal.value)


def test_missing_reports_every_absent_credential(monkeypatch):
    """One restart should teach an operator the whole set, not the first name in it."""
    monkeypatch.delenv(config.DATABASE_URL_VARIABLE, raising=False)
    for variable in config.DATABASE_CREDENTIALS + config.OBJECT_STORAGE_CREDENTIALS:
        monkeypatch.delenv(variable, raising=False)

    assert set(config.missing()) == set(
        config.DATABASE_CREDENTIALS + config.OBJECT_STORAGE_CREDENTIALS
    )


def test_database_url_stands_in_for_the_postgres_credentials(monkeypatch):
    """A complete URL carries the credentials, so demanding them too is a false alarm.

    This is not hypothetical: it is how `tests/conftest.py` binds this suite to its own
    database, and a check that ignored `DATABASE_URL` would refuse to start the suite.
    """
    monkeypatch.setenv(config.DATABASE_URL_VARIABLE, "postgresql+psycopg2://u:p@host/db")
    for variable in config.DATABASE_CREDENTIALS:
        monkeypatch.delenv(variable, raising=False)
    for variable in config.OBJECT_STORAGE_CREDENTIALS:
        monkeypatch.setenv(variable, "configured")

    assert config.missing() == ()


def test_validate_passes_when_everything_is_configured(monkeypatch):
    monkeypatch.delenv(config.DATABASE_URL_VARIABLE, raising=False)
    for variable in config.DATABASE_CREDENTIALS + config.OBJECT_STORAGE_CREDENTIALS:
        monkeypatch.setenv(variable, "configured")

    config.validate()


def test_validate_refuses_and_lists_what_is_missing(monkeypatch):
    monkeypatch.delenv(config.DATABASE_URL_VARIABLE, raising=False)
    for variable in config.DATABASE_CREDENTIALS:
        monkeypatch.setenv(variable, "configured")
    monkeypatch.delenv(config.MINIO_ROOT_PASSWORD_VARIABLE, raising=False)
    monkeypatch.setenv(config.MINIO_ROOT_USER_VARIABLE, "configured")

    with pytest.raises(config.MissingConfiguration) as refusal:
        config.validate()

    assert config.MINIO_ROOT_PASSWORD_VARIABLE in str(refusal.value)


def test_importing_the_application_without_credentials_exits_non_zero(tmp_path):
    """The gate, seen the way uvicorn sees it: a process that will not start.

    Run from `tmp_path` rather than the repository, because `app.config` calls `load_dotenv`
    and would otherwise find the developer's own `.env` by walking up from the working
    directory — which would configure the very process this test needs unconfigured.

    `PYTHONPATH` is what makes `app` importable from there. The environment is otherwise
    emptied down to the few variables an interpreter needs, so nothing inherited from this
    suite can satisfy the check by accident.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(API_ROOT),
            "HOME": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "MissingConfiguration" in result.stderr
    assert config.POSTGRES_USER_VARIABLE in result.stderr
