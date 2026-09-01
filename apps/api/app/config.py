"""Every credential this service refuses to start without (R1-T5).

Until now each secret was read where it was needed and each read carried a development
default beside it: `POSTGRES_PASSWORD` fell back to `deepguard`, `MINIO_ROOT_PASSWORD` to
`deepguard123`. Those defaults are what this module removes. They were convenient exactly
once — on the machine the values were invented on — and everywhere else they were a way for
a deployment to come up *working* against the wrong credentials: a production stack that had
never been given a password would connect happily to a PostgreSQL that had been created with
the same forgotten default, and nothing in the logs would say so.

So the rule here is that a credential is either configured or the process does not run.
Three properties make that a rule rather than a preference:

- **there are no defaults for secrets.** `require` raises; it has no second argument. A
  variable that is not set is not substituted, guessed, or read from a neighbouring one;
- **empty is missing.** A compose file that lists a variable the host never set passes it
  through as the empty string, so `POSTGRES_PASSWORD=` reaches the process looking configured
  and connects as a passwordless user. Treating that as absence is the whole point of reading
  the environment strictly, and it is the case that actually happens;
- **the failure is at startup and names every variable at once.** `validate` is called from
  `app/__init__.py`, which Python runs before any module inside the package — so both the API
  and the worker refuse to start, together, with one message listing everything that is
  missing rather than one exception per import about whichever was read first.

What is deliberately *not* here:

- **topology.** `POSTGRES_HOST`, `POSTGRES_PORT`, `MINIO_ENDPOINT` and `MINIO_SECURE` keep
  their defaults. They are not secrets, a wrong value fails loudly on the first connection
  rather than quietly succeeding against something it should not reach, and compose sets all
  four explicitly anyway;
- **the detector credentials.** `NVIDIA_API_KEY`, the two function ids and `HUGGINGFACE_TOKEN`
  are optional by design and documented as such: without them an analysis still completes and
  records a failed or unavailable signal for the detector that could not run. Making them
  required would turn a degraded signal into a stack that will not boot;
- **`DEEPGUARD_ENV` and `DEEPGUARD_WEB_ORIGIN`.** Both already fail secure when unset — the
  session cookie is marked `Secure` and every dashboard mutation is refused — and a fail-secure
  default is a stronger guarantee than a startup check, because it cannot be satisfied by
  setting the variable to the wrong thing.
"""

import os

from dotenv import load_dotenv

# The same `.env` the rest of the package loads, read here first because this module is what
# `app/__init__.py` calls before anything else in the package is imported. Loading it more
# than once is harmless: `load_dotenv` does not overwrite a variable that is already set, so
# the real environment a container is started with always wins over a file.
load_dotenv()

# An explicit, fully-formed connection string. When it is set it carries the credentials
# itself and the three variables below are not consulted — `app.db.session` returns it
# unchanged — so demanding them as well would refuse a deployment that is perfectly
# configured. The test suite is the reader that makes this matter: `tests/conftest.py` binds
# the suite to its own database this way.
DATABASE_URL_VARIABLE = "DATABASE_URL"

# The PostgreSQL account and database `app.db.session` builds its URL from.
POSTGRES_USER_VARIABLE = "POSTGRES_USER"
POSTGRES_PASSWORD_VARIABLE = "POSTGRES_PASSWORD"
POSTGRES_DB_VARIABLE = "POSTGRES_DB"

DATABASE_CREDENTIALS = (
    POSTGRES_USER_VARIABLE,
    POSTGRES_PASSWORD_VARIABLE,
    POSTGRES_DB_VARIABLE,
)

# The MinIO root credentials `app.storage` authenticates with. Object storage holds the
# originals, which are the forensic artifacts this service exists to keep byte-for-byte, so
# these are not a lesser class of secret than the database's.
MINIO_ROOT_USER_VARIABLE = "MINIO_ROOT_USER"
MINIO_ROOT_PASSWORD_VARIABLE = "MINIO_ROOT_PASSWORD"

OBJECT_STORAGE_CREDENTIALS = (
    MINIO_ROOT_USER_VARIABLE,
    MINIO_ROOT_PASSWORD_VARIABLE,
)


class MissingConfiguration(RuntimeError):
    """A credential this process cannot run without was not configured."""


def is_set(variable: str) -> bool:
    """Whether the environment actually carries a value for `variable`.

    Whitespace is not a value. A `.env` written as `POSTGRES_PASSWORD= ` is the same
    oversight as one written without the space, and reading the second as configured while
    refusing the first would make the check depend on an invisible character.
    """
    return bool(os.getenv(variable, "").strip())


def require(variable: str) -> str:
    """The value of a required credential, or a refusal naming it.

    Deliberately has no default parameter. A caller that wants a fallback wants `os.getenv`
    and is not asking for a secret — every reader of this function is one for which running
    on a guessed value is worse than not running at all.

    The value is returned as configured, not stripped: a password may legitimately begin or
    end with a space, and silently trimming one would produce an authentication failure that
    no amount of comparing the `.env` to the database would ever explain. Only the emptiness
    test above ignores whitespace.
    """
    value = os.getenv(variable)

    if value is None or not value.strip():
        raise MissingConfiguration(
            f"{variable} is not set. It has no default: set it in the environment "
            "or in the repository's .env file before starting this service."
        )

    return value


def missing() -> tuple[str, ...]:
    """Every required credential this environment does not carry, in a stable order.

    The database group is skipped entirely when `DATABASE_URL` is set, matching how
    `app.db.session` actually resolves its connection: reporting `POSTGRES_PASSWORD` as
    missing to a deployment that passed a complete URL would be a false alarm about a
    variable nothing is going to read.
    """
    absent: list[str] = []

    if not is_set(DATABASE_URL_VARIABLE):
        absent.extend(name for name in DATABASE_CREDENTIALS if not is_set(name))

    absent.extend(name for name in OBJECT_STORAGE_CREDENTIALS if not is_set(name))

    return tuple(absent)


def validate() -> None:
    """Refuse to start unless every required credential is configured.

    Called from `app/__init__.py`, so it runs once, before any module in this package — the
    API, the worker, `create_admin.py` and `alembic` all go through it, and none of them had
    to be taught to.

    Raising `MissingConfiguration` from a package import is what stops uvicorn: the ASGI
    application is never constructed, the process exits non-zero, and the restart policy in
    `docker-compose.yml` puts the container into a visible restart loop with this message in
    its log. That is the intended shape of the failure — a stack that is obviously broken is
    safer than one that is quietly running on `deepguard`/`deepguard123`.

    The message lists everything that is absent rather than the first one found. An operator
    filling in a `.env` should learn the whole set in one restart, not one variable per
    attempt.
    """
    absent = missing()

    if not absent:
        return

    raise MissingConfiguration(
        "This service will not start: required configuration is missing "
        f"({', '.join(absent)}). Set the variables listed above in the environment or in "
        "the repository's .env file — see .env.example. None of them has a default."
    )
