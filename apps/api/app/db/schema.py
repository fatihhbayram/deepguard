"""Whether the database is at the schema this code was written against (R1-T3).

The worker reads and writes tables it never declared at runtime; Alembic is what makes them
exist. Between the two sits a window nothing was watching: a deployment that starts the new
worker image before `alembic upgrade head` has run gives a process built for one schema a
database holding another. The failure that produces is not a clean one — the worker claims
a job, gets a `column does not exist` several statements in, fails that analysis, and takes
the next one. A rollout that forgot a migration therefore looked like a burst of failed
analyses rather than like a rollout that forgot a migration.

So the question is asked once, before anything is claimed, and it is asked the way Alembic
itself asks it: the revisions stamped in `alembic_version` against the head revisions of the
migration directory that shipped in this image. Comparing the two is the whole check.

Deliberately not `Base.metadata` reflection. Comparing declared tables against real ones
would answer a similar question and answer it wrongly in both directions — a data-only
migration changes no table, and a column added by hand would satisfy a reflection check
while leaving the version table saying something else. The stamped revision is the fact the
deployment procedure actually produces, so it is the fact worth checking.

Multiple heads are treated as a schema that is not ready rather than as an error of their
own. A repository with two heads has no single "current" schema to be at, and a worker is
not the place to discover that — it says what it found and refuses, which is what it would
do for any other mismatch.
"""

import logging
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# The API directory, where `alembic.ini` sits beside `app/`. Resolved from this file rather
# than from the working directory: the worker is started as `python -m app.worker` and
# nothing guarantees where from.
API_ROOT = Path(__file__).resolve().parent.parent.parent
ALEMBIC_CONFIG = API_ROOT / "alembic.ini"


class SchemaNotReady(Exception):
    """The database is not at the revision this code expects.

    Carries both revision sets so the log line can say what to do about it. Neither is a
    secret — they are migration identifiers committed to the repository — and neither
    quotes a connection string, a credential or any row.
    """

    def __init__(self, expected: frozenset[str], found: frozenset[str]) -> None:
        self.expected = expected
        self.found = found
        super().__init__(
            "The database schema is not at the expected Alembic revision "
            f"(expected {_render(expected)}, found {_render(found)}). "
            "Run `alembic upgrade head` before starting the worker."
        )


def _render(revisions: frozenset[str]) -> str:
    """Name a revision set for a log line, including the empty one."""
    return ", ".join(sorted(revisions)) if revisions else "none"


def expected_revisions() -> frozenset[str]:
    """The head revision(s) of the migration directory shipped in this image."""
    script = ScriptDirectory.from_config(Config(str(ALEMBIC_CONFIG)))
    return frozenset(script.get_heads())


def applied_revisions(engine: Engine) -> frozenset[str]:
    """The revision(s) stamped in the database, or nothing if it has never been migrated.

    A database with no `alembic_version` table answers the empty set rather than raising:
    "never migrated" is one of the states this check exists to catch, and it is not
    different in kind from "migrated to the wrong revision".
    """
    with engine.connect() as connection:
        return frozenset(MigrationContext.configure(connection).get_current_heads())


def check_schema_ready(engine: Engine) -> None:
    """Raise `SchemaNotReady` unless the database is exactly at this image's head.

    Exact set equality, in both directions. A database *behind* head is the obvious case; a
    database *ahead* of it is the rollback case, and it is refused too — an old worker
    against a new schema is the same class of mismatch, and letting it through would mean a
    failed deployment silently ran half its fleet against tables it does not know about.

    Anything the database itself raises — it is not there, the credentials are wrong —
    propagates as it is. That is not a schema answer and dressing it up as one would tell an
    operator to run a migration against a server they cannot reach.
    """
    expected = expected_revisions()
    found = applied_revisions(engine)

    if expected != found:
        raise SchemaNotReady(expected=expected, found=found)

    logger.info("Database schema is at the expected revision %s.", _render(expected))
