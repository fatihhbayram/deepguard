"""The one place both processes pass through before anything else (R1-T5).

Python initialises a package before any module inside it, so this file runs once for the
API, once for the worker, and once for every tool that imports `app.*` — `create_admin.py`
and `alembic` included. That makes it the only spot where a configuration check is
guaranteed to happen *before* `app.db.session` builds an engine or `app.storage` builds a
MinIO client, which is what "fail at startup" has to mean if it is to be true of both
processes and not just of the one somebody remembered to add a check to.

Nothing else belongs here. This is a gate, not a module: an import side effect that is worth
its surprise only because the alternative is a stack that starts on guessed credentials.
"""

from app.config import validate

validate()
