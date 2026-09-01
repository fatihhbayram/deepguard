"""One request, one id, in every log line it causes (R1-T4).

DeepGuard answers a submission in one process and does the work in another, minutes later.
Until now those two halves said nothing to each other in the logs: the API wrote "analysis
accepted", the worker wrote "claimed job 3f2b…", and connecting the two meant looking a
`job_id` up in PostgreSQL by hand. When the question is "what happened to the upload this
person made at 14:02", that is one query too many, and it is the question production
operation is made of.

So a correlation id travels the whole way. The web application mints one per browser
request and sends it on; this module accepts it at the API boundary, binds it to a
`ContextVar` that every log record on that request reads, and `analysis_jobs.request_id`
carries it across the queue to the worker, which binds the same value for the length of the
job. Grepping one id therefore returns the browser request, the API request and the
worker's whole analysis.

A `ContextVar` rather than a parameter threaded through every function, and that is the
point of using one: the request id is ambient context, not an argument any of this code has
a use for. Threading it would mean adding a parameter to the storage client, the probe, the
risk engine and everything else that might one day log — which is the same reason
`logging` itself is not passed around.

**An id from a caller is never trusted as it stands.** It reaches log lines, so a value
with a newline in it could forge a log record, and one of unbounded length could fill a
disk. `accepted_request_id` therefore admits a short, boring token or nothing at all, and
anything it refuses is replaced by a fresh id rather than being sanitized into a
half-truth: a mangled id would still correlate two systems, which is exactly what makes a
forged one dangerous.

Structured output is the second half of the task. In production every line is a JSON object
on stdout for whatever the host environment ingests; in development it is the readable text
a person reads in a terminal. No SaaS agent, no APM SDK and no new dependency: the standard
library emits both shapes, and a log aggregator's job is the host's.
"""

import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The header the id travels in, in the spelling everything else uses. Lowercase because
# that is how ASGI presents headers and how they are compared here; HTTP header names are
# case-insensitive, so a client sending `X-Request-ID` is matched by the same check.
REQUEST_ID_HEADER = "x-request-id"

# What a request id may look like when it arrives from outside. Deliberately narrow: this
# value ends up inside log lines and inside a database column, so the characters that could
# do something there — a newline that forges a second log record, a quote that breaks a
# field, anything non-ASCII that a downstream parser reads differently — are simply not
# admitted. 64 is `analysis_jobs.request_id`'s width and comfortably fits a UUID.
MAX_REQUEST_ID_LENGTH = 64
REQUEST_ID_PATTERN = re.compile(rf"\A[A-Za-z0-9._-]{{1,{MAX_REQUEST_ID_LENGTH}}}\Z")

# Which deployment this is. The same variable `app.web_auth` reads to decide whether the
# session cookie is `Secure`, named again here rather than imported from it: this module is
# imported by the worker, which has no session, no cookie and no reason to pull in the
# argon2 machinery that module carries. The two spellings must agree, which is why both say
# so.
ENVIRONMENT_VARIABLE = "DEEPGUARD_ENV"

# The deployments that get readable logs. A closed list, and the same fail-secure direction
# `app.web_auth` takes with the cookie: an unset or unrecognised environment is treated as
# production, so a host nobody configured emits the machine-readable shape rather than
# quietly losing structure a log pipeline was expecting.
DEVELOPMENT_ENVIRONMENTS = frozenset({"development", "test"})

LOG_LEVEL_VARIABLE = "LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

# Marks the handler this module installed, so `configure_logging` can be called twice — the
# API imports it and a test may call it again — without stacking a second copy of every log
# line. Only handlers carrying this attribute are removed; pytest's capture handler and
# anything else on the root logger are left exactly where they are.
_HANDLER_MARKER = "_deepguard_handler"

# The record attributes `logging` puts on every record itself. Everything a caller passed
# through `extra=` is whatever is left over, and that is how the JSON formatter finds the
# structured fields without being told about them.
_STANDARD_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"request_id", "message", "asctime", "taskName"}

# The current request's id, or nothing outside a request. `ContextVar` rather than a global,
# so concurrent requests in one process do not overwrite each other's value: an `asyncio`
# task inherits a copy of the context it was created in, which is exactly one request's
# worth.
_request_id: ContextVar[str | None] = ContextVar("deepguard_request_id", default=None)


def new_request_id() -> str:
    """Mint an id for a request that arrived without a usable one."""
    return str(uuid.uuid4())


def accepted_request_id(value: str | None) -> str | None:
    """The caller's request id if it is one this system may repeat, otherwise nothing.

    Membership of a small character set and a length bound, and nothing more clever than
    that. It is not sanitization — a value is admitted whole or refused whole — because a
    trimmed or escaped id would still be accepted as *this request's* id, and an id is a
    claim about which other log lines belong with these. A caller that sends nonsense gets
    a fresh id, which is honest: nothing correlates to it because nothing else saw it.
    """
    if value is None:
        return None

    return value if REQUEST_ID_PATTERN.fullmatch(value) else None


def current_request_id() -> str | None:
    """The id of the request or job this code is running under, or nothing outside one."""
    return _request_id.get()


def bind_request_id(request_id: str | None) -> Token:
    """Make `request_id` the one every log record from here on reports.

    Returns the token that undoes it. The caller resets rather than clearing, because
    "before this" is not always "nothing": nested binding is not used today, and a `set(None)`
    on the way out would quietly make it wrong the day it is.
    """
    return _request_id.set(request_id)


def reset_request_id(token: Token) -> None:
    """Undo one `bind_request_id`, restoring whatever was bound before it."""
    _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Put the current request id on every record, from wherever it was emitted.

    A filter rather than a `LoggerAdapter` or an argument at each call site: this has to
    reach records that code which has never heard of this module emits — SQLAlchemy,
    uvicorn, a library's warning — and a filter on the handler is the one place all of them
    pass through.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """One log record as one JSON object on one line.

    The production shape. Every line is independently parseable, which is what lets a host's
    log pipeline index `request_id` as a field rather than by pattern-matching a sentence.

    Whatever a caller passed in `extra=` is carried through as top-level keys, so a call
    site that wants to record `job_id` needs nothing added here. Values that will not
    serialize are rendered with `repr` rather than taking the log line down: a formatter
    that raises turns a diagnostic into a second failure.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and key not in payload:
                payload[key] = value

        if record.exc_info:
            # The traceback as one string rather than as a list of frames: it is read by a
            # person, and a pipeline that shows it will show it as it was written.
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=repr)


class DevelopmentFormatter(logging.Formatter):
    """The readable shape, with the request id appended when there is one.

    Appended rather than given a fixed column, because most lines a developer reads — the
    worker starting up, a schema check — belong to no request, and `[None]` in front of
    every one of them is noise that trains the eye to skip the field.

    Appended to the *message*, though, not to the whole record. A log line carrying a
    traceback is several lines long, and putting the id after the last of them would attach
    it to whatever the innermost frame happened to print rather than to the event it belongs
    to — which is exactly the line an eye scanning for a request id lands on.
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        request_id = getattr(record, "request_id", None)

        if not request_id:
            return formatted

        message, newline, rest = formatted.partition("\n")

        return f"{message} request_id={request_id}{newline}{rest}"


def development_logging() -> bool:
    """Whether this deployment gets readable logs instead of JSON."""
    return os.getenv(ENVIRONMENT_VARIABLE, "").strip().lower() in DEVELOPMENT_ENVIRONMENTS


def configure_logging() -> None:
    """Install the one handler both processes log through.

    Called at import by `app.main` and at startup by `app.worker`, so the API and the worker
    emit the same shape without either having to know the other exists.

    It also takes uvicorn's loggers over. Uvicorn configures its own handlers before it
    imports the application, so by the time this runs there are two logging setups in the
    process and half the production output — every access line and every startup message —
    would still be plain text. Clearing those handlers and letting the records propagate
    puts them through the formatter here instead, which is what "JSON logs in production"
    has to mean if it is to be true of the whole stream.

    Idempotent by marker, not by clearing: only a handler this function installed is
    removed, so a second call replaces its own work and leaves pytest's capture handler —
    and anything else on the root logger — untouched.
    """
    root = logging.getLogger()

    for existing in [h for h in root.handlers if getattr(h, _HANDLER_MARKER, False)]:
        root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARKER, True)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(
        DevelopmentFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        if development_logging()
        else JsonFormatter()
    )

    root.addHandler(handler)
    root.setLevel(os.getenv(LOG_LEVEL_VARIABLE, DEFAULT_LOG_LEVEL).upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True


def _header(scope: Scope, name: str) -> str | None:
    """One request header from an ASGI scope, decoded, or nothing if it is absent.

    Bytes that are not ASCII are treated as an absent header rather than decoded loosely.
    The only header read here has a character set this narrow anyway, and a lenient decode
    would be the first step of admitting a value the pattern exists to refuse.
    """
    wanted = name.encode()
    for key, value in scope.get("headers", ()):
        if key.lower() == wanted:
            try:
                return value.decode("ascii")
            except UnicodeDecodeError:
                return None

    return None


class RequestId:
    """Give every HTTP request an id, bind it to the logs, and echo it back.

    ASGI middleware rather than `BaseHTTPMiddleware`, matching `app.request_limits` and for
    the same reason: it has to be outside everything, including the body guard, so a request
    refused for its size is still a request with an id in the log.

    The id is accepted from the caller when the caller sent a usable one, which is what makes
    a trace span two services — the web application mints it, and this request is the same
    request as far as anyone reading the logs is concerned. Otherwise one is minted here, so
    there is no such thing as an unidentified request.

    It is echoed in the response so a caller can record which id its call was handled under
    without having to guess whether the one it sent was accepted.

    The binding is reset on the way out. `asyncio` gives each request its own context, so a
    leak is unlikely rather than possible — but "unlikely" is not a property to rest a
    correlation id on, and a value that outlived its request would attribute one caller's
    log lines to another.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = (
            accepted_request_id(_header(scope, REQUEST_ID_HEADER)) or new_request_id()
        )

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )
            await send(message)

        token = bind_request_id(request_id)
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(token)
