"""Tests for the request id and the log shape it travels in (R1-T4).

Three separable claims, and they are tested apart because they fail apart:

1. the middleware gives every request an id, accepts a usable one from the caller and
   refuses anything else;
2. a log record emitted under that id carries it, in JSON in production and in readable
   text in development;
3. the id survives the queue — the API writes it onto the job, and the worker's claim reads
   it back.

The third needs real PostgreSQL, as every persistence claim in this suite does, and lives in
`tests/test_persistence.py` and `tests/test_worker.py` beside the rows it is about. What is
here is the first two, which need nothing but the middleware and a logger.
"""

import asyncio
import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.observability import (
    MAX_REQUEST_ID_LENGTH,
    REQUEST_ID_HEADER,
    DevelopmentFormatter,
    JsonFormatter,
    RequestId,
    RequestIdFilter,
    accepted_request_id,
    bind_request_id,
    configure_logging,
    current_request_id,
    development_logging,
    new_request_id,
    reset_request_id,
)


class FakeSession:
    """Enough of a session for `/health`, which is the route these tests drive."""

    def execute(self, statement):
        return None


@pytest.fixture
def client():
    app.dependency_overrides[get_session] = lambda: FakeSession()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def scope(headers=()):
    return {"type": "http", "method": "GET", "path": "/health", "headers": list(headers)}


async def empty_body():
    return {"type": "http.request", "body": b"", "more_body": False}


def run(request_scope):
    """Drive one request through the middleware and return what the id was inside it."""
    seen = {}
    sent = []

    async def inner(scope, receive, send):
        # Read inside the application, which is the only place the answer means anything:
        # what matters is the value a route's log record would pick up, not what the
        # middleware happened to compute.
        seen["request_id"] = current_request_id()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def send(message):
        sent.append(message)

    asyncio.run(RequestId(inner)(request_scope, empty_body, send))

    return seen.get("request_id"), sent


def response_header(sent, name):
    start = next(m for m in sent if m["type"] == "http.response.start")
    return next(
        (value.decode() for key, value in start["headers"] if key == name.encode()),
        None,
    )


# --------------------------------------------------------------------------- #
# What may be accepted as an id                                                #
# --------------------------------------------------------------------------- #


def test_a_uuid_is_accepted_as_a_request_id():
    minted = new_request_id()

    assert accepted_request_id(minted) == minted


@pytest.mark.parametrize(
    "value",
    [
        "trace-01.b_9",
        "A" * MAX_REQUEST_ID_LENGTH,
    ],
)
def test_a_plain_bounded_token_is_accepted(value):
    assert accepted_request_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        # The one that matters most: a newline lets a caller write a second log record of
        # its own choosing into this process's output.
        "abc\ndef",
        "abc\rdef",
        # Longer than the column and longer than anything worth logging.
        "A" * (MAX_REQUEST_ID_LENGTH + 1),
        # Not a token: quotes and braces are what a JSON or logfmt reader downstream would
        # trip over, and spaces make a field boundary ambiguous.
        'id" ,"level":"ERROR',
        "id with spaces",
        "id/with/slashes",
        # Non-ASCII, which the header decode refuses before this is even reached.
        "kimlik-ç",
        "",
        None,
    ],
)
def test_anything_else_is_refused_rather_than_sanitized(value):
    assert accepted_request_id(value) is None


# --------------------------------------------------------------------------- #
# The middleware                                                               #
# --------------------------------------------------------------------------- #


def test_a_request_without_an_id_is_given_one():
    bound, sent = run(scope())

    assert bound is not None
    # A real id, not a placeholder: it has to be unique per request to be worth logging.
    uuid.UUID(bound)
    assert response_header(sent, REQUEST_ID_HEADER) == bound


def test_the_callers_id_is_kept_so_the_trace_spans_both_services():
    caller = "web-9f2c41ab"

    bound, sent = run(scope([(b"x-request-id", caller.encode())]))

    assert bound == caller
    assert response_header(sent, REQUEST_ID_HEADER) == caller


def test_the_header_is_matched_case_insensitively():
    caller = "web-9f2c41ab"

    bound, _ = run(scope([(b"X-Request-ID", caller.encode())]))

    assert bound == caller


def test_an_unusable_id_is_replaced_rather_than_repeated():
    forged = b'abc\nERROR the-database-was-deleted'

    bound, sent = run(scope([(b"x-request-id", forged)]))

    assert bound is not None
    assert "\n" not in bound
    uuid.UUID(bound)
    # And the refused value is not echoed either: a response header is a log line somewhere
    # else.
    assert response_header(sent, REQUEST_ID_HEADER) == bound


def test_an_id_that_is_not_ascii_is_replaced():
    bound, _ = run(scope([(b"x-request-id", "kimlik-\xe7".encode("latin-1"))]))

    uuid.UUID(bound)


def test_two_requests_do_not_share_an_id():
    first, _ = run(scope())
    second, _ = run(scope())

    assert first != second


def test_the_binding_does_not_outlive_the_request():
    run(scope())

    assert current_request_id() is None


def test_a_real_request_through_the_application_is_answered_with_its_id(client):
    caller = "web-integration-1"

    response = client.get("/health", headers={"X-Request-ID": caller})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == caller


def test_a_real_request_without_an_id_is_still_answered_with_one(client):
    response = client.get("/health")

    uuid.UUID(response.headers[REQUEST_ID_HEADER])


def test_non_http_traffic_is_passed_through_untouched():
    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    asyncio.run(RequestId(inner)({"type": "lifespan"}, empty_body, lambda m: None))

    assert calls == ["lifespan"]


# --------------------------------------------------------------------------- #
# What a log record looks like                                                 #
# --------------------------------------------------------------------------- #


def record(message="something happened", **extra):
    made = logging.LogRecord(
        "app.test", logging.INFO, "app/test.py", 10, message, None, None
    )
    made.__dict__.update(extra)
    RequestIdFilter().filter(made)

    return made


def test_a_json_line_carries_the_bound_request_id():
    token = bind_request_id("web-1234")
    try:
        line = json.loads(JsonFormatter().format(record("Claimed job.")))
    finally:
        reset_request_id(token)

    assert line["request_id"] == "web-1234"
    assert line["message"] == "Claimed job."
    assert line["level"] == "INFO"
    assert line["logger"] == "app.test"
    assert line["timestamp"].endswith("+00:00")


def test_a_json_line_outside_a_request_simply_has_no_request_id():
    line = json.loads(JsonFormatter().format(record("Worker started.")))

    # Absent rather than null or a placeholder: this line belongs to no request, and a field
    # that was always present would make "not correlated" indistinguishable from "correlated
    # to something called none".
    assert "request_id" not in line


def test_extra_fields_reach_the_json_line():
    line = json.loads(JsonFormatter().format(record("Queued.", analysis_id="abc")))

    assert line["analysis_id"] == "abc"


def test_a_value_that_cannot_be_serialized_does_not_take_the_line_down():
    line = json.loads(JsonFormatter().format(record("Queued.", session=object())))

    assert isinstance(line["session"], str)
    assert line["message"] == "Queued."


def test_a_traceback_travels_with_the_line():
    try:
        raise ValueError("no")
    except ValueError:
        made = record("Job failed.")
        made.exc_info = __import__("sys").exc_info()

    line = json.loads(JsonFormatter().format(made))

    assert "ValueError: no" in line["exception"]


def test_the_development_line_is_readable_and_still_carries_the_id():
    formatter = DevelopmentFormatter("%(levelname)s %(name)s %(message)s")

    token = bind_request_id("web-1234")
    try:
        line = formatter.format(record("Claimed job."))
    finally:
        reset_request_id(token)

    assert line == "INFO app.test Claimed job. request_id=web-1234"


def test_the_development_id_stays_on_the_message_and_not_after_the_traceback():
    formatter = DevelopmentFormatter("%(levelname)s %(message)s")

    token = bind_request_id("web-1234")
    try:
        # Built under the binding, as a real record is: the filter reads the bound id at the
        # moment the record is handled.
        try:
            raise ValueError("no")
        except ValueError:
            made = record("Job failed.")
            made.exc_info = __import__("sys").exc_info()

        lines = formatter.format(made).splitlines()
    finally:
        reset_request_id(token)

    # On the event, not trailing the innermost frame of the traceback: the first line is
    # what an eye scanning a terminal for a request id reads.
    assert lines[0] == "INFO Job failed. request_id=web-1234"
    assert "ValueError: no" in lines[-1]
    assert "request_id" not in lines[-1]


def test_the_development_line_says_nothing_when_there_is_no_request():
    formatter = DevelopmentFormatter("%(levelname)s %(name)s %(message)s")

    assert formatter.format(record("Worker started.")) == "INFO app.test Worker started."


# --------------------------------------------------------------------------- #
# Which shape a deployment gets                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("environment", ["development", "test", "DEVELOPMENT"])
def test_development_deployments_get_readable_logs(monkeypatch, environment):
    monkeypatch.setenv("DEEPGUARD_ENV", environment)

    assert development_logging() is True


@pytest.mark.parametrize("environment", ["production", "staging", "", "typo"])
def test_everything_else_gets_json(monkeypatch, environment):
    # Including the unset and the misspelt case: an environment nobody configured emits the
    # machine-readable shape rather than silently dropping structure a pipeline expects.
    monkeypatch.setenv("DEEPGUARD_ENV", environment)

    assert development_logging() is False


def test_an_unset_environment_gets_json(monkeypatch):
    monkeypatch.delenv("DEEPGUARD_ENV", raising=False)

    assert development_logging() is False


def test_configuring_twice_does_not_double_every_line(monkeypatch):
    # `app.main` has already configured logging by the time this module is imported, so the
    # claim is not "one handler exists" but "there is never a second copy of ours" — two
    # would mean every log line printed twice, which is how this fails in a deployment.
    monkeypatch.delenv("DEEPGUARD_ENV", raising=False)

    configure_logging()
    configure_logging()

    ours = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_deepguard_handler", False)
    ]

    assert len(ours) == 1
