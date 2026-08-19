"""Tests for the pre-parser request-body guard.

Driven at the ASGI layer, because that is where the guard lives and what it protects:
these prove the inner application never runs, or never receives the excess bytes, which
a response code alone cannot show. Nothing here allocates a real 100 MiB payload — the
same 1 MiB block is fed repeatedly, so what is counted is real while memory is not.

`tests/test_upload.py` covers the other half: the route's own file limit, still returning
413 for a file of `MAX_UPLOAD_BYTES + 1`, and normal uploads still reaching the pipeline.
"""

import asyncio
import json

import pytest

from app.api.analyses import MAX_UPLOAD_BYTES
from app.request_limits import (
    MAX_REQUEST_BYTES,
    MULTIPART_OVERHEAD_BYTES,
    UploadRequestSizeLimit,
)

BLOCK_SIZE = 1024 * 1024
# One buffer, reused for every chunk: the counter sees the bytes, the test never holds
# them.
BLOCK = b"\0" * BLOCK_SIZE


class SpyApp:
    """Stand-in for everything behind the guard: FastAPI, its parser and the route."""

    def __init__(self, *, status=200, raises=None, drain=True):
        self.status = status
        self.raises = raises
        self.drain = drain
        self.calls = 0
        self.body_bytes = 0
        self.disconnected = False

    async def __call__(self, scope, receive, send):
        self.calls += 1

        if self.drain:
            more = True
            while more:
                message = await receive()
                if message["type"] == "http.disconnect":
                    self.disconnected = True
                    break
                self.body_bytes += len(message.get("body", b""))
                more = message.get("more_body", False)

        if self.raises is not None:
            raise self.raises

        await send(
            {"type": "http.response.start", "status": self.status, "headers": []}
        )
        await send({"type": "http.response.body", "body": b"downstream"})


def upload_scope(*, content_length=None, method="POST", path="/api/v1/analyses"):
    headers = [(b"content-type", b"multipart/form-data; boundary=x")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))

    return {"type": "http", "method": method, "path": path, "headers": headers}


class Stream:
    """A request body delivered in chunks, tracking what was actually pulled from it."""

    def __init__(self, total_bytes):
        self.remaining = total_bytes
        self.delivered = 0

    async def __call__(self):
        if self.remaining <= 0:
            return {"type": "http.request", "body": b"", "more_body": False}

        size = min(BLOCK_SIZE, self.remaining)
        self.remaining -= size
        self.delivered += size

        return {
            "type": "http.request",
            "body": BLOCK[:size],
            "more_body": self.remaining > 0,
        }


def run(app, scope, receive):
    """Drive one request through the guard and collect the response messages."""
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(UploadRequestSizeLimit(app)(scope, receive, send))

    return sent


def status_of(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def body_of(sent):
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


def test_the_request_limit_leaves_room_for_multipart_framing():
    # The two limits are deliberately different: this guard bounds the HTTP body, the
    # route bounds the file. A file of exactly the business limit must survive its own
    # multipart envelope.
    assert MAX_REQUEST_BYTES == MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
    assert MAX_REQUEST_BYTES > MAX_UPLOAD_BYTES


def test_an_oversized_content_length_is_rejected_before_the_app_runs():
    app = SpyApp()

    sent = run(app, upload_scope(content_length=MAX_REQUEST_BYTES + 1), None)

    assert status_of(sent) == 413
    # Nothing behind the guard was reached, so no parser was ever started. `receive` is
    # None here: touching the body at all would have raised.
    assert app.calls == 0


def test_the_rejection_says_only_that_the_request_was_too_large():
    sent = run(SpyApp(), upload_scope(content_length=MAX_REQUEST_BYTES + 1), None)

    assert json.loads(body_of(sent)) == {
        "detail": f"Request body exceeds the {MAX_REQUEST_BYTES} byte limit."
    }


def test_a_declared_length_within_the_limit_reaches_the_app():
    app = SpyApp()
    stream = Stream(BLOCK_SIZE)

    sent = run(app, upload_scope(content_length=BLOCK_SIZE), stream)

    assert status_of(sent) == 200
    assert app.calls == 1
    assert app.body_bytes == BLOCK_SIZE


def test_a_file_at_the_business_limit_survives_its_multipart_framing():
    # Exactly what the route is required to accept, plus the boundary and part headers
    # wrapped around it. Rejecting this would make the guard, not the route, the real
    # file limit.
    framed = MAX_UPLOAD_BYTES + 512
    app = SpyApp()
    stream = Stream(framed)

    sent = run(app, upload_scope(content_length=framed), stream)

    assert status_of(sent) == 200
    assert app.body_bytes == framed


@pytest.mark.parametrize("content_length", [None, "not-a-number"])
def test_a_streamed_body_without_a_usable_length_is_still_bounded(content_length):
    # Chunked transfer, or a header that states nothing usable: the only trustworthy
    # figure is what actually arrives.
    oversized = MAX_REQUEST_BYTES + 4 * BLOCK_SIZE
    app = SpyApp()
    stream = Stream(oversized)

    headers = [(b"content-type", b"multipart/form-data; boundary=x")]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    scope = {**upload_scope(), "headers": headers}

    sent = run(app, scope, stream)

    assert status_of(sent) == 413
    # The body stopped arriving at the limit instead of being read to the end.
    assert stream.remaining > 0
    assert stream.delivered <= MAX_REQUEST_BYTES + BLOCK_SIZE


def test_the_app_never_receives_the_bytes_past_the_limit():
    app = SpyApp()
    stream = Stream(MAX_REQUEST_BYTES + 4 * BLOCK_SIZE)

    run(app, upload_scope(), stream)

    # The chunk that crossed the limit is dropped, not forwarded, and the app is told
    # the stream is gone.
    assert app.body_bytes <= MAX_REQUEST_BYTES
    assert app.disconnected


def test_the_apps_own_response_to_a_cut_body_is_replaced_with_413():
    # A parser that loses its stream answers however it likes — 400, in FastAPI's case.
    # The client asked too much of the server and must be told that.
    app = SpyApp(status=400)
    stream = Stream(MAX_REQUEST_BYTES + BLOCK_SIZE)

    sent = run(app, upload_scope(), stream)

    assert status_of(sent) == 413
    assert b"downstream" not in body_of(sent)
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]


def test_an_app_that_raises_on_a_cut_body_still_answers_413():
    app = SpyApp(raises=RuntimeError("stream vanished"))
    stream = Stream(MAX_REQUEST_BYTES + BLOCK_SIZE)

    sent = run(app, upload_scope(), stream)

    assert status_of(sent) == 413


def test_an_unrelated_failure_is_not_disguised_as_a_size_rejection():
    app = SpyApp(raises=RuntimeError("something else broke"), drain=False)

    with pytest.raises(RuntimeError, match="something else broke"):
        run(app, upload_scope(content_length=BLOCK_SIZE), Stream(BLOCK_SIZE))


def test_other_routes_are_passed_straight_through():
    # Even claiming an absurd size: the guard covers the upload endpoint only.
    for scope in (
        upload_scope(content_length=MAX_REQUEST_BYTES + 1, method="GET"),
        upload_scope(content_length=MAX_REQUEST_BYTES + 1, path="/health"),
        upload_scope(content_length=MAX_REQUEST_BYTES + 1, path="/api/v1/analyses/other"),
    ):
        app = SpyApp()

        sent = run(app, scope, Stream(0))

        assert app.calls == 1
        assert status_of(sent) == 200


def test_non_http_scopes_are_passed_straight_through():
    app = SpyApp(drain=False)

    sent = run(app, {"type": "lifespan"}, None)

    assert app.calls == 1
    assert status_of(sent) == 200
