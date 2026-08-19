"""Pre-parser bound on the upload request body.

The route already refuses a file larger than `MAX_UPLOAD_BYTES` while it streams the
upload in chunks, and that limit stays authoritative for the file itself. It can only
act once FastAPI has already parsed the multipart body into an `UploadFile`, though —
parsing that happens before the route function runs and is therefore beyond the route's
reach. This is the layer in front of it: it counts raw request bytes as the server hands
them over, so an oversized body stops arriving before the parser can finish consuming it.

Deliberately narrow: only `POST /api/v1/analyses`, the one endpoint that accepts a body.
"""

import json
import logging

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.analyses import MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

UPLOAD_PATH = "/api/v1/analyses"
UPLOAD_METHOD = "POST"

# A multipart body is the file plus its framing: the boundary lines around each part,
# the part headers, and the closing boundary. That framing is a few hundred bytes for
# this endpoint's single file part, so 64 KiB is a wide but bounded allowance — enough
# that a legitimate file of exactly MAX_UPLOAD_BYTES is never rejected here for its
# envelope, and small enough that it does not meaningfully raise the ceiling.
MULTIPART_OVERHEAD_BYTES = 64 * 1024

# The request-layer ceiling. It is intentionally above the file limit: this guard bounds
# the HTTP body, while the route decides what the file itself is allowed to be.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES

async def _send_too_large(send: Send) -> None:
    """Answer 413 directly, in the same JSON shape the route's own errors use.

    It says only that the request was too large: parser internals and byte accounting
    stay in the server log.
    """
    body = json.dumps(
        {"detail": f"Request body exceeds the {MAX_REQUEST_BYTES} byte limit."}
    ).encode()

    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # Nothing further is read from this request, so the connection cannot be
                # reused for another one.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _declared_length(scope: Scope) -> int | None:
    """Read the request's Content-Length, if it states a usable one.

    A header that is absent or not a plain byte count proves nothing about the body's
    size, so it is treated as unknown and left to the byte counter.
    """
    for name, value in scope.get("headers", ()):
        if name.lower() == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None

    return None


class _BoundedBody:
    """Counts the body as it arrives and cuts the request off once it is too large.

    Only the length of each chunk is kept, never the chunks themselves: this adds no
    buffering of its own, and the body still reaches the parser one chunk at a time.
    """

    def __init__(self, receive: Receive, send: Send) -> None:
        self._receive = receive
        self._send = send
        self._received = 0
        self.exceeded = False
        self._app_started_response = False
        self._replaced_response = False

    async def receive(self) -> Message:
        message = await self._receive()
        if self.exceeded or message["type"] != "http.request":
            return message

        self._received += len(message.get("body", b""))
        if self._received <= MAX_REQUEST_BYTES:
            return message

        self.exceeded = True
        # The offending chunk is dropped rather than handed on, and the stream is
        # reported as gone so the parser stops here instead of waiting for a body that
        # will never be forwarded.
        return {"type": "http.disconnect"}

    async def send(self, message: Message) -> None:
        """Forward the app's response, or replace it once the body was cut off.

        A parser that loses its stream mid-body reports that however it sees fit, and
        that is not the answer this client is owed. As long as nothing has been sent
        yet, the app's response is dropped and the 413 is sent in its place.
        """
        if self._replaced_response:
            return

        if self.exceeded and not self._app_started_response:
            if message["type"] == "http.response.start":
                await _send_too_large(self._send)
                self._replaced_response = True
            return

        self._app_started_response = True
        await self._send(message)

    @property
    def responded(self) -> bool:
        return self._app_started_response or self._replaced_response


class UploadRequestSizeLimit:
    """Bound the upload endpoint's request body before anything parses it.

    Runs as ASGI middleware, so it sits in front of routing and therefore in front of
    FastAPI's multipart parsing. Every other route is passed straight through.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != UPLOAD_METHOD
            or scope.get("path") != UPLOAD_PATH
        ):
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > MAX_REQUEST_BYTES:
            # The client stated the size itself: answer before reading a single body
            # byte, so no parser is ever started.
            logger.info("Rejected an upload declaring %s bytes.", declared)
            await _send_too_large(send)
            return

        # A declared length can be absent, wrong or chunked away entirely, so what
        # actually arrives is counted regardless.
        body = _BoundedBody(receive, send)
        try:
            await self.app(scope, body.receive, body.send)
        except Exception:
            # An app that raises rather than responds when its stream disappears must
            # still not turn a bounded request into a 500. Anything unrelated to the cut
            # body propagates untouched.
            if not body.exceeded or body.responded:
                raise
            logger.info("Bounded an upload request that exceeded the body limit.")
            await _send_too_large(send)
