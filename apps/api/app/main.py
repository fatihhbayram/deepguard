from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.analyses import router as analyses_router
from app.api.auth import router as auth_router
from app.api.public_v1.analyses import router as public_analyses_router
from app.api.url_analyses import router as url_analyses_router
from app.db.session import get_session
from app.observability import RequestId, configure_logging
from app.request_limits import UploadRequestSizeLimit

# Before the application object, so that anything the routers log while they are being
# imported already goes through the one configured handler. Uvicorn has set its own logging
# up by the time it imports this module, and this is what takes that over — see
# `configure_logging`.
configure_logging()

app = FastAPI(title="DeepGuard API")

# In front of routing, so an oversized body is bounded before FastAPI parses the upload.
app.add_middleware(UploadRequestSizeLimit)

# Outside that, because `add_middleware` puts the most recently added layer outermost. The
# order is deliberate: a request refused for its size never reaches the guard's inner
# application, and it should still be a request with an id in the log and in the 413 it gets
# back.
app.add_middleware(RequestId)

app.include_router(analyses_router)
# The internal URL submission, under the same `/api/v1` prefix as the upload beside it. Its
# own module because `app.downloader` imports the upload ceiling from `app.api.analyses`,
# so the route that drives the downloader cannot live there without closing an import cycle.
app.include_router(url_analyses_router)
# Web sign-in, under the same internal `/api/v1` prefix. It authenticates browsers by
# cookie and has nothing to do with the API-key surface below. Since R1-T2 the analyses
# routes above demand one of these sessions — per route, through `require_user`, and never
# through a blanket middleware, so `/health` and the public surface stay untouched by it.
app.include_router(auth_router)
# The external B2B surface. Mounted under its own prefix and carrying its own API-key
# dependency, so the internal routes above stay exactly as unauthenticated as they were.
app.include_router(public_analyses_router)


@app.get("/health")
def health(session: Session = Depends(get_session)) -> JSONResponse:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "unavailable"},
        )

    return JSONResponse(content={"status": "ok", "database": "ok"})
