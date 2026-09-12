from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.admin_jobs import router as admin_jobs_router
from app.api.admin_users import router as admin_users_router
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
# Account administration, on the same internal prefix again and behind `require_admin` rather
# than the `require_user` the routes above carry. It is mounted separately from the auth
# router deliberately: `/api/v1/auth` is what every signed-in browser talks to about itself,
# and `/api/v1/admin` is what one role talks to about everybody else — one module per audience
# keeps the privileged routes from sitting one forgotten dependency away from the unprivileged
# ones.
app.include_router(admin_users_router)
# The operational half of the same administrative prefix (R8-T3): the detection queue rather
# than the account table. Its own module beside `admin_users` for the reason that one is its
# own module — one audience per file — and read-only by design: there is no retry route here,
# because a requeued job would write a second forensic signal per provider over the first
# one's. The reason is stated in full at the top of `admin_jobs.py`.
app.include_router(admin_jobs_router)
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
