from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.analyses import router as analyses_router
from app.api.public_v1.analyses import router as public_analyses_router
from app.api.url_analyses import router as url_analyses_router
from app.db.session import get_session
from app.request_limits import UploadRequestSizeLimit

app = FastAPI(title="DeepGuard API")

# In front of routing, so an oversized body is bounded before FastAPI parses the upload.
app.add_middleware(UploadRequestSizeLimit)

app.include_router(analyses_router)
# The internal URL submission, under the same `/api/v1` prefix as the upload beside it. Its
# own module because `app.downloader` imports the upload ceiling from `app.api.analyses`,
# so the route that drives the downloader cannot live there without closing an import cycle.
app.include_router(url_analyses_router)
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
