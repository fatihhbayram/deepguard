from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.analyses import router as analyses_router
from app.db.session import get_session

app = FastAPI(title="DeepGuard API")

app.include_router(analyses_router)


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
