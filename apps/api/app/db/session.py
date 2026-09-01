import os
from collections.abc import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.config import require

load_dotenv()


def database_url() -> URL:
    """Build the PostgreSQL URL from the environment provisioned in docker-compose.

    The credentials are required and have no defaults (R1-T5): a URL assembled from a
    fallback `deepguard`/`deepguard` would connect successfully to the wrong server on any
    host that happened to have been created with the same development values, which is worse
    than not connecting at all. `app.config.require` is what refuses, and `app/__init__.py`
    has already refused for the whole process before this function is ever called.

    Host and port keep their defaults. They are not secrets, and a wrong one fails on the
    first connection with the address it tried in the message.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return make_url(url)

    return URL.create(
        drivername="postgresql+psycopg2",
        username=require("POSTGRES_USER"),
        password=require("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=require("POSTGRES_DB"),
    )


engine = create_engine(database_url(), pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
