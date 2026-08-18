import os
from collections.abc import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


def database_url() -> URL:
    """Build the PostgreSQL URL from the environment provisioned in docker-compose."""
    url = os.getenv("DATABASE_URL")
    if url:
        return make_url(url)

    return URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "deepguard"),
        password=os.getenv("POSTGRES_PASSWORD", "deepguard"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "deepguard"),
    )


engine = create_engine(database_url(), pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
