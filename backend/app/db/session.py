"""Database engine, session factory, and dependency lifecycle."""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Create engine with resilient connection parameters.
# Note: create_engine does not immediately establish network connections;
# connections are created lazily on first checkout from the pool.
engine = create_engine(
    settings.sync_database_url,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    connect_args={
        "connect_timeout": int(settings.DB_CONNECT_TIMEOUT_SECONDS)
    },
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a transactional database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
