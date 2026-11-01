"""Health checking and system diagnostics service."""

import socket
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from app.core.config import settings
from app.core.logging import logger
from app.db.session import engine
from app.schemas.health import DatabaseHealth, HealthCheckResponse


def check_database_health() -> DatabaseHealth:
    """Safely probes PostgreSQL connectivity and verifies pgvector extension status."""
    # Fast TCP pre-check to prevent long connection timeouts when database host is down
    try:
        host = settings.POSTGRES_SERVER
        # Normalize localhost to 127.0.0.1 for faster socket resolution on Windows
        target_host = "127.0.0.1" if host in ("localhost", "127.0.0.1") else host
        with socket.create_connection((target_host, settings.POSTGRES_PORT), timeout=0.8):
            pass
    except (socket.timeout, TimeoutError, OSError, ConnectionRefusedError) as exc:
        logger.debug("Database port check failed on %s:%d - %s", settings.POSTGRES_SERVER, settings.POSTGRES_PORT, type(exc).__name__)
        return DatabaseHealth(
            connected=False,
            vector_extension=False,
            error="PostgreSQL service unavailable or connection refused",
        )

    # Database port is open; verify SQLAlchemy query execution and pgvector extension
    try:
        with engine.connect() as connection:
            # 1. Test basic connectivity
            connection.execute(text("SELECT 1;"))

            # 2. Check for pgvector extension
            ext_check = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
            )
            has_vector = ext_check.scalar() is not None

            return DatabaseHealth(
                connected=True,
                vector_extension=has_vector,
                error=None if has_vector else "pgvector extension is not enabled in database",
            )
    except SQLAlchemyError as exc:
        logger.warning("Database connectivity query failed: %s", type(exc).__name__)
        return DatabaseHealth(
            connected=False,
            vector_extension=False,
            error="PostgreSQL service unavailable or connection refused",
        )
    except Exception as exc:
        logger.error("Unexpected error during database health check: %s", type(exc).__name__)
        return DatabaseHealth(
            connected=False,
            vector_extension=False,
            error="Unexpected error probing database connection",
        )


def get_system_health() -> HealthCheckResponse:
    """Aggregates overall application health and dependencies."""
    db_health = check_database_health()
    overall_status = "healthy" if db_health.connected and db_health.vector_extension else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
        database=db_health,
    )
