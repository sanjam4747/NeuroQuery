"""Logging configuration for NeuroQuery."""

import logging
import sys
from app.core.config import settings


def setup_logging() -> None:
    """Configure centralized application logger."""
    log_level = logging.DEBUG if settings.APP_DEBUG else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Silence overly verbose external loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


logger = logging.getLogger("neuroquery")
