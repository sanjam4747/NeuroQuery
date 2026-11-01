"""API v1 Endpoints package."""

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.root import router as root_router

__all__ = ["health_router", "root_router"]
