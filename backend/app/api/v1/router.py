"""API v1 master router."""

from fastapi import APIRouter
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.root import router as root_router
from app.api.v1.endpoints.documents import router as documents_router

api_router = APIRouter()

# Include versioned sub-routers
api_router.include_router(root_router, tags=["API Info"])
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(documents_router)
