"""API v1 Endpoints package."""

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.root import router as root_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.embeddings import router as embeddings_router
from app.api.v1.endpoints.retrieval import router as retrieval_router
from app.api.v1.endpoints.assistant import router as assistant_router

__all__ = ["health_router", "root_router", "documents_router", "embeddings_router", "retrieval_router", "assistant_router"]

