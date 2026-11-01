"""API endpoints for semantic embeddings model information and configuration."""

from fastapi import APIRouter, status
from app.schemas.embeddings import EmbeddingModelInfo
from app.services.embeddings import embedding_service

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.get(
    "/info",
    response_model=EmbeddingModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get Active Embedding Model Info",
    description="Reports the configured Sentence-Transformers model name, vector dimensionality, sequence limit, device, and metric.",
)
def get_embedding_model_info() -> EmbeddingModelInfo:
    info = embedding_service.get_info()
    return EmbeddingModelInfo(**info)
