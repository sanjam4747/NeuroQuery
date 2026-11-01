"""API routes for Phase 4 hybrid retrieval and relevance scoring."""

import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.config import settings
from app.core.logging import logger
from app.schemas.retrieval import (
    RebuildIndexResponse,
    RetrievalInfoResponse,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
)
from app.services.bm25_retrieval import bm25_service
from app.services.dense_retrieval import dense_retrieval_service
from app.services.embeddings import embedding_service
from app.services.reranker import reranker_service
from app.services.retrieval import retrieval_service

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-stage hybrid retrieval",
    description="Performs dense vector retrieval, sparse BM25 retrieval, RRF fusion, and cross-encoder reranking.",
)
def search_documents(
    request: RetrievalQueryRequest,
    db: Session = Depends(get_db),
) -> Any:
    """
    Search document passages using the hybrid retrieval pipeline.

    - Dense Semantic Search (BAAI/bge-small-en-v1.5)
    - Sparse Lexical Search (BM25Okapi)
    - Reciprocal Rank Fusion (RRF k=60)
    - Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    """
    try:
        response = retrieval_service.search(db, request)
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Retrieval search endpoint error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval pipeline failed: {exc}",
        )


@router.get(
    "/info",
    response_model=RetrievalInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get retrieval engine status and telemetry",
)
def get_retrieval_info(
    db: Session = Depends(get_db),
) -> Any:
    """Retrieves operational telemetry and model configuration for the hybrid retrieval engine."""
    # Ensure BM25 status is populated
    bm25_stat = bm25_service.get_status()

    return RetrievalInfoResponse(
        embedding_model=embedding_service.model_name,
        embedding_dimension=embedding_service.dimension,
        reranker_model=reranker_service.model_name,
        dense_available=True,
        bm25_indexed=bm25_stat["is_indexed"],
        bm25_total_chunks=bm25_stat["total_chunks"],
        bm25_total_documents=bm25_stat["total_documents"],
        bm25_last_updated=bm25_stat["last_updated_at"],
        rrf_constant=settings.RRF_CONSTANT,
        reranking_enabled_by_default=settings.ENABLE_RERANKING,
        device=embedding_service.resolve_device(),
    )


@router.post(
    "/rebuild-index",
    response_model=RebuildIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Rebuild the sparse BM25 index from persistent chunks",
)
def rebuild_bm25_index(
    db: Session = Depends(get_db),
) -> Any:
    """Reconstructs the in-memory BM25 index from all completed document chunks in the database."""
    start_time = time.perf_counter()
    try:
        count = bm25_service.build_index(db)
        duration_ms = (time.perf_counter() - start_time) * 1000
        return RebuildIndexResponse(
            status="completed",
            indexed_chunks=count,
            indexed_documents=bm25_service.total_documents,
            duration_ms=round(duration_ms, 2),
        )
    except Exception as exc:
        logger.error("Failed to rebuild BM25 index: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"BM25 index rebuild failed: {exc}",
        )
