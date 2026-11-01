"""Unified hybrid retrieval orchestrator coordinating dense, BM25, RRF, and reranking."""

import time
from typing import List, Optional
import uuid
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.schemas.retrieval import (
    RetrievalMetadata,
    RetrievalQueryRequest,
    RetrievalQueryResponse,
    RetrievedChunkResult,
)
from app.services.bm25_retrieval import bm25_service
from app.services.dense_retrieval import dense_retrieval_service
from app.services.reranker import reranker_service
from app.services.rrf import rrf_service


class RetrievalService:
    """
    Orchestrates the multi-stage hybrid retrieval pipeline:
    1. Dense Semantic Vector Search (pgvector cosine distance)
    2. Sparse Lexical Search (BM25Okapi)
    3. Reciprocal Rank Fusion (RRF k=60)
    4. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    """

    def search(
        self,
        db: Session,
        request: RetrievalQueryRequest,
    ) -> RetrievalQueryResponse:
        """
        Executes the end-to-end hybrid retrieval pipeline with comprehensive latency tracking.

        Args:
            db: Database session.
            request: Validated search query request.

        Returns:
            RetrievalQueryResponse containing ranked results and latency metrics.
        """
        total_start = time.perf_counter()
        query = request.query

        # 1. Dense Semantic Retrieval
        dense_start = time.perf_counter()
        try:
            dense_candidates = dense_retrieval_service.search(
                db=db,
                query=query,
                top_k=settings.DENSE_RETRIEVAL_TOP_K,
                document_ids=request.document_ids,
                similarity_threshold=request.similarity_threshold,
            )
        except Exception as exc:
            logger.error("Dense retrieval failed for query '%s': %s", query[:30], exc, exc_info=True)
            dense_candidates = []
        dense_latency_ms = (time.perf_counter() - dense_start) * 1000

        # 2. Sparse BM25 Retrieval
        bm25_start = time.perf_counter()
        try:
            bm25_candidates = bm25_service.search(
                db=db,
                query=query,
                top_k=settings.BM25_RETRIEVAL_TOP_K,
                document_ids=request.document_ids,
            )
        except Exception as exc:
            logger.error("BM25 retrieval failed for query '%s': %s", query[:30], exc, exc_info=True)
            bm25_candidates = []
        bm25_latency_ms = (time.perf_counter() - bm25_start) * 1000

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_start = time.perf_counter()
        fused_candidates = rrf_service.fuse(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            rrf_k=settings.RRF_CONSTANT,
            top_k=settings.HYBRID_RETRIEVAL_TOP_K,
        )
        rrf_latency_ms = (time.perf_counter() - rrf_start) * 1000

        # 4. Cross-Encoder Reranking
        reranker_start = time.perf_counter()
        reranked_candidates = reranker_service.rerank(
            query=query,
            candidates=fused_candidates,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking,
        )
        reranker_latency_ms = (time.perf_counter() - reranker_start) * 1000

        total_latency_ms = (time.perf_counter() - total_start) * 1000

        # 5. Format results
        results: List[RetrievedChunkResult] = []
        for cand in reranked_candidates:
            results.append(
                RetrievedChunkResult(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    filename=cand.filename,
                    text=cand.text,
                    character_count=cand.character_count,
                    page_start=cand.page_start,
                    page_end=cand.page_end,
                    metadata_json=cand.metadata_json,
                    dense_score=cand.dense_score,
                    dense_rank=cand.dense_rank,
                    bm25_score=cand.bm25_score,
                    bm25_rank=cand.bm25_rank,
                    rrf_score=cand.rrf_score,
                    rrf_rank=cand.rrf_rank,
                    reranker_score=cand.reranker_score,
                    final_rank=cand.final_rank,
                )
            )

        metadata = RetrievalMetadata(
            dense_candidates=len(dense_candidates),
            bm25_candidates=len(bm25_candidates),
            fused_candidates=len(fused_candidates),
            reranked_candidates=len(fused_candidates) if request.enable_reranking else 0,
            returned_results=len(results),
            reranking_enabled=request.enable_reranking and settings.ENABLE_RERANKING,
            dense_latency_ms=round(dense_latency_ms, 2),
            bm25_latency_ms=round(bm25_latency_ms, 2),
            rrf_latency_ms=round(rrf_latency_ms, 2),
            reranker_latency_ms=round(reranker_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
        )

        logger.info(
            "Hybrid retrieval completed in %.1f ms (%d results from %d dense / %d BM25 / %d fused)",
            total_latency_ms,
            len(results),
            len(dense_candidates),
            len(bm25_candidates),
            len(fused_candidates),
        )

        return RetrievalQueryResponse(
            query=query,
            results=results,
            retrieval_metadata=metadata,
        )


# Singleton instance
retrieval_service = RetrievalService()
