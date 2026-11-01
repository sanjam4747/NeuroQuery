"""Tests for cross-encoder reranker service."""

import uuid
import pytest
from app.services.reranker import reranker_service, RerankerService
from app.services.rrf import HybridCandidate


def test_reranker_singleton_and_device():
    """Verify reranker singleton initialization and device resolution."""
    assert reranker_service.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device = reranker_service.resolve_device()
    assert device in ("cpu", "cuda")


def test_reranker_empty_candidates():
    """Verify empty candidate list returns empty without error."""
    assert reranker_service.rerank("test query", []) == []


def test_reranker_bypass_when_disabled():
    """Verify disabling reranker preserves RRF ordering without running model inference."""
    c1 = HybridCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="doc1.pdf",
        text="First passage",
        character_count=13,
        page_start=1,
        page_end=1,
        rrf_score=0.032,
        rrf_rank=1,
    )
    c2 = HybridCandidate(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="doc2.pdf",
        text="Second passage",
        character_count=14,
        page_start=2,
        page_end=2,
        rrf_score=0.016,
        rrf_rank=2,
    )

    results = reranker_service.rerank(
        query="test query",
        candidates=[c1, c2],
        top_k=2,
        enable_reranking=False,
    )

    assert len(results) == 2
    assert results[0].chunk_id == c1.chunk_id
    assert results[0].final_rank == 1
    assert results[0].reranker_score is None
    assert results[1].chunk_id == c2.chunk_id
    assert results[1].final_rank == 2


def test_reranker_scoring_and_ranking_order():
    """Verify cross-encoder scores relevant passage higher and elevates its final rank."""
    query = "What database extension is used for vector search?"
    doc_id = uuid.uuid4()

    # Highly relevant chunk
    chunk_relevant = HybridCandidate(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        filename="db.pdf",
        text="PostgreSQL utilizes the pgvector extension and HNSW indexes for approximate nearest neighbor vector search.",
        character_count=105,
        page_start=1,
        page_end=1,
        dense_score=0.88,
        dense_rank=2,  # Suppose it was lower in initial fusion
        rrf_score=0.015,
        rrf_rank=2,
    )

    # Irrelevant chunk that happened to be rank 1 in fusion
    chunk_irrelevant = HybridCandidate(
        chunk_id=uuid.uuid4(),
        document_id=doc_id,
        filename="recipes.txt",
        text="To prepare homemade pizza dough, combine warm water with dry yeast and olive oil.",
        character_count=82,
        page_start=1,
        page_end=1,
        bm25_score=14.0,
        bm25_rank=1,
        rrf_score=0.016,
        rrf_rank=1,
    )

    results = reranker_service.rerank(
        query=query,
        candidates=[chunk_irrelevant, chunk_relevant],
        top_k=2,
        enable_reranking=True,
    )

    assert len(results) == 2
    # Reranker should promote the relevant chunk to rank 1
    assert results[0].chunk_id == chunk_relevant.chunk_id
    assert results[0].final_rank == 1
    assert results[0].reranker_score is not None
    assert results[1].reranker_score is not None
    assert results[0].reranker_score > results[1].reranker_score
    # Preserves historical RRF and dense metrics
    assert results[0].dense_rank == 2
    assert results[0].rrf_rank == 2
