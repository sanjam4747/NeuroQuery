"""Tests for Reciprocal Rank Fusion (RRF) algorithm and deduplication."""

import math
import uuid
import pytest
from app.services.bm25_retrieval import BM25Candidate
from app.services.dense_retrieval import DenseCandidate
from app.services.rrf import rrf_service, RRFService


def test_rrf_mathematical_calculation():
    """Verify standard formula RRF(d) = sum(1 / (k + rank)) with k=60."""
    chunk_shared = uuid.uuid4()
    doc_id = uuid.uuid4()

    # Shared chunk at rank 1 in dense (score=0.90) and rank 2 in BM25 (score=15.0)
    dense_list = [
        DenseCandidate(
            chunk_id=chunk_shared,
            document_id=doc_id,
            filename="doc.pdf",
            text="Shared candidate passage.",
            character_count=25,
            page_start=1,
            page_end=1,
            score=0.90,
            rank=1,
        )
    ]

    bm25_list = [
        BM25Candidate(
            chunk_id=chunk_shared,
            document_id=doc_id,
            filename="doc.pdf",
            text="Shared candidate passage.",
            character_count=25,
            page_start=1,
            page_end=1,
            score=15.0,
            rank=2,
        )
    ]

    fused = rrf_service.fuse(dense_list, bm25_list, rrf_k=60, top_k=5)
    assert len(fused) == 1

    expected_rrf = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    assert math.isclose(fused[0].rrf_score, expected_rrf, rel_tol=1e-5)
    assert fused[0].dense_score == 0.90
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_score == 15.0
    assert fused[0].bm25_rank == 2
    assert fused[0].rrf_rank == 1


def test_rrf_candidate_deduplication_and_disjoint_candidates():
    """Verify chunks appearing in only dense or only BM25 are correctly merged and deduplicated."""
    chunk_dense_only = uuid.uuid4()
    chunk_shared = uuid.uuid4()
    chunk_bm25_only = uuid.uuid4()
    doc_id = uuid.uuid4()

    dense_candidates = [
        DenseCandidate(
            chunk_id=chunk_shared,
            document_id=doc_id,
            filename="doc.pdf",
            text="Shared chunk",
            character_count=12,
            page_start=1,
            page_end=1,
            score=0.95,
            rank=1,
        ),
        DenseCandidate(
            chunk_id=chunk_dense_only,
            document_id=doc_id,
            filename="doc.pdf",
            text="Dense only chunk",
            character_count=16,
            page_start=1,
            page_end=1,
            score=0.85,
            rank=2,
        ),
    ]

    bm25_candidates = [
        BM25Candidate(
            chunk_id=chunk_bm25_only,
            document_id=doc_id,
            filename="doc.pdf",
            text="BM25 only chunk",
            character_count=15,
            page_start=1,
            page_end=1,
            score=18.5,
            rank=1,
        ),
        BM25Candidate(
            chunk_id=chunk_shared,
            document_id=doc_id,
            filename="doc.pdf",
            text="Shared chunk",
            character_count=12,
            page_start=1,
            page_end=1,
            score=14.0,
            rank=2,
        ),
    ]

    fused = rrf_service.fuse(dense_candidates, bm25_candidates, rrf_k=60, top_k=10)
    assert len(fused) == 3

    # Shared chunk has contributions from both systems -> highest RRF score
    assert fused[0].chunk_id == chunk_shared
    assert fused[0].rrf_rank == 1
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank == 2

    # Verify single-system chunks
    dense_cand = next(c for c in fused if c.chunk_id == chunk_dense_only)
    assert dense_cand.dense_score == 0.85
    assert dense_cand.dense_rank == 2
    assert dense_cand.bm25_score is None
    assert dense_cand.bm25_rank is None
    assert math.isclose(dense_cand.rrf_score, 1.0 / (60 + 2), rel_tol=1e-5)

    bm25_cand = next(c for c in fused if c.chunk_id == chunk_bm25_only)
    assert bm25_cand.bm25_score == 18.5
    assert bm25_cand.bm25_rank == 1
    assert bm25_cand.dense_score is None
    assert bm25_cand.dense_rank is None
    assert math.isclose(bm25_cand.rrf_score, 1.0 / (60 + 1), rel_tol=1e-5)


def test_rrf_deterministic_tie_breaking():
    """Verify deterministic ranking order when candidates have identical RRF scores."""
    chunk_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    chunk_b = uuid.UUID("00000000-0000-0000-0000-000000000002")
    doc_id = uuid.uuid4()

    # Both chunks appear only in dense retrieval at rank 1
    dense_candidates = [
        DenseCandidate(
            chunk_id=chunk_a,
            document_id=doc_id,
            filename="a.pdf",
            text="A",
            character_count=1,
            page_start=1,
            page_end=1,
            score=0.90,
            rank=1,
        ),
        DenseCandidate(
            chunk_id=chunk_b,
            document_id=doc_id,
            filename="b.pdf",
            text="B",
            character_count=1,
            page_start=1,
            page_end=1,
            score=0.85,  # Lower dense score
            rank=1,
        ),
    ]

    fused = rrf_service.fuse(dense_candidates, [], rrf_k=60, top_k=2)
    assert len(fused) == 2
    # Chunk A has higher dense score as tie-breaker
    assert fused[0].chunk_id == chunk_a
    assert fused[1].chunk_id == chunk_b
