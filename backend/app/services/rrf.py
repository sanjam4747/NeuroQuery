"""Reciprocal Rank Fusion (RRF) service for combining dense and sparse retrieval rankings."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
from app.core.config import settings
from app.core.logging import logger
from app.services.bm25_retrieval import BM25Candidate
from app.services.dense_retrieval import DenseCandidate


@dataclass
class HybridCandidate:
    """Represents a fused candidate chunk produced via Reciprocal Rank Fusion."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    character_count: int
    page_start: Optional[int]
    page_end: Optional[int]
    metadata_json: Dict[str, Any] = field(default_factory=dict)

    # Retrieval Metrics Preservation
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None

    # Fusion Metrics
    rrf_score: float = 0.0
    rrf_rank: int = 1


class RRFService:
    """
    Executes Reciprocal Rank Fusion over dense semantic and sparse BM25 candidate lists.

    Formula:
        RRF(d) = sum_{m in {dense, bm25}} [ 1 / (k + rank_m(d)) ]

    Design details:
    - Combines ordinal ranks rather than raw uncalibrated scores (which have different distributions).
    - Preserves original dense and BM25 scores and ranks for downstream explainability and telemetry.
    - Handles candidates appearing in one or both retrieval systems seamlessly.
    - Deterministic tie-breaking using dense score, BM25 score, and stable chunk UUID string.
    """

    @staticmethod
    def fuse(
        dense_candidates: List[DenseCandidate],
        bm25_candidates: List[BM25Candidate],
        rrf_k: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[HybridCandidate]:
        """
        Merges and deduplicates dense and BM25 candidate lists using RRF.

        Args:
            dense_candidates: Ranked list from dense semantic search.
            bm25_candidates: Ranked list from sparse BM25 search.
            rrf_k: Configurable smoothing constant (defaults to settings.RRF_CONSTANT, standard 60).
            top_k: Maximum number of fused candidates to return (defaults to settings.HYBRID_RETRIEVAL_TOP_K).

        Returns:
            List of HybridCandidate objects sorted descending by RRF score.
        """
        k = rrf_k if rrf_k is not None else settings.RRF_CONSTANT
        eff_top_k = top_k or settings.HYBRID_RETRIEVAL_TOP_K

        # Maps chunk_id -> dict of candidate properties and accumulated RRF score
        fused_map: Dict[uuid.UUID, Dict[str, Any]] = {}

        # 1. Process dense candidates
        for c in dense_candidates:
            cid = c.chunk_id
            rrf_contrib = 1.0 / (k + c.rank)
            fused_map[cid] = {
                "chunk_id": cid,
                "document_id": c.document_id,
                "filename": c.filename,
                "text": c.text,
                "character_count": c.character_count,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "metadata_json": c.metadata_json,
                "dense_score": c.score,
                "dense_rank": c.rank,
                "bm25_score": None,
                "bm25_rank": None,
                "rrf_score": rrf_contrib,
            }

        # 2. Process BM25 candidates
        for c in bm25_candidates:
            cid = c.chunk_id
            rrf_contrib = 1.0 / (k + c.rank)

            if cid in fused_map:
                # Chunk retrieved by both dense and sparse systems
                fused_map[cid]["rrf_score"] += rrf_contrib
                fused_map[cid]["bm25_score"] = c.score
                fused_map[cid]["bm25_rank"] = c.rank
            else:
                # Chunk retrieved solely by BM25
                fused_map[cid] = {
                    "chunk_id": cid,
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "text": c.text,
                    "character_count": c.character_count,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "metadata_json": c.metadata_json,
                    "dense_score": None,
                    "dense_rank": None,
                    "bm25_score": c.score,
                    "bm25_rank": c.rank,
                    "rrf_score": rrf_contrib,
                }

        # 3. Deterministic sorting:
        # Primary: RRF score descending
        # Tie-breaker 1: Dense score descending (if present)
        # Tie-breaker 2: BM25 score descending (if present)
        # Tie-breaker 3: Chunk ID string ascending
        sorted_candidates = sorted(
            fused_map.values(),
            key=lambda item: (
                -item["rrf_score"],
                -(item["dense_score"] if item["dense_score"] is not None else -999.0),
                -(item["bm25_score"] if item["bm25_score"] is not None else -999.0),
                str(item["chunk_id"]),
            ),
        )

        # 4. Construct HybridCandidate list with assigned RRF rank
        results: List[HybridCandidate] = []
        for rank_idx, item in enumerate(sorted_candidates[:eff_top_k], start=1):
            results.append(
                HybridCandidate(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    filename=item["filename"],
                    text=item["text"],
                    character_count=item["character_count"],
                    page_start=item["page_start"],
                    page_end=item["page_end"],
                    metadata_json=item["metadata_json"],
                    dense_score=item["dense_score"],
                    dense_rank=item["dense_rank"],
                    bm25_score=item["bm25_score"],
                    bm25_rank=item["bm25_rank"],
                    rrf_score=item["rrf_score"],
                    rrf_rank=rank_idx,
                )
            )

        logger.debug(
            "RRF fusion combined %d dense and %d BM25 candidates into %d unique candidates (k=%d, top_k=%d)",
            len(dense_candidates),
            len(bm25_candidates),
            len(results),
            k,
            eff_top_k,
        )
        return results


# Singleton instance
rrf_service = RRFService()
