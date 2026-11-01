"""Pydantic request and response schemas for Phase 4 Hybrid Retrieval."""

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class RetrievalQueryRequest(BaseModel):
    """Payload for natural-language hybrid search query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language user query to retrieve relevant document passages for.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of final ranked results to return.",
    )
    document_ids: Optional[List[uuid.UUID]] = Field(
        default=None,
        description="Optional list of document UUIDs to filter retrieval scope.",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking on top fused candidates.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff for dense candidates.",
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace only.")
        return cleaned


class RetrievedChunkResult(BaseModel):
    """Detailed ranked chunk result with full provenance and explainable scoring metrics."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    character_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    # Retrieval Scoring Explainability
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: float
    rrf_rank: int
    reranker_score: Optional[float] = None
    final_rank: int


class RetrievalMetadata(BaseModel):
    """Telemetry and timing breakdown for the hybrid retrieval pipeline."""

    dense_candidates: int
    bm25_candidates: int
    fused_candidates: int
    reranked_candidates: int
    returned_results: int
    reranking_enabled: bool
    dense_latency_ms: float
    bm25_latency_ms: float
    rrf_latency_ms: float
    reranker_latency_ms: float
    total_latency_ms: float


class RetrievalQueryResponse(BaseModel):
    """Complete response payload for a hybrid search query."""

    query: str
    results: List[RetrievedChunkResult]
    retrieval_metadata: RetrievalMetadata


class RetrievalInfoResponse(BaseModel):
    """Telemetry and configuration status for the hybrid retrieval engine."""

    embedding_model: str
    embedding_dimension: int
    reranker_model: str
    dense_available: bool
    bm25_indexed: bool
    bm25_total_chunks: int
    bm25_total_documents: int
    bm25_last_updated: Optional[str] = None
    rrf_constant: int
    reranking_enabled_by_default: bool
    device: str


class RebuildIndexResponse(BaseModel):
    """Response payload when rebuilding the sparse BM25 index."""

    status: str
    indexed_chunks: int
    indexed_documents: int
    duration_ms: float
