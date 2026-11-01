"""Pydantic schemas for Phase 5 Retrieval-Augmented Generation (RAG)."""

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.retrieval import RetrievalMetadata


class RAGQueryRequest(BaseModel):
    """Payload for natural-language question answering over document collections."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question to answer using document evidence.",
    )
    document_ids: Optional[List[uuid.UUID]] = Field(
        default=None,
        description="Optional list of document UUIDs to scope retrieval.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top reranked chunks to select as potential evidence.",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking before evidence selection.",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff for dense candidates.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional LLM generation temperature (default: 0.0 for deterministic grounding).",
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        ge=64,
        le=4096,
        description="Maximum generation token budget.",
    )

    @field_validator("question")
    @classmethod
    def clean_question(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        return cleaned


class RAGCitationItem(BaseModel):
    """Verified citation item tied directly to a persistent database chunk."""

    source_id: str = Field(..., description="Context identifier, e.g. 'S1', 'S2'")
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text_snippet: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float
    reranker_score: Optional[float] = None
    final_rank: int


class RAGClaimItem(BaseModel):
    """An individual atomic factual claim mapped to verified citations."""

    claim_text: str
    citation_ids: List[str]
    citations: List[RAGCitationItem] = Field(default_factory=list)
    is_supported: bool = Field(default=True, description="Whether claim has at least one verified supporting citation")
    unsupported_citation_ids: List[str] = Field(default_factory=list, description="Invalid or hallucinated citation tags cited for this claim")


class RAGGenerationMetadata(BaseModel):
    """Telemetry regarding the LLM provider execution."""

    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    generation_latency_ms: float


class RAGDiagnostics(BaseModel):
    """Granular latency and resource profiling across the full RAG pipeline."""

    query_validation_ms: float
    retrieval_latency_ms: float
    context_building_ms: float
    inference_latency_ms: float
    citation_validation_ms: float
    total_latency_ms: float
    evidence_chunks_retrieved: int
    evidence_chunks_selected: int
    context_tokens_used: int


class RAGQueryResponse(BaseModel):
    """Complete response payload for a document question-answering request."""

    question: str
    answer: str
    is_answered: bool
    claims: List[RAGClaimItem]
    citations: List[RAGCitationItem]
    evidence_sources: List[RAGCitationItem]
    insufficient_evidence_reason: Optional[str] = None
    invalid_citation_ids: List[str] = Field(default_factory=list)
    parsing_status: str = Field(default="valid_structured", description="One of: valid_structured, recovered_with_repair, valid_abstention")
    retrieval_metadata: RetrievalMetadata
    generation_metadata: RAGGenerationMetadata
    diagnostics: RAGDiagnostics


class RAGHealthResponse(BaseModel):
    """Diagnostic health status for the RAG and LLM provider subsystems."""

    status: str  # "healthy", "degraded", "unavailable"
    provider: str
    model: str
    provider_available: bool
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
