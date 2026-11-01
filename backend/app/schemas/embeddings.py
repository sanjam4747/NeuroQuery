"""Pydantic schemas for semantic embeddings and model metadata."""

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EmbeddingModelInfo(BaseModel):
    """Metadata regarding active embedding model and inference configuration."""

    model_name: str = Field(..., description="Active Sentence-Transformers model name.")
    dimension: int = Field(..., description="Vector dimensionality (e.g. 384).")
    max_sequence_length: int = Field(..., description="Maximum token sequence length supported.")
    device: str = Field(..., description="Inference device: cpu or cuda.")
    batch_size: int = Field(..., description="Configured batch size for vector generation.")
    normalize: bool = Field(..., description="Whether vectors are L2-normalized for cosine similarity.")
    metric: str = Field("cosine", description="Target vector similarity metric.")
    is_loaded: bool = Field(..., description="Whether model weights are loaded in memory.")
    description: str = Field(..., description="Model description and optimization notes.")


class ReembedResponse(BaseModel):
    """Result of re-embedding a document's chunks."""

    document_id: uuid.UUID = Field(..., description="Target document identifier.")
    status: str = Field(..., description="Updated embedding status: completed or failed.")
    total_embedded_chunks: int = Field(..., description="Number of chunks re-embedded.")
    embedding_model: str = Field(..., description="Model identifier used.")
    message: str = Field(..., description="Result message.")
