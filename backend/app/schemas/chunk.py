"""Pydantic schemas for Document Chunk representations."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkResponse(BaseModel):
    """Chunk details with provenance metadata."""

    id: uuid.UUID = Field(..., description="Unique chunk identifier.")
    document_id: uuid.UUID = Field(..., description="Parent document identifier.")
    chunk_index: int = Field(..., description="0-indexed position within parent document.")
    text: str = Field(..., description="Chunk text content.")
    character_count: int = Field(..., description="Length of text in characters.")
    page_start: Optional[int] = Field(None, description="Starting page (PDF).")
    page_end: Optional[int] = Field(None, description="Ending page (PDF).")
    metadata_json: Dict[str, Any] = Field(default_factory=dict, description="Structural provenance metadata.")
    created_at: datetime = Field(..., description="Creation timestamp in UTC.")

    model_config = ConfigDict(from_attributes=True)


class ChunkListResponse(BaseModel):
    """Paginated list of chunks for a document."""

    items: List[ChunkResponse] = Field(..., description="List of chunks.")
    total: int = Field(..., description="Total chunks count for this document.")
    skip: int = Field(..., description="Pagination offset.")
    limit: int = Field(..., description="Pagination page size limit.")
