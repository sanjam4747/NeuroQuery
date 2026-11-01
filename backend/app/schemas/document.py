"""Pydantic schemas for Document representations."""

import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Public document metadata payload (no internal storage paths exposed)."""

    id: uuid.UUID = Field(..., description="Unique document identifier.")
    filename: str = Field(..., description="Unique internal system filename.")
    original_filename: str = Field(..., description="User's original upload filename.")
    file_type: str = Field(..., description="Document format: pdf, docx, txt.")
    file_size: int = Field(..., description="File size in bytes.")
    status: str = Field(..., description="Processing status: uploaded, processing, completed, failed.")
    total_pages: Optional[int] = Field(None, description="Total page count (for PDFs).")
    total_characters: int = Field(0, description="Total extracted character count.")
    total_chunks: int = Field(0, description="Total number of chunks generated.")
    error_message: Optional[str] = Field(None, description="Diagnostic error message if status is failed.")
    embedding_status: str = Field("pending", description="Embedding generation status: pending, generating, completed, failed.")
    embedding_model: Optional[str] = Field(None, description="Model identifier used for chunk embeddings.")
    total_embedded_chunks: int = Field(0, description="Total chunks with generated embeddings.")
    embedding_error: Optional[str] = Field(None, description="Diagnostic error message if embedding failed.")
    created_at: datetime = Field(..., description="Upload timestamp in UTC.")
    updated_at: datetime = Field(..., description="Last update timestamp in UTC.")

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Paginated list of document summaries."""

    items: List[DocumentResponse] = Field(..., description="List of documents.")
    total: int = Field(..., description="Total matching documents count.")
    skip: int = Field(..., description="Pagination offset.")
    limit: int = Field(..., description="Pagination page size limit.")
