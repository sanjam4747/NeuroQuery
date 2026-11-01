"""SQLAlchemy ORM model for document chunks."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional
from sqlalchemy import DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base):
    """Represents a discrete text passage extracted and chunked from a source document."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="0-indexed sequence position within parent document",
    )
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Cleaned, normalized chunk text payload",
    )
    character_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Length of the chunk text in characters",
    )
    page_start: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="1-indexed starting page number (for PDFs)",
    )
    page_end: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="1-indexed ending page number (for PDFs)",
    )
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Provenance metadata (e.g. section, paragraph index, char offsets)",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} index={self.chunk_index} len={self.character_count}>"
