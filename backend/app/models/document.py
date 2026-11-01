"""SQLAlchemy ORM model for ingested documents."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk


class Document(Base):
    """Represents an ingested document and its extraction lifecycle state."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique internal storage filename to prevent collisions",
    )
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Original uploaded filename preserved for display",
    )
    file_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        comment="Normalized file extension/type: pdf, docx, txt",
    )
    file_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="File size in bytes",
    )
    storage_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Relative path from storage root directory",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="uploaded",
        index=True,
        comment="Processing status: uploaded, processing, completed, failed",
    )
    total_pages: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total page count (for paginated documents like PDF)",
    )
    total_characters: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total character count extracted from the document",
    )
    total_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of chunks generated",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Diagnostic failure message if status is failed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Cascading deletion: deleting a document removes all associated chunks
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.original_filename} status={self.status}>"
