"""Add vector embedding columns to chunks and documents.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-01 15:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add vector embedding and model tracking columns to chunks
    op.add_column("chunks", sa.Column("embedding", Vector(384), nullable=True))
    op.add_column("chunks", sa.Column("embedding_model", sa.String(length=100), nullable=True))
    op.add_column("chunks", sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))

    # 2. Add embedding tracking columns to documents
    op.add_column(
        "documents",
        sa.Column("embedding_status", sa.String(length=20), server_default="pending", nullable=False),
    )
    op.add_column("documents", sa.Column("embedding_model", sa.String(length=100), nullable=True))
    op.add_column(
        "documents",
        sa.Column("total_embedded_chunks", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column("documents", sa.Column("embedding_error", sa.Text(), nullable=True))

    # 3. Create index on document embedding_status
    op.create_index(op.f("ix_documents_embedding_status"), "documents", ["embedding_status"], unique=False)

    # 4. Create HNSW cosine similarity index on PostgreSQL with pgvector
    bind = op.get_bind()
    if bind and bind.dialect.name == "postgresql":
        op.create_index(
            "ix_chunks_embedding_cosine",
            "chunks",
            ["embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind and bind.dialect.name == "postgresql":
        op.drop_index(
            "ix_chunks_embedding_cosine",
            table_name="chunks",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )

    op.drop_index(op.f("ix_documents_embedding_status"), table_name="documents")

    op.drop_column("documents", "embedding_error")
    op.drop_column("documents", "total_embedded_chunks")
    op.drop_column("documents", "embedding_model")
    op.drop_column("documents", "embedding_status")

    op.drop_column("chunks", "embedded_at")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")
