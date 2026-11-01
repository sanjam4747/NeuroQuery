"""Initial migration: Enable pgvector extension.

Revision ID: 0001
Revises: None
Create Date: 2026-11-01 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension in PostgreSQL
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    # Disable pgvector extension
    op.execute("DROP EXTENSION IF EXISTS vector;")
