"""Dense semantic retrieval service utilizing pgvector cosine similarity."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embeddings import embedding_service


@dataclass
class DenseCandidate:
    """Represents a candidate chunk retrieved via dense semantic vector search."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    character_count: int
    page_start: Optional[int]
    page_end: Optional[int]
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # Cosine similarity in range [-1.0, 1.0]
    rank: int = 1       # 1-indexed rank position


class DenseRetrievalService:
    """
    Manages vector similarity search against persisted chunk embeddings.

    Design details:
    - Queries are embedded using the active singleton EmbeddingService (BAAI/bge-small-en-v1.5, 384d).
    - In PostgreSQL + pgvector, similarity search runs at the database level using cosine distance:
      `Chunk.embedding.cosine_distance(query_vector)` mapped to `similarity = 1 - distance`.
    - In SQLite (used in local test suites and offline development), automatically utilizes an in-memory
      vector cosine evaluation fallback without crashing.
    - Excludes chunks without embeddings and documents that are not in 'completed' state.
    - Supports filtering by document IDs and an optional similarity threshold cutoff.
    """

    def search(
        self,
        db: Session,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[uuid.UUID]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[DenseCandidate]:
        """
        Executes dense semantic retrieval for a natural-language query.

        Args:
            db: Active SQLAlchemy database session.
            query: User's search query string.
            top_k: Number of candidate chunks to retrieve (defaults to settings.DENSE_RETRIEVAL_TOP_K).
            document_ids: Optional list of document UUIDs to scope retrieval.
            similarity_threshold: Optional cosine similarity cutoff.

        Returns:
            List of DenseCandidate objects ordered by similarity descending.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        eff_top_k = top_k or settings.DENSE_RETRIEVAL_TOP_K
        eff_threshold = similarity_threshold if similarity_threshold is not None else settings.DENSE_SIMILARITY_THRESHOLD

        # Generate query embedding using shared singleton service
        query_vector = embedding_service.embed_text(clean_query)
        if len(query_vector) != settings.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query vector dimension mismatch: got {len(query_vector)}, expected {settings.EMBEDDING_DIMENSION}."
            )

        # Detect database dialect
        dialect_name = db.bind.dialect.name if db.bind else "sqlite"
        is_postgres = dialect_name == "postgresql"

        candidates: List[DenseCandidate] = []

        if is_postgres:
            # PostgreSQL + pgvector database-level cosine distance query
            distance_expr = Chunk.embedding.cosine_distance(query_vector)
            similarity_expr = 1.0 - distance_expr

            stmt = (
                select(Chunk, Document.filename, similarity_expr.label("similarity"))
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Chunk.embedding.is_not(None),
                    Document.status == "completed",
                )
            )

            if document_ids:
                stmt = stmt.where(Chunk.document_id.in_(document_ids))

            if eff_threshold is not None:
                stmt = stmt.where(similarity_expr >= eff_threshold)

            stmt = stmt.order_by(distance_expr.asc(), Chunk.id.asc()).limit(eff_top_k)

            rows = db.execute(stmt).all()
            for rank_idx, (chunk, fname, sim) in enumerate(rows, start=1):
                candidates.append(
                    DenseCandidate(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        filename=fname,
                        text=chunk.text,
                        character_count=chunk.character_count,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        metadata_json=chunk.metadata_json or {},
                        score=round(float(sim), 4),
                        rank=rank_idx,
                    )
                )
        else:
            # SQLite fallback for offline development and fast unit testing
            stmt = (
                select(Chunk, Document.filename)
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Chunk.embedding.is_not(None),
                    Document.status == "completed",
                )
            )

            if document_ids:
                stmt = stmt.where(Chunk.document_id.in_(document_ids))

            rows = db.execute(stmt).all()
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)

            scored_chunks = []
            for chunk, fname in rows:
                if chunk.embedding is not None:
                    c_vec = np.array(chunk.embedding, dtype=np.float32)
                    c_norm = np.linalg.norm(c_vec)
                    if c_norm > 0 and q_norm > 0:
                        sim = float(np.dot(c_vec, q_vec) / (c_norm * q_norm))
                    else:
                        sim = 0.0

                    if eff_threshold is None or sim >= eff_threshold:
                        scored_chunks.append((chunk, fname, sim))

            # Deterministic sort: similarity score descending, chunk ID ascending
            scored_chunks.sort(key=lambda item: (-item[2], str(item[0].id)))
            top_rows = scored_chunks[:eff_top_k]

            for rank_idx, (chunk, fname, sim) in enumerate(top_rows, start=1):
                candidates.append(
                    DenseCandidate(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        filename=fname,
                        text=chunk.text,
                        character_count=chunk.character_count,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        metadata_json=chunk.metadata_json or {},
                        score=round(float(sim), 4),
                        rank=rank_idx,
                    )
                )

        logger.debug(
            "Dense retrieval returned %d candidates for query '%s' (top_k=%d, postgres=%s)",
            len(candidates),
            clean_query[:30],
            eff_top_k,
            is_postgres,
        )
        return candidates


# Singleton instance
dense_retrieval_service = DenseRetrievalService()
