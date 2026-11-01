"""Sparse retrieval service using BM25 with intelligent lexical tokenization."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import threading
from typing import Any, Dict, List, Optional, Set
import uuid
import math
import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.logging import logger
from app.models.chunk import Chunk
from app.models.document import Document


class LuceneBM25(BM25Okapi):
    """
    Production-grade BM25 model implementing Lucene / Elasticsearch non-negative IDF:
    IDF(q) = ln(1.0 + (N - n + 0.5) / (n + 0.5))

    Standard raw Robertson BM25 IDF produces zero or negative weights whenever a term
    appears in half or more documents (common in small collections or when N <= 2).
    Lucene's formulation guarantees strictly positive IDF for all matching tokens
    while preserving monotonic rarity discrimination.
    """

    def _calc_idf(self, nd: Dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        self.average_idf = sum(self.idf.values()) / (len(self.idf) or 1)


# Conservative functional stopwords to avoid diluting BM25 inverse document frequencies
# Domain concepts, technical acronyms, and numeric tokens are strictly preserved.
COMMON_FUNCTIONAL_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "from", "in", "into", "of", "off", "on", "onto",
    "out", "over", "to", "up", "with", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "must", "it", "its",
    "this", "that", "these", "those", "there", "their", "they",
}

# Regex token matcher: captures words, numbers, identifiers, and hyphenated/dotted technical symbols
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_\-\./#+]+", re.UNICODE)


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Deterministic tokenizer for BM25 indexing and querying.

    Preserves:
    - Technical terminology and acronyms (e.g. 'pgvector', 'HNSW', 'BERT', 'NLP')
    - Identifiers, version numbers, and file extensions (e.g. 'v1.5', 'report.pdf', 'id_123')
    - Numeric values and quantities (e.g. '384', '512', '25MB')
    - Hyphenated technical compounds (e.g. 'cross-encoder', 'token-weighted')

    Filters:
    - Pure punctuation
    - Common grammatical functional stopwords
    """
    if not text:
        return []

    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group(0).strip(".-_")
        if not token:
            continue
        # Skip pure functional stopwords unless they are numeric or acronyms
        if token in COMMON_FUNCTIONAL_STOPWORDS and not token.isdigit():
            continue
        tokens.append(token)

    return tokens


@dataclass
class BM25Candidate:
    """Represents a candidate chunk retrieved via sparse BM25 keyword matching."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    character_count: int
    page_start: Optional[int]
    page_end: Optional[int]
    metadata_json: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # BM25 score
    rank: int = 1       # 1-indexed rank position


class BM25Service:
    """
    Manages in-memory BM25 index lifecycle and sparse keyword retrieval.

    Design details:
    - Thread-safe index management via re-entrant lock.
    - Tokenized corpus is maintained in memory for sub-millisecond retrieval.
    - Lifecycle synchronization:
      - Automatically builds on startup or first query if uninitialized.
      - Incrementally receives new chunks when documents complete ingestion.
      - Automatically purges deleted document chunks when documents are removed.
    - Returns ranked chunks with non-zero BM25 scores.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: List[uuid.UUID] = []
        self._tokenized_corpus: List[List[str]] = []
        self._chunks_map: Dict[uuid.UUID, Dict[str, Any]] = {}
        self._last_updated_at: Optional[datetime] = None
        self._is_initialized: bool = False

    @property
    def is_indexed(self) -> bool:
        """Indicates whether the BM25 index currently contains searchable chunks."""
        with self._lock:
            return self._bm25 is not None and len(self._chunk_ids) > 0

    @property
    def total_chunks(self) -> int:
        """Total number of chunks currently indexed in BM25."""
        with self._lock:
            return len(self._chunk_ids)

    @property
    def total_documents(self) -> int:
        """Total number of unique documents represented in the BM25 index."""
        with self._lock:
            doc_ids = {meta["document_id"] for meta in self._chunks_map.values()}
            return len(doc_ids)

    @property
    def last_updated_at(self) -> Optional[datetime]:
        """UTC timestamp of last index modification."""
        with self._lock:
            return self._last_updated_at

    def get_status(self) -> Dict[str, Any]:
        """Telemetry snapshot of BM25 index status."""
        with self._lock:
            return {
                "is_indexed": self.is_indexed,
                "total_chunks": self.total_chunks,
                "total_documents": self.total_documents,
                "last_updated_at": self._last_updated_at.isoformat() if self._last_updated_at else None,
            }

    def build_index(self, db: Session) -> int:
        """
        Builds or rebuilds the BM25 index from all completed document chunks in the database.

        Returns:
            Count of indexed chunks.
        """
        with self._lock:
            logger.info("Building BM25 index from persistent database chunks...")
            stmt = (
                select(Chunk, Document.filename)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.status == "completed")
                .order_by(Chunk.created_at.asc())
            )
            rows = db.execute(stmt).all()

            chunk_ids: List[uuid.UUID] = []
            corpus: List[List[str]] = []
            chunks_map: Dict[uuid.UUID, Dict[str, Any]] = {}

            for chunk, fname in rows:
                tokens = tokenize_for_bm25(chunk.text)
                chunk_ids.append(chunk.id)
                corpus.append(tokens)
                chunks_map[chunk.id] = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": fname,
                    "text": chunk.text,
                    "character_count": chunk.character_count,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "metadata_json": chunk.metadata_json or {},
                }

            if corpus:
                self._bm25 = LuceneBM25(corpus)
            else:
                self._bm25 = None

            self._chunk_ids = chunk_ids
            self._tokenized_corpus = corpus
            self._chunks_map = chunks_map
            self._last_updated_at = datetime.now(timezone.utc)
            self._is_initialized = True

            logger.info(
                "BM25 index built successfully with %d chunks across %d documents.",
                len(chunk_ids),
                len({m["document_id"] for m in chunks_map.values()}),
            )
            return len(chunk_ids)

    def add_document_chunks(self, chunks: List[Chunk], filename: str) -> None:
        """
        Incrementally registers chunks of a newly ingested or updated document.
        Reconstructs the BM25Okapi instance with updated corpus.
        """
        if not chunks:
            return

        with self._lock:
            # Avoid duplicate registration
            existing_ids = set(self._chunk_ids)
            added_count = 0

            for chunk in chunks:
                if chunk.id in existing_ids:
                    # Update existing metadata
                    self._chunks_map[chunk.id] = {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "filename": filename,
                        "text": chunk.text,
                        "character_count": chunk.character_count,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "metadata_json": chunk.metadata_json or {},
                    }
                    continue

                tokens = tokenize_for_bm25(chunk.text)
                self._chunk_ids.append(chunk.id)
                self._tokenized_corpus.append(tokens)
                self._chunks_map[chunk.id] = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": filename,
                    "text": chunk.text,
                    "character_count": chunk.character_count,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "metadata_json": chunk.metadata_json or {},
                }
                added_count += 1

            if self._tokenized_corpus:
                self._bm25 = LuceneBM25(self._tokenized_corpus)
            else:
                self._bm25 = None

            self._last_updated_at = datetime.now(timezone.utc)
            logger.info("Added %d new chunks to BM25 index (total=%d).", added_count, len(self._chunk_ids))

    def remove_document_chunks(self, document_id: uuid.UUID) -> int:
        """
        Removes all chunks associated with a deleted document and refreshes the BM25 index.

        Returns:
            Number of chunks removed.
        """
        with self._lock:
            initial_count = len(self._chunk_ids)
            new_chunk_ids: List[uuid.UUID] = []
            new_corpus: List[List[str]] = []
            new_chunks_map: Dict[uuid.UUID, Dict[str, Any]] = {}

            for idx, c_id in enumerate(self._chunk_ids):
                meta = self._chunks_map.get(c_id)
                if meta and meta["document_id"] == document_id:
                    continue
                new_chunk_ids.append(c_id)
                new_corpus.append(self._tokenized_corpus[idx])
                if meta:
                    new_chunks_map[c_id] = meta

            removed_count = initial_count - len(new_chunk_ids)
            self._chunk_ids = new_chunk_ids
            self._tokenized_corpus = new_corpus
            self._chunks_map = new_chunks_map

            if self._tokenized_corpus:
                self._bm25 = LuceneBM25(self._tokenized_corpus)
            else:
                self._bm25 = None

            self._last_updated_at = datetime.now(timezone.utc)
            logger.info("Evicted %d chunks for deleted document %s from BM25 index.", removed_count, document_id)
            return removed_count

    def search(
        self,
        db: Optional[Session],
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[uuid.UUID]] = None,
    ) -> List[BM25Candidate]:
        """
        Executes sparse lexical keyword retrieval using BM25Okapi.

        Args:
            db: Optional database session to build index if uninitialized.
            query: User's search query string.
            top_k: Number of candidates to retrieve (defaults to settings.BM25_RETRIEVAL_TOP_K).
            document_ids: Optional list of document UUIDs to scope retrieval.

        Returns:
            List of BM25Candidate objects ordered by score descending.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        eff_top_k = top_k or settings.BM25_RETRIEVAL_TOP_K

        with self._lock:
            # Auto-initialize from DB if uninitialized
            if not self._is_initialized and db is not None:
                self.build_index(db)

            if self._bm25 is None or not self._chunk_ids:
                return []

            query_tokens = tokenize_for_bm25(clean_query)
            if not query_tokens:
                return []

            scores = self._bm25.get_scores(query_tokens)
            doc_filter = set(document_ids) if document_ids else None

            scored_candidates: List[tuple[int, float]] = []
            for idx, score in enumerate(scores):
                if score <= 0.0:
                    continue
                c_id = self._chunk_ids[idx]
                meta = self._chunks_map.get(c_id)
                if not meta:
                    continue
                if doc_filter is not None and meta["document_id"] not in doc_filter:
                    continue
                scored_candidates.append((idx, float(score)))

            # Deterministic sort: score descending, chunk ID ascending
            scored_candidates.sort(key=lambda item: (-item[1], str(self._chunk_ids[item[0]])))
            top_scored = scored_candidates[:eff_top_k]

            results: List[BM25Candidate] = []
            for rank_idx, (c_idx, score) in enumerate(top_scored, start=1):
                c_id = self._chunk_ids[c_idx]
                meta = self._chunks_map[c_id]
                results.append(
                    BM25Candidate(
                        chunk_id=meta["chunk_id"],
                        document_id=meta["document_id"],
                        filename=meta["filename"],
                        text=meta["text"],
                        character_count=meta["character_count"],
                        page_start=meta["page_start"],
                        page_end=meta["page_end"],
                        metadata_json=meta["metadata_json"],
                        score=round(score, 4),
                        rank=rank_idx,
                    )
                )

            logger.debug(
                "BM25 retrieval returned %d candidates for query '%s' (top_k=%d, tokens=%s)",
                len(results),
                clean_query[:30],
                eff_top_k,
                query_tokens[:5],
            )
            return results


# Singleton instance
bm25_service = BM25Service()
