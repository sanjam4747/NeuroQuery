"""Tests for sparse BM25 retrieval service and lexical tokenization."""

import uuid
import pytest
from sqlalchemy.orm import Session
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.bm25_retrieval import BM25Service, tokenize_for_bm25


def test_bm25_tokenization_preserves_technical_terms():
    """Verify tokenizer preserves technical acronyms, numbers, hyphens, and identifiers."""
    text = "Deploying pgvector v0.8.0 with HNSW index on port 5432 and 384-dimensional embeddings."
    tokens = tokenize_for_bm25(text)

    assert "pgvector" in tokens
    assert "v0.8.0" in tokens
    assert "hnsw" in tokens
    assert "5432" in tokens
    assert "384-dimensional" in tokens
    # Common functional stopwords should be excluded
    assert "with" not in tokens
    assert "on" not in tokens
    assert "and" not in tokens


def test_bm25_empty_query():
    """Verify empty query returns empty candidates."""
    service = BM25Service()
    assert service.search(db=None, query="") == []
    assert service.search(db=None, query="   \t\n  ") == []


def test_bm25_empty_index():
    """Verify search on an empty unindexed service returns empty list."""
    service = BM25Service()
    assert service.is_indexed is False
    assert service.search(db=None, query="test query") == []


def test_bm25_indexing_and_exact_keyword_retrieval(test_db: Session):
    """Verify BM25 indexes chunks and retrieves exact keyword matches with scores."""
    service = BM25Service()

    doc1 = Document(
        id=uuid.uuid4(),
        filename="database_guide.pdf",
        original_filename="database_guide.pdf",
        file_type="pdf",
        storage_path="mock/db",
        file_size=1024,
        status="completed",
    )
    doc2 = Document(
        id=uuid.uuid4(),
        filename="cooking_recipes.txt",
        original_filename="cooking_recipes.txt",
        file_type="txt",
        storage_path="mock/cook",
        file_size=512,
        status="completed",
    )
    test_db.add_all([doc1, doc2])
    test_db.flush()

    chunk1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc1.id,
        chunk_index=0,
        text="PostgreSQL with pgvector enables HNSW indexing for rapid cosine similarity queries.",
        character_count=85,
        page_start=1,
        page_end=1,
    )
    chunk2 = Chunk(
        id=uuid.uuid4(),
        document_id=doc2.id,
        chunk_index=0,
        text="A classic Italian pasta recipe uses durum wheat semolina, eggs, and freshly grated parmesan.",
        character_count=92,
        page_start=None,
        page_end=None,
    )
    test_db.add_all([chunk1, chunk2])
    test_db.commit()

    count = service.build_index(test_db)
    assert count == 2
    assert service.is_indexed is True
    assert service.total_chunks == 2
    assert service.total_documents == 2

    # Query for exact technical keywords
    results = service.search(db=test_db, query="pgvector HNSW indexing", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == chunk1.id
    assert results[0].score > 0.0
    assert results[0].rank == 1

    # Query for exact recipe keywords
    cook_results = service.search(db=test_db, query="durum wheat semolina pasta", top_k=2)
    assert len(cook_results) >= 1
    assert cook_results[0].chunk_id == chunk2.id

    # Document filtering
    filtered = service.search(db=test_db, query="pgvector", document_ids=[doc2.id])
    assert len(filtered) == 0


def test_bm25_incremental_add_and_eviction():
    """Verify adding new document chunks and evicting deleted document chunks."""
    service = BM25Service()

    doc_id1 = uuid.uuid4()
    doc_id2 = uuid.uuid4()

    chunk1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc_id1,
        chunk_index=0,
        text="First document chunk discussing information retrieval architectures.",
        character_count=65,
    )
    chunk2 = Chunk(
        id=uuid.uuid4(),
        document_id=doc_id2,
        chunk_index=0,
        text="Second document chunk describing machine learning algorithms.",
        character_count=60,
    )

    # Incrementally add doc1
    service.add_document_chunks([chunk1], filename="doc1.txt")
    assert service.total_chunks == 1
    assert service.total_documents == 1

    # Incrementally add doc2
    service.add_document_chunks([chunk2], filename="doc2.txt")
    assert service.total_chunks == 2
    assert service.total_documents == 2

    # Query doc1
    res1 = service.search(db=None, query="retrieval architectures")
    assert len(res1) == 1
    assert res1[0].chunk_id == chunk1.id

    # Evict doc1
    removed = service.remove_document_chunks(doc_id1)
    assert removed == 1
    assert service.total_chunks == 1
    assert service.total_documents == 1

    # Query doc1 again - should return empty
    res_after = service.search(db=None, query="retrieval architectures")
    assert len(res_after) == 0

    # Query doc2 - should remain intact
    res2 = service.search(db=None, query="machine learning algorithms")
    assert len(res2) == 1
    assert res2[0].chunk_id == chunk2.id
