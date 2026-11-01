"""Tests for dense semantic vector retrieval service."""

import math
import uuid
import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.dense_retrieval import dense_retrieval_service, DenseCandidate
from app.services.embeddings import embedding_service


def test_dense_retrieval_query_embedding_and_dimension(test_db: Session):
    """Verify dense retrieval generates a valid 384d unit query vector."""
    query = "semantic document retrieval"
    vec = embedding_service.embed_text(query)
    assert len(vec) == 384
    assert math.isclose(np.linalg.norm(vec), 1.0, rel_tol=1e-4)


def test_dense_retrieval_empty_query(test_db: Session):
    """Verify empty or whitespace query returns empty candidates."""
    assert dense_retrieval_service.search(test_db, "") == []
    assert dense_retrieval_service.search(test_db, "   \n\t  ") == []


def test_dense_retrieval_empty_database(test_db: Session):
    """Verify dense search with no documents in DB returns empty list."""
    results = dense_retrieval_service.search(test_db, "sample query", top_k=5)
    assert results == []


def test_dense_retrieval_ordering_and_filtering(test_db: Session):
    """Verify dense search orders by cosine similarity and respects document filtering."""
    # Create two test documents
    doc1 = Document(
        id=uuid.uuid4(),
        filename="ai_overview.pdf",
        original_filename="ai_overview.pdf",
        file_type="pdf",
        storage_path="mock/path1",
        file_size=1024,
        status="completed",
        embedding_status="completed",
    )
    doc2 = Document(
        id=uuid.uuid4(),
        filename="cooking_recipes.txt",
        original_filename="cooking_recipes.txt",
        file_type="txt",
        storage_path="mock/path2",
        file_size=512,
        status="completed",
        embedding_status="completed",
    )
    test_db.add_all([doc1, doc2])
    test_db.flush()

    # Generate embeddings for two distinct texts
    ai_text = "Artificial intelligence and machine learning models for neural retrieval."
    cook_text = "Baking artisan bread with flour, water, yeast, and salt in a hot oven."

    ai_vec = embedding_service.embed_text(ai_text)
    cook_vec = embedding_service.embed_text(cook_text)

    chunk1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc1.id,
        chunk_index=0,
        text=ai_text,
        character_count=len(ai_text),
        page_start=1,
        page_end=1,
        embedding=ai_vec,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    chunk2 = Chunk(
        id=uuid.uuid4(),
        document_id=doc2.id,
        chunk_index=0,
        text=cook_text,
        character_count=len(cook_text),
        embedding=cook_vec,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    test_db.add_all([chunk1, chunk2])
    test_db.commit()

    # Search for AI query
    ai_query = "neural network retrieval algorithms"
    results = dense_retrieval_service.search(test_db, ai_query, top_k=2)

    assert len(results) == 2
    # First result should be the AI chunk with higher similarity
    assert results[0].chunk_id == chunk1.id
    assert results[0].filename == "ai_overview.pdf"
    assert results[0].rank == 1
    assert results[0].score > results[1].score

    # Test document filtering
    filtered = dense_retrieval_service.search(test_db, ai_query, top_k=2, document_ids=[doc2.id])
    assert len(filtered) == 1
    assert filtered[0].chunk_id == chunk2.id


def test_dense_retrieval_excludes_unembedded_or_uncompleted_chunks(test_db: Session):
    """Verify chunks lacking embeddings or from uncompleted documents are excluded."""
    doc_pending = Document(
        id=uuid.uuid4(),
        filename="pending.pdf",
        original_filename="pending.pdf",
        file_type="pdf",
        storage_path="mock/path_p",
        file_size=1024,
        status="processing",  # Not completed
    )
    test_db.add(doc_pending)
    test_db.flush()

    chunk_no_emb = Chunk(
        id=uuid.uuid4(),
        document_id=doc_pending.id,
        chunk_index=0,
        text="Text in a pending document without embeddings.",
        character_count=45,
        embedding=None,  # No embedding
    )
    test_db.add(chunk_no_emb)
    test_db.commit()

    results = dense_retrieval_service.search(test_db, "pending document", top_k=5)
    assert not any(r.chunk_id == chunk_no_emb.id for r in results)
