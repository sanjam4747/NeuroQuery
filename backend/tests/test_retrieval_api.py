"""API integration tests for retrieval search, info, and index rebuild endpoints."""

import uuid
import pytest
from sqlalchemy.orm import Session
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embeddings import embedding_service


def test_retrieval_info_endpoint(client):
    """Verify GET /api/v1/retrieval/info returns model names, dimensions, and index state."""
    res = client.get("/api/v1/retrieval/info")
    assert res.status_code == 200
    data = res.json()
    assert data["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert data["embedding_dimension"] == 384
    assert data["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert data["dense_available"] is True
    assert "bm25_indexed" in data
    assert data["rrf_constant"] == 60


def test_retrieval_search_validation_errors(client):
    """Verify validation errors for missing, empty, or whitespace queries."""
    res_empty = client.post("/api/v1/retrieval/search", json={"query": ""})
    assert res_empty.status_code == 422

    res_whitespace = client.post("/api/v1/retrieval/search", json={"query": "   \n\t  "})
    assert res_whitespace.status_code == 422

    res_missing = client.post("/api/v1/retrieval/search", json={})
    assert res_missing.status_code == 422


def test_retrieval_search_end_to_end(client, test_db: Session):
    """Verify POST /api/v1/retrieval/search executes dense, BM25, RRF, and reranking."""
    # Seed document and chunks
    doc = Document(
        id=uuid.uuid4(),
        filename="search_test_doc.pdf",
        original_filename="search_test_doc.pdf",
        file_type="pdf",
        storage_path="mock/search_path",
        file_size=2048,
        status="completed",
        embedding_status="completed",
    )
    test_db.add(doc)
    test_db.flush()

    text1 = "FastAPI powers asynchronous web microservices with automatic OpenAPI documentation."
    text2 = "PostgreSQL 16 with pgvector extension enables fast cosine similarity search."

    v1 = embedding_service.embed_text(text1)
    v2 = embedding_service.embed_text(text2)

    c1 = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=0,
        text=text1,
        character_count=len(text1),
        page_start=1,
        page_end=1,
        embedding=v1,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    c2 = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=1,
        text=text2,
        character_count=len(text2),
        page_start=2,
        page_end=2,
        embedding=v2,
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    test_db.add_all([c1, c2])
    test_db.commit()

    # Rebuild BM25 index via API
    rebuild_res = client.post("/api/v1/retrieval/rebuild-index")
    assert rebuild_res.status_code == 200
    assert rebuild_res.json()["status"] == "completed"
    assert rebuild_res.json()["indexed_chunks"] >= 2

    # Execute search query
    search_payload = {
        "query": "vector similarity search with pgvector",
        "top_k": 2,
        "enable_reranking": True,
    }
    search_res = client.post("/api/v1/retrieval/search", json=search_payload)
    assert search_res.status_code == 200
    data = search_res.json()

    assert data["query"] == search_payload["query"]
    assert len(data["results"]) >= 1

    # Chunk 2 should be ranked top
    top_result = data["results"][0]
    assert top_result["chunk_id"] == str(c2.id)
    assert top_result["filename"] == "search_test_doc.pdf"
    assert top_result["page_start"] == 2
    assert top_result["final_rank"] == 1
    assert top_result["rrf_score"] > 0.0
    assert top_result["reranker_score"] is not None

    # Verify telemetry metadata
    meta = data["retrieval_metadata"]
    assert meta["dense_candidates"] >= 1
    assert meta["bm25_candidates"] >= 1
    assert meta["fused_candidates"] >= 1
    assert meta["reranking_enabled"] is True
    assert meta["total_latency_ms"] > 0.0


def test_retrieval_search_with_document_filter(client, test_db: Session):
    """Verify search respects document_ids filtering."""
    doc_target = Document(
        id=uuid.uuid4(),
        filename="target.txt",
        original_filename="target.txt",
        file_type="txt",
        storage_path="mock/target",
        file_size=500,
        status="completed",
        embedding_status="completed",
    )
    doc_other = Document(
        id=uuid.uuid4(),
        filename="other.txt",
        original_filename="other.txt",
        file_type="txt",
        storage_path="mock/other",
        file_size=500,
        status="completed",
        embedding_status="completed",
    )
    test_db.add_all([doc_target, doc_other])
    test_db.flush()

    t_text = "Target document discussing quantum teleportation protocols."
    o_text = "Other document discussing classical networking protocols."

    t_vec = embedding_service.embed_text(t_text)
    o_vec = embedding_service.embed_text(o_text)

    c_t = Chunk(id=uuid.uuid4(), document_id=doc_target.id, chunk_index=0, text=t_text, character_count=len(t_text), embedding=t_vec)
    c_o = Chunk(id=uuid.uuid4(), document_id=doc_other.id, chunk_index=0, text=o_text, character_count=len(o_text), embedding=o_vec)
    test_db.add_all([c_t, c_o])
    test_db.commit()

    client.post("/api/v1/retrieval/rebuild-index")

    # Filter to doc_target only
    res = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "networking protocols",
            "document_ids": [str(doc_target.id)],
            "top_k": 5,
        },
    )
    assert res.status_code == 200
    results = res.json()["results"]
    for r in results:
        assert r["document_id"] == str(doc_target.id)
