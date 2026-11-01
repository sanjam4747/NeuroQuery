"""Tests for semantic embedding generation, vector storage, and re-embedding."""

import math
import uuid
import pytest
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.chunk import Chunk
from app.services.embeddings import EmbeddingService, embedding_service
from app.services.pipeline import pipeline_service
from tests.conftest import make_txt


def test_embedding_service_singleton_and_properties():
    """Verify service properties and default English-optimized configuration."""
    assert embedding_service.model_name == "BAAI/bge-small-en-v1.5"
    assert embedding_service.dimension == 384
    assert embedding_service.max_sequence_length == 512

    info = embedding_service.get_info()
    assert info["dimension"] == 384
    assert info["max_sequence_length"] == 512
    assert info["metric"] == "cosine"
    assert info["normalize"] is True
    assert "English" in info["description"]


def test_embedding_dimensionality_and_normalization():
    """Verify generated embedding has exactly 384 dimensions and unit L2 norm."""
    text = "NeuroQuery is an NLP-powered document research platform."
    vec = embedding_service.embed_text(text)

    assert isinstance(vec, list)
    assert len(vec) == 384
    # L2 norm for cosine similarity should be approximately 1.0
    l2_norm = math.sqrt(sum(x * x for x in vec))
    assert abs(l2_norm - 1.0) < 1e-4


def test_embedding_batch_generation():
    """Verify batched embedding generates matching vectors and throughput metrics."""
    texts = [
        "First chunk discussing semantic retrieval.",
        "Second chunk covering vector indexing.",
        "Third chunk describing cosine similarity.",
    ]
    vectors, metrics = embedding_service.embed_batch(texts, batch_size=2)

    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 384
        norm = np.linalg.norm(v)
        assert abs(norm - 1.0) < 1e-4

    assert metrics["total_chunks"] == 3
    assert metrics["dimension"] == 384
    assert metrics["duration_ms"] > 0
    assert metrics["throughput_chunks_per_sec"] > 0
    assert len(metrics["token_counts"]) == 3


def test_embedding_empty_and_invalid_inputs():
    """Verify empty texts and invalid inputs are rejected or handled gracefully."""
    # Empty string should raise ValueError
    with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
        embedding_service.embed_text("   ")

    # Empty batch returns empty lists
    vecs, metrics = embedding_service.embed_batch([])
    assert vecs == []
    assert metrics["total_chunks"] == 0


def test_embedding_oversized_chunk_detection(caplog):
    """Verify chunks exceeding model's 512 max_sequence_length are explicitly flagged."""
    # Create an artificial text with 600 single-character words (> 512 tokens)
    dense_text = " ".join([f"w{i}" for i in range(600)])
    token_count = embedding_service.count_tokens(dense_text)
    assert token_count > 512

    vecs, metrics = embedding_service.embed_batch([dense_text])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
    assert 0 in metrics["oversized_chunks"]
    assert metrics["token_counts"][0] == token_count


def test_pipeline_ingest_persists_embeddings(test_db: Session, tmp_path):
    """Integration test verifying end-to-end ingestion generates and persists embeddings in DB."""
    from app.services.storage import storage_service
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path

    try:
        content = "Natural Language Processing enables document intelligence. Dense vectors capture deep semantics."
        txt_bytes = make_txt(content)

        doc = pipeline_service.ingest_document(test_db, txt_bytes, "vectors_test.txt")

        assert doc.status == "completed"
        assert doc.embedding_status == "completed"
        assert doc.embedding_model == "BAAI/bge-small-en-v1.5"
        assert doc.total_embedded_chunks == doc.total_chunks
        assert doc.total_embedded_chunks > 0

        # Verify DB Chunk records have 384d embedding
        chunks = test_db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()
        assert len(chunks) == doc.total_chunks
        for chunk in chunks:
            assert chunk.has_embedding is True
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384
            assert chunk.embedding_model == "BAAI/bge-small-en-v1.5"
            assert chunk.embedded_at is not None
            assert "token_count" in chunk.metadata_json
    finally:
        storage_service.base_dir = orig_base


def test_pipeline_reembedding_without_duplicate_chunks(test_db: Session, tmp_path):
    """Verify re-embedding updates existing chunks in-place without duplicating records."""
    from app.services.storage import storage_service
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path

    try:
        content = "Re-embedding test passage. Chunks should be updated in place."
        txt_bytes = make_txt(content)
        doc = pipeline_service.ingest_document(test_db, txt_bytes, "reembed_test.txt")

        initial_chunk_count = doc.total_chunks
        initial_chunk_ids = [c.id for c in test_db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()]

        # Trigger re-embedding
        updated_doc = pipeline_service.reembed_document(test_db, doc.id)

        assert updated_doc.embedding_status == "completed"
        assert updated_doc.total_embedded_chunks == initial_chunk_count

        # Verify no duplicate chunk records were created
        current_chunks = test_db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()
        assert len(current_chunks) == initial_chunk_count
        assert [c.id for c in current_chunks] == initial_chunk_ids
        for c in current_chunks:
            assert c.embedding is not None
            assert len(c.embedding) == 384
    finally:
        storage_service.base_dir = orig_base


def test_pipeline_embedding_failure_handling(test_db: Session, monkeypatch, tmp_path):
    """Verify document transitions to failed state without orphaned chunks if embedding fails."""
    from app.services.storage import storage_service
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path

    def broken_embed_batch(*args, **kwargs):
        raise RuntimeError("Simulated GPU/Inference Out of Memory error")

    monkeypatch.setattr(embedding_service, "embed_batch", broken_embed_batch)

    try:
        content = "Content that will fail during embedding inference."
        txt_bytes = make_txt(content)
        doc = pipeline_service.ingest_document(test_db, txt_bytes, "fail_embed.txt")

        assert doc.status == "failed"
        assert doc.embedding_status == "failed"
        assert "Simulated GPU/Inference Out of Memory" in (doc.embedding_error or "")

        # Verify no chunks persisted
        chunks = test_db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()
        assert len(chunks) == 0
    finally:
        storage_service.base_dir = orig_base


def test_chunk_below_token_limit():
    """Verify chunk below token limit is processed as a single window without pooling."""
    text = "Machine learning models transform unstructured document text into rich semantic vectors."
    windows = embedding_service.split_into_token_windows(text)
    assert len(windows) == 1
    sub_text, tc, start, end = windows[0]
    assert start == 0
    assert end == tc
    assert tc < 512

    vecs, metrics = embedding_service.embed_batch([text])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
    assert math.isclose(np.linalg.norm(vecs[0]), 1.0, rel_tol=1e-4)
    assert metrics["sub_chunk_counts"][0] == 1
    assert 0 not in metrics["oversized_chunks"]


def test_chunk_exactly_at_token_limit():
    """Verify chunk exactly at the 510 content-token boundary fits in a single window."""
    model = embedding_service.get_model()
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        pytest.skip("Tokenizer not available")

    # Generate text that tokenizes to exactly 510 tokens
    raw_tokens = [tok.convert_tokens_to_ids("test")] * 510
    exact_text = tok.decode(raw_tokens, skip_special_tokens=True)
    actual_tokens = len(tok.encode(exact_text, add_special_tokens=False))

    windows = embedding_service.split_into_token_windows(exact_text)
    assert len(windows) == 1
    assert windows[0][1] <= 510
    assert windows[0][2] == 0
    assert windows[0][3] == len(windows[0][0].split()) or windows[0][3] == actual_tokens

    vecs, metrics = embedding_service.embed_batch([exact_text])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
    assert math.isclose(np.linalg.norm(vecs[0]), 1.0, rel_tol=1e-4)
    assert metrics["sub_chunk_counts"][0] == 1
    assert 0 not in metrics["oversized_chunks"]


def test_chunk_exceeding_token_limit():
    """Verify chunk exceeding 512 tokens is split into sub-windows and token-weighted pooled."""
    model = embedding_service.get_model()
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        pytest.skip("Tokenizer not available")

    sentence = "Artificial intelligence and neural search revolutionize document question answering. "
    exceeding_text = sentence * 65  # ~715 tokens (> 512)
    total_tokens = len(tok.encode(exceeding_text, add_special_tokens=False))
    assert total_tokens > 512

    windows = embedding_service.split_into_token_windows(exceeding_text)
    assert len(windows) == 2
    assert windows[0][2] == 0
    assert windows[-1][3] == total_tokens
    # Verify overlap
    assert windows[1][2] < windows[0][3]

    vecs, metrics = embedding_service.embed_batch([exceeding_text])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
    assert math.isclose(np.linalg.norm(vecs[0]), 1.0, rel_tol=1e-4)
    assert 0 in metrics["oversized_chunks"]
    assert metrics["sub_chunk_counts"][0] == 2
    assert metrics["chunk_details"][0]["is_oversized"] is True
    assert len(metrics["chunk_details"][0]["sub_chunks"]) == 2


def test_very_large_chunk_multiple_embedding_inputs():
    """Verify a very large chunk (~1500 tokens) requires >= 3 embedding inputs and pools correctly."""
    model = embedding_service.get_model()
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        pytest.skip("Tokenizer not available")

    passage = "Advanced dense retrieval blends semantic vector similarity with sparse lexical matching. "
    large_text = passage * 110  # ~1430 tokens
    total_tokens = len(tok.encode(large_text, add_special_tokens=False))
    assert total_tokens > 1200

    windows = embedding_service.split_into_token_windows(large_text)
    assert len(windows) >= 3

    vecs, metrics = embedding_service.embed_batch([large_text])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
    assert math.isclose(np.linalg.norm(vecs[0]), 1.0, rel_tol=1e-4)
    assert 0 in metrics["oversized_chunks"]
    assert metrics["sub_chunk_counts"][0] >= 3
    assert metrics["total_sub_windows"] >= 3


def test_no_content_silently_omitted_across_windows():
    """Verify that 100% of tokens from index 0 to N are covered without gaps or omission."""
    model = embedding_service.get_model()
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        pytest.skip("Tokenizer not available")

    passage = "Token continuity test verifying that every index from beginning to end is represented. "
    text = passage * 60
    token_ids = tok.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    assert total_tokens > 512

    windows = embedding_service.split_into_token_windows(text)
    assert len(windows) > 1

    # Check start and end boundaries
    assert windows[0][2] == 0, "First window must start at token 0"
    assert windows[-1][3] == total_tokens, f"Final window must end at token {total_tokens}"

    # Check contiguous coverage: no gaps between consecutive windows
    covered_indices = set()
    for w_idx, (_, count, start, end) in enumerate(windows):
        assert end > start, "Window must contain positive number of tokens"
        assert count == end - start, "Token count must match boundary difference"
        covered_indices.update(range(start, end))

    # Assert every single token index is covered
    all_indices = set(range(total_tokens))
    missing = all_indices - covered_indices
    assert len(missing) == 0, f"Tokens were silently omitted: {missing}"


def test_pipeline_oversized_chunk_status_and_metadata(test_db: Session, tmp_path):
    """Verify end-to-end pipeline handles oversized chunks with lossless sub-chunk metadata."""
    from app.services.storage import storage_service
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path

    try:
        model = embedding_service.get_model()
        tok = getattr(model, "tokenizer", None)
        if tok is None:
            pytest.skip("Tokenizer not available")

        # 1500 chars of dense tokens produces ~890 tokens in a single chunk
        normal_content = "This is a normal paragraph with standard length."
        dense_content = " ".join([f"x{i}" for i in range(400)])[:1500]
        full_doc = f"{normal_content}\n\n{dense_content}"
        txt_bytes = make_txt(full_doc)

        doc = pipeline_service.ingest_document(test_db, txt_bytes, "oversized_pipeline_test.txt")

        assert doc.status == "completed"
        assert doc.embedding_status == "completed"
        assert doc.total_chunks >= 2

        chunks = test_db.scalars(
            select(Chunk)
            .where(Chunk.document_id == doc.id)
            .order_by(Chunk.chunk_index)
        ).all()

        # Find the oversized chunk
        oversized = [c for c in chunks if c.metadata_json.get("oversized_chunk") is True]
        assert len(oversized) >= 1
        oc = oversized[0]

        # Verify lossless metadata
        assert oc.metadata_json["embedding_method"] == "token_weighted_subchunk_pooling"
        assert oc.metadata_json["sub_chunk_count"] >= 2
        assert len(oc.metadata_json["sub_chunks"]) >= 2
        assert "zero content discarded" in oc.metadata_json["oversized_info"]
        assert len(oc.embedding) == 384
        assert math.isclose(np.linalg.norm(oc.embedding), 1.0, rel_tol=1e-4)

        # Verify re-embedding on this document preserves sub-chunk metadata and does not duplicate chunks
        initial_chunk_ids = [c.id for c in chunks]
        reembed_doc = pipeline_service.reembed_document(test_db, doc.id)
        assert reembed_doc.embedding_status == "completed"

        reembed_chunks = test_db.scalars(
            select(Chunk)
            .where(Chunk.document_id == doc.id)
            .order_by(Chunk.chunk_index)
        ).all()
        assert [c.id for c in reembed_chunks] == initial_chunk_ids
        oc_reembedded = [c for c in reembed_chunks if c.metadata_json.get("oversized_chunk") is True][0]
        assert oc_reembedded.metadata_json["sub_chunk_count"] >= 2
    finally:
        storage_service.base_dir = orig_base


def test_api_embeddings_info(client):
    """Verify GET /api/v1/embeddings/info returns active model metadata."""
    res = client.get("/api/v1/embeddings/info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "BAAI/bge-small-en-v1.5"
    assert data["dimension"] == 384
    assert data["max_sequence_length"] == 512
    assert data["metric"] == "cosine"
    assert data["normalize"] is True


def test_api_reembed_endpoint(client, test_db: Session, tmp_path):
    """Verify POST /api/v1/documents/{id}/reembed triggers re-embedding via HTTP."""
    from app.services.storage import storage_service
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path

    try:
        txt_bytes = make_txt("Testing HTTP re-embedding endpoint functionality.")
        doc = pipeline_service.ingest_document(test_db, txt_bytes, "api_reembed.txt")

        res = client.post(f"/api/v1/documents/{doc.id}/reembed")
        assert res.status_code == 200
        data = res.json()
        assert data["document_id"] == str(doc.id)
        assert data["status"] == "completed"
        assert data["total_embedded_chunks"] == doc.total_chunks
        assert "BAAI/bge-small-en-v1.5" in data["embedding_model"]

        # Check document embedding status endpoint
        status_res = client.get(f"/api/v1/documents/{doc.id}/embeddings")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["embedding_status"] == "completed"
        assert status_data["total_embedded_chunks"] == doc.total_chunks
    finally:
        storage_service.base_dir = orig_base


def test_direct_model_input_tokens_bounded_and_special_tokens():
    """
    Verify that token-window model inputs:
    1. Are created with exact model tokenizer special tokens [CLS] (101) and [SEP] (102).
    2. Strictly respect the model's 512-token ceiling (bounded <= 452 tokens).
    3. Pass token IDs directly to the model without decode/re-tokenize transformations.
    4. Produce unit-norm 384d vectors directly from token IDs.
    """
    model = embedding_service.get_model()
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        pytest.skip("Tokenizer not available")

    cls_id = getattr(tok, "cls_token_id", 101) or 101
    sep_id = getattr(tok, "sep_token_id", 102) or 102

    # Dense passage of ~800 tokens to force multiple windows
    sentence = "Comprehensive validation of model token inputs ensuring sequence length constraints. "
    text = sentence * 70

    windows = embedding_service.tokenize_into_windows(text, window_size=450, overlap_size=50)
    assert len(windows) >= 2, "Must segment into multiple sub-windows"

    for w_idx, win in enumerate(windows):
        input_ids = win["input_ids"]
        token_count = win["token_count"]

        # 1. Sequence length <= 512
        assert len(input_ids) <= 512, f"Window {w_idx} sequence length {len(input_ids)} exceeds 512 limit"
        assert len(input_ids) == token_count + 2, f"Window {w_idx} input_ids must equal content tokens + 2 special tokens"
        assert len(input_ids) <= 452, f"Window {w_idx} should be bounded by window_size + 2 (452)"

        # 2. Special tokens at exact boundaries
        assert input_ids[0] == cls_id, f"Window {w_idx} must start with [CLS] token {cls_id}"
        assert input_ids[-1] == sep_id, f"Window {w_idx} must end with [SEP] token {sep_id}"

        # 3. Content tokens match exactly without special tokens
        assert input_ids[1:-1] == win["token_ids"]

    # 4. Direct embedding verification through _embed_token_batches
    raw_vecs = embedding_service._embed_token_batches([w["input_ids"] for w in windows], batch_size=32)
    assert raw_vecs.shape == (len(windows), 384)
    for v in raw_vecs:
        assert math.isclose(np.linalg.norm(v), 1.0, rel_tol=1e-4)


