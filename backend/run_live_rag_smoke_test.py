"""
NeuroQuery Phase 5 - Reproducible Live RAG Smoke Test Script
Executes an end-to-end test using a real multi-page PDF document,
live local embedding generation (bge-small-en-v1.5), and live local
LLM inference via Ollama (qwen2.5:1.5b-instruct).
"""

import asyncio
import os
import sys
import time
from pathlib import Path
import pymupdf

# Set paths
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.db.base import Base
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.rag import RAGQueryRequest
from app.services.bm25_retrieval import bm25_service
from app.services.chunker import DocumentChunker
from app.services.embeddings import embedding_service
from app.services.extractors import PdfExtractor
from app.services.llm.ollama_provider import OllamaProvider
from app.services.rag.orchestrator import RAGOrchestrator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def generate_real_pdf_file(filepath: Path) -> Path:
    """Creates a real 3-page factual PDF document on disk."""
    doc = pymupdf.open()

    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "NeuroQuery Architecture Report\n\n"
        "Chapter 1: Dense Retrieval Architecture\n"
        "NeuroQuery integrates dense semantic search powered by the BAAI/bge-small-en-v1.5 model.\n"
        "This model generates 384-dimensional dense vectors with L2 normalization for exact cosine similarity.\n"
        "Chunks are mapped with token-aware bounding windows."
    )

    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "Chapter 2: Hybrid Reranking and Context Assembly\n"
        "Sparse retrieval uses Lucene BM25 with deterministic technical tokenization.\n"
        "Dense and sparse rankings are fused using Reciprocal Rank Fusion with a constant k=60.\n"
        "Reranking is conducted by the cross-encoder model ms-marco-MiniLM-L-6-v2.\n"
        "The context assembly allocates a strict maximum ceiling of 2048 tokens for evidence chunks."
    )

    p3 = doc.new_page()
    p3.insert_text(
        (50, 72),
        "Chapter 3: Verification and Grounding Guarantee\n"
        "The CitationValidator enforces strict JSON schemas on generation outputs.\n"
        "Any claim asserting facts must cite an available source tag like S1 or S2.\n"
        "Unsupported claims or hallucinated tags are pruned or trigger validation errors.\n"
        "Unanswerable queries result in honest, deterministic abstention."
    )

    doc.save(str(filepath))
    doc.close()
    return filepath


async def run_smoke_test():
    print("=" * 78)
    print("      NEUROQUERY PHASE 5: LIVE RAG SMOKE TEST (REAL PDF + OLLAMA)")
    print("=" * 78)

    # 1. Create real test PDF
    artifacts_dir = backend_dir / "storage" / "smoke_test_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = artifacts_dir / "neuroquery_architecture_spec.pdf"
    generate_real_pdf_file(pdf_path)
    print(f"\n[STEP 1] Created Real Test PDF: {pdf_path.name} (3 pages, {pdf_path.stat().st_size} bytes)")

    # 2. Extract text with page provenance
    extractor = PdfExtractor()
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    extracted_doc = extractor.extract(pdf_bytes)
    print(f"[STEP 2] Extracted {extracted_doc.total_pages} pages ({len(extracted_doc.units)} extractable units)")
    for u in extracted_doc.units:
        print(f"         * Page {u.page_number}: {len(u.text.split())} words, sample: '{u.text[:60]}...'")

    # 3. Chunking
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    chunks_data = chunker.chunk_document(extracted_doc)
    print(f"[STEP 3] Chunked into {len(chunks_data)} chunks with page tracking.")
    for c in chunks_data:
        print(f"         * Chunk {c.chunk_index}: pp. {c.page_start}-{c.page_end}, chars: {c.character_count}")

    # 4. Generate Embeddings using local Sentence-Transformers
    print(f"[STEP 4] Generating embeddings locally via BAAI/bge-small-en-v1.5...")
    embed_service = embedding_service
    texts = [c.text for c in chunks_data]
    embeddings, _ = embed_service.embed_batch(texts)
    print(f"         * Generated {len(embeddings)} vectors, dimension: {len(embeddings[0])}")

    # 5. Populate in-memory SQLite database for test session
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    Session = sessionmaker(bind=test_engine)
    db = Session()

    import uuid
    doc_id = uuid.uuid4()
    doc_record = Document(
        id=doc_id,
        filename=pdf_path.name,
        original_filename=pdf_path.name,
        file_type="pdf",
        file_size=len(pdf_bytes),
        storage_path=str(pdf_path),
        status="completed",
        total_pages=extracted_doc.total_pages,
        total_characters=sum(c.character_count for c in chunks_data),
        total_chunks=len(chunks_data),
        embedding_status="completed",
        total_embedded_chunks=len(chunks_data),
    )
    db.add(doc_record)
    db.flush()

    chunk_objects = []
    for c, emb in zip(chunks_data, embeddings):
        chunk_obj = Chunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            chunk_index=c.chunk_index,
            text=c.text,
            character_count=c.character_count,
            page_start=c.page_start,
            page_end=c.page_end,
            metadata_json=c.metadata,
            embedding=emb,
            embedding_model="BAAI/bge-small-en-v1.5",
        )
        db.add(chunk_obj)
        chunk_objects.append(chunk_obj)
    db.commit()

    # Register into BM25 index
    bm25_service.add_document_chunks(chunk_objects, filename=pdf_path.name)

    # 6. Initialize Live Ollama Orchestrator
    ollama_provider = OllamaProvider(
        base_url="http://127.0.0.1:11434",
        model_name="qwen2.5:1.5b-instruct",
        timeout_seconds=60.0,
    )
    orchestrator = RAGOrchestrator(
        llm_provider=ollama_provider,
    )

    # 7. Query 1: Positive Grounded Question
    q1 = "What cross-encoder model is used for reranking, and what is the RRF constant?"
    print("\n" + "-" * 78)
    print(f"[TEST 1] Grounded Question: \"{q1}\"")
    print("-" * 78)

    t0 = time.perf_counter()
    req1 = RAGQueryRequest(question=q1)
    res1 = await orchestrator.answer_question(db=db, request=req1)
    dur1 = (time.perf_counter() - t0) * 1000

    print(f"Status: answered={res1.is_answered}, parsing={res1.parsing_status}, total_time={dur1:.1f}ms")
    print(f"\nGenerated Answer:\n{res1.answer}\n")
    print("Diagnostics:")
    print(f"  - Inference Latency: {res1.diagnostics.inference_latency_ms:.1f} ms")
    print(f"  - Context Tokens:    {res1.diagnostics.context_tokens_used} tokens")
    print(f"  - Context Sources:   {len(res1.evidence_sources)} sources assembled")
    print(f"  - Valid Citations:   {len(res1.citations)}")

    for cit in res1.citations:
        print(f"    -> [{cit.source_id}] {cit.filename} (Page {cit.page_start}), Chunk: {cit.chunk_id}")
        print(f"       Quote/Excerpt: \"{cit.text_snippet[:100]}...\"")

    print("\nAnswer Claims Breakdown:")
    for i, claim in enumerate(res1.claims):
        print(f"  [{i+1}] \"{claim.claim_text}\"")
        print(f"      Citations: {claim.citation_ids} | Supported: {claim.is_supported}")

    # Verifications for Q1
    assert res1.is_answered is True, "Query 1 must be answered!"
    assert any("ms-marco" in res1.answer.lower() or "ms-marco" in c.claim_text.lower() for c in res1.claims), "Answer must cite ms-marco"
    assert any("60" in res1.answer or "60" in c.claim_text for c in res1.claims), "Answer must mention k=60"
    assert len(res1.citations) > 0, "Query 1 must have at least one citation"
    assert any((c.page_start or 1) <= 2 <= (c.page_end or 2) for c in res1.citations), "Citations must point to Page 2 evidence"
    print(">>> [PASS] Grounded query correctly answered, claims backed by valid source citations from Page 2!")

    # 8. Query 2: Ungrounded / Unrelated Question
    q2 = "What is the average flight speed of an unladen European swallow?"
    print("\n" + "-" * 78)
    print(f"[TEST 2] Ungrounded Question: \"{q2}\"")
    print("-" * 78)

    t0 = time.perf_counter()
    req2 = RAGQueryRequest(question=q2)
    res2 = await orchestrator.answer_question(db=db, request=req2)
    dur2 = (time.perf_counter() - t0) * 1000

    print(f"Status: answered={res2.is_answered}, parsing={res2.parsing_status}, total_time={dur2:.1f}ms")
    print(f"\nGenerated Answer / Refusal:\n{res2.answer}\n")
    print("Diagnostics:")
    print(f"  - Insufficient Evidence Reason: {res2.insufficient_evidence_reason}")
    print(f"  - Citations Count:              {len(res2.citations)}")
    print(f"  - Claims Count:                 {len(res2.claims)}")
    print(f"  - Inference Latency:            {res2.diagnostics.inference_latency_ms:.1f} ms")

    # Verifications for Q2
    assert res2.is_answered is False, "Query 2 must abstain!"
    assert len(res2.citations) == 0, "Query 2 must not have citations on abstention"
    assert res2.insufficient_evidence_reason is not None, "Query 2 must provide insufficient evidence reason"
    print(">>> [PASS] System appropriately abstained on ungrounded question!")

    print("\n" + "=" * 78)
    print("      ALL LIVE RAG SMOKE TEST VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
