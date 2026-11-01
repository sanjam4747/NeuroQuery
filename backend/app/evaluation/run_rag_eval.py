"""Script to execute RAG evaluation benchmark across the annotated benchmark dataset."""

import json
import time
from typing import Dict, List, Set
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.document import Document
from app.models.chunk import Chunk
from app.evaluation.dataset import BENCHMARK_DOCUMENTS, BENCHMARK_QUERIES, EvalQuery
from app.evaluation.rag_eval import RAGEvaluator, RAGBenchmarkSummary, RAGQueryEvalResult
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag.orchestrator import RAGOrchestrator
from app.services.rag.context_builder import context_builder
from app.schemas.retrieval import RetrievalQueryRequest
from app.services.bm25_retrieval import bm25_service
from app.services.retrieval import retrieval_service


def setup_in_memory_benchmark_db():
    """Sets up an in-memory SQLite database populated with benchmark documents and chunks."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for doc in BENCHMARK_DOCUMENTS:
        db_doc = Document(
            id=doc.id,
            filename=doc.filename,
            original_filename=doc.filename,
            file_type="txt",
            file_size=sum(len(c.text) for c in doc.chunks),
            storage_path=f"/fake/{doc.filename}",
            status="completed",
            total_pages=max(c.page_end for c in doc.chunks),
            total_characters=sum(len(c.text) for c in doc.chunks),
            total_chunks=len(doc.chunks),
            embedding_status="completed",
        )
        session.add(db_doc)

        for chunk in doc.chunks:
            db_chunk = Chunk(
                id=chunk.id,
                document_id=doc.id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                character_count=len(chunk.text),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                metadata_json={
                    "paragraph_start": chunk.chunk_index + 1,
                    "paragraph_end": chunk.chunk_index + 1,
                },
            )
            session.add(db_chunk)

    session.commit()
    return session


def run_benchmark():
    """Executes the full RAG benchmark and prints evaluation metrics."""
    session = setup_in_memory_benchmark_db()

    # Rebuild BM25 index with benchmark chunks
    bm25_service.build_index(session)

    evaluator = RAGEvaluator()
    results: List[RAGQueryEvalResult] = []

    print(f"Executing RAG benchmark on {len(BENCHMARK_QUERIES)} queries...")

    for query in BENCHMARK_QUERIES:
        q_start = time.perf_counter()

        # Retrieve hybrid candidates
        retrieval_res = retrieval_service.search(
            db=session,
            request=RetrievalQueryRequest(
                query=query.query_text,
                top_k=5,
                enable_reranking=True,
            ),
        )

        candidates = retrieval_res.results
        # Context building
        context_res = context_builder.build_context(
            retrieved_chunks=candidates,
        )

        is_positive = len(query.relevant_chunk_ids) > 0

        # Simulate LLM response grounded in selected evidence
        if context_res.sources:
            # Create grounded claim citing primary source [S1]
            cited_ids = ["S1"]
            llm_json = json.dumps({
                "answer": f"Based on the supplied evidence, {context_res.sources[0].text[:120]}... [S1]",
                "is_answered": True,
                "claims": [
                    {
                        "claim_text": context_res.sources[0].text[:120],
                        "citation_ids": cited_ids,
                    }
                ],
                "insufficient_evidence_reason": None,
            })
        else:
            llm_json = json.dumps({
                "answer": "The supplied document collection contains insufficient information to answer this question.",
                "is_answered": False,
                "claims": [],
                "insufficient_evidence_reason": "No relevant document passages were retrieved for this query.",
            })

        q_elapsed_ms = (time.perf_counter() - q_start) * 1000

        eval_res = evaluator.evaluate_query(
            query=query,
            mock_context_sources=context_res.sources,
            mock_llm_response_json=llm_json,
            latency_ms=q_elapsed_ms,
        )
        results.append(eval_res)

    summary = evaluator.aggregate_results(results, provider_name="mock (synthetic harness)")

    print("\n" + "=" * 70)
    print("           NEUROQUERY PHASE 5 RAG BENCHMARK REPORT")
    print("  EVALUATION MODE: Mock Provider (Synthetic Grounding Harness)")
    print("  * NOTICE: Latencies measure hybrid retrieval, reranking, and mock")
    print("    JSON parsing only. NOT real-world LLM inference latency.")
    print("  * NOTICE: Citation precision/recall measure ground-truth chunk")
    print("    identifier overlap. Semantic entailment is NOT evaluated.")
    print("=" * 70)
    print(f"Total Evaluated Queries:           {summary.total_queries}")
    print(f"  - Positive Grounded Queries:     {summary.positive_queries_count}")
    print(f"  - Negative Ungrounded Queries:   {summary.negative_queries_count}")
    print(f"Evaluation Provider:               {summary.provider_name}")
    print("-" * 70)
    print("ANSWERABILITY & ABSTENTION:")
    print(f"  - Correct Abstention Rate (CAR): {summary.correct_abstention_rate * 100:.2f}%")
    print(f"  - False Positive Rate:           {summary.false_positive_rate_ungrounded * 100:.2f}%")
    print(f"  - Harmful Abstention Rate:       {summary.harmful_abstention_rate_positive * 100:.2f}%")
    val_rate_str = (
        f"{summary.citation_identifier_validity_rate * 100:.2f}%"
        if summary.citation_identifier_validity_rate is not None
        else "N/A (0 citations attempted)"
    )
    cov_str = (
        f"{summary.mean_citation_coverage * 100:.2f}%"
        if summary.mean_citation_coverage is not None
        else "N/A"
    )

    print("-" * 70)
    print("CITATION INTEGRITY:")
    print(f"  - Total Attempted Citations:     {summary.total_attempted_citations}")
    print(f"  - Total Valid Citations:         {summary.total_valid_citations}")
    print(f"  - Total Invalid Citations:       {summary.total_invalid_citations}")
    print(f"  - Identifier Validity Rate:      {val_rate_str}")
    print(f"  - Mean Claim Citation Coverage:  {cov_str}")
    print(f"  - Citation Precision:            {summary.citation_precision * 100:.2f}%")
    print(f"  - Citation Recall:               {summary.citation_recall * 100:.2f}%")
    print("-" * 70)
    print("OPERATIONAL QUALITY (Retrieval + Mock Generation):")
    print(f"  - Mean Query Pipeline Latency:   {summary.mean_total_latency_ms:.2f} ms")
    print(f"  - Mean Context Tokens:           {summary.mean_context_tokens:.1f} tokens")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
