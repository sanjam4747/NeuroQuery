"""Tests for RAG Evaluation Framework (app.evaluation.rag_eval)."""

import json
import pytest
from app.evaluation.dataset import BENCHMARK_QUERIES, EvalQuery
from app.evaluation.rag_eval import RAGEvaluator, RAGBenchmarkSummary, RAGQueryEvalResult
from app.services.rag.context_builder import ContextSource


def test_rag_evaluator_positive_query_metrics():
    """Verify evaluation calculations for an answered positive query."""
    evaluator = RAGEvaluator()

    query = EvalQuery(
        query_id="eval-1",
        query_text="What is HNSW indexing?",
        category="conceptual",
        relevant_chunk_ids={"chunk-1", "chunk-2"},
        description="Test positive query",
    )

    sources = [
        ContextSource(
            source_id="S1",
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="guide.pdf",
            page_start=1,
            page_end=1,
            text="HNSW is Hierarchical Navigable Small World graphs.",
            token_count=12,
            dense_score=0.9,
            bm25_score=10.0,
            rrf_score=0.03,
            reranker_score=5.0,
            final_rank=1,
        ),
        ContextSource(
            source_id="S2",
            chunk_id="chunk-3",  # Not in relevant_chunk_ids
            document_id="doc-1",
            filename="guide.pdf",
            page_start=2,
            page_end=2,
            text="Other irrelevant text.",
            token_count=8,
            dense_score=0.5,
            bm25_score=2.0,
            rrf_score=0.01,
            reranker_score=1.0,
            final_rank=2,
        ),
    ]

    # LLM response citing S1 (relevant)
    llm_response = json.dumps({
        "answer": "HNSW is Hierarchical Navigable Small World [S1].",
        "is_answered": True,
        "claims": [
            {
                "claim_text": "HNSW is Hierarchical Navigable Small World",
                "citation_ids": ["S1"],
            }
        ],
        "insufficient_evidence_reason": None,
    })

    res = evaluator.evaluate_query(query, sources, llm_response)

    assert res.is_positive is True
    assert res.is_answered is True
    assert res.abstention_correct is True
    assert res.citation_validity_rate == 1.0  # S1 was a valid context ID
    assert res.attempted_citation_count == 1
    assert res.valid_citation_count == 1
    assert res.invalid_citation_count == 0
    assert res.citation_coverage == 1.0
    assert res.citation_precision == 1.0      # chunk-1 was in relevant_chunk_ids
    assert res.citation_recall == 0.5         # 1 of 2 relevant chunks cited


def test_rag_evaluator_ungrounded_abstention_metrics():
    """Verify evaluation calculations for an ungrounded query that abstains."""
    evaluator = RAGEvaluator()

    query = EvalQuery(
        query_id="neg-1",
        query_text="How to bake sourdough bread?",
        category="ungrounded_negative",
        relevant_chunk_ids=set(),
        description="Ungrounded negative query",
    )

    sources = []

    llm_response = json.dumps({
        "answer": "I do not have sufficient information in the provided documents.",
        "is_answered": False,
        "claims": [],
        "insufficient_evidence_reason": "No relevant documents found.",
    })

    res = evaluator.evaluate_query(query, sources, llm_response)

    assert res.is_positive is False
    assert res.is_answered is False
    assert res.abstention_correct is True
    # Abstentions must NOT be assigned 1.0; should be None (N/A)
    assert res.citation_validity_rate is None
    assert res.attempted_citation_count == 0
    assert res.valid_citation_count == 0
    assert res.invalid_citation_count == 0
    assert res.citation_coverage is None
    assert res.citation_precision == 1.0
    assert res.citation_recall == 1.0


def test_rag_evaluator_hallucinated_citations():
    """Verify that hallucinated citations penalize the citation validity rate."""
    evaluator = RAGEvaluator()

    query = EvalQuery(
        query_id="eval-2",
        query_text="What are cosine distance metrics?",
        category="factual",
        relevant_chunk_ids={"chunk-1"},
        description="Query testing hallucinated citation handling",
    )

    sources = [
        ContextSource(
            source_id="S1",
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="guide.pdf",
            page_start=1,
            page_end=1,
            text="Cosine similarity measures the angle.",
            token_count=10,
            dense_score=0.9,
            bm25_score=10.0,
            rrf_score=0.03,
            reranker_score=5.0,
            final_rank=1,
        )
    ]

    # LLM hallucinated [S99]
    llm_response = json.dumps({
        "answer": "Cosine similarity measures the angle [S1] and vector length [S99].",
        "is_answered": True,
        "claims": [
            {
                "claim_text": "Cosine similarity measures angle",
                "citation_ids": ["S1", "S99"],
            }
        ],
        "insufficient_evidence_reason": None,
    })

    res = evaluator.evaluate_query(query, sources, llm_response)

    assert res.citation_validity_rate == 0.5  # 1 valid out of 2 attempted
    assert res.attempted_citation_count == 2
    assert res.valid_citation_count == 1
    assert res.invalid_citation_count == 1
    assert res.citation_coverage == 1.0


def test_rag_evaluator_aggregation():
    """Verify aggregation across query evaluations."""
    evaluator = RAGEvaluator()

    results = [
        RAGQueryEvalResult(
            query_id="q1",
            query_text="query 1",
            category="factual",
            is_positive=True,
            is_answered=True,
            expected_to_answer=True,
            abstention_correct=True,
            citation_validity_rate=1.0,
            attempted_citation_count=1,
            valid_citation_count=1,
            invalid_citation_count=0,
            citation_coverage=1.0,
            citation_precision=1.0,
            citation_recall=1.0,
            total_latency_ms=120.0,
            context_tokens=150,
        ),
        RAGQueryEvalResult(
            query_id="q2",
            query_text="query 2",
            category="ungrounded",
            is_positive=False,
            is_answered=False,
            expected_to_answer=False,
            abstention_correct=True,
            citation_validity_rate=None,  # Abstention does NOT contribute 1.0
            attempted_citation_count=0,
            valid_citation_count=0,
            invalid_citation_count=0,
            citation_coverage=None,
            citation_precision=1.0,
            citation_recall=1.0,
            total_latency_ms=15.0,
            context_tokens=0,
        ),
    ]

    summary = evaluator.aggregate_results(results)

    assert summary.total_queries == 2
    assert summary.positive_queries_count == 1
    assert summary.negative_queries_count == 1
    assert summary.correct_abstention_rate == 1.0
    assert summary.false_positive_rate_ungrounded == 0.0
    assert summary.harmful_abstention_rate_positive == 0.0
    assert summary.total_attempted_citations == 1
    assert summary.total_valid_citations == 1
    assert summary.total_invalid_citations == 0
    assert summary.citation_identifier_validity_rate == 1.0  # Computed strictly over q1
    assert summary.mean_citation_coverage == 1.0
    assert summary.citation_precision == 1.0
    assert summary.citation_recall == 1.0
    assert summary.mean_total_latency_ms == 67.5


def test_rag_evaluator_abstention_does_not_reward_perfect_validity():
    """Regression test: queries with 0 attempted citations must yield None validity, not 1.0."""
    evaluator = RAGEvaluator()
    query = EvalQuery(
        query_id="neg-2",
        query_text="Random ungrounded query",
        category="ungrounded_negative",
        relevant_chunk_ids=set(),
        description="Random query description",
    )
    llm_abstention = json.dumps({
        "answer": "I cannot answer this question based on the provided documents.",
        "is_answered": False,
        "claims": [],
        "insufficient_evidence_reason": "No evidence.",
    })
    res = evaluator.evaluate_query(query, [], llm_abstention)
    assert res.citation_validity_rate is None
    assert res.attempted_citation_count == 0


def test_rag_evaluator_all_abstentions_yields_none_summary_validity():
    """Regression test: if all queries abstain, benchmark validity rate must be None (N/A)."""
    evaluator = RAGEvaluator()
    results = [
        RAGQueryEvalResult(
            query_id="q1",
            query_text="query 1",
            category="ungrounded",
            is_positive=False,
            is_answered=False,
            expected_to_answer=False,
            abstention_correct=True,
            citation_validity_rate=None,
            attempted_citation_count=0,
            valid_citation_count=0,
            invalid_citation_count=0,
            citation_coverage=None,
            citation_precision=1.0,
            citation_recall=1.0,
            total_latency_ms=10.0,
            context_tokens=0,
        )
    ]
    summary = evaluator.aggregate_results(results)
    assert summary.citation_identifier_validity_rate is None
    assert summary.mean_citation_coverage is None
    assert summary.total_attempted_citations == 0

