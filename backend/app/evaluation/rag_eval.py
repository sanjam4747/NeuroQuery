"""RAG Evaluation Framework for measuring answerability, citation integrity, and operational quality."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set
import numpy as np

from app.evaluation.dataset import BENCHMARK_DOCUMENTS, BENCHMARK_QUERIES, EvalQuery
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag.citation_validator import CitationValidator
from app.services.rag.context_builder import ContextBuilder, context_builder
from app.services.rag.orchestrator import RAGOrchestrator


@dataclass
class RAGQueryEvalResult:
    """Evaluation metrics for a single RAG query execution."""

    query_id: str
    query_text: str
    category: str
    is_positive: bool
    is_answered: bool
    expected_to_answer: bool
    abstention_correct: bool
    citation_validity_rate: Optional[float]  # None (N/A) when 0 citations were attempted
    attempted_citation_count: int
    valid_citation_count: int
    invalid_citation_count: int
    citation_coverage: Optional[float]  # None (N/A) when no claims exist or query abstained
    citation_precision: float
    citation_recall: float
    total_latency_ms: float
    context_tokens: int


@dataclass
class RAGBenchmarkSummary:
    """
    Aggregated evaluation summary across the RAG benchmark suite.

    Mathematical Definitions & Metrics:
    1. Citation Identifier Validity Rate:
       |Valid Citations| / (|Valid Citations| + |Invalid/Hallucinated IDs|)
       Evaluated ONLY on queries where at least one citation was attempted.
       Reports None (N/A) if zero citations were attempted. Abstentions do NOT receive 100%.

    2. Citation Precision:
       |Cited Chunks ∩ Ground-Truth Relevant Chunks| / |Cited Chunks|
       Measures agreement between cited chunks and human-annotated relevant chunks.
       NOTE: Does not measure claim-level NLI/entailment.

    3. Citation Recall:
       |Cited Chunks ∩ Ground-Truth Relevant Chunks| / |Ground-Truth Relevant Chunks|
       Measures the proportion of relevant chunks cited.

    4. Correct Abstention Rate (CAR):
       |Negative Queries with is_answered=False| / |Total Negative Queries|
       Measures honest refusal when evidence is absent.

    5. False Positive Rate (FPR):
       1.0 - CAR (Negative queries where the model fabricated an answer).

    6. Harmful Abstention Rate (HAR):
       |Positive Queries with is_answered=False| / |Total Positive Queries|
       Measures unjustified refusal when sufficient evidence exists in corpus.

    7. Citation Coverage:
       |Claims citing >=1 source| / |Total Claims|
       Measures the proportion of answer claims that have citation backing.
    """

    total_queries: int
    positive_queries_count: int
    negative_queries_count: int
    # Abstention & Answerability
    correct_abstention_rate: float
    false_positive_rate_ungrounded: float
    harmful_abstention_rate_positive: float
    # Citation Quality
    total_attempted_citations: int
    total_valid_citations: int
    total_invalid_citations: int
    citation_identifier_validity_rate: Optional[float]  # None (N/A) when 0 attempted
    mean_citation_coverage: Optional[float]  # None (N/A) when no claims generated
    citation_precision: float
    citation_recall: float
    # Operational Quality
    mean_total_latency_ms: float
    mean_context_tokens: float
    provider_name: str = "mock"


class RAGEvaluator:
    """
    Evaluates RAG generation quality across benchmark queries:
    - Structural citation validity: checks for non-existent source tags.
    - Reference-based citation precision: verifies cited chunks match ground truth.
    - Abstention correctness on ungrounded queries vs positive queries.
    """

    def __init__(self, orchestrator: Optional[RAGOrchestrator] = None):
        self.orchestrator = orchestrator or RAGOrchestrator(
            llm_provider=MockLLMProvider(model_name="mock-rag-eval"),
        )

    def evaluate_query(
        self,
        query: EvalQuery,
        mock_context_sources: List[Any],
        mock_llm_response_json: str,
        latency_ms: Optional[float] = None,
    ) -> RAGQueryEvalResult:
        """
        Evaluates a single query run against ground truth annotations.
        """
        source_map = {src.source_id: src for src in mock_context_sources}
        val_res = CitationValidator.parse_and_validate(mock_llm_response_json, source_map)

        is_positive = len(query.relevant_chunk_ids) > 0
        expected_to_answer = is_positive

        # Abstention correctness
        if is_positive:
            abstention_correct = val_res.is_answered
        else:
            abstention_correct = not val_res.is_answered

        # Citation validity & count tracking
        valid_citations = len(val_res.citations)
        invalid_citations = len(val_res.invalid_citation_ids)
        attempted_citations = valid_citations + invalid_citations

        # DO NOT assign 1.0 when zero citations were attempted; return None (N/A)
        if attempted_citations > 0:
            cit_validity_rate = round(valid_citations / attempted_citations, 4)
        else:
            cit_validity_rate = None

        # Claim citation coverage
        if val_res.claims:
            claims_with_citations = sum(1 for c in val_res.claims if c.citation_ids)
            citation_coverage = round(claims_with_citations / len(val_res.claims), 4)
        else:
            citation_coverage = None

        # Reference-based citation precision & recall
        cited_chunk_ids = {str(c.chunk_id) for c in val_res.citations}
        relevant_ids = {str(x) for x in query.relevant_chunk_ids}

        if cited_chunk_ids and relevant_ids:
            hits = len(cited_chunk_ids & relevant_ids)
            cit_precision = hits / len(cited_chunk_ids)
            cit_recall = hits / len(relevant_ids)
        elif not relevant_ids:
            cit_precision = 1.0 if not cited_chunk_ids else 0.0
            cit_recall = 1.0
        else:
            cit_precision = 0.0
            cit_recall = 0.0

        context_tokens = sum(src.token_count for src in mock_context_sources)
        exec_latency = latency_ms if latency_ms is not None else 10.0

        return RAGQueryEvalResult(
            query_id=query.query_id,
            query_text=query.query_text,
            category=query.category,
            is_positive=is_positive,
            is_answered=val_res.is_answered,
            expected_to_answer=expected_to_answer,
            abstention_correct=abstention_correct,
            citation_validity_rate=cit_validity_rate,
            attempted_citation_count=attempted_citations,
            valid_citation_count=valid_citations,
            invalid_citation_count=invalid_citations,
            citation_coverage=citation_coverage,
            citation_precision=round(cit_precision, 4),
            citation_recall=round(cit_recall, 4),
            total_latency_ms=round(exec_latency, 2),
            context_tokens=context_tokens,
        )

    def aggregate_results(
        self,
        results: List[RAGQueryEvalResult],
        provider_name: str = "mock",
    ) -> RAGBenchmarkSummary:
        """Aggregates per-query results into a comprehensive RAGBenchmarkSummary."""
        pos_results = [r for r in results if r.is_positive]
        neg_results = [r for r in results if not r.is_positive]

        # Negative abstention
        car = (
            sum(1 for r in neg_results if not r.is_answered) / len(neg_results)
            if neg_results
            else 1.0
        )
        fpr = 1.0 - car

        # Positive harmful abstention (failing to answer when answerable)
        harmful_abstention = (
            sum(1 for r in pos_results if not r.is_answered) / len(pos_results)
            if pos_results
            else 0.0
        )

        valid_rates = [r.citation_validity_rate for r in results if r.citation_validity_rate is not None]
        mean_validity_rate = round(float(np.mean(valid_rates)), 4) if valid_rates else None

        coverage_rates = [r.citation_coverage for r in results if r.citation_coverage is not None]
        mean_coverage = round(float(np.mean(coverage_rates)), 4) if coverage_rates else None

        pos_prec = float(np.mean([r.citation_precision for r in pos_results])) if pos_results else 1.0
        pos_rec = float(np.mean([r.citation_recall for r in pos_results])) if pos_results else 1.0

        total_attempted = sum(r.attempted_citation_count for r in results)
        total_valid = sum(r.valid_citation_count for r in results)
        total_invalid = sum(r.invalid_citation_count for r in results)

        return RAGBenchmarkSummary(
            total_queries=len(results),
            positive_queries_count=len(pos_results),
            negative_queries_count=len(neg_results),
            correct_abstention_rate=round(car, 4),
            false_positive_rate_ungrounded=round(fpr, 4),
            harmful_abstention_rate_positive=round(harmful_abstention, 4),
            total_attempted_citations=total_attempted,
            total_valid_citations=total_valid,
            total_invalid_citations=total_invalid,
            citation_identifier_validity_rate=mean_validity_rate,
            mean_citation_coverage=mean_coverage,
            citation_precision=round(pos_prec, 4),
            citation_recall=round(pos_rec, 4),
            mean_total_latency_ms=round(float(np.mean([r.total_latency_ms for r in results])), 2),
            mean_context_tokens=round(float(np.mean([r.context_tokens for r in results])), 1),
            provider_name=provider_name,
        )
