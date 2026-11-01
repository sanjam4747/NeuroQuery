"""Unit tests for standard IR metrics, abstention metrics, dataset integrity, and deterministic tie-breaking."""

import uuid
import pytest
from app.evaluation.dataset import BENCHMARK_DOCUMENTS, BENCHMARK_QUERIES
from app.evaluation.metrics import (
    correct_abstention_rate,
    empty_result_rate_positive,
    false_positive_rate_ungrounded,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.services.bm25_retrieval import BM25Candidate, BM25Service
from app.services.dense_retrieval import DenseCandidate, DenseRetrievalService
from app.services.reranker import HybridCandidate, RerankerService
from app.services.rrf import RRFService


def test_standard_precision_at_k():
    """Verify precision@k calculation under standard IR definitions."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c6"}

    # Top 3: ["c1", "c2", "c3"] -> 2 hits out of 3 = 2/3
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3, rel=1e-3)

    # Top 5: ["c1", "c2", "c3", "c4", "c5"] -> 2 hits out of 5 = 2/5
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5, rel=1e-3)

    # Top 1: ["c1"] -> 1 hit out of 1 = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)

    # Zero hits
    assert precision_at_k(["c7", "c8"], relevant, k=2) == 0.0

    # Edge cases: empty retrieved or k <= 0
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_standard_recall_at_k():
    """Verify recall@k calculation."""
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c6"}  # Total 3 relevant

    # Top 1: ["c1"] -> 1 / 3
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3, rel=1e-3)

    # Top 3: ["c1", "c2", "c3"] -> 2 / 3
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3, rel=1e-3)

    # If all retrieved:
    all_retrieved = ["c6", "c1", "c3"]
    assert recall_at_k(all_retrieved, relevant, k=3) == pytest.approx(1.0)

    # No hits or empty retrieved
    assert recall_at_k(["c9"], relevant, k=5) == 0.0
    assert recall_at_k([], relevant, k=5) == 0.0


def test_standard_mean_reciprocal_rank():
    """Verify standard reciprocal rank computation."""
    relevant = {"c3", "c5"}

    # First result is relevant
    assert mean_reciprocal_rank(["c3", "c1", "c2"], relevant) == 1.0

    # Second result is relevant
    assert mean_reciprocal_rank(["c1", "c3", "c2"], relevant) == 0.5

    # Third result is relevant
    assert mean_reciprocal_rank(["c1", "c2", "c5"], relevant) == pytest.approx(1 / 3, rel=1e-3)

    # No relevant result found
    assert mean_reciprocal_rank(["c1", "c2", "c8"], relevant) == 0.0

    # Empty retrieved list on positive query
    assert mean_reciprocal_rank([], relevant) == 0.0


def test_standard_ndcg_at_k():
    """Verify normalized discounted cumulative gain calculation under binary relevance."""
    relevant = {"c1", "c2"}

    # Perfect ranking: ["c1", "c2", "c3"] -> DCG = IDCG -> nDCG = 1.0
    assert ndcg_at_k(["c1", "c2", "c3"], relevant, k=3) == pytest.approx(1.0)

    # Suboptimal ranking: ["c3", "c1", "c2"]
    score_suboptimal = ndcg_at_k(["c3", "c1", "c2"], relevant, k=3)
    assert 0.6 < score_suboptimal < 0.8
    assert score_suboptimal < 1.0

    # Zero hits or empty retrieved
    assert ndcg_at_k(["c4", "c5"], relevant, k=2) == 0.0
    assert ndcg_at_k([], relevant, k=2) == 0.0


def test_negative_queries_standard_vs_custom():
    """
    Verify that negative queries (empty relevance set) are evaluated cleanly:
    - Standard IR: Precision = 0.0, Recall = 0.0, MRR = 0.0, nDCG = 0.0.
    - Custom scoring: returns 1.0 when empty, 0.0 when false positives are retrieved.
    """
    empty_relevant = set()

    # Case A: System returned empty list []
    # Standard IR: no hits / k -> 0.0, MRR -> 0.0, nDCG (IDCG=0) -> 0.0
    assert precision_at_k([], empty_relevant, k=5, custom_negative_scoring=False) == 0.0
    assert recall_at_k([], empty_relevant, k=5, custom_negative_scoring=False) == 0.0
    assert mean_reciprocal_rank([], empty_relevant, custom_negative_scoring=False) == 0.0
    assert ndcg_at_k([], empty_relevant, k=5, custom_negative_scoring=False) == 0.0

    # Custom scoring (legacy): awards 1.0 for empty returns
    assert precision_at_k([], empty_relevant, k=5, custom_negative_scoring=True) == 1.0
    assert recall_at_k([], empty_relevant, k=5, custom_negative_scoring=True) == 1.0
    assert mean_reciprocal_rank([], empty_relevant, custom_negative_scoring=True) == 1.0
    assert ndcg_at_k([], empty_relevant, k=5, custom_negative_scoring=True) == 1.0

    # Case B: System returned false positive results ['c1', 'c2']
    assert precision_at_k(["c1", "c2"], empty_relevant, k=5, custom_negative_scoring=False) == 0.0
    assert mean_reciprocal_rank(["c1", "c2"], empty_relevant, custom_negative_scoring=False) == 0.0
    assert ndcg_at_k(["c1", "c2"], empty_relevant, k=5, custom_negative_scoring=False) == 0.0

    assert precision_at_k(["c1", "c2"], empty_relevant, k=5, custom_negative_scoring=True) == 0.0
    assert mean_reciprocal_rank(["c1", "c2"], empty_relevant, custom_negative_scoring=True) == 0.0
    assert ndcg_at_k(["c1", "c2"], empty_relevant, k=5, custom_negative_scoring=True) == 0.0


def test_abstention_metrics():
    """Verify Correct Abstention Rate (CAR), False Positive Rate (FPR), and Empty Result Rate (ERR)."""
    # 5 ungrounded queries: 3 empty, 2 with false positives
    ungrounded_runs = [[], [], [], ["c1"], ["c2", "c3"]]
    assert correct_abstention_rate(ungrounded_runs) == 3 / 5  # 0.60
    assert false_positive_rate_ungrounded(ungrounded_runs) == 2 / 5  # 0.40

    # 4 positive queries: 3 returned results, 1 erroneously returned empty
    positive_runs = [["c1"], ["c2"], [], ["c3"]]
    assert empty_result_rate_positive(positive_runs) == 1 / 4  # 0.25


def test_metrics_multiple_relevant_chunks():
    """Verify metrics calculation for queries with multiple relevant target chunks."""
    relevant = {"c1", "c2", "c3"}  # 3 relevant documents
    retrieved = ["c1", "c9", "c2", "c8", "c3"]  # Hits at ranks 1, 3, 5

    # Precision@5 = 3 / 5 = 0.6
    assert precision_at_k(retrieved, relevant, k=5) == 0.6

    # Recall@5 = 3 / 3 = 1.0
    assert recall_at_k(retrieved, relevant, k=5) == 1.0

    # MRR = 1 / 1 = 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    # nDCG@5: DCG = 1/log2(2) + 1/log2(4) + 1/log2(6) = 1.0 + 0.5 + 0.38685 = 1.88685
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.63093 + 0.5 = 2.13093
    # nDCG = 1.88685 / 2.13093 ~= 0.88546
    assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.88685 / 2.13093, rel=1e-3)


def test_metrics_duplicate_resistance():
    """Verify that duplicate chunk IDs in retrieved results do not artificially inflate hits or DCG."""
    relevant = {"c1"}
    retrieved_with_duplicates = ["c1", "c1", "c2"]

    # Precision@3: only 1 unique relevant chunk in top 3 -> 1 / 3
    assert precision_at_k(retrieved_with_duplicates, relevant, k=3) == pytest.approx(1 / 3, rel=1e-3)

    # Recall@3: only 1 unique relevant chunk found out of 1 -> 1.0
    assert recall_at_k(retrieved_with_duplicates, relevant, k=3) == pytest.approx(1.0)

    # nDCG@3: duplicate c1 at rank 2 does not contribute again to DCG
    assert ndcg_at_k(retrieved_with_duplicates, relevant, k=3) == pytest.approx(1.0)


def test_metrics_fewer_than_k_available():
    """Verify metrics behavior when fewer results than k are retrieved."""
    relevant = {"c1", "c2"}
    retrieved_two = ["c1", "c2"]

    # Precision@5 when only 2 chunks retrieved: 2 / 5 = 0.4
    assert precision_at_k(retrieved_two, relevant, k=5) == 0.4

    # Recall@5: 2 / 2 = 1.0
    assert recall_at_k(retrieved_two, relevant, k=5) == 1.0

    # MRR: rank 1 -> 1.0
    assert mean_reciprocal_rank(retrieved_two, relevant) == 1.0

    # nDCG@5: DCG@2 / IDCG@2 = 1.0
    assert ndcg_at_k(retrieved_two, relevant, k=5) == 1.0


def test_deterministic_tie_breaking_rrf():
    """Verify that RRF produces strictly deterministic order when scores are equal."""
    rrf_service = RRFService()
    id_a = uuid.UUID("11111111-0000-0000-0000-000000000001")
    id_b = uuid.UUID("22222222-0000-0000-0000-000000000002")

    # Tied dense and BM25 scores
    dense = [
        DenseCandidate(id_b, uuid.uuid4(), "b.txt", "text b", 10, 1, 1, {}, 0.85, 1),
        DenseCandidate(id_a, uuid.uuid4(), "a.txt", "text a", 10, 1, 1, {}, 0.85, 1),
    ]
    bm25 = [
        BM25Candidate(id_b, uuid.uuid4(), "b.txt", "text b", 10, 1, 1, {}, 5.0, 1),
        BM25Candidate(id_a, uuid.uuid4(), "a.txt", "text a", 10, 1, 1, {}, 5.0, 1),
    ]

    # Run 10 times to verify deterministic stability
    for _ in range(10):
        fused = rrf_service.fuse(dense, bm25, top_k=2)
        # Because UUID '11111111...' < '22222222...', id_a must strictly precede id_b
        assert fused[0].chunk_id == id_a
        assert fused[1].chunk_id == id_b


def test_deterministic_tie_breaking_reranker():
    """Verify that RerankerService produces strictly deterministic order when cross-encoder scores are identical."""
    id_x = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
    id_y = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")

    candidates = [
        HybridCandidate(id_y, uuid.uuid4(), "y.txt", "text y", 10, 1, 1, {}, 0.8, 1, 5.0, 1, 0.03, 1),
        HybridCandidate(id_x, uuid.uuid4(), "x.txt", "text x", 10, 1, 1, {}, 0.8, 1, 5.0, 1, 0.03, 1),
    ]

    # If reranking is disabled, fallback preserves deterministic order
    reranker = RerankerService()
    for _ in range(5):
        res = reranker.rerank("test query", candidates, enable_reranking=False)
        assert res[0].chunk_id == id_y
        assert res[1].chunk_id == id_x


def test_benchmark_dataset_integrity():
    """Verify evaluation dataset contains >= 30 queries and valid ground truth annotations."""
    assert len(BENCHMARK_QUERIES) >= 30, f"Expected >= 30 queries, got {len(BENCHMARK_QUERIES)}"
    assert len(BENCHMARK_DOCUMENTS) >= 5

    all_chunks = [chunk for doc in BENCHMARK_DOCUMENTS for chunk in doc.chunks]
    assert len(all_chunks) >= 10
    all_chunk_ids = {c.id for c in all_chunks}

    categories = set()
    negative_count = 0

    for query in BENCHMARK_QUERIES:
        assert query.query_text.strip(), "Query text cannot be empty"
        categories.add(query.category)
        if query.category == "negative":
            negative_count += 1
            assert len(query.relevant_chunk_ids) == 0, "Negative queries must have empty relevant chunk sets"
        else:
            assert len(query.relevant_chunk_ids) > 0, f"Query '{query.query_text}' missing relevant chunk annotations"
            for chunk_id in query.relevant_chunk_ids:
                assert chunk_id in all_chunk_ids, f"Chunk {chunk_id} not found in benchmark chunks"

    expected_categories = {
        "exact_keyword",
        "semantic_paraphrase",
        "technical_term",
        "numeric_identifier",
        "negative",
    }
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"
    assert negative_count >= 2, "Expected at least 2 negative queries"
