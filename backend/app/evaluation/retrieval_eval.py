"""
Systematic Retrieval Quality Benchmark Evaluator.

Executes comparative offline evaluations across four retrieval modalities:
1. Dense Semantic Retrieval Only (BAAI/bge-small-en-v1.5)
2. Sparse Lexical Retrieval Only (LuceneBM25)
3. Hybrid Retrieval via Reciprocal Rank Fusion (RRF k=60)
4. Hybrid Retrieval with Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)

Rigorously separates:
- Standard IR metrics on positive queries (Precision@5, Recall@5, MRR, nDCG@5).
- Dedicated abstention metrics on ungrounded queries (Correct Abstention Rate, False Positive Rate).
- Harmful abstention metrics on positive queries (Empty Result Rate).
- Deterministic secondary key tie-breaking across all ranking stages.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Set
import uuid
import numpy as np

from app.core.logging import logger
from app.evaluation.dataset import (
    BENCHMARK_DOCUMENTS,
    BENCHMARK_QUERIES,
    EvalChunk,
    EvalDocument,
    EvalQuery,
)
from app.evaluation.metrics import (
    correct_abstention_rate,
    empty_result_rate_positive,
    false_positive_rate_ungrounded,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.services.bm25_retrieval import BM25Candidate, LuceneBM25, tokenize_for_bm25
from app.services.dense_retrieval import DenseCandidate
from app.services.embeddings import embedding_service
from app.services.reranker import reranker_service
from app.services.rrf import rrf_service


@dataclass
class QueryEvalResult:
    """Individual query performance metrics under a specific retrieval configuration."""

    query_id: str
    query_text: str
    category: str
    configuration: str
    retrieved_chunk_ids: List[uuid.UUID]
    relevant_chunk_ids: Set[uuid.UUID]
    # Standard IR metrics
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    # Historical custom negative scoring
    legacy_precision_at_5: float
    legacy_mrr: float
    legacy_ndcg_at_5: float
    latency_ms: float


@dataclass
class ConfigurationSummary:
    """Aggregated evaluation metrics for a single retrieval configuration."""

    configuration_name: str
    # 1. Positive queries only (27 queries with ground-truth relevant documents)
    positive_precision_at_5: float
    positive_recall_at_5: float
    positive_mrr: float
    positive_ndcg_at_5: float
    empty_result_rate_positive: float

    # 2. Ungrounded queries only (5 queries with no relevant documents)
    correct_abstention_rate: float
    false_positive_rate_ungrounded: float

    # 3. Standard IR overall macro-average across all 32 queries (strictly standard definitions)
    standard_overall_precision_at_5: float
    standard_overall_recall_at_5: float
    standard_overall_mrr: float
    standard_overall_ndcg_at_5: float

    # 4. Historical comparison: Phase 4 custom scoring convention across all 32 queries
    legacy_custom_mean_mrr: float
    legacy_custom_mean_ndcg_at_5: float

    # 5. Latency profiling (warm execution)
    mean_latency_ms: float
    p95_latency_ms: float

    # Category-specific breakdowns
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BenchmarkEvaluator:
    """
    Executes systematic comparative retrieval benchmarking over the 32 queries in BENCHMARK_QUERIES.
    """

    def __init__(self, documents: List[EvalDocument] = BENCHMARK_DOCUMENTS):
        self.documents = documents
        self.chunks: List[EvalChunk] = []
        self.chunk_map: Dict[uuid.UUID, EvalChunk] = {}
        self.chunk_doc_map: Dict[uuid.UUID, str] = {}

        for doc in self.documents:
            for ch in doc.chunks:
                self.chunks.append(ch)
                self.chunk_map[ch.id] = ch
                self.chunk_doc_map[ch.id] = doc.filename

        # State for benchmark indices
        self._dense_embeddings: Optional[np.ndarray] = None
        self._bm25: Optional[LuceneBM25] = None
        self._tokenized_corpus: List[List[str]] = []
        self._is_indexed: bool = False

    def build_benchmark_indices(self) -> None:
        """Embeds all benchmark chunks, initializes BM25 index, and pre-warms models."""
        if self._is_indexed:
            return

        logger.info("Initializing benchmark indices for %d chunks...", len(self.chunks))

        # 1. Generate dense embeddings
        texts = [ch.text for ch in self.chunks]
        vecs, _ = embedding_service.embed_batch(texts)
        self._dense_embeddings = np.array(vecs, dtype=np.float32)

        # 2. Build BM25 index
        self._tokenized_corpus = [tokenize_for_bm25(ch.text) for ch in self.chunks]
        self._bm25 = LuceneBM25(self._tokenized_corpus)

        # 3. Pre-warm cross-encoder model to prevent cold disk-load from distorting benchmark timing
        _ = reranker_service.get_model()
        self._is_indexed = True
        logger.info("Benchmark indices ready and models pre-warmed.")

    def run_dense_only(self, query: str, top_k: int = 5) -> tuple[List[uuid.UUID], float]:
        """Executes Dense Semantic Retrieval with deterministic chunk-ID tie-breaking."""
        start = time.perf_counter()
        q_vec = np.array(embedding_service.embed_text(query), dtype=np.float32)
        sims = np.dot(self._dense_embeddings, q_vec)

        # Deterministic sorting: similarity score descending, chunk ID string ascending
        ranked_indices = sorted(
            range(len(self.chunks)),
            key=lambda idx: (-float(sims[idx]), str(self.chunks[idx].id)),
        )[:top_k]

        retrieved_ids = [self.chunks[idx].id for idx in ranked_indices]
        latency_ms = (time.perf_counter() - start) * 1000
        return retrieved_ids, latency_ms

    def run_bm25_only(self, query: str, top_k: int = 5) -> tuple[List[uuid.UUID], float]:
        """Executes Sparse BM25 Retrieval with deterministic chunk-ID tie-breaking."""
        start = time.perf_counter()
        tokens = tokenize_for_bm25(query)
        if not tokens or self._bm25 is None:
            return [], (time.perf_counter() - start) * 1000

        scores = self._bm25.get_scores(tokens)
        scored = [(idx, float(s)) for idx, s in enumerate(scores) if s > 0.0]

        # Deterministic sorting: score descending, chunk ID string ascending
        scored.sort(key=lambda item: (-item[1], str(self.chunks[item[0]].id)))

        retrieved_ids = [self.chunks[idx].id for idx, _ in scored[:top_k]]
        latency_ms = (time.perf_counter() - start) * 1000
        return retrieved_ids, latency_ms

    def run_hybrid_rrf(self, query: str, top_k: int = 5, rrf_k: int = 60) -> tuple[List[uuid.UUID], float]:
        """Executes Hybrid Dense + BM25 with Reciprocal Rank Fusion."""
        start = time.perf_counter()

        # 1. Dense top-20 with deterministic tie-breaking
        q_vec = np.array(embedding_service.embed_text(query), dtype=np.float32)
        dense_sims = np.dot(self._dense_embeddings, q_vec)
        dense_top_indices = sorted(
            range(len(self.chunks)),
            key=lambda idx: (-float(dense_sims[idx]), str(self.chunks[idx].id)),
        )[:20]

        dense_cands = [
            DenseCandidate(
                chunk_id=self.chunks[i].id,
                document_id=uuid.uuid4(),
                filename=self.chunk_doc_map[self.chunks[i].id],
                text=self.chunks[i].text,
                character_count=len(self.chunks[i].text),
                page_start=self.chunks[i].page_start,
                page_end=self.chunks[i].page_end,
                score=float(dense_sims[i]),
                rank=r,
            )
            for r, i in enumerate(dense_top_indices, start=1)
        ]

        # 2. BM25 top-20 with deterministic tie-breaking
        tokens = tokenize_for_bm25(query)
        bm25_cands: List[BM25Candidate] = []
        if tokens and self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
            scored = [(idx, float(s)) for idx, s in enumerate(scores) if s > 0.0]
            scored.sort(key=lambda item: (-item[1], str(self.chunks[item[0]].id)))
            bm25_cands = [
                BM25Candidate(
                    chunk_id=self.chunks[i].id,
                    document_id=uuid.uuid4(),
                    filename=self.chunk_doc_map[self.chunks[i].id],
                    text=self.chunks[i].text,
                    character_count=len(self.chunks[i].text),
                    page_start=self.chunks[i].page_start,
                    page_end=self.chunks[i].page_end,
                    score=s,
                    rank=r,
                )
                for r, (i, s) in enumerate(scored[:20], start=1)
            ]

        # 3. RRF Fusion (deterministic)
        fused = rrf_service.fuse(dense_cands, bm25_cands, rrf_k=rrf_k, top_k=top_k)
        retrieved_ids = [c.chunk_id for c in fused]
        latency_ms = (time.perf_counter() - start) * 1000
        return retrieved_ids, latency_ms

    def run_hybrid_reranked(self, query: str, top_k: int = 5, rrf_k: int = 60) -> tuple[List[uuid.UUID], float]:
        """Executes Hybrid Dense + BM25 + Cross-Encoder Reranking."""
        start = time.perf_counter()

        # 1. Dense top-20
        q_vec = np.array(embedding_service.embed_text(query), dtype=np.float32)
        dense_sims = np.dot(self._dense_embeddings, q_vec)
        dense_top_indices = sorted(
            range(len(self.chunks)),
            key=lambda idx: (-float(dense_sims[idx]), str(self.chunks[idx].id)),
        )[:20]

        dense_cands = [
            DenseCandidate(
                chunk_id=self.chunks[i].id,
                document_id=uuid.uuid4(),
                filename=self.chunk_doc_map[self.chunks[i].id],
                text=self.chunks[i].text,
                character_count=len(self.chunks[i].text),
                page_start=self.chunks[i].page_start,
                page_end=self.chunks[i].page_end,
                score=float(dense_sims[i]),
                rank=r,
            )
            for r, i in enumerate(dense_top_indices, start=1)
        ]

        # 2. BM25 top-20
        tokens = tokenize_for_bm25(query)
        bm25_cands: List[BM25Candidate] = []
        if tokens and self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
            scored = [(idx, float(s)) for idx, s in enumerate(scores) if s > 0.0]
            scored.sort(key=lambda item: (-item[1], str(self.chunks[item[0]].id)))
            bm25_cands = [
                BM25Candidate(
                    chunk_id=self.chunks[i].id,
                    document_id=uuid.uuid4(),
                    filename=self.chunk_doc_map[self.chunks[i].id],
                    text=self.chunks[i].text,
                    character_count=len(self.chunks[i].text),
                    page_start=self.chunks[i].page_start,
                    page_end=self.chunks[i].page_end,
                    score=s,
                    rank=r,
                )
                for r, (i, s) in enumerate(scored[:20], start=1)
            ]

        # 3. RRF Fusion (top 20 candidates for cross-encoder reranker)
        fused = rrf_service.fuse(dense_cands, bm25_cands, rrf_k=rrf_k, top_k=20)

        # 4. Cross-Encoder Reranking
        reranked = reranker_service.rerank(query=query, candidates=fused, top_k=top_k, enable_reranking=True)
        retrieved_ids = [c.chunk_id for c in reranked]
        latency_ms = (time.perf_counter() - start) * 1000
        return retrieved_ids, latency_ms

    def evaluate_all(self, queries: Optional[List[EvalQuery]] = None) -> Dict[str, ConfigurationSummary]:
        """
        Runs all 32 benchmark queries across all 4 configurations and aggregates performance.
        Separates positive-query IR metrics from ungrounded-query abstention metrics.
        """
        queries = queries or BENCHMARK_QUERIES
        self.build_benchmark_indices()

        configs = [
            ("Dense Only", self.run_dense_only),
            ("BM25 Only", self.run_bm25_only),
            ("Hybrid (RRF)", self.run_hybrid_rrf),
            ("Hybrid + Reranker", self.run_hybrid_reranked),
        ]

        summaries: Dict[str, ConfigurationSummary] = {}

        for cfg_name, runner in configs:
            results: List[QueryEvalResult] = []
            latencies: List[float] = []

            for q in queries:
                retrieved, lat_ms = runner(q.query_text, top_k=5)
                latencies.append(lat_ms)

                # Standard IR metrics
                p5 = precision_at_k(retrieved, q.relevant_chunk_ids, k=5, custom_negative_scoring=False)
                r5 = recall_at_k(retrieved, q.relevant_chunk_ids, k=5, custom_negative_scoring=False)
                mrr = mean_reciprocal_rank(retrieved, q.relevant_chunk_ids, custom_negative_scoring=False)
                ndcg = ndcg_at_k(retrieved, q.relevant_chunk_ids, k=5, custom_negative_scoring=False)

                # Historical custom scoring
                leg_p5 = precision_at_k(retrieved, q.relevant_chunk_ids, k=5, custom_negative_scoring=True)
                leg_mrr = mean_reciprocal_rank(retrieved, q.relevant_chunk_ids, custom_negative_scoring=True)
                leg_ndcg = ndcg_at_k(retrieved, q.relevant_chunk_ids, k=5, custom_negative_scoring=True)

                results.append(
                    QueryEvalResult(
                        query_id=q.query_id,
                        query_text=q.query_text,
                        category=q.category,
                        configuration=cfg_name,
                        retrieved_chunk_ids=retrieved,
                        relevant_chunk_ids=q.relevant_chunk_ids,
                        precision_at_5=p5,
                        recall_at_5=r5,
                        mrr=mrr,
                        ndcg_at_5=ndcg,
                        legacy_precision_at_5=leg_p5,
                        legacy_mrr=leg_mrr,
                        legacy_ndcg_at_5=leg_ndcg,
                        latency_ms=lat_ms,
                    )
                )

            # Separate positive queries and ungrounded queries
            pos_results = [r for r in results if len(r.relevant_chunk_ids) > 0]
            ungrounded_results = [r for r in results if len(r.relevant_chunk_ids) == 0]

            pos_retrieved = [r.retrieved_chunk_ids for r in pos_results]
            ungrounded_retrieved = [r.retrieved_chunk_ids for r in ungrounded_results]

            # 1. Positive queries metrics (27 queries)
            pos_p5 = float(np.mean([r.precision_at_5 for r in pos_results])) if pos_results else 0.0
            pos_r5 = float(np.mean([r.recall_at_5 for r in pos_results])) if pos_results else 0.0
            pos_mrr = float(np.mean([r.mrr for r in pos_results])) if pos_results else 0.0
            pos_ndcg = float(np.mean([r.ndcg_at_5 for r in pos_results])) if pos_results else 0.0
            err_pos = empty_result_rate_positive(pos_retrieved)

            # 2. Ungrounded queries metrics (5 queries)
            car = correct_abstention_rate(ungrounded_retrieved)
            fpr = false_positive_rate_ungrounded(ungrounded_retrieved)

            # 3. Standard IR overall macro-average across all 32 queries
            std_all_p5 = float(np.mean([r.precision_at_5 for r in results]))
            std_all_r5 = float(np.mean([r.recall_at_5 for r in results]))
            std_all_mrr = float(np.mean([r.mrr for r in results]))
            std_all_ndcg = float(np.mean([r.ndcg_at_5 for r in results]))

            # 4. Legacy custom macro-average
            leg_all_mrr = float(np.mean([r.legacy_mrr for r in results]))
            leg_all_ndcg = float(np.mean([r.legacy_ndcg_at_5 for r in results]))

            # Category breakdowns
            categories = sorted({q.category for q in queries})
            cat_metrics: Dict[str, Dict[str, float]] = {}
            for cat in categories:
                cat_results = [r for r in results if r.category == cat]
                cat_metrics[cat] = {
                    "precision_at_5": round(float(np.mean([r.precision_at_5 for r in cat_results])), 4),
                    "recall_at_5": round(float(np.mean([r.recall_at_5 for r in cat_results])), 4),
                    "mrr": round(float(np.mean([r.mrr for r in cat_results])), 4),
                    "ndcg_at_5": round(float(np.mean([r.ndcg_at_5 for r in cat_results])), 4),
                    "mean_latency_ms": round(float(np.mean([r.latency_ms for r in cat_results])), 2),
                }

            summaries[cfg_name] = ConfigurationSummary(
                configuration_name=cfg_name,
                positive_precision_at_5=round(pos_p5, 4),
                positive_recall_at_5=round(pos_r5, 4),
                positive_mrr=round(pos_mrr, 4),
                positive_ndcg_at_5=round(pos_ndcg, 4),
                empty_result_rate_positive=round(err_pos, 4),
                correct_abstention_rate=round(car, 4),
                false_positive_rate_ungrounded=round(fpr, 4),
                standard_overall_precision_at_5=round(std_all_p5, 4),
                standard_overall_recall_at_5=round(std_all_r5, 4),
                standard_overall_mrr=round(std_all_mrr, 4),
                standard_overall_ndcg_at_5=round(std_all_ndcg, 4),
                legacy_custom_mean_mrr=round(leg_all_mrr, 4),
                legacy_custom_mean_ndcg_at_5=round(leg_all_ndcg, 4),
                mean_latency_ms=round(float(np.mean(latencies)), 2),
                p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
                category_metrics=cat_metrics,
            )

        return summaries


# CLI Runner for manual testing and reporting
if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    print("Executing Phase 4 comparative retrieval evaluation across 32 queries...")
    results = evaluator.evaluate_all()

    print("\n" + "=" * 90)
    print("TABLE 1: POSITIVE QUERIES RETRIEVAL PERFORMANCE (27 queries with ground truth)")
    print("=" * 90)
    print(f"{'Configuration':<22} | {'Precision@5':<11} | {'Recall@5':<10} | {'MRR':<8} | {'nDCG@5':<8} | {'Empty Rate':<10}")
    print("-" * 90)
    for name, s in results.items():
        print(
            f"{s.configuration_name:<22} | {s.positive_precision_at_5:<11.4f} | {s.positive_recall_at_5:<10.4f} | "
            f"{s.positive_mrr:<8.4f} | {s.positive_ndcg_at_5:<8.4f} | {s.empty_result_rate_positive:<10.4f}"
        )
    print("=" * 90)

    print("\n" + "=" * 90)
    print("TABLE 2: UNGROUNDED QUERIES ABSTENTION PERFORMANCE (5 negative queries)")
    print("=" * 90)
    print(f"{'Configuration':<22} | {'Correct Abstention Rate (CAR)':<30} | {'False Positive Rate (FPR)':<28}")
    print("-" * 90)
    for name, s in results.items():
        print(
            f"{s.configuration_name:<22} | {s.correct_abstention_rate:<30.4f} | {s.false_positive_rate_ungrounded:<28.4f}"
        )
    print("=" * 90)

    print("\n" + "=" * 90)
    print("TABLE 3: STANDARD IR OVERALL MACRO-AVERAGE (All 32 queries, strict IR definitions)")
    print("=" * 90)
    print(f"{'Configuration':<22} | {'Precision@5':<11} | {'Recall@5':<10} | {'MRR':<8} | {'nDCG@5':<8}")
    print("-" * 90)
    for name, s in results.items():
        print(
            f"{s.configuration_name:<22} | {s.standard_overall_precision_at_5:<11.4f} | {s.standard_overall_recall_at_5:<10.4f} | "
            f"{s.standard_overall_mrr:<8.4f} | {s.standard_overall_ndcg_at_5:<8.4f}"
        )
    print("=" * 90)

    print("\n" + "=" * 90)
    print("TABLE 4: HISTORICAL COMPARISON (Legacy Custom Negative Scoring)")
    print("=" * 90)
    print(f"{'Configuration':<22} | {'Legacy MRR':<12} | {'Legacy nDCG@5':<14} | {'Mean Latency':<12} | {'p95 Latency':<12}")
    print("-" * 90)
    for name, s in results.items():
        print(
            f"{s.configuration_name:<22} | {s.legacy_custom_mean_mrr:<12.4f} | {s.legacy_custom_mean_ndcg_at_5:<14.4f} | "
            f"{s.mean_latency_ms:<10.2f} ms | {s.p95_latency_ms:<10.2f} ms"
        )
    print("=" * 90)
