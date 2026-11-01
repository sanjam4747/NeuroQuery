"""Information retrieval evaluation package for benchmarking and metrics."""

from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from app.evaluation.dataset import (
    BENCHMARK_DOCUMENTS,
    BENCHMARK_QUERIES,
    EvalDocument,
    EvalChunk,
    EvalQuery,
)
from app.evaluation.retrieval_eval import (
    BenchmarkEvaluator,
    QueryEvalResult,
    ConfigurationSummary,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "BENCHMARK_DOCUMENTS",
    "BENCHMARK_QUERIES",
    "EvalDocument",
    "EvalChunk",
    "EvalQuery",
    "BenchmarkEvaluator",
    "QueryEvalResult",
    "ConfigurationSummary",
]
