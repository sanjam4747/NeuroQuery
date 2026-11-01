"""
Standard Information Retrieval (IR) evaluation metrics and abstention metrics.

Separation of Concerns:
1. Standard IR Metrics (Precision@K, Recall@K, MRR, nDCG@K):
   - Defined strictly for queries where ground-truth relevant documents exist (|Relevant| > 0).
   - If an empty result list is returned for a positive query, metrics evaluate to 0.0 (missed retrieval).
   - If a query has NO relevant documents (|Relevant| == 0, e.g. ungrounded/negative query):
     - Standard fixed-denominator Precision@K: 0 / K = 0.0.
     - Recall@K is mathematically undefined (0/0) and set to 0.0 with documentation.
     - MRR is 0.0 (no relevant document was retrieved because none exists).
     - nDCG@K: when IDCG == 0, standard convention (e.g. scikit-learn, trec_eval) defines nDCG = 0.0.
   - For historical comparison only, custom_negative_scoring=True retains the Phase 4 legacy convention
     (where empty returns on ungrounded queries were awarded 1.0).

2. Abstention Metrics (For Ungrounded and Positive Queries):
   - Correct Abstention Rate (CAR): Fraction of ungrounded queries where system correctly returned [] (|Retrieved| == 0).
   - False Positive Rate (FPR): Fraction of ungrounded queries where system returned at least 1 candidate (|Retrieved| > 0).
   - Empty-Result Rate on Positive Queries (Harmful Abstention / Under-retrieval): Fraction of positive queries where system erroneously returned [].
"""

import math
from typing import List, Sequence, Set, Union
import uuid

IdType = Union[uuid.UUID, str]


def precision_at_k(
    retrieved: Sequence[IdType],
    relevant: Set[IdType],
    k: int,
    custom_negative_scoring: bool = False,
) -> float:
    """
    Computes Precision@K: the fraction of retrieved top-K chunks that are relevant.

    Standard Definition:
        Precision@K = |Retrieved_K intersect Relevant| / K

    Edge Cases:
        - k <= 0: returns 0.0
        - |Relevant| == 0 (ungrounded query):
            - If custom_negative_scoring is True (legacy Phase 4 convention): returns 1.0 if len(retrieved) == 0 else 0.0.
            - Otherwise (Standard IR): returns 0.0 (0 hits / K).
        - len(retrieved) < k: correctly divided by k (fixed denominator).
        - Duplicate candidate IDs in retrieved: deduplicated within top-k so duplicates cannot inflate hits.
    """
    if k <= 0:
        return 0.0

    if not relevant:
        if custom_negative_scoring:
            return 1.0 if len(retrieved) == 0 else 0.0
        return 0.0

    top_k_items = list(retrieved)[:k]
    if not top_k_items:
        return 0.0

    unique_retrieved_in_k = set(top_k_items)
    hits = len(unique_retrieved_in_k & relevant)
    return hits / k


def recall_at_k(
    retrieved: Sequence[IdType],
    relevant: Set[IdType],
    k: int,
    custom_negative_scoring: bool = False,
) -> float:
    """
    Computes Recall@K: the fraction of all relevant chunks found within the top-K results.

    Standard Definition:
        Recall@K = |Retrieved_K intersect Relevant| / |Relevant|

    Edge Cases:
        - |Relevant| == 0:
            - If custom_negative_scoring is True: returns 1.0 (no relevant items missed).
            - Otherwise (Standard IR): 0/0 is mathematically undefined; returns 0.0.
        - k <= 0: returns 0.0.
        - Duplicate candidate IDs in retrieved: deduplicated within top-k.
    """
    if not relevant:
        if custom_negative_scoring:
            return 1.0
        return 0.0

    if k <= 0:
        return 0.0

    top_k_items = list(retrieved)[:k]
    unique_retrieved_in_k = set(top_k_items)
    hits = len(unique_retrieved_in_k & relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(
    retrieved: Sequence[IdType],
    relevant: Set[IdType],
    custom_negative_scoring: bool = False,
) -> float:
    """
    Computes Reciprocal Rank (RR): reciprocal of the 1-indexed position of the first relevant result.

    Standard Definition:
        RR = 1 / rank* (where rank* is the first 1-indexed position of a relevant item).
        If no relevant item is retrieved, RR = 0.0.

    Edge Cases:
        - |Relevant| == 0:
            - If custom_negative_scoring is True: returns 1.0 if len(retrieved) == 0 else 0.0.
            - Otherwise (Standard IR): returns 0.0 (no relevant items exist, so none retrieved).
    """
    if not relevant:
        if custom_negative_scoring:
            return 1.0 if len(retrieved) == 0 else 0.0
        return 0.0

    for rank_idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[IdType],
    relevant: Set[IdType],
    k: int,
    custom_negative_scoring: bool = False,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@K).

    Standard Binary Formulation:
        DCG@K = sum_{i=1}^K [ rel_i / log2(i + 1) ]
        IDCG@K = sum_{i=1}^{min(|relevant|, K)} [ 1 / log2(i + 1) ]
        nDCG@K = DCG@K / IDCG@K

    Edge Cases:
        - |Relevant| == 0 (IDCG == 0.0):
            - If custom_negative_scoring is True: returns 1.0 if len(retrieved) == 0 else 0.0.
            - Otherwise (Standard IR convention): returns 0.0 when IDCG == 0.0.
        - k <= 0: returns 0.0.
        - Deduplication: uses a 'seen' set to prevent identical candidate IDs from accumulating multiple discount gains.
    """
    if not relevant:
        if custom_negative_scoring:
            return 1.0 if len(retrieved) == 0 else 0.0
        return 0.0

    if k <= 0:
        return 0.0

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0

    top_k_items = list(retrieved)[:k]
    dcg = 0.0
    seen: Set[IdType] = set()
    for idx, item in enumerate(top_k_items, start=1):
        if item in relevant and item not in seen:
            seen.add(item)
            dcg += 1.0 / math.log2(idx + 1)

    return dcg / idcg


# =========================================================================
# Specialized Abstention & Error Metrics
# =========================================================================

def correct_abstention_rate(retrieval_runs: Sequence[Sequence[IdType]]) -> float:
    """
    Fraction of ungrounded queries where the system correctly returned an empty result set ([]).

    CAR = (Count of ungrounded queries with |retrieved| == 0) / (Total ungrounded queries)
    """
    if not retrieval_runs:
        return 0.0
    correct_count = sum(1 for retrieved in retrieval_runs if len(retrieved) == 0)
    return correct_count / len(retrieval_runs)


def false_positive_rate_ungrounded(retrieval_runs: Sequence[Sequence[IdType]]) -> float:
    """
    Fraction of ungrounded queries where the system erroneously returned candidates.

    FPR = (Count of ungrounded queries with |retrieved| > 0) / (Total ungrounded queries) = 1.0 - CAR
    """
    if not retrieval_runs:
        return 0.0
    fp_count = sum(1 for retrieved in retrieval_runs if len(retrieved) > 0)
    return fp_count / len(retrieval_runs)


def empty_result_rate_positive(retrieval_runs: Sequence[Sequence[IdType]]) -> float:
    """
    Fraction of positive queries where the system erroneously returned an empty result set ([]),
    indicating harmful under-retrieval / false abstention.

    ERR = (Count of positive queries with |retrieved| == 0) / (Total positive queries)
    """
    if not retrieval_runs:
        return 0.0
    empty_count = sum(1 for retrieved in retrieval_runs if len(retrieved) == 0)
    return empty_count / len(retrieval_runs)
