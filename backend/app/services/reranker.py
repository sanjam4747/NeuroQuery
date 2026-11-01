"""Cross-Encoder reranking service using Sentence-Transformers."""

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional
import uuid
import torch
from sentence_transformers import CrossEncoder
from app.core.config import settings
from app.core.logging import logger
from app.services.rrf import HybridCandidate


@dataclass
class RerankedCandidate:
    """Represents a final candidate chunk scored by the cross-encoder reranker."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    character_count: int
    page_start: Optional[int]
    page_end: Optional[int]
    metadata_json: Dict[str, Any] = field(default_factory=dict)

    # Historical Retrieval & Fusion Metrics
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = 0.0
    rrf_rank: int = 1

    # Reranking Metrics
    reranker_score: Optional[float] = None
    final_rank: int = 1


class RerankerService:
    """
    Manages loading and batched inference for cross-encoder reranking.

    Design details:
    - Default Model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
    - Joint query-document attention scoring: evaluates true interaction between query and chunk.
    - Singleton Lifecycle: Model weights are loaded once upon first invocation and reused.
    - Device Agnostic: Supports CPU inference and detects CUDA when available.
    - Operates only on top fused candidates from RRF (default 20 candidates), keeping latency low.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._model: Optional[CrossEncoder] = None
        self._device: Optional[str] = None
        self._load_error: Optional[str] = None

    @property
    def model_name(self) -> str:
        """Configured cross-encoder model identifier."""
        return settings.RERANKER_MODEL_NAME

    def resolve_device(self) -> str:
        """Determines computation device for reranking."""
        configured = settings.RERANKER_DEVICE.lower()
        if configured == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested for reranker but not available. Falling back to CPU.")
            return "cpu"
        elif configured == "cpu":
            return "cpu"
        else:  # "auto"
            return "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self) -> CrossEncoder:
        """
        Lazily loads and returns the singleton CrossEncoder instance.
        Thread-safe across multiple concurrent requests.
        """
        with self._lock:
            if self._model is not None:
                return self._model

            if self._load_error:
                raise RuntimeError(f"Reranker model initialization previously failed: {self._load_error}")

            self._device = self.resolve_device()
            logger.info("Loading cross-encoder reranker '%s' on device '%s'...", self.model_name, self._device)
            start_time = time.perf_counter()

            try:
                self._model = CrossEncoder(self.model_name, device=self._device)
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.info("CrossEncoder reranker loaded successfully in %.1f ms", duration_ms)
                return self._model
            except Exception as exc:
                self._load_error = str(exc)
                logger.error("Failed to load cross-encoder model '%s': %s", self.model_name, exc, exc_info=True)
                raise RuntimeError(f"Failed to initialize reranker model '{self.model_name}': {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: List[HybridCandidate],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
        enable_reranking: bool = True,
    ) -> List[RerankedCandidate]:
        """
        Scores candidate query-document pairs and orders them by cross-encoder relevance.

        Args:
            query: User search query.
            candidates: List of fused HybridCandidate objects from RRF.
            top_k: Number of final ranked candidates to return (defaults to settings.RERANKER_TOP_K).
            batch_size: Batch size for model inference (defaults to settings.RERANKER_BATCH_SIZE).
            enable_reranking: If False, preserves RRF ranking and bypasses model inference.

        Returns:
            List of RerankedCandidate objects ordered by relevance.
        """
        if not candidates:
            return []

        eff_top_k = top_k or settings.RERANKER_TOP_K
        eff_batch_size = batch_size or settings.RERANKER_BATCH_SIZE

        # Fast path if reranking is explicitly disabled
        if not enable_reranking or not settings.ENABLE_RERANKING:
            results: List[RerankedCandidate] = []
            for rank_idx, cand in enumerate(candidates[:eff_top_k], start=1):
                results.append(
                    RerankedCandidate(
                        chunk_id=cand.chunk_id,
                        document_id=cand.document_id,
                        filename=cand.filename,
                        text=cand.text,
                        character_count=cand.character_count,
                        page_start=cand.page_start,
                        page_end=cand.page_end,
                        metadata_json=cand.metadata_json,
                        dense_score=cand.dense_score,
                        dense_rank=cand.dense_rank,
                        bm25_score=cand.bm25_score,
                        bm25_rank=cand.bm25_rank,
                        rrf_score=cand.rrf_score,
                        rrf_rank=cand.rrf_rank,
                        reranker_score=None,
                        final_rank=rank_idx,
                    )
                )
            return results

        # Construct (query, passage) pairs for the cross-encoder
        pairs = [(query, cand.text) for cand in candidates]

        try:
            model = self.get_model()
            scores = model.predict(
                pairs,
                batch_size=eff_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Associate scores with candidates
            scored_candidates: List[tuple[HybridCandidate, float]] = []
            for idx, score_val in enumerate(scores):
                scored_candidates.append((candidates[idx], float(score_val)))

            # Deterministic sort: cross-encoder score descending, RRF score descending, chunk ID ascending
            scored_candidates.sort(key=lambda item: (-item[1], -item[0].rrf_score, str(item[0].chunk_id)))

            final_results: List[RerankedCandidate] = []
            for rank_idx, (cand, score) in enumerate(scored_candidates[:eff_top_k], start=1):
                final_results.append(
                    RerankedCandidate(
                        chunk_id=cand.chunk_id,
                        document_id=cand.document_id,
                        filename=cand.filename,
                        text=cand.text,
                        character_count=cand.character_count,
                        page_start=cand.page_start,
                        page_end=cand.page_end,
                        metadata_json=cand.metadata_json,
                        dense_score=cand.dense_score,
                        dense_rank=cand.dense_rank,
                        bm25_score=cand.bm25_score,
                        bm25_rank=cand.bm25_rank,
                        rrf_score=cand.rrf_score,
                        rrf_rank=cand.rrf_rank,
                        reranker_score=round(score, 4),
                        final_rank=rank_idx,
                    )
                )

            logger.debug(
                "Cross-encoder reranked %d candidates for query '%s' (top_k=%d)",
                len(candidates),
                query[:30],
                eff_top_k,
            )
            return final_results

        except Exception as exc:
            logger.error("Reranking failed (%s). Falling back to RRF rankings.", exc, exc_info=True)
            # Resilient fallback to RRF order
            fallback_results: List[RerankedCandidate] = []
            for rank_idx, cand in enumerate(candidates[:eff_top_k], start=1):
                fallback_results.append(
                    RerankedCandidate(
                        chunk_id=cand.chunk_id,
                        document_id=cand.document_id,
                        filename=cand.filename,
                        text=cand.text,
                        character_count=cand.character_count,
                        page_start=cand.page_start,
                        page_end=cand.page_end,
                        metadata_json=cand.metadata_json,
                        dense_score=cand.dense_score,
                        dense_rank=cand.dense_rank,
                        bm25_score=cand.bm25_score,
                        bm25_rank=cand.bm25_rank,
                        rrf_score=cand.rrf_score,
                        rrf_rank=cand.rrf_rank,
                        reranker_score=None,
                        final_rank=rank_idx,
                    )
                )
            return fallback_results


# Singleton instance
reranker_service = RerankerService()
