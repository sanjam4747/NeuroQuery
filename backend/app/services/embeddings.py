"""Dedicated semantic embedding service using Sentence-Transformers."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    """
    Manages loading, execution, token auditing, and normalization for semantic embeddings.

    Design details:
    - Default Model: 'BAAI/bge-small-en-v1.5' (384 dimensions, 512 max token sequence length).
    - Optimized for English language semantic document retrieval.
    - Cosine Metric: Embeddings are strictly L2 normalized (||v|| = 1.0) so that cosine similarity
      is equivalent to the inner dot product <u, v>.
    - Singleton Lifecycle: Model weights are loaded once into memory on first invocation and reused.
    - Device Agnostic: Automatically detects CUDA or falls back to CPU based on settings.
    - Safety: Inspects token count per chunk to detect and flag chunks exceeding sequence limits.
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[str] = None
        self._load_error: Optional[str] = None

    @property
    def model_name(self) -> str:
        """Configured embedding model identifier."""
        return settings.EMBEDDING_MODEL_NAME

    @property
    def dimension(self) -> int:
        """Expected output vector dimensionality."""
        return settings.EMBEDDING_DIMENSION

    @property
    def max_sequence_length(self) -> int:
        """Maximum token sequence length supported by the model (512 for BGE-small)."""
        if self._model is not None:
            return getattr(self._model, "max_seq_length", 512)
        return 512

    def resolve_device(self) -> str:
        """Determines computation device based on configuration and hardware availability."""
        configured = settings.EMBEDDING_DEVICE.lower()
        if configured == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested in configuration but not available. Falling back to CPU.")
            return "cpu"
        elif configured == "cpu":
            return "cpu"
        else:  # "auto"
            return "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self) -> SentenceTransformer:
        """
        Lazily loads and returns the singleton SentenceTransformer instance.
        Thread-safe in standard Python runtime; logs initialization metrics.
        """
        if self._model is not None:
            return self._model

        if self._load_error:
            raise RuntimeError(f"Embedding model initialization previously failed: {self._load_error}")

        self._device = self.resolve_device()
        logger.info(
            "Loading embedding model '%s' on device '%s' (English optimized, %d dimensions)...",
            self.model_name,
            self._device,
            self.dimension,
        )
        start_time = time.perf_counter()

        try:
            self._model = SentenceTransformer(self.model_name, device=self._device)
            load_duration = (time.perf_counter() - start_time) * 1000
            logger.info("Embedding model loaded successfully in %.1f ms", load_duration)
            return self._model
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Failed to load embedding model '%s': %s", self.model_name, exc, exc_info=True)
            raise RuntimeError(f"Failed to initialize embedding model '{self.model_name}': {exc}") from exc

    def tokenize_into_windows(
        self,
        text: str,
        window_size: int = 450,
        overlap_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Tokenizes text using the model tokenizer and segments token IDs into bounded sub-windows.
        Each window contains exact token IDs and fully formatted input_ids ([CLS] + IDs + [SEP])
        strictly bounded within the 512-token limit.

        Args:
            text: Input text passage.
            window_size: Maximum content tokens per window (default 450, leaving margin below BERT 512).
            overlap_size: Overlapping content tokens between consecutive windows (default 50).

        Returns:
            List of dicts with keys:
            - 'sub_text': Decoded string for display and provenance.
            - 'token_count': Number of content tokens in the window.
            - 'start_token_idx': Start token offset in original passage.
            - 'end_token_idx': End token offset in original passage.
            - 'token_ids': List of content token IDs.
            - 'input_ids': Complete model input sequence with [CLS] and [SEP] (length <= window_size + 2).
        """
        clean = text.strip() if text else ""
        if not clean:
            return []

        model = self.get_model()
        tokenizer = getattr(model, "tokenizer", None)
        max_content_tokens = getattr(self, "max_sequence_length", 512) - 2

        if tokenizer is None:
            # Fallback heuristic when tokenizer is unavailable
            char_window = window_size * 4
            char_overlap = overlap_size * 4
            char_step = max(1, char_window - char_overlap)
            windows: List[Dict[str, Any]] = []
            c_start = 0
            while c_start < len(clean):
                c_end = min(c_start + char_window, len(clean))
                sub = clean[c_start:c_end]
                cnt = max(1, len(sub) // 4)
                windows.append({
                    "sub_text": sub,
                    "token_count": cnt,
                    "start_token_idx": c_start,
                    "end_token_idx": c_end,
                    "token_ids": [],
                    "input_ids": [],
                })
                if c_end >= len(clean):
                    break
                c_start += char_step
            return windows

        # Encode tokens without special tokens
        try:
            token_ids = tokenizer.encode(clean, add_special_tokens=False, verbose=False)
        except TypeError:
            token_ids = tokenizer.encode(clean, add_special_tokens=False)

        cls_id = getattr(tokenizer, "cls_token_id", 101) or 101
        sep_id = getattr(tokenizer, "sep_token_id", 102) or 102
        total_tokens = len(token_ids)

        if total_tokens <= max_content_tokens:
            return [{
                "sub_text": clean,
                "token_count": total_tokens,
                "start_token_idx": 0,
                "end_token_idx": total_tokens,
                "token_ids": token_ids,
                "input_ids": [cls_id] + token_ids + [sep_id],
            }]

        # Segment oversized sequence into overlapping token windows
        step = max(1, window_size - overlap_size)
        windows: List[Dict[str, Any]] = []
        start_idx = 0
        while start_idx < total_tokens:
            end_idx = min(start_idx + window_size, total_tokens)
            sub_token_ids = token_ids[start_idx:end_idx]
            sub_text = tokenizer.decode(
                sub_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            windows.append({
                "sub_text": sub_text,
                "token_count": len(sub_token_ids),
                "start_token_idx": start_idx,
                "end_token_idx": end_idx,
                "token_ids": sub_token_ids,
                "input_ids": [cls_id] + sub_token_ids + [sep_id],
            })
            if end_idx >= total_tokens:
                break
            start_idx += step

        return windows

    def split_into_token_windows(
        self,
        text: str,
        window_size: int = 450,
        overlap_size: int = 50,
    ) -> List[Tuple[str, int, int, int]]:
        """
        Splits text into token-bounded sub-windows to safely embed passages exceeding max_sequence_length.
        Guarantees 100% token coverage with zero content omission.

        Returns:
            List of (sub_text, token_count, start_token_idx, end_token_idx).
        """
        id_windows = self.tokenize_into_windows(text, window_size, overlap_size)
        return [
            (w["sub_text"], w["token_count"], w["start_token_idx"], w["end_token_idx"])
            for w in id_windows
        ]

    def _embed_token_batches(
        self,
        token_id_sequences: List[List[int]],
        batch_size: int,
    ) -> np.ndarray:
        """
        Embeds a list of token ID sequences directly through the model without
        decoding to text or re-tokenizing.

        Each sequence includes special tokens [CLS] and [SEP] with length <= max_seq_length.
        """
        if not token_id_sequences:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self.get_model()
        tokenizer = getattr(model, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
        if pad_id is None:
            pad_id = 0
        device = self.resolve_device()

        all_vectors: List[np.ndarray] = []

        for b_start in range(0, len(token_id_sequences), batch_size):
            b_seqs = token_id_sequences[b_start : b_start + batch_size]
            max_len = max(len(seq) for seq in b_seqs)

            batch_input_ids: List[List[int]] = []
            batch_attention_mask: List[List[int]] = []
            batch_token_type_ids: List[List[int]] = []

            for seq in b_seqs:
                if len(seq) > self.max_sequence_length:
                    raise ValueError(
                        f"Token sequence length {len(seq)} exceeds model limit {self.max_sequence_length}."
                    )
                pad_len = max_len - len(seq)
                batch_input_ids.append(seq + [pad_id] * pad_len)
                batch_attention_mask.append([1] * len(seq) + [0] * pad_len)
                batch_token_type_ids.append([0] * max_len)

            features = {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long, device=device),
                "token_type_ids": torch.tensor(batch_token_type_ids, dtype=torch.long, device=device),
            }

            with torch.no_grad():
                out = model(features)
                vecs = out["sentence_embedding"].cpu().numpy().astype(np.float32)
                all_vectors.append(vecs)

        return np.vstack(all_vectors)

    def count_tokens(self, text: str) -> int:
        """Returns the exact token count of text according to the model tokenizer."""
        model = self.get_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            try:
                encoded = tokenizer.encode(text, add_special_tokens=True, verbose=False)
                return len(encoded)
            except Exception:
                try:
                    encoded = tokenizer.encode(text, add_special_tokens=True)
                    return len(encoded)
                except Exception as exc:
                    logger.warning("Tokenizer count error: %s. Using heuristic estimate.", exc)
        # Fallback heuristic: 1 token ≈ 4 characters
        return max(1, len(text) // 4)

    def embed_text(self, text: str) -> List[float]:
        """
        Generates a normalized embedding vector for a single text input.
        If the text exceeds model sequence length, it is automatically segmented into
        overlapping sub-windows, embedded without truncation, and token-weighted pooled.

        Returns:
            384-dimensional list of floats.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            raise ValueError("Cannot generate embedding for empty text content.")

        embeddings, _ = self.embed_batch([clean_text])
        if not embeddings:
            raise RuntimeError("Embedding generation returned empty output.")
        return embeddings[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Generates normalized embedding vectors for a list of text passages in batches.
        Oversized chunks are split into sub-windows, passed directly as token IDs,
        and pooled weighted by sub-window token counts.

        Returns:
            Tuple of:
            - List of 384-dimensional float vectors
            - Metrics dictionary (duration_ms, throughput_chunks_per_sec, oversized_chunks,
              token_counts, sub_chunk_counts, chunk_details)
        """
        if not texts:
            return [], {
                "duration_ms": 0.0,
                "throughput_chunks_per_sec": 0.0,
                "total_chunks": 0,
                "total_sub_windows": 0,
                "oversized_chunks": [],
                "token_counts": [],
                "sub_chunk_counts": [],
                "chunk_details": [],
            }

        # Validate non-emptiness
        for idx, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Cannot generate embedding for empty text content at index {idx}.")

        model = self.get_model()
        tokenizer = getattr(model, "tokenizer", None)
        eff_batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        # 1. Segment chunks into token-bounded windows with direct token IDs
        chunk_windows: List[List[Dict[str, Any]]] = []
        token_counts: List[int] = []
        sub_chunk_counts: List[int] = []
        oversized_chunks: List[int] = []

        all_input_ids: List[List[int]] = []
        all_sub_texts: List[str] = []
        sub_index_map: List[List[int]] = []

        for idx, text in enumerate(texts):
            windows = self.tokenize_into_windows(text)
            actual_tc = self.count_tokens(text)
            token_counts.append(actual_tc)
            sub_chunk_counts.append(len(windows))
            chunk_windows.append(windows)

            if len(windows) > 1:
                oversized_chunks.append(idx)
                logger.info(
                    "Chunk #%d is oversized (%d tokens > %d max). "
                    "Segmented into %d overlapping token-bounded sub-windows for lossless embedding.",
                    idx,
                    actual_tc,
                    self.max_sequence_length,
                    len(windows),
                )

            chunk_map = []
            for w in windows:
                chunk_map.append(len(all_input_ids))
                all_input_ids.append(w["input_ids"])
                all_sub_texts.append(w["sub_text"])
            sub_index_map.append(chunk_map)

        # 2. Run batched model inference directly with token IDs
        start_time = time.perf_counter()
        try:
            if tokenizer is not None and all_input_ids and all_input_ids[0]:
                raw_sub_embeddings = self._embed_token_batches(
                    all_input_ids,
                    batch_size=eff_batch_size,
                )
            else:
                raw_sub_embeddings = model.encode(
                    all_sub_texts,
                    batch_size=eff_batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=False,
                    convert_to_numpy=True,
                )
        except Exception as exc:
            logger.error("Embedding inference failed on batch of %d sub-windows: %s", len(all_input_ids), exc, exc_info=True)
            raise RuntimeError(f"Embedding inference failed: {exc}") from exc

        duration_sec = time.perf_counter() - start_time
        duration_ms = duration_sec * 1000
        throughput = len(texts) / duration_sec if duration_sec > 0 else float(len(texts))

        # 3. Aggregate sub-vectors per chunk with token-weighted pooling and L2 normalization
        vector_list: List[List[float]] = []
        chunk_details: List[Dict[str, Any]] = []

        for idx, windows in enumerate(chunk_windows):
            chunk_sub_indices = sub_index_map[idx]
            sub_vecs = raw_sub_embeddings[chunk_sub_indices]
            sub_token_weights = np.array([float(w["token_count"]) for w in windows], dtype=np.float32)

            if len(windows) == 1:
                composite = sub_vecs[0]
            else:
                total_w = sub_token_weights.sum()
                weights = sub_token_weights / total_w if total_w > 0 else np.ones(len(windows)) / len(windows)
                composite = np.zeros(self.dimension, dtype=np.float32)
                for w_idx in range(len(windows)):
                    composite += weights[w_idx] * sub_vecs[w_idx]

            # Apply L2 unit-norm normalization
            norm = np.linalg.norm(composite)
            if norm > 0:
                final_vec = composite / norm
            else:
                final_vec = composite

            dim = len(final_vec)
            if dim != self.dimension:
                raise ValueError(f"Embedding dimensionality mismatch: model produced {dim}d, expected {self.dimension}d.")

            vector_list.append([float(val) for val in final_vec])
            chunk_details.append({
                "chunk_index": idx,
                "token_count": token_counts[idx],
                "sub_chunk_count": len(windows),
                "is_oversized": len(windows) > 1,
                "sub_chunks": [
                    {
                        "sub_index": s_i,
                        "token_count": w["token_count"],
                        "token_start": w["start_token_idx"],
                        "token_end": w["end_token_idx"],
                    }
                    for s_i, w in enumerate(windows)
                ],
            })

        logger.info(
            "Generated %d embeddings (%dd) across %d sub-windows in %.1f ms (%.1f chunks/sec, %d oversized)",
            len(texts),
            self.dimension,
            len(all_sub_texts),
            duration_ms,
            throughput,
            len(oversized_chunks),
        )

        metrics: Dict[str, Any] = {
            "duration_ms": round(duration_ms, 2),
            "throughput_chunks_per_sec": round(throughput, 1),
            "total_chunks": len(texts),
            "total_sub_windows": len(all_sub_texts),
            "oversized_chunks": oversized_chunks,
            "token_counts": token_counts,
            "sub_chunk_counts": sub_chunk_counts,
            "chunk_details": chunk_details,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self._device or "cpu",
        }

        return vector_list, metrics

    def get_info(self) -> Dict[str, Any]:
        """Provides metadata about active model and execution configuration."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": self.max_sequence_length,
            "device": self._device or self.resolve_device(),
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
            "normalize": settings.EMBEDDING_NORMALIZE,
            "metric": "cosine",
            "is_loaded": self._model is not None,
            "description": "Sentence-Transformers dense semantic embeddings optimized for English document search.",
        }


embedding_service = EmbeddingService()
