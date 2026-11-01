"""Evidence selection, passage deduplication, and token-aware context budgeting."""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set
import uuid

from app.core.config import settings
from app.core.logging import logger
from app.schemas.retrieval import RetrievedChunkResult


@dataclass
class ContextSource:
    """Provenance-rich metadata for an evidence chunk selected into the context window."""

    source_id: str  # e.g. "S1", "S2"
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    token_count: int
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    reranker_score: Optional[float] = None
    final_rank: int = 1


@dataclass
class ContextBuildResult:
    """Outcome of token budgeting, evidence selection, and context assembly."""

    formatted_context: str
    sources: List[ContextSource]
    source_map: Dict[str, ContextSource]
    total_context_tokens: int
    budget_limit_tokens: int
    deduplicated_count: int
    omitted_due_to_budget_count: int


# Functional stopwords for informational overlap filtering
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "from", "in", "into", "of", "off", "on", "onto",
    "out", "over", "to", "up", "with", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "may", "might", "must", "it", "its",
    "this", "that", "these", "those", "there", "their", "they",
}


def estimate_token_count(text: str) -> int:
    """
    Conservative token estimation for LLM context budgeting.
    Standard English text averages ~3.8 to 4.0 characters per token for modern subword tokenizers
    (e.g. Qwen2.5 BPE, Llama 3 SentencePiece, BERT). We use 3.8 to prevent overfilling the model's
    configured context window.

    Explicit Limitations:
    1. Heuristic accuracy: Calibrated for English prose and technical terminology.
    2. Non-Latin scripts: For CJK, Cyrillic, Arabic, or Devanagari text, tokenizers may emit
       1-2 chars/token, meaning this heuristic will underestimate token consumption.
    3. Dense code/data: High-frequency punctuation and hex hashes can produce more tokens.
    4. Exactness: If exact token-level capacity is required, the target model's native
       tokenizer should be integrated.
    """
    if not text:
        return 0
    cleaned = re.sub(r"\s+", " ", text).strip()
    return max(1, int(len(cleaned) / 3.8))


def compute_word_jaccard_overlap(text1: str, text2: str) -> float:
    """Computes Jaccard word set similarity between two passages to identify near-duplicates."""
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def extract_content_words(text: str) -> Set[str]:
    """Extracts non-stopword content tokens from a text passage."""
    raw_words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", text.lower())
    return {w for w in raw_words if w not in STOPWORDS}


class ContextBuilder:
    """
    Constructs an optimized, bounded, provenance-preserving context block
    from reranked retrieval candidates.
    """

    def __init__(
        self,
        max_context_tokens: Optional[int] = None,
        max_evidence_chunks: Optional[int] = None,
        overlap_threshold: Optional[float] = None,
        max_evidence_tokens: Optional[int] = None,
    ):
        # max_context_tokens represents the overall model context capacity (default: settings.LLM_CONTEXT_WINDOW_TOKENS)
        self.max_context_tokens = max_context_tokens or settings.LLM_CONTEXT_WINDOW_TOKENS
        self.max_evidence_chunks = max_evidence_chunks or settings.RAG_MAX_EVIDENCE_CHUNKS
        self.overlap_threshold = overlap_threshold or settings.RAG_SIMILARITY_OVERLAP_THRESHOLD
        # max_evidence_tokens represents an explicit cap on evidence passages (default: settings.RAG_CONTEXT_MAX_TOKENS)
        self.max_evidence_tokens = max_evidence_tokens or settings.RAG_CONTEXT_MAX_TOKENS

    def build_context(
        self,
        retrieved_chunks: List[RetrievedChunkResult],
        question: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        reserved_overhead_tokens: Optional[int] = None,
    ) -> ContextBuildResult:
        """
        Selects top relevance-ordered chunks, filters redundant near-duplicates while preserving
        unique facts, bounds token consumption against model context capacity, and assigns stable
        source identifiers ([S1], [S2], ...).

        Args:
            retrieved_chunks: Ranked chunks from hybrid retrieval and reranking.
            question: Optional user question text for precise overhead accounting.
            system_prompt: Optional system prompt text for precise overhead accounting.
            max_output_tokens: Optional output token reservation (defaults to LLM_MAX_OUTPUT_TOKENS).
            reserved_overhead_tokens: Explicit total overhead override if provided.

        Returns:
            ContextBuildResult containing formatted evidence block and provenance mapping.
        """
        # Calculate dynamic overhead budget accounting for system instructions, question, and output reservation
        if reserved_overhead_tokens is not None:
            overhead = reserved_overhead_tokens
        elif question is not None or system_prompt is not None or max_output_tokens is not None:
            sys_tokens = estimate_token_count(system_prompt) if system_prompt else 350
            q_tokens = estimate_token_count(question) if question else 50
            out_tokens = max_output_tokens if max_output_tokens is not None else settings.LLM_MAX_OUTPUT_TOKENS
            safety_buffer = 100  # For XML tags, source formatting, and schema instructions
            overhead = sys_tokens + q_tokens + out_tokens + safety_buffer
        else:
            # Standalone invocation without prompt context: assume overhead is 0 so max_context_tokens is treated directly as the evidence budget
            overhead = 0

        available_budget = max(0, self.max_context_tokens - overhead)
        effective_budget = min(available_budget, self.max_evidence_tokens)

        if effective_budget == 0:
            logger.warning(
                "Context budget exhausted by prompt overhead alone (max=%d, overhead=%d). Zero evidence selected.",
                self.max_context_tokens,
                overhead,
            )

        sources: List[ContextSource] = []
        source_map: Dict[str, ContextSource] = {}
        accepted_texts: List[str] = []
        accepted_content_words: Set[str] = set()
        current_tokens = 0
        dedup_count = 0
        omitted_budget_count = 0

        # Iterate in reranked relevance order
        for idx, chunk in enumerate(retrieved_chunks):
            if len(sources) >= self.max_evidence_chunks or effective_budget == 0:
                break

            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            # 1. Deduplication check:
            # - Exact identical text is always skipped.
            # - For high Jaccard overlap (>= overlap_threshold), we inspect content words.
            #   If the candidate contains >= 15% new content words not present in accepted passages,
            #   we preserve it because it carries distinct factual information.
            is_duplicate = False
            chunk_content = extract_content_words(chunk_text)

            for prev_text in accepted_texts:
                clean_prev = re.sub(r"\s+", " ", prev_text).strip()
                clean_curr = re.sub(r"\s+", " ", chunk_text).strip()

                if clean_curr == clean_prev:
                    is_duplicate = True
                    break

                overlap = compute_word_jaccard_overlap(chunk_text, prev_text)
                if overlap >= self.overlap_threshold:
                    # Check whether current chunk contains new factual information
                    new_words = chunk_content - accepted_content_words
                    unique_fraction = len(new_words) / max(1, len(chunk_content))
                    if unique_fraction < 0.15:
                        is_duplicate = True
                        break

            if is_duplicate:
                dedup_count += 1
                logger.debug("Deduplicated redundant chunk %s (rank %d)", chunk.chunk_id, chunk.final_rank)
                continue

            # 2. Token budgeting check
            chunk_tokens = estimate_token_count(chunk_text)
            source_overhead_tokens = 25  # Overhead for <document_evidence id=...> framing
            projected_total = current_tokens + chunk_tokens + source_overhead_tokens

            if projected_total > effective_budget:
                omitted_budget_count += 1
                logger.info(
                    "Chunk %s omitted to respect context budget (projected=%d, budget=%d)",
                    chunk.chunk_id,
                    projected_total,
                    effective_budget,
                )
                continue

            # 3. Assign stable source identifier [S1], [S2], ...
            source_idx = len(sources) + 1
            source_id = f"S{source_idx}"

            src = ContextSource(
                source_id=source_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk_text,
                token_count=chunk_tokens,
                dense_score=chunk.dense_score,
                bm25_score=chunk.bm25_score,
                rrf_score=chunk.rrf_score,
                reranker_score=chunk.reranker_score,
                final_rank=chunk.final_rank,
            )

            sources.append(src)
            source_map[source_id] = src
            accepted_texts.append(chunk_text)
            accepted_content_words.update(chunk_content)
            current_tokens += chunk_tokens + source_overhead_tokens

        # 4. Assemble grounded XML-structured context block
        context_blocks: List[str] = []
        for src in sources:
            page_info = ""
            if src.page_start is not None:
                if src.page_end and src.page_end != src.page_start:
                    page_info = f" pages=\"{src.page_start}-{src.page_end}\""
                else:
                    page_info = f" page=\"{src.page_start}\""

            block = (
                f'<document_evidence id="{src.source_id}" filename="{src.filename}"{page_info}>\n'
                f"{src.text}\n"
                f"</document_evidence>"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        logger.info(
            "Context constructed: %d evidence chunks selected (%d tokens, budget=%d, deduped=%d, budget_omitted=%d)",
            len(sources),
            current_tokens,
            effective_budget,
            dedup_count,
            omitted_budget_count,
        )

        return ContextBuildResult(
            formatted_context=formatted_context,
            sources=sources,
            source_map=source_map,
            total_context_tokens=current_tokens,
            budget_limit_tokens=effective_budget,
            deduplicated_count=dedup_count,
            omitted_due_to_budget_count=omitted_budget_count,
        )


# Singleton instance
context_builder = ContextBuilder()
