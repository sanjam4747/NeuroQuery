"""Unit tests for ContextBuilder, token budgeting, and passage deduplication."""

import uuid
import pytest
from app.schemas.retrieval import RetrievedChunkResult
from app.services.rag.context_builder import (
    ContextBuilder,
    compute_word_jaccard_overlap,
    estimate_token_count,
)


def make_chunk_result(
    text: str,
    final_rank: int = 1,
    page_start: int = 1,
    page_end: int = 1,
    filename: str = "doc1.pdf",
) -> RetrievedChunkResult:
    return RetrievedChunkResult(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        text=text,
        character_count=len(text),
        page_start=page_start,
        page_end=page_end,
        rrf_score=0.03,
        rrf_rank=final_rank,
        reranker_score=5.0 - final_rank * 0.1,
        final_rank=final_rank,
    )


def test_token_estimation():
    """Verify conservative token estimation heuristic."""
    assert estimate_token_count("") == 0
    short_text = "This is a short sentence."
    tokens = estimate_token_count(short_text)
    assert 5 <= tokens <= 10

    long_text = "word " * 100
    long_tokens = estimate_token_count(long_text)
    assert 100 <= long_tokens <= 150


def test_passage_deduplication():
    """Verify near-identical passages are filtered out."""
    p1 = "Reciprocal rank fusion combines rankings from multiple retrieval systems using the formula 1 over 60 plus rank."
    p2 = "Reciprocal rank fusion combines rankings from multiple retrieval systems using the formula 1 over 60 plus rank."  # Exact duplicate
    p3 = "Dense vector search computes cosine distance between 384-dimensional dense vectors."

    builder = ContextBuilder(max_context_tokens=1000, max_evidence_chunks=5)
    chunks = [
        make_chunk_result(p1, final_rank=1),
        make_chunk_result(p2, final_rank=2),
        make_chunk_result(p3, final_rank=3),
    ]

    res = builder.build_context(chunks)
    assert len(res.sources) == 2  # p2 was deduplicated
    assert res.deduplicated_count == 1
    assert res.sources[0].source_id == "S1"
    assert res.sources[1].source_id == "S2"


def test_token_budgeting_omission():
    """Verify that chunks exceeding the token budget are omitted."""
    # Create very low token budget
    builder = ContextBuilder(max_context_tokens=150, max_evidence_chunks=5)

    c1 = make_chunk_result("Short passage that easily fits inside the context budget.", final_rank=1)
    c2 = make_chunk_result("Another huge paragraph " + "filled with repetition " * 40, final_rank=2)
    c3 = make_chunk_result("Third small passage.", final_rank=3)

    res = builder.build_context([c1, c2, c3], reserved_overhead_tokens=50)
    assert len(res.sources) >= 1
    assert res.omitted_due_to_budget_count >= 1
    assert res.total_context_tokens <= 150


def test_source_identifier_mapping():
    """Verify source IDs map directly to chunk provenance."""
    doc_id = uuid.uuid4()
    c1 = make_chunk_result("First evidence passage.", final_rank=1, page_start=3, page_end=4, filename="guide.pdf")
    c1.document_id = doc_id

    builder = ContextBuilder(max_context_tokens=1000)
    res = builder.build_context([c1])

    assert "S1" in res.source_map
    src = res.source_map["S1"]
    assert src.chunk_id == c1.chunk_id
    assert src.document_id == doc_id
    assert src.filename == "guide.pdf"
    assert src.page_start == 3
    assert src.page_end == 4
    assert '<document_evidence id="S1" filename="guide.pdf" pages="3-4">' in res.formatted_context


def test_partially_overlapping_passages_with_different_facts_preserved():
    """Verify deduplication does NOT discard overlapping passages that contain unique factual information."""
    p1 = "The Apollo 11 mission launched on July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins."
    # p2 shares the launch intro but adds the unique landing site and module facts
    p2 = "The Apollo 11 mission launched on July 16, 1969, carrying astronauts who landed Lunar Module Eagle in the Sea of Tranquility on July 20."

    builder = ContextBuilder(max_context_tokens=1000, max_evidence_chunks=5, overlap_threshold=0.5)
    chunks = [
        make_chunk_result(p1, final_rank=1),
        make_chunk_result(p2, final_rank=2),
    ]

    res = builder.build_context(chunks)
    # Both passages must be preserved because p2 contains unique factual tokens
    assert len(res.sources) == 2
    assert res.deduplicated_count == 0
    assert res.sources[0].source_id == "S1"
    assert res.sources[1].source_id == "S2"


def test_redundant_passage_without_new_facts_discarded():
    """Verify that a passage with near-identical phrasing and no new facts is filtered out."""
    p1 = "The Apollo 11 mission launched on July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins to space."
    p2 = "The Apollo 11 mission launched on July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins."

    builder = ContextBuilder(max_context_tokens=1000, max_evidence_chunks=5, overlap_threshold=0.7)
    chunks = [
        make_chunk_result(p1, final_rank=1),
        make_chunk_result(p2, final_rank=2),
    ]

    res = builder.build_context(chunks)
    assert len(res.sources) == 1
    assert res.deduplicated_count == 1


def test_empty_evidence_handling():
    """Verify behavior when no evidence chunks are provided."""
    builder = ContextBuilder(max_context_tokens=1000)
    res = builder.build_context([])

    assert len(res.sources) == 0
    assert res.formatted_context == ""
    assert res.total_context_tokens == 0
    assert res.deduplicated_count == 0
    assert res.omitted_due_to_budget_count == 0


def test_output_reservation_1024_tokens_enforced():
    """Verify that a 1,024-token output reservation is strictly subtracted from total context."""
    # System prompt ~8 tokens, question ~5 tokens, output 1024, buffer 100 -> overhead = 1137
    # With max_context_tokens=1170, remaining evidence budget is 33 tokens.
    builder = ContextBuilder(max_context_tokens=1170, max_evidence_chunks=5)

    c1 = make_chunk_result("This text is roughly 15 tokens.", final_rank=1)
    c2 = make_chunk_result("This second text is roughly 30 tokens and should not fit into remaining budget.", final_rank=2)

    res = builder.build_context(
        [c1, c2],
        question="What is the answer?",
        system_prompt="You are a helpful assistant.",
        max_output_tokens=1024,
    )

    # Output reservation of 1024 tokens was strictly enforced, allowing only c1 or omitting c2
    assert res.omitted_due_to_budget_count >= 1
    assert res.total_context_tokens <= 1400


def test_extremely_long_question_exhausts_budget():
    """Verify that an extremely long question exhausts budget cleanly and selects 0 chunks rather than overflowing."""
    builder = ContextBuilder(max_context_tokens=2048, max_evidence_chunks=5)

    # 3000-word question (exceeding total context tokens)
    long_q = "what is the impact of " * 600
    c1 = make_chunk_result("Valuable evidence chunk.", final_rank=1)

    res = builder.build_context(
        [c1],
        question=long_q,
        system_prompt="You are a helpful assistant.",
        max_output_tokens=1024,
    )

    # Budget exhausted by long question + 1024 output reservation
    assert len(res.sources) == 0
    assert res.formatted_context == ""

