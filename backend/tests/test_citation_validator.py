"""Unit tests for CitationValidator, strict structured output validation, and provenance verification."""

import json
from typing import Optional
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.rag.citation_validator import (
    CitationValidator,
    RAGResponseValidationError,
    ValidatedCitation,
)
from app.services.rag.context_builder import ContextSource


def make_context_source(
    source_id: str,
    filename: str = "report.pdf",
    page: int = 1,
    chunk_id: Optional[uuid.UUID] = None,
    document_id: Optional[uuid.UUID] = None,
) -> ContextSource:
    return ContextSource(
        source_id=source_id,
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        filename=filename,
        page_start=page,
        page_end=page,
        text=f"Text content for passage {source_id}.",
        token_count=15,
        dense_score=0.88,
        bm25_score=4.5,
        rrf_score=0.03,
        reranker_score=3.2,
        final_rank=1,
    )


def test_structured_json_parsing_and_valid_citations():
    """Verify clean parsing of valid structured output with accurate citations."""
    s1 = make_context_source("S1", "vector_guide.pdf", page=2)
    s2 = make_context_source("S2", "nlp_handbook.pdf", page=10)
    source_map = {"S1": s1, "S2": s2}

    output = json.dumps({
        "answer": "Hybrid search blends dense and sparse vectors [S1]. It uses reciprocal rank fusion [S2].",
        "is_answered": True,
        "claims": [
            {
                "claim_text": "Hybrid search blends dense and sparse vectors.",
                "citation_ids": ["S1"],
            },
            {
                "claim_text": "It uses reciprocal rank fusion.",
                "citation_ids": ["S2"],
            },
        ],
        "insufficient_evidence_reason": None,
    })

    res = CitationValidator.parse_and_validate(output, source_map)
    assert res.is_answered is True
    assert res.parsing_status == "valid_structured"
    assert len(res.claims) == 2
    assert len(res.citations) == 2
    assert len(res.invalid_citation_ids) == 0

    # Verify claim 1 support status
    assert res.claims[0].is_supported is True
    assert res.claims[0].citation_ids == ["S1"]
    assert res.claims[0].citations[0].source_id == "S1"

    # Verify provenance attached to S1
    cit1 = res.citations[0]
    assert cit1.source_id == "S1"
    assert cit1.filename == "vector_guide.pdf"
    assert cit1.page_start == 2
    assert cit1.chunk_id == s1.chunk_id


def test_rejection_of_hallucinated_citations():
    """Verify that citations to non-existent source identifiers are strictly rejected and flagged."""
    s1 = make_context_source("S1")
    source_map = {"S1": s1}  # Only S1 is provided

    output = json.dumps({
        "answer": "Document AI extracts facts from text [S1]. Quantum teleportation is also discussed [S99].",
        "is_answered": True,
        "claims": [
            {
                "claim_text": "Document AI extracts facts from text.",
                "citation_ids": ["S1"],
            },
            {
                "claim_text": "Quantum teleportation is also discussed.",
                "citation_ids": ["S99"],
            },
        ],
        "insufficient_evidence_reason": None,
    })

    res = CitationValidator.parse_and_validate(output, source_map)
    assert len(res.citations) == 1
    assert res.citations[0].source_id == "S1"
    assert "S99" in res.invalid_citation_ids

    # Claim 1 has valid citation S1 -> is_supported = True
    assert res.claims[0].is_supported is True
    assert res.claims[0].citation_ids == ["S1"]

    # Claim 2 cited only S99 (invalid) -> is_supported = False, unsupported_citation_ids = ['S99']
    assert res.claims[1].is_supported is False
    assert res.claims[1].citation_ids == []
    assert res.claims[1].unsupported_citation_ids == ["S99"]


def test_markdown_code_fenced_json_parsing():
    """Verify that JSON wrapped in markdown ```json ... ``` blocks parses cleanly and records repair status."""
    s1 = make_context_source("S1")
    source_map = {"S1": s1}

    fenced_output = """```json
    {
        "answer": "A factual statement [S1].",
        "is_answered": true,
        "claims": [
            {
                "claim_text": "A factual statement.",
                "citation_ids": ["S1"]
            }
        ],
        "insufficient_evidence_reason": null
    }
    ```"""

    res = CitationValidator.parse_and_validate(fenced_output, source_map)
    assert res.is_answered is True
    assert res.parsing_status == "recovered_with_repair"
    assert len(res.citations) == 1
    assert res.citations[0].source_id == "S1"


def test_trailing_comma_repair():
    """Verify bounded repair of JSON containing trailing commas."""
    s1 = make_context_source("S1")
    source_map = {"S1": s1}

    # JSON with invalid trailing comma in object and array
    trailing_comma_output = """
    {
        "answer": "Statement with trailing commas [S1].",
        "is_answered": true,
        "claims": [
            {
                "claim_text": "Statement with trailing commas.",
                "citation_ids": ["S1", ],
            },
        ],
        "insufficient_evidence_reason": null,
    }
    """

    res = CitationValidator.parse_and_validate(trailing_comma_output, source_map)
    assert res.is_answered is True
    assert res.parsing_status == "recovered_with_repair"
    assert len(res.claims) == 1
    assert res.claims[0].citation_ids == ["S1"]


def test_plain_text_without_json_raises_validation_error():
    """Verify that raw plain text without structured JSON is strictly rejected rather than accepted."""
    s1 = make_context_source("S1")
    source_map = {"S1": s1}

    plain_output = "According to the documentation [S1], embedding dimension is 384."

    with pytest.raises(RAGResponseValidationError) as exc_info:
        CitationValidator.parse_and_validate(plain_output, source_map)

    assert "could not be parsed as valid JSON" in str(exc_info.value)


def test_empty_or_whitespace_output_raises_validation_error():
    """Verify that empty output raises a structured validation error."""
    source_map = {"S1": make_context_source("S1")}

    with pytest.raises(RAGResponseValidationError) as exc_info:
        CitationValidator.parse_and_validate("   ", source_map)

    assert "empty response output" in str(exc_info.value)


def test_missing_claims_field_raises_validation_error():
    """Verify that response lacking 'claims' array is rejected."""
    source_map = {"S1": make_context_source("S1")}
    output = json.dumps({
        "answer": "Answer text without claims.",
        "is_answered": True,
    })

    with pytest.raises(RAGResponseValidationError) as exc_info:
        CitationValidator.parse_and_validate(output, source_map)

    assert "missing required 'claims' array" in str(exc_info.value)


def test_malformed_claim_items_raise_validation_error():
    """Verify that claims with non-dict elements or missing claim_text raise validation errors."""
    source_map = {"S1": make_context_source("S1")}

    # Case A: claims contains strings instead of objects
    output_a = json.dumps({
        "answer": "Answer text.",
        "is_answered": True,
        "claims": ["string claim not an object"],
    })
    with pytest.raises(RAGResponseValidationError) as exc_info:
        CitationValidator.parse_and_validate(output_a, source_map)
    assert "must be a JSON object" in str(exc_info.value)

    # Case B: claim lacks claim_text
    output_b = json.dumps({
        "answer": "Answer text.",
        "is_answered": True,
        "claims": [{"citation_ids": ["S1"]}],
    })
    with pytest.raises(RAGResponseValidationError) as exc_info:
        CitationValidator.parse_and_validate(output_b, source_map)
    assert "missing non-empty 'claim_text'" in str(exc_info.value)

    # Case C: citation_ids is not a list
    output_c = json.dumps({
        "answer": "Answer text.",
        "is_answered": True,
        "claims": [{"claim_text": "Valid text", "citation_ids": "not_a_list"}],
    })
    with pytest.raises(RAGResponseValidationError) as exc_info:
        CitationValidator.parse_and_validate(output_c, source_map)
    assert "must be a list of source tags" in str(exc_info.value)


def test_explicit_abstention_parsing():
    """Verify honest abstention handling."""
    source_map = {"S1": make_context_source("S1")}

    output = json.dumps({
        "answer": "The supplied documents do not contain information regarding server deployment on Kubernetes.",
        "is_answered": False,
        "claims": [],
        "insufficient_evidence_reason": "No passages mention Kubernetes.",
    })

    res = CitationValidator.parse_and_validate(output, source_map)
    assert res.is_answered is False
    assert res.parsing_status == "valid_abstention"
    assert res.insufficient_evidence_reason == "No passages mention Kubernetes."
    assert len(res.claims) == 0
    assert len(res.citations) == 0


def test_database_persistence_verification_purges_deleted_chunk():
    """Verify that verify_chunk_persistence drops concurrently deleted chunks from citations."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    doc_id = uuid.uuid4()
    chunk1_id = uuid.uuid4()
    chunk2_id = uuid.uuid4()

    doc = Document(
        id=doc_id,
        filename="test.pdf",
        original_filename="test.pdf",
        file_type="pdf",
        file_size=100,
        storage_path="/path/test.pdf",
        status="completed",
        total_pages=1,
        total_characters=100,
        total_chunks=1,
        embedding_status="completed",
    )
    session.add(doc)

    # Only chunk1 exists in DB, chunk2 was deleted/not inserted
    chunk1 = Chunk(
        id=chunk1_id,
        document_id=doc_id,
        chunk_index=0,
        text="Persisted chunk content.",
        character_count=23,
    )
    session.add(chunk1)
    session.commit()

    s1 = make_context_source("S1", chunk_id=chunk1_id, document_id=doc_id)
    s2 = make_context_source("S2", chunk_id=chunk2_id, document_id=doc_id)
    source_map = {"S1": s1, "S2": s2}

    output = json.dumps({
        "answer": "Claim supported by S1 [S1]. Claim supported by deleted S2 [S2].",
        "is_answered": True,
        "claims": [
            {"claim_text": "Claim 1", "citation_ids": ["S1"]},
            {"claim_text": "Claim 2", "citation_ids": ["S2"]},
        ],
        "insufficient_evidence_reason": None,
    })

    raw_res = CitationValidator.parse_and_validate(output, source_map)
    assert len(raw_res.citations) == 2

    # Run persistence check
    verified_res = CitationValidator.verify_chunk_persistence(session, raw_res)

    # S2 must be purged because it doesn't exist in DB
    assert len(verified_res.citations) == 1
    assert verified_res.citations[0].source_id == "S1"
    assert "S2" in verified_res.invalid_citation_ids

    # Claim 1 remains supported
    assert verified_res.claims[0].is_supported is True
    assert verified_res.claims[0].citation_ids == ["S1"]

    # Claim 2 is now marked unsupported
    assert verified_res.claims[1].is_supported is False
    assert verified_res.claims[1].citation_ids == []
    assert "S2" in verified_res.claims[1].unsupported_citation_ids
