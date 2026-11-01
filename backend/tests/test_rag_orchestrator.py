"""Integration tests for RAGOrchestrator pipeline."""

import uuid
import pytest
from sqlalchemy.orm import Session
from app.models.chunk import Chunk
from app.models.document import Document
from app.schemas.rag import RAGQueryRequest
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag.orchestrator import RAGOrchestrator


@pytest.fixture
def rag_test_corpus(test_db: Session):
    """Creates a sample document and chunk for testing the end-to-end RAG orchestrator."""
    doc = Document(
        id=uuid.uuid4(),
        filename="rag_test_doc.txt",
        original_filename="rag_test_doc.txt",
        file_type="txt",
        file_size=500,
        storage_path="storage/test.txt",
        status="completed",
    )
    test_db.add(doc)
    test_db.flush()

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=0,
        text="NeuroQuery uses BAAI bge-small-en-v1.5 to generate 384-dimensional dense semantic embeddings.",
        character_count=92,
        page_start=1,
        page_end=1,
        embedding=[0.05] * 384,
    )
    test_db.add(chunk)
    test_db.commit()
    test_db.refresh(doc)
    test_db.refresh(chunk)
    return doc, chunk


@pytest.mark.asyncio
async def test_rag_orchestrator_success(test_db: Session, rag_test_corpus):
    """Verify full end-to-end RAG question answering pipeline."""
    doc, chunk = rag_test_corpus
    mock_provider = MockLLMProvider()
    orchestrator = RAGOrchestrator(llm_provider=mock_provider)

    req = RAGQueryRequest(
        question="What embedding model and dimension does NeuroQuery use?",
        document_ids=[doc.id],
        top_k=5,
    )

    response = await orchestrator.answer_question(db=test_db, request=req)
    assert response.question == req.question
    assert response.is_answered is True
    assert len(response.answer) > 0
    assert len(response.evidence_sources) >= 1
    assert response.diagnostics.total_latency_ms > 0
    assert response.generation_metadata.provider == "mock"


@pytest.mark.asyncio
async def test_rag_orchestrator_empty_corpus_abstention(test_db: Session):
    """Verify early abstention when no candidate documents exist in database."""
    mock_provider = MockLLMProvider()
    orchestrator = RAGOrchestrator(llm_provider=mock_provider)

    # Search against non-existent document ID
    req = RAGQueryRequest(
        question="What is the meaning of life?",
        document_ids=[uuid.uuid4()],
        top_k=5,
    )

    response = await orchestrator.answer_question(db=test_db, request=req)
    assert response.is_answered is False
    assert "No relevant document passages were found" in response.answer
    # Fast path: LLM was not even invoked
    assert mock_provider.call_count == 0


@pytest.mark.asyncio
async def test_rag_orchestrator_validation():
    """Verify orchestrator rejects empty query."""
    orchestrator = RAGOrchestrator(llm_provider=MockLLMProvider())
    with pytest.raises(Exception):
        RAGQueryRequest(question="   ")
