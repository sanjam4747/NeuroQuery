"""Tests for document processing pipeline orchestration and cascading lifecycle."""

from pathlib import Path
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.pipeline import pipeline_service
from app.services.storage import storage_service
from tests.conftest import make_docx, make_pdf, make_txt


def test_pipeline_ingest_pdf_success(test_db: Session, tmp_path: Path):
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path
    try:
        pdf_bytes = make_pdf(["NeuroQuery Ingestion Page 1", "NeuroQuery Ingestion Page 2"])
        doc = pipeline_service.ingest_document(test_db, pdf_bytes, "pipeline_test.pdf")

        assert doc.status == "completed"
        assert doc.file_type == "pdf"
        assert doc.total_pages == 2
        assert doc.total_chunks > 0
        assert doc.total_characters > 0

        # Verify DB records
        chunks = test_db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()
        assert len(chunks) == doc.total_chunks

        # Verify physical file existence
        file_path = storage_service.get_file_path(doc.storage_path)
        assert file_path.exists()
    finally:
        storage_service.base_dir = orig_base


def test_pipeline_ingest_failure_marks_failed(test_db: Session, tmp_path: Path):
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path
    try:
        # A blank PDF has no extractable text, which triggers pipeline error
        pdf_bytes = make_pdf([""])
        doc = pipeline_service.ingest_document(test_db, pdf_bytes, "blank.pdf")

        assert doc.status == "failed"
        assert doc.error_message is not None
        assert "no extractable text was found" in doc.error_message

        # No orphaned chunks
        chunks = test_db.scalars(select(Chunk).where(Chunk.document_id == doc.id)).all()
        assert len(chunks) == 0
    finally:
        storage_service.base_dir = orig_base


def test_pipeline_delete_cascades_and_removes_file(test_db: Session, tmp_path: Path):
    orig_base = storage_service.base_dir
    storage_service.base_dir = tmp_path
    try:
        txt_bytes = make_txt("Testing deletion and cascading behavior in NeuroQuery.")
        doc = pipeline_service.ingest_document(test_db, txt_bytes, "delete_me.txt")
        doc_id = doc.id
        storage_path = doc.storage_path

        file_path = storage_service.get_file_path(storage_path)
        assert file_path.exists()

        # Delete document
        pipeline_service.delete_document(test_db, doc)

        # Verify document is gone
        assert test_db.get(Document, doc_id) is None

        # Verify cascading deletion of chunks
        chunks = test_db.scalars(select(Chunk).where(Chunk.document_id == doc_id)).all()
        assert len(chunks) == 0

        # Verify file is deleted from disk
        assert not file_path.exists()
    finally:
        storage_service.base_dir = orig_base
