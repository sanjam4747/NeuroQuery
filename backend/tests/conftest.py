"""Pytest fixtures and isolated testing configuration."""

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, List
import docx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure backend and project root are in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(backend_dir)
for p in [backend_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services.storage import storage_service


@pytest.fixture
def test_db() -> Generator[Session, None, None]:
    """Isolated in-memory SQLite database session for unit and integration tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(test_db: Session, tmp_path: Path) -> Generator[TestClient, None, None]:
    """Test client fixture with overridden database and isolated storage directory."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    orig_base_dir = storage_service.base_dir
    storage_service.base_dir = tmp_path

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    storage_service.base_dir = orig_base_dir


# ==============================================================================
# Sample Document Generators for Synthetic Testing
# ==============================================================================

def make_pdf(pages: List[str]) -> bytes:
    """Generates an in-memory valid PDF document with specified page contents."""
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), page_text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_docx(paragraphs: List[str]) -> bytes:
    """Generates an in-memory valid DOCX document with specified paragraphs."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_txt(content: str, encoding: str = "utf-8") -> bytes:
    """Generates an in-memory plain text document."""
    return content.encode(encoding)
