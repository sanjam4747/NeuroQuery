"""Tests for document upload validation, size limits, and signature checks."""

import pytest
from app.core.config import settings
from app.services.validation import DocumentValidator, ValidationError
from tests.conftest import make_docx, make_pdf, make_txt


def test_validate_valid_pdf():
    pdf_bytes = make_pdf(["NeuroQuery Phase 2 Validation Test."])
    file_type, filename = DocumentValidator.validate_upload("sample.pdf", pdf_bytes)
    assert file_type == "pdf"
    assert filename == "sample.pdf"


def test_validate_valid_docx():
    docx_bytes = make_docx(["NeuroQuery DOCX extraction testing."])
    file_type, filename = DocumentValidator.validate_upload("research.docx", docx_bytes)
    assert file_type == "docx"
    assert filename == "research.docx"


def test_validate_valid_txt():
    txt_bytes = make_txt("NeuroQuery plain text document.")
    file_type, filename = DocumentValidator.validate_upload("notes.txt", txt_bytes)
    assert file_type == "txt"
    assert filename == "notes.txt"


def test_validate_empty_document():
    with pytest.raises(ValidationError, match="Uploaded file is empty"):
        DocumentValidator.validate_upload("empty.txt", b"")


def test_validate_unsupported_format():
    with pytest.raises(ValidationError, match="Unsupported file format"):
        DocumentValidator.validate_upload("script.py", b"print('hello')")

    with pytest.raises(ValidationError, match="Unsupported file format"):
        DocumentValidator.validate_upload("image.png", b"\x89PNG\r\n\x1a\n")


def test_validate_upload_size_limit(monkeypatch):
    # Temporarily set limit to 1 KB for testing
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 1024)
    oversized_data = b"x" * 2048

    with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
        DocumentValidator.validate_upload("big.txt", oversized_data)


def test_validate_malformed_pdf():
    fake_pdf = b"Not A PDF Header At All"
    with pytest.raises(ValidationError, match="does not contain a valid %PDF- header"):
        DocumentValidator.validate_upload("fake.pdf", fake_pdf)


def test_validate_malformed_docx():
    fake_docx = b"PK\x03\x04CorruptZipData"
    with pytest.raises(ValidationError, match="Corrupt or unreadable Word document archive"):
        DocumentValidator.validate_upload("corrupt.docx", fake_docx)


def test_validate_binary_as_txt():
    binary_content = b"Some initial text\x00\x01\x02binary junk"
    with pytest.raises(ValidationError, match="Binary or null byte sequence detected"):
        DocumentValidator.validate_upload("binary.txt", binary_content)
