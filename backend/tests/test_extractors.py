"""Tests for format-specific text extractors (PDF, DOCX, TXT)."""

import pytest
from app.services.extractors import get_extractor, PdfExtractor, DocxExtractor, TxtExtractor
from tests.conftest import make_docx, make_pdf, make_txt


def test_pdf_extraction_multipage():
    extractor = get_extractor("pdf")
    assert isinstance(extractor, PdfExtractor)

    pages = [
        "First page content discussing semantic search.",
        "Second page content analyzing retrieval architectures.",
    ]
    pdf_bytes = make_pdf(pages)
    doc = extractor.extract(pdf_bytes)

    assert doc.total_pages == 2
    assert len(doc.units) == 2
    assert doc.units[0].page_number == 1
    assert "First page" in doc.units[0].text
    assert doc.units[1].page_number == 2
    assert "Second page" in doc.units[1].text


def test_pdf_skips_empty_pages():
    extractor = get_extractor("pdf")
    pages = [
        "Actual page 1 text.",
        "",  # Empty page
        "Actual page 3 text.",
    ]
    pdf_bytes = make_pdf(pages)
    doc = extractor.extract(pdf_bytes)

    assert doc.total_pages == 3
    assert len(doc.units) == 2
    assert doc.units[0].page_number == 1
    assert doc.units[1].page_number == 3


def test_pdf_no_extractable_text():
    extractor = get_extractor("pdf")
    pdf_bytes = make_pdf([""])  # Blank PDF
    with pytest.raises(ValueError, match="no extractable text was found"):
        extractor.extract(pdf_bytes)


def test_docx_extraction():
    extractor = get_extractor("docx")
    assert isinstance(extractor, DocxExtractor)

    paragraphs = [
        "NeuroQuery architecture paragraph 1.",
        "NeuroQuery retrieval pipeline paragraph 2.",
    ]
    docx_bytes = make_docx(paragraphs)
    doc = extractor.extract(docx_bytes)

    assert doc.total_pages is None  # No fabricated page numbers for DOCX
    assert len(doc.units) == 2
    assert doc.units[0].page_number is None
    assert doc.units[0].metadata["paragraph_index"] == 1
    assert "paragraph 1" in doc.units[0].text
    assert doc.units[1].metadata["paragraph_index"] == 2


def test_txt_extraction_utf8_and_latin1():
    extractor = get_extractor("txt")
    assert isinstance(extractor, TxtExtractor)

    # UTF-8 text
    txt_bytes = make_txt("Section 1: Overview\n\nSection 2: Architecture")
    doc = extractor.extract(txt_bytes)
    assert doc.total_pages is None
    assert len(doc.units) == 2
    assert "Section 1" in doc.units[0].text
    assert "Section 2" in doc.units[1].text

    # Latin-1 fallback with special characters
    latin1_bytes = "Café résumé señor\n\nPróxima sección".encode("latin-1")
    doc_latin1 = extractor.extract(latin1_bytes)
    assert len(doc_latin1.units) == 2
    assert "Café" in doc_latin1.units[0].text
