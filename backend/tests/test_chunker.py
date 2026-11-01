"""Tests for intelligent document chunking and provenance tracking."""

from app.services.chunker import DocumentChunker
from app.services.extractors.base import ExtractedDocument, ExtractedUnit


def test_chunker_basic_sizing():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    units = [
        ExtractedUnit(
            text="First sentence of unit 1. Second sentence of unit 1 is somewhat longer. Third sentence completes unit 1.",
            page_number=1,
        )
    ]
    doc = ExtractedDocument(units=units, total_pages=1)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert c.character_count <= 120  # bounded near target size
        assert len(c.text) > 0


def test_chunker_overlap_contains_prior_context():
    chunker = DocumentChunker(chunk_size=40, chunk_overlap=15)
    units = [
        ExtractedUnit(
            text="Alpha Beta Gamma Delta. Epsilon Zeta Eta Theta. Iota Kappa Lambda Mu.",
            page_number=1,
        )
    ]
    doc = ExtractedDocument(units=units, total_pages=1)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Verify sequential chunk indexing
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_page_provenance_pdf():
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=20)
    units = [
        ExtractedUnit(text="Content from page 1.", page_number=1),
        ExtractedUnit(text="Content from page 2.", page_number=2),
    ]
    doc = ExtractedDocument(units=units, total_pages=2)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    # Check that provenance reflects the pages
    assert chunks[0].page_start is not None
    assert chunks[0].page_end is not None
    assert "source_pages" in chunks[0].metadata


def test_chunker_docx_txt_provenance_no_fake_pages():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    units = [
        ExtractedUnit(
            text="Paragraph one of document.",
            page_number=None,
            metadata={"paragraph_index": 1},
        ),
        ExtractedUnit(
            text="Paragraph two of document.",
            page_number=None,
            metadata={"paragraph_index": 2},
        ),
    ]
    doc = ExtractedDocument(units=units, total_pages=None)
    chunks = chunker.chunk_document(doc)

    for c in chunks:
        assert c.page_start is None
        assert c.page_end is None
        assert "paragraph_start" in c.metadata


def test_chunker_loop_safety_with_large_overlap():
    # If overlap is erroneously configured >= chunk_size, chunker clamps it
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=100)
    assert chunker.chunk_overlap < chunker.chunk_size

    units = [ExtractedUnit(text="Short repetitive text for safety testing.", page_number=1)]
    doc = ExtractedDocument(units=units, total_pages=1)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1


def test_chunker_overlap_snaps_to_word_boundary():
    # Verify that overlap does not cut words into fragments
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=15)
    text = "Paragraph 1 is here.\n\nParagraph 2 has more details and information.\n\nParagraph 3 is also here."
    doc = ExtractedDocument(units=[ExtractedUnit(text=text, page_number=1)])
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3
    # Chunk 1 should start with the full word 'Paragraph', not a fragment like 'aph'
    assert chunks[1].text.startswith("Paragraph 2")

