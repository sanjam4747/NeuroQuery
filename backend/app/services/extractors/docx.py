"""DOCX document text extractor using python-docx."""

import io
import docx
from app.core.logging import logger
from app.services.extractors.base import BaseExtractor, ExtractedDocument, ExtractedUnit


class DocxExtractor(BaseExtractor):
    """Extracts text paragraphs and tables from Word (.docx) files."""

    def extract(self, file_bytes: bytes) -> ExtractedDocument:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
        except Exception as exc:
            logger.warning("Failed to parse DOCX document: %s", exc)
            raise ValueError(f"Corrupt or unreadable Word document: {exc}")

        units: list[ExtractedUnit] = []
        para_idx = 0

        # Extract paragraphs in document order
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                para_idx += 1
                style_name = p.style.name if p.style else None
                units.append(
                    ExtractedUnit(
                        text=text,
                        page_number=None,  # No fabricated page numbers for DOCX
                        section=f"Paragraph {para_idx}",
                        metadata={
                            "paragraph_index": para_idx,
                            "style": style_name,
                        },
                    )
                )

        # Also extract table rows if any tables exist
        for table_idx, table in enumerate(doc.tables):
            for row_idx, row in enumerate(table.rows):
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    row_text = " | ".join(row_cells)
                    units.append(
                        ExtractedUnit(
                            text=row_text,
                            page_number=None,
                            section=f"Table {table_idx + 1} Row {row_idx + 1}",
                            metadata={
                                "table_index": table_idx + 1,
                                "row_index": row_idx + 1,
                            },
                        )
                    )

        if not units:
            raise ValueError("DOCX document contains no readable text content.")

        return ExtractedDocument(units=units, total_pages=None)
