"""PDF document text extractor using PyMuPDF."""

import pymupdf
from app.core.logging import logger
from app.services.extractors.base import BaseExtractor, ExtractedDocument, ExtractedUnit


class PdfExtractor(BaseExtractor):
    """Extracts text page-by-page from PDF files using PyMuPDF."""

    def extract(self, file_bytes: bytes) -> ExtractedDocument:
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            logger.warning("Failed to parse PDF binary structure: %s", exc)
            raise ValueError(f"Corrupt or unreadable PDF document: {exc}")

        try:
            if doc.is_encrypted:
                raise ValueError("PDF is encrypted or password-protected and cannot be read.")

            total_pages = len(doc)
            units: list[ExtractedUnit] = []

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1
                try:
                    text = page.get_text("text")
                except Exception as exc:
                    logger.warning("Error reading text on PDF page %d: %s", page_num, exc)
                    text = ""

                # Skip pages with no extractable text without crashing
                if text and text.strip():
                    units.append(
                        ExtractedUnit(
                            text=text,
                            page_number=page_num,
                            section=f"Page {page_num}",
                            metadata={"page": page_num, "total_pages": total_pages},
                        )
                    )

            if not units:
                raise ValueError(
                    f"PDF contains {total_pages} page(s) but no extractable text was found "
                    "(the document may contain only scanned images or blank pages)."
                )

            return ExtractedDocument(units=units, total_pages=total_pages)
        finally:
            doc.close()
