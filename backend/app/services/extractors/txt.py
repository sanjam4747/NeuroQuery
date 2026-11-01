"""Plain text document text extractor with encoding fallback handling."""

import re
from app.services.extractors.base import BaseExtractor, ExtractedDocument, ExtractedUnit


class TxtExtractor(BaseExtractor):
    """Extracts text from plain text (.txt) documents preserving paragraph boundaries."""

    ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

    def extract(self, file_bytes: bytes) -> ExtractedDocument:
        decoded_text: str | None = None

        for encoding in self.ENCODINGS:
            try:
                decoded_text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            raise ValueError("Unable to decode text file using supported encodings (UTF-8, Latin-1, CP1252).")

        # Split into meaningful paragraphs (double newline or line breaks)
        raw_paragraphs = re.split(r"\n\s*\n", decoded_text)
        units: list[ExtractedUnit] = []
        para_idx = 0

        for para in raw_paragraphs:
            clean_para = para.strip()
            if clean_para:
                para_idx += 1
                units.append(
                    ExtractedUnit(
                        text=clean_para,
                        page_number=None,  # No fabricated page numbers for TXT
                        section=f"Section {para_idx}",
                        metadata={"paragraph_index": para_idx},
                    )
                )

        if not units:
            raise ValueError("Text document is empty or contains only whitespace.")

        return ExtractedDocument(units=units, total_pages=None)
