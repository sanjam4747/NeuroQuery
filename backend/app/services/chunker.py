"""Intelligent document chunking service with provenance tracking."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from app.core.config import settings
from app.services.extractors.base import ExtractedDocument, ExtractedUnit


@dataclass
class GeneratedChunk:
    """Represents a generated text chunk with provenance metadata."""

    chunk_index: int
    text: str
    character_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """
    Splits extracted document content into deterministic, bounded chunks.

    Design details:
    - Sizing: Character-based with documented token approximation.
      (Default: 1500 chars ≈ 350-400 tokens; overlap 200 chars ≈ 50 tokens).
    - Boundary selection hierarchy:
      1. Paragraph breaks (double newlines)
      2. Sentence endings (. ! ? followed by whitespace)
      3. Word boundaries (spaces, newlines)
      4. Hard character boundary fallback for long unbroken strings.
    - Provenance: Retains source page numbers (PDF) or paragraph indexes (DOCX/TXT).
    - Loop safety: Strictly advances by at least 1 character on every iteration.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Safety: overlap must always be strictly less than chunk_size
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def chunk_document(self, document: ExtractedDocument) -> List[GeneratedChunk]:
        """Chunks an extracted document into ordered, provenance-annotated chunks."""
        if not document.units:
            return []

        # 1. Build unified text and interval-based provenance map
        full_text, intervals = self._build_provenance_map(document.units)
        total_len = len(full_text)

        if total_len == 0 or not full_text.strip():
            return []

        chunks: List[GeneratedChunk] = []
        start = 0
        chunk_idx = 0

        while start < total_len:
            # Skip any leading whitespace at chunk boundary
            while start < total_len and full_text[start] in (" ", "\t", "\n", "\r"):
                start += 1

            if start >= total_len:
                break

            target_end = min(start + self.chunk_size, total_len)

            if target_end >= total_len:
                end = total_len
            else:
                end = self._find_best_split_point(full_text, start, target_end)

            chunk_text = full_text[start:end].strip()

            if chunk_text:
                page_start, page_end, provenance_meta = self._resolve_provenance(intervals, start, end)

                chunks.append(
                    GeneratedChunk(
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        character_count=len(chunk_text),
                        page_start=page_start,
                        page_end=page_end,
                        metadata={
                            "char_start": start,
                            "char_end": end,
                            **provenance_meta,
                        },
                    )
                )
                chunk_idx += 1

            if end >= total_len:
                break

            # Calculate next starting position with overlap, snapping to word boundary
            raw_start = max(start + 1, end - self.chunk_overlap)
            candidate = raw_start

            # If raw_start lands in the middle of a word, snap backward to word start
            while (
                candidate > start
                and candidate < total_len
                and full_text[candidate].isalnum()
                and full_text[candidate - 1].isalnum()
            ):
                candidate -= 1

            if candidate > start:
                next_start = candidate
            else:
                # If moving backward would stall, snap forward to next word
                candidate = raw_start
                while candidate < end and full_text[candidate].isalnum():
                    candidate += 1
                next_start = candidate if candidate > start else raw_start

            start = next_start

        return chunks

    def _build_provenance_map(
        self, units: List[ExtractedUnit]
    ) -> Tuple[str, List[Tuple[int, int, Optional[int], Optional[int]]]]:
        """
        Concatenates extracted units and tracks character offsets:
        Returns:
            (full_text, [(start_char, end_char, page_number, paragraph_index)])
        """
        parts: List[str] = []
        intervals: List[Tuple[int, int, Optional[int], Optional[int]]] = []
        current_offset = 0

        for idx, unit in enumerate(units):
            text = unit.text.strip()
            if not text:
                continue

            if idx > 0 and parts:
                parts.append("\n\n")
                current_offset += 2

            start = current_offset
            parts.append(text)
            current_offset += len(text)
            end = current_offset

            page = unit.page_number
            para = unit.metadata.get("paragraph_index")
            intervals.append((start, end, page, para))

        return "".join(parts), intervals

    def _find_best_split_point(self, text: str, start: int, target_end: int) -> int:
        """Finds natural semantic break point between start + (size/2) and target_end."""
        min_split = start + max(1, self.chunk_size // 2)
        search_window = text[min_split:target_end]

        # 1. Look for paragraph break (\n\n)
        last_para = search_window.rfind("\n\n")
        if last_para != -1:
            return min_split + last_para + 2

        # 2. Look for sentence break ([.!?]\s+)
        sentence_matches = list(re.finditer(r"[.!?]\s+", search_window))
        if sentence_matches:
            last_match = sentence_matches[-1]
            return min_split + last_match.end()

        # 3. Look for single line break or space
        last_space = max(search_window.rfind(" "), search_window.rfind("\n"))
        if last_space != -1:
            return min_split + last_space + 1

        # 4. Fallback: hard cut at target_end
        return target_end

    def _resolve_provenance(
        self,
        intervals: List[Tuple[int, int, Optional[int], Optional[int]]],
        start: int,
        end: int,
    ) -> Tuple[Optional[int], Optional[int], Dict[str, Any]]:
        """Resolves page ranges and paragraph ranges overlapping [start, end]."""
        pages: List[int] = []
        paragraphs: List[int] = []

        for int_start, int_end, page, para in intervals:
            # Check overlap between [start, end] and [int_start, int_end]
            if max(start, int_start) < min(end, int_end):
                if page is not None:
                    pages.append(page)
                if para is not None:
                    paragraphs.append(para)

        meta: Dict[str, Any] = {}
        page_start: Optional[int] = None
        page_end: Optional[int] = None

        if pages:
            unique_pages = sorted(list(set(pages)))
            page_start = unique_pages[0]
            page_end = unique_pages[-1]
            meta["source_pages"] = unique_pages

        if paragraphs:
            unique_paras = sorted(list(set(paragraphs)))
            meta["paragraph_start"] = unique_paras[0]
            meta["paragraph_end"] = unique_paras[-1]

        return page_start, page_end, meta


chunker_service = DocumentChunker()
