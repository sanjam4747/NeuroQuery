"""Base extraction contracts and data structures for document parsing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedUnit:
    """Represents a discrete structural unit (page, paragraph, section) from extraction."""

    text: str
    page_number: Optional[int] = None  # 1-indexed; None for non-paginated documents
    section: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Complete document extraction payload with metadata."""

    units: List[ExtractedUnit]
    total_pages: Optional[int] = None

    @property
    def total_characters(self) -> int:
        return sum(len(unit.text) for unit in self.units)

    @property
    def full_text(self) -> str:
        return "\n\n".join(unit.text for unit in self.units if unit.text.strip())


class BaseExtractor(ABC):
    """Abstract interface for format-specific extractors."""

    @abstractmethod
    def extract(self, file_bytes: bytes) -> ExtractedDocument:
        """Extracts text content and structural units from raw file bytes."""
        pass
