"""Text extraction modules and factory for NeuroQuery."""

from app.services.extractors.base import BaseExtractor, ExtractedDocument, ExtractedUnit
from app.services.extractors.pdf import PdfExtractor
from app.services.extractors.docx import DocxExtractor
from app.services.extractors.txt import TxtExtractor

_EXTRACTORS: dict[str, type[BaseExtractor]] = {
    "pdf": PdfExtractor,
    "docx": DocxExtractor,
    "txt": TxtExtractor,
}


def get_extractor(file_type: str) -> BaseExtractor:
    """Returns the extractor instance for the given normalized file type."""
    normalized_type = file_type.lower().lstrip(".")
    extractor_cls = _EXTRACTORS.get(normalized_type)
    if not extractor_cls:
        raise ValueError(f"No extractor registered for file type '{file_type}'.")
    return extractor_cls()


__all__ = [
    "BaseExtractor",
    "ExtractedDocument",
    "ExtractedUnit",
    "PdfExtractor",
    "DocxExtractor",
    "TxtExtractor",
    "get_extractor",
]
