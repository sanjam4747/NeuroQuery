"""Document upload validation service."""

import io
import os
import zipfile
from typing import Tuple
from app.core.config import settings


class ValidationError(ValueError):
    """Custom validation exception for document uploads."""
    pass


class DocumentValidator:
    """Validates document size, extensions, and content signatures."""

    @classmethod
    def validate_upload(cls, filename: str, file_bytes: bytes) -> Tuple[str, str]:
        """
        Validates uploaded file size, extension, and content signature.
        Returns:
            Tuple of (normalized_file_type, sanitized_filename)
        Raises:
            ValidationError if any check fails.
        """
        if not filename or not filename.strip():
            raise ValidationError("Filename cannot be empty.")

        # 1. Size Validation
        size = len(file_bytes)
        if size == 0:
            raise ValidationError("Uploaded file is empty (0 bytes).")

        if size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            actual_mb = round(size / (1024 * 1024), 2)
            raise ValidationError(
                f"File size ({actual_mb} MB) exceeds maximum allowed limit of {max_mb} MB."
            )

        # 2. Extension Validation
        _, ext = os.path.splitext(filename.lower())
        if ext not in settings.ALLOWED_EXTENSIONS:
            allowed_str = ", ".join(settings.ALLOWED_EXTENSIONS)
            raise ValidationError(
                f"Unsupported file format '{ext or 'unknown'}'. Supported formats: {allowed_str}."
            )

        file_type = ext.lstrip(".")

        # 3. Content Signature & Structure Validation
        cls.validate_content_signature(file_type, file_bytes)

        return file_type, filename.strip()

    @classmethod
    def validate_content_signature(cls, file_type: str, file_bytes: bytes) -> None:
        """Inspects magic bytes and headers to ensure content matches extension."""
        if file_type == "pdf":
            # PDF header must contain %PDF- in the first 1024 bytes
            header = file_bytes[:1024]
            if b"%PDF-" not in header:
                raise ValidationError("Invalid PDF: File does not contain a valid %PDF- header signature.")

        elif file_type == "docx":
            # DOCX is an OpenXML ZIP archive starting with PK\x03\x04
            if not file_bytes.startswith(b"PK\x03\x04"):
                raise ValidationError("Invalid DOCX: File does not match the Microsoft Word OpenXML signature.")

            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
                    namelist = zf.namelist()
                    if "[Content_Types].xml" not in namelist and "word/document.xml" not in namelist:
                        raise ValidationError("Invalid DOCX: Archive missing required document structure.")
            except zipfile.BadZipFile:
                raise ValidationError("Invalid DOCX: Corrupt or unreadable Word document archive.")

        elif file_type == "txt":
            # Ensure file is valid text, not binary executable
            if b"\x00" in file_bytes[:1024]:
                raise ValidationError("Invalid TXT: Binary or null byte sequence detected in plain text file.")

            decoded = False
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    file_bytes.decode(encoding)
                    decoded = True
                    break
                except UnicodeDecodeError:
                    continue

            if not decoded:
                raise ValidationError("Invalid TXT: Unable to decode text content with UTF-8 or standard fallbacks.")
