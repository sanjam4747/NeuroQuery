"""Document storage service handling safe file persistence and deletion."""

import os
import re
import uuid
from pathlib import Path
from typing import Tuple
from app.core.config import settings
from app.core.logging import logger


class StorageService:
    """Manages secure on-disk persistence of uploaded documents."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.get_storage_path()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitizes filename removing unsafe characters and preventing path traversal."""
        # Strip directory components
        basename = os.path.basename(filename).strip()
        # Remove null bytes and non-printable chars
        basename = re.sub(r"[\x00-\x1f\x7f]", "", basename)
        # Replace spaces and special characters with underscores, keeping alphanumeric, dots, dashes, underscores
        clean_name = re.sub(r"[^\w\.\-]", "_", basename)
        # Avoid hidden/empty filenames
        clean_name = clean_name.lstrip(".")
        if not clean_name:
            clean_name = "unnamed_document"
        return clean_name

    def save_file(self, file_bytes: bytes, original_filename: str) -> Tuple[str, str, Path]:
        """
        Saves file bytes to disk with a collision-proof unique name.
        Returns:
            Tuple of (unique_filename, relative_storage_path, absolute_path)
        """
        clean_name = self._sanitize_filename(original_filename)
        doc_uuid = uuid.uuid4().hex[:12]
        unique_filename = f"{doc_uuid}_{clean_name}"

        dest_path = (self.base_dir / unique_filename).resolve()

        # Prevent path traversal: ensure dest_path is within base_dir
        if not str(dest_path).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Unsafe storage destination path detected for: {original_filename}")

        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        # Store relative path from backend root for portable storage references
        relative_path = f"{settings.STORAGE_DIR}/{unique_filename}"
        logger.info("Stored file %s (size: %d bytes) to %s", original_filename, len(file_bytes), unique_filename)

        return unique_filename, relative_path, dest_path

    def delete_file(self, storage_path: str) -> bool:
        """
        Deletes the file from disk using relative storage_path or filename.
        Returns True if deleted, False if not found.
        """
        filename = os.path.basename(storage_path)
        file_path = (self.base_dir / filename).resolve()

        # Path traversal guard
        if not str(file_path).startswith(str(self.base_dir.resolve())):
            logger.warning("Path traversal attempt in delete_file: %s", storage_path)
            return False

        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                logger.info("Deleted physical file: %s", file_path)
                return True
            except OSError as exc:
                logger.error("Failed to delete physical file %s: %s", file_path, exc)
                return False
        return False

    def get_file_path(self, storage_path: str) -> Path:
        """Resolves physical file path from relative storage path with safety check."""
        filename = os.path.basename(storage_path)
        resolved = (self.base_dir / filename).resolve()
        if not str(resolved).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Invalid storage path: {storage_path}")
        return resolved


storage_service = StorageService()
