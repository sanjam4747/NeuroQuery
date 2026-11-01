"""Document ingestion and chunking processing pipeline."""

import uuid
from typing import Optional
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.storage import storage_service
from app.services.validation import DocumentValidator, ValidationError
from app.services.extractors import get_extractor, ExtractedDocument, ExtractedUnit
from app.services.preprocessing import normalize_text
from app.services.chunker import chunker_service


class DocumentProcessingPipeline:
    """Orchestrates document ingestion, extraction, preprocessing, chunking, and persistence."""

    def ingest_document(
        self,
        db: Session,
        file_bytes: bytes,
        original_filename: str,
    ) -> Document:
        """
        Executes end-to-end ingestion pipeline:
        1. Validates file size, extension, and content signature.
        2. Saves raw file to secure storage.
        3. Creates initial Document record with status='processing'.
        4. Extracts text structural units.
        5. Normalizes text conservatively.
        6. Generates semantic chunks with provenance.
        7. Persists chunks and marks document status='completed'.
        """
        # 1. Validation
        file_type, sanitized_orig_name = DocumentValidator.validate_upload(
            original_filename, file_bytes
        )

        # 2. Storage
        unique_filename, relative_storage_path, _ = storage_service.save_file(
            file_bytes, sanitized_orig_name
        )

        # 3. Create initial Document record
        doc = Document(
            id=uuid.uuid4(),
            filename=unique_filename,
            original_filename=sanitized_orig_name,
            file_type=file_type,
            file_size=len(file_bytes),
            storage_path=relative_storage_path,
            status="processing",
            total_pages=None,
            total_characters=0,
            total_chunks=0,
            error_message=None,
        )

        db.add(doc)
        db.commit()
        db.refresh(doc)

        # 4-7. Extraction, Preprocessing, Chunking, Persistence
        try:
            # 4. Text Extraction
            extractor = get_extractor(file_type)
            extracted: ExtractedDocument = extractor.extract(file_bytes)

            # 5. Preprocessing & Normalization
            normalized_units = []
            for unit in extracted.units:
                norm_text = normalize_text(unit.text)
                if norm_text:
                    normalized_units.append(
                        ExtractedUnit(
                            text=norm_text,
                            page_number=unit.page_number,
                            section=unit.section,
                            metadata=unit.metadata,
                        )
                    )

            normalized_doc = ExtractedDocument(
                units=normalized_units,
                total_pages=extracted.total_pages,
            )

            # 6. Chunking
            generated_chunks = chunker_service.chunk_document(normalized_doc)

            if not generated_chunks:
                raise ValueError("No readable text chunks could be produced from document.")

            # 7. Persistence
            chunk_records = [
                Chunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    character_count=chunk.character_count,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    metadata_json=chunk.metadata,
                )
                for chunk in generated_chunks
            ]

            db.add_all(chunk_records)

            # Update Document summary
            doc.status = "completed"
            doc.total_pages = extracted.total_pages
            doc.total_characters = sum(c.character_count for c in chunk_records)
            doc.total_chunks = len(chunk_records)
            doc.error_message = None

            db.commit()
            db.refresh(doc)
            logger.info("Successfully ingested document %s (%d chunks)", doc.original_filename, doc.total_chunks)
            return doc

        except Exception as exc:
            db.rollback()
            logger.error("Error processing document %s: %s", original_filename, exc, exc_info=True)
            doc.status = "failed"
            doc.error_message = str(exc)
            db.commit()
            db.refresh(doc)
            return doc

    def delete_document(self, db: Session, document: Document) -> None:
        """Deletes a document from the database (cascading chunks) and removes stored file."""
        storage_path = document.storage_path
        db.delete(document)
        db.commit()

        # Delete physical file
        removed = storage_service.delete_file(storage_path)
        if not removed:
            logger.warning(
                "Physical file %s could not be found or unlinked during deletion of document %s",
                storage_path,
                document.id,
            )
        else:
            logger.info("Deleted document %s and removed storage file %s", document.id, storage_path)


pipeline_service = DocumentProcessingPipeline()
