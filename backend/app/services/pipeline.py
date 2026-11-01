"""Document ingestion and chunking processing pipeline."""

import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.logging import logger
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.storage import storage_service
from app.services.validation import DocumentValidator, ValidationError
from app.services.extractors import get_extractor, ExtractedDocument, ExtractedUnit
from app.services.preprocessing import normalize_text
from app.services.chunker import chunker_service
from app.services.embeddings import embedding_service


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

            # 7. Semantic Embedding Generation (Phase 3)
            doc.embedding_status = "generating"
            chunk_texts = [chunk.text for chunk in generated_chunks]
            embeddings, emb_metrics = embedding_service.embed_batch(chunk_texts)

            now_utc = datetime.now(timezone.utc)
            chunk_records = []
            for idx, chunk in enumerate(generated_chunks):
                meta = dict(chunk.metadata)
                token_count = emb_metrics["token_counts"][idx]
                sub_chunk_count = emb_metrics["sub_chunk_counts"][idx]
                meta["token_count"] = token_count
                meta["sub_chunk_count"] = sub_chunk_count
                if idx in emb_metrics["oversized_chunks"]:
                    meta["oversized_chunk"] = True
                    meta["embedding_method"] = "token_weighted_subchunk_pooling"
                    meta["sub_chunks"] = emb_metrics["chunk_details"][idx]["sub_chunks"]
                    meta["oversized_info"] = (
                        f"Chunk has {token_count} tokens (> {embedding_service.max_sequence_length}). "
                        f"Safely embedded via {sub_chunk_count} token-weighted sub-windows with zero content discarded."
                    )
                else:
                    meta["oversized_chunk"] = False
                    meta["embedding_method"] = "direct"

                chunk_records.append(
                    Chunk(
                        id=uuid.uuid4(),
                        document_id=doc.id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        character_count=chunk.character_count,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        metadata_json=meta,
                        embedding=embeddings[idx],
                        embedding_model=embedding_service.model_name,
                        embedded_at=now_utc,
                    )
                )

            # 8. Persistence (Atomic commit)
            db.add_all(chunk_records)

            # Update Document summary & embedding metrics
            doc.status = "completed"
            doc.total_pages = extracted.total_pages
            doc.total_characters = sum(c.character_count for c in chunk_records)
            doc.total_chunks = len(chunk_records)
            doc.error_message = None

            doc.embedding_status = "completed"
            doc.embedding_model = embedding_service.model_name
            doc.total_embedded_chunks = len(chunk_records)
            doc.embedding_error = None

            db.commit()
            db.refresh(doc)
            logger.info(
                "Successfully ingested document %s (%d chunks, %d embedded in %.1f ms)",
                doc.original_filename,
                doc.total_chunks,
                doc.total_embedded_chunks,
                emb_metrics["duration_ms"],
            )
            return doc

        except Exception as exc:
            db.rollback()
            logger.error("Error processing document %s: %s", original_filename, exc, exc_info=True)
            doc.status = "failed"
            doc.error_message = str(exc)
            doc.embedding_status = "failed"
            doc.embedding_error = str(exc)
            db.commit()
            db.refresh(doc)
            return doc

    def reembed_document(self, db: Session, document_id: uuid.UUID) -> Document:
        """
        Regenerates semantic embeddings for all existing chunks of a document.
        Updates chunks in-place without duplicating records.
        """
        doc = db.get(Document, document_id)
        if not doc:
            raise ValueError(f"Document with ID '{document_id}' not found.")

        chunks = db.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        ).all()

        if not chunks:
            raise ValueError(f"Document '{document_id}' has no chunks available to embed.")

        doc.embedding_status = "generating"
        db.commit()

        try:
            chunk_texts = [c.text for c in chunks]
            embeddings, emb_metrics = embedding_service.embed_batch(chunk_texts)

            now_utc = datetime.now(timezone.utc)
            for idx, chunk in enumerate(chunks):
                chunk.embedding = embeddings[idx]
                chunk.embedding_model = embedding_service.model_name
                chunk.embedded_at = now_utc

                meta = dict(chunk.metadata_json or {})
                token_count = emb_metrics["token_counts"][idx]
                sub_chunk_count = emb_metrics["sub_chunk_counts"][idx]
                meta["token_count"] = token_count
                meta["sub_chunk_count"] = sub_chunk_count
                if idx in emb_metrics["oversized_chunks"]:
                    meta["oversized_chunk"] = True
                    meta["embedding_method"] = "token_weighted_subchunk_pooling"
                    meta["sub_chunks"] = emb_metrics["chunk_details"][idx]["sub_chunks"]
                    meta["oversized_info"] = (
                        f"Chunk has {token_count} tokens (> {embedding_service.max_sequence_length}). "
                        f"Safely embedded via {sub_chunk_count} token-weighted sub-windows with zero content discarded."
                    )
                    if "oversized_warning" in meta:
                        del meta["oversized_warning"]
                else:
                    meta["oversized_chunk"] = False
                    meta["embedding_method"] = "direct"
                chunk.metadata_json = meta

            doc.embedding_status = "completed"
            doc.embedding_model = embedding_service.model_name
            doc.total_embedded_chunks = len(chunks)
            doc.embedding_error = None

            db.commit()
            db.refresh(doc)
            logger.info(
                "Successfully re-embedded document %s (%d chunks in %.1f ms)",
                doc.original_filename,
                len(chunks),
                emb_metrics["duration_ms"],
            )
            return doc
        except Exception as exc:
            db.rollback()
            logger.error("Re-embedding failed for document %s: %s", document_id, exc, exc_info=True)
            doc.embedding_status = "failed"
            doc.embedding_error = str(exc)
            db.commit()
            db.refresh(doc)
            raise RuntimeError(f"Re-embedding failed: {exc}") from exc

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
