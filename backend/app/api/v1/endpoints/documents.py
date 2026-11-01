"""API endpoints for Document ingestion, inspection, chunk retrieval, and deletion."""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.schemas.chunk import ChunkListResponse, ChunkResponse
from app.services.pipeline import pipeline_service
from app.services.validation import ValidationError

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Ingest Document",
    description="Uploads a PDF, DOCX, or TXT document, extracts text, normalizes content, and generates semantic chunks.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, or TXT)"),
    db: Session = Depends(get_db),
) -> Document:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename.",
        )

    try:
        file_bytes = await file.read()
        document = pipeline_service.ingest_document(
            db=db,
            file_bytes=file_bytes,
            original_filename=file.filename,
        )
        return document
    except ValidationError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred processing document: {exc}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="Returns a paginated list of ingested documents ordered by creation date descending.",
)
def list_documents(
    skip: int = Query(0, ge=0, description="Offset record index"),
    limit: int = Query(20, ge=1, le=100, description="Page size limit"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status (uploaded, processing, completed, failed)"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    query = select(Document)
    count_query = select(func.count(Document.id))

    if status_filter:
        query = query.where(Document.status == status_filter.lower())
        count_query = count_query.where(Document.status == status_filter.lower())

    total = db.scalar(count_query) or 0
    documents = db.scalars(
        query.order_by(Document.created_at.desc()).offset(skip).limit(limit)
    ).all()

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document Details",
    description="Retrieves metadata, processing state, and summary metrics for a specific document.",
)
def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return DocumentResponse.model_validate(doc)


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Get Document Chunks",
    description="Retrieves ordered text chunks with page and provenance metadata for a document.",
)
def get_document_chunks(
    document_id: uuid.UUID,
    skip: int = Query(0, ge=0, description="Chunk offset index"),
    limit: int = Query(50, ge=1, le=200, description="Chunk page size limit"),
    db: Session = Depends(get_db),
) -> ChunkListResponse:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    count_query = select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    total = db.scalar(count_query) or 0

    chunks_query = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index.asc())
        .offset(skip)
        .limit(limit)
    )
    chunks = db.scalars(chunks_query).all()

    return ChunkListResponse(
        items=[ChunkResponse.model_validate(c) for c in chunks],
        total=total,
        skip=skip,
        limit=limit,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Permanently deletes the document record, its cascading chunks, and the stored file on disk.",
)
def delete_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    filename = doc.original_filename
    pipeline_service.delete_document(db, doc)

    return {
        "message": f"Document '{filename}' and all associated chunks were successfully deleted.",
        "id": str(document_id),
    }
