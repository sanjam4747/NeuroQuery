"""API endpoints for Phase 5 Assistant and Retrieval-Augmented Generation (RAG)."""

import time
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.db.session import get_db
from app.schemas.rag import RAGHealthResponse, RAGQueryRequest, RAGQueryResponse
from app.services.llm.base import (
    LLMException,
    LLMGenerationError,
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)
from app.services.llm.factory import get_llm_provider
from app.services.rag.citation_validator import RAGResponseValidationError
from app.services.rag.orchestrator import rag_orchestrator

router = APIRouter(prefix="/assistant", tags=["Assistant (RAG)"])


@router.get(
    "/health",
    response_model=RAGHealthResponse,
    summary="Check LLM provider status and model availability",
)
async def check_assistant_health() -> RAGHealthResponse:
    """
    Performs a non-blocking diagnostic health check against the configured LLM inference provider.
    Verifies that the daemon is reachable and the configured model is downloaded.
    """
    provider = get_llm_provider()
    health = await provider.health_check()

    overall_status = "healthy" if health.is_available else "unavailable"

    return RAGHealthResponse(
        status=overall_status,
        provider=health.provider_name,
        model=health.model_name,
        provider_available=health.is_available,
        latency_ms=health.latency_ms,
        details=health.details,
        error_message=health.error_message,
    )


@router.post(
    "/ask",
    response_model=RAGQueryResponse,
    summary="Ask a document-grounded question using hybrid retrieval and LLM generation",
)
async def ask_assistant(
    request: RAGQueryRequest,
    db: Session = Depends(get_db),
) -> RAGQueryResponse:
    """
    Executes end-to-end document question-answering with factual verification:
    1. Validates and scopes query to selected document IDs
    2. Performs hybrid vector and BM25 retrieval with RRF
    3. Reranks evidence candidates with cross-encoder
    4. Selects non-overlapping evidence passages within token budget
    5. Prompts configured LLM with prompt-injection defense
    6. Verifies that all citations correspond to actual retrieved database passages
    7. Returns grounded answer, claim breakdown, and full telemetry
    """
    try:
        response = await rag_orchestrator.answer_question(db=db, request=request)
        return response

    except LLMProviderUnavailableError as exc:
        logger.error("RAG request failed: LLM provider unavailable (%s)", exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "LLM Provider Unavailable",
                "message": exc.message,
                "provider": exc.provider,
                "model": exc.model,
            },
        ) from exc

    except LLMModelNotFoundError as exc:
        logger.error("RAG request failed: LLM model missing (%s)", exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "LLM Model Not Found",
                "message": exc.message,
                "provider": exc.provider,
                "model": exc.model,
            },
        ) from exc

    except LLMTimeoutError as exc:
        logger.warning("RAG request timed out: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": "LLM Generation Timeout",
                "message": exc.message,
                "provider": exc.provider,
                "model": exc.model,
            },
        ) from exc

    except LLMGenerationError as exc:
        logger.error("RAG generation error: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "LLM Generation Failed",
                "message": exc.message,
                "provider": exc.provider,
                "model": exc.model,
            },
        ) from exc

    except RAGResponseValidationError as exc:
        logger.error("RAG model response validation failed: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error": "Invalid Model Response",
                "message": exc.message,
                "details": exc.details,
            },
        ) from exc

    except ValueError as exc:
        logger.warning("Invalid RAG request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Bad Request", "message": str(exc)},
        ) from exc

    except Exception as exc:
        logger.error("Unexpected error in RAG question answering: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while generating the document answer.",
            },
        ) from exc
