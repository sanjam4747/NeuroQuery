"""Central RAG Orchestrator coordinating retrieval, context engineering, LLM inference, and citation verification."""

import asyncio
import time
from typing import List, Optional
import uuid
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.schemas.rag import (
    RAGCitationItem,
    RAGClaimItem,
    RAGDiagnostics,
    RAGGenerationMetadata,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.schemas.retrieval import RetrievalMetadata, RetrievalQueryRequest
from app.services.llm.base import GenerationConfig, LLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.rag.citation_validator import CitationValidator, ValidatedCitation
from app.services.rag.context_builder import ContextBuilder, context_builder, estimate_token_count
from app.services.rag.prompt_builder import PromptBuilder
from app.services.retrieval import retrieval_service


class RAGOrchestrator:
    """
    Coordinates the complete end-to-end question answering pipeline:
    1. Query normalization and document scope verification
    2. Hybrid Retrieval (Dense Vector + LuceneBM25 + RRF)
    3. Cross-Encoder Reranking
    4. Evidence Selection, Deduplication, and Token-Aware Context Engineering
    5. Grounded Prompt Construction with Prompt-Injection Defenses
    6. LLM Inference via Provider Abstraction
    7. Structured Output Parsing and Citation Verification
    8. Diagnostic Profiling and Response Assembly
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        context_builder_service: Optional[ContextBuilder] = None,
    ):
        self._custom_provider = llm_provider
        self._context_builder = context_builder_service or context_builder

    def get_provider(self) -> LLMProvider:
        """Returns the active or injected LLM provider."""
        if self._custom_provider:
            return self._custom_provider
        return get_llm_provider()

    async def answer_question(
        self,
        db: Session,
        request: RAGQueryRequest,
    ) -> RAGQueryResponse:
        """
        Executes the full RAG pipeline asynchronously without blocking the event loop.
        """
        overall_start = time.perf_counter()
        question = request.question.strip()

        # Step 1: Query Validation & Normalization
        val_start = time.perf_counter()
        if not question:
            raise ValueError("Question cannot be empty.")

        q_tokens = estimate_token_count(question)
        if q_tokens > 1000:
            raise ValueError(f"Question exceeds maximum allowed input token capacity (estimated {q_tokens} tokens > 1000 limit).")

        val_latency_ms = (time.perf_counter() - val_start) * 1000

        # Step 2: Hybrid Retrieval and Cross-Encoder Reranking (executed in threadpool to prevent event loop blocking)
        retrieval_start = time.perf_counter()
        retrieval_req = RetrievalQueryRequest(
            query=question,
            top_k=request.top_k or settings.RAG_MAX_EVIDENCE_CHUNKS,
            document_ids=request.document_ids,
            enable_reranking=request.enable_reranking,
            similarity_threshold=request.similarity_threshold,
        )

        retrieval_response = await asyncio.to_thread(
            retrieval_service.search,
            db=db,
            request=retrieval_req,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
        retrieved_chunks = retrieval_response.results

        # Fast-Path: Early Abstention if zero candidates found in the corpus
        if not retrieved_chunks:
            total_time_ms = (time.perf_counter() - overall_start) * 1000
            provider = self.get_provider()
            logger.info("Zero retrieval candidates found for question '%s'. Returning early abstention.", question[:30])

            diagnostics = RAGDiagnostics(
                query_validation_ms=round(val_latency_ms, 2),
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                context_building_ms=0.0,
                inference_latency_ms=0.0,
                citation_validation_ms=0.0,
                total_latency_ms=round(total_time_ms, 2),
                evidence_chunks_retrieved=0,
                evidence_chunks_selected=0,
                context_tokens_used=0,
            )

            gen_meta = RAGGenerationMetadata(
                provider=provider.provider_name,
                model=provider.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                generation_latency_ms=0.0,
            )

            return RAGQueryResponse(
                question=question,
                answer="No relevant document passages were found in the selected collection to answer this question.",
                is_answered=False,
                claims=[],
                citations=[],
                evidence_sources=[],
                insufficient_evidence_reason="No matching text passages were retrieved from the document corpus.",
                invalid_citation_ids=[],
                parsing_status="valid_abstention",
                retrieval_metadata=retrieval_response.retrieval_metadata,
                generation_metadata=gen_meta,
                diagnostics=diagnostics,
            )

        # Step 3: Evidence Selection, Deduplication, and Token-Aware Context Engineering
        ctx_start = time.perf_counter()
        system_prompt = PromptBuilder.get_system_prompt()
        max_out = request.max_output_tokens or settings.LLM_MAX_OUTPUT_TOKENS

        context_result = self._context_builder.build_context(
            retrieved_chunks=retrieved_chunks,
            question=question,
            system_prompt=system_prompt,
            max_output_tokens=max_out,
        )
        ctx_latency_ms = (time.perf_counter() - ctx_start) * 1000

        # If context budget could accommodate zero evidence
        if not context_result.sources:
            total_time_ms = (time.perf_counter() - overall_start) * 1000
            provider = self.get_provider()
            logger.warning("Context builder selected 0 sources due to budget limit for question '%s'", question[:30])

            diagnostics = RAGDiagnostics(
                query_validation_ms=round(val_latency_ms, 2),
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                context_building_ms=round(ctx_latency_ms, 2),
                inference_latency_ms=0.0,
                citation_validation_ms=0.0,
                total_latency_ms=round(total_time_ms, 2),
                evidence_chunks_retrieved=len(retrieved_chunks),
                evidence_chunks_selected=0,
                context_tokens_used=0,
            )

            gen_meta = RAGGenerationMetadata(
                provider=provider.provider_name,
                model=provider.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                generation_latency_ms=0.0,
            )

            return RAGQueryResponse(
                question=question,
                answer="The available model context budget was insufficient to include evidence passages for this question.",
                is_answered=False,
                claims=[],
                citations=[],
                evidence_sources=[],
                insufficient_evidence_reason="Context budget exhausted by input prompt and output token reservation.",
                invalid_citation_ids=[],
                parsing_status="valid_abstention",
                retrieval_metadata=retrieval_response.retrieval_metadata,
                generation_metadata=gen_meta,
                diagnostics=diagnostics,
            )

        # Step 4: Grounded Prompt Construction
        user_prompt = PromptBuilder.build_user_prompt(
            question=question,
            formatted_context=context_result.formatted_context,
        )
        json_schema = PromptBuilder.get_json_schema()

        # Step 5: LLM Generation via Provider Abstraction
        inf_start = time.perf_counter()
        provider = self.get_provider()
        gen_config = GenerationConfig(
            temperature=request.temperature if request.temperature is not None else settings.LLM_TEMPERATURE,
            max_tokens=max_out,
            timeout_seconds=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )

        gen_result = await provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=json_schema,
            config=gen_config,
        )
        inf_latency_ms = (time.perf_counter() - inf_start) * 1000

        # Step 6: Structured Response Parsing and Citation Verification
        cit_start = time.perf_counter()
        validation_res = CitationValidator.parse_and_validate(
            raw_output=gen_result.raw_text,
            source_map=context_result.source_map,
        )

        # Persistence check: verify cited chunks actually exist in database
        validation_res = CitationValidator.verify_chunk_persistence(
            db=db,
            result=validation_res,
        )
        cit_latency_ms = (time.perf_counter() - cit_start) * 1000

        total_latency_ms = (time.perf_counter() - overall_start) * 1000

        # Step 7: Map to API Schemas
        def to_citation_item(c: ValidatedCitation) -> RAGCitationItem:
            return RAGCitationItem(
                source_id=c.source_id,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                page_start=c.page_start,
                page_end=c.page_end,
                text_snippet=c.text_snippet,
                dense_score=c.dense_score,
                bm25_score=c.bm25_score,
                rrf_score=c.rrf_score,
                reranker_score=c.reranker_score,
                final_rank=c.final_rank,
            )

        citation_items = [to_citation_item(c) for c in validation_res.citations]
        evidence_items = [to_citation_item(c) for c in validation_res.evidence_sources]

        claim_items: List[RAGClaimItem] = []
        for claim in validation_res.claims:
            claim_items.append(
                RAGClaimItem(
                    claim_text=claim.claim_text,
                    citation_ids=claim.citation_ids,
                    citations=[to_citation_item(c) for c in claim.citations],
                    is_supported=claim.is_supported,
                    unsupported_citation_ids=claim.unsupported_citation_ids,
                )
            )

        diagnostics = RAGDiagnostics(
            query_validation_ms=round(val_latency_ms, 2),
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            context_building_ms=round(ctx_latency_ms, 2),
            inference_latency_ms=round(inf_latency_ms, 2),
            citation_validation_ms=round(cit_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            evidence_chunks_retrieved=len(retrieved_chunks),
            evidence_chunks_selected=len(context_result.sources),
            context_tokens_used=context_result.total_context_tokens,
        )

        gen_meta = RAGGenerationMetadata(
            provider=provider.provider_name,
            model=gen_result.model,
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
            generation_latency_ms=round(gen_result.duration_ms, 2),
        )

        return RAGQueryResponse(
            question=question,
            answer=validation_res.answer,
            is_answered=validation_res.is_answered,
            claims=claim_items,
            citations=citation_items,
            evidence_sources=evidence_items,
            insufficient_evidence_reason=validation_res.insufficient_evidence_reason,
            invalid_citation_ids=validation_res.invalid_citation_ids,
            parsing_status=validation_res.parsing_status,
            retrieval_metadata=retrieval_response.retrieval_metadata,
            generation_metadata=gen_meta,
            diagnostics=diagnostics,
        )


# Singleton instance
rag_orchestrator = RAGOrchestrator()
