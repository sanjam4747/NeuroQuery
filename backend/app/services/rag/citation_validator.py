"""Structured answer parsing, citation verification, and provenance resolution."""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Set
import uuid
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import logger
from app.models.chunk import Chunk
from app.services.rag.context_builder import ContextSource


class RAGResponseValidationError(Exception):
    """Raised when LLM output violates the structured JSON schema or cannot be safely parsed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


@dataclass
class ValidatedCitation:
    """A verified source citation tied to a real retrieved document chunk."""

    source_id: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_start: Optional[int]
    page_end: Optional[int]
    text_snippet: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    reranker_score: Optional[float] = None
    final_rank: int = 1


@dataclass
class ValidatedClaim:
    """An individual answer claim associated with verified citations and support status."""

    claim_text: str
    citation_ids: List[str]
    citations: List[ValidatedCitation] = field(default_factory=list)
    is_supported: bool = True
    unsupported_citation_ids: List[str] = field(default_factory=list)


@dataclass
class CitationValidationResult:
    """Outcome of structured parsing and citation verification."""

    answer: str
    is_answered: bool
    claims: List[ValidatedClaim]
    citations: List[ValidatedCitation]
    invalid_citation_ids: List[str]
    insufficient_evidence_reason: Optional[str]
    evidence_sources: List[ValidatedCitation]
    parsing_status: str = "valid_structured"  # "valid_structured", "recovered_with_repair", "valid_abstention"


CITATION_TAG_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")


class CitationValidator:
    """
    Parses LLM output, strictly enforces structured JSON schema, validates citation
    identifiers against the context source map, and enriches claims with persisted document provenance.

    Important distinction:
    Identifier validation verifies that a citation maps to a real retrieved passage.
    It does NOT prove semantic entailment (future NLI layer).
    """

    @classmethod
    def parse_and_validate(
        cls,
        raw_output: str,
        source_map: Dict[str, ContextSource],
    ) -> CitationValidationResult:
        """
        Parses structured model output and verifies all citation references.

        Args:
            raw_output: Raw text output from LLM provider.
            source_map: Map from source ID (e.g. 'S1') to ContextSource from ContextBuilder.

        Returns:
            CitationValidationResult with verified citations, claims, and abstention status.

        Raises:
            RAGResponseValidationError: If output is unparseable or violates schema constraints.
        """
        evidence_sources = [
            cls._build_validated_citation(src) for src in source_map.values()
        ]

        if not raw_output or not raw_output.strip():
            raise RAGResponseValidationError(
                message="LLM provider returned empty response output.",
                details={"reason": "empty_output"},
            )

        # 1. Bounded JSON extraction and repair
        parsed_data, parsing_status = cls._extract_and_repair_json(raw_output)

        if parsed_data is None:
            logger.error("LLM response failed structured JSON parsing: %s...", raw_output[:120].strip())
            raise RAGResponseValidationError(
                message="Language model returned output that could not be parsed as valid JSON.",
                details={
                    "reason": "json_parse_failure",
                    "sample": raw_output[:200].strip(),
                },
            )

        # 2. Strict Schema Revalidation
        if not isinstance(parsed_data, dict):
            raise RAGResponseValidationError(
                message="Language model response must be a JSON object, got " + type(parsed_data).__name__,
                details={"reason": "root_not_object"},
            )

        is_answered = parsed_data.get("is_answered")
        if is_answered is None:
            # Fallback check: if answer is present, treat as answered
            is_answered = bool(parsed_data.get("answer"))
        elif not isinstance(is_answered, bool):
            is_answered = bool(is_answered)

        answer_text = parsed_data.get("answer")
        insufficient_reason = parsed_data.get("insufficient_evidence_reason")
        if insufficient_reason is not None:
            insufficient_reason = str(insufficient_reason).strip()

        # Handle explicit abstention
        if not is_answered:
            ans_str = str(answer_text).strip() if answer_text else "The supplied documents do not contain sufficient evidence to answer this question."
            return CitationValidationResult(
                answer=ans_str,
                is_answered=False,
                claims=[],
                citations=[],
                invalid_citation_ids=[],
                insufficient_evidence_reason=insufficient_reason or "Insufficient evidence in document collection.",
                evidence_sources=evidence_sources,
                parsing_status="valid_abstention",
            )

        # For answered responses, answer_text must be non-empty string
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            raise RAGResponseValidationError(
                message="Structured response indicated is_answered=True but answer text is empty or not a string.",
                details={"reason": "missing_answer_text"},
            )
        answer_text = answer_text.strip()

        # Validate claims array
        raw_claims = parsed_data.get("claims")
        if raw_claims is None:
            raise RAGResponseValidationError(
                message="Structured response missing required 'claims' array.",
                details={"reason": "missing_claims_field"},
            )
        if not isinstance(raw_claims, list):
            raise RAGResponseValidationError(
                message="Structured response 'claims' field must be a list, got " + type(raw_claims).__name__,
                details={"reason": "invalid_claims_type"},
            )

        validated_claims: List[ValidatedClaim] = []
        all_cited_ids: Set[str] = set()
        invalid_ids: Set[str] = set()

        for idx, c in enumerate(raw_claims):
            if not isinstance(c, dict):
                raise RAGResponseValidationError(
                    message=f"Claim item at index {idx} must be a JSON object, got {type(c).__name__}.",
                    details={"index": idx},
                )

            claim_text = c.get("claim_text")
            if not claim_text or not isinstance(claim_text, str) or not claim_text.strip():
                raise RAGResponseValidationError(
                    message=f"Claim item at index {idx} missing non-empty 'claim_text'.",
                    details={"index": idx},
                )
            claim_text = claim_text.strip()

            raw_cit_ids = c.get("citation_ids", [])
            if not isinstance(raw_cit_ids, list):
                raise RAGResponseValidationError(
                    message=f"Claim 'citation_ids' at index {idx} must be a list of source tags.",
                    details={"index": idx},
                )

            clean_cit_ids: List[str] = []
            claim_citations: List[ValidatedCitation] = []
            unsupported_cids: List[str] = []

            for raw_id in raw_cit_ids:
                cid = str(raw_id).strip().upper().strip("[]")
                if not cid:
                    continue
                if cid in source_map:
                    if cid not in clean_cit_ids:
                        clean_cit_ids.append(cid)
                        claim_citations.append(cls._build_validated_citation(source_map[cid]))
                        all_cited_ids.add(cid)
                else:
                    unsupported_cids.append(cid)
                    invalid_ids.add(cid)
                    logger.warning("Rejected hallucinated citation identifier: %s in claim %d", cid, idx)

            is_supported = len(claim_citations) > 0

            validated_claims.append(
                ValidatedClaim(
                    claim_text=claim_text,
                    citation_ids=clean_cit_ids,
                    citations=claim_citations,
                    is_supported=is_supported,
                    unsupported_citation_ids=unsupported_cids,
                )
            )

        # Also extract any in-text citations from answer text (e.g. [S1])
        in_text_matches = CITATION_TAG_PATTERN.findall(answer_text)
        for match in in_text_matches:
            cid_clean = match.upper().strip("[]")
            if cid_clean in source_map:
                all_cited_ids.add(cid_clean)
            else:
                invalid_ids.add(cid_clean)

        # Collect distinct verified citations ordered by source ID index
        distinct_citations = [
            cls._build_validated_citation(source_map[cid])
            for cid in sorted(all_cited_ids, key=lambda s: int(s[1:]) if s[1:].isdigit() else 999)
        ]

        # Contradiction guard: if model claimed is_answered=True but has ZERO valid citations,
        # or if model explicitly admitted in insufficient_evidence_reason that evidence is absent:
        if len(distinct_citations) == 0 and insufficient_reason and any(
            phrase in insufficient_reason.lower()
            for phrase in [
                "does not contain",
                "not contain",
                "no information",
                "insufficient",
                "not found",
                "cannot answer",
                "no evidence",
                "lacks information",
            ]
        ):
            logger.info(
                "Model set is_answered=True with 0 valid citations and admitted insufficient evidence ('%s'). Overriding to honest abstention.",
                insufficient_reason[:80],
            )
            return CitationValidationResult(
                answer=f"The available document evidence does not contain sufficient information to answer this question. ({insufficient_reason})",
                is_answered=False,
                claims=[],
                citations=[],
                invalid_citation_ids=sorted(invalid_ids),
                insufficient_evidence_reason=insufficient_reason,
                evidence_sources=evidence_sources,
                parsing_status="valid_abstention",
            )

        return CitationValidationResult(
            answer=answer_text,
            is_answered=True,
            claims=validated_claims,
            citations=distinct_citations,
            invalid_citation_ids=sorted(invalid_ids),
            insufficient_evidence_reason=None,
            evidence_sources=evidence_sources,
            parsing_status=parsing_status,
        )

    @classmethod
    def _extract_and_repair_json(cls, raw: str) -> tuple[Optional[Dict[str, Any]], str]:
        """
        Attempts bounded repair and extraction of JSON object from raw LLM output.
        Returns: (parsed_dict, parsing_status)
        """
        cleaned = raw.strip()

        # Pass 1: Direct JSON parse
        try:
            val = json.loads(cleaned)
            if isinstance(val, dict):
                return val, "valid_structured"
        except json.JSONDecodeError:
            pass

        # Pass 2: Markdown fence extraction
        fence_match = JSON_FENCE_PATTERN.search(cleaned)
        if fence_match:
            try:
                val = json.loads(fence_match.group(1))
                if isinstance(val, dict):
                    return val, "recovered_with_repair"
            except json.JSONDecodeError:
                pass

        # Pass 3: Search for outermost curly braces
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = cleaned[first_brace : last_brace + 1]
            try:
                val = json.loads(candidate)
                if isinstance(val, dict):
                    return val, "recovered_with_repair"
            except json.JSONDecodeError:
                # Pass 4: Trailing comma repair
                repaired = TRAILING_COMMA_PATTERN.sub(r"\1", candidate)
                try:
                    val = json.loads(repaired)
                    if isinstance(val, dict):
                        return val, "recovered_with_repair"
                except json.JSONDecodeError:
                    pass

        return None, "failed"

    @classmethod
    def verify_chunk_persistence(
        cls,
        db: Session,
        result: CitationValidationResult,
    ) -> CitationValidationResult:
        """
        Verifies that every cited chunk currently exists in the persistent database.
        Purges any chunk that was deleted or invalidated, updating claims accordingly.
        """
        if not result.citations:
            return result

        cited_uuids = [c.chunk_id for c in result.citations]
        stmt = select(Chunk.id).where(Chunk.id.in_(cited_uuids))
        existing_uuids = set(db.scalars(stmt).all())

        missing_uuids = set(cited_uuids) - existing_uuids
        if not missing_uuids:
            return result

        logger.warning("Purging deleted/inaccessible chunks from citations: %s", missing_uuids)

        # Filter global citations
        surviving_citations = [c for c in result.citations if c.chunk_id in existing_uuids]
        missing_source_ids = {c.source_id for c in result.citations if c.chunk_id in missing_uuids}

        # Update claims
        updated_claims: List[ValidatedClaim] = []
        for claim in result.claims:
            surviving_claim_citations = [c for c in claim.citations if c.chunk_id in existing_uuids]
            surviving_cids = [cid for cid in claim.citation_ids if cid not in missing_source_ids]
            purged_cids = [cid for cid in claim.citation_ids if cid in missing_source_ids]

            updated_claims.append(
                ValidatedClaim(
                    claim_text=claim.claim_text,
                    citation_ids=surviving_cids,
                    citations=surviving_claim_citations,
                    is_supported=len(surviving_claim_citations) > 0,
                    unsupported_citation_ids=sorted(set(claim.unsupported_citation_ids + purged_cids)),
                )
            )

        updated_invalid_ids = sorted(set(result.invalid_citation_ids + list(missing_source_ids)))

        return CitationValidationResult(
            answer=result.answer,
            is_answered=result.is_answered,
            claims=updated_claims,
            citations=surviving_citations,
            invalid_citation_ids=updated_invalid_ids,
            insufficient_evidence_reason=result.insufficient_evidence_reason,
            evidence_sources=result.evidence_sources,
            parsing_status=result.parsing_status,
        )

    @classmethod
    def _build_validated_citation(cls, src: ContextSource) -> ValidatedCitation:
        """Converts internal ContextSource to an immutable ValidatedCitation object."""
        return ValidatedCitation(
            source_id=src.source_id,
            chunk_id=src.chunk_id,
            document_id=src.document_id,
            filename=src.filename,
            page_start=src.page_start,
            page_end=src.page_end,
            text_snippet=src.text,
            dense_score=src.dense_score,
            bm25_score=src.bm25_score,
            rrf_score=src.rrf_score,
            reranker_score=src.reranker_score,
            final_rank=src.final_rank,
        )


citation_validator = CitationValidator()
