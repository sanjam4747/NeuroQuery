"""Mock LLM Provider for unit testing and deterministic offline evaluation."""

import json
from typing import Any, Callable, Dict, List, Optional
from app.services.llm.base import (
    GenerationConfig,
    GenerationResult,
    LLMGenerationError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMTimeoutError,
    ProviderHealth,
)


class MockLLMProvider(LLMProvider):
    """
    Deterministic test double implementing LLMProvider.
    Allows testing full RAG orchestration, structured parsing, citation validation,
    and failure states without requiring an active Ollama process.
    """

    def __init__(
        self,
        model_name: str = "mock-model-v1",
        is_healthy: bool = True,
        mock_response_generator: Optional[Callable[[str], str]] = None,
    ):
        self._model_name = model_name
        self._is_healthy = is_healthy
        self._mock_response_generator = mock_response_generator
        self.call_count: int = 0
        self.last_prompt: Optional[str] = None
        self.last_system_prompt: Optional[str] = None

        # Configurable fault injection
        self.simulate_unreachable: bool = False
        self.simulate_model_missing: bool = False
        self.simulate_timeout: bool = False
        self.simulate_malformed_json: bool = False
        self.simulate_hallucinated_citation: bool = False
        self.simulate_insufficient_evidence: bool = False

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def health_check(self) -> ProviderHealth:
        if self.simulate_unreachable or not self._is_healthy:
            return ProviderHealth(
                provider_name=self.provider_name,
                model_name=self.model_name,
                is_available=False,
                error_message="Mock provider simulated unreachable daemon",
            )
        if self.simulate_model_missing:
            return ProviderHealth(
                provider_name=self.provider_name,
                model_name=self.model_name,
                is_available=False,
                error_message=f"Model '{self.model_name}' not downloaded",
            )
        return ProviderHealth(
            provider_name=self.provider_name,
            model_name=self.model_name,
            is_available=True,
            latency_ms=1.5,
            details={"mode": "mock"},
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt

        if self.simulate_unreachable:
            raise LLMProviderUnavailableError("Simulated connection error", self.provider_name, self.model_name)
        if self.simulate_model_missing:
            raise LLMModelNotFoundError("Simulated model missing", self.provider_name, self.model_name)
        if self.simulate_timeout:
            raise LLMTimeoutError("Simulated request timeout", self.provider_name, self.model_name)

        if self._mock_response_generator:
            text = self._mock_response_generator(prompt)
            return GenerationResult(
                raw_text=text,
                model=self.model_name,
                prompt_tokens=len(prompt) // 4,
                completion_tokens=len(text) // 4,
                total_tokens=(len(prompt) + len(text)) // 4,
                duration_ms=10.0,
            )

        if self.simulate_malformed_json:
            return GenerationResult(
                raw_text="This is raw invalid text that is not valid JSON at all.",
                model=self.model_name,
                prompt_tokens=50,
                completion_tokens=15,
                total_tokens=65,
                duration_ms=5.0,
            )

        if self.simulate_insufficient_evidence:
            data = {
                "answer": "The supplied documents do not contain sufficient evidence to answer this question.",
                "is_answered": False,
                "claims": [],
                "insufficient_evidence_reason": "No relevant technical passages were found discussing this topic.",
            }
            return GenerationResult(
                raw_text=json.dumps(data),
                model=self.model_name,
                prompt_tokens=80,
                completion_tokens=25,
                total_tokens=105,
                duration_ms=5.0,
            )

        if self.simulate_hallucinated_citation:
            data = {
                "answer": "According to the analysis, hybrid search scales linearly [S99].",
                "is_answered": True,
                "claims": [
                    {
                        "claim_text": "Hybrid search scales linearly.",
                        "citation_ids": ["S99"],
                    }
                ],
                "insufficient_evidence_reason": None,
            }
            return GenerationResult(
                raw_text=json.dumps(data),
                model=self.model_name,
                prompt_tokens=120,
                completion_tokens=30,
                total_tokens=150,
                duration_ms=8.0,
            )

        # Standard well-formed grounded response
        data = {
            "answer": "Hybrid search combines dense vector search with sparse BM25 retrieval to maximize precision and recall [S1]. The results are fused using reciprocal rank fusion [S2].",
            "is_answered": True,
            "claims": [
                {
                    "claim_text": "Hybrid search combines dense vector search with sparse BM25 retrieval to maximize precision and recall.",
                    "citation_ids": ["S1"],
                },
                {
                    "claim_text": "The results are fused using reciprocal rank fusion.",
                    "citation_ids": ["S2"],
                },
            ],
            "insufficient_evidence_reason": None,
        }
        return GenerationResult(
            raw_text=json.dumps(data),
            model=self.model_name,
            prompt_tokens=150,
            completion_tokens=45,
            total_tokens=195,
            duration_ms=12.0,
        )
