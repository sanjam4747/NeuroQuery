"""LLM Provider abstraction interface and structured generation contracts."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderHealth(BaseModel):
    """Diagnostic health information from an LLM inference provider."""

    provider_name: str
    model_name: str
    is_available: bool
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class GenerationConfig(BaseModel):
    """Runtime generation parameters for LLM inference."""

    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    stop_sequences: List[str] = Field(default_factory=list)


class GenerationResult(BaseModel):
    """Complete output payload from an LLM generation request."""

    raw_text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_ms: float = 0.0
    finish_reason: Optional[str] = None


# =========================================================================
# LLM Provider Exception Hierarchy
# =========================================================================

class LLMException(Exception):
    """Base exception for all LLM inference and provider failures."""

    def __init__(self, message: str, provider: str, model: str):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model


class LLMProviderUnavailableError(LLMException):
    """Raised when the LLM service daemon or endpoint is unreachable."""
    pass


class LLMModelNotFoundError(LLMException):
    """Raised when the requested model is not downloaded or found on the provider."""
    pass


class LLMTimeoutError(LLMException):
    """Raised when generation exceeds the configured request timeout."""
    pass


class LLMGenerationError(LLMException):
    """Raised when inference fails due to bad input, payload rejection, or runtime error."""
    pass


class LLMProvider(ABC):
    """
    Abstract interface for LLM inference providers (e.g. Ollama, Mock, OpenAI-compatible).
    Ensures that RAG orchestration is completely decoupled from any specific inference backend.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the inference provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the configured model."""
        pass

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Performs a non-blocking diagnostic health check on the inference provider."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Executes a prompt completion or chat generation request.

        Args:
            prompt: User-facing prompt or assembled grounded context.
            system_prompt: Optional system instruction prompt.
            json_schema: Optional JSON schema constraint for structured generation.
            config: Optional runtime generation parameters.

        Returns:
            GenerationResult containing raw response text and execution metadata.
        """
        pass
