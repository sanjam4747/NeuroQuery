"""LLM inference provider abstraction package."""

from app.services.llm.base import (
    GenerationConfig,
    GenerationResult,
    LLMException,
    LLMGenerationError,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMTimeoutError,
    ProviderHealth,
)
from app.services.llm.factory import get_llm_provider, set_default_llm_provider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.ollama_provider import OllamaProvider

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "LLMException",
    "LLMGenerationError",
    "LLMModelNotFoundError",
    "LLMProvider",
    "LLMProviderUnavailableError",
    "LLMTimeoutError",
    "ProviderHealth",
    "get_llm_provider",
    "set_default_llm_provider",
    "MockLLMProvider",
    "OllamaProvider",
]
