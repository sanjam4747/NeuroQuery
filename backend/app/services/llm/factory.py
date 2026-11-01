"""Factory and lifecycle resolver for LLM providers."""

from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.llm.base import LLMProvider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.ollama_provider import OllamaProvider

_active_provider: Optional[LLMProvider] = None


def get_llm_provider(override_type: Optional[str] = None) -> LLMProvider:
    """
    Returns the configured LLMProvider singleton or an explicit override.
    """
    global _active_provider
    if _active_provider is not None and override_type is None:
        return _active_provider

    provider_type = (override_type or settings.LLM_PROVIDER).strip().lower()

    if provider_type == "ollama":
        provider = OllamaProvider(
            base_url=settings.LLM_BASE_URL,
            model_name=settings.LLM_MODEL_NAME,
            timeout_seconds=settings.LLM_REQUEST_TIMEOUT_SECONDS,
            max_concurrent_requests=settings.LLM_MAX_CONCURRENT_REQUESTS,
        )
    elif provider_type == "mock":
        provider = MockLLMProvider()
    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider_type}'. Supported providers: 'ollama', 'mock'."
        )

    if override_type is None:
        _active_provider = provider
    return provider


def set_default_llm_provider(provider: Optional[LLMProvider]) -> None:
    """Explicitly sets or clears the active provider (used by test suites)."""
    global _active_provider
    _active_provider = provider
