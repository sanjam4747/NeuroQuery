"""Unit tests for LLM provider abstraction, Ollama error mapping, and MockLLMProvider."""

import pytest
import httpx
from app.services.llm.base import (
    GenerationConfig,
    LLMGenerationError,
    LLMModelNotFoundError,
    LLMProviderUnavailableError,
    LLMTimeoutError,
)
from app.services.llm.factory import get_llm_provider, set_default_llm_provider
from app.services.llm.mock_provider import MockLLMProvider
from app.services.llm.ollama_provider import OllamaProvider


@pytest.mark.asyncio
async def test_mock_provider_generation():
    """Verify MockLLMProvider returns valid structured responses and tracks call count."""
    provider = MockLLMProvider(model_name="mock-test-model")
    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-test-model"

    health = await provider.health_check()
    assert health.is_available is True
    assert health.latency_ms is not None

    result = await provider.generate(
        prompt="What is hybrid search?",
        system_prompt="Answer factually.",
    )
    assert provider.call_count == 1
    assert "Hybrid search" in result.raw_text
    assert result.prompt_tokens is not None
    assert result.completion_tokens is not None


@pytest.mark.asyncio
async def test_mock_provider_fault_injection():
    """Verify MockLLMProvider simulates errors accurately."""
    provider = MockLLMProvider()

    # Unreachable
    provider.simulate_unreachable = True
    with pytest.raises(LLMProviderUnavailableError) as exc_info:
        await provider.generate("test prompt")
    assert "Simulated connection error" in exc_info.value.message

    health = await provider.health_check()
    assert health.is_available is False

    # Model missing
    provider.simulate_unreachable = False
    provider.simulate_model_missing = True
    with pytest.raises(LLMModelNotFoundError):
        await provider.generate("test prompt")

    # Timeout
    provider.simulate_model_missing = False
    provider.simulate_timeout = True
    with pytest.raises(LLMTimeoutError):
        await provider.generate("test prompt")


@pytest.mark.asyncio
async def test_ollama_provider_error_mapping(monkeypatch):
    """Verify OllamaProvider maps low-level httpx errors to high-level domain exceptions."""
    provider = OllamaProvider(base_url="http://127.0.0.1:11434", model_name="qwen2.5:1.5b-instruct")

    # 1. Connection Error
    async def mock_post_connect_error(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_connect_error)
    with pytest.raises(LLMProviderUnavailableError) as exc:
        await provider.generate("test prompt")
    assert "Cannot connect to Ollama daemon" in exc.value.message

    # 2. Timeout Error
    async def mock_post_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("Read timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_timeout)
    with pytest.raises(LLMTimeoutError) as exc:
        await provider.generate("test prompt")
    assert "timed out" in exc.value.message

    # 3. Model Not Found (HTTP 404)
    class Mock404Response:
        status_code = 404
        text = "model 'qwen2.5:1.5b-instruct' not found"

    async def mock_post_404(*args, **kwargs):
        return Mock404Response()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_404)
    with pytest.raises(LLMModelNotFoundError):
        await provider.generate("test prompt")

    # 4. Internal Server Error (HTTP 500)
    class Mock500Response:
        status_code = 500
        text = "internal server error"

    async def mock_post_500(*args, **kwargs):
        return Mock500Response()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_500)
    with pytest.raises(LLMGenerationError):
        await provider.generate("test prompt")


def test_provider_factory_and_injection():
    """Verify get_llm_provider and set_default_llm_provider."""
    mock_p = MockLLMProvider(model_name="injected-mock")
    set_default_llm_provider(mock_p)

    active = get_llm_provider()
    assert active.provider_name == "mock"
    assert active.model_name == "injected-mock"

    set_default_llm_provider(None)  # Reset
