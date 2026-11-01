"""Ollama inference provider implementation using official REST API."""

import asyncio
import time
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings
from app.core.logging import logger
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


class OllamaProvider(LLMProvider):
    """
    Client for Ollama local inference daemon.

    Features:
    - Asynchronous HTTP communication via httpx.AsyncClient.
    - Concurrency bounding via asyncio.Semaphore to avoid CPU/VRAM starvation.
    - Structured output generation via Ollama's native 'format: json' mode.
    - Explicit exception classification (connection failure, missing model, timeout).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_concurrent_requests: Optional[int] = None,
    ):
        self._base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self._model_name = model_name or settings.LLM_MODEL_NAME
        self._timeout_seconds = timeout_seconds or settings.LLM_REQUEST_TIMEOUT_SECONDS
        max_concurrent = max_concurrent_requests or settings.LLM_MAX_CONCURRENT_REQUESTS
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    async def health_check(self) -> ProviderHealth:
        """
        Queries Ollama /api/tags to verify daemon accessibility and presence of configured model.
        """
        start = time.perf_counter()
        url = f"{self._base_url}/api/tags"

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                latency_ms = (time.perf_counter() - start) * 1000

                if response.status_code != 200:
                    return ProviderHealth(
                        provider_name=self.provider_name,
                        model_name=self.model_name,
                        is_available=False,
                        latency_ms=round(latency_ms, 2),
                        error_message=f"Ollama returned HTTP status {response.status_code}",
                    )

                data = response.json()
                models: List[Dict[str, Any]] = data.get("models", [])
                available_names = [m.get("name", "") for m in models]

                # Match model name with or without tag suffix (e.g. 'qwen2.5:1.5b-instruct')
                model_found = any(
                    self.model_name == name
                    or self.model_name == name.split(":")[0]
                    or name.startswith(f"{self.model_name}:")
                    for name in available_names
                )

                if not model_found:
                    return ProviderHealth(
                        provider_name=self.provider_name,
                        model_name=self.model_name,
                        is_available=False,
                        latency_ms=round(latency_ms, 2),
                        details={"available_models": available_names},
                        error_message=(
                            f"Model '{self.model_name}' is not downloaded in Ollama. "
                            f"Run 'ollama pull {self.model_name}' to install it."
                        ),
                    )

                return ProviderHealth(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    is_available=True,
                    latency_ms=round(latency_ms, 2),
                    details={"available_models_count": len(available_names)},
                )

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderHealth(
                provider_name=self.provider_name,
                model_name=self.model_name,
                is_available=False,
                latency_ms=round(latency_ms, 2),
                error_message=(
                    f"Ollama daemon is unreachable at '{self._base_url}'. "
                    "Ensure Ollama is running ('ollama serve')."
                ),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ProviderHealth(
                provider_name=self.provider_name,
                model_name=self.model_name,
                is_available=False,
                latency_ms=round(latency_ms, 2),
                error_message=f"Unexpected health check error: {exc}",
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Executes an inference request via Ollama's /api/chat endpoint.
        """
        cfg = config or GenerationConfig(
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            timeout_seconds=self._timeout_seconds,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            },
        }

        # Enable structured JSON mode
        if json_schema or settings.RAG_ENABLE_STRUCTURED_OUTPUT:
            payload["format"] = json_schema if json_schema else "json"

        if cfg.stop_sequences:
            payload["options"]["stop"] = cfg.stop_sequences

        url = f"{self._base_url}/api/chat"
        start = time.perf_counter()

        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
                    response = await client.post(url, json=payload)

                    duration_ms = (time.perf_counter() - start) * 1000

                    if response.status_code == 404:
                        raise LLMModelNotFoundError(
                            message=f"Model '{self._model_name}' not found on Ollama server at {self._base_url}.",
                            provider=self.provider_name,
                            model=self._model_name,
                        )

                    if response.status_code != 200:
                        raise LLMGenerationError(
                            message=f"Ollama returned HTTP error {response.status_code}: {response.text}",
                            provider=self.provider_name,
                            model=self._model_name,
                        )

                    res_json = response.json()
                    msg = res_json.get("message", {})
                    content = msg.get("content", "")

                    prompt_tokens = res_json.get("prompt_eval_count")
                    completion_tokens = res_json.get("eval_count")
                    total_tokens = None
                    if prompt_tokens is not None and completion_tokens is not None:
                        total_tokens = prompt_tokens + completion_tokens

                    return GenerationResult(
                        raw_text=content,
                        model=res_json.get("model", self._model_name),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        duration_ms=round(duration_ms, 2),
                        finish_reason="done" if res_json.get("done") else None,
                    )

            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise LLMProviderUnavailableError(
                    message=(
                        f"Cannot connect to Ollama daemon at {self._base_url}. "
                        "Please verify Ollama is running and accessible."
                    ),
                    provider=self.provider_name,
                    model=self._model_name,
                ) from exc

            except httpx.ReadTimeout as exc:
                raise LLMTimeoutError(
                    message=(
                        f"Ollama generation timed out after {cfg.timeout_seconds} seconds. "
                        "The model may be overloaded or prompt is too complex."
                    ),
                    provider=self.provider_name,
                    model=self._model_name,
                ) from exc

            except LLMException:
                raise

            except Exception as exc:
                raise LLMGenerationError(
                    message=f"Unexpected error communicating with Ollama: {exc}",
                    provider=self.provider_name,
                    model=self._model_name,
                ) from exc
