"""Concurrency, semaphore bounding, timeout handling, and resource leak tests."""

import asyncio
import time
from typing import Any, Dict, List, Optional
import uuid
import pytest

from app.schemas.rag import RAGQueryRequest
from app.services.llm.base import (
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    LLMTimeoutError,
    ProviderHealth,
)
from app.services.rag.orchestrator import RAGOrchestrator


class MockSlowProvider(LLMProvider):
    """Mock LLM provider that simulates latency and tracks concurrency."""

    def __init__(self, max_concurrent: int = 2, delay: float = 0.05):
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.delay = delay
        self.active_count = 0
        self.peak_concurrency = 0

    @property
    def provider_name(self) -> str:
        return "mock_slow"

    @property
    def model_name(self) -> str:
        return "slow-test-model"

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, model_name=self.model_name, is_available=True, latency_ms=1.0)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        async with self._semaphore:
            self.active_count += 1
            if self.active_count > self.peak_concurrency:
                self.peak_concurrency = self.active_count

            try:
                await asyncio.sleep(self.delay)
                return GenerationResult(
                    raw_text='{"answer": "Concurrent test answer [S1].", "is_answered": true, "claims": [{"claim_text": "Concurrent test answer.", "citation_ids": ["S1"]}], "insufficient_evidence_reason": null}',
                    model=self.model_name,
                    prompt_tokens=50,
                    completion_tokens=20,
                    total_tokens=70,
                    duration_ms=self.delay * 1000,
                )
            finally:
                self.active_count -= 1


class MockTimeoutProvider(LLMProvider):
    """Mock LLM provider that simulates timeout and verifies permit cleanup."""

    def __init__(self, max_concurrent: int = 1):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.permits_released = 0

    @property
    def provider_name(self) -> str:
        return "mock_timeout"

    @property
    def model_name(self) -> str:
        return "timeout-model"

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, model_name=self.model_name, is_available=True, latency_ms=1.0)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        async with self._semaphore:
            # Simulate a timeout raising an LLMTimeoutError
            raise LLMTimeoutError(
                message="Simulated inference timeout.",
                provider=self.provider_name,
                model=self.model_name,
            )


@pytest.mark.asyncio
async def test_semaphore_concurrency_bounding():
    """Verify that semaphore limits max active concurrent LLM inferences to configured bound."""
    provider = MockSlowProvider(max_concurrent=2, delay=0.08)

    # Launch 6 concurrent generation tasks
    tasks = [
        provider.generate(f"Query {i}")
        for i in range(6)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 6
    # Active concurrent requests must NEVER exceed the configured semaphore limit of 2
    assert provider.peak_concurrency == 2
    assert provider.active_count == 0


@pytest.mark.asyncio
async def test_timeout_releases_semaphore_permits():
    """Verify that failed requests and timeouts release semaphore permits without leaking."""
    provider = MockTimeoutProvider(max_concurrent=1)

    # First request raises timeout
    with pytest.raises(LLMTimeoutError):
        await provider.generate("Failing query")

    # Second request must not deadlock on semaphore!
    with pytest.raises(LLMTimeoutError):
        await provider.generate("Second query")

    # Ensure semaphore is not locked
    assert not provider._semaphore.locked()


@pytest.mark.asyncio
async def test_cancellation_releases_semaphore_permit():
    """Verify that cancelling an in-flight inference task properly releases semaphore permit."""
    provider = MockSlowProvider(max_concurrent=1, delay=1.0)

    task = asyncio.create_task(provider.generate("Query to cancel"))
    await asyncio.sleep(0.02)  # Allow task to enter semaphore

    assert provider._semaphore.locked()

    # Cancel task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Semaphore must be immediately unlocked and usable
    assert not provider._semaphore.locked()
    assert provider.active_count == 0


@pytest.mark.asyncio
async def test_event_loop_non_blocking_during_retrieval():
    """Verify that offloading synchronous operations to asyncio.to_thread prevents event loop lag."""
    loop = asyncio.get_running_loop()

    def heavy_cpu_task():
        time.sleep(0.05)
        return "done"

    start_time = loop.time()

    # Run heavy task in threadpool while running a concurrent sleep on the event loop
    async def loop_tick():
        await asyncio.sleep(0.01)
        return loop.time() - start_time

    heavy_coro = asyncio.to_thread(heavy_cpu_task)
    tick_coro = loop_tick()

    heavy_res, tick_latency = await asyncio.gather(heavy_coro, tick_coro)

    assert heavy_res == "done"
    # Event loop tick should execute well before the 50ms heavy task finishes
    assert tick_latency < 0.04
