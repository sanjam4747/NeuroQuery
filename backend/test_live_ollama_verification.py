"""Live verification script for local Ollama service and qwen2.5:1.5b-instruct."""

import asyncio
import json
import time
import httpx

from app.core.config import settings
from app.services.llm.ollama_provider import OllamaProvider
from app.services.rag.citation_validator import CitationValidator
from app.services.rag.context_builder import ContextSource
from app.services.rag.prompt_builder import PromptBuilder


async def run_live_ollama_checks():
    print("=" * 70)
    print("            NEUROQUERY LIVE OLLAMA VERIFICATION")
    print("=" * 70)

    # 1. GET /api/tags
    print("\n1. Testing GET /api/tags...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        res = await client.get("http://127.0.0.1:11434/api/tags")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        tags = res.json()
        models = [m["name"] for m in tags.get("models", [])]
        print(f"   [PASS] HTTP 200 OK. Available models: {models}")

    # 2. Presence of qwen2.5:1.5b-instruct
    print("\n2. Checking presence of qwen2.5:1.5b-instruct...")
    target_model = "qwen2.5:1.5b-instruct"
    model_found = any(target_model in m for m in models)
    assert model_found, f"Model {target_model} not found in {models}"
    print(f"   [PASS] Target model '{target_model}' is present and verified.")

    # 3. Provider Health Check
    provider = OllamaProvider(
        base_url="http://127.0.0.1:11434",
        model_name=target_model,
        timeout_seconds=120.0,
    )
    health = await provider.health_check()
    print(f"\n3. Provider Health Check:")
    print(f"   - Is Available: {health.is_available}")
    print(f"   - Latency: {health.latency_ms:.2f} ms")
    assert health.is_available, f"Provider health check failed: {health.error_message}"

    # 4. Actual Generation with Evidence (Positive grounded query)
    print("\n4. Testing Live Generation on Grounded Query...")
    source_s1 = ContextSource(
        source_id="S1",
        chunk_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        filename="neuroquery_spec.pdf",
        page_start=3,
        page_end=3,
        text="NeuroQuery uses BAAI/bge-small-en-v1.5 to compute 384-dimensional dense semantic embeddings with L2 normalization.",
        token_count=25,
    )
    allowed_sources = {"S1": source_s1}

    evidence_block = (
        '<document_evidence id="S1" filename="neuroquery_spec.pdf" page="3">\n'
        'NeuroQuery uses BAAI/bge-small-en-v1.5 to compute 384-dimensional dense semantic embeddings with L2 normalization.\n'
        '</document_evidence>'
    )
    question = "What embedding model and dimension does NeuroQuery use?"
    sys_prompt = PromptBuilder.get_system_prompt()
    user_prompt = PromptBuilder.build_user_prompt(question=question, formatted_context=evidence_block)
    schema = PromptBuilder.get_json_schema()

    gen_start = time.perf_counter()
    gen_result = await provider.generate(
        prompt=user_prompt,
        system_prompt=sys_prompt,
        json_schema=schema,
    )
    gen_latency_ms = (time.perf_counter() - gen_start) * 1000

    print(f"   - Generation Latency: {gen_latency_ms:.2f} ms")
    print(f"   - Prompt Tokens: {gen_result.prompt_tokens}")
    print(f"   - Completion Tokens: {gen_result.completion_tokens}")
    print(f"   - Raw LLM Text: {gen_result.raw_text}")

    # Validate structured JSON & citations
    val_res = CitationValidator.parse_and_validate(gen_result.raw_text, allowed_sources)
    print(f"   - Is Answered: {val_res.is_answered}")
    print(f"   - Answer: {val_res.answer}")
    print(f"   - Validated Citations: {[c.source_id for c in val_res.citations]}")
    print(f"   - Invalid Citation IDs: {val_res.invalid_citation_ids}")
    print(f"   - Claims Count: {len(val_res.claims)}")
    for i, c in enumerate(val_res.claims):
        print(f"     * Claim {i+1}: '{c.claim_text}' (citations: {c.citation_ids}, supported: {c.is_supported})")

    assert val_res.is_answered is True, "Expected model to answer with evidence"
    assert "S1" in [c.source_id for c in val_res.citations], "Expected model to cite [S1]"
    assert len(val_res.invalid_citation_ids) == 0, f"Unexpected invalid citations: {val_res.invalid_citation_ids}"
    assert all(c.is_supported for c in val_res.claims), "Expected claims to be supported"
    print("   [PASS] Grounded question answered with valid citations and structured JSON.")

    # 5. Live Handling of Insufficient Evidence / Unrelated Query
    print("\n5. Testing Live Handling of Insufficient Evidence / Unrelated Query...")
    unrelated_question = "How do you bake sourdough bread with wild yeast?"
    unrelated_prompt = PromptBuilder.build_user_prompt(question=unrelated_question, formatted_context=evidence_block)

    abstain_start = time.perf_counter()
    abstain_gen = await provider.generate(
        prompt=unrelated_prompt,
        system_prompt=sys_prompt,
        json_schema=schema,
    )
    abstain_latency_ms = (time.perf_counter() - abstain_start) * 1000

    print(f"   - Generation Latency: {abstain_latency_ms:.2f} ms")
    print(f"   - Raw LLM Text: {abstain_gen.raw_text}")

    abstain_val = CitationValidator.parse_and_validate(abstain_gen.raw_text, allowed_sources)
    print(f"   - Is Answered: {abstain_val.is_answered}")
    print(f"   - Answer: {abstain_val.answer}")
    print(f"   - Insufficient Reason: {abstain_val.insufficient_evidence_reason}")
    print(f"   - Invalid Citation IDs: {abstain_val.invalid_citation_ids}")

    assert abstain_val.is_answered is False, "Expected model to abstain when evidence is insufficient"
    assert abstain_val.insufficient_evidence_reason is not None, "Expected insufficient_evidence_reason"
    assert len(abstain_val.citations) == 0, "Expected 0 citations on abstention"
    print("   [PASS] Model correctly abstained on unrelated question with valid JSON and honest reason.")

    print("\n" + "=" * 70)
    print("      ALL LIVE OLLAMA VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_live_ollama_checks())
