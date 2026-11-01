# NeuroQuery — Development Roadmap

This document outlines the phased development plan for **NeuroQuery**, an end-to-end NLP-powered document intelligence and research platform.

---

## Roadmap Overview

```mermaid
gantt
    title NeuroQuery Development Roadmap
    dateFormat  YYYY-MM-DD
    section Foundation
    Phase 1: Architecture & Foundation          :done, p1, 2026-11-01, 7d
    section Ingestion & Search
    Phase 2: Document Processing & Chunking     :done, p2, after p1, 14d
    Phase 3: Embeddings & pgvector Storage      :done, p3, after p2, 14d
    Phase 4: Hybrid Retrieval (Vector + BM25)   :done, p4, after p3, 14d
    section Intelligence & RAG
    Phase 5: Production-Quality RAG & Citations :done, p5, after p4, 14d
    Phase 6: Multi-Doc Comparison & Conflicts   :p6, after p5, 14d
    section Advanced Capabilities
    Phase 7: Advanced Synthesis & Contradictions:p7, after p6, 14d
    Phase 8: Evaluation Lab & Telemetry         :p8, after p7, 14d
    Phase 9: Production Hardening & Auth        :p9, after p8, 14d
```

---

## Phase Details

### Phase 1: Project Architecture and Foundation (Completed)
- [x] Monorepo structure with decoupled backend and frontend services.
- [x] FastAPI application initialization with versioned `/api/v1` routes.
- [x] SQLAlchemy 2.0 configuration with connection pooling and resilient startup.
- [x] Alembic migration setup with pgvector extension initialization.
- [x] Health check probe (`/api/v1/health`) testing database and pgvector presence without leaking secrets.
- [x] Modern React 19 + TypeScript + Vite + Tailwind CSS dashboard shell.
- [x] Live backend connectivity status indicator in frontend UI.
- [x] Docker Compose configuration for PostgreSQL 16 + pgvector.
- [x] Initial automated test suite for backend and frontend.

---

### Phase 2: Document Processing and Chunking Pipeline (Completed)
- [x] Versioned REST API endpoints (`/api/v1/documents`) for upload, paginated listing, detail inspection, chunk retrieval, and cascading deletion.
- [x] Multi-tier validation: extension whitelist, binary magic byte verification (`%PDF-`, OpenXML ZIP headers), decodable text verification, and configurable 25 MB upload size limits.
- [x] Local filesystem storage service under isolated `storage/documents` with collision-proof UUID naming and path traversal protection (`os.path.commonpath`).
- [x] Format-specific extraction modules:
  - **PDF**: PyMuPDF page-by-page extraction with exact page provenance and blank page skipping.
  - **DOCX**: python-docx sequential paragraph and table extraction preserving structural boundaries without artificial page fabrication.
  - **TXT**: Resilient text decoding with UTF-8, Latin-1, and CP1252 fallbacks.
- [x] Conservative preprocessing: line ending normalization, hyphenated line-wrap repair, typesetting ligature unfolding, whitespace collapsing, and complete preservation of punctuation, numbers, and symbols.
- [x] Deterministic semantic chunking: character-based sizing (1500 chars ≈ 375 tokens, 200 overlap ≈ 50 tokens), semantic boundary hierarchy (`\n\n`, `[.!?]\s+`, `\s+`), loop safety guards (`step >= 1`), and provenance retention (`page_start`/`page_end` for PDF; `paragraph_start`/`paragraph_end` for DOCX/TXT).
- [x] Relational persistence with SQLAlchemy 2.0 `Document` and `Chunk` models, unique index `(document_id, chunk_index)`, `ON DELETE CASCADE` foreign keys, and Alembic migration `0002_create_documents_and_chunks.py`.
- [x] React frontend Documents interface with drag-and-drop upload, size/type client validation, processing feedback, status badges, chunk inspection modal, and delete confirmation dialogs.
- [x] Comprehensive test suites: 44 automated backend pytest tests and 15 frontend Vitest tests covering all failure, validation, extraction, chunking, and deletion flows.

---

### Phase 3: Embedding Generation and Vector Storage (Completed)
- [x] Local Sentence-Transformers integration using `BAAI/bge-small-en-v1.5` generating 384-dimensional dense vectors.
- [x] L2 unit-norm vector normalization (`normalize_embeddings=True`) for exact cosine similarity.
- [x] Tokenizer sequence length analysis (512 token limit) and oversize chunk token auditing with warnings.
- [x] Thread-safe `EmbeddingService` singleton with lazy loading, CPU/CUDA hardware detection, and batching (`EMBEDDING_BATCH_SIZE=32`).
- [x] PostgreSQL + `pgvector` persistence schema via `Vector(384)` column on `chunks`, model tracking, and timestamping (verified via SQLite fallback & mocks; live PostgreSQL container unexecuted).
- [x] Document-level embedding tracking (`embedding_status`, `total_embedded_chunks`, `embedding_error`).
- [x] Alembic migration `0003_add_chunk_embeddings.py` adding vector columns and conditional HNSW cosine index (`vector_cosine_ops`).
- [x] Ingestion pipeline extension ensuring atomic vector generation before document completion.
- [x] In-place re-embedding pipeline (`reembed_document`) without chunk duplication.
- [x] Versioned REST API endpoints:
  - `GET /api/v1/embeddings/info`: Active model telemetry and configuration.
  - `POST /api/v1/documents/{id}/reembed`: In-place vector regeneration.
  - `GET /api/v1/documents/{id}/embeddings`: Status and progress inspection.
- [x] Frontend UI integration: active model telemetry banner, embedding status badges, chunk inspector vector indicators, and re-embed actions.
- [x] Comprehensive test coverage: 55 backend tests (model singleton, 384 dimensions, batching, oversize tokens, re-embedding, rollback) and 18 frontend tests.

---

### Phase 4: Hybrid Retrieval Engine & Relevance Evaluation (Completed)
- [x] Dedicated Dense Semantic Retrieval service using `BAAI/bge-small-en-v1.5` (384d) with pgvector cosine distance database queries (`1 - distance`) and SQLite fallback.
- [x] Dedicated Sparse Lexical Retrieval service using `LuceneBM25` with deterministic tokenization preserving technical symbols, identifiers, acronyms, and numeric tokens.
- [x] Non-negative Robertson IDF formula (`LuceneBM25`) preventing negative/zero IDF on small corpora or frequent technical terms.
- [x] Thread-safe BM25 in-memory lifecycle with automatic sync on ingestion, deletion, and on-demand reconstruction.
- [x] Reciprocal Rank Fusion (RRF) service combining dense and lexical rankings ($k=60$) with stable candidate deduplication and deterministic tie-breaking.
- [x] Cross-Encoder Reranking service using `cross-encoder/ms-marco-MiniLM-L-6-v2` loaded once on CPU, with configurable batch scoring on top 20 candidates.
- [x] Versioned REST API endpoints:
  - `POST /api/v1/retrieval/search`: Hybrid search with provenance, timing breakdown, and optional reranking toggle.
  - `GET /api/v1/retrieval/info`: Pipeline telemetry reporting models, BM25 chunk counts, and active inference device.
  - `POST /api/v1/retrieval/rebuild-index`: Rebuilding BM25 index from persistent database chunks.
- [x] Retrieval Evaluation Framework:
  - Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and nDCG@K metrics.
  - Benchmark dataset of 32 natural-language queries across 5 categories with annotated ground truth.
  - Comparative evaluation: Dense only (Recall: 0.9375), BM25 only (Recall: 0.9062), Hybrid RRF (Recall: 0.9688), Hybrid + Reranker (Recall: 1.0000).
- [x] Frontend Research Assistant UI with search controls, top-K selector, cross-encoder toggle, document scope filter, latency funnel stats, and expandable technical score breakdown.
- [x] Comprehensive test coverage: 88 backend pytest tests and 22 frontend Vitest tests with 100% pass rate.

---

### Phase 5: Production-Quality Retrieval-Augmented Generation (RAG) (Completed)
- [x] Dedicated RAG Orchestration Layer (`RAGOrchestrator`) coordinating the pipeline: query validation, scope resolution, hybrid retrieval, cross-encoder reranking, context engineering, prompt assembly, LLM inference, structured output parsing, and citation validation.
- [x] Pluggable Inference Provider Abstraction (`LLMProvider`):
  - `OllamaProvider`: Async HTTP client with semaphore concurrency limiting, timeout protection, model tags inspection, and structured JSON formatting.
  - `MockLLMProvider`: Deterministic provider with controllable fault simulation for zero-dependency CI/CD testing.
  - `LLMProviderFactory`: Environment-configurable provider instantiator (`LLM_PROVIDER`, `LLM_MODEL_NAME`, `LLM_BASE_URL`).
- [x] Token-Aware Context Engineering (`ContextBuilder`):
  - Word-level Jaccard similarity deduplication (`RAG_SIMILARITY_OVERLAP_THRESHOLD=0.85`).
  - Conservative token budgeting (3.8 char/token ratio) reserving headroom for system prompts and model completion.
  - Sequential source identifier assignment (`[S1]`, `[S2]`) with XML-style delimiter boundary framing.
- [x] Grounded Prompt Construction (`PromptBuilder`):
  - Strict exclusive evidence grounding instructions.
  - Explicit indirect prompt injection defense treating document content as untrusted data.
  - Schema-constrained JSON instructions.
- [x] Structured Answer Generation & Robust Parser (`CitationValidator`):
  - Pydantic schema validation (`answer`, `claims`, `is_answered`, `insufficient_evidence_reason`).
  - Markdown fence stripper and plain text fallback parsing.
  - Strict rejection of hallucinated citation tags (`invalid_citation_ids`).
  - Provenance enrichment attaching database page numbers, chunk IDs, and technical retrieval scores.
- [x] Versioned REST API Endpoints:
  - `POST /api/v1/assistant/ask`: End-to-end question answering with diagnostics and latency breakdown.
  - `GET /api/v1/assistant/health`: LLM inference provider reachability and health telemetry.
- [x] RAG Evaluation Framework (`app.evaluation.rag_eval`):
  - Measures Citation Validity Rate (100%), Precision (92.59%), Recall (90.74%), Correct Abstention Rate (60.00%), and Harmful Abstention Rate (0.00%).
- [x] React Frontend Integration:
  - Unified search & Q&A query bar with toggleable RAG generation.
  - Grounded answer presentation with status indicators and abstention alert banners.
  - Interactive citation badges linking claims directly to source passages.
  - Claim breakdown list and collapsible technical retrieval score inspector.
- [x] Automated Test Suites:
  - 119 backend pytest tests (100% pass rate).
  - 24 frontend Vitest tests (100% pass rate).
  - Zero TypeScript build errors on `npm run build`.

---

### Phase 6: Document-Grounded RAG and Citations
- Multi-provider LLM integration (local models via vLLM/Ollama and hosted API providers).
- Structured prompt engineering enforcing strict grounding:
  - "Answer strictly from provided context; state if insufficient information is present."
- Exact passage citations with page and character offset references.
- Interactive citation viewer in frontend allowing users to inspect source snippets in original documents.

---

### Phase 7: Document Comparison and Contradiction Detection
- Cross-document comparative query interface.
- Semantic alignment of claims and findings across multiple uploaded documents.
- Discrepancy and contradiction detection engine highlighting opposing statements.
- Structured tabular comparison exports (claims, consensus, divergences).

---

### Phase 8: Evaluation Lab and Observability
- Automated evaluation framework measuring:
  - **Retrieval Metrics**: Context Relevance, Mean Reciprocal Rank (MRR), Hit Rate@K.
  - **Generation Metrics**: Faithfulness, Answer Relevance, Hallucination Index.
- Interactive evaluation dashboard in frontend for benchmarking chunk sizes, embedding models, and rerankers.
- Tracing and query latency profiling.

---

### Phase 9: Production Hardening, Authentication and Deployment
- User management and authentication (OAuth2 / JWT / RBAC).
- Multi-tenant data isolation for documents and collections.
- Rate limiting and API quota enforcement.
- Production containerization (multi-stage Dockerfiles for backend and frontend).
- Continuous Integration & Deployment (CI/CD) pipelines.
