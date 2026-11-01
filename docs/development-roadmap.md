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
    Phase 2: Document Processing & Chunking     :active, p2, after p1, 14d
    Phase 3: Embeddings & pgvector Storage      :p3, after p2, 14d
    Phase 4: Hybrid Retrieval (Vector + BM25)   :p4, after p3, 14d
    section Intelligence & RAG
    Phase 5: Reranking & Context Compression    :p5, after p4, 10d
    Phase 6: Grounded RAG & Citations           :p6, after p5, 14d
    section Advanced Capabilities
    Phase 7: Multi-Doc Comparison & Conflicts   :p7, after p6, 14d
    Phase 8: Evaluation Lab & Telemetry         :p8, after p7, 14d
    Phase 9: Production Hardening & Auth        :p9, after p8, 14d
```

---

## Phase Details

### Phase 1: Project Architecture and Foundation (Current)
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

### Phase 2: Document Processing and Chunking Pipeline
- Document ingestion endpoints supporting `.pdf`, `.docx`, and `.txt` uploads.
- Text extraction with layout awareness (headers, tables, paragraphs, page numbers).
- Preprocessing and normalization pipeline (whitespace, encoding, header/footer cleaning).
- Chunking strategies:
  - Recursive character chunking with configurable overlap.
  - Sentence-aware semantic boundary chunking.
  - Metadata preservation (document ID, page number, section title, character offsets).
- Document and chunk database schemas.

---

### Phase 3: Embedding Generation and Vector Storage
- Text embedding generation service supporting Hugging Face / Sentence-Transformers (e.g., `BAAI/bge-large-en-v1.5`) and OpenAI embeddings.
- pgvector integration with SQLAlchemy:
  - Vector column definitions (`Vector(dim)`).
  - HNSW (Hierarchical Navigable Small World) index creation for sub-millisecond approximate nearest neighbor search.
  - Batched chunk embedding generation with progress tracking.
- Storage of embeddings linked directly to source document chunks.

---

### Phase 4: Hybrid Retrieval Engine
- Full-text search implementation using PostgreSQL `tsvector` / BM25 algorithms.
- Vector similarity search (Cosine similarity / L2 distance via pgvector).
- Hybrid search fusion:
  - Reciprocal Rank Fusion (RRF) algorithm.
  - Normalized score weighting (alpha vector + (1-alpha) BM25).
- Filter support (by document ID, collection, date, metadata attributes).

---

### Phase 5: Reranking and Context Optimization
- Cross-encoder reranking service (e.g., `bge-reranker-large` / Cohere Rerank API).
- Dynamic context window budgeting to prevent context overflow.
- Redundant passage deduplication and filtering.
- Context window compression to focus LLM attention on highest-relevance tokens.

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
