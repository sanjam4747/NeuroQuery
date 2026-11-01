# NeuroQuery — System Architecture

## 1. System Overview

**NeuroQuery** is an end-to-end NLP-powered document intelligence and research platform. It enables semantic analysis, hybrid retrieval (vector similarity + BM25 keyword search), document-grounded question answering (RAG) with verifiable citations, cross-document comparison, and retrieval/generation evaluation.

This document details the architectural foundation established in **Phase 1: Project Architecture and Foundation**.

---

## 2. Architecture Diagram

```mermaid
flowchart TB
    subgraph Client ["Client Layer (Browser)"]
        UI["React 19 + TypeScript + Vite"]
        TW["Tailwind CSS UI Shell"]
        State["React Router & Health Hooks"]
    end

    subgraph Gateway ["API & Application Layer"]
        FastAPI["FastAPI App (ASGI / Uvicorn)"]
        CORS["CORS Middleware"]
        V1Router["API Router (/api/v1)"]
        HealthSvc["Health & Telemetry Service"]
        RootSvc["API Info Service"]
    end

    subgraph DataLayer ["Data & Storage Layer"]
        SQLA["SQLAlchemy 2.0 ORM & Core Engine"]
        Alembic["Alembic Migrations"]
        PG["PostgreSQL 16 Engine"]
        PGVector["pgvector Extension (HNSW/IVFFlat)"]
    end

    UI --> |REST / JSON via Axios| FastAPI
    FastAPI --> CORS
    CORS --> V1Router
    V1Router --> HealthSvc
    V1Router --> RootSvc
    HealthSvc --> |Safe Health Probe| SQLA
    SQLA --> |Connection Pool| PG
    PG --- PGVector
    Alembic -.-> |Applies Schema & Extensions| PG
```

---

## 3. Technology Stack & Technical Rationale

| Layer | Technology | Selection Rationale |
| :--- | :--- | :--- |
| **Frontend Framework** | React 19, TypeScript | Strict type safety, rich component ecosystem, predictable state management. |
| **Build Tool** | Vite | Instant HMR, native ES modules in development, highly optimized Rollup production builds. |
| **Styling** | Tailwind CSS | Utility-first, zero runtime overhead, responsive layout primitives, seamless dark mode. |
| **Icons & UI** | Lucide React | Lightweight, tree-shakeable, modern consistent icon set. |
| **HTTP Client** | Axios | Request/response interceptors, timeout handling, typed response contracts. |
| **Backend Framework** | FastAPI (Python 3.11+) | Native async support, automatic OpenAPI/Swagger documentation, Pydantic validation, high throughput. |
| **Data Validation** | Pydantic v2 & `pydantic-settings` | High-performance Rust-backed schema validation, environment variable parsing. |
| **ORM / Data Access** | SQLAlchemy 2.0 | Modern typed 2.0 query syntax, robust connection pooling, transactional consistency. |
| **Database Migrations** | Alembic | Trackable, version-controlled schema evolution with custom migration steps. |
| **Database & Vectors** | PostgreSQL 16 + `pgvector` | Battle-tested relational database combined with vector similarity indexing (Cosine, L2, Inner Product) in a single unified data store. |

---

## 4. Repository Structure & Separation of Concerns

```text
NeuroQuery/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints/      # Request handlers (health, root, future docs/rag)
│   │   │       └── router.py       # Aggregate v1 routing definitions
│   │   ├── core/                   # Global configuration, settings, logging
│   │   ├── db/                     # Engine creation, session lifecycle, declarative base
│   │   ├── models/                 # SQLAlchemy ORM entity models
│   │   ├── schemas/                # Pydantic validation & serialization models
│   │   ├── services/               # Core business & domain logic
│   │   ├── utils/                  # Generic reusable helper functions
│   │   └── main.py                 # FastAPI application factory & lifecycle
│   ├── alembic/                    # Database migration environment & versions
│   ├── tests/                      # Automated unit and integration tests
│   ├── requirements.txt            # Production & dev dependencies
│   └── .env.example                # Safe backend environment template
├── frontend/
│   ├── src/
│   │   ├── assets/                 # Static visual assets and styles
│   │   ├── components/             # Reusable UI components (Header, Sidebar, Badges)
│   │   ├── hooks/                  # Custom React hooks (useHealthCheck, useTheme)
│   │   ├── layouts/                # Base layout templates (MainLayout)
│   │   ├── pages/                  # Route views (Dashboard, Documents, Assistant, etc.)
│   │   ├── services/               # HTTP client instances and API endpoint wrappers
│   │   ├── types/                  # Shared TypeScript interfaces & types
│   │   ├── utils/                  # Helper utilities (cn, formatting)
│   │   ├── App.tsx                 # Root application component with routing
│   │   ├── index.css               # Global Tailwind CSS directives & theme variables
│   │   └── main.tsx                # Frontend entrypoint
│   ├── public/                     # Static assets served as-is (favicon, icons)
│   ├── package.json                # Frontend dependencies and scripts
│   └── vite.config.ts              # Vite configuration with dev proxy
├── docs/                           # Architecture, roadmap, and onboarding documentation
├── docker-compose.yml              # Local container orchestrations (PostgreSQL + pgvector)
├── .gitignore                      # Multi-language Git exclusion rules
├── LICENSE                         # Project open-source license
└── README.md                       # Main documentation & getting started guide
```

---

## 5. Backend Architecture & Resilience Patterns

### 5.1 Graceful Database Degradation
The backend must boot and serve HTTP requests even if PostgreSQL is offline or restarting.
- **Engine Initialization**: The SQLAlchemy engine is instantiated at startup without making eager blocking network calls.
- **Safe Health Probing**: The `/api/v1/health` endpoint tests connectivity using a short socket timeout.
  - If PostgreSQL is reachable: Executes `SELECT 1` and verifies `pgvector` presence via `pg_extension`. Reports `"connected": true` and `"vector_extension": true`.
  - If PostgreSQL is unreachable: Catches connection exceptions safely, logs the diagnostic message internally, and responds with `"connected": false` and `"vector_extension": false` without crashing the application or returning a 500 error that crashes health checks.
- **No Credential Leaks**: Connection errors in responses are sanitized to prevent database credentials, hostnames, or internal IPs from leaking to clients.

### 5.2 Centralized Configuration
- Configured via `pydantic-settings` in `app/core/config.py`.
- Enforces strict type validation on environment variables (ports, database URLs, CORS origins).
- Follows the Twelve-Factor App methodology.

---

## 6. Database & Migration Strategy

- **SQLAlchemy 2.0 Declarative Base**: Base metadata defined in `app/db/base.py`.
- **Alembic Environment**: Configured in `backend/alembic/` to import `Base.metadata`.
- **pgvector Integration**:
  - The migration script includes `op.execute("CREATE EXTENSION IF NOT EXISTS vector;")` to ensure vector operations (`vector` type, `<=>`, `<#>` operators) are available.
  - Future phases will define embedding columns using `from pgvector.sqlalchemy import Vector`.

---

## 7. Frontend Architecture

- **Layout Structure**: Collapsible responsive sidebar, dynamic header with environment and real-time backend connection status badge.
- **Real-time Health Checking**: `useHealthCheck` hook polls the backend health status periodically (every 30 seconds) and provides immediate status indication (Connected, Degraded, Disconnected).
- **Theme Support**: Light mode as default research UI, with dark mode toggle powered by Tailwind CSS class strategy.
- **Extensible Pages**:
  - **Dashboard**: System overview, architecture summary, live telemetry, and roadmap tracking.
  - **Documents**: Planned ingestion pipeline placeholder.
  - **Research Assistant**: Planned conversational RAG placeholder.
  - **Document Comparison**: Planned multi-document contradiction & diff placeholder.
  - **Evaluation Lab**: Planned retrieval & faithfulness metrics placeholder.

---

## 8. Security & Compliance

1. **Environment Separation**: No `.env` files are tracked in source control. Comprehensive `.env.example` files guide developers.
2. **CORS Governance**: Configurable allowed origins (`BACKEND_CORS_ORIGINS`) defaulting strictly to development host ports (`http://localhost:5173`, `http://127.0.0.1:5173`).
3. **No Unauthenticated Secret Leaks**: Health and diagnostic endpoints report operational boolean status without exposing underlying connection strings or credentials.
4. **Storage Security**: Filenames are sanitized and prepended with random UUID prefixes to prevent namespace collisions and directory traversal attacks (`../`). Absolute storage paths are strictly withheld from API responses.

---

## 9. Document Ingestion & Chunking Architecture (Phase 2)

```mermaid
flowchart TD
    Upload[Client Upload: PDF / DOCX / TXT] --> Val[DocumentValidator: Magic Bytes & Size Limits]
    Val -->|Valid| Store[StorageService: Path Traversal Guard & UUID Renaming]
    Store --> DBRec[SQLAlchemy: Insert Document Record 'uploaded']
    DBRec --> Ext[Format Extractors: PyMuPDF / python-docx / txt]
    Ext --> Norm[Text Preprocessor: Conservative Normalization]
    Norm --> Chunk[DocumentChunker: Semantic Sizing & Overlap]
    Chunk --> DBPersist[SQLAlchemy: Bulk Insert Chunks + Status 'completed']
    DBPersist --> Done[Client Response: Document & Chunk Metadata]

    Val -->|Invalid| Err[Reject 400 Bad Request]
    Ext -->|Failure| Fail[Mark Document 'failed' + Safe Error + Clean Disk]
    Chunk -->|Failure| Fail
```

### 9.1 Multi-Tier File Validation
- **Extension & Magic Bytes**: Files are inspected for both appropriate file extensions and file header signatures:
  - **PDF**: Verified against `%PDF-` binary magic bytes. Encrypted PDFs are detected and rejected.
  - **DOCX**: Verified against OpenXML ZIP container headers (`PK\x03\x04`). Corrupted or non-OpenXML archives are rejected.
  - **TXT**: Verified for decodability with standard encodings (UTF-8, Latin-1, CP1252) while rejecting non-text binary streams.
- **Size Limits**: Enforces a configurable upload ceiling (default 25 MB / `MAX_UPLOAD_SIZE_BYTES=26214400`) before file streaming commences.
- **Empty File Protection**: 0-byte uploads or documents with no extractable text are rejected with clear error codes.

### 9.2 Isolated Document Storage
- **Directory**: Uploaded documents are saved under `storage/documents` (configurable via `STORAGE_DIR`), isolated completely from frontend and backend application source trees.
- **Collision Resistance**: Stored files are named `{uuid}_{sanitized_basename}`. Original filenames are preserved as metadata in the database.
- **Path Traversal Protection**: Basenames are sanitized using `re.sub(r'[^a-zA-Z0-9_.-]', '_', ...)` and strictly verified with `os.path.commonpath` to ensure target paths remain within the designated storage root.
- **Disk Synchronization**: Deleting a document removes its associated file from local disk.

### 9.3 Format-Specific Text Extraction
- **PDF Extraction (`PyMuPDF`)**:
  - Extracts text page-by-page, recording exact page provenance (`page_number`).
  - Gracefully skips blank or image-only pages without crashing.
  - Detects completely blank PDFs and flags an unextractable document error.
- **DOCX Extraction (`python-docx`)**:
  - Iterates through document paragraphs and tables in sequential order.
  - Preserves paragraph structural provenance (`paragraph_index`) without fabricating artificial page numbers.
- **Plain Text Extraction (`TXT`)**:
  - Automatically handles UTF-8, UTF-8-BOM, Latin-1, and CP1252 encodings with fallbacks.
  - Splits paragraphs along double line breaks (`\r?\n\r?\n`) preserving natural document structure.

### 9.4 Conservative Text Preprocessing
- **Line Ending Normalization**: Standardizes all line breaks to `\n`.
- **Artifact Cleaning**: Heuristically mends broken PDF hyphenated words across line wraps (`word-\nbreak` -> `wordbreak`).
- **Ligature Normalization**: Unfolds common typesetting ligatures (`ﬁ` -> `fi`, `ﬂ` -> `fl`, `ﬀ` -> `ff`, etc.).
- **Whitespace Collapsing**: Collapses repeated internal spacing and excessive line breaks while preserving paragraph boundaries.
- **Information Preservation**: Does **not** perform aggressive stopword removal, stemming, or lemmatization, ensuring search and retrieval models receive verbatim document context.

### 9.5 Intelligent Semantic Chunking
- **Sizing Strategy**: Uses character-based sizing (default: 1500 characters ≈ 375 tokens, overlap: 200 characters ≈ 50 tokens).
- **Semantic Boundary Hierarchy**: Prioritizes splitting at paragraph breaks (`\n\n`), then sentence endings (`[.!?]\s+`), and finally word boundaries (`\s+`) to prevent mid-sentence or mid-word cuts.
- **Infinite Loop Safeguards**: Guarantees advance step is at least 1 character (`step = max(1, chunk_size - chunk_overlap)`) even if overlap is set equal to chunk size.
- **Provenance Tracking**:
  - **PDF chunks**: Calculate exact span `page_start` and `page_end`.
  - **DOCX/TXT chunks**: Record `paragraph_start` and `paragraph_end` in `metadata_json`.

### 9.6 Database Schema & Cascading Deletion
- **`documents` Table**: Primary key UUID, file metadata (`filename`, `original_filename`, `file_type`, `file_size`), processing state (`uploaded`, `processing`, `completed`, `failed`), metrics (`total_pages`, `total_characters`, `total_chunks`), `error_message`, and UTC timestamps.
- **`chunks` Table**: Primary key UUID, foreign key `document_id` referencing `documents.id` with `ON DELETE CASCADE`, sequential `chunk_index`, verbatim `text`, `character_count`, `page_start`, `page_end`, `metadata_json`, and a unique constraint on `(document_id, chunk_index)`.
- **Integrity**: Deleting a parent document automatically removes all associated chunks via PostgreSQL foreign key cascades.

---

## 10. Phase 3: Semantic Embedding Generation & Vector Storage Architecture

### 10.1 Embedding Model Selection & Configuration
- **Model**: `BAAI/bge-small-en-v1.5` loaded locally via `sentence-transformers`.
- **Dimensionality**: Exactly 384 dimensions.
- **Similarity Metric**: Cosine similarity, achieved by computing unit-norm L2 embeddings (`normalize_embeddings=True`).
- **Privacy & Security**: Zero document content or metadata is transmitted to external APIs; inference runs completely on-premises.
- **Language Optimization**: Documented as specialized and optimized for English text.
- **Hardware Acceleration**: Automatic device detection (`cuda` if available, gracefully falling back to `cpu`).

### 10.2 Tokenizer Compatibility & Lossless Oversized Chunk Handling
- **Tokenizer Limits**: The underlying BERT architecture of `BAAI/bge-small-en-v1.5` enforces a strict sequence length ceiling of `512` wordpiece tokens (accounting for `[CLS]` and `[SEP]` boundary tokens, yielding 510 usable content tokens).
- **Chunk Sizing Analysis**: Phase 2 character-based chunking creates windows of 1,500 characters with 200 character overlap. In typical prose, 1,500 characters translates to approximately 280–350 tokens, comfortably within the 512-token budget.
- **Handling Oversized Chunks (Lossless Token-Weighted Sub-Span Pooling)**:
  - When dense or repetitive alphanumeric text causes a chunk to exceed 510 tokens, the chunk is **never** silently truncated.
  - **Sub-Window Token Segmentation**: The chunk's token sequence is segmented into overlapping sub-windows ($W = 450$ tokens, overlap $O = 50$ tokens) via `split_into_token_windows`. This ensures that every token is covered with zero omission while preserving context across window boundaries.
  - **Inference & Token-Weighted Pooling**: Each sub-window is encoded by the embedding model. Rather than relying on simple unweighted averaging, sub-vectors are combined using **token-weighted linear combination**:
    $$V = \frac{\sum_{k} t_k \cdot v_k}{\sum_{k} t_k}$$
    where $t_k$ is the token length of window $k$ and $v_k$ is its dense representation.
  - **Unit L2 Normalization**: The resulting composite vector is L2-normalized:
    $$\hat{v} = \frac{V}{\|V\|_2}$$
    ensuring it resides on the unit hypersphere with shape `(384,)` and an L2 norm of `1.0`, completely compatible with cosine similarity vector indices.
  - **Provenance & Storage Implication**:
    - **No Duplicate Database Chunks**: Original Phase 2 chunk IDs, text, sequential indices, and parent document relationships remain strictly 1:1. Creating duplicate database chunks would distort document structure, complicate cascading deletion, and skew BM25 retrieval.
    - **Metadata Tracking**: Full provenance is recorded in `chunk.metadata_json`:
      - `token_count`: Total token count of the chunk.
      - `oversized_chunk`: Boolean flag set to `True`.
      - `sub_chunk_count`: Number of sub-windows processed.
      - `embedding_method`: `"token_weighted_subchunk_pooling"`.
      - `sub_chunks`: Array of token span bounds `[{"window": k, "token_start": s, "token_end": e, "token_count": n}]`.
      - `oversized_info`: Human-readable description documenting that all tokens were represented without data loss.
    - **Retrieval Implication**: In Phase 4, cosine distance against `chunks.embedding` using the HNSW index operates seamlessly on the 384d vector representing the entire semantic content of the chunk. There is no need for complex multi-vector aggregation or rank-deduplication during vector search.


### 10.3 Embedding Service Singleton
- **Lifecycle & Memory Management**: Encapsulated in `EmbeddingService` as a thread-safe singleton. Model weights are loaded once upon initialization and shared across all ingestion and re-embedding requests, eliminating repeated load overhead.
- **Batch Processing**: Embeddings are generated in configurable batches (default: 32 chunks) to prevent GPU/CPU memory spikes.
- **Resilience & Validation**:
  - Validates output vectors for non-emptiness and exactly 384 dimensions.
  - Returns empty lists for empty inputs.
  - Catches inference exceptions gracefully and propagates domain-specific errors without leaving orphan records.

### 10.4 Relational Database & pgvector Integration
- **`chunks` Table Vector Columns**:
  - `embedding`: `Vector(384)` storing float32 vector representations.
  - `embedding_model`: String recording the model identifier that generated the vector.
  - `embedded_at`: UTC timestamp of vector generation.
- **`documents` Table Tracking Columns**:
  - `embedding_status`: Status enum (`pending`, `generating`, `completed`, `failed`).
  - `embedding_model`: Active model used for document chunks.
  - `total_embedded_chunks`: Integer count of chunks successfully embedded.
  - `embedding_error`: Error message if embedding pipeline encounters a failure.
- **Index Strategy**: Alembic migration `0003_add_chunk_embeddings.py` defines a Hierarchical Navigable Small World (`hnsw`) index utilizing `vector_cosine_ops` when running against PostgreSQL with pgvector.
- **Safe Downgrades**: Migration scripts cleanly drop indices and columns on downgrade without corrupting existing records.

### 10.5 Ingestion & Re-Embedding Pipeline
- **Atomic Execution**: Document ingestion executes in a single logical transaction: validation -> extraction -> preprocessing -> chunking -> batch vectorization -> atomic database commit.
- **Failure Isolation**: If embedding generation fails, the document is marked `embedding_status="failed"` with a descriptive error message; completed documents are never marked completed without verified vectors.
- **In-Place Re-Embedding**: Dedicated `reembed_document` pipeline allows regenerating vectors when the embedding model is upgraded or following an operational failure, updating chunk vectors in-place without duplicating chunks.

### 10.6 API & Telemetry Endpoints
- `GET /api/v1/embeddings/info`: Reports model name, dimensionality (384), similarity metric (`cosine`), max sequence length (512), inference device (`cpu`/`cuda`), and status.
- `POST /api/v1/documents/{id}/reembed`: Triggers chunk re-embedding for a document.
- `GET /api/v1/documents/{id}/embeddings`: Retrieves document embedding status and chunk completion count.


