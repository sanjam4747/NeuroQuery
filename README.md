# NeuroQuery — Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-0.5+-blue?style=flat)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NeuroQuery** is an end-to-end NLP-powered document research and intelligence platform. It provides document parsing, semantic chunking, high-dimensional vector embeddings, hybrid retrieval (dense vector similarity + sparse BM25 keyword matching), cross-encoder reranking, and document-grounded question answering (RAG) with verifiable citations, multi-document contradiction detection, and retrieval evaluation.

> [!NOTE]
> **Current Status**: **Phase 4 — Hybrid Retrieval Engine & Relevance Evaluation** is complete and practically verified. The retrieval engine combines dense semantic search (`BAAI/bge-small-en-v1.5`, 384d, pgvector cosine search), sparse lexical retrieval (`LuceneBM25` with non-negative Robertson IDF), Reciprocal Rank Fusion ($k=60$), and Cross-Encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`), achieving 100% recall across our reproducible 32-query benchmark evaluation suite. Detailed in [docs/architecture.md](docs/architecture.md).

---

## Architecture & System Overview

NeuroQuery is built as a modular, decoupled full-stack platform:
- **Backend**: Asynchronous **FastAPI** REST engine using **SQLAlchemy 2.0** ORM, **Alembic** migrations, and **PostgreSQL 16** with the **pgvector** extension.
- **Frontend**: High-performance **React 19** + **TypeScript** SPA powered by **Vite** and styled with **Tailwind CSS**, featuring dark mode, live telemetry health probing, and interactive navigation.
- **Resilient Probing**: The backend boots and operates gracefully even when the database is offline, honestly reporting degraded status through `/api/v1/health` without masking errors or leaking secrets.

Detailed architectural specifications and component diagrams are documented in [docs/architecture.md](docs/architecture.md).

---

## Repository Structure

```text
NeuroQuery/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/                 # Versioned API routes (/health, /)
│   │   ├── core/                   # Centralized settings & logging
│   │   ├── db/                     # SQLAlchemy 2.0 engine, sessions & Base
│   │   ├── models/                 # ORM models (speculative models deferred)
│   │   ├── schemas/                # Pydantic v2 schemas
│   │   ├── services/               # System & health probe business logic
│   │   ├── utils/                  # Shared helper utilities
│   │   └── main.py                 # FastAPI application factory & lifespan
│   ├── tests/                      # Pytest automated test suite
│   ├── alembic/                    # Migration scripts (pgvector extension)
│   ├── requirements.txt            # Python dependencies
│   ├── pytest.ini                  # Pytest test configuration
│   └── .env.example                # Safe environment template
│
├── frontend/
│   ├── src/
│   │   ├── components/             # Reusable UI components (HealthBadge, Sidebar, Header)
│   │   ├── pages/                  # Dashboard & Roadmap milestone placeholders
│   │   ├── layouts/                # Main responsive application layout
│   │   ├── services/               # Axios API client & healthService
│   │   ├── hooks/                  # React hooks (useHealthCheck, useTheme)
│   │   ├── types/                  # Shared TypeScript interfaces
│   │   ├── utils/                  # Tailwind cn utility
│   │   ├── App.tsx                 # Master router configuration
│   │   └── main.tsx                # Client entrypoint
│   ├── public/                     # Static assets (SVG favicon)
│   ├── vite.config.ts              # Vite dev server + Vitest config
│   ├── tailwind.config.js          # Tailwind CSS styling tokens
│   └── package.json                # Dependencies and scripts
│
├── docs/
│   ├── architecture.md             # In-depth architectural design
│   └── development-roadmap.md      # 9-phase development plan
│
├── docker-compose.yml              # Local PostgreSQL 16 + pgvector container
├── .gitignore                      # Multi-tier exclusion rules
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```

---

## Local Development Setup (Windows & Cross-Platform)

### 1. Prerequisites
- **Python**: Version 3.11 or higher (Tested on Python 3.13)
- **Node.js**: Version 18 or higher (Tested on Node v24)
- **Database**: Docker Desktop OR a local PostgreSQL 16+ installation with the `pgvector` extension.

---

### 2. Database Setup

#### Option A: Using Docker Compose (Recommended)
Run the bundled PostgreSQL + pgvector container:
```powershell
docker compose up -d db
```
This launches a PostgreSQL 16 container with `pgvector` pre-installed on port `5432`.

#### Option B: Standalone PostgreSQL on Windows
1. Install PostgreSQL 16 from [postgresql.org](https://www.postgresql.org/download/windows/).
2. Follow the [pgvector installation instructions for Windows](https://github.com/pgvector/pgvector#windows).
3. Create the database and user:
   ```sql
   CREATE USER neuroquery_user WITH PASSWORD 'neuroquery_pass';
   CREATE DATABASE neuroquery OWNER neuroquery_user;
   ```

---

### 3. Backend Setup

1. **Navigate to the backend directory and create a virtual environment**:
   ```powershell
   cd backend
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```powershell
   cp .env.example .env
   ```
   *(Update database credentials in `.env` if using custom credentials).*

4. **Run database migrations**:
   ```powershell
   alembic upgrade head
   ```
   *(This enables the `pgvector` extension in PostgreSQL).*

5. **Start the FastAPI backend server**:
   ```powershell
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
   - API Root: `http://127.0.0.1:8000/api/v1/`
   - Health Probe: `http://127.0.0.1:8000/api/v1/health`
   - Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

---

### 4. Frontend Setup

1. **Navigate to the frontend directory**:
   ```powershell
   cd frontend
   ```

2. **Install dependencies**:
   ```powershell
   npm install
   ```

3. **Configure environment variables** (optional, defaults to proxy):
   ```powershell
   cp .env.example .env
   ```

4. **Start the Vite development server**:
   ```powershell
   npm run dev
   ```
   Access the dashboard at `http://localhost:5173`.

---

## Running Automated Tests

### Backend Tests (Pytest)
Run all 62 automated unit and integration tests (validation, extractors, preprocessing, chunking, word boundary snapping, storage security, pipeline orchestrator, embedding service singleton, 384d dimensionality, vector normalization, direct token ID tensor batch inference, lossless sub-window token pooling for oversized chunks, in-place re-embedding, and API endpoints):
```powershell
backend\.venv\Scripts\pytest backend/tests -v
```

### Frontend Tests (Vitest)
Run all 18 automated component and integration tests (Documents page, model telemetry banner, chunk inspector modal, 384d vector badges, file validation, navigation, health status badge, re-embedding action):
```powershell
cd frontend
npm test
```

### Production Build Verification
Verify type safety and compile production bundle:
```powershell
cd frontend
npm run build
```

---

## Configuration Variables

Configured in `backend/.env` with safe fallbacks in `backend/app/core/config.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `STORAGE_DIR` | `storage/documents` | Directory where uploaded files are stored locally on disk. |
| `MAX_UPLOAD_SIZE_BYTES` | `26214400` (25 MB) | Maximum permitted file upload size. |
| `ALLOWED_EXTENSIONS` | `[".pdf", ".docx", ".txt"]` | Whitelist of supported file extensions. |
| `CHUNK_SIZE` | `1500` | Target chunk size in characters (~375 tokens). |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks in characters (~50 tokens). |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Hugging Face Sentence-Transformers model identifier. |
| `EMBEDDING_DIMENSION` | `384` | Semantic vector embedding dimensionality. |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding generation. |
| `EMBEDDING_DEVICE` | `auto` | Inference device (`auto`, `cpu`, or `cuda`). |
| `EMBEDDING_NORMALIZE` | `True` | Whether to compute L2 unit-norm vectors for exact cosine similarity. |
| `DENSE_RETRIEVAL_TOP_K` | `20` | Candidate passages retrieved by pgvector dense cosine search. |
| `BM25_RETRIEVAL_TOP_K` | `20` | Candidate passages retrieved by LuceneBM25 lexical search. |
| `RRF_CONSTANT` | `60` | Smoothing constant $k$ for Reciprocal Rank Fusion ($1/(k + rank)$). |
| `HYBRID_RETRIEVAL_TOP_K` | `20` | Candidates forwarded from RRF fusion into the cross-encoder. |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sentence-Transformers cross-encoder model identifier. |
| `RERANKER_TOP_K` | `5` | Final top-K ranked results returned to the client. |
| `RERANKER_BATCH_SIZE` | `16` | Batch size for cross-encoder inference. |
| `ENABLE_RERANKING` | `True` | Global default toggle for cross-encoder reranking stage. |
| `DATABASE_URL` | `postgresql+psycopg://...` | Connection URI for PostgreSQL with pgvector. |

---

## API Specifications

### Hybrid Retrieval Endpoints (`/api/v1/retrieval`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/retrieval/search` | Executes multi-stage hybrid search (Dense + BM25 + RRF + Reranker) with timing telemetry and source provenance. |
| `GET` | `/api/v1/retrieval/info` | Reports active embedding model, vector dimensions, reranker model, BM25 index status, and inference hardware. |
| `POST` | `/api/v1/retrieval/rebuild-index` | Reconstructs the in-memory BM25 index from persistent document chunks in the database. |

### Document Processing Endpoints (`/api/v1/documents`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/documents` | Ingests a document (`.pdf`, `.docx`, `.txt`), extracts text, normalizes content, chunks, generates 384d embeddings, and persists atomically. |
| `GET` | `/api/v1/documents` | Retrieves a paginated list of ingested documents with chunk and embedding status. |
| `GET` | `/api/v1/documents/{id}` | Retrieves metadata and processing metrics for a single document. (Storage paths are never exposed). |
| `GET` | `/api/v1/documents/{id}/chunks` | Retrieves paginated chunks with provenance and vector embedding flags. |
| `GET` | `/api/v1/documents/{id}/embeddings` | Retrieves embedding status, active model, and total embedded chunks for a document. |
| `POST` | `/api/v1/documents/{id}/reembed` | Re-generates embeddings for all chunks of a document in-place without chunk duplication. |
| `DELETE` | `/api/v1/documents/{id}` | Permanently deletes the document record, cascades to all generated chunks and vectors, and removes the file from disk. |
| `GET` | `/api/v1/embeddings/info` | Reports active embedding model name, 384 dimensions, cosine metric, inference device, and max sequence length (512). |

### Health Check Endpoint

#### `GET /api/v1/health`
Probes API operational state, PostgreSQL connection, and pgvector extension availability.

**Degraded Response (Database Offline)**:
```json
{
  "status": "degraded",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-01T12:55:19.597Z",
  "database": {
    "connected": false,
    "vector_extension": false,
    "error": "PostgreSQL service unavailable or connection refused"
  }
}
```

**Healthy Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-11-01T12:00:00.000Z",
  "database": {
    "connected": true,
    "vector_extension": true,
    "error": null
  }
}
```

---

## Development Roadmap

| Phase | Milestone | Status |
| :---: | :--- | :---: |
| **1** | **Project Architecture & Foundation** | **Completed** |
| **2** | **Document Processing & Chunking Pipeline** | **Completed** |
| **3** | **Embedding Generation & pgvector Storage** | **Completed** |
| **4** | **Hybrid Retrieval Engine & Relevance Evaluation** | **Completed** |
| 5 | Cross-Encoder Context Optimization & Token Compression | Planned |
| 6 | Grounded RAG & Verifiable Citations | Planned |
| 7 | Multi-Document Comparison & Contradictions | Planned |
| 8 | Evaluation Lab & Observability Metrics | Planned |
| 9 | Production Hardening, Authentication & CI/CD | Planned |

See [docs/development-roadmap.md](docs/development-roadmap.md) for full phase breakdowns.

---

## License

This project is licensed under the [MIT License](LICENSE).
