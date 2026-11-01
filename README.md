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
> **Current Status**: **Phase 1 — Project Architecture & Foundation** is complete. Future NLP and RAG capabilities will be introduced iteratively in subsequent phases according to the [Development Roadmap](docs/development-roadmap.md).

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
Run all 8 automated tests (API root, configuration, database degraded/healthy state handling):
```powershell
backend\.venv\Scripts\pytest backend/tests -v
```

### Frontend Tests (Vitest)
Run all 9 automated component and integration tests:
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

## API Specifications

### `GET /api/v1/health`
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
| 2 | Document Ingestion & Chunking Pipeline | Planned |
| 3 | Embedding Generation & pgvector Storage | Planned |
| 4 | Hybrid Retrieval Engine (Vector + BM25) | Planned |
| 5 | Cross-Encoder Reranking & Context Compression | Planned |
| 6 | Grounded RAG & Verifiable Citations | Planned |
| 7 | Multi-Document Comparison & Contradictions | Planned |
| 8 | Evaluation Lab & Observability Metrics | Planned |
| 9 | Production Hardening, Authentication & CI/CD | Planned |

See [docs/development-roadmap.md](docs/development-roadmap.md) for full phase breakdowns.

---

## License

This project is licensed under the [MIT License](LICENSE).
