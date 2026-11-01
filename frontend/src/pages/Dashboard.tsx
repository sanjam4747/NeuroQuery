import React from 'react';
import { useOutletContext } from 'react-router-dom';
import {
  Server,
  Database,
  Binary,
  GitBranch,
  ShieldCheck,
  Cpu,
  Layers,
  ArrowRight,
  ExternalLink,
  Clock,
  Sparkles,
} from 'lucide-react';
import { StatusCard } from '../components/StatusCard';
import type { HealthCheckResponse, PlannedFeature } from '../types';

interface ContextType {
  health: HealthCheckResponse | null;
  loading: boolean;
  refetch: () => void;
}

export const Dashboard: React.FC = () => {
  const { health } = useOutletContext<ContextType>();

  const isApiOnline = health?.status === 'healthy' || health?.status === 'degraded';
  const isDbConnected = Boolean(health?.database.connected);
  const isVectorReady = Boolean(health?.database.vector_extension);

  const plannedFeatures: PlannedFeature[] = [
    {
      title: 'Foundation & Telemetry',
      phase: 'Phase 1',
      description: 'FastAPI backend, SQLAlchemy 2.0 ORM, pgvector initialization, and React dashboard shell.',
      status: 'completed',
      category: 'Ingestion',
    },
    {
      title: 'Document Ingestion & Chunking',
      phase: 'Phase 2',
      description: 'PDF, DOCX, and TXT parsing, layout-aware normalization, and semantic sliding-window chunking.',
      status: 'completed',
      category: 'Ingestion',
    },
    {
      title: 'Embeddings & pgvector Storage',
      phase: 'Phase 3',
      description: 'Local BAAI/bge-small-en-v1.5 Sentence-Transformers (384d), L2 cosine normalization, and pgvector persistence.',
      status: 'completed',
      category: 'Retrieval',
    },
    {
      title: 'Hybrid Retrieval Engine',
      phase: 'Phase 4',
      description: 'Dual-stream retrieval blending pgvector semantic similarity with Lucene BM25 sparse search, RRF fusion, and Cross-Encoder reranking.',
      status: 'completed',
      category: 'Retrieval',
    },
    {
      title: 'Cross-Encoder Reranking',
      phase: 'Phase 5',
      description: 'Deep passage reranking and context compression to optimize LLM input tokens.',
      status: 'planned',
      category: 'Retrieval',
    },
    {
      title: 'Grounded RAG & Citations',
      phase: 'Phase 6',
      description: 'Verifiable document question answering with exact page and passage attribution.',
      status: 'planned',
      category: 'Generation',
    },
    {
      title: 'Multi-Document Comparison',
      phase: 'Phase 7',
      description: 'Cross-corpus discrepancy analysis and semantic contradiction detection.',
      status: 'planned',
      category: 'Analysis',
    },
    {
      title: 'Evaluation Lab',
      phase: 'Phase 8',
      description: 'Automated retrieval benchmarking, faithfulness scoring, and hallucination metrics.',
      status: 'planned',
      category: 'Evaluation',
    },
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Hero / Project Branding */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-brand-950 to-slate-900 text-white p-6 sm:p-8 shadow-lg">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-500/20 text-brand-300 text-xs font-semibold mb-4 border border-brand-400/20">
            <Cpu className="w-3.5 h-3.5" />
            <span>Document Intelligence Platform</span>
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white mb-3">
            NeuroQuery
          </h1>
          <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
            An end-to-end NLP-powered document research and intelligence platform. Built for
            verifiable semantic retrieval, hybrid dense/sparse search, grounded RAG answering with
            precise citations, cross-document contradiction detection, and retrieval evaluation.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-4 text-xs font-mono text-slate-400">
            <div className="flex items-center gap-1.5 bg-white/5 px-2.5 py-1 rounded-md border border-white/10">
              <GitBranch className="w-3.5 h-3.5 text-brand-400" />
              <span>Current: Phase 4 (Hybrid Retrieval Engine & Relevance Evaluation)</span>
            </div>
            <div className="flex items-center gap-1.5 bg-white/5 px-2.5 py-1 rounded-md border border-white/10">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>Production Architecture</span>
            </div>
            {health?.timestamp && (
              <div className="flex items-center gap-1.5 bg-white/5 px-2.5 py-1 rounded-md border border-white/10">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <span>Last Telemetry: {new Date(health.timestamp).toLocaleTimeString()}</span>
              </div>
            )}
          </div>
        </div>

        {/* Decorative background glow */}
        <div className="absolute -right-10 -bottom-10 w-80 h-80 bg-brand-500/10 rounded-full blur-3xl pointer-events-none" />
      </div>

      {/* Real-time System Status Cards */}
      <section aria-labelledby="system-status-heading">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 id="system-status-heading" className="text-lg font-bold text-slate-900 dark:text-slate-100">
              System Infrastructure & Telemetry
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Live status probed directly from the backend API health monitor
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatusCard
            title="FastAPI Backend"
            status={isApiOnline ? 'online' : 'offline'}
            icon={Server}
            description="Core asynchronous REST API engine serving versioned /api/v1 routes."
            detail={isApiOnline ? `API Version: ${health?.version || '0.1.0'}` : 'Service Unreachable'}
            badge="Port 8000"
          />

          <StatusCard
            title="PostgreSQL 16 Engine"
            status={isDbConnected ? 'online' : 'degraded'}
            icon={Database}
            description="Relational persistence layer managed via SQLAlchemy 2.0 and Alembic migrations."
            detail={
              isDbConnected
                ? 'Connection Pool Active'
                : health?.database.error || 'Connection Refused (Offline)'
            }
            badge="Port 5432"
          />

          <StatusCard
            title="pgvector Extension"
            status={isVectorReady ? 'online' : 'offline'}
            icon={Binary}
            description="Vector similarity search index for high-dimensional semantic embeddings."
            detail={isVectorReady ? 'Extension Ready (HNSW/IVFFlat)' : 'Pending DB Migration'}
            badge="Vector Ops"
          />

          <StatusCard
            title="API v1 Gateway"
            status={isApiOnline ? 'online' : 'offline'}
            icon={Layers}
            description="CORS-secured API routes with standardized error handling and schemas."
            detail="Prefix: /api/v1"
            badge="REST"
          />
        </div>
      </section>

      {/* Planned Capabilities & Roadmap Matrix */}
      <section aria-labelledby="roadmap-heading">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 id="roadmap-heading" className="text-lg font-bold text-slate-900 dark:text-slate-100">
              Development Roadmap & Architectural Phases
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              NeuroQuery is engineered through phased milestones to ensure rigorous architectural integrity
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {plannedFeatures.map((feat) => {
            const isCompleted = feat.status === 'completed';
            const isCurrent = feat.phase === 'Phase 3';
            return (
              <div
                key={feat.phase}
                className={`p-5 rounded-xl border transition-all ${
                  isCurrent
                    ? 'bg-brand-50/40 dark:bg-brand-950/20 border-brand-200 dark:border-brand-800 shadow-sm'
                    : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span
                    className={`text-xs font-semibold px-2.5 py-0.5 rounded-full ${
                      isCurrent
                        ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                        : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'
                    }`}
                  >
                    {feat.phase}
                  </span>
                  <span className="text-[11px] font-mono font-medium text-slate-400">
                    {feat.category}
                  </span>
                </div>

                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 mb-1.5 flex items-center gap-1.5">
                  {isCurrent && <Sparkles className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400" />}
                  {feat.title}
                </h3>
                <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed mb-4">
                  {feat.description}
                </p>

                <div className="pt-3 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-xs">
                  <span className="text-slate-500 dark:text-slate-400">Status</span>
                  <span
                    className={`font-medium ${
                      isCompleted
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-slate-400'
                    }`}
                  >
                    {isCompleted ? 'Implemented & Verified' : 'Roadmap Milestone'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Developer Quick Reference */}
      <section aria-labelledby="developer-quick-ref-heading">
        <h2 id="developer-quick-ref-heading" className="text-lg font-bold text-slate-900 dark:text-slate-100 mb-4">
          Quick Developer Actions
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="group p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-brand-300 dark:hover:border-brand-700 transition-all flex items-center justify-between"
          >
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 group-hover:text-brand-600 transition-colors">
                OpenAPI Specification
              </h3>
              <p className="text-xs text-slate-500">Interactive Swagger UI</p>
            </div>
            <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-brand-500 transition-colors" />
          </a>

          <a
            href="http://127.0.0.1:8000/api/v1/health"
            target="_blank"
            rel="noreferrer"
            className="group p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-brand-300 dark:hover:border-brand-700 transition-all flex items-center justify-between"
          >
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 group-hover:text-brand-600 transition-colors">
                Raw Health Payload
              </h3>
              <p className="text-xs text-slate-500">Direct /api/v1/health JSON</p>
            </div>
            <ExternalLink className="w-4 h-4 text-slate-400 group-hover:text-brand-500 transition-colors" />
          </a>

          <a
            href="https://github.com/sanjam4747/NeuroQuery/blob/main/docs/architecture.md"
            target="_blank"
            rel="noreferrer"
            className="group p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-brand-300 dark:hover:border-brand-700 transition-all flex items-center justify-between"
          >
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 group-hover:text-brand-600 transition-colors">
                Architecture Spec
              </h3>
              <p className="text-xs text-slate-500">View docs/architecture.md</p>
            </div>
            <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-brand-500 transition-colors" />
          </a>
        </div>
      </section>
    </div>
  );
};
