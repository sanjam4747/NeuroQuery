import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowLeft, Clock, CheckCircle2 } from 'lucide-react';

export const AssistantPage: React.FC = () => {
  return (
    <div className="space-y-6 max-w-4xl mx-auto py-6">
      <Link
        to="/"
        className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-900 dark:hover:text-slate-100 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Dashboard</span>
      </Link>

      <div className="p-8 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 rounded-xl bg-purple-50 text-purple-600 dark:bg-purple-950 dark:text-purple-400">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                Research Assistant & Grounded RAG
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-100 text-purple-800 dark:bg-purple-950 dark:text-purple-300">
                Phase 6 Milestone
              </span>
            </div>
            <p className="text-xs text-slate-500">Scheduled for Phase 6 implementation</p>
          </div>
        </div>

        <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed mb-6">
          The Research Assistant will provide document-grounded question answering powered by
          hybrid retrieval (pgvector dense + BM25 sparse) and verifiable passage citations.
        </p>

        <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 mb-6">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
            Planned Technical Capabilities
          </h2>
          <ul className="space-y-2.5 text-xs text-slate-600 dark:text-slate-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500 shrink-0 mt-0.5" />
              <span>
                <strong>Hybrid Retrieval (Phase 4)</strong>: Reciprocal Rank Fusion of vector distance and BM25 scores.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500 shrink-0 mt-0.5" />
              <span>
                <strong>Passage Reranking (Phase 5)</strong>: Cross-encoder scoring to elevate highest-precision context.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500 shrink-0 mt-0.5" />
              <span>
                <strong>Grounded Generation (Phase 6)</strong>: Context-bounded LLM generation with zero hallucination enforcement.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-purple-500 shrink-0 mt-0.5" />
              <span>
                <strong>Verifiable Citations</strong>: Interactive citations mapping answers directly to source pages and character offsets.
              </span>
            </li>
          </ul>
        </div>

        <div className="flex items-center gap-2 p-3.5 rounded-lg bg-blue-50 dark:bg-blue-950/40 text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-900/40 text-xs">
          <Clock className="w-4 h-4 shrink-0" />
          <span>
            This feature requires the hybrid retrieval and reranking engines from prior phases. No simulated responses are provided in Phase 1.
          </span>
        </div>
      </div>
    </div>
  );
};
