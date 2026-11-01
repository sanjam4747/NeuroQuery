import React from 'react';
import { Link } from 'react-router-dom';
import { FlaskConical, ArrowLeft, Clock, CheckCircle2 } from 'lucide-react';

export const EvaluationPage: React.FC = () => {
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
          <div className="p-3 rounded-xl bg-teal-50 text-teal-600 dark:bg-teal-950 dark:text-teal-400">
            <FlaskConical className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                Evaluation Lab & Observability
              </h1>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300">
                Phase 8 Milestone
              </span>
            </div>
            <p className="text-xs text-slate-500">Scheduled for Phase 8 implementation</p>
          </div>
        </div>

        <p className="text-sm text-slate-600 dark:text-slate-400 leading-relaxed mb-6">
          The Evaluation Lab provides automated empirical benchmarking of retrieval recall, MRR,
          context relevance, answer faithfulness, and latency telemetry.
        </p>

        <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-800 mb-6">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
            Planned Technical Capabilities
          </h2>
          <ul className="space-y-2.5 text-xs text-slate-600 dark:text-slate-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
              <span>
                <strong>Retrieval Metrics</strong>: Context Precision, Context Recall, Mean Reciprocal Rank (MRR), and Hit Rate@K.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
              <span>
                <strong>Generation Faithfulness</strong>: Automated verification verifying that statements are directly grounded in retrieved context.
              </span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
              <span>
                <strong>Model Benchmarking</strong>: Comparative experiments contrasting chunk sizes, embedding models, and rerankers.
              </span>
            </li>
          </ul>
        </div>

        <div className="flex items-center gap-2 p-3.5 rounded-lg bg-blue-50 dark:bg-blue-950/40 text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-900/40 text-xs">
          <Clock className="w-4 h-4 shrink-0" />
          <span>
            Evaluation frameworks require a live retrieval corpus and ground truth test sets established in prior phases.
          </span>
        </div>
      </div>
    </div>
  );
};
