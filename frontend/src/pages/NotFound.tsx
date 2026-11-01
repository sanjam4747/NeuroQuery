import React from 'react';
import { Link } from 'react-router-dom';
import { HelpCircle, ArrowLeft } from 'lucide-react';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center p-6">
      <div className="p-4 rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-400 mb-4">
        <HelpCircle className="w-10 h-10" />
      </div>
      <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">Page Not Found</h1>
      <p className="text-sm text-slate-500 dark:text-slate-400 max-w-sm mb-6">
        The requested path does not exist in NeuroQuery or has not yet been registered.
      </p>
      <Link
        to="/"
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700 transition-colors shadow-sm"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Return to Dashboard</span>
      </Link>
    </div>
  );
};
