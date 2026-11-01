import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, RefreshCw } from 'lucide-react';
import type { HealthCheckResponse } from '../types';

interface HealthBadgeProps {
  health: HealthCheckResponse | null;
  loading: boolean;
  onRefresh?: () => void;
  showDetails?: boolean;
}

export const HealthBadge: React.FC<HealthBadgeProps> = ({
  health,
  loading,
  onRefresh,
  showDetails = false,
}) => {
  if (loading && !health) {
    return (
      <div
        data-testid="health-badge-checking"
        className="inline-flex items-center gap-2 px-2.5 py-1 text-xs font-medium rounded-full bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border border-slate-200 dark:border-slate-700"
      >
        <span className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" />
        <span>Probing API...</span>
      </div>
    );
  }

  const status = health?.status || 'disconnected';

  const badgeConfig = {
    healthy: {
      bg: 'bg-emerald-50 dark:bg-emerald-950/40',
      text: 'text-emerald-700 dark:text-emerald-400',
      border: 'border-emerald-200 dark:border-emerald-800/60',
      dot: 'bg-emerald-500',
      icon: CheckCircle2,
      label: 'API Operational',
    },
    degraded: {
      bg: 'bg-amber-50 dark:bg-amber-950/40',
      text: 'text-amber-700 dark:text-amber-400',
      border: 'border-amber-200 dark:border-amber-800/60',
      dot: 'bg-amber-500',
      icon: AlertTriangle,
      label: 'API Degraded',
    },
    disconnected: {
      bg: 'bg-rose-50 dark:bg-rose-950/40',
      text: 'text-rose-700 dark:text-rose-400',
      border: 'border-rose-200 dark:border-rose-800/60',
      dot: 'bg-rose-500',
      icon: XCircle,
      label: 'API Offline',
    },
    checking: {
      bg: 'bg-slate-100 dark:bg-slate-800',
      text: 'text-slate-600 dark:text-slate-400',
      border: 'border-slate-200 dark:border-slate-700',
      dot: 'bg-slate-400',
      icon: RefreshCw,
      label: 'Checking...',
    },
  }[status];

  const Icon = badgeConfig.icon;

  return (
    <div className="inline-flex items-center gap-2">
      <div
        data-testid={`health-badge-${status}`}
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border ${badgeConfig.bg} ${badgeConfig.text} ${badgeConfig.border}`}
      >
        <span className={`w-2 h-2 rounded-full ${badgeConfig.dot} ${status === 'healthy' ? 'animate-pulse' : ''}`} />
        <Icon className="w-3.5 h-3.5" />
        <span>{badgeConfig.label}</span>
      </div>

      {showDetails && health && (
        <span className="text-xs text-slate-500 dark:text-slate-400 hidden sm:inline">
          {health.database.connected
            ? health.database.vector_extension
              ? 'PostgreSQL & pgvector ready'
              : 'PostgreSQL connected (pgvector missing)'
            : 'PostgreSQL disconnected'}
        </span>
      )}

      {onRefresh && (
        <button
          onClick={onRefresh}
          disabled={loading}
          title="Refresh health check"
          aria-label="Refresh health check"
          className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors rounded hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      )}
    </div>
  );
};
