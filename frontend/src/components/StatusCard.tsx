import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface StatusCardProps {
  title: string;
  status: 'online' | 'degraded' | 'offline' | 'pending';
  icon: LucideIcon;
  description: string;
  detail?: string;
  badge?: string;
}

export const StatusCard: React.FC<StatusCardProps> = ({
  title,
  status,
  icon: Icon,
  description,
  detail,
  badge,
}) => {
  const statusStyles = {
    online: {
      border: 'border-emerald-200 dark:border-emerald-800/40',
      bg: 'bg-emerald-500/5',
      iconColor: 'text-emerald-600 dark:text-emerald-400',
      badgeBg: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300',
      statusText: 'Operational',
    },
    degraded: {
      border: 'border-amber-200 dark:border-amber-800/40',
      bg: 'bg-amber-500/5',
      iconColor: 'text-amber-600 dark:text-amber-400',
      badgeBg: 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300',
      statusText: 'Degraded',
    },
    offline: {
      border: 'border-rose-200 dark:border-rose-800/40',
      bg: 'bg-rose-500/5',
      iconColor: 'text-rose-600 dark:text-rose-400',
      badgeBg: 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300',
      statusText: 'Unavailable',
    },
    pending: {
      border: 'border-slate-200 dark:border-slate-700',
      bg: 'bg-slate-500/5',
      iconColor: 'text-slate-500 dark:text-slate-400',
      badgeBg: 'bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-300',
      statusText: 'Planned',
    },
  }[status];

  return (
    <div
      className={`p-4 rounded-xl border bg-white dark:bg-slate-900 shadow-sm transition-all duration-200 hover:shadow-md ${statusStyles.border}`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className={`p-2.5 rounded-lg ${statusStyles.bg} ${statusStyles.iconColor}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex items-center gap-1.5">
          {badge && (
            <span className="px-2 py-0.5 text-[11px] font-medium rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
              {badge}
            </span>
          )}
          <span
            className={`px-2 py-0.5 text-xs font-semibold rounded-full ${statusStyles.badgeBg}`}
          >
            {statusStyles.statusText}
          </span>
        </div>
      </div>

      <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-1">{title}</h3>
      <p className="text-xs text-slate-600 dark:text-slate-400 line-clamp-2">{description}</p>

      {detail && (
        <div className="mt-3 pt-2.5 border-t border-slate-100 dark:border-slate-800/80">
          <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400 block truncate">
            {detail}
          </span>
        </div>
      )}
    </div>
  );
};
