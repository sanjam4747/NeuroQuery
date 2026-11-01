import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FileText,
  Sparkles,
  GitCompare,
  FlaskConical,
  BrainCircuit,
  BookOpen,
  Layers,
  ChevronRight,
} from 'lucide-react';
import type { HealthCheckResponse } from '../types';

interface SidebarProps {
  health: HealthCheckResponse | null;
  loading: boolean;
  isOpen: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ health, isOpen, onClose }) => {
  const navItems = [
    {
      to: '/',
      label: 'Dashboard',
      icon: LayoutDashboard,
      phase: 'Phase 1',
      activeStatus: 'Active',
    },
    {
      to: '/documents',
      label: 'Documents',
      icon: FileText,
      phase: 'Phase 2',
      activeStatus: 'Active',
    },
    {
      to: '/assistant',
      label: 'Research Assistant',
      icon: Sparkles,
      phase: 'Phase 6',
      activeStatus: 'Planned',
    },
    {
      to: '/comparison',
      label: 'Doc Comparison',
      icon: GitCompare,
      phase: 'Phase 7',
      activeStatus: 'Planned',
    },
    {
      to: '/evaluation',
      label: 'Evaluation Lab',
      icon: FlaskConical,
      phase: 'Phase 8',
      activeStatus: 'Planned',
    },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-30 bg-slate-900/40 backdrop-blur-sm lg:hidden"
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-40 w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col transition-transform duration-200 lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand Header */}
        <div className="h-16 px-5 flex items-center gap-3 border-b border-slate-100 dark:border-slate-800">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-700 to-brand-500 flex items-center justify-center text-white shadow-sm shadow-brand-500/20">
            <BrainCircuit className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-base tracking-tight text-slate-900 dark:text-white">
                NeuroQuery
              </span>
              <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300 rounded border border-brand-200 dark:border-brand-800">
                v0.1
              </span>
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 font-medium">
              Document Intelligence
            </p>
          </div>
        </div>

        {/* Navigation Links */}
        <div className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <div className="px-3 pb-2 text-[11px] font-semibold tracking-wider uppercase text-slate-400 dark:text-slate-500">
            Platform Modules
          </div>

          {navItems.map((item) => {
            const Icon = item.icon;
            const isPhaseActive = item.phase === 'Phase 1' || item.phase === 'Phase 2';

            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                onClick={onClose}
                className={({ isActive }) =>
                  `group flex items-center justify-between px-3 py-2.5 text-sm font-medium rounded-lg transition-all ${
                    isActive
                      ? 'bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-100'
                  }`
                }
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4 shrink-0 transition-transform group-hover:scale-110" />
                  <span>{item.label}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span
                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                      isPhaseActive
                        ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300'
                        : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
                    }`}
                  >
                    {item.phase}
                  </span>
                  <ChevronRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </NavLink>
            );
          })}

          <div className="pt-6 px-3 pb-2 text-[11px] font-semibold tracking-wider uppercase text-slate-400 dark:text-slate-500">
            Documentation & System
          </div>

          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-between px-3 py-2 text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-100 rounded-lg transition-colors"
          >
            <div className="flex items-center gap-3">
              <BookOpen className="w-4 h-4" />
              <span>Swagger API Docs</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">/docs</span>
          </a>

          <a
            href="http://127.0.0.1:8000/api/v1/health"
            target="_blank"
            rel="noreferrer"
            className="flex items-center justify-between px-3 py-2 text-sm font-medium text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-100 rounded-lg transition-colors"
          >
            <div className="flex items-center gap-3">
              <Layers className="w-4 h-4" />
              <span>Health Endpoint</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">/health</span>
          </a>
        </div>

        {/* Footer telemetry */}
        <div className="p-3 border-t border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/50">
          <div className="p-2.5 rounded-lg border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-950 text-xs">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-slate-500 dark:text-slate-400 font-medium">Status</span>
              <span
                className={`font-semibold capitalize ${
                  health?.status === 'healthy'
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : health?.status === 'degraded'
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-rose-600 dark:text-rose-400'
                }`}
              >
                {health?.status || 'Connecting...'}
              </span>
            </div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
              {health?.database.connected ? 'PostgreSQL Connected' : 'DB Not Connected'}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
