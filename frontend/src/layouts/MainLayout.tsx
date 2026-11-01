import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { Header } from '../components/Header';
import { useHealthCheck } from '../hooks/useHealthCheck';
import { useTheme } from '../hooks/useTheme';

export const MainLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const { health, loading, refetch } = useHealthCheck();
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-200">
      <Sidebar
        health={health}
        loading={loading}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="lg:pl-64 flex flex-col min-h-screen">
        <Header
          health={health}
          loading={loading}
          onRefreshHealth={refetch}
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          theme={theme}
          onToggleTheme={toggleTheme}
        />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          <Outlet context={{ health, loading, refetch }} />
        </main>
      </div>
    </div>
  );
};
