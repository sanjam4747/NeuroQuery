import { useState, useEffect, useCallback } from 'react';
import { healthService } from '../services/healthService';
import type { HealthCheckResponse } from '../types';

const POLL_INTERVAL = Number(import.meta.env.VITE_POLL_INTERVAL_MS) || 30000;

export function useHealthCheck() {
  const [health, setHealth] = useState<HealthCheckResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const checkHealth = useCallback(async () => {
    setLoading(true);
    try {
      const data = await healthService.getHealth();
      setHealth(data);
      setLastChecked(new Date());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return {
    health,
    loading,
    lastChecked,
    refetch: checkHealth,
  };
}
