import { describe, it, expect, vi } from 'vitest';
import { healthService } from '../services/healthService';
import { apiClient } from '../services/api';

describe('healthService', () => {
  it('returns valid health data on successful API response', async () => {
    const mockData = {
      status: 'healthy',
      version: '0.1.0',
      environment: 'development',
      timestamp: '2026-11-01T12:00:00Z',
      database: {
        connected: true,
        vector_extension: true,
        error: null,
      },
    };

    vi.spyOn(apiClient, 'get').mockResolvedValueOnce({ data: mockData });

    const result = await healthService.getHealth();
    expect(result.status).toBe('healthy');
    expect(result.database.connected).toBe(true);
    expect(result.database.vector_extension).toBe(true);
  });

  it('handles network failure gracefully and reports disconnected status honestly', async () => {
    vi.spyOn(apiClient, 'get').mockRejectedValueOnce(new Error('Network error (connection refused)'));

    const result = await healthService.getHealth();
    expect(result.status).toBe('disconnected');
    expect(result.database.connected).toBe(false);
    expect(result.database.vector_extension).toBe(false);
    expect(result.database.error).toContain('Network error');
  });
});
