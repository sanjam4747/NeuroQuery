import { apiClient } from './api';
import type { HealthCheckResponse, ApiRootInfo } from '../types';

export const healthService = {
  /**
   * Fetches health status from backend.
   * If backend is completely offline, returns a typed disconnected state.
   */
  async getHealth(): Promise<HealthCheckResponse> {
    try {
      const response = await apiClient.get<HealthCheckResponse>('/health');
      return response.data;
    } catch (error: any) {
      return {
        status: 'disconnected',
        version: '0.0.0',
        environment: 'offline',
        timestamp: new Date().toISOString(),
        database: {
          connected: false,
          vector_extension: false,
          error: error.message || 'Unable to connect to backend server',
        },
      };
    }
  },

  /**
   * Fetches root API information.
   */
  async getApiInfo(): Promise<ApiRootInfo | null> {
    try {
      const response = await apiClient.get<ApiRootInfo>('/');
      return response.data;
    } catch {
      return null;
    }
  },
};
