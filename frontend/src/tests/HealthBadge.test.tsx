import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { HealthBadge } from '../components/HealthBadge';
import type { HealthCheckResponse } from '../types';

describe('HealthBadge Component', () => {
  it('renders "Probing API..." when loading without health data', () => {
    render(<HealthBadge health={null} loading={true} />);
    expect(screen.getByTestId('health-badge-checking')).toBeInTheDocument();
    expect(screen.getByText('Probing API...')).toBeInTheDocument();
  });

  it('renders "API Operational" when status is healthy', () => {
    const mockHealth: HealthCheckResponse = {
      status: 'healthy',
      version: '0.1.0',
      environment: 'development',
      timestamp: '2026-11-01T12:00:00Z',
      database: {
        connected: true,
        vector_extension: true,
      },
    };

    render(<HealthBadge health={mockHealth} loading={false} />);
    expect(screen.getByTestId('health-badge-healthy')).toBeInTheDocument();
    expect(screen.getByText('API Operational')).toBeInTheDocument();
  });

  it('renders "API Degraded" when database is unavailable', () => {
    const mockHealth: HealthCheckResponse = {
      status: 'degraded',
      version: '0.1.0',
      environment: 'development',
      timestamp: '2026-11-01T12:00:00Z',
      database: {
        connected: false,
        vector_extension: false,
        error: 'PostgreSQL connection refused',
      },
    };

    render(<HealthBadge health={mockHealth} loading={false} showDetails={true} />);
    expect(screen.getByTestId('health-badge-degraded')).toBeInTheDocument();
    expect(screen.getByText('API Degraded')).toBeInTheDocument();
    expect(screen.getByText('PostgreSQL disconnected')).toBeInTheDocument();
  });

  it('renders "API Offline" when backend is unreachable', () => {
    const mockHealth: HealthCheckResponse = {
      status: 'disconnected',
      version: '0.0.0',
      environment: 'offline',
      timestamp: '2026-11-01T12:00:00Z',
      database: {
        connected: false,
        vector_extension: false,
        error: 'Network error',
      },
    };

    render(<HealthBadge health={mockHealth} loading={false} />);
    expect(screen.getByTestId('health-badge-disconnected')).toBeInTheDocument();
    expect(screen.getByText('API Offline')).toBeInTheDocument();
  });
});
