import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '../App';
import { healthService } from '../services/healthService';

describe('App Integration', () => {
  it('renders application branding, sidebar navigation, and dashboard overview', async () => {
    vi.spyOn(healthService, 'getHealth').mockResolvedValueOnce({
      status: 'healthy',
      version: '0.1.0',
      environment: 'development',
      timestamp: '2026-11-01T12:00:00Z',
      database: {
        connected: true,
        vector_extension: true,
      },
    });

    render(<App />);

    // Brand and hero checks
    expect(screen.getAllByText('NeuroQuery').length).toBeGreaterThan(0);
    expect(screen.getByText('Document Intelligence Platform')).toBeInTheDocument();
    expect(screen.getByText('Current: Phase 1 (Foundation)')).toBeInTheDocument();

    // Check navigation links
    expect(screen.getAllByText('Documents').length).toBeGreaterThan(0);
    expect(screen.getByText('Research Assistant')).toBeInTheDocument();
    expect(screen.getByText('Doc Comparison')).toBeInTheDocument();
    expect(screen.getAllByText('Evaluation Lab').length).toBeGreaterThan(0);
  });

  it('navigates to placeholder page when clicking on sidebar links', async () => {
    const user = userEvent.setup();
    render(<App />);

    const docsLink = screen.getByText('Documents');
    await user.click(docsLink);

    expect(
      screen.getByText('Document Ingestion & Management')
    ).toBeInTheDocument();
    expect(screen.getByText('Phase 2 Milestone')).toBeInTheDocument();

    // Click back to Dashboard
    const backBtn = screen.getByText('Back to Dashboard');
    await user.click(backBtn);

    expect(screen.getByText('System Infrastructure & Telemetry')).toBeInTheDocument();
  });

  it('displays real-time backend connection status badge', async () => {
    vi.spyOn(healthService, 'getHealth').mockResolvedValueOnce({
      status: 'degraded',
      version: '0.1.0',
      environment: 'development',
      timestamp: '2026-11-01T12:00:00Z',
      database: {
        connected: false,
        vector_extension: false,
        error: 'PostgreSQL connection refused',
      },
    });

    render(<App />);

    expect(await screen.findByText('API Degraded')).toBeInTheDocument();
  });
});
