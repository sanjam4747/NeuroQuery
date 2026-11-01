import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { App } from '../App';
import { healthService } from '../services/healthService';
import { documentService } from '../services/documentService';

describe('App Integration', () => {
  beforeEach(() => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValue({
      items: [],
      total: 0,
      skip: 0,
      limit: 50,
    });
    vi.spyOn(documentService, 'getEmbeddingModelInfo').mockResolvedValue({
      model_name: 'BAAI/bge-small-en-v1.5',
      dimension: 384,
      max_sequence_length: 512,
      device: 'cpu',
      batch_size: 32,
      normalize: true,
      metric: 'cosine',
      is_loaded: true,
      description: 'English text optimization',
    });
  });
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
    expect(screen.getByText('Current: Phase 3 (Semantic Embeddings & Vector Storage)')).toBeInTheDocument();

    // Check navigation links
    expect(screen.getAllByText('Documents').length).toBeGreaterThan(0);
    expect(screen.getByText('Research Assistant')).toBeInTheDocument();
    expect(screen.getByText('Doc Comparison')).toBeInTheDocument();
    expect(screen.getAllByText('Evaluation Lab').length).toBeGreaterThan(0);
  });

  it('navigates to documents page when clicking on sidebar links', async () => {
    const user = userEvent.setup();
    render(<App />);

    const docsLink = screen.getAllByText('Documents')[0];
    await user.click(docsLink);

    expect(
      screen.getByText('Document Ingestion & Chunking')
    ).toBeInTheDocument();
    expect(screen.getByText('Phase 3 Active')).toBeInTheDocument();

    // Click back to Dashboard
    const backBtn = screen.getAllByText('Dashboard')[0];
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
