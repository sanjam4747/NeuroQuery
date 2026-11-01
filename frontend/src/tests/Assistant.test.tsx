import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AssistantPage } from '../pages/Assistant';
import { retrievalService } from '../services/retrievalService';
import { assistantService } from '../services/assistantService';
import { documentService } from '../services/documentService';

describe('AssistantPage Search Integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(retrievalService, 'getInfo').mockResolvedValue({
      status: 'operational',
      embedding_model: 'BAAI/bge-small-en-v1.5',
      embedding_dimensions: 384,
      reranker_model: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
      dense_retrieval_available: true,
      bm25_index_status: {
        total_chunks: 14,
        total_documents: 7,
        is_initialized: true,
        last_updated_at: '2026-11-01T12:00:00Z',
      },
      rrf_configuration: {
        rrf_constant: 60,
        dense_top_k: 20,
        bm25_top_k: 20,
        hybrid_top_k: 20,
      },
      reranking_enabled_by_default: true,
      active_inference_device: 'cpu',
    });

    vi.spyOn(documentService, 'getDocuments').mockResolvedValue({
      items: [
        {
          id: 'doc-123',
          filename: 'pgvector_guide.pdf',
          original_filename: 'pgvector_guide.pdf',
          file_type: 'pdf',
          file_size: 1024,
          status: 'completed',
          total_pages: 1,
          total_characters: 500,
          total_chunks: 2,
          error_message: null,
          embedding_status: 'completed',
          embedding_model: 'BAAI/bge-small-en-v1.5',
          total_embedded_chunks: 2,
          embedding_error: null,
          created_at: '2026-11-01T10:00:00Z',
          updated_at: '2026-11-01T10:00:00Z',
        },
      ],
      total: 1,
      skip: 0,
      limit: 100,
    });
  });

  it('renders search input, controls, and initial suggested queries', async () => {
    render(<AssistantPage />);

    expect(screen.getByText('Hybrid Retrieval Engine')).toBeInTheDocument();
    expect(screen.getByText('Phase 4 Active')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Enter natural-language query/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
    expect(screen.getAllByText(/Cross-Encoder Reranking/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Suggested Benchmark Queries/i)).toBeInTheDocument();
  });

  it('executes search query and displays ranked results with provenance and latency', async () => {
    const user = userEvent.setup();
    const mockSearch = vi.spyOn(retrievalService, 'search').mockResolvedValueOnce({
      query: 'pgvector cosine similarity',
      results: [
        {
          chunk_id: 'c-1',
          document_id: 'doc-123',
          filename: 'pgvector_guide.pdf',
          text: 'pgvector provides vector similarity search with HNSW indexes.',
          character_count: 65,
          page_start: 1,
          page_end: 1,
          dense_score: 0.88,
          dense_rank: 1,
          bm25_score: 14.2,
          bm25_rank: 1,
          rrf_score: 0.03278,
          rrf_rank: 1,
          reranker_score: 4.85,
          final_rank: 1,
        },
      ],
      retrieval_metadata: {
        dense_candidates: 20,
        bm25_candidates: 15,
        fused_candidates: 25,
        reranked_candidates: 20,
        returned_results: 1,
        reranking_enabled: true,
        dense_latency_ms: 35.2,
        bm25_latency_ms: 0.4,
        rrf_latency_ms: 0.1,
        reranker_latency_ms: 85.0,
        latency_ms: 120.7,
      },
    });

    render(<AssistantPage />);

    const input = screen.getByPlaceholderText(/Enter natural-language query/i);
    await user.type(input, 'pgvector cosine similarity');

    const searchBtn = screen.getByRole('button', { name: /search/i });
    await user.click(searchBtn);

    await waitFor(() => {
      expect(mockSearch).toHaveBeenCalledWith({
        query: 'pgvector cosine similarity',
        top_k: 5,
        document_ids: null,
        enable_reranking: true,
      });
    });

    // Check result display
    expect(await screen.findByText(/pgvector provides vector similarity search with HNSW indexes/i)).toBeInTheDocument();
    expect(screen.getAllByText('pgvector_guide.pdf').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Page 1')).toBeInTheDocument();
    expect(screen.getByText('#1')).toBeInTheDocument();
    expect(screen.getByText(/120.7 ms/i)).toBeInTheDocument();
  });

  it('expands technical score breakdown on click', async () => {
    const user = userEvent.setup();
    vi.spyOn(retrievalService, 'search').mockResolvedValueOnce({
      query: 'test query',
      results: [
        {
          chunk_id: 'c-99',
          document_id: 'doc-123',
          filename: 'report.pdf',
          text: 'Technical score detail test passage.',
          character_count: 36,
          page_start: 3,
          page_end: 4,
          dense_score: 0.8123,
          dense_rank: 2,
          bm25_score: 8.5,
          bm25_rank: 1,
          rrf_score: 0.032,
          rrf_rank: 1,
          reranker_score: 6.21,
          final_rank: 1,
        },
      ],
      retrieval_metadata: {
        dense_candidates: 10,
        bm25_candidates: 10,
        fused_candidates: 15,
        reranked_candidates: 10,
        returned_results: 1,
        reranking_enabled: true,
        latency_ms: 95,
      },
    });

    render(<AssistantPage />);

    const input = screen.getByPlaceholderText(/Enter natural-language query/i);
    await user.type(input, 'test query');
    await user.click(screen.getByRole('button', { name: /search/i }));

    const expandBtn = await screen.findByText(/View Technical Scores/i);
    await user.click(expandBtn);

    expect(screen.getByText('Dense Semantic')).toBeInTheDocument();
    expect(screen.getByText('0.8123')).toBeInTheDocument();
    expect(screen.getByText('BM25 Sparse')).toBeInTheDocument();
    expect(screen.getByText('8.50')).toBeInTheDocument();
    expect(screen.getByText('RRF Fusion Score')).toBeInTheDocument();
    expect(screen.getByText('Cross-Encoder Score')).toBeInTheDocument();
    expect(screen.getByText('6.210')).toBeInTheDocument();
  });

  it('displays empty state when no matching chunks are found', async () => {
    const user = userEvent.setup();
    vi.spyOn(retrievalService, 'search').mockResolvedValueOnce({
      query: 'unrelated astronaut recipe',
      results: [],
      retrieval_metadata: {
        dense_candidates: 0,
        bm25_candidates: 0,
        fused_candidates: 0,
        reranked_candidates: 0,
        returned_results: 0,
        reranking_enabled: true,
        latency_ms: 22,
      },
    });

    render(<AssistantPage />);

    const input = screen.getByPlaceholderText(/Enter natural-language query/i);
    await user.type(input, 'unrelated astronaut recipe');
    await user.click(screen.getByRole('button', { name: /search/i }));

    expect(await screen.findByText(/No matching document chunks found/i)).toBeInTheDocument();
  });

  it('renders grounded RAG answer with citation chips and claim breakdown', async () => {
    const user = userEvent.setup();
    const mockAsk = vi.spyOn(assistantService, 'ask').mockResolvedValueOnce({
      question: 'What is pgvector used for?',
      answer: 'pgvector provides vector similarity search capabilities with HNSW indexing [S1].',
      is_answered: true,
      claims: [
        {
          claim_text: 'pgvector provides vector similarity search capabilities with HNSW indexing.',
          citation_ids: ['S1'],
          citations: [
            {
              source_id: 'S1',
              chunk_id: 'chunk-101',
              document_id: 'doc-123',
              filename: 'pgvector_guide.pdf',
              page_start: 1,
              page_end: 1,
              text_snippet: 'pgvector provides vector similarity search capabilities with HNSW indexing.',
              dense_score: 0.89,
              bm25_score: 12.5,
              rrf_score: 0.033,
              reranker_score: 5.12,
              final_rank: 1,
            },
          ],
        },
      ],
      citations: [
        {
          source_id: 'S1',
          chunk_id: 'chunk-101',
          document_id: 'doc-123',
          filename: 'pgvector_guide.pdf',
          page_start: 1,
          page_end: 1,
          text_snippet: 'pgvector provides vector similarity search capabilities with HNSW indexing.',
          dense_score: 0.89,
          bm25_score: 12.5,
          rrf_score: 0.033,
          reranker_score: 5.12,
          final_rank: 1,
        },
      ],
      evidence_sources: [
        {
          source_id: 'S1',
          chunk_id: 'chunk-101',
          document_id: 'doc-123',
          filename: 'pgvector_guide.pdf',
          page_start: 1,
          page_end: 1,
          text_snippet: 'pgvector provides vector similarity search capabilities with HNSW indexing.',
          dense_score: 0.89,
          bm25_score: 12.5,
          rrf_score: 0.033,
          reranker_score: 5.12,
          final_rank: 1,
        },
      ],
      insufficient_evidence_reason: null,
      invalid_citation_ids: [],
      retrieval_metadata: {
        dense_candidates: 5,
        bm25_candidates: 5,
        fused_candidates: 5,
        reranked_candidates: 5,
        returned_results: 1,
        reranking_enabled: true,
        latency_ms: 45.0,
      },
      generation_metadata: {
        provider: 'ollama',
        model: 'qwen2.5:1.5b-instruct',
        prompt_tokens: 220,
        completion_tokens: 35,
        total_tokens: 255,
        generation_latency_ms: 180.0,
      },
      diagnostics: {
        query_validation_ms: 0.8,
        retrieval_latency_ms: 45.0,
        context_building_ms: 1.2,
        inference_latency_ms: 180.0,
        citation_validation_ms: 0.5,
        total_latency_ms: 227.5,
        evidence_chunks_retrieved: 5,
        evidence_chunks_selected: 1,
        context_tokens_used: 220,
      },
    });

    render(<AssistantPage />);

    const input = screen.getByPlaceholderText(/Enter natural-language query/i);
    await user.type(input, 'What is pgvector used for?');
    await user.click(screen.getByRole('button', { name: /search/i }));

    await waitFor(() => {
      expect(mockAsk).toHaveBeenCalledWith({
        question: 'What is pgvector used for?',
        top_k: 5,
        document_ids: null,
        enable_reranking: true,
      });
    });

    // Check Grounded Answer card
    expect(await screen.findByText('Grounded Answer')).toBeInTheDocument();
    expect(screen.getByText(/pgvector provides vector similarity search capabilities with HNSW indexing \[S1\]/i)).toBeInTheDocument();

    // Check Citation chips
    expect(screen.getByRole('button', { name: /\[S1\] pgvector_guide\.pdf \(p\. 1\)/i })).toBeInTheDocument();

    // Check Claim Breakdown
    expect(screen.getByText('Factual Claims Breakdown')).toBeInTheDocument();
    expect(screen.getAllByText(/pgvector provides vector similarity search capabilities with HNSW indexing\./i).length).toBeGreaterThanOrEqual(1);
  });

  it('renders abstention UI when evidence is insufficient', async () => {
    const user = userEvent.setup();
    vi.spyOn(assistantService, 'ask').mockResolvedValueOnce({
      question: 'What is the recipe for Mars soil?',
      answer: 'The supplied document collection contains no evidence regarding Mars soil recipes.',
      is_answered: false,
      claims: [],
      citations: [],
      evidence_sources: [],
      insufficient_evidence_reason: 'No retrieved document contains information related to Mars soil recipes.',
      invalid_citation_ids: [],
      retrieval_metadata: {
        dense_candidates: 0,
        bm25_candidates: 0,
        fused_candidates: 0,
        reranked_candidates: 0,
        returned_results: 0,
        reranking_enabled: true,
        latency_ms: 12.0,
      },
      generation_metadata: {
        provider: 'ollama',
        model: 'qwen2.5:1.5b-instruct',
        generation_latency_ms: 0.0,
      },
      diagnostics: {
        query_validation_ms: 0.5,
        retrieval_latency_ms: 12.0,
        context_building_ms: 0.1,
        inference_latency_ms: 0.0,
        citation_validation_ms: 0.1,
        total_latency_ms: 12.7,
        evidence_chunks_retrieved: 0,
        evidence_chunks_selected: 0,
        context_tokens_used: 0,
      },
    });

    render(<AssistantPage />);

    const input = screen.getByPlaceholderText(/Enter natural-language query/i);
    await user.type(input, 'What is the recipe for Mars soil?');
    await user.click(screen.getByRole('button', { name: /search/i }));

    expect(await screen.findByText('Insufficient Evidence')).toBeInTheDocument();
    expect(screen.getByText(/contains no evidence regarding Mars soil recipes/i)).toBeInTheDocument();
    expect(screen.getByText(/No retrieved document contains information related to Mars soil recipes/i)).toBeInTheDocument();
  });
});
