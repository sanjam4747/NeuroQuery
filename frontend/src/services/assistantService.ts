import { apiClient } from './api';
import type { RetrievalMetadata } from './retrievalService';

export interface RAGCitationItem {
  source_id: string;
  chunk_id: string;
  document_id: string;
  filename: string;
  page_start?: number | null;
  page_end?: number | null;
  text_snippet: string;
  dense_score?: number | null;
  bm25_score?: number | null;
  rrf_score: number;
  reranker_score?: number | null;
  final_rank: number;
}

export interface RAGClaimItem {
  claim_text: string;
  citation_ids: string[];
  citations: RAGCitationItem[];
  is_supported?: boolean;
  unsupported_citation_ids?: string[];
}

export interface RAGGenerationMetadata {
  provider: string;
  model: string;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
  generation_latency_ms: number;
}

export interface RAGDiagnostics {
  query_validation_ms: number;
  retrieval_latency_ms: number;
  context_building_ms: number;
  inference_latency_ms: number;
  citation_validation_ms: number;
  total_latency_ms: number;
  evidence_chunks_retrieved: number;
  evidence_chunks_selected: number;
  context_tokens_used: number;
}

export interface RAGQueryRequest {
  question: string;
  document_ids?: string[] | null;
  top_k?: number;
  enable_reranking?: boolean;
  similarity_threshold?: number | null;
  temperature?: number | null;
  max_output_tokens?: number | null;
}

export interface RAGQueryResponse {
  question: string;
  answer: string;
  is_answered: boolean;
  claims: RAGClaimItem[];
  citations: RAGCitationItem[];
  evidence_sources: RAGCitationItem[];
  insufficient_evidence_reason?: string | null;
  invalid_citation_ids: string[];
  parsing_status?: string;
  retrieval_metadata: RetrievalMetadata;
  generation_metadata: RAGGenerationMetadata;
  diagnostics: RAGDiagnostics;
}

export interface RAGHealthResponse {
  status: 'healthy' | 'degraded' | 'unavailable';
  provider: string;
  model: string;
  provider_available: boolean;
  latency_ms?: number | null;
  details: Record<string, unknown>;
  error_message?: string | null;
}

export const assistantService = {
  async ask(request: RAGQueryRequest): Promise<RAGQueryResponse> {
    const response = await apiClient.post<RAGQueryResponse>('/assistant/ask', request);
    return response.data;
  },

  async getHealth(): Promise<RAGHealthResponse> {
    const response = await apiClient.get<RAGHealthResponse>('/assistant/health');
    return response.data;
  },
};
