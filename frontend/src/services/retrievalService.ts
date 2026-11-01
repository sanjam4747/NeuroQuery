import { apiClient } from './api';

export interface RetrievedChunkResult {
  chunk_id: string;
  document_id: string;
  filename: string;
  text: string;
  character_count: number;
  page_start?: number | null;
  page_end?: number | null;
  metadata_json?: Record<string, unknown>;
  dense_score?: number | null;
  dense_rank?: number | null;
  bm25_score?: number | null;
  bm25_rank?: number | null;
  rrf_score: number;
  rrf_rank: number;
  reranker_score?: number | null;
  final_rank: number;
}

export interface RetrievalMetadata {
  dense_candidates: number;
  bm25_candidates: number;
  fused_candidates: number;
  reranked_candidates: number;
  returned_results: number;
  reranking_enabled: boolean;
  dense_latency_ms?: number | null;
  bm25_latency_ms?: number | null;
  rrf_latency_ms?: number | null;
  reranker_latency_ms?: number | null;
  latency_ms: number;
}

export interface RetrievalQueryRequest {
  query: string;
  top_k?: number;
  document_ids?: string[] | null;
  enable_reranking?: boolean;
  similarity_threshold?: number | null;
}

export interface RetrievalQueryResponse {
  query: string;
  results: RetrievedChunkResult[];
  retrieval_metadata: RetrievalMetadata;
}

export interface BM25IndexStatus {
  total_chunks: number;
  total_documents: number;
  is_initialized: boolean;
  last_updated_at?: string | null;
}

export interface RRFConfiguration {
  rrf_constant: number;
  dense_top_k: number;
  bm25_top_k: number;
  hybrid_top_k: number;
}

export interface RetrievalInfoResponse {
  status: string;
  embedding_model: string;
  embedding_dimensions: number;
  reranker_model: string;
  dense_retrieval_available: boolean;
  bm25_index_status: BM25IndexStatus;
  rrf_configuration: RRFConfiguration;
  reranking_enabled_by_default: boolean;
  active_inference_device: string;
}

export interface RebuildIndexResponse {
  status: string;
  message: string;
  indexed_chunks: number;
  timestamp: string;
}

export const retrievalService = {
  async search(request: RetrievalQueryRequest): Promise<RetrievalQueryResponse> {
    const response = await apiClient.post<RetrievalQueryResponse>('/retrieval/search', request);
    return response.data;
  },

  async getInfo(): Promise<RetrievalInfoResponse> {
    const response = await apiClient.get<RetrievalInfoResponse>('/retrieval/info');
    return response.data;
  },

  async rebuildIndex(): Promise<RebuildIndexResponse> {
    const response = await apiClient.post<RebuildIndexResponse>('/retrieval/rebuild-index');
    return response.data;
  },
};
