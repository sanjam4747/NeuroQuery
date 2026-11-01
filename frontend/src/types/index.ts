export interface DatabaseHealth {
  connected: boolean;
  vector_extension: boolean;
  error?: string | null;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'disconnected';
  version: string;
  environment: string;
  timestamp: string;
  database: DatabaseHealth;
}

export interface ApiRootInfo {
  name: string;
  version: string;
  docs_url: string;
  status: string;
}

export type ConnectionStatus = 'healthy' | 'degraded' | 'disconnected' | 'checking';

export interface PlannedFeature {
  title: string;
  phase: string;
  description: string;
  status: 'completed' | 'in-progress' | 'planned';
  category: 'Ingestion' | 'Retrieval' | 'Generation' | 'Analysis' | 'Evaluation';
}

// ==============================================================================
// Phase 2 Document & Chunk Models
// ==============================================================================

export type DocumentStatus = 'uploaded' | 'processing' | 'completed' | 'failed';

export interface DocumentItem {
  id: string;
  filename: string;
  original_filename: string;
  file_type: 'pdf' | 'docx' | 'txt' | string;
  file_size: number;
  status: DocumentStatus;
  total_pages: number | null;
  total_characters: number;
  total_chunks: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  skip: number;
  limit: number;
}

export interface ChunkItem {
  id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  character_count: number;
  page_start: number | null;
  page_end: number | null;
  metadata_json: Record<string, any>;
  created_at: string;
}

export interface ChunkListResponse {
  items: ChunkItem[];
  total: number;
  skip: number;
  limit: number;
}
