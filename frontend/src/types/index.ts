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
