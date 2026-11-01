import { apiClient } from './api';
import type {
  DocumentItem,
  DocumentListResponse,
  ChunkListResponse,
  EmbeddingModelInfo,
  ReembedResponse,
} from '../types';

export const documentService = {
  /**
   * Retrieves a paginated list of ingested documents.
   */
  async getDocuments(skip = 0, limit = 20, status?: string): Promise<DocumentListResponse> {
    const params: Record<string, any> = { skip, limit };
    if (status) {
      params.status = status;
    }
    const response = await apiClient.get<DocumentListResponse>('/documents', { params });
    return response.data;
  },

  /**
   * Retrieves metadata and processing metrics for a single document.
   */
  async getDocument(documentId: string): Promise<DocumentItem> {
    const response = await apiClient.get<DocumentItem>(`/documents/${documentId}`);
    return response.data;
  },

  /**
   * Retrieves paginated chunks with provenance for a specific document.
   */
  async getDocumentChunks(documentId: string, skip = 0, limit = 50): Promise<ChunkListResponse> {
    const response = await apiClient.get<ChunkListResponse>(`/documents/${documentId}/chunks`, {
      params: { skip, limit },
    });
    return response.data;
  },

  /**
   * Uploads a document (PDF, DOCX, TXT) for ingestion and chunking.
   */
  async uploadDocument(file: File): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<DocumentItem>('/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Permanently deletes a document, its cascading chunks, and storage file.
   */
  async deleteDocument(documentId: string): Promise<{ message: string; id: string }> {
    const response = await apiClient.delete<{ message: string; id: string }>(`/documents/${documentId}`);
    return response.data;
  },

  /**
   * Triggers re-generation of semantic vector embeddings for a document's chunks.
   */
  async reembedDocument(documentId: string): Promise<ReembedResponse> {
    const response = await apiClient.post<ReembedResponse>(`/documents/${documentId}/reembed`);
    return response.data;
  },

  /**
   * Retrieves metadata regarding the active Sentence-Transformers embedding model.
   */
  async getEmbeddingModelInfo(): Promise<EmbeddingModelInfo> {
    const response = await apiClient.get<EmbeddingModelInfo>('/embeddings/info');
    return response.data;
  },
};

