import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { DocumentsPage } from '../pages/Documents';
import { documentService } from '../services/documentService';
import type { DocumentItem, ChunkListResponse, EmbeddingModelInfo } from '../types';

const mockModelInfo: EmbeddingModelInfo = {
  model_name: 'BAAI/bge-small-en-v1.5',
  dimension: 384,
  max_sequence_length: 512,
  device: 'cpu',
  batch_size: 32,
  normalize: true,
  metric: 'cosine',
  is_loaded: true,
  description: 'English text optimization',
};

const mockDocs: DocumentItem[] = [
  {
    id: '11111111-1111-1111-1111-111111111111',
    filename: '1111_research_paper.pdf',
    original_filename: 'research_paper.pdf',
    file_type: 'pdf',
    file_size: 1048576,
    status: 'completed',
    total_pages: 5,
    total_characters: 12500,
    total_chunks: 10,
    embedding_status: 'completed',
    embedding_model: 'BAAI/bge-small-en-v1.5',
    total_embedded_chunks: 10,
    embedding_error: null,
    error_message: null,
    created_at: '2026-11-01T12:00:00Z',
    updated_at: '2026-11-01T12:00:05Z',
  },
  {
    id: '22222222-2222-2222-2222-222222222222',
    filename: '2222_corrupted.docx',
    original_filename: 'corrupted.docx',
    file_type: 'docx',
    file_size: 4096,
    status: 'failed',
    total_pages: null,
    total_characters: 0,
    total_chunks: 0,
    embedding_status: 'failed',
    embedding_model: null,
    total_embedded_chunks: 0,
    embedding_error: 'Extraction failed',
    error_message: 'File is not a valid DOCX document',
    created_at: '2026-11-01T12:10:00Z',
    updated_at: '2026-11-01T12:10:02Z',
  },
];

const mockChunks: ChunkListResponse = {
  items: [
    {
      id: 'chunk-1',
      document_id: '11111111-1111-1111-1111-111111111111',
      chunk_index: 0,
      text: 'First chunk of research paper content detailing experimental setup.',
      character_count: 67,
      page_start: 1,
      page_end: 1,
      has_embedding: true,
      embedding_model: 'BAAI/bge-small-en-v1.5',
      embedded_at: '2026-11-01T12:00:05Z',
      metadata_json: { source_page: 1, token_count: 15 },
      created_at: '2026-11-01T12:00:05Z',
    },
    {
      id: 'chunk-2',
      document_id: '11111111-1111-1111-1111-111111111111',
      chunk_index: 1,
      text: 'Second chunk of research paper spanning pages 1 and 2.',
      character_count: 54,
      page_start: 1,
      page_end: 2,
      has_embedding: true,
      embedding_model: 'BAAI/bge-small-en-v1.5',
      embedded_at: '2026-11-01T12:00:05Z',
      metadata_json: { source_page: 1, token_count: 12 },
      created_at: '2026-11-01T12:00:05Z',
    },
  ],
  total: 2,
  skip: 0,
  limit: 20,
};

describe('DocumentsPage Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(documentService, 'getEmbeddingModelInfo').mockResolvedValue(mockModelInfo);
  });

  const renderPage = () => {
    return render(
      <BrowserRouter>
        <DocumentsPage />
      </BrowserRouter>
    );
  };

  it('renders document table with ingested documents and statistics', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: mockDocs,
      total: 2,
      skip: 0,
      limit: 50,
    });

    renderPage();

    expect(await screen.findByText('research_paper.pdf')).toBeInTheDocument();
    expect(screen.getByText('corrupted.docx')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText('Failed')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
  });

  it('renders empty state when no documents have been uploaded', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: [],
      total: 0,
      skip: 0,
      limit: 50,
    });

    renderPage();

    expect(await screen.findByText('No Documents Ingested')).toBeInTheDocument();
  });

  it('validates file size on upload (> 25MB rejected)', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: [],
      total: 0,
      skip: 0,
      limit: 50,
    });

    renderPage();

    const oversizedFile = new File(['a'.repeat(100)], 'huge_file.pdf', { type: 'application/pdf' });
    Object.defineProperty(oversizedFile, 'size', { value: 26 * 1024 * 1024 + 1 });

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [oversizedFile] } });

    expect(await screen.findByText(/exceeds 25 MB limit/i)).toBeInTheDocument();
  });

  it('validates file extension on upload (.exe rejected)', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: [],
      total: 0,
      skip: 0,
      limit: 50,
    });

    renderPage();

    const invalidFile = new File(['binary content'], 'script.exe', { type: 'application/octet-stream' });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    expect(await screen.findByText(/Unsupported file format/i)).toBeInTheDocument();
  });

  it('opens chunk inspector modal and displays chunk provenance', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: mockDocs,
      total: 2,
      skip: 0,
      limit: 50,
    });

    vi.spyOn(documentService, 'getDocumentChunks').mockResolvedValueOnce(mockChunks);

    renderPage();

    expect(await screen.findByText('research_paper.pdf')).toBeInTheDocument();

    const inspectBtn = screen.getAllByTitle('Inspect Chunks')[0];
    const user = userEvent.setup();
    await user.click(inspectBtn);

    expect(await screen.findByText('Chunk Inspector')).toBeInTheDocument();
    expect(screen.getByText(/First chunk of research paper/i)).toBeInTheDocument();
    expect(screen.getByText(/Pages 1–2/)).toBeInTheDocument();
  });

  it('handles document deletion with confirmation dialog', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValue({
      items: mockDocs,
      total: 2,
      skip: 0,
      limit: 50,
    });

    const deleteSpy = vi.spyOn(documentService, 'deleteDocument').mockResolvedValueOnce({
      message: 'Document deleted successfully',
    } as any);

    renderPage();

    expect(await screen.findByText('research_paper.pdf')).toBeInTheDocument();

    const deleteButtons = screen.getAllByTitle('Delete Document');
    const user = userEvent.setup();
    await user.click(deleteButtons[0]);

    // Check confirmation modal appears
    expect(await screen.findByText('Delete Document')).toBeInTheDocument();
    expect(
      screen.getByText(/Are you sure you want to delete/i)
    ).toBeInTheDocument();

    // Click confirm delete
    const confirmBtn = screen.getByRole('button', { name: /Delete Permanently/i });
    await user.click(confirmBtn);

    expect(deleteSpy).toHaveBeenCalledWith('11111111-1111-1111-1111-111111111111');
  });

  it('renders embedding model telemetry banner and document embedding status', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: mockDocs,
      total: 2,
      skip: 0,
      limit: 50,
    });

    renderPage();

    // Verify model telemetry banner
    expect(await screen.findByText('BAAI/bge-small-en-v1.5')).toBeInTheDocument();
    expect(screen.getByText('384 Dimensions')).toBeInTheDocument();
    expect(screen.getByText('Device: CPU')).toBeInTheDocument();

    // Verify embedding column statuses
    expect(screen.getByText('Embedded')).toBeInTheDocument();
    expect(screen.getByText('(10/10)')).toBeInTheDocument();
    expect(screen.getByText('Embed Failed')).toBeInTheDocument();
  });

  it('triggers document re-embedding on user action', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValue({
      items: mockDocs,
      total: 2,
      skip: 0,
      limit: 50,
    });

    const reembedSpy = vi.spyOn(documentService, 'reembedDocument').mockResolvedValueOnce({
      document_id: '11111111-1111-1111-1111-111111111111',
      status: 'completed',
      total_embedded_chunks: 10,
      embedding_model: 'BAAI/bge-small-en-v1.5',
      message: 'Successfully re-embedded 10 chunks using BAAI/bge-small-en-v1.5.',
    });

    renderPage();

    expect(await screen.findByText('research_paper.pdf')).toBeInTheDocument();

    const reembedBtn = screen.getAllByTitle('Re-embed Chunks')[0];
    const user = userEvent.setup();
    await user.click(reembedBtn);

    expect(reembedSpy).toHaveBeenCalledWith('11111111-1111-1111-1111-111111111111');
    expect(
      await screen.findByText(/Successfully re-embedded 10 chunks/i)
    ).toBeInTheDocument();
  });

  it('displays vector embedding indicator and token count in chunk inspector', async () => {
    vi.spyOn(documentService, 'getDocuments').mockResolvedValueOnce({
      items: mockDocs,
      total: 2,
      skip: 0,
      limit: 50,
    });

    vi.spyOn(documentService, 'getDocumentChunks').mockResolvedValueOnce(mockChunks);

    renderPage();

    expect(await screen.findByText('research_paper.pdf')).toBeInTheDocument();

    const inspectBtn = screen.getAllByTitle('Inspect Chunks')[0];
    const user = userEvent.setup();
    await user.click(inspectBtn);

    expect(await screen.findByText('Chunk Inspector')).toBeInTheDocument();
    expect(screen.getAllByText('384d Vector').length).toBe(2);
    expect(screen.getByText('~15 tokens')).toBeInTheDocument();
  });
});
