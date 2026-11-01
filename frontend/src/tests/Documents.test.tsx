import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { DocumentsPage } from '../pages/Documents';
import { documentService } from '../services/documentService';
import type { DocumentItem, ChunkListResponse } from '../types';

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
      metadata_json: { source_page: 1 },
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
      metadata_json: { source_page: 1 },
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
});
