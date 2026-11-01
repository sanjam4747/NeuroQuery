import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  Upload,
  Trash2,
  Eye,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Clock,
  FileCode,
  FileSpreadsheet,
  X,
  ChevronLeft,
  ChevronRight,
  ArrowLeft,
  Layers,
  Sparkles,
  Cpu,
  Binary,
} from 'lucide-react';
import { documentService } from '../services/documentService';
import type { DocumentItem, ChunkItem, EmbeddingModelInfo } from '../types';

export const DocumentsPage: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [totalDocs, setTotalDocs] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  // Embedding model info
  const [modelInfo, setModelInfo] = useState<EmbeddingModelInfo | null>(null);
  const [reembeddingId, setReembeddingId] = useState<string | null>(null);

  // Inspector modal state
  const [inspectDoc, setInspectDoc] = useState<DocumentItem | null>(null);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunksLoading, setChunksLoading] = useState<boolean>(false);
  const [chunksPage, setChunksPage] = useState<number>(0);
  const [totalChunks, setTotalChunks] = useState<number>(0);
  const CHUNKS_PER_PAGE = 20;

  // Delete modal state
  const [deleteTarget, setDeleteTarget] = useState<DocumentItem | null>(null);
  const [deleting, setDeleting] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    try {
      const res = await documentService.getDocuments(0, 50);
      setDocuments(res.items);
      setTotalDocs(res.total);
    } catch (err: any) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchModelInfo = useCallback(async () => {
    try {
      const info = await documentService.getEmbeddingModelInfo();
      setModelInfo(info);
    } catch (err: any) {
      console.warn('Failed to fetch embedding model info:', err);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
    fetchModelInfo();
  }, [fetchDocuments, fetchModelInfo]);

  // Re-embed handling
  const handleReembed = async (doc: DocumentItem) => {
    setReembeddingId(doc.id);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const res = await documentService.reembedDocument(doc.id);
      setUploadSuccess(
        res.message ||
          `Successfully re-embedded ${res.total_embedded_chunks} chunks for "${doc.original_filename}" using ${res.embedding_model}.`
      );
      await fetchDocuments();
      if (inspectDoc?.id === doc.id) {
        await loadChunks(doc.id, chunksPage);
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Re-embedding failed.';
      setUploadError(`Re-embedding failed: ${detail}`);
    } finally {
      setReembeddingId(null);
    }
  };

  // Load chunks when inspectDoc changes
  const loadChunks = useCallback(async (docId: string, page: number) => {
    setChunksLoading(true);
    try {
      const res = await documentService.getDocumentChunks(docId, page * CHUNKS_PER_PAGE, CHUNKS_PER_PAGE);
      setChunks(res.items);
      setTotalChunks(res.total);
      setChunksPage(page);
    } catch (err: any) {
      console.error('Failed to load chunks:', err);
    } finally {
      setChunksLoading(false);
    }
  }, []);

  const openInspector = (doc: DocumentItem) => {
    setInspectDoc(doc);
    loadChunks(doc.id, 0);
  };

  const closeInspector = () => {
    setInspectDoc(null);
    setChunks([]);
    setChunksPage(0);
    setTotalChunks(0);
  };

  // Upload Handling
  const handleFile = async (file: File) => {
    setUploadError(null);
    setUploadSuccess(null);

    // Client-side extension validation
    const validExtensions = ['.pdf', '.docx', '.txt'];
    const lowerName = file.name.toLowerCase();
    const isValidExt = validExtensions.some((ext) => lowerName.endsWith(ext));
    if (!isValidExt) {
      setUploadError(`Unsupported file format. Please upload a .pdf, .docx, or .txt document.`);
      return;
    }

    // Client-side size validation (25MB)
    const MAX_SIZE = 25 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      setUploadError(`File size exceeds 25 MB limit (${(file.size / (1024 * 1024)).toFixed(2)} MB).`);
      return;
    }

    setUploading(true);
    try {
      const uploadedDoc = await documentService.uploadDocument(file);
      if (uploadedDoc.status === 'failed') {
        setUploadError(`Ingestion warning: ${uploadedDoc.error_message || 'Processing failed'}`);
      } else {
        const embeddedMsg =
          uploadedDoc.embedding_status === 'completed'
            ? ` and generated ${uploadedDoc.total_embedded_chunks} vector embeddings (${uploadedDoc.embedding_model || '384d'})`
            : '';
        setUploadSuccess(
          `Successfully ingested "${uploadedDoc.original_filename}" into ${uploadedDoc.total_chunks} semantic chunks${embeddedMsg}.`
        );
      }
      await fetchDocuments();
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Upload failed.';
      setUploadError(detail);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      await documentService.deleteDocument(deleteTarget.id);
      setDeleteTarget(null);
      await fetchDocuments();
      if (inspectDoc?.id === deleteTarget.id) {
        closeInspector();
      }
    } catch (err: any) {
      console.error('Delete failed:', err);
    } finally {
      setDeleting(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getFormatBadge = (fileType: string) => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300 border border-rose-200 dark:border-rose-900/40">
            <FileText className="w-3 h-3" />
            PDF
          </span>
        );
      case 'docx':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border border-blue-200 dark:border-blue-900/40">
            <FileSpreadsheet className="w-3 h-3" />
            DOCX
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
            <FileCode className="w-3 h-3" />
            TXT
          </span>
        );
    }
  };

  const getStatusBadge = (status: string, error?: string | null) => {
    switch (status) {
      case 'completed':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
            <CheckCircle2 className="w-3 h-3" />
            Completed
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
            <RefreshCw className="w-3 h-3 animate-spin" />
            Processing
          </span>
        );
      case 'failed':
        return (
          <span
            title={error || 'Processing failed'}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border border-rose-200 dark:border-rose-800 cursor-help"
          >
            <AlertCircle className="w-3 h-3" />
            Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300">
            <Clock className="w-3 h-3" />
            {status}
          </span>
        );
    }
  };

  const getEmbeddingStatusBadge = (
    status?: string,
    embeddedCount?: number,
    totalCount?: number,
    error?: string | null
  ) => {
    switch (status) {
      case 'completed':
        return (
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800"
            title={`${embeddedCount ?? 0} of ${totalCount ?? 0} chunks embedded`}
          >
            <Sparkles className="w-3 h-3" />
            <span>Embedded</span>
            <span className="text-[10px] font-mono opacity-80">
              ({embeddedCount ?? 0}/{totalCount ?? 0})
            </span>
          </span>
        );
      case 'generating':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
            <RefreshCw className="w-3 h-3 animate-spin" />
            Embedding...
          </span>
        );
      case 'failed':
        return (
          <span
            title={error || 'Embedding generation failed'}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border border-rose-200 dark:border-rose-800 cursor-help"
          >
            <AlertCircle className="w-3 h-3" />
            Embed Failed
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400">
            <Clock className="w-3 h-3" />
            Pending
          </span>
        );
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto py-2">
      {/* Breadcrumb & Navigation Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 transition-colors mb-2"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Dashboard</span>
          </Link>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-white tracking-tight">
              Document Ingestion & Chunking
            </h1>
            <span className="px-2 py-0.5 text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 rounded border border-emerald-200 dark:border-emerald-800">
              Phase 3 Active
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Upload PDF, DOCX, and TXT files for layout-aware text extraction, semantic chunking, and pgvector embeddings
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchDocuments}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 transition-colors shadow-sm disabled:opacity-50"
            title="Refresh document repository"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Model Telemetry Banner */}
      {modelInfo && (
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 px-4 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs">
          <div className="flex items-center gap-2">
            <div className="p-1 rounded-md bg-brand-50 text-brand-600 dark:bg-brand-950 dark:text-brand-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <span className="font-semibold text-slate-900 dark:text-slate-100">
                Active Embedding Model:{' '}
              </span>
              <span className="font-mono text-brand-600 dark:text-brand-400 font-bold">
                {modelInfo.model_name}
              </span>
              <span className="text-slate-400 ml-2">
                ({modelInfo.description || 'English-optimized'})
              </span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-slate-600 dark:text-slate-400">
            <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
              {modelInfo.dimension} Dimensions
            </span>
            <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 capitalize">
              {modelInfo.metric}
            </span>
            <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
              Device: {modelInfo.device.toUpperCase()}
            </span>
            <span className="px-2 py-0.5 rounded bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
              Max Tokens: {modelInfo.max_sequence_length}
            </span>
          </div>
        </div>
      )}

      {/* Upload Dropzone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-6 sm:p-8 text-center cursor-pointer transition-all ${
          dragActive
            ? 'border-brand-500 bg-brand-50/50 dark:bg-brand-950/20 shadow-md'
            : 'border-slate-300 dark:border-slate-800 hover:border-brand-400 dark:hover:border-brand-700 bg-white dark:bg-slate-900'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          className="hidden"
          disabled={uploading}
        />

        <div className="max-w-md mx-auto space-y-3">
          <div className="w-12 h-12 mx-auto rounded-xl bg-brand-50 text-brand-600 dark:bg-brand-950 dark:text-brand-400 flex items-center justify-center shadow-sm">
            {uploading ? (
              <RefreshCw className="w-6 h-6 animate-spin text-brand-600 dark:text-brand-400" />
            ) : (
              <Upload className="w-6 h-6" />
            )}
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              {uploading ? 'Processing Document Pipeline...' : 'Click to upload or drag and drop'}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Supports <strong className="text-slate-700 dark:text-slate-300">PDF</strong>,{' '}
              <strong className="text-slate-700 dark:text-slate-300">DOCX</strong>, and{' '}
              <strong className="text-slate-700 dark:text-slate-300">TXT</strong> up to 25 MB
            </p>
          </div>

          {uploading && (
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 text-xs font-medium animate-pulse">
              <Sparkles className="w-3.5 h-3.5" />
              <span>Extracting text, cleaning, and generating chunks...</span>
            </div>
          )}
        </div>
      </div>

      {/* Upload Feedback Banners */}
      {uploadError && (
        <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-300 text-xs flex items-start gap-3">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Upload Failed:</span> {uploadError}
          </div>
          <button
            onClick={() => setUploadError(null)}
            className="p-1 hover:bg-rose-100 dark:hover:bg-rose-900 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {uploadSuccess && (
        <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300 text-xs flex items-start gap-3">
          <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold">Ingestion Complete:</span> {uploadSuccess}
          </div>
          <button
            onClick={() => setUploadSuccess(null)}
            className="p-1 hover:bg-emerald-100 dark:hover:bg-emerald-900 rounded"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Documents Repository Table */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
        <div className="p-4 sm:p-5 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="text-sm sm:text-base font-bold text-slate-900 dark:text-slate-100">
              Ingested Documents
            </h2>
            <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-mono">
              {totalDocs}
            </span>
          </div>
        </div>

        {loading ? (
          <div className="p-12 text-center text-slate-400">
            <RefreshCw className="w-6 h-6 mx-auto animate-spin mb-2" />
            <p className="text-xs">Loading document repository...</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="p-12 text-center max-w-sm mx-auto">
            <div className="w-12 h-12 mx-auto rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-400 flex items-center justify-center mb-3">
              <FileText className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-1">
              No Documents Ingested
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Upload a PDF, DOCX, or plain text file above to extract and generate semantic chunks.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400 font-semibold border-b border-slate-100 dark:border-slate-800 uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="py-3.5 px-4 sm:px-6">Document</th>
                  <th className="py-3.5 px-3">Format</th>
                  <th className="py-3.5 px-3">Size</th>
                  <th className="py-3.5 px-3">Pages</th>
                  <th className="py-3.5 px-3">Chunks</th>
                  <th className="py-3.5 px-3">Embeddings</th>
                  <th className="py-3.5 px-3">Status</th>
                  <th className="py-3.5 px-3">Uploaded</th>
                  <th className="py-3.5 px-4 sm:px-6 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                {documents.map((doc) => (
                  <tr
                    key={doc.id}
                    className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="py-3.5 px-4 sm:px-6 font-medium text-slate-900 dark:text-slate-100 max-w-[220px] truncate">
                      <span title={doc.original_filename}>{doc.original_filename}</span>
                    </td>
                    <td className="py-3.5 px-3">{getFormatBadge(doc.file_type)}</td>
                    <td className="py-3.5 px-3 font-mono text-slate-500 dark:text-slate-400">
                      {formatFileSize(doc.file_size)}
                    </td>
                    <td className="py-3.5 px-3 font-mono text-slate-500 dark:text-slate-400">
                      {doc.total_pages !== null ? `${doc.total_pages}` : '—'}
                    </td>
                    <td className="py-3.5 px-3">
                      <span className="inline-flex items-center gap-1 font-mono font-semibold text-brand-600 dark:text-brand-400">
                        <Layers className="w-3 h-3" />
                        {doc.total_chunks}
                      </span>
                    </td>
                    <td className="py-3.5 px-3">
                      {getEmbeddingStatusBadge(
                        doc.embedding_status,
                        doc.total_embedded_chunks,
                        doc.total_chunks,
                        doc.embedding_error
                      )}
                    </td>
                    <td className="py-3.5 px-3">{getStatusBadge(doc.status, doc.error_message)}</td>
                    <td className="py-3.5 px-3 text-slate-500 dark:text-slate-400 text-[11px] font-mono">
                      {new Date(doc.created_at).toLocaleDateString()}
                    </td>
                    <td className="py-3.5 px-4 sm:px-6 text-right">
                      <div className="inline-flex items-center gap-1">
                        <button
                          onClick={() => handleReembed(doc)}
                          disabled={reembeddingId === doc.id || doc.status === 'processing'}
                          className="p-1.5 text-slate-500 hover:text-brand-600 dark:hover:text-brand-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-40"
                          title="Re-embed Chunks"
                        >
                          <Sparkles
                            className={`w-4 h-4 ${
                              reembeddingId === doc.id ? 'animate-spin text-brand-600 dark:text-brand-400' : ''
                            }`}
                          />
                        </button>
                        <button
                          onClick={() => openInspector(doc)}
                          className="p-1.5 text-slate-500 hover:text-brand-600 dark:hover:text-brand-400 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                          title="Inspect Chunks"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setDeleteTarget(doc)}
                          className="p-1.5 text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/50 rounded-lg transition-colors"
                          title="Delete Document"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Chunk Inspector Modal */}
      {inspectDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="p-4 sm:p-5 border-b border-slate-100 dark:border-slate-800 flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-semibold px-2 py-0.5 rounded bg-brand-50 text-brand-700 dark:bg-brand-950 dark:text-brand-300 border border-brand-200 dark:border-brand-800">
                    Chunk Inspector
                  </span>
                  <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 truncate max-w-md">
                    {inspectDoc.original_filename}
                  </h3>
                  {getFormatBadge(inspectDoc.file_type)}
                </div>
                <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400 font-mono">
                  <span>Size: {formatFileSize(inspectDoc.file_size)}</span>
                  <span>•</span>
                  <span>Pages: {inspectDoc.total_pages ?? 'N/A'}</span>
                  <span>•</span>
                  <span>Characters: {inspectDoc.total_characters.toLocaleString()}</span>
                  <span>•</span>
                  <span>Total Chunks: {totalChunks}</span>
                </div>
              </div>
              <button
                onClick={closeInspector}
                className="p-1.5 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body: Chunk List */}
            <div className="flex-1 p-4 sm:p-6 overflow-y-auto space-y-4">
              {chunksLoading ? (
                <div className="py-12 text-center text-slate-400">
                  <RefreshCw className="w-6 h-6 mx-auto animate-spin mb-2" />
                  <p className="text-xs">Loading chunks...</p>
                </div>
              ) : chunks.length === 0 ? (
                <div className="py-12 text-center text-slate-400 text-xs">
                  No chunks generated for this document.
                </div>
              ) : (
                chunks.map((chunk) => (
                  <div
                    key={chunk.id}
                    className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/50 space-y-2.5"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold text-brand-600 dark:text-brand-400">
                          Chunk #{chunk.chunk_index + 1}
                        </span>
                        {chunk.page_start !== null && (
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                            {chunk.page_start === chunk.page_end
                              ? `Page ${chunk.page_start}`
                              : `Pages ${chunk.page_start}–${chunk.page_end}`}
                          </span>
                        )}
                        {chunk.metadata_json?.paragraph_start !== undefined && (
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                            Para {chunk.metadata_json.paragraph_start}–{chunk.metadata_json.paragraph_end}
                          </span>
                        )}
                        {chunk.has_embedding ? (
                          <span
                            title={`Embedded with ${chunk.embedding_model || '384d model'}`}
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800"
                          >
                            <Binary className="w-3 h-3" />
                            384d Vector
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-slate-200 dark:bg-slate-800 text-slate-500">
                            No Vector
                          </span>
                        )}
                        {chunk.metadata_json?.token_count !== undefined && (
                          <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                            ~{chunk.metadata_json.token_count} tokens
                          </span>
                        )}
                        {(chunk.metadata_json?.oversized_chunk || chunk.metadata_json?.oversized_warning) && (
                          <span
                            title={
                              chunk.metadata_json.oversized_info ||
                              `Chunk exceeds 512 tokens. Safely embedded via ${chunk.metadata_json.sub_chunk_count || 'multiple'} sub-windows.`
                            }
                            className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-400 border border-blue-200 dark:border-blue-800"
                          >
                            {chunk.metadata_json.sub_chunk_count
                              ? `Sub-Pooled (${chunk.metadata_json.sub_chunk_count}x)`
                              : 'Oversized (>512)'}
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-slate-400 text-[11px]">
                        {chunk.character_count} chars
                      </span>
                    </div>

                    <div className="p-3 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800/80 text-xs text-slate-700 dark:text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
                      {chunk.text}
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Modal Footer: Pagination */}
            {totalChunks > CHUNKS_PER_PAGE && (
              <div className="p-3 sm:p-4 border-t border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/50 flex items-center justify-between text-xs">
                <span className="text-slate-500 font-mono">
                  Showing {chunksPage * CHUNKS_PER_PAGE + 1} to{' '}
                  {Math.min((chunksPage + 1) * CHUNKS_PER_PAGE, totalChunks)} of {totalChunks} chunks
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => loadChunks(inspectDoc.id, chunksPage - 1)}
                    disabled={chunksPage === 0 || chunksLoading}
                    className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <span className="font-mono px-2">Page {chunksPage + 1}</span>
                  <button
                    onClick={() => loadChunks(inspectDoc.id, chunksPage + 1)}
                    disabled={(chunksPage + 1) * CHUNKS_PER_PAGE >= totalChunks || chunksLoading}
                    className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 disabled:opacity-40 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-rose-50 text-rose-600 dark:bg-rose-950 dark:text-rose-400">
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
                  Delete Document
                </h3>
                <p className="text-xs text-slate-500">This action cannot be undone</p>
              </div>
            </div>

            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Are you sure you want to delete <strong className="text-slate-900 dark:text-slate-100">{deleteTarget.original_filename}</strong>?
              This will permanently delete the document record, all {deleteTarget.total_chunks} associated chunks, and remove the file from storage.
            </p>

            <div className="pt-2 flex items-center justify-end gap-2 text-xs">
              <button
                onClick={() => setDeleteTarget(null)}
                disabled={deleting}
                className="px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteConfirm}
                disabled={deleting}
                className="px-3 py-1.5 rounded-lg bg-rose-600 text-white font-medium hover:bg-rose-700 disabled:opacity-50 flex items-center gap-1.5 shadow-sm"
              >
                {deleting && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                <span>{deleting ? 'Deleting...' : 'Delete Permanently'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
