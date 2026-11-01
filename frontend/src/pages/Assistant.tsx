import React, { useState, useEffect, useCallback } from 'react';
import {
  Search,
  Sliders,
  RefreshCw,
  FileText,
  Clock,
  Sparkles,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Database,
  Cpu,
  Zap,
  Filter,
} from 'lucide-react';
import { retrievalService } from '../services/retrievalService';
import type {
  RetrievedChunkResult,
  RetrievalMetadata,
  RetrievalInfoResponse,
} from '../services/retrievalService';
import { documentService } from '../services/documentService';
import type { DocumentItem } from '../types';

export const AssistantPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RetrievedChunkResult[]>([]);
  const [metadata, setMetadata] = useState<RetrievalMetadata | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  // Search parameters
  const [topK, setTopK] = useState<number>(5);
  const [enableReranking, setEnableReranking] = useState<boolean>(true);
  const [selectedDocId, setSelectedDocId] = useState<string>('all');
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  // Pipeline telemetry & index status
  const [pipelineInfo, setPipelineInfo] = useState<RetrievalInfoResponse | null>(null);
  const [rebuildingIndex, setRebuildingIndex] = useState(false);
  const [indexSuccessMsg, setIndexSuccessMsg] = useState<string | null>(null);

  // Expanded score details per chunk
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  // Fetch pipeline status and document list on mount
  const fetchStatus = useCallback(async () => {
    try {
      const [info, docsRes] = await Promise.all([
        retrievalService.getInfo(),
        documentService.getDocuments(0, 100),
      ]);
      setPipelineInfo(info);
      setDocuments(docsRes.items || []);
    } catch (err) {
      console.warn('Failed to load retrieval info or documents:', err);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await retrievalService.search({
        query: trimmed,
        top_k: topK,
        document_ids: selectedDocId !== 'all' ? [selectedDocId] : null,
        enable_reranking: enableReranking,
      });
      setResults(response.results);
      setMetadata(response.retrieval_metadata);
    } catch (err: unknown) {
      const msg =
        err && typeof err === 'object' && 'response' in err
          ? (err as { response?: { data?: { detail?: string; message?: string } } }).response?.data?.message ||
            (err as { response?: { data?: { detail?: string; message?: string } } }).response?.data?.detail ||
            'Retrieval failed. Please try again.'
          : 'Failed to connect to retrieval engine.';
      setError(msg);
      setResults([]);
      setMetadata(null);
    } finally {
      setLoading(false);
    }
  };

  const handleRebuildIndex = async () => {
    setRebuildingIndex(true);
    setIndexSuccessMsg(null);
    try {
      const res = await retrievalService.rebuildIndex();
      setIndexSuccessMsg(`BM25 Index rebuilt: ${res.indexed_chunks} chunks indexed.`);
      await fetchStatus();
      setTimeout(() => setIndexSuccessMsg(null), 4000);
    } catch (err) {
      setError('Failed to rebuild BM25 index.');
    } finally {
      setRebuildingIndex(false);
    }
  };

  const toggleExpand = (chunkId: string) => {
    setExpandedChunkId((prev) => (prev === chunkId ? null : chunkId));
  };

  const sampleQueries = [
    'pgvector HNSW index parameters and cosine similarity',
    'maximum upload limit for PDF and DOCX files',
    'Reciprocal Rank Fusion k constant formula',
    'Sentence-Transformers 384 dimensions',
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto py-2">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-500" />
              Hybrid Retrieval Engine
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300">
              Phase 4 Active
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Multi-stage search blending dense semantic vectors, Lucene BM25 lexical search, RRF fusion, and Cross-Encoder reranking.
          </p>
        </div>

        {/* Index Status & Rebuild Action */}
        <div className="flex items-center gap-3">
          {pipelineInfo && (
            <div className="hidden sm:flex flex-col text-right text-xs text-slate-500 dark:text-slate-400">
              <span>
                BM25: <strong>{pipelineInfo.bm25_index_status.total_chunks}</strong> chunks
              </span>
              <span>
                Device: <strong>{pipelineInfo.active_inference_device}</strong>
              </span>
            </div>
          )}
          <button
            onClick={handleRebuildIndex}
            disabled={rebuildingIndex}
            title="Rebuild in-memory BM25 index from persistent database chunks"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${rebuildingIndex ? 'animate-spin' : ''}`} />
            <span>{rebuildingIndex ? 'Indexing...' : 'Rebuild Index'}</span>
          </button>
        </div>
      </div>

      {indexSuccessMsg && (
        <div className="p-3 text-xs rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
          {indexSuccessMsg}
        </div>
      )}

      {/* Search Input Box */}
      <form onSubmit={handleSearch} className="space-y-4">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter natural-language query or technical terms (e.g., 'pgvector cosine distance', 'chunk overlap')..."
            className="w-full pl-12 pr-28 py-3.5 text-sm rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-sm"
          />
          <Search className="w-5 h-5 text-slate-400 absolute left-4 top-4" />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="absolute right-2 top-2 px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold shadow-sm transition disabled:opacity-50"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Parameter Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-900/50 text-xs">
          <div className="flex flex-wrap items-center gap-4">
            {/* Cross-Encoder Toggle */}
            <label className="flex items-center gap-2 cursor-pointer text-slate-700 dark:text-slate-300 select-none">
              <input
                type="checkbox"
                checked={enableReranking}
                onChange={(e) => setEnableReranking(e.target.checked)}
                className="w-4 h-4 rounded text-indigo-600 focus:ring-indigo-500 border-slate-300 dark:border-slate-700"
              />
              <span className="font-medium">Cross-Encoder Reranking</span>
              <span className="text-slate-400 text-[11px]">(ms-marco-MiniLM-L-6-v2)</span>
            </label>

            {/* Top-K Selector */}
            <div className="flex items-center gap-1.5 text-slate-700 dark:text-slate-300">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span>Top K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md px-2 py-0.5 text-xs text-slate-900 dark:text-slate-100"
              >
                <option value={3}>3 Chunks</option>
                <option value={5}>5 Chunks</option>
                <option value={10}>10 Chunks</option>
                <option value={20}>20 Chunks</option>
              </select>
            </div>

            {/* Document Filter */}
            {documents.length > 0 && (
              <div className="flex items-center gap-1.5 text-slate-700 dark:text-slate-300">
                <Filter className="w-3.5 h-3.5 text-slate-400" />
                <span>Scope:</span>
                <select
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  className="bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md px-2 py-0.5 text-xs text-slate-900 dark:text-slate-100 max-w-[180px] truncate"
                >
                  <option value="all">All Documents ({documents.length})</option>
                  {documents.map((doc) => (
                    <option key={doc.id} value={doc.id}>
                      {doc.filename}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Model info snippet */}
          {pipelineInfo && (
            <div className="text-[11px] text-slate-400 flex items-center gap-3">
              <span className="flex items-center gap-1">
                <Database className="w-3 h-3 text-indigo-400" />
                Dense: {pipelineInfo.embedding_dimensions}d
              </span>
              <span className="flex items-center gap-1">
                <Cpu className="w-3 h-3 text-purple-400" />
                RRF k={pipelineInfo.rrf_configuration.rrf_constant}
              </span>
            </div>
          )}
        </div>
      </form>

      {/* Latency & Candidate Funnel Telemetry */}
      {metadata && !loading && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <div className="text-[11px] text-slate-400 font-medium">Total Latency</div>
            <div className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1 mt-0.5">
              <Clock className="w-4 h-4 text-emerald-500" />
              {metadata.latency_ms} ms
            </div>
          </div>
          <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <div className="text-[11px] text-slate-400 font-medium">Dense Stream</div>
            <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mt-0.5">
              {metadata.dense_candidates} cands
            </div>
            {metadata.dense_latency_ms !== null && metadata.dense_latency_ms !== undefined && (
              <div className="text-[10px] text-slate-400">{metadata.dense_latency_ms} ms</div>
            )}
          </div>
          <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <div className="text-[11px] text-slate-400 font-medium">BM25 Stream</div>
            <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mt-0.5">
              {metadata.bm25_candidates} cands
            </div>
            {metadata.bm25_latency_ms !== null && metadata.bm25_latency_ms !== undefined && (
              <div className="text-[10px] text-slate-400">{metadata.bm25_latency_ms} ms</div>
            )}
          </div>
          <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <div className="text-[11px] text-slate-400 font-medium">RRF Fusion</div>
            <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mt-0.5">
              {metadata.fused_candidates} fused
            </div>
            {metadata.rrf_latency_ms !== null && metadata.rrf_latency_ms !== undefined && (
              <div className="text-[10px] text-slate-400">{metadata.rrf_latency_ms} ms</div>
            )}
          </div>
          <div className="p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <div className="text-[11px] text-slate-400 font-medium">Reranker Stage</div>
            <div className="text-sm font-semibold text-slate-700 dark:text-slate-300 mt-0.5">
              {metadata.reranking_enabled ? `${metadata.returned_results} selected` : 'Bypassed'}
            </div>
            {metadata.reranker_latency_ms !== null && metadata.reranker_latency_ms !== undefined && (
              <div className="text-[10px] text-slate-400">{metadata.reranker_latency_ms} ms</div>
            )}
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 text-rose-800 dark:text-rose-200 text-xs">
          <AlertCircle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold">Retrieval Request Error</div>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="space-y-3">
          {[1, 2, 3].map((n) => (
            <div
              key={n}
              className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 space-y-3 animate-pulse"
            >
              <div className="flex items-center justify-between">
                <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/4"></div>
                <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-16"></div>
              </div>
              <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-full"></div>
              <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-5/6"></div>
              <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      )}

      {/* Results List */}
      {!loading && hasSearched && results.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>
              Retrieved <strong>{results.length}</strong> most relevant chunks
            </span>
            <span>Sorted by relevance rank</span>
          </div>

          <div className="space-y-3">
            {results.map((result) => {
              const isExpanded = expandedChunkId === result.chunk_id;
              return (
                <div
                  key={result.chunk_id}
                  className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm transition hover:border-slate-300 dark:hover:border-slate-700 space-y-3"
                >
                  {/* Top Metadata Line */}
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <span className="flex items-center justify-center w-6 h-6 rounded-full bg-indigo-50 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 font-bold text-xs">
                        #{result.final_rank}
                      </span>
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-800 dark:text-slate-200">
                        <FileText className="w-3.5 h-3.5 text-slate-400" />
                        <span className="truncate max-w-[240px] sm:max-w-md">{result.filename}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {result.page_start !== null && result.page_start !== undefined && (
                        <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                          {result.page_start === result.page_end
                            ? `Page ${result.page_start}`
                            : `Pages ${result.page_start} - ${result.page_end}`}
                        </span>
                      )}
                      <span className="text-[11px] text-slate-400">
                        {result.character_count} chars
                      </span>
                    </div>
                  </div>

                  {/* Chunk Text Passage */}
                  <div className="p-3.5 rounded-xl bg-slate-50/80 dark:bg-slate-800/40 text-xs sm:text-sm text-slate-700 dark:text-slate-200 leading-relaxed font-sans border border-slate-100 dark:border-slate-800/60 select-text">
                    {result.text}
                  </div>

                  {/* Explainable Scoring Details Accordion */}
                  <div className="pt-1">
                    <button
                      onClick={() => toggleExpand(result.chunk_id)}
                      className="inline-flex items-center gap-1 text-[11px] font-medium text-indigo-600 dark:text-indigo-400 hover:underline"
                    >
                      <span>{isExpanded ? 'Hide Technical Scores' : 'View Technical Scores & Ranks'}</span>
                      {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-2.5 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                        <div>
                          <span className="text-slate-400 text-[10px] block">Dense Semantic</span>
                          <span className="font-semibold text-slate-800 dark:text-slate-200">
                            {result.dense_score != null ? result.dense_score.toFixed(4) : 'N/A'}
                          </span>
                          <span className="text-[10px] text-slate-400 ml-1">
                            {result.dense_rank ? `(#${result.dense_rank})` : ''}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-[10px] block">BM25 Sparse</span>
                          <span className="font-semibold text-slate-800 dark:text-slate-200">
                            {result.bm25_score != null ? result.bm25_score.toFixed(2) : 'N/A'}
                          </span>
                          <span className="text-[10px] text-slate-400 ml-1">
                            {result.bm25_rank ? `(#${result.bm25_rank})` : ''}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-[10px] block">RRF Fusion Score</span>
                          <span className="font-semibold text-slate-800 dark:text-slate-200">
                            {result.rrf_score.toFixed(5)}
                          </span>
                          <span className="text-[10px] text-slate-400 ml-1">
                            (Rank #{result.rrf_rank})
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-[10px] block">Cross-Encoder Score</span>
                          <span className="font-semibold text-slate-800 dark:text-slate-200">
                            {result.reranker_score !== null && result.reranker_score !== undefined
                              ? result.reranker_score.toFixed(3)
                              : 'Bypassed'}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Empty Search Results */}
      {!loading && hasSearched && results.length === 0 && !error && (
        <div className="p-8 text-center rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 space-y-2">
          <AlertCircle className="w-8 h-8 text-slate-400 mx-auto" />
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
            No matching document chunks found
          </h2>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Try adjusting your search terms, removing document filters, or click &ldquo;Rebuild Index&rdquo; to sync the lexical search engine.
          </p>
        </div>
      )}

      {/* Initial Guidance Card with Suggested Queries */}
      {!hasSearched && (
        <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 space-y-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <Zap className="w-4 h-4 text-amber-500" />
            <span>Suggested Benchmark Queries</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {sampleQueries.map((sq, i) => (
              <button
                key={i}
                type="button"
                onClick={() => {
                  setQuery(sq);
                }}
                className="text-left p-3 rounded-xl border border-slate-200 dark:border-slate-800 hover:border-indigo-400 dark:hover:border-indigo-600 bg-slate-50/50 dark:bg-slate-800/30 text-xs text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition"
              >
                &ldquo;{sq}&rdquo;
              </button>
            ))}
          </div>

          <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/40 text-[11px] text-slate-500 leading-relaxed border border-slate-100 dark:border-slate-800">
            <strong>Hybrid Architecture Note:</strong> Natural-language queries query both BGE-small dense embeddings (via pgvector cosine similarity) and Lucene BM25 sparse index concurrently. Reciprocal Rank Fusion ($k=60$) balances ranks before the Cross-Encoder elevates top-precision candidates.
          </div>
        </div>
      )}
    </div>
  );
};
