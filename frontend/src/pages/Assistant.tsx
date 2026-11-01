import React, { useState, useEffect, useCallback } from 'react';
import {
  Search,
  RefreshCw,
  Clock,
  Sparkles,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Zap,
  Filter,
  Bot,
  CheckCircle2,
  HelpCircle,
  BookOpen,
} from 'lucide-react';
import { retrievalService } from '../services/retrievalService';
import type {
  RetrievedChunkResult,
  RetrievalMetadata,
  RetrievalInfoResponse,
} from '../services/retrievalService';
import { assistantService } from '../services/assistantService';
import type {
  RAGQueryResponse,
  RAGHealthResponse,
} from '../services/assistantService';
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
  const [enableRAG, setEnableRAG] = useState<boolean>(true);
  const [selectedDocId, setSelectedDocId] = useState<string>('all');
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  // Phase 5 RAG state
  const [ragResponse, setRagResponse] = useState<RAGQueryResponse | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [llmHealth, setLlmHealth] = useState<RAGHealthResponse | null>(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  // Pipeline telemetry & index status
  const [pipelineInfo, setPipelineInfo] = useState<RetrievalInfoResponse | null>(null);
  const [rebuildingIndex, setRebuildingIndex] = useState(false);
  const [indexSuccessMsg, setIndexSuccessMsg] = useState<string | null>(null);

  // Expanded score details per chunk
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  // Fetch pipeline status and document list on mount
  const fetchStatus = useCallback(async () => {
    try {
      const [info, docsRes, healthRes] = await Promise.all([
        retrievalService.getInfo(),
        documentService.getDocuments(0, 100),
        assistantService.getHealth().catch(() => null),
      ]);
      setPipelineInfo(info);
      setDocuments(docsRes.items || []);
      setLlmHealth(healthRes);
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
    setRagResponse(null);
    setSelectedCitationId(null);

    try {
      if (enableRAG) {
        // Execute RAG Question Answering pipeline (Phase 5)
        try {
          const ragRes = await assistantService.ask({
            question: trimmed,
            top_k: topK,
            document_ids: selectedDocId !== 'all' ? [selectedDocId] : null,
            enable_reranking: enableReranking,
          });
          setRagResponse(ragRes);
          setMetadata(ragRes.retrieval_metadata);

          // Convert evidence sources to RetrievedChunkResult format for unified transparency view
          const mappedChunks: RetrievedChunkResult[] = ragRes.evidence_sources.map((src, i) => ({
            chunk_id: src.chunk_id,
            document_id: src.document_id,
            filename: src.filename,
            text: src.text_snippet,
            character_count: src.text_snippet.length,
            page_start: src.page_start,
            page_end: src.page_end,
            dense_score: src.dense_score,
            bm25_score: src.bm25_score,
            rrf_score: src.rrf_score,
            rrf_rank: i + 1,
            reranker_score: src.reranker_score,
            final_rank: src.final_rank,
          }));
          setResults(mappedChunks);
        } catch (ragErr: any) {
          // If LLM provider is unavailable, fallback to hybrid retrieval and report informative notice
          console.warn('RAG generation unavailable, falling back to direct retrieval:', ragErr);
          const response = await retrievalService.search({
            query: trimmed,
            top_k: topK,
            document_ids: selectedDocId !== 'all' ? [selectedDocId] : null,
            enable_reranking: enableReranking,
          });
          setResults(response.results);
          setMetadata(response.retrieval_metadata);
          setError(
            ragErr.response?.data?.message ||
              ragErr.response?.data?.detail?.message ||
              'LLM inference provider is currently unavailable. Displaying retrieved candidate passages.'
          );
        }
      } else {
        // Phase 4 direct retrieval search only
        const response = await retrievalService.search({
          query: trimmed,
          top_k: topK,
          document_ids: selectedDocId !== 'all' ? [selectedDocId] : null,
          enable_reranking: enableReranking,
        });
        setResults(response.results);
        setMetadata(response.retrieval_metadata);
      }
    } catch (err: unknown) {
      const msg =
        err && typeof err === 'object' && 'response' in err
          ? (err as { response?: { data?: { detail?: string; message?: string } } }).response?.data?.message ||
            (err as { response?: { data?: { detail?: string; message?: string } } }).response?.data?.detail ||
            'Retrieval failed. Please try again.'
          : 'Failed to connect to retrieval engine.';
      setError(String(msg));
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
    } catch {
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
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-500" />
              Hybrid Retrieval Engine
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800 dark:bg-indigo-950 dark:text-indigo-300">
              Phase 4 Active
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 flex items-center gap-1">
              <Bot className="w-3.5 h-3.5" />
              Phase 5 Grounded RAG
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Multi-stage search blending dense semantic vectors, Lucene BM25 lexical search, RRF fusion, and Cross-Encoder reranking.
          </p>
        </div>

        {/* Index & LLM Status */}
        <div className="flex items-center gap-3">
          {llmHealth && (
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 text-xs">
              <span className={`w-2 h-2 rounded-full ${llmHealth.provider_available ? 'bg-emerald-500' : 'bg-amber-500'}`} />
              <span className="text-slate-600 dark:text-slate-400">
                LLM: <strong>{llmHealth.provider}</strong>
              </span>
            </div>
          )}
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
            {rebuildingIndex ? 'Rebuilding...' : 'Rebuild Index'}
          </button>
        </div>
      </div>

      {indexSuccessMsg && (
        <div className="p-3 text-xs bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300 rounded-xl flex items-center gap-2">
          <Zap className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
          {indexSuccessMsg}
        </div>
      )}

      {/* Query Formulation Card */}
      <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter natural-language query or question to retrieve and answer..."
              className="w-full pl-10 pr-24 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500"
            />
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-1.5 top-1.5 bottom-1.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-xs flex items-center gap-1.5 transition disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="w-3.5 h-3.5" />
                  Search
                </>
              )}
            </button>
          </div>

          {/* Search Controls */}
          <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600 dark:text-slate-400 pt-1">
            <div className="flex items-center gap-1.5">
              <Filter className="w-3.5 h-3.5 text-slate-400" />
              <span>Document Scope:</span>
              <select
                value={selectedDocId}
                onChange={(e) => setSelectedDocId(e.target.value)}
                className="px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none"
              >
                <option value="all">All Documents ({documents.length})</option>
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.original_filename} ({d.file_type.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span>Top K:</span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none"
              >
                <option value={3}>Top 3</option>
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
              </select>
            </div>

            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={enableRAG}
                onChange={(e) => setEnableRAG(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="font-semibold text-emerald-600 dark:text-emerald-400">Generate Answer (RAG)</span>
            </label>

            <label className="flex items-center gap-1.5 cursor-pointer ml-auto">
              <input
                type="checkbox"
                checked={enableReranking}
                onChange={(e) => setEnableReranking(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Cross-Encoder Reranking</span>
            </label>
          </div>
        </form>

        {/* Suggested Queries */}
        <div className="pt-2 border-t border-slate-100 dark:border-slate-800/80">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 block mb-2">
            Suggested Benchmark Queries
          </span>
          <div className="flex flex-wrap gap-2">
            {sampleQueries.map((sq, i) => (
              <button
                key={i}
                onClick={() => {
                  setQuery(sq);
                }}
                className="text-xs px-2.5 py-1 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40 text-slate-600 dark:text-slate-300 hover:border-indigo-400 hover:text-indigo-600 dark:hover:text-indigo-400 transition"
              >
                {sq}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 rounded-xl border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 text-red-800 dark:text-red-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* ========================================================================= */}
      {/* PHASE 5: GROUNDED RAG ANSWER PRESENTATION                                  */}
      {/* ========================================================================= */}
      {ragResponse && (
        <div className="space-y-6">
          <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <CheckCircle2 className={`w-5 h-5 ${ragResponse.is_answered ? 'text-emerald-500' : 'text-amber-500'}`} />
                <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
                  {ragResponse.is_answered ? 'Grounded Answer' : 'Insufficient Evidence'}
                </h2>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Clock className="w-3.5 h-3.5 text-indigo-500" />
                <span>{ragResponse.diagnostics.total_latency_ms} ms</span>
                <span className="text-slate-300 dark:text-slate-700">•</span>
                <span>{ragResponse.generation_metadata.model}</span>
                <button
                  type="button"
                  onClick={() => setShowDiagnostics((prev) => !prev)}
                  className="ml-2 text-indigo-600 dark:text-indigo-400 hover:underline flex items-center gap-0.5"
                >
                  {showDiagnostics ? 'Hide Breakdown' : 'Breakdown'}
                  {showDiagnostics ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
              </div>
            </div>

            {showDiagnostics && (
              <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl border border-slate-200 dark:border-slate-700 text-xs grid grid-cols-2 sm:grid-cols-5 gap-2 font-mono">
                <div>
                  <span className="text-slate-400 block text-[10px]">Validation</span>
                  <strong>{ragResponse.diagnostics.query_validation_ms} ms</strong>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">Retrieval</span>
                  <strong>{ragResponse.diagnostics.retrieval_latency_ms} ms</strong>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">Context</span>
                  <strong>{ragResponse.diagnostics.context_building_ms} ms</strong>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">Inference</span>
                  <strong>{ragResponse.diagnostics.inference_latency_ms} ms</strong>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px]">Citations</span>
                  <strong>{ragResponse.diagnostics.citation_validation_ms} ms</strong>
                </div>
              </div>
            )}

            <div className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed whitespace-pre-line">
              {ragResponse.answer}
            </div>

            {!ragResponse.is_answered && ragResponse.insufficient_evidence_reason && (
              <div className="p-3 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-300 text-xs rounded-xl flex items-start gap-2">
                <HelpCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" />
                <div>
                  <strong>Abstention Reason:</strong> {ragResponse.insufficient_evidence_reason}
                </div>
              </div>
            )}

            {ragResponse.citations.length > 0 && (
              <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center gap-2 flex-wrap text-xs">
                <span className="font-semibold text-slate-500 dark:text-slate-400">Cited Sources:</span>
                {ragResponse.citations.map((c) => (
                  <button
                    key={c.source_id}
                    onClick={() => setSelectedCitationId(c.source_id)}
                    className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-xs font-semibold transition ${
                      selectedCitationId === c.source_id
                        ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                        : 'bg-indigo-50 dark:bg-indigo-950/50 text-indigo-700 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800 hover:bg-indigo-100'
                    }`}
                  >
                    [{c.source_id}] {c.filename} {c.page_start ? `(p. ${c.page_start})` : ''}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Factual Claims List */}
          {ragResponse.claims.length > 0 && (
            <div className="p-6 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
                  <BookOpen className="w-4 h-4 text-indigo-500" />
                  Factual Claims Breakdown
                </h3>
                <span className="text-[11px] text-slate-400">
                  Note: Citation badges confirm source presence; not semantic entailment.
                </span>
              </div>
              <div className="space-y-2">
                {ragResponse.claims.map((claim, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-xl border text-xs flex flex-col sm:flex-row sm:items-start justify-between gap-3 ${
                      claim.is_supported === false
                        ? 'border-amber-200 dark:border-amber-900/60 bg-amber-50/50 dark:bg-amber-950/20'
                        : 'border-slate-100 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-800/40'
                    }`}
                  >
                    <span className="text-slate-700 dark:text-slate-300 leading-relaxed">
                      {claim.claim_text}
                    </span>
                    <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                      {claim.citation_ids.map((cid) => (
                        <span
                          key={cid}
                          className="px-2 py-0.5 rounded-md font-bold bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 text-[11px]"
                        >
                          [{cid}]
                        </span>
                      ))}
                      {claim.is_supported === false && (
                        <span className="px-2 py-0.5 rounded-md font-semibold bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 text-[11px]">
                          Unsupported (No Verified Source)
                        </span>
                      )}
                      {claim.unsupported_citation_ids &&
                        claim.unsupported_citation_ids.map((cid) => (
                          <span
                            key={cid}
                            className="px-2 py-0.5 rounded-md font-semibold bg-red-100 dark:bg-red-950 text-red-700 dark:text-red-300 text-[11px] line-through"
                            title="Unrecognized / hallucinated citation tag"
                          >
                            [{cid}]
                          </span>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* SEARCH / RETRIEVAL EVIDENCE INSPECTOR (TRANSPARENCY VIEW)                  */}
      {/* ========================================================================= */}
      {hasSearched && (
        <div className="space-y-4">
          {metadata && (
            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs text-slate-600 dark:text-slate-400 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <span>Dense Candidates: <strong>{metadata.dense_candidates}</strong></span>
                <span>BM25 Candidates: <strong>{metadata.bm25_candidates}</strong></span>
                <span>Fused (RRF): <strong>{metadata.fused_candidates}</strong></span>
                {metadata.reranking_enabled && (
                  <span>Reranked: <strong>{metadata.reranked_candidates}</strong></span>
                )}
              </div>
              <div className="flex items-center gap-2 font-mono">
                <Clock className="w-3.5 h-3.5 text-indigo-500" />
                <span>
                  Latency: <strong>{metadata.latency_ms} ms</strong>
                </span>
              </div>
            </div>
          )}

          {results.length === 0 ? (
            <div className="p-8 text-center text-xs text-slate-500 border border-dashed border-slate-200 dark:border-slate-800 rounded-2xl">
              No matching document chunks found.
            </div>
          ) : (
            <div className="space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500 px-1">
                {enableRAG ? 'Evidence Passages' : 'Ranked Retrieval Results'} ({results.length})
              </h3>
              {results.map((res) => (
                <div
                  key={res.chunk_id}
                  className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm space-y-3"
                >
                  <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-100 dark:border-slate-800">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-indigo-600 dark:text-indigo-400">
                        #{res.final_rank}
                      </span>
                      <span className="font-semibold text-slate-800 dark:text-slate-200">
                        {res.filename}
                      </span>
                      {res.page_start && (
                        <span className="text-slate-400">
                          Page {res.page_start}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => toggleExpand(res.chunk_id)}
                      className="text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 hover:underline flex items-center gap-1"
                    >
                      {expandedChunkId === res.chunk_id ? 'Hide Scores' : 'View Technical Scores'}
                      {expandedChunkId === res.chunk_id ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>
                  </div>

                  <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-mono">
                    {res.text}
                  </p>

                  {expandedChunkId === res.chunk_id && (
                    <div className="p-3 bg-slate-50 dark:bg-slate-800/60 rounded-xl grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                      <div>
                        <span className="text-slate-400 block">Dense Semantic</span>
                        <strong>{res.dense_score != null ? res.dense_score.toFixed(4) : 'N/A'}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 block">BM25 Sparse</span>
                        <strong>{res.bm25_score != null ? res.bm25_score.toFixed(2) : 'N/A'}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 block">RRF Fusion Score</span>
                        <strong>{res.rrf_score.toFixed(5)}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 block">Cross-Encoder Score</span>
                        <strong>{res.reranker_score != null ? res.reranker_score.toFixed(3) : 'N/A'}</strong>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
