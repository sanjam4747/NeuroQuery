"""Evaluation dataset and reference corpus for Phase 4 hybrid retrieval benchmarking."""

from dataclasses import dataclass, field
from typing import Dict, List, Set
import uuid

# Deterministic namespace UUIDs for evaluation documents and chunks
EVAL_NAMESPACE = uuid.UUID("11111111-2222-3333-4444-555555555555")


def make_eval_id(name: str) -> uuid.UUID:
    """Generates a deterministic UUID based on unique passage name."""
    return uuid.uuid5(EVAL_NAMESPACE, name)


@dataclass
class EvalDocument:
    """Synthetic reference document for retrieval benchmarking."""
    id: uuid.UUID
    filename: str
    title: str
    chunks: List["EvalChunk"] = field(default_factory=list)


@dataclass
class EvalChunk:
    """Individual chunk within an evaluation document with assigned ID."""
    id: uuid.UUID
    chunk_index: int
    text: str
    page_start: int
    page_end: int


@dataclass
class EvalQuery:
    """A natural language benchmark query with known relevant ground-truth chunk IDs."""
    query_id: str
    query_text: str
    category: str  # "exact_keyword", "semantic_paraphrase", "technical_term", "numeric_identifier", "negative"
    relevant_chunk_ids: Set[uuid.UUID]
    description: str


# ---------------------------------------------------------------------------
# 1. Benchmark Reference Documents & Chunks
# ---------------------------------------------------------------------------

DOC1_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc1_chunk0"),
        chunk_index=0,
        text=(
            "NeuroQuery document ingestion supports PDF, DOCX, and plain text formats. "
            "File validation enforces a 25 MB maximum upload limit and inspects magic byte headers "
            "such as %PDF- and OpenXML ZIP structures to ensure content authenticity before storage."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc1_chunk1"),
        chunk_index=1,
        text=(
            "PyMuPDF extracts text page by page while preserving exact page numbers and provenance. "
            "Blank and scanned pages lacking decodable text layers are cleanly skipped without crashing. "
            "Document storage isolates uploaded files under secure storage paths protected against path traversal."
        ),
        page_start=1,
        page_end=2,
    ),
]

DOC2_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc2_chunk0"),
        chunk_index=0,
        text=(
            "Conservative text preprocessing normalizes line endings to newline characters and repairs "
            "broken PDF hyphenated line wraps such as word- break into wordbreak. Typesetting ligatures "
            "like fi and fl are normalized to standard characters while numbers and punctuation remain intact."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc2_chunk1"),
        chunk_index=1,
        text=(
            "The semantic chunking engine uses character-based sliding windows with a default target size of "
            "1500 characters and 200 characters overlap. Boundaries prioritize paragraph breaks, sentence "
            "terminators, and whitespace boundaries to prevent mid-sentence or mid-word truncation."
        ),
        page_start=1,
        page_end=2,
    ),
]

DOC3_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc3_chunk0"),
        chunk_index=0,
        text=(
            "Dense semantic vector representations are computed locally via BAAI/bge-small-en-v1.5 using "
            "Sentence-Transformers. Each generated embedding has exactly 384 dimensions and is L2-normalized "
            "to unit length so that cosine similarity is equivalent to the dot product."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc3_chunk1"),
        chunk_index=1,
        text=(
            "For oversized chunks exceeding the model's 512 token limit, sub-window token segmentation splits "
            "the passage into overlapping 450-token windows with 50-token overlap. Vectors are combined using "
            "token-weighted linear combination pooling into a single unit-norm 384-dimensional vector without data loss."
        ),
        page_start=1,
        page_end=2,
    ),
]

DOC4_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc4_chunk0"),
        chunk_index=0,
        text=(
            "Persistent vector storage uses PostgreSQL 16 with the pgvector extension. The chunks table defines "
            "an embedding column of type vector(384) with an HNSW index configured using vector_cosine_ops "
            "for efficient nearest-neighbor approximate search at scale."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc4_chunk1"),
        chunk_index=1,
        text=(
            "Alembic migration 0003 manages pgvector schema creation and clean downgrades. Foreign key constraints "
            "with ON DELETE CASCADE guarantee that deleting a document record automatically removes all generated "
            "chunks and vector embeddings without orphan records."
        ),
        page_start=1,
        page_end=2,
    ),
]

DOC5_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc5_chunk0"),
        chunk_index=0,
        text=(
            "Sparse keyword retrieval uses BM25Okapi over tokenized document chunks. Tokenization preserves "
            "technical acronyms, dates, numbers, and identifiers while filtering high-frequency grammatical stopwords. "
            "The in-memory BM25 index automatically updates when new documents are ingested or deleted."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc5_chunk1"),
        chunk_index=1,
        text=(
            "BM25 is fundamentally distinct from dense retrieval: it excels at finding exact product names, error codes, "
            "and technical identifiers, whereas dense semantic search finds conceptually related passages even when "
            "vocabulary differs."
        ),
        page_start=1,
        page_end=2,
    ),
]

DOC6_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc6_chunk0"),
        chunk_index=0,
        text=(
            "Reciprocal Rank Fusion (RRF) combines dense vector and BM25 candidate lists into a unified ranking. "
            "The formula sums 1 / (k + rank) with a default constant k=60. Candidates are deduplicated by chunk ID "
            "and ordered deterministically without summing uncalibrated raw scores."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc6_chunk1"),
        chunk_index=1,
        text=(
            "Cross-encoder reranking applies cross-encoder/ms-marco-MiniLM-L-6-v2 to score query-passage pairs jointly. "
            "Reranking only evaluates the top fused candidates from RRF (default top 20), achieving high precision "
            "while maintaining low end-to-end retrieval latency on CPU."
        ),
        page_start=1,
        page_end=2,
    ),
]

DOC7_CHUNKS = [
    EvalChunk(
        id=make_eval_id("doc7_chunk0"),
        chunk_index=0,
        text=(
            "The backend REST API is built with FastAPI and runs on default port 8000. Versioned endpoints under /api/v1 "
            "include health checks, document ingestion, chunk retrieval, in-place re-embedding, and hybrid search. "
            "CORS settings allow communication with the Vite React frontend on port 5173."
        ),
        page_start=1,
        page_end=1,
    ),
    EvalChunk(
        id=make_eval_id("doc7_chunk1"),
        chunk_index=1,
        text=(
            "Evaluation metrics for information retrieval include Precision@K, Recall@K, Mean Reciprocal Rank (MRR), "
            "and Normalized Discounted Cumulative Gain (nDCG@K). Latency profiling breaks down dense, BM25, fusion, "
            "and cross-encoder execution times."
        ),
        page_start=1,
        page_end=2,
    ),
]

BENCHMARK_DOCUMENTS: List[EvalDocument] = [
    EvalDocument(id=make_eval_id("doc1"), filename="ingestion_and_formats.pdf", title="Document Ingestion & Formats", chunks=DOC1_CHUNKS),
    EvalDocument(id=make_eval_id("doc2"), filename="preprocessing_chunking.pdf", title="Preprocessing and Chunking", chunks=DOC2_CHUNKS),
    EvalDocument(id=make_eval_id("doc3"), filename="embeddings_bge.pdf", title="Semantic Embeddings & BGE", chunks=DOC3_CHUNKS),
    EvalDocument(id=make_eval_id("doc4"), filename="pgvector_persistence.pdf", title="PostgreSQL and pgvector", chunks=DOC4_CHUNKS),
    EvalDocument(id=make_eval_id("doc5"), filename="bm25_sparse_search.pdf", title="BM25 Sparse Retrieval", chunks=DOC5_CHUNKS),
    EvalDocument(id=make_eval_id("doc6"), filename="hybrid_rrf_reranking.pdf", title="Hybrid RRF and Reranking", chunks=DOC6_CHUNKS),
    EvalDocument(id=make_eval_id("doc7"), filename="api_evaluation_metrics.pdf", title="API and Evaluation Metrics", chunks=DOC7_CHUNKS),
]


# ---------------------------------------------------------------------------
# 2. Benchmark Evaluation Queries (32 queries across 5 distinct categories)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES: List[EvalQuery] = [
    # Category 1: Exact Keyword Queries (7 queries)
    EvalQuery(
        query_id="q01_bge_model",
        query_text="BAAI/bge-small-en-v1.5",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc3_chunk0")},
        description="Exact Hugging Face embedding model name",
    ),
    EvalQuery(
        query_id="q02_reranker_model",
        query_text="cross-encoder/ms-marco-MiniLM-L-6-v2",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc6_chunk1")},
        description="Exact cross-encoder reranker model identifier",
    ),
    EvalQuery(
        query_id="q03_pgvector_hnsw",
        query_text="pgvector HNSW index vector_cosine_ops",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc4_chunk0")},
        description="PostgreSQL vector index syntax and operator class",
    ),
    EvalQuery(
        query_id="q04_bm25okapi",
        query_text="BM25Okapi tokenized document chunks",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc5_chunk0")},
        description="Exact sparse lexical algorithm name",
    ),
    EvalQuery(
        query_id="q05_pymupdf_extraction",
        query_text="PyMuPDF extract page by page provenance",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc1_chunk1")},
        description="PDF parser library and page extraction",
    ),
    EvalQuery(
        query_id="q06_rrf_formula",
        query_text="Reciprocal Rank Fusion 1 / (k + rank)",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc6_chunk0")},
        description="Exact mathematical formula for RRF",
    ),
    EvalQuery(
        query_id="q07_alembic_0003",
        query_text="Alembic migration 0003 pgvector",
        category="exact_keyword",
        relevant_chunk_ids={make_eval_id("doc4_chunk1")},
        description="Exact migration identifier and schema",
    ),

    # Category 2: Semantic Paraphrase Queries (7 queries)
    EvalQuery(
        query_id="q08_safe_text_division",
        query_text="how does the system avoid cutting sentences in half when breaking documents down?",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc2_chunk1")},
        description="Semantic question describing chunking boundary hierarchy",
    ),
    EvalQuery(
        query_id="q09_handling_large_passages",
        query_text="what happens if a chunk contains way too many tokens for the embedding model?",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc3_chunk1")},
        description="Semantic paraphrase of oversized chunk token pooling",
    ),
    EvalQuery(
        query_id="q10_score_fusion_rationale",
        query_text="why don't we just add vector distance directly to the keyword frequency score?",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc6_chunk0")},
        description="Conceptual explanation of why RRF merges ranks rather than uncalibrated scores",
    ),
    EvalQuery(
        query_id="q11_cleaning_raw_text",
        query_text="mending line breaks, joined words, and special typographical symbols after extraction",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc2_chunk0")},
        description="Paraphrase of preprocessing normalization and ligature unfolding",
    ),
    EvalQuery(
        query_id="q12_cascade_deletion",
        query_text="removing an uploaded file and all its associated vector embeddings cleanly",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc4_chunk1")},
        description="Paraphrase of cascading delete without orphan records",
    ),
    EvalQuery(
        query_id="q13_when_sparse_helps",
        query_text="scenarios where keyword search performs better than semantic embeddings",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc5_chunk1")},
        description="Conceptual distinction between BM25 exact matching and dense retrieval",
    ),
    EvalQuery(
        query_id="q14_two_stage_ranking",
        query_text="running heavy transformer scoring only on a small subset of candidate passages",
        category="semantic_paraphrase",
        relevant_chunk_ids={make_eval_id("doc6_chunk1")},
        description="Two-stage retrieval and reranker efficiency on CPU",
    ),

    # Category 3: Technical Terminology Queries (7 queries)
    EvalQuery(
        query_id="q15_token_weighted_pooling",
        query_text="token-weighted linear combination pooling sub-window segmentation",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc3_chunk1")},
        description="Specific technical algorithm for oversized chunk embedding",
    ),
    EvalQuery(
        query_id="q16_hnsw_cosine",
        query_text="hierarchical navigable small world cosine graph traversal",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc4_chunk0")},
        description="Underlying graph index structure for pgvector",
    ),
    EvalQuery(
        query_id="q17_rrf_k_parameter",
        query_text="Reciprocal Rank Fusion constant k=60 smoothing parameter",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc6_chunk0")},
        description="RRF hyperparameter tuning and rank dampening",
    ),
    EvalQuery(
        query_id="q18_ligatures_hyphens",
        query_text="typesetting ligatures fi fl and hyphenated line wrap normalization",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc2_chunk0")},
        description="Typography and text normalization technicalities",
    ),
    EvalQuery(
        query_id="q19_magic_bytes",
        query_text="magic byte headers %PDF- and OpenXML ZIP content verification",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc1_chunk0")},
        description="File binary format signatures for upload security",
    ),
    EvalQuery(
        query_id="q20_cross_encoder_joint_attention",
        query_text="cross-encoder joint attention query-passage pair scoring",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc6_chunk1")},
        description="Full cross-attention scoring mechanism",
    ),
    EvalQuery(
        query_id="q21_retrieval_ranking_metrics",
        query_text="Normalized Discounted Cumulative Gain nDCG and Mean Reciprocal Rank MRR",
        category="technical_term",
        relevant_chunk_ids={make_eval_id("doc7_chunk1")},
        description="Standard ranking evaluation metrics",
    ),

    # Category 4: Numeric and Identifier Queries (6 queries)
    EvalQuery(
        query_id="q22_vector_dimensionality",
        query_text="384 dimensions dense embedding vector",
        category="numeric_identifier",
        relevant_chunk_ids={make_eval_id("doc3_chunk0"), make_eval_id("doc4_chunk0")},
        description="Exact dimensional output of BGE-small",
    ),
    EvalQuery(
        query_id="q23_chunk_target_size",
        query_text="1500 characters target chunk size and 200 characters overlap",
        category="numeric_identifier",
        relevant_chunk_ids={make_eval_id("doc2_chunk1")},
        description="Configured sliding window character parameters",
    ),
    EvalQuery(
        query_id="q24_max_file_size",
        query_text="25 MB maximum upload file size limit",
        category="numeric_identifier",
        relevant_chunk_ids={make_eval_id("doc1_chunk0")},
        description="Upload payload size threshold",
    ),
    EvalQuery(
        query_id="q25_token_limits",
        query_text="512 token limit and 450-token window size with 50-token overlap",
        category="numeric_identifier",
        relevant_chunk_ids={make_eval_id("doc3_chunk1")},
        description="Tokenizer sequence length ceiling and sub-window dimensions",
    ),
    EvalQuery(
        query_id="q26_server_port",
        query_text="port 8000 FastAPI backend and port 5173 Vite frontend",
        category="numeric_identifier",
        relevant_chunk_ids={make_eval_id("doc7_chunk0")},
        description="Network configuration and CORS ports",
    ),
    EvalQuery(
        query_id="q27_top_candidates_count",
        query_text="top 20 candidate passages from RRF for reranking",
        category="numeric_identifier",
        relevant_chunk_ids={make_eval_id("doc6_chunk1")},
        description="Candidate pool size passed to cross-encoder",
    ),

    # Category 5: Negative Queries with No Relevant Results (5 queries)
    EvalQuery(
        query_id="q28_neg_sourdough",
        query_text="how to bake artisanal sourdough bread using whole wheat flour and hydration",
        category="negative",
        relevant_chunk_ids=set(),
        description="Completely ungrounded culinary query expecting zero relevant chunks",
    ),
    EvalQuery(
        query_id="q29_neg_mars_rover",
        query_text="Perseverance Mars exploration rover atmospheric entry and descent telemetry",
        category="negative",
        relevant_chunk_ids=set(),
        description="Space exploration topic not present in document intelligence corpus",
    ),
    EvalQuery(
        query_id="q30_neg_quantum_crypto",
        query_text="BB84 quantum key distribution and entangled photon polarization states",
        category="negative",
        relevant_chunk_ids=set(),
        description="Quantum cryptography query expecting no matching document chunks",
    ),
    EvalQuery(
        query_id="q31_neg_ancient_rome",
        query_text="Julius Caesar crossing the Rubicon river and the Roman civil war legions",
        category="negative",
        relevant_chunk_ids=set(),
        description="Ancient history query with no relevant documents",
    ),
    EvalQuery(
        query_id="q32_neg_gardening",
        query_text="organic pest management for heirloom tomato blight and soil nitrogen deficiency",
        category="negative",
        relevant_chunk_ids=set(),
        description="Agricultural query expecting zero relevance in benchmark corpus",
    ),
]
