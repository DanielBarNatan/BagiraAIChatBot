"""
Hybrid retrieval over the Chroma vector store + BM25 keyword index.
1. Semantic search via Chroma (vector similarity)
2. Keyword search via BM25 (exact word matching)
3. Merge & deduplicate candidates from both
4. Rerank combined candidates with a cross-encoder
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from rank_bm25 import BM25Okapi

from config.settings import (
    VECTOR_STORE_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    DATA_CHUNKS,
    get,
)

logger = logging.getLogger(__name__)

CHROMA_COLLECTION_NAME = "devops_assistant"

RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATE_MULTIPLIER = 4

BM25_ENABLED = os.environ.get("BM25_ENABLED", "1").strip().lower() in ("1", "true", "yes")

_cross_encoder = None
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[tuple[str, dict]] | None = None

_TOKENIZE_RE = re.compile(r"[a-zA-Z0-9\u0590-\u05FF]+")


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase alphanumeric + Hebrew tokens."""
    return _TOKENIZE_RE.findall(text.lower())


def _load_bm25_index() -> tuple[BM25Okapi, list[tuple[str, dict]]]:
    """Build BM25 index from chunks.jsonl. Loaded once, cached globally."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None and _bm25_chunks is not None:
        return _bm25_index, _bm25_chunks

    chunks_path = DATA_CHUNKS / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            "Run: python scripts/chunk_documents.py"
        )

    chunks: list[tuple[str, dict]] = []
    tokenized_corpus: list[list[str]] = []

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = entry.pop("text", "")
            meta = entry
            chunks.append((text, meta))
            tokenized_corpus.append(_tokenize(text))

    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_chunks = chunks
    logger.info("BM25 index loaded: %d chunks", len(chunks))
    return _bm25_index, _bm25_chunks


def _bm25_search(query: str, top_k: int) -> list[tuple[str, dict]]:
    """Return top_k chunks by BM25 keyword relevance."""
    bm25, chunks = _load_bm25_index()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (heavy import, cached after first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def _rerank(query: str, chunks: list[tuple[str, dict]], top_k: int) -> list[tuple[str, dict]]:
    """Re-score chunks with a cross-encoder and return the top_k best."""
    if not chunks:
        return chunks
    encoder = _get_cross_encoder()
    pairs = [[query, text] for text, _meta in chunks]
    scores = encoder.predict(pairs)
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _score, chunk in scored[:top_k]]


def _get_index():
    """Load or create VectorStoreIndex from persisted Chroma. Raises if DB missing."""
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}. "
            "Run: python scripts/generate_embeddings.py"
        )
    db = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    try:
        collection = db.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        raise FileNotFoundError(
            f"Chroma collection '{CHROMA_COLLECTION_NAME}' not found. "
            "Run: python scripts/generate_embeddings.py"
        ) from e
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = OpenAIEmbedding(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=get(OPENAI_API_KEY),
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index


def _merge_chunks(
    semantic: list[tuple[str, dict]],
    keyword: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    """Merge two chunk lists, deduplicating by text content.
    Preserves order: semantic results first, then keyword-only results."""
    seen_texts: set[str] = set()
    merged: list[tuple[str, dict]] = []
    for text, meta in semantic:
        key = text[:200]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append((text, meta))
    for text, meta in keyword:
        key = text[:200]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append((text, meta))
    return merged


def retrieve(query: str, top_k: int | None = None) -> list[tuple[str, dict]]:
    """
    Hybrid retrieval: combines semantic (vector) search with BM25 keyword
    search, then reranks all candidates with a cross-encoder.
    Returns list of (text, metadata) for the top-k best chunks.
    """
    top_k = top_k if top_k is not None else RETRIEVER_TOP_K
    candidate_k = top_k * RERANK_CANDIDATE_MULTIPLIER if RERANK_ENABLED else top_k

    index = _get_index()
    retriever = index.as_retriever(similarity_top_k=candidate_k)
    nodes = retriever.retrieve(query)
    semantic_chunks = [(node.text, node.metadata or {}) for node in nodes]

    keyword_chunks: list[tuple[str, dict]] = []
    if BM25_ENABLED:
        try:
            keyword_chunks = _bm25_search(query, top_k=candidate_k)
        except FileNotFoundError:
            logger.warning("BM25 index unavailable — falling back to semantic-only retrieval")
        except Exception:
            logger.exception("BM25 search failed — falling back to semantic-only retrieval")

    if keyword_chunks:
        combined = _merge_chunks(semantic_chunks, keyword_chunks)
    else:
        combined = semantic_chunks

    if RERANK_ENABLED:
        combined = _rerank(query, combined, top_k)

    return combined
