"""
Semantic search over the Chroma vector store.
Loads persisted index, retrieves candidate chunks, then reranks with a
cross-encoder for higher-precision results.
"""
from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from config.settings import (
    VECTOR_STORE_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    RETRIEVER_TOP_K,
    get,
)

CHROMA_COLLECTION_NAME = "devops_assistant"

RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1").strip().lower() in ("1", "true", "yes")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATE_MULTIPLIER = 2

_cross_encoder = None


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


def retrieve(query: str, top_k: int | None = None) -> list[tuple[str, dict]]:
    """
    Return the top-k most relevant chunks for the query.
    When reranking is enabled, fetches extra candidates from the vector store
    and then re-scores them with a cross-encoder for higher precision.
    Returns list of (text, metadata) for each chunk.
    """
    top_k = top_k if top_k is not None else RETRIEVER_TOP_K
    candidate_k = top_k * RERANK_CANDIDATE_MULTIPLIER if RERANK_ENABLED else top_k

    index = _get_index()
    retriever = index.as_retriever(similarity_top_k=candidate_k)
    nodes = retriever.retrieve(query)
    chunks = [(node.text, node.metadata or {}) for node in nodes]

    if RERANK_ENABLED:
        chunks = _rerank(query, chunks, top_k)

    return chunks
