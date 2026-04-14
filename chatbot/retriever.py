"""
Semantic search over the Chroma vector store.
Loads persisted index and returns top-k relevant chunks for a query.
"""
from pathlib import Path

# Ensure project root on path when chatbot is used as package or from main
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
    Returns list of (text, metadata) for each chunk.
    """
    top_k = top_k if top_k is not None else RETRIEVER_TOP_K
    index = _get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [(node.text, node.metadata or {}) for node in nodes]
