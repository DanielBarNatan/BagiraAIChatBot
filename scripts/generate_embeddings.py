"""
Load chunks from data/chunks/chunks.jsonl, generate OpenAI embeddings, and store in Chroma.
Persists to vector_db/chroma_store. Idempotent: re-run clears and re-indexes.
Run from project root so config is importable.
"""
import hashlib
import json
import sys
from pathlib import Path

# Add project root to path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from config.settings import (
    DATA_CHUNKS,
    VECTOR_STORE_PATH,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    get,
    ensure_dirs,
    validate,
    update_pipeline_status,
)

# LlamaIndex/Chroma imports after path setup

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHROMA_COLLECTION_NAME = "devops_assistant"


def _load_chunks():
    """Load chunk records from chunks.jsonl."""
    path = DATA_CHUNKS / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No chunks file at {path}. Run scripts/chunk_documents.py first.")
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _chunk_id(record: dict) -> str:
    """Stable id for deduplication (hash of text + source)."""
    key = f"{record.get('text', '')}|{record.get('source_document', '')}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def generate_embeddings() -> int:
    """
    Read chunks, embed with OpenAI, store in Chroma at vector_db/chroma_store.
    Returns number of documents indexed.
    """
    validate(require_azure=False)  # Only OpenAI required for this script
    ensure_dirs()
    chunks = _load_chunks()
    if not chunks:
        logger.warning("No chunks found in chunks.jsonl")
        return 0

    import chromadb
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding

    # Persistent Chroma client
    db = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    # Clear existing collection so re-run is idempotent
    try:
        db.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    collection = db.get_or_create_collection(
        CHROMA_COLLECTION_NAME,
        metadata={"description": "DevOps AI Assistant chunks"},
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    embed_model = OpenAIEmbedding(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=get(OPENAI_API_KEY),
    )

    documents = []
    for rec in chunks:
        text = rec.get("text", "")
        if not text:
            continue
        doc_id = _chunk_id(rec)
        meta = {k: v for k, v in rec.items() if k != "text" and v is not None}
        doc = Document(text=text, doc_id=doc_id, metadata=meta)
        documents.append(doc)

    logger.info("Building index for %s documents with OpenAI embeddings...", len(documents))
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info("Index persisted to %s", VECTOR_STORE_PATH)
    update_pipeline_status("generate_embeddings", len(documents))
    return len(documents)


if __name__ == "__main__":
    try:
        generate_embeddings()
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)
