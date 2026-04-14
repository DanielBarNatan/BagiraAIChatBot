"""
Orchestrates retrieval, prompt building, and LLM call to produce a grounded answer.
"""
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import RETRIEVER_TOP_K
from chatbot.retriever import retrieve
from chatbot.prompt_builder import build_prompt


def _get_llm():
    from config.settings import OPENAI_API_KEY, OPENAI_CHAT_MODEL, get
    from openai import OpenAI
    client = OpenAI(api_key=get(OPENAI_API_KEY))
    return client, OPENAI_CHAT_MODEL


def _deduplicate_sources(chunks: list[tuple[str, dict]]) -> list[dict]:
    """Return unique source metadata dicts, keyed by (document_type, source_document)."""
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for _text, meta in chunks:
        key = (meta.get("document_type", ""), meta.get("source_document", ""))
        if key not in seen:
            seen.add(key)
            sources.append(meta)
    return sources


def answer(question: str) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks, build context-only prompt, call OpenAI.
    Returns (reply_text, deduplicated_sources).
    """
    chunks = retrieve(question, top_k=RETRIEVER_TOP_K)
    if not chunks:
        return (
            "I couldn't find any relevant information in the knowledge base. "
            "Make sure you have run the ingestion and embedding scripts (fetch wiki/work items, "
            "clean, chunk, generate_embeddings) so that the vector store is populated."
        ), []
    prompt = build_prompt(chunks, question)
    client, model = _get_llm()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    reply = (response.choices[0].message.content or "").strip()
    sources = _deduplicate_sources(chunks)
    return reply, sources
