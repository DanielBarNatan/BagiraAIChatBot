"""
Build a strict context-only prompt so the LLM answers only from retrieved documents.
"""
from urllib.parse import quote

from config.settings import (
    AZURE_DEVOPS_ORG,
    AZURE_DEVOPS_PROJECT,
    AZURE_DEVOPS_WIKI_ID,
    get,
)

MAX_CONTEXT_CHARS = 30_000


def _azure_base_url() -> str | None:
    org = get(AZURE_DEVOPS_ORG)
    project = get(AZURE_DEVOPS_PROJECT)
    if org and project:
        return f"https://dev.azure.com/{quote(org, safe='')}/{quote(project, safe='')}"
    return None


def _source_label(meta: dict) -> str:
    """Build a source label with a markdown link when possible."""
    doc_type = meta.get("document_type", "unknown")
    base = _azure_base_url()

    if doc_type == "wiki":
        title = meta.get("page_title", "untitled")
        wiki_path = meta.get("wiki_path", "")
        wiki_id = get(AZURE_DEVOPS_WIKI_ID)
        if base and wiki_id and wiki_path:
            encoded_path = quote(wiki_path, safe="/")
            url = f"{base}/_wiki/wikis/{quote(wiki_id, safe='')}?pagePath={encoded_path}"
            return f"[Source: wiki | [{wiki_path}]({url})]"
        if wiki_path:
            return f"[Source: wiki | {wiki_path}]"
        return f"[Source: wiki | {title}]"

    if doc_type == "pbi":
        wid = meta.get("work_item_id", "")
        if base and wid:
            url = f"{base}/_workitems/edit/{wid}"
            return f"[Source: pbi | [PBI {wid}]({url})]"
        return f"[Source: pbi | {wid}]" if wid else "[Source: pbi]"

    source_doc = meta.get("source_document", "")
    return f"[Source: {doc_type} | {source_doc}]" if source_doc else f"[Source: {doc_type}]"


def build_prompt(
    retrieved_chunks: list[tuple[str, dict]],
    user_question: str,
) -> str:
    """
    Build the prompt from (text, metadata) tuples.
    Each context block is prefixed with a source label so the LLM can cite it.
    Truncates total context if too long.
    """
    instruction = (
        "Answer the question using ONLY the context below. "
        "If the context contains a clear and direct answer, provide it confidently with source citations. "
        "If the context does not directly answer the question but contains related information, "
        "provide a best-effort answer based on what you found and start your response with: "
        "\"I couldn't find an exact answer, but based on related information I found, "
        "here is what may be relevant (note — this may not be fully accurate):\"\n"
        "Only if the context contains absolutely nothing relevant to the question, "
        "say you do not know.\n"
        "When citing sources, reproduce the markdown link from the source label "
        "exactly as it appears (do not strip or rewrite the URL).\n\n"
    )
    labeled_blocks = [
        f"{_source_label(meta)}\n{text}" for text, meta in retrieved_chunks
    ]
    context_block = "\n\n---\n\n".join(labeled_blocks)
    if len(context_block) > MAX_CONTEXT_CHARS:
        context_block = context_block[:MAX_CONTEXT_CHARS] + "\n\n[... context truncated ...]"
    return (
        f"{instruction}Context:\n{context_block}\n\nQuestion:\n{user_question}"
    )


def build_stats_context(stats_text: str, user_question: str) -> str:
    """Build a prompt that includes structured PBI statistics for aggregate queries."""
    instruction = (
        "You are given structured PBI statistics data below. "
        "Use this data to answer the user's question accurately. "
        "Present numbers clearly and format your answer in a readable way.\n\n"
    )
    return f"{instruction}PBI Statistics:\n{stats_text}\n\nQuestion:\n{user_question}"
