"""
DevOps AI Knowledge Assistant — Streamlit web UI.
Provides a chat interface with source attribution and sidebar pipeline controls.
Run with: streamlit run app.py
"""
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import streamlit as st

from urllib.parse import quote

from config.settings import (
    validate,
    SHOW_SOURCES,
    AZURE_DEVOPS_ORG,
    AZURE_DEVOPS_PROJECT,
    AZURE_DEVOPS_WIKI_ID,
    get,
)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bagira AI Assistant",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HASH_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")


def _format_source_name(raw_name: str) -> str:
    """Strip trailing 8-hex hash and replace underscores with spaces."""
    name = _HASH_SUFFIX_RE.sub("", raw_name)
    return name.replace("_", " ")


def _azure_base_url() -> str | None:
    """Return https://dev.azure.com/{org}/{project} or None if env vars are missing."""
    org = get(AZURE_DEVOPS_ORG)
    project = get(AZURE_DEVOPS_PROJECT)
    if org and project:
        return f"https://dev.azure.com/{quote(org, safe='')}/{quote(project, safe='')}"
    return None


def _render_sources(sources: list[dict]) -> None:
    """Show deduplicated sources inside an expander, with clickable Azure DevOps links."""
    if not sources:
        return
    base = _azure_base_url()
    wiki_id = get(AZURE_DEVOPS_WIKI_ID)
    with st.expander("Sources", expanded=False):
        for src in sources:
            doc_type = src.get("document_type", "unknown")
            if doc_type == "wiki":
                title = _format_source_name(src.get("page_title", "untitled"))
                wiki_path = src.get("wiki_path", "")
                if base and wiki_id and wiki_path:
                    encoded_path = quote(wiki_path, safe="/")
                    url = f"{base}/_wiki/wikis/{quote(wiki_id, safe='')}?pagePath={encoded_path}"
                    st.markdown(f"- **\\[wiki\\]** [{title}]({url})")
                else:
                    st.markdown(f"- **\\[wiki\\]** {title}")
            elif doc_type == "pbi":
                wid = src.get("work_item_id", "unknown")
                if base and wid != "unknown":
                    url = f"{base}/_workitems/edit/{wid}"
                    st.markdown(f"- **\\[pbi\\]** [PBI {wid}]({url})")
                else:
                    st.markdown(f"- **\\[pbi\\]** PBI {wid}")
            else:
                source_doc = _format_source_name(src.get("source_document", "unknown"))
                st.markdown(f"- **\\[{doc_type}\\]** {source_doc}")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar — data pipeline controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Chat")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Data Pipeline")
    st.caption("Fetch and index data from Azure DevOps.")

    def _run_step(label: str, func, **kwargs):
        """Run a pipeline function with spinner and status feedback."""
        with st.spinner(f"Running {label}..."):
            try:
                result = func(**kwargs)
                st.success(f"{label} completed (result: {result})")
            except Exception as exc:
                st.error(f"{label} failed: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch Wiki", use_container_width=True):
            from scripts.fetch_wiki import fetch_wiki
            _run_step("Fetch Wiki", fetch_wiki)
    with col2:
        if st.button("Fetch Work Items", use_container_width=True):
            from scripts.fetch_work_items import fetch_work_items
            _run_step("Fetch Work Items (PBI + Bug + Task)", fetch_work_items)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Clean", use_container_width=True):
            from scripts.clean_documents import clean_documents
            _run_step("Clean Documents", clean_documents)
    with col4:
        if st.button("Chunk", use_container_width=True):
            from scripts.chunk_documents import chunk_documents
            _run_step("Chunk Documents", chunk_documents)

    if st.button("Generate Embeddings", use_container_width=True):
        from scripts.generate_embeddings import generate_embeddings
        _run_step("Generate Embeddings", generate_embeddings)

    st.divider()

    if st.button("Run Full Pipeline", type="primary", use_container_width=True):
        from scripts.fetch_wiki import fetch_wiki
        from scripts.fetch_work_items import fetch_work_items
        from scripts.clean_documents import clean_documents
        from scripts.chunk_documents import chunk_documents
        from scripts.generate_embeddings import generate_embeddings

        steps = [
            ("Fetch Wiki", fetch_wiki),
            ("Fetch Work Items (PBI + Bug + Task)", fetch_work_items),
            ("Clean Documents", clean_documents),
            ("Chunk Documents", chunk_documents),
            ("Generate Embeddings", generate_embeddings),
        ]
        progress = st.progress(0, text="Starting pipeline...")
        for i, (label, func) in enumerate(steps):
            progress.progress((i) / len(steps), text=f"Running {label}...")
            try:
                result = func()
                st.success(f"{label} — done (result: {result})")
            except Exception as exc:
                st.error(f"{label} — failed: {exc}")
                break
        else:
            progress.progress(1.0, text="Pipeline complete!")

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
st.title("Bagira AI Assistant")
st.caption("Ask questions about your system (Wiki + PBI). Answers are grounded in retrieved documents.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and SHOW_SOURCES:
            _render_sources(msg.get("sources", []))

if question := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                from chatbot.chat_engine import answer
                reply, sources = answer(question, history=st.session_state.messages)
            except FileNotFoundError as exc:
                reply = (
                    f"**Setup required:** {exc}\n\n"
                    "Use the sidebar to run the pipeline, or run:\n"
                    "```\npython scripts/generate_embeddings.py\n```"
                )
                sources = []
            except Exception as exc:
                reply = f"**Error:** {exc}"
                sources = []

        st.markdown(reply)
        if SHOW_SOURCES:
            _render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources,
    })
