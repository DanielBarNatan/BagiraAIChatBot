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
    with st.expander("View sources", expanded=False):
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
# Avatars
# ---------------------------------------------------------------------------
USER_AVATAR = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E"
    "%3Ccircle cx='20' cy='20' r='20' fill='%23334155'/%3E"
    "%3Ccircle cx='20' cy='15' r='6' fill='%2394a3b8'/%3E"
    "%3Cellipse cx='20' cy='30' rx='10' ry='7' fill='%2394a3b8'/%3E"
    "%3C/svg%3E"
)
BOT_AVATAR = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E"
    "%3Ccircle cx='20' cy='20' r='20' fill='%231e1b4b'/%3E"
    "%3Cpath d='M20 6l2.5 6.5L29 15l-6.5 2.5L20 24l-2.5-6.5L11 15l6.5-2.5z' "
    "fill='%23a78bfa'/%3E"
    "%3Cpath d='M30 22l1.5 3.5L35 27l-3.5 1.5L30 32l-1.5-3.5L25 27l3.5-1.5z' "
    "fill='%23c4b5fd' opacity='0.7'/%3E"
    "%3Cpath d='M12 25l1 2.5L16 29l-3 1L12 32.5l-1-2.5L8 29l3-1z' "
    "fill='%23c4b5fd' opacity='0.5'/%3E"
    "%3C/svg%3E"
)

# ---------------------------------------------------------------------------
# Custom CSS — clean dark theme + RTL support
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── RTL support ── */
    .stMarkdown p, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        unicode-bidi: plaintext;
        direction: auto;
        text-align: start;
    }

    /* ── Chat message cards ── */
    .stChatMessage {
        background: transparent;
        border: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 0;
        padding: 1.1rem 0.5rem;
        margin-bottom: 0;
    }

    /* Avatar images */
    .stChatMessage img[data-testid],
    .stChatMessage [data-testid="chatAvatarIcon-user"],
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        border-radius: 50%;
    }

    /* ── Message text ── */
    .stChatMessage [data-testid="stMarkdownContainer"] {
        line-height: 1.7;
        font-weight: 400;
        color: rgba(235, 235, 245, 0.88);
    }

    /* ── Chat input ── */
    .stChatInput > div {
        background: #1a1a2e !important;
        border: 1px solid #2a2a3e !important;
        border-radius: 24px !important;
        transition: border-color 0.2s ease;
    }
    .stChatInput > div:focus-within {
        border-color: #5b5ea6 !important;
    }

    /* ── Sources expander ── */
    .stChatMessage .stExpander {
        border: 1px solid #1e1e30;
        border-radius: 8px;
        background: #13131f;
        margin-top: 0.6rem;
    }
    .stChatMessage .stExpander summary {
        font-size: 0.8rem;
        font-weight: 500;
        color: rgba(235, 235, 245, 0.5);
    }
    .stChatMessage .stExpander [data-testid="stMarkdownContainer"] {
        font-size: 0.84rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d0d17 !important;
        border-right: 1px solid #1a1a2e;
    }
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        border: 1px solid #252538;
        background: #151525;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1c1c30;
        border-color: #5b5ea6;
    }

    /* ── Custom header ── */
    .bagira-header {
        padding: 0.25rem 0 1rem 0;
        border-bottom: 1px solid #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .bagira-header h1 {
        font-size: 1.75rem;
        font-weight: 600;
        margin: 0 0 0.2rem 0;
        color: #c4bfe0;
    }
    .bagira-header p {
        font-size: 0.84rem;
        color: rgba(235, 235, 245, 0.35);
        margin: 0;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

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
st.markdown(
    '<div class="bagira-header">'
    "<h1>Bagira AI Assistant</h1>"
    "<p>Ask about your system (Wiki + PBI) in English or Hebrew "
    "| שאל שאלות על המערכת בעברית או באנגלית</p>"
    "</div>",
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    _av = BOT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=_av):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and SHOW_SOURCES:
            _render_sources(msg.get("sources", []))

if question := st.chat_input("Ask a question... | ...שאל שאלה"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Thinking..."):
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
