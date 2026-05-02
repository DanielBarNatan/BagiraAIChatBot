"""
Split cleaned documents into fixed-size chunks with metadata.
Reads from data/cleaned/wiki/ and data/cleaned/pbi/; writes JSONL to data/chunks/chunks.jsonl.
Chunk size 800-1000 chars with overlap; split on sentence/paragraph where possible.
Run from project root so config is importable.
"""
import json
import re
import sys
from pathlib import Path

# Add project root to path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_CLEANED_WIKI, DATA_CLEANED_PBI, DATA_CHUNKS, ensure_dirs

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter key-value pairs and return (meta_dict, body).
    Only handles simple `key: "value"` or `key: value` lines (no nested YAML)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_fm = m.group(1)
    body = text[m.end():]
    meta: dict[str, str] = {}
    for line in raw_fm.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip().strip('"')
        meta[key.strip()] = val
    return meta, body


def _split_into_sentences_or_paragraphs(text: str) -> list[str]:
    """Split text into segments (paragraphs first, then sentences) for cleaner chunk boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] if text.strip() else []
    segments = []
    for p in paragraphs:
        if len(p) <= CHUNK_SIZE:
            segments.append(p)
        else:
            # Split long paragraph by sentences
            for s in re.split(r"(?<=[.!?])\s+", p):
                s = s.strip()
                if s:
                    segments.append(s)
    return segments


def _build_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Build overlapping chunks. Prefer splitting on segment boundaries;
    if a segment exceeds chunk_size, split it by character with overlap.
    """
    segments = _split_into_sentences_or_paragraphs(text)
    chunks = []
    current = []
    current_len = 0
    for seg in segments:
        if current_len + len(seg) + (1 if current else 0) <= chunk_size:
            current.append(seg)
            current_len += len(seg) + (1 if current else 0)
        else:
            if current:
                chunk_text = " ".join(current)
                chunks.append(chunk_text)
            # Start new chunk with overlap: keep last segment(s) that fit in overlap
            overlap_remaining = overlap
            overlap_parts = []
            for s in reversed(current):
                if overlap_remaining >= len(s):
                    overlap_parts.append(s)
                    overlap_remaining -= len(s)
                else:
                    break
            current = list(reversed(overlap_parts))
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
            if len(seg) <= chunk_size:
                current.append(seg)
                current_len += len(seg) + 1
            else:
                # Split long segment by character with overlap
                start = 0
                while start < len(seg):
                    end = min(start + chunk_size, len(seg))
                    chunks.append(seg[start:end])
                    start = end - overlap if end < len(seg) else len(seg)
                current = []
                current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents() -> int:
    """
    Chunk all cleaned documents and write data/chunks/chunks.jsonl.
    Each line is JSON: { "text", "source_document", "document_type", "page_title" or "work_item_id" }.
    Returns total number of chunks written.
    """
    ensure_dirs()
    out_path = DATA_CHUNKS / "chunks.jsonl"
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for doc_dir, doc_type in [(DATA_CLEANED_WIKI, "wiki"), (DATA_CLEANED_PBI, "pbi")]:
            if not doc_dir.exists():
                continue
            for path in sorted(doc_dir.iterdir()):
                if path.is_dir() or path.name.startswith("."):
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    logger.warning("Skip %s: %s", path, e)
                    continue
                if not text.strip():
                    continue
                fm_meta, body = _parse_frontmatter(text) if doc_type == "wiki" else ({}, text)
                title = fm_meta.get("page_title") or path.stem
                wiki_path = fm_meta.get("wiki_path", "")
                work_item_id = title.replace("pbi_", "") if doc_type == "pbi" and title.lower().startswith("pbi_") else None
                # Prepend wiki path/title (or PBI id/title) to chunk text so
                # embeddings and search can match on page/item identity.
                header_lines = []
                if doc_type == "wiki":
                    if wiki_path:
                        header_lines.append(f"[Wiki path: {wiki_path}]")
                    if title:
                        header_lines.append(f"[Title: {title}]")
                elif doc_type == "pbi":
                    if work_item_id:
                        header_lines.append(f"[PBI {work_item_id}: {title}]")
                    elif title:
                        header_lines.append(f"[Title: {title}]")
                chunk_header = "\n".join(header_lines) + "\n\n" if header_lines else ""

                chunks = _build_chunks(body if fm_meta else text)
                for i, chunk_text in enumerate(chunks):
                    text_with_header = chunk_header + chunk_text
                    meta = {
                        "text": text_with_header,
                        "source_document": path.name,
                        "document_type": doc_type,
                    }
                    if doc_type == "wiki":
                        meta["page_title"] = title
                        if wiki_path:
                            meta["wiki_path"] = wiki_path
                    else:
                        meta["work_item_id"] = work_item_id or title
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    total += 1
    logger.info("Wrote %s chunks to %s", total, out_path)
    return total


if __name__ == "__main__":
    try:
        chunk_documents()
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)
