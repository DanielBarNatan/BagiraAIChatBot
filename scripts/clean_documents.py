"""
Clean raw documents: strip HTML, remove images, normalize whitespace.
Reads from data/raw/wiki/ and data/raw/pbi/; writes to data/cleaned/wiki/ and data/cleaned/pbi/.
Run from project root so config is importable.
"""
import re
import sys
from pathlib import Path

# Add project root to path when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_RAW_WIKI,
    DATA_RAW_PBI,
    DATA_CLEANED_WIKI,
    DATA_CLEANED_PBI,
    ensure_dirs,
)

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _strip_html(text: str) -> str:
    """Remove HTML tags. Use BeautifulSoup if available, else regex."""
    if HAS_BS4:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    return re.sub(r"<[^>]+>", " ", text)


def _remove_markdown_images(text: str) -> str:
    """Remove markdown image syntax ![](url) or ![alt](url)."""
    return re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and newlines, trim."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split YAML frontmatter from body. Returns (frontmatter_block, body).
    frontmatter_block includes the --- delimiters so it can be re-attached as-is."""
    m = _FRONTMATTER_RE.match(text)
    if m:
        return text[: m.end()], text[m.end():]
    return "", text


def clean_text(text: str, preserve_frontmatter: bool = False) -> str:
    """Apply all cleaning steps to a single document text.
    If preserve_frontmatter is True, YAML frontmatter is kept intact."""
    frontmatter = ""
    if preserve_frontmatter:
        frontmatter, text = _split_frontmatter(text)
    text = _strip_html(text)
    text = _remove_markdown_images(text)
    text = _normalize_whitespace(text)
    return frontmatter + text


def clean_documents() -> tuple[int, int]:
    """
    Clean all raw wiki and PBI files and save to data/cleaned/.
    Returns (wiki_count, pbi_count) of files written.
    """
    ensure_dirs()
    wiki_count = 0
    for path in DATA_RAW_WIKI.glob("*"):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in (".md", ".txt", ".html"):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Skip %s: %s", path, e)
            continue
        cleaned = clean_text(raw, preserve_frontmatter=True)
        out = DATA_CLEANED_WIKI / path.name
        out.write_text(cleaned, encoding="utf-8")
        wiki_count += 1
    pbi_count = 0
    for path in DATA_RAW_PBI.glob("*.txt"):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Skip %s: %s", path, e)
            continue
        cleaned = clean_text(raw)
        out = DATA_CLEANED_PBI / path.name
        out.write_text(cleaned, encoding="utf-8")
        pbi_count += 1
    logger.info("Cleaned %s wiki and %s PBI files -> data/cleaned/", wiki_count, pbi_count)
    return wiki_count, pbi_count


if __name__ == "__main__":
    try:
        clean_documents()
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)
