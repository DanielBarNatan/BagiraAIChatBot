"""
Fetch Azure DevOps Wiki pages and save as markdown under data/raw/wiki/.
Uses PAT authentication. Run from project root so config is importable.

Each file begins with YAML frontmatter (wiki_path, page_title) so downstream
chunking/embeddings can see hierarchy; filenames include a path hash to avoid
collisions when different wiki paths sanitize to the same title.
"""
import base64
import hashlib
import logging
import re
import sys
from pathlib import Path
from urllib.parse import quote

import requests

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
    AZURE_DEVOPS_ORG,
    AZURE_DEVOPS_PAT,
    AZURE_DEVOPS_PROJECT,
    AZURE_DEVOPS_WIKI_ID,
    DATA_RAW_WIKI,
    get,
    ensure_dirs,
    validate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_VERSION = "7.1"


def _auth_headers() -> dict:
    """Build Authorization header for Azure DevOps PAT (Basic base64(:PAT))."""
    pat = get(AZURE_DEVOPS_PAT)
    if not pat:
        raise ValueError("AZURE_DEVOPS_PAT is not set")
    encoded = base64.b64encode(f":{pat}".encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}


def _sanitize_filename(name: str) -> str:
    """Make a safe filename from a page title or path."""
    name = re.sub(r"[^\w\s\-.]", "", name)
    name = re.sub(r"\s+", "_", name).strip("_") or "page"
    return name[:200]


def _wiki_path_fingerprint(wiki_path: str) -> str:
    """Short stable id for a wiki path (unique per path for filename suffix)."""
    normalized = wiki_path.strip() or "/"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]


def _yaml_double_quoted(s: str) -> str:
    """YAML double-quoted scalar for values that may contain special characters."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _format_wiki_file_body(wiki_path: str, page_title: str, content: str) -> str:
    """YAML frontmatter plus raw page markdown body (what Azure returns as content)."""
    wp = _yaml_double_quoted(wiki_path)
    pt = _yaml_double_quoted(page_title)
    header = f"---\nwiki_path: {wp}\npage_title: {pt}\n---\n\n"
    return header + (content or "")


def _wiki_output_filename(page_title: str, wiki_path: str) -> str:
    """
    Human-readable stem from title plus path fingerprint so two different paths
    with the same title never share one file.
    """
    base = _sanitize_filename(page_title) or _sanitize_filename(
        wiki_path.strip("/").replace("/", "_") or "page"
    )
    base = base[:120]
    if not base:
        base = "page"
    return f"{base}_{_wiki_path_fingerprint(wiki_path)}.md"


def _get_wiki_id(session: requests.Session, base_url: str) -> str:
    """
    Resolve wiki identifier to the actual API id/name that works in URL paths.
    Lists all wikis and matches by AZURE_DEVOPS_WIKI_ID (name or id), else project wiki.
    """
    url = f"{base_url}/_apis/wiki/wikis?api-version={API_VERSION}"
    r = session.get(url, headers=_auth_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    wikis = data.get("value", [])
    logger.info("Available wikis: %s", [(w.get("name"), w.get("id")) for w in wikis])

    hint = get(AZURE_DEVOPS_WIKI_ID)
    if hint:
        hint_normalized = re.sub(r"[\s\-_.]", "", hint).lower()
        for w in wikis:
            name = w.get("name", "")
            wid = w.get("id", "")
            if hint in (name, wid):
                chosen = wid or name
                logger.info("Matched wiki hint %r (exact) -> id=%s", hint, chosen)
                return chosen
            if re.sub(r"[\s\-_.]", "", name).lower().startswith(hint_normalized):
                chosen = wid or name
                logger.info("Matched wiki hint %r (fuzzy) -> id=%s, name=%s", hint, chosen, name)
                return chosen

    for w in wikis:
        if w.get("name") == get(AZURE_DEVOPS_PROJECT):
            chosen = w.get("id") or w.get("name")
            logger.info("Using project wiki -> id=%s", chosen)
            return chosen

    if wikis:
        chosen = wikis[0].get("id") or wikis[0].get("name")
        logger.info("Falling back to first wiki -> id=%s", chosen)
        return chosen

    raise ValueError("No wikis found in the project")


def _get_page(session: requests.Session, base_url: str, wiki_id: str, page_path: str) -> dict | None:
    """Fetch a single wiki page by path. Returns None on 404 or error."""
    encoded_wiki_id = quote(wiki_id, safe="")
    url = f"{base_url}/_apis/wiki/wikis/{encoded_wiki_id}/pages"
    params = {"path": page_path, "includeContent": "true", "recursionLevel": "oneLevel", "api-version": API_VERSION}
    try:
        r = session.get(url, headers=_auth_headers(), params=params, timeout=30)
        if r.status_code == 404:
            logger.debug("Page not found (404): %s", page_path)
            return None
        if not r.ok:
            logger.warning("HTTP %s for page %s: %s", r.status_code, page_path, r.text[:500])
            return None
        return r.json()
    except requests.RequestException as e:
        logger.warning("Failed to get page %s: %s", page_path, e)
        return None


def _collect_pages_recursive(
    session: requests.Session,
    base_url: str,
    wiki_id: str,
    path: str,
    collected: list[dict],
) -> None:
    """
    Recursively collect wiki pages starting at path.
    Azure DevOps returns a single page; subPages (if any) are listed and we fetch each.
    """
    logger.info("Fetching page: %s", path)
    page = _get_page(session, base_url, wiki_id, path)
    if not page:
        logger.warning("No page returned for path: %s", path)
        return
    collected.append(page)
    subs = page.get("subPages", []) or []
    logger.info("Page %s has %d subPages", path, len(subs))
    for sub in subs:
        sub_path = sub.get("path") or sub.get("path", path + "/" + str(sub.get("id", "")))
        if sub_path and sub_path != path:
            _collect_pages_recursive(session, base_url, wiki_id, sub_path, collected)


def fetch_wiki() -> int:
    """
    Fetch all wiki pages and save to data/raw/wiki/.
    Returns the number of pages saved.
    """
    validate(require_azure=True)
    ensure_dirs()
    org = get(AZURE_DEVOPS_ORG)
    project = get(AZURE_DEVOPS_PROJECT)
    base_url = f"https://dev.azure.com/{org}/{project}"
    session = requests.Session()
    wiki_id = _get_wiki_id(session, base_url)
    logger.info("Using wiki id: %s", wiki_id)
    collected: list[dict] = []
    _collect_pages_recursive(session, base_url, wiki_id, "/", collected)
    logger.info("Collected %s pages from wiki tree", len(collected))
    saved = 0
    for page in collected:
        content = page.get("content") or ""
        wiki_path = page.get("path") or page.get("id") or "/"
        if isinstance(wiki_path, str):
            wiki_path = wiki_path.strip() or "/"
        else:
            wiki_path = str(wiki_path)
        page_title = page.get("pageTitle") or wiki_path
        full_text = _format_wiki_file_body(wiki_path, page_title, content)
        out_path = DATA_RAW_WIKI / _wiki_output_filename(page_title, wiki_path)
        if out_path.exists() and out_path.read_text(encoding="utf-8") == full_text:
            continue
        out_path.write_text(full_text, encoding="utf-8")
        saved += 1
        logger.info("Saved: %s -> %s", wiki_path, out_path.name)
    logger.info("Done. Saved %s wiki pages to %s", saved, DATA_RAW_WIKI)
    return saved


if __name__ == "__main__":
    try:
        fetch_wiki()
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)
