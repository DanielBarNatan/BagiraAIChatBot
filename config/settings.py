"""
Configuration for the DevOps AI Knowledge Assistant.
Loads from environment variables (use python-dotenv in main.py or at app entry).
Validates required vars at import or via validate().
"""
import os
from pathlib import Path

# Project root: directory containing main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Paths (relative to project root) ---
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_RAW_WIKI = DATA_RAW / "wiki"
DATA_RAW_PBI = DATA_RAW / "pbi"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_CLEANED_WIKI = DATA_CLEANED / "wiki"
DATA_CLEANED_PBI = DATA_CLEANED / "pbi"
DATA_CHUNKS = PROJECT_ROOT / "data" / "chunks"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_db" / "chroma_store"

# --- Environment variable names ---
OPENAI_API_KEY = "OPENAI_API_KEY"
AZURE_DEVOPS_PAT = "AZURE_DEVOPS_PAT"
AZURE_DEVOPS_ORG = "AZURE_DEVOPS_ORG"
AZURE_DEVOPS_PROJECT = "AZURE_DEVOPS_PROJECT"
AZURE_DEVOPS_WIKI_ID = "AZURE_DEVOPS_WIKI_ID"

# --- Optional settings with defaults ---
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o")
RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "5"))
SHOW_SOURCES = os.environ.get("SHOW_SOURCES", "1").strip().lower() in ("1", "true", "yes")


def get(key: str, default: str = "") -> str:
    """Get an environment variable."""
    return os.environ.get(key, default).strip()


def validate(require_azure: bool = True) -> None:
    """
    Validate that required environment variables are set.
    Raises ValueError with a clear message if any are missing.
    """
    missing = []
    if not get(OPENAI_API_KEY):
        missing.append(OPENAI_API_KEY)
    if require_azure:
        if not get(AZURE_DEVOPS_PAT):
            missing.append(AZURE_DEVOPS_PAT)
        if not get(AZURE_DEVOPS_ORG):
            missing.append(AZURE_DEVOPS_ORG)
        if not get(AZURE_DEVOPS_PROJECT):
            missing.append(AZURE_DEVOPS_PROJECT)
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and set the values."
        )


def ensure_dirs() -> None:
    """Create project data and vector_db directories if they do not exist."""
    for path in (
        DATA_RAW_WIKI,
        DATA_RAW_PBI,
        DATA_CLEANED_WIKI,
        DATA_CLEANED_PBI,
        DATA_CHUNKS,
        VECTOR_STORE_PATH,
    ):
        path.mkdir(parents=True, exist_ok=True)


# --- Pipeline status tracking ---
PIPELINE_STATUS_PATH = PROJECT_ROOT / "data" / "pipeline_status.json"


def update_pipeline_status(step: str, count: int) -> None:
    """Record when a pipeline step last ran and how many items it produced."""
    import json
    from datetime import datetime, timezone

    status: dict = {}
    if PIPELINE_STATUS_PATH.exists():
        try:
            status = json.loads(PIPELINE_STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            status = {}

    status[step] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": count,
    }
    PIPELINE_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_pipeline_status() -> dict:
    """Return the saved pipeline status dict, or empty dict if unavailable."""
    import json

    if not PIPELINE_STATUS_PATH.exists():
        return {}
    try:
        return json.loads(PIPELINE_STATUS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
