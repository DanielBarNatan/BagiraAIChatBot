"""
Fetch Azure DevOps Work Items (PBI) and save as text under data/raw/pbi/.
Uses PAT authentication. Run from project root so config is importable.
"""
import base64
import sys
from pathlib import Path

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
    DATA_RAW_PBI,
    get,
    ensure_dirs,
    validate,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_VERSION = "7.0"
BATCH_SIZE = 200  # Azure DevOps allows up to 200 IDs per work items request


def _auth_headers() -> dict:
    """Build Authorization header for Azure DevOps PAT (Basic base64(:PAT))."""
    pat = get(AZURE_DEVOPS_PAT)
    if not pat:
        raise ValueError("AZURE_DEVOPS_PAT is not set")
    encoded = base64.b64encode(f":{pat}".encode()).decode()
    return {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}


def _wiql_query_ids(session: requests.Session, base_url: str) -> list[int]:
    """Run Wiql to get all Product Backlog Item IDs."""
    url = f"{base_url}/_apis/wit/wiql?api-version={API_VERSION}"
    body = {
        "query": (
            "SELECT [System.Id], [System.Title], [System.Description], [System.State] "
            "FROM WorkItems WHERE [System.WorkItemType] = 'Product Backlog Item' ORDER BY [System.ChangedDate] DESC"
        )
    }
    r = session.post(url, headers=_auth_headers(), json=body, timeout=30)
    r.raise_for_status()
    data = r.json()
    ids = []
    for row in data.get("workItems", []) or []:
        wid = row.get("id")
        if wid is not None:
            ids.append(int(wid))
    return ids


def _get_work_items_batch(session: requests.Session, base_url: str, ids: list[int]) -> list[dict]:
    """Fetch work item details for a batch of IDs."""
    if not ids:
        return []
    url = f"{base_url}/_apis/wit/workitems"
    params = {"ids": ",".join(map(str, ids)), "$expand": "All", "api-version": API_VERSION}
    r = session.get(url, headers=_auth_headers(), params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("value", [])


def _work_item_to_text(wi: dict) -> str:
    """Build a single text representation of a work item (title, description, state, acceptance criteria)."""
    fields = wi.get("fields") or {}
    title = fields.get("System.Title") or fields.get("System.Id") or "Untitled"
    state = fields.get("System.State") or ""
    description = fields.get("System.Description") or ""
    # Acceptance criteria often in custom field or "Microsoft.VSTS.Common.AcceptanceCriteria"
    acceptance = (
        fields.get("Microsoft.VSTS.Common.AcceptanceCriteria")
        or fields.get("Acceptance Criteria")
        or ""
    )
    parts = [f"Title: {title}", f"State: {state}"]
    if description:
        parts.append(f"Description:\n{description}")
    if acceptance:
        parts.append(f"Acceptance Criteria:\n{acceptance}")
    return "\n\n".join(parts)


def fetch_work_items() -> int:
    """
    Fetch all PBI work items and save to data/raw/pbi/pbi_{id}.txt.
    Returns the number of PBIs saved.
    """
    validate(require_azure=True)
    ensure_dirs()
    org = get(AZURE_DEVOPS_ORG)
    project = get(AZURE_DEVOPS_PROJECT)
    base_url = f"https://dev.azure.com/{org}/{project}"
    session = requests.Session()
    ids = _wiql_query_ids(session, base_url)
    logger.info("Found %s PBI work items", len(ids))
    saved = 0
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        try:
            items = _get_work_items_batch(session, base_url, batch)
        except requests.RequestException as e:
            logger.warning("Batch request failed for ids %s: %s", batch[:5], e)
            continue
        for wi in items:
            wid = wi.get("id")
            if wid is None:
                continue
            text = _work_item_to_text(wi)
            out_path = DATA_RAW_PBI / f"pbi_{wid}.txt"
            out_path.write_text(text, encoding="utf-8")
            saved += 1
    logger.info("Saved %s PBI files to %s", saved, DATA_RAW_PBI)
    return saved


if __name__ == "__main__":
    try:
        fetch_work_items()
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)
