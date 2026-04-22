"""
Fetch Azure DevOps Work Items (PBI, Bug, Task, Feature, Epic) and save as
text under data/raw/pbi/.
Uses PAT authentication. Run from project root so config is importable.
"""
import base64
import json
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


WORK_ITEM_TYPES = [
    "Product Backlog Item",
    "Bug",
    "Task",
]

WIQL_LIMIT = 20000
OLDEST_YEAR = 2016


def _wiql_run(session: requests.Session, url: str, query: str) -> list[int] | None:
    """Execute a WIQL query and return IDs, or None on failure."""
    body = {"query": query}
    r = session.post(url, headers=_auth_headers(), json=body, timeout=30)
    if r.status_code != 200:
        logger.warning("WIQL failed — status: %s, response: %s", r.status_code, r.text)
        return None
    data = r.json()
    return [int(row["id"]) for row in (data.get("workItems") or []) if row.get("id") is not None]


def _wiql_query_ids(session: requests.Session, base_url: str) -> list[int]:
    """
    Fetch all work item IDs, splitting by year when a type exceeds the
    20,000 WIQL result limit.  Excludes items with State = 'Removed'.
    """
    url = f"{base_url}/_apis/wit/wiql?api-version={API_VERSION}&$top={WIQL_LIMIT}"
    all_ids: list[int] = []

    for wi_type in WORK_ITEM_TYPES:
        logger.info("Fetching %s IDs...", wi_type)
        query = (
            "SELECT [System.Id] FROM WorkItems "
            f"WHERE [System.WorkItemType] = '{wi_type}' "
            "AND [System.State] <> 'Removed' "
            "ORDER BY [System.ChangedDate] DESC"
        )
        ids = _wiql_run(session, url, query)

        if ids is not None and len(ids) < WIQL_LIMIT:
            logger.info("  Found %s %s items", len(ids), wi_type)
            all_ids.extend(ids)
            continue

        logger.info("  %s exceeded limit or query failed — splitting by year...", wi_type)
        type_ids: set[int] = set()
        current_year = 2026
        for year in range(current_year, OLDEST_YEAR - 1, -1):
            year_query = (
                "SELECT [System.Id] FROM WorkItems "
                f"WHERE [System.WorkItemType] = '{wi_type}' "
                "AND [System.State] <> 'Removed' "
                f"AND [System.CreatedDate] >= '{year}-01-01' "
                f"AND [System.CreatedDate] < '{year + 1}-01-01' "
                "ORDER BY [System.ChangedDate] DESC"
            )
            year_ids = _wiql_run(session, url, year_query)
            if year_ids is None:
                continue
            if len(year_ids) >= WIQL_LIMIT:
                # Split year into two halves
                for half_start, half_end in [(f"{year}-01-01", f"{year}-07-01"), (f"{year}-07-01", f"{year + 1}-01-01")]:
                    half_query = (
                        "SELECT [System.Id] FROM WorkItems "
                        f"WHERE [System.WorkItemType] = '{wi_type}' "
                        "AND [System.State] <> 'Removed' "
                        f"AND [System.CreatedDate] >= '{half_start}' "
                        f"AND [System.CreatedDate] < '{half_end}' "
                        "ORDER BY [System.ChangedDate] DESC"
                    )
                    half_ids = _wiql_run(session, url, half_query)
                    if half_ids:
                        type_ids.update(half_ids)
                        logger.info("    %s (%s to %s): %s items", wi_type, half_start, half_end, len(half_ids))
            else:
                type_ids.update(year_ids)
                if year_ids:
                    logger.info("    %s %s: %s items", wi_type, year, len(year_ids))

        logger.info("  Total %s: %s items", wi_type, len(type_ids))
        all_ids.extend(type_ids)

    return all_ids


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


def _extract_person(field_value) -> str:
    """Extract display name from an Azure DevOps identity field (dict or string)."""
    if isinstance(field_value, dict):
        return field_value.get("displayName", "")
    return str(field_value) if field_value else ""


def _work_item_to_text(wi: dict) -> str:
    """Build a single text representation of a work item."""
    fields = wi.get("fields") or {}
    title = fields.get("System.Title") or fields.get("System.Id") or "Untitled"
    wi_type = fields.get("System.WorkItemType") or ""
    state = fields.get("System.State") or ""
    iteration = fields.get("System.IterationPath") or ""
    created_by = _extract_person(fields.get("System.CreatedBy"))
    assigned_to = _extract_person(fields.get("System.AssignedTo"))
    description = fields.get("System.Description") or ""
    acceptance = (
        fields.get("Microsoft.VSTS.Common.AcceptanceCriteria")
        or fields.get("Acceptance Criteria")
        or ""
    )
    parts = [f"Title: {title}"]
    if wi_type:
        parts.append(f"Type: {wi_type}")
    parts.append(f"State: {state}")
    if iteration:
        parts.append(f"Iteration: {iteration}")
    if created_by:
        parts.append(f"Created By: {created_by}")
    if assigned_to:
        parts.append(f"Assigned To: {assigned_to}")
    if description:
        parts.append(f"Description:\n{description}")
    if acceptance:
        parts.append(f"Acceptance Criteria:\n{acceptance}")
    return "\n\n".join(parts)


def _work_item_to_index_entry(wi: dict) -> dict:
    """Build a structured metadata dict for the work item index."""
    fields = wi.get("fields") or {}
    return {
        "id": wi.get("id"),
        "title": fields.get("System.Title") or "Untitled",
        "work_item_type": fields.get("System.WorkItemType") or "",
        "state": fields.get("System.State") or "",
        "iteration": fields.get("System.IterationPath") or "",
        "created_by": _extract_person(fields.get("System.CreatedBy")),
        "assigned_to": _extract_person(fields.get("System.AssignedTo")),
    }


def fetch_work_items() -> int:
    """
    Fetch all work items (PBI, Bug, Task, Feature, Epic) and save to
    data/raw/pbi/pbi_{id}.txt.  Also writes data/raw/pbi/pbi_index.json
    with structured metadata for aggregate queries.
    Returns the number of work items saved.
    """
    validate(require_azure=True)
    ensure_dirs()
    org = get(AZURE_DEVOPS_ORG)
    project = get(AZURE_DEVOPS_PROJECT)
    base_url = f"https://dev.azure.com/{org}/{project}"
    session = requests.Session()
    ids = _wiql_query_ids(session, base_url)
    logger.info("Found %s work items", len(ids))
    saved = 0
    index_entries: list[dict] = []
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
            index_entries.append(_work_item_to_index_entry(wi))
            saved += 1
    index_path = DATA_RAW_PBI / "pbi_index.json"
    index_path.write_text(json.dumps(index_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %s work item files + pbi_index.json to %s", saved, DATA_RAW_PBI)
    return saved


if __name__ == "__main__":
    try:
        fetch_work_items()
    except Exception as e:
        logger.exception("%s", e)
        sys.exit(1)
