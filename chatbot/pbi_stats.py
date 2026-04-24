"""
Structured work item statistics from pbi_index.json.
Provides aggregate queries (counts by state, type, iteration, person, date) that RAG cannot answer.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from config.settings import DATA_RAW_PBI

_INDEX_PATH = DATA_RAW_PBI / "pbi_index.json"
_cache: list[dict] | None = None


def load_index() -> list[dict]:
    """Load and cache the PBI index. Raises FileNotFoundError if missing."""
    global _cache
    if _cache is not None:
        return _cache
    if not _INDEX_PATH.exists():
        raise FileNotFoundError(
            f"PBI index not found at {_INDEX_PATH}. "
            "Run: python scripts/fetch_work_items.py"
        )
    _cache = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    return _cache


def invalidate_cache() -> None:
    global _cache
    _cache = None


def count_by_field(field: str) -> dict[str, int]:
    """Return {value: count} for a given field."""
    index = load_index()
    if field == "created_date":
        counter = Counter(
            (item.get("created_date") or "Unknown")[:7]  # group by YYYY-MM
            for item in index
        )
        return dict(sorted(counter.items()))
    counter = Counter(item.get(field, "") or "Unknown" for item in index)
    return dict(counter.most_common())


def items_by_field(field: str, value: str) -> list[dict]:
    """Return PBI items where *field* matches *value* (case-insensitive).
    For created_date, supports prefix matching (e.g. '2026', '2026-03', '2026-03-15')."""
    index = load_index()
    value_lower = value.lower()
    if field == "created_date":
        return [item for item in index if (item.get("created_date") or "").startswith(value)]
    return [item for item in index if (item.get(field) or "").lower() == value_lower]


def summary() -> str:
    """Human-readable summary of all work item statistics."""
    index = load_index()
    total = len(index)
    lines = [f"Total Work Items: {total}"]

    type_counts = count_by_field("work_item_type")
    lines.append("\nBy Type:")
    for wi_type, count in type_counts.items():
        lines.append(f"  {wi_type}: {count}")

    state_counts = count_by_field("state")
    lines.append("\nBy State:")
    for state, count in state_counts.items():
        lines.append(f"  {state}: {count}")

    iteration_counts = count_by_field("iteration")
    top_iterations = dict(list(iteration_counts.items())[:10])
    lines.append(f"\nBy Iteration (top {len(top_iterations)}):")
    for iteration, count in top_iterations.items():
        lines.append(f"  {iteration}: {count}")

    creator_counts = count_by_field("created_by")
    lines.append("\nBy Created By:")
    for person, count in creator_counts.items():
        lines.append(f"  {person}: {count}")

    assignee_counts = count_by_field("assigned_to")
    lines.append("\nBy Assigned To:")
    for person, count in assignee_counts.items():
        lines.append(f"  {person}: {count}")

    return "\n".join(lines)


def query(action: str, field: str = "", value: str = "") -> str:
    """
    Single entry-point used by the chat engine tool-calling integration.
    Actions: "summary", "count_by_field", "items_by_field".
    """
    if action == "summary":
        return summary()
    if action == "count_by_field":
        if not field:
            return "Error: 'field' is required (state, work_item_type, iteration, created_by, or assigned_to)."
        counts = count_by_field(field)
        lines = [f"{k}: {v}" for k, v in counts.items()]
        return "\n".join(lines) if lines else "No data found."
    if action == "items_by_field":
        if not field or not value:
            return "Error: 'field' and 'value' are required."
        items = items_by_field(field, value)
        if not items:
            return f"No work items found where {field} = '{value}'."
        lines = [f"Found {len(items)} work items where {field} = '{value}':"]
        for item in items:
            wi_type = item.get('work_item_type', 'PBI')
            created = item.get('created_date', '')
            date_part = f", Created: {created}" if created else ""
            lines.append(f"  [{wi_type}] #{item['id']}: {item['title']} (State: {item['state']}{date_part})")
        return "\n".join(lines)
    return f"Unknown action: {action}. Use summary, count_by_field, or items_by_field."
