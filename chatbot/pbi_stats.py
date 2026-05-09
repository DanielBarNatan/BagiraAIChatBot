"""
Structured work item statistics from pbi_index.json.
Provides aggregate queries (counts by state, type, iteration, person, date) that RAG cannot answer.
"""
import json
from collections import Counter
from datetime import datetime, date, timedelta
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


def _parse_date(s: str) -> date | None:
    """Parse a YYYY-MM-DD (or longer ISO) string into a date object."""
    try:
        return datetime.fromisoformat(s[:10]).date()
    except (ValueError, TypeError):
        return None


def _filter_by_date_range(
    items: list[dict], date_from: str = "", date_to: str = ""
) -> list[dict]:
    """Filter items whose created_date falls within [date_from, date_to] inclusive."""
    d_from = _parse_date(date_from) if date_from else None
    d_to = _parse_date(date_to) if date_to else None
    if not d_from and not d_to:
        return items
    result = []
    for item in items:
        d = _parse_date(item.get("created_date") or "")
        if d is None:
            continue
        if d_from and d < d_from:
            continue
        if d_to and d > d_to:
            continue
        result.append(item)
    return result


def items_by_field(field: str, value: str) -> list[dict]:
    """Return PBI items where *field* matches *value* (case-insensitive).
    For created_date, supports prefix matching (e.g. '2026', '2026-03', '2026-03-15')."""
    index = load_index()
    value_lower = value.lower()
    if field == "created_date":
        return [item for item in index if (item.get("created_date") or "").startswith(value)]
    return [item for item in index if (item.get(field) or "").lower() == value_lower]


def _item_matches_filters(item: dict, filters: dict[str, str]) -> bool:
    """Check if a work item matches all filters (AND logic).
    Skips date_from/date_to keys (handled separately by _filter_by_date_range)."""
    for field, value in filters.items():
        if field in ("date_from", "date_to"):
            continue
        item_val = item.get(field) or ""
        if field == "created_date":
            if not item_val.startswith(value):
                return False
        else:
            if item_val.lower() != value.lower():
                return False
    return True


def items_by_filters(
    filters: dict[str, str],
    date_from: str = "",
    date_to: str = "",
) -> list[dict]:
    """Return work items matching ALL the given field-value filters.
    Also applies date_from/date_to range if provided (either as params or inside filters)."""
    df = date_from or filters.get("date_from", "")
    dt = date_to or filters.get("date_to", "")
    index = load_index()
    items = [item for item in index if _item_matches_filters(item, filters)]
    if df or dt:
        items = _filter_by_date_range(items, df, dt)
    return items


def count_by_field_with_filters(
    group_by: str, filters: dict[str, str],
    date_from: str = "", date_to: str = "",
) -> dict[str, int]:
    """Count items grouped by *group_by*, after applying multi-field filters + date range."""
    filtered = items_by_filters(filters, date_from, date_to) if filters else load_index()
    if not filters and (date_from or date_to):
        filtered = _filter_by_date_range(filtered, date_from, date_to)
    if group_by == "created_date":
        counter = Counter(
            (item.get("created_date") or "Unknown")[:7]
            for item in filtered
        )
        return dict(sorted(counter.items()))
    counter = Counter(item.get(group_by, "") or "Unknown" for item in filtered)
    return dict(counter.most_common())


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


def _sort_and_limit(items: list[dict], sort_by: str = "",
                    sort_order: str = "desc", limit: int = 0) -> list[dict]:
    """Sort items by a field and optionally limit the result count."""
    if sort_by:
        reverse = sort_order.lower() != "asc"
        items = sorted(items, key=lambda x: x.get(sort_by) or "", reverse=reverse)
    if limit > 0:
        items = items[:limit]
    return items


def _format_items(items: list[dict], filter_desc: str) -> str:
    """Format a list of work items into a human-readable string."""
    if not items:
        return f"No work items found matching {filter_desc}."
    lines = [f"Found {len(items)} work items matching {filter_desc}:"]
    for item in items:
        wi_type = item.get('work_item_type', 'PBI')
        created = item.get('created_date', '')
        date_part = f", Created: {created}" if created else ""
        lines.append(f"  [{wi_type}] #{item['id']}: {item['title']} (State: {item['state']}{date_part})")
    return "\n".join(lines)


def query(action: str, field: str = "", value: str = "",
          filters: dict[str, str] | None = None,
          group_by: str = "", sort_by: str = "",
          sort_order: str = "desc", limit: int = 0,
          date_from: str = "", date_to: str = "") -> str:
    """
    Single entry-point used by the chat engine tool-calling integration.
    Actions: "summary", "count_by_field", "items_by_field",
             "filter_items", "filter_count".
    sort_by, sort_order, and limit apply to items_by_field and filter_items.
    date_from / date_to (YYYY-MM-DD) filter created_date to a range.
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
        if date_from or date_to:
            items = _filter_by_date_range(items, date_from, date_to)
        items = _sort_and_limit(items, sort_by, sort_order, limit)
        desc = f"{field} = '{value}'"
        if date_from or date_to:
            desc += f" (date range: {date_from or '...'} to {date_to or '...'})"
        return _format_items(items, desc)
    if action == "filter_items":
        if not filters and not date_from and not date_to:
            return "Error: 'filters' dict or date_from/date_to is required for filter_items."
        items = items_by_filters(filters or {}, date_from, date_to)
        items = _sort_and_limit(items, sort_by, sort_order, limit)
        parts = [f"{k} = '{v}'" for k, v in (filters or {}).items() if k not in ("date_from", "date_to")]
        if date_from or date_to:
            parts.append(f"date range: {date_from or '...'} to {date_to or '...'}")
        desc = ", ".join(parts) if parts else "all items"
        return _format_items(items, desc)
    if action == "filter_count":
        if not filters and not date_from and not date_to:
            return "Error: 'filters' dict or date_from/date_to is required for filter_count."
        gb = group_by or ""
        if gb:
            counts = count_by_field_with_filters(gb, filters or {}, date_from, date_to)
            parts = [f"{k} = '{v}'" for k, v in (filters or {}).items() if k not in ("date_from", "date_to")]
            if date_from or date_to:
                parts.append(f"date range: {date_from or '...'} to {date_to or '...'}")
            desc = ", ".join(parts) if parts else "all items"
            lines = [f"Counts grouped by '{gb}' where {desc}:"]
            total = sum(counts.values())
            for k, v in counts.items():
                lines.append(f"  {k}: {v}")
            lines.append(f"  Total: {total}")
            return "\n".join(lines)
        items = items_by_filters(filters or {}, date_from, date_to)
        parts = [f"{k} = '{v}'" for k, v in (filters or {}).items() if k not in ("date_from", "date_to")]
        if date_from or date_to:
            parts.append(f"date range: {date_from or '...'} to {date_to or '...'}")
        desc = ", ".join(parts) if parts else "all items"
        return f"Count of work items where {desc}: {len(items)}"
    return f"Unknown action: {action}. Use summary, count_by_field, items_by_field, filter_items, or filter_count."
