"""
Orchestrates retrieval, prompt building, and LLM call to produce a grounded answer.
Supports both RAG-based knowledge search and structured PBI statistics via tool calling.
"""
import datetime
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import RETRIEVER_TOP_K
from chatbot.retriever import retrieve
from chatbot.prompt_builder import build_prompt, build_stats_context


def _get_llm():
    from config.settings import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_FAST_MODEL, get
    from openai import OpenAI
    client = OpenAI(api_key=get(OPENAI_API_KEY))
    return client, OPENAI_CHAT_MODEL, OPENAI_FAST_MODEL


def _deduplicate_sources(chunks: list[tuple[str, dict]]) -> list[dict]:
    """Return unique source metadata dicts, keyed by (document_type, source_document)."""
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for _text, meta in chunks:
        key = (meta.get("document_type", ""), meta.get("source_document", ""))
        if key not in seen:
            seen.add(key)
            sources.append(meta)
    return sources


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "work_item_statistics",
            "description": (
                "Query structured Azure DevOps work item data for aggregate or "
                "statistical questions. Covers PBIs, Bugs, and Tasks. "
                "Use this for questions about counts, totals, breakdowns by state "
                "(Done, Active, Removed, New, etc.), by work item type, by sprint/"
                "iteration, by creation date, or questions about how many items "
                "a specific person created or is assigned to. "
                "Supports multi-field filtering via 'filter_items' and 'filter_count' "
                "actions — use these when the question combines multiple criteria "
                "(e.g. 'Bugs created by X that are Done')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["summary", "count_by_field", "items_by_field",
                                 "filter_items", "filter_count"],
                        "description": (
                            "summary: full overview of all work item stats. "
                            "count_by_field: count work items grouped by a single field. "
                            "items_by_field: list work items matching a single field value. "
                            "filter_items: list work items matching multiple field-value "
                            "filters simultaneously (AND logic). "
                            "filter_count: count work items matching multiple filters, "
                            "optionally grouped by a field via 'group_by'."
                        ),
                    },
                    "field": {
                        "type": "string",
                        "enum": ["state", "work_item_type", "iteration", "created_by", "assigned_to", "created_date"],
                        "description": "The field to query on. Required for count_by_field and items_by_field.",
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "The value to match. Required for items_by_field. "
                            "Examples: a state like 'Done', a type like 'Bug', "
                            "an iteration like 'ProjectName\\Sprint 24', a person's name, "
                            "or a date/prefix like '2026', '2026-03', or '2026-03-15'."
                        ),
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Multi-field filter for filter_items / filter_count. "
                            "Keys are field names (state, work_item_type, iteration, "
                            "created_by, assigned_to, created_date), values are the "
                            "values to match. All conditions are ANDed together. "
                            "Example: {\"created_by\": \"Daniel\", \"state\": \"Done\", "
                            "\"work_item_type\": \"Bug\"}"
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["state", "work_item_type", "iteration", "created_by", "assigned_to", "created_date"],
                        "description": (
                            "Optional. For filter_count only: group the filtered results "
                            "by this field and return counts per group."
                        ),
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["created_date", "state", "work_item_type", "iteration", "title"],
                        "description": (
                            "Optional. Sort results by this field. Works with "
                            "items_by_field and filter_items. "
                            "Use 'created_date' to find the most recent or oldest items."
                        ),
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": (
                            "Optional. Sort direction: 'desc' (newest/largest first, default) "
                            "or 'asc' (oldest/smallest first)."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Optional. Maximum number of items to return. "
                            "Use with sort_by to get e.g. the 5 most recent items. "
                            "0 or omitted means return all."
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "description": (
                            "Optional. Start of date range (YYYY-MM-DD, inclusive). "
                            "Filters work items whose created_date >= this date. "
                            "Use for questions like 'last week', 'last 2 weeks', 'since March'. "
                            "Today's date is available in the system message."
                        ),
                    },
                    "date_to": {
                        "type": "string",
                        "description": (
                            "Optional. End of date range (YYYY-MM-DD, inclusive). "
                            "Filters work items whose created_date <= this date. "
                            "Defaults to today if omitted and date_from is set."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": (
                "Search the knowledge base (Wiki pages and work item descriptions) for "
                "detailed information. Use this for questions about system architecture, "
                "features, processes, or any question that needs descriptive content "
                "rather than aggregate numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant documents.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _execute_tool_call(name: str, arguments: dict) -> tuple[str, list[dict]]:
    """
    Execute a tool call and return (result_text, sources).
    Sources are only populated for knowledge_search.
    """
    if name == "work_item_statistics":
        from chatbot.pbi_stats import query as pbi_query
        result = pbi_query(
            action=arguments.get("action", "summary"),
            field=arguments.get("field", ""),
            value=arguments.get("value", ""),
            filters=arguments.get("filters"),
            group_by=arguments.get("group_by", ""),
            date_from=arguments.get("date_from", ""),
            date_to=arguments.get("date_to", ""),
            sort_by=arguments.get("sort_by", ""),
            sort_order=arguments.get("sort_order", "desc"),
            limit=arguments.get("limit", 0),
        )
        return result, []

    if name == "knowledge_search":
        query_text = arguments.get("query", "")
        chunks = retrieve(query_text, top_k=RETRIEVER_TOP_K)
        if not chunks:
            return "No relevant documents found in the knowledge base.", []
        context = build_prompt(chunks, query_text)
        sources = _deduplicate_sources(chunks)
        return context, sources

    return f"Unknown tool: {name}", []


MAX_HISTORY_PAIRS = 5


def _build_system_messages(history: list[dict] | None, question: str) -> list[dict]:
    """Build the messages list with system prompt, history, and user question."""
    messages = [
        {
            "role": "system",
            "content": (
                f"Today's date is {datetime.date.today().isoformat()}.\n\n"
                "You are a DevOps AI Knowledge Assistant. Provide thorough, accurate answers "
                "grounded in the team's documentation and work item data.\n\n"

                "## Tools\n"
                "1. **work_item_statistics** — counts, listings, and breakdowns of PBIs/Bugs/Tasks.\n"
                "2. **knowledge_search** — wiki pages, system architecture, feature docs.\n\n"

                "## Examples\n"
                "- 'How many bugs are open?' → count_by_field, field=state\n"
                "- 'How many items did Daniel create?' → filter_count, "
                "filters={\"created_by\":\"Daniel\"}, group_by=work_item_type\n"
                "- 'Done bugs Daniel created' → filter_count, "
                "filters={\"created_by\":\"Daniel\",\"state\":\"Done\",\"work_item_type\":\"Bug\"}\n"
                "- 'Break down Daniel's items by state' → filter_count, "
                "filters={\"created_by\":\"Daniel\"}, group_by=state\n"
                "- 'What items did Daniel create last month?' → filter_items, "
                "filters={\"created_by\":\"Daniel\"}, date_from/date_to as YYYY-MM-DD. "
                "ALWAYS compute exact dates from today. Use filter_items when the user "
                "wants to SEE items; filter_count only for 'how many'.\n"
                "- 'Last item Daniel created' → items_by_field, field=created_by, "
                "value=Daniel, sort_by=created_date, sort_order=desc, limit=1\n"
                "- 'How does authentication work?' → knowledge_search\n"
                "- Both stats + details needed → call BOTH tools\n"
                "IMPORTANT: When a question mentions a specific person, ALWAYS use "
                "filter_count or filter_items with filters={\"created_by\":\"Name\"} — "
                "NEVER use count_by_field, which counts ALL items ignoring the person.\n\n"

                "## Rules\n"
                "- Call tools immediately; never promise to look something up.\n"
                "- When user asks to see/list/show items, use filter_items (NOT filter_count).\n"
                "- Display rule for work items: <=5 per type → list all with ID, title, state, "
                "date. >5 per type → show count + 3 examples. Group by type. Never drop items.\n"
                "- Cite source documents/wiki pages. Use conversation history for follow-ups.\n"
                "- If context is insufficient, say so honestly.\n\n"

                "## Language\n"
                "Reply in the user's language. Keep technical terms and titles in "
                "their original language."
            ),
        },
    ]

    if history:
        recent = history[-(MAX_HISTORY_PAIRS * 2):]
        for msg in recent:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})
    return messages


def answer(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """
    Answer a user question using tool calling. The LLM decides whether to use
    RAG knowledge search, PBI statistics, or both.
    Accepts optional conversation history (list of {"role", "content"} dicts)
    to support follow-up questions. Only the last MAX_HISTORY_PAIRS exchanges
    are included to control token usage.
    Returns (reply_text, deduplicated_sources).
    """
    client, model, fast_model = _get_llm()

    messages = _build_system_messages(history, question)

    response = client.chat.completions.create(
        model=fast_model,
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )
    message = response.choices[0].message
    all_sources: list[dict] = []

    if not message.tool_calls:
        # No tool call — fast model answered directly; re-ask with full model
        # for higher-quality direct answers.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        reply = (response.choices[0].message.content or "").strip()
        return reply if reply else "I'm not sure how to answer that.", all_sources

    messages.append(message)

    for tool_call in message.tool_calls:
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        result_text, sources = _execute_tool_call(tool_call.function.name, args)
        all_sources.extend(sources)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    reply = (final_response.choices[0].message.content or "").strip()
    return reply, all_sources


def answer_stream(question: str, history: list[dict] | None = None):
    """
    Streaming variant of answer(). Yields (chunk_text, None) for each token,
    then yields ("", sources) as the final item with the deduplicated sources.
    The tool-decision and retrieval steps run non-streaming (fast);
    only the final answer generation is streamed.
    """
    client, model, fast_model = _get_llm()

    messages = _build_system_messages(history, question)

    response = client.chat.completions.create(
        model=fast_model,
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )
    message = response.choices[0].message
    all_sources: list[dict] = []

    if not message.tool_calls:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content, None
        yield "", all_sources
        return

    messages.append(message)

    for tool_call in message.tool_calls:
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        result_text, sources = _execute_tool_call(tool_call.function.name, args)
        all_sources.extend(sources)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_text,
        })

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content, None
    yield "", all_sources
