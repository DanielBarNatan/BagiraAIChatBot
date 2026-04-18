"""
Orchestrates retrieval, prompt building, and LLM call to produce a grounded answer.
Supports both RAG-based knowledge search and structured PBI statistics via tool calling.
"""
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
    from config.settings import OPENAI_API_KEY, OPENAI_CHAT_MODEL, get
    from openai import OpenAI
    client = OpenAI(api_key=get(OPENAI_API_KEY))
    return client, OPENAI_CHAT_MODEL


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
                "statistical questions. Covers PBIs, Bugs, Tasks, Features, and Epics. "
                "Use this for questions about counts, totals, breakdowns by state "
                "(Done, Active, Removed, New, etc.), by work item type, by sprint/"
                "iteration, or questions about how many items a specific person "
                "created or is assigned to."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["summary", "count_by_field", "items_by_field"],
                        "description": (
                            "summary: full overview of all work item stats. "
                            "count_by_field: count work items grouped by a field. "
                            "items_by_field: list work items matching a specific field value."
                        ),
                    },
                    "field": {
                        "type": "string",
                        "enum": ["state", "work_item_type", "iteration", "created_by", "assigned_to"],
                        "description": "The field to query on. Required for count_by_field and items_by_field.",
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "The value to match. Required for items_by_field. "
                            "Examples: a state like 'Done', a type like 'Bug', "
                            "an iteration like 'ProjectName\\Sprint 24', or a person's name."
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


def answer(question: str) -> tuple[str, list[dict]]:
    """
    Answer a user question using tool calling. The LLM decides whether to use
    RAG knowledge search, PBI statistics, or both.
    Returns (reply_text, deduplicated_sources).
    """
    client, model = _get_llm()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a DevOps AI Knowledge Assistant. You have two tools:\n"
                "1. work_item_statistics — for aggregate/statistical questions about "
                "work items (PBIs, Bugs, Tasks, Features, Epics). Supports counts by "
                "state, by type, by sprint/iteration, by person, and listing items.\n"
                "2. knowledge_search — for detailed knowledge questions about system "
                "architecture, features, wiki content, or work item descriptions.\n"
                "Choose the appropriate tool based on the question. You may call both "
                "if the question needs both statistics and detailed information."
            ),
        },
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        temperature=0,
    )
    message = response.choices[0].message
    all_sources: list[dict] = []

    if not message.tool_calls:
        reply = (message.content or "").strip()
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
