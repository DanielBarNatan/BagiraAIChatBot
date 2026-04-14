"""
DevOps AI Knowledge Assistant — CLI chat loop.
Load .env, validate config, then prompt for questions and print grounded answers.
"""
from pathlib import Path
import sys

# Load environment before importing config
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from urllib.parse import quote

from config.settings import (
    validate,
    SHOW_SOURCES,
    AZURE_DEVOPS_ORG,
    AZURE_DEVOPS_PROJECT,
    AZURE_DEVOPS_WIKI_ID,
    get,
)

def main():
    # Validate required env vars (OpenAI required; Azure only for ingestion scripts)
    try:
        validate(require_azure=False)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    print("DevOps AI Knowledge Assistant")
    print("Ask questions about your system (Wiki + PBI). Type 'quit' or press Enter to exit.\n")

    try:
        from chatbot.chat_engine import answer
    except FileNotFoundError as e:
        print(f"Setup required: {e}")
        print("Run: python scripts/generate_embeddings.py")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() == "quit":
            print("Goodbye.")
            break
        print("Searching...")
        try:
            reply, sources = answer(user_input)
            print(f"Assistant: {reply}\n")
            if SHOW_SOURCES and sources:
                org = get(AZURE_DEVOPS_ORG)
                project = get(AZURE_DEVOPS_PROJECT)
                wiki_id = get(AZURE_DEVOPS_WIKI_ID)
                base = f"https://dev.azure.com/{quote(org, safe='')}/{quote(project, safe='')}" if org and project else ""
                print("  Sources:")
                for src in sources:
                    doc_type = src.get("document_type", "unknown")
                    if doc_type == "wiki":
                        title = src.get("page_title", "untitled")
                        wiki_path = src.get("wiki_path", "")
                        if base and wiki_id and wiki_path:
                            url = f"{base}/_wiki/wikis/{quote(wiki_id, safe='')}?pagePath={quote(wiki_path, safe='/')}"
                            print(f"    [wiki] {title}\n           {url}")
                        else:
                            print(f"    [wiki] {title}")
                    elif doc_type == "pbi":
                        wid = src.get("work_item_id", "unknown")
                        if base and wid != "unknown":
                            url = f"{base}/_workitems/edit/{wid}"
                            print(f"    [pbi]  PBI {wid}\n           {url}")
                        else:
                            print(f"    [pbi]  PBI {wid}")
                    else:
                        source_doc = src.get("source_document", "unknown")
                        print(f"    [{doc_type}] {source_doc}")
                print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
