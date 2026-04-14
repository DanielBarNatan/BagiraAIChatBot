"""Unit tests for prompt builder."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot.prompt_builder import build_prompt


def test_build_prompt_includes_instruction_and_context():
    docs = ["Context A.", "Context B."]
    question = "What is X?"
    prompt = build_prompt(docs, question)
    assert "ONLY the context" in prompt or "only the context" in prompt
    assert "Context A." in prompt
    assert "Context B." in prompt
    assert "What is X?" in prompt


def test_build_prompt_empty_documents():
    prompt = build_prompt([], "Question?")
    assert "Question?" in prompt
