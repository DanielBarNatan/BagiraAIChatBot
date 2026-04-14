"""Unit tests for document cleaning logic."""
import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the cleaning logic (script adds path and defines clean_text)
import scripts.clean_documents as clean_mod


def test_clean_text_strips_html():
    html = "<p>Hello <b>world</b></p>"
    assert "Hello" in clean_mod.clean_text(html) and "world" in clean_mod.clean_text(html)
    assert "<" not in clean_mod.clean_text(html)


def test_clean_text_removes_markdown_images():
    text = "See here: ![alt](http://example.com/img.png) and more."
    result = clean_mod.clean_text(text)
    assert "![alt]" not in result
    assert "example.com" not in result or "img" not in result


def test_clean_text_normalizes_whitespace():
    text = "  foo   \n\n\n   bar  "
    result = clean_mod.clean_text(text)
    assert "foo" in result and "bar" in result
    assert "     " not in result  # no long space runs
