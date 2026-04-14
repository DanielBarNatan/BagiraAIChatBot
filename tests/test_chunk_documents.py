"""Unit tests for chunking logic."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.chunk_documents as chunk_mod


def test_build_chunks_respects_size():
    # One long segment should be split into chunks of ~CHUNK_SIZE
    long_text = "A. " + ("word " * 400)  # ~2k chars
    chunks = chunk_mod._build_chunks(long_text, chunk_size=200, overlap=20)
    assert len(chunks) >= 2
    for c in chunks[:-1]:
        assert len(c) <= 220  # chunk_size + some margin


def test_build_chunks_short_text_single_chunk():
    short = "This is a short paragraph."
    chunks = chunk_mod._build_chunks(short, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert short in chunks[0]
