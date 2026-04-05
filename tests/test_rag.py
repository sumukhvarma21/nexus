import pytest
from unittest.mock import patch, MagicMock
from rag.retrieval import _merge_chunks
from rag.query_processor import decompose_query, generate_hypothetical_document


# --- Pure logic tests (no LLM/ChromaDB needed) ---

def test_merge_chunks_deduplicates():
    chunk = {"content": "a" * 201, "source_file": "doc.pdf", "score": 0.9}
    result = _merge_chunks([[chunk], [chunk]])
    assert len(result) == 1


def test_merge_chunks_sorts_by_score():
    low = {"content": "aaa", "source_file": "a.pdf", "score": 0.3}
    high = {"content": "bbb", "source_file": "b.pdf", "score": 0.9}
    result = _merge_chunks([[low, high]])
    assert result[0]["score"] == 0.9


# --- Integration tests (hit real LLM, need .env) ---

@pytest.mark.integration
def test_decompose_compound_query():
    subs = decompose_query(
        "What is the payment schedule and what happens if payments are late?"
    )
    assert len(subs) >= 2


@pytest.mark.integration
def test_hyde_returns_string():
    doc = generate_hypothetical_document("What is a neural network?")
    assert isinstance(doc, str) and len(doc) > 50
