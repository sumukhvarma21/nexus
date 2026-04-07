"""
Tests for the Phase 5 MCP server tools.

All tests are unit tests — no real API calls are made.
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_tool(name: str):
    """Import a tool function by name from mcp_server.server."""
    import importlib
    mod = importlib.import_module("mcp.server")
    return getattr(mod, name)


# ---------------------------------------------------------------------------
# chat tool
# ---------------------------------------------------------------------------

def test_chat_tool_returns_answer():
    """chat() should forward to run_chat and return its dict unchanged."""
    fake_response = {
        "answer": "Paris is the capital of France.",
        "agent_called": "rag_agent",
        "retrieved_context": ["France is a country in Europe. Its capital is Paris."],
        "web_results": [],
        "session_id": "abc-123",
    }

    with patch("agents.orchestrator.run_chat", return_value=fake_response) as mock_run:
        from mcp_server.server import chat
        result = chat(message="What is the capital of France?", session_id="abc-123")

    mock_run.assert_called_once_with("What is the capital of France?", "abc-123")
    assert result["answer"] == "Paris is the capital of France."
    assert result["agent_called"] == "rag_agent"
    assert result["session_id"] == "abc-123"


# ---------------------------------------------------------------------------
# query_documents tool
# ---------------------------------------------------------------------------

def test_query_documents_returns_chunks():
    """query_documents() should wrap retrieve_multi_step results under 'chunks'."""
    fake_chunks = [
        {"content": "Chunk A text.", "source_file": "doc.pdf", "score": 0.95},
        {"content": "Chunk B text.", "source_file": "doc.pdf", "score": 0.80},
    ]

    with patch("rag.retrieval.retrieve_multi_step", return_value=fake_chunks):
        from mcp_server.server import query_documents
        result = query_documents(query="What is in the document?")

    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert result["chunks"][0]["content"] == "Chunk A text."
    assert result["chunks"][1]["score"] == 0.80


# ---------------------------------------------------------------------------
# ingest_document tool
# ---------------------------------------------------------------------------

def test_ingest_document_returns_status():
    """ingest_document() should return ingest_file's metadata dict."""
    fake_meta = {
        "file": "sample.txt",
        "parent_chunks": 3,
        "child_chunks": 12,
        "source_pages": 1,
    }
    test_file_path = "/Users/sumukh/Code/nexus/test_files/sample.txt"

    with patch("rag.ingestion.ingest_file", return_value=fake_meta) as mock_ingest:
        from mcp_server.server import ingest_document
        result = ingest_document(file_path=test_file_path)

    mock_ingest.assert_called_once_with(test_file_path)
    assert result["file"] == "sample.txt"
    assert result["parent_chunks"] == 3
    assert result["child_chunks"] == 12


# ---------------------------------------------------------------------------
# web_search tool
# ---------------------------------------------------------------------------

def test_web_search_no_api_key():
    """web_search() should return a graceful error when TAVILY_API_KEY is not set."""
    with patch("mcp.server.settings") as mock_settings:
        mock_settings.tavily_api_key = ""
        from mcp_server.server import web_search
        result = web_search(query="latest AI news")

    assert "error" in result
    assert "TAVILY_API_KEY" in result["error"]


def test_web_search_with_api_key():
    """web_search() should call TavilyClient.search when key is configured."""
    fake_results = [
        {"title": "AI News", "url": "https://example.com", "content": "AI is advancing."}
    ]
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": fake_results}

    # TavilyClient is imported locally inside web_search(), so patch via the tavily module
    with patch("mcp.server.settings") as mock_settings, \
         patch("tavily.TavilyClient", return_value=mock_client):
        mock_settings.tavily_api_key = "test-key-xyz"
        from mcp_server.server import web_search
        result = web_search(query="latest AI news")

    assert "results" in result
    assert result["results"][0]["title"] == "AI News"


# ---------------------------------------------------------------------------
# read_emails tool (stub)
# ---------------------------------------------------------------------------

def test_read_emails_stub():
    """read_emails() should return not_configured status until Google MCP is wired up."""
    from mcp_server.server import read_emails
    result = read_emails(max_results=5, query="from:boss@example.com")

    assert result["status"] == "not_configured"
    assert "message" in result
    assert "Google MCP" in result["message"] or "not_configured" in result["status"]


# ---------------------------------------------------------------------------
# read_calendar tool (stub)
# ---------------------------------------------------------------------------

def test_read_calendar_stub():
    """read_calendar() should return not_configured status until Google MCP is wired up."""
    from mcp_server.server import read_calendar
    result = read_calendar(start_date="2026-04-01", end_date="2026-04-30")

    assert result["status"] == "not_configured"
    assert "message" in result
    assert "Google MCP" in result["message"] or "not_configured" in result["status"]
