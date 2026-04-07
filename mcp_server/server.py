"""
Nexus MCP Server — exposes Nexus capabilities as MCP tools via FastMCP.

Run standalone:
    python mcp_server/server.py

Or mount inside a FastAPI app with mcp.get_asgi_app().
"""

from fastmcp import FastMCP

from config import settings

mcp = FastMCP("nexus")


# ---------------------------------------------------------------------------
# Chat tool
# ---------------------------------------------------------------------------

@mcp.tool()
def chat(message: str, session_id: str = "") -> dict:
    """
    Send a message to the Nexus orchestrator and get a response.

    Args:
        message:    The user's question or instruction.
        session_id: Optional session UUID for multi-turn conversations.
                    A new UUID is generated if not provided.

    Returns:
        dict with keys: answer, agent_called, retrieved_context,
                        web_results, session_id
    """
    from agents.orchestrator import run_chat
    return run_chat(message, session_id)


# ---------------------------------------------------------------------------
# Document tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_documents(query: str) -> dict:
    """
    Retrieve relevant document chunks from the Nexus knowledge base.

    Uses multi-step retrieval (query decomposition + bi-encoder + cross-encoder
    reranking + parent-chunk expansion).

    Args:
        query: The search query.

    Returns:
        dict with key 'chunks': list of dicts each containing
        content, source_file, and score.
    """
    from rag.retrieval import retrieve_multi_step
    chunks = retrieve_multi_step(query)
    return {"chunks": chunks}


@mcp.tool()
def ingest_document(file_path: str) -> dict:
    """
    Ingest a local file into the Nexus ChromaDB knowledge base.

    Supported file types: .pdf, .txt

    Args:
        file_path: Absolute path to the file on disk.

    Returns:
        dict with keys: file, parent_chunks, child_chunks, source_pages
    """
    from rag.ingestion import ingest_file
    return ingest_file(file_path)


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------

@mcp.tool()
def web_search(query: str) -> dict:
    """
    Search the web using Tavily and return relevant results.

    Requires TAVILY_API_KEY to be set in the environment / .env file.

    Args:
        query: The search query.

    Returns:
        dict with key 'results': list of result dicts (title, url, content),
        or 'error' key if Tavily is not configured.
    """
    if not settings.tavily_api_key:
        return {
            "error": (
                "Tavily API key is not configured. "
                "Set TAVILY_API_KEY in your .env file to enable web search."
            )
        }

    from tavily import TavilyClient
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query)
    results = response.get("results", [])
    return {"results": results}


# ---------------------------------------------------------------------------
# Email tool (stub — requires Google MCP integration)
# ---------------------------------------------------------------------------

@mcp.tool()
def read_emails(max_results: int = 10, query: str = "") -> dict:
    """
    Read emails from Gmail.

    Currently not configured — requires connecting a Google Workspace MCP server.

    TODO: Google MCP integration
        To implement this properly:
        1. Run the official Google Workspace MCP server
           (https://github.com/googleworkspace/google-workspace-mcp) as a
           subprocess or sidecar process.
        2. Open an MCP ClientSession to its stdio/SSE transport.
        3. Call the 'gmail_get_emails' (or equivalent) tool via
           session.call_tool(...) and forward the result.
        4. Handle OAuth2 token refresh and credential storage.

    Args:
        max_results: Maximum number of emails to return (default 10).
        query:       Optional Gmail search query (e.g., 'from:boss@example.com').

    Returns:
        dict with status 'not_configured' until Google MCP is connected.
    """
    # Stub implementation
    return {
        "status": "not_configured",
        "message": "Connect Google MCP server to enable this tool",
    }


# ---------------------------------------------------------------------------
# Calendar tool (stub — requires Google MCP integration)
# ---------------------------------------------------------------------------

@mcp.tool()
def read_calendar(start_date: str, end_date: str) -> dict:
    """
    Read Google Calendar events between two dates.

    Currently not configured — requires connecting a Google Workspace MCP server.

    TODO: Google MCP integration
        To implement this properly:
        1. Run the official Google Workspace MCP server
           (https://github.com/googleworkspace/google-workspace-mcp) as a
           subprocess or sidecar process.
        2. Open an MCP ClientSession to its stdio/SSE transport.
        3. Call the 'calendar_list_events' (or equivalent) tool via
           session.call_tool(...) with the provided date range and forward
           the result.
        4. Handle OAuth2 token refresh and credential storage.

    Args:
        start_date: ISO-8601 date string, e.g. '2026-04-01'.
        end_date:   ISO-8601 date string, e.g. '2026-04-30'.

    Returns:
        dict with status 'not_configured' until Google MCP is connected.
    """
    # Stub implementation
    return {
        "status": "not_configured",
        "message": "Connect Google MCP server to enable this tool",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
