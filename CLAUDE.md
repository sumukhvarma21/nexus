# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

General-purpose multi-agent system built with Python · FastAPI · LangGraph · Gemini Flash 2.5 · ChromaDB → Qdrant · FastMCP. Local-first, AWS Serverless at scale.

See `documents/` for the phased build plan. Each phase has its own document.

## Commands

```bash
# Setup
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn api.routes:app --reload

# Run tests
pytest tests/

# Run a single test
pytest tests/test_rag.py::test_ingest -v
```

## Architecture

```
User (UI / MCP Client)
        ↓
   MCP Server (mcp/server.py)        ← FastMCP, exposes all tools
        ↓
 FastAPI Backend (api/routes.py)     ← HTTP layer, session management
        ↓
 LangGraph Orchestrator (agents/orchestrator.py)   ← supervisor routing
    ├── RAG Agent (agents/rag_agent.py)
    ├── Web Search Agent (agents/web_search_agent.py)
    └── Email/Cal Agent (agents/email_calendar_agent.py)
        ↓
   Gemini Flash 2.5                  ← LLM for all agents + routing
        ↓
 ChromaDB (local dev) / Qdrant (prod)
```

**Routing:** Supervisor uses Gemini structured output `{"agent": "rag_agent"}` to dispatch. All agents return to a synthesizer node before responding.

**RAG pipeline:** Two-stage retrieval — bi-encoder (BAAI/bge-base-en-v1.5) does ANN search (top-20), cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) reranks to top-5. Parent-child chunking: child=256 tokens retrieved, parent=1024 tokens sent to LLM.

**Memory:** Short-term = LangGraph state with summarization buffer. Long-term = separate ChromaDB collection (retrieval-based) + SQLite for structured user facts. Session ID (UUID) ties them together.

**MCP vs REST:** MCP (`/mcp/server.py`) is the primary interface for LLM clients and Claude Desktop. REST (`/api/routes.py`) is the HTTP layer the UI and MCP server call.

## Key Config

- `config.py` — Pydantic Settings, switches between local and AWS automatically
- `.env` — `GOOGLE_API_KEY`, `LANGCHAIN_API_KEY` (LangSmith), `TAVILY_API_KEY`
- Phase 7: ChromaDB → Qdrant migration, Lambda + ECS Fargate split (Lambda for short calls, ECS for long LangGraph workflows)

## Phase Documents

| File | Content |
|------|---------|
| `documents/phase-0.md` | Foundations & scaffolding |
| `documents/phase-1.md` | RAG pipeline (bi-encoder + reranking) |
| `documents/phase-2.md` | Multi-step RAG (decomposition, HyDE, iterative) |
| `documents/phase-3.md` | LangGraph orchestrator + agent nodes |
| `documents/phase-4.md` | Short-term + long-term memory |
| `documents/phase-5.md` | MCP server + Email/Calendar agent |
| `documents/phase-6.md` | Chainlit UI |
| `documents/phase-7.md` | AWS serverless deployment |
