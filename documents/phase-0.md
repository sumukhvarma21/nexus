# Phase 0 — Foundations & Project Scaffolding

**Goal:** Clean repo, working skeleton, environment setup. Understand the full architecture before writing any business logic.

## Architecture Overview

```
User (UI / MCP Client)
        ↓
   MCP Server              ← exposes the whole system as tools
        ↓
 FastAPI Backend           ← HTTP layer, session management
        ↓
 LangGraph Orchestrator    ← supervisor node, routes to sub-agents
    ├── RAG Agent          ← retrieves from your vector store
    ├── Web Search Agent   ← searches the web
    └── Email/Cal Agent    ← reads Gmail, GCal via MCP
        ↓
   Gemini Flash 2.5        ← LLM backbone for all agents
        ↓
 ChromaDB (local) / Qdrant (prod)   ← vector store
```

## Why LangGraph

| | LangGraph | CrewAI | Custom Router |
|---|---|---|---|
| Control | Explicit graph, you own every edge | High abstraction, magic underneath | Full control |
| Debugging | Built-in state tracing | Hard to inspect | Depends on you |
| Production-readiness | High | Medium | High |

**Chosen:** Explicit state machines map cleanly to multi-agent routing, first-class streaming + async for AWS Lambda.

## Why Gemini Flash 2.5

| Model | Context Window | Cost |
|---|---|---|
| Gemini Flash 2.5 | 1M tokens | Free tier generous |
| GPT-4o-mini | 128K tokens | Cheap |
| Claude Haiku | 200K tokens | Cheap |

**Chosen:** Free tier for learning, 1M context window for large-doc RAG.

## Deliverables

- [ ] Repo initialized with full folder structure
- [ ] `requirements.txt` with pinned versions
- [ ] `.env` + `config.py` with Pydantic Settings
- [ ] `docker-compose.yml` for ChromaDB local
- [ ] FastAPI app boots, `/health` returns 200

## Packages

```
fastapi
uvicorn[standard]
pydantic-settings
python-dotenv
httpx
```

## Next

→ [Phase 1: RAG Pipeline](phase-1.md)
