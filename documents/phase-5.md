# Phase 5 — MCP Server + Email/Calendar Agent

**Goal:** Expose the entire system via MCP. Add Gmail and GCal tools. MCP is the standard interface for all clients.

**Depends on:** [Phase 4](phase-4.md)

## What MCP Is

MCP (Model Context Protocol) is an open standard for exposing tools and resources to LLMs. USB-C for AI tools — any MCP-compatible client can plug in.

```
MCP Client (UI / Claude Desktop / another agent)
        ↓  JSON-RPC over SSE
MCP Server (FastMCP)
        ↓
FastAPI Backend → LangGraph Orchestrator → Agents
```

| | MCP | REST API |
|---|---|---|
| Tool discovery | Client auto-discovers tools + schemas | Manual docs |
| LLM-native | Designed for LLM tool use | General purpose |
| Streaming | First-class SSE | Needs custom SSE/WS |

## Gmail/GCal Integration

| Option | Effort | Approach |
|---|---|---|
| Direct Gmail API (OAuth2) | High | Write integration yourself |
| **Existing Google MCP servers** ✓ | Low | Connect your MCP to Google's MCP |
| LangChain Gmail toolkit | Medium | Wrapper over Gmail API |

**Chosen:** Google publishes official MCP servers for Gmail and GCal. Email agent calls your MCP server, which connects to Google's MCP. No raw OAuth2 to maintain.

## MCP Tools to Expose

```
chat(message, session_id)             → full orchestrator response (streaming)
query_documents(query, doc_filter?)   → RAG results
ingest_document(file_path)            → ingestion status
web_search(query)                     → search results
read_emails(max_results, query?)      → email list + summaries
read_calendar(start_date, end_date)   → events list
list_sessions()                       → active session IDs
```

## Deliverables

- [ ] `mcp/server.py` — FastMCP server exposing all tools above
- [ ] `agents/email_calendar_agent.py` — LangGraph node using Gmail + GCal MCP tools
- [ ] Orchestrator updated to route email and calendar intent
- [ ] Test with Claude Desktop as MCP client
- [ ] End-to-end: "What meetings do I have tomorrow and summarize related emails?"
- [ ] Streaming responses working through MCP

## Packages

```
fastmcp
mcp
```

## Next

→ [Phase 6: UI](phase-6.md)
