# Phase 6 — Simple UI (Chainlit)

**Goal:** Minimal functional chat UI. Not a design project — enough to demo and use daily.

**Depends on:** [Phase 5](phase-5.md)

## UI Options

| Option | Effort | Best for |
|---|---|---|
| **Chainlit** ✓ | Very low | LLM chat apps exactly like this |
| Gradio | Very low | ML demos |
| Streamlit | Low | Data apps |
| Next.js | High | Deployed product |

**Chosen:** Purpose-built for LLM chat. File upload, streaming, session management, source citation display — all out of the box.

**For AWS later:** Chainlit can be containerized to ECS, or replaced with Next.js hitting FastAPI. MCP and FastAPI layers stay unchanged.

## UI Features

- Chat interface with streaming token-by-token responses
- File upload → triggers `/ingest`
- Agent attribution per response ("Answered by: RAG Agent")
- Source chunk display — expandable citations
- Toggle panel: web search on/off, memory on/off
- Session display in sidebar

## Deliverables

- [ ] `ui/app.py` — Chainlit app wired to FastAPI backend
- [ ] Streaming responses rendering correctly
- [ ] File upload working end-to-end (upload → ingest → queryable)
- [ ] Agent attribution label per message
- [ ] Source chunks as expandable citations
- [ ] Settings toggle panel

## Packages

```
chainlit
```

## Next

→ [Phase 7: AWS Deployment](phase-7.md)
