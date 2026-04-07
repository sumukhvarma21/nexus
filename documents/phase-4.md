# Phase 4 — Memory (Short-Term + Long-Term)

**Goal:** Agents remember conversation context within a session (short-term) and persist useful knowledge across sessions (long-term).

**Depends on:** [Phase 3](phase-3.md)

## Memory Types

```
┌──────────────┬──────────────────────────────────────────────┐
│ Short-term   │ Current conversation context                  │
│ (In-context) │ Lives in LangGraph state as message list      │
│              │ Lost when session ends                         │
├──────────────┼──────────────────────────────────────────────┤
│ Long-term    │ Persisted across sessions                      │
│ (External)   │ Stored in vector DB or key-value store        │
│              │ Retrieved at session start                     │
├──────────────┼──────────────────────────────────────────────┤
│ Episodic     │ "Last Tuesday you asked about X"               │
│              │ Stored as past interaction logs                │
├──────────────┼──────────────────────────────────────────────┤
│ Semantic     │ Facts extracted from conversations             │
│              │ "User is working on project X"                 │
└──────────────┴──────────────────────────────────────────────┘
```

**Implementing:** Short-term + Long-term semantic.

## Short-Term Strategy

| Strategy | Tradeoff |
|---|---|
| Full history | Simple, breaks on long chats |
| Sliding window | Simple, loses early context |
| **Summarization buffer** ✓ | Best balance |
| Token-aware trimming | Most precise |

**Chosen:** After K messages, Gemini summarizes older messages into one "history summary". Recent messages kept verbatim. LangChain has this built in.

## Long-Term Memory

- **Retrieval-based:** Store past interactions as embeddings in ChromaDB. At session start, embed new query, retrieve relevant past memories (same as RAG but over history).
- **Structured extraction:** After each session, LLM extracts key facts ("user prefers formal tone") → stored in SQLite.

**Both implemented.** Session ID (UUID) ties everything together.

## Deliverables

- [ ] `memory/short_term.py` — summarization buffer integrated into LangGraph state
- [ ] `memory/long_term.py` — ChromaDB memory collection + retrieval + write
- [ ] SQLite store for structured user/session facts
- [ ] Session ID tracking in FastAPI (UUID per conversation)
- [ ] Memory injected into agent context at session start
- [ ] Memory written and indexed at session end
- [ ] Integration test: reference something from session 1 in session 2

## Packages

```
sqlalchemy
aiosqlite
```

## Future Improvement — Query Condensation

When a user asks a follow-up question ("what if I pay late?"), the raw query has no context.
Before retrieval, condense the conversation history + current query into a single standalone query using an LLM call.

**Where:** inside `agents/rag_agent.py`, before the retrieval call.
**When to trigger:** only if `state["messages"]` is non-empty (skip on first turn).
**How:**

```python
def condense_query(messages: list[BaseMessage], current_query: str) -> str:
    """Rewrite current_query as a standalone question using conversation history."""
    ...
    # returns e.g. "What are the consequences of late payment in the service contract?"
```

**Why not now:** session message threading (the prerequisite) is not yet wired across turns.
Implement after short-term session store is in place.

## Next

→ [Phase 5: MCP Server](phase-5.md)
