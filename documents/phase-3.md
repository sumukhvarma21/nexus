# Phase 3 — LangGraph Orchestrator + Agent Nodes

**Goal:** Build the multi-agent system. Supervisor routes queries to specialist agents. Each agent is a LangGraph node.

**Depends on:** [Phase 2](phase-2.md)

## LangGraph Core Concepts

```
Nodes  = agents or functions (transform state)
Edges  = transitions between nodes (conditional or fixed)
State  = shared TypedDict passed through the entire graph
```

**Supervisor pattern (chosen):**
```
          ┌──────────────────────────────┐
          │       Supervisor Node         │
          │  (reads query, picks agent)   │
          └──────┬───────────────────────┘
                 │ routes to:
       ┌─────────┼──────────┐
       ↓         ↓          ↓
   RAG Agent  Web Agent  Email Agent
       └─────────┼──────────┘
                 ↓ all return to:
          ┌──────┴──────┐
          │  Synthesizer │
          └─────────────┘
```

**Why supervisor over peer-to-peer:** Hub-and-spoke means one node always knows current state — inspectable and no infinite loops.

## Agent State

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]      # full conversation history
    query: str                       # current user query
    sub_queries: list[str]           # decomposed sub-questions
    retrieved_context: list[str]     # RAG results
    web_results: list[str]           # web search results
    agent_called: str                # which agent handled this turn
    final_answer: str                # synthesized response
    next: str                        # routing decision (supervisor output)
```

State design is the most important decision — everything flows through it. Good state = easy debugging + easy to add agents later.

## Routing Strategy

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **LLM-based** ✓ | Ask LLM "which agent?" | Flexible | Token cost |
| Classifier | Fine-tuned intent | Fast | Needs training data |
| Keyword/regex | Pattern match | Very fast | Brittle |

**Chosen:** Gemini returns `{"agent": "rag_agent"}` via structured output. Easy to inspect routing decisions and add new agents.

## Deliverables

- [ ] `agents/orchestrator.py` — LangGraph supervisor with structured routing
- [ ] `agents/rag_agent.py` — wraps Phase 1+2 RAG pipeline as LangGraph node
- [ ] `agents/web_search_agent.py` — Tavily web search as LangGraph node
- [ ] Full `AgentState` TypedDict
- [ ] LangSmith tracing enabled
- [ ] `POST /chat` endpoint — replaces `/query`, goes through orchestrator
- [ ] Test: "What does my document say about X?" → RAG agent
- [ ] Test: "What's the latest news on Y?" → Web agent

## Packages

```
langgraph
langsmith
tavily-python
```

## Next

→ [Phase 4: Memory](phase-4.md)
