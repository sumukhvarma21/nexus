# General-Purpose Multi-Agent System — Build Plan

> **Stack:** Python · FastAPI · LangGraph · Gemini Flash 2.5 · ChromaDB → Qdrant · FastMCP
> **Target:** Local-first, AWS Serverless at scale
> **Philosophy:** Learn the theory + tradeoffs before implementing each phase

---

## Project Structure (Final Vision)

```
agent-system/
├── agents/                       # Individual agent definitions
│   ├── orchestrator.py           # LangGraph router / supervisor
│   ├── rag_agent.py              # Document retrieval agent
│   ├── web_search_agent.py       # Web search agent
│   └── email_calendar_agent.py   # Gmail + GCal agent
├── rag/                          # RAG pipeline
│   ├── ingestion.py              # Chunking + embedding + storing
│   ├── retrieval.py              # Bi-encoder + reranking
│   └── query_processor.py        # Decomposition, HyDE, iterative
├── memory/                       # Short-term + long-term memory
│   ├── short_term.py
│   └── long_term.py
├── mcp/                          # MCP server exposing agents as tools
│   └── server.py
├── tools/                        # Tool definitions
│   ├── search.py
│   └── gmail_gcal.py
├── api/                          # FastAPI layer
│   └── routes.py
├── ui/                           # Simple frontend (final phase)
│   └── app.py                    # Chainlit app
├── config.py                     # Centralized config + env
├── requirements.txt
└── .env
```

---

## Phase 0 — Foundations & Project Scaffolding

**Goal:** Clean repo, working skeleton, environment setup. Understand the full architecture before writing any business logic.

### Theory: The Architecture You're Building

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

### Theory: Why LangGraph over Alternatives

| | LangGraph | CrewAI | Custom Router |
|---|---|---|---|
| Control | Explicit graph, you own every edge | High abstraction, magic underneath | Full control |
| Debugging | Built-in state tracing | Hard to inspect | Depends on you |
| Learning curve | Medium | Low | High |
| Production-readiness | High | Medium | High |
| Best for | Complex stateful workflows | Quick prototypes | Specific domains |

**Chosen: LangGraph** — you want to learn what's actually happening, state machines map cleanly to multi-agent routing, and it has first-class support for streaming + async which matters for AWS Lambda.

### Theory: Why Gemini Flash 2.5 over Alternatives

| Model | Context Window | Cost | Strengths |
|---|---|---|---|
| Gemini Flash 2.5 | 1M tokens | Free tier generous | Long context, multimodal |
| GPT-4o-mini | 128K tokens | Cheap | Strong ecosystem |
| Claude Haiku | 200K tokens | Cheap | Strong reasoning |

**Chosen: Gemini Flash 2.5** — free tier works for learning, 1M context window is critical for large-doc RAG, multimodal by default for future use.

### Deliverables

- [ ] Repo initialized with folder structure above
- [ ] `requirements.txt` with pinned versions
- [ ] `.env` + `config.py` with centralized settings (Pydantic Settings)
- [ ] `docker-compose.yml` for ChromaDB local
- [ ] FastAPI app boots, `/health` returns 200
- [ ] Understand the full architecture diagram before Phase 1

### Packages Introduced

```
fastapi
uvicorn[standard]
pydantic-settings
python-dotenv
httpx
```

---

## Phase 1 — RAG Pipeline (Built from Scratch)

**Goal:** Ingest documents, chunk intelligently, embed, store, retrieve with bi-encoder + cross-encoder reranking. Better than your previous fixed-size implementation.

### Theory: Chunking Strategies

| Strategy | How it works | Best for | Weakness |
|---|---|---|---|
| **Fixed-size** | Split every N chars/tokens | Simple baseline | Cuts mid-sentence, loses context |
| **Recursive character** | Tries paragraph → sentence → word boundaries | General text | Still size-based |
| **Sentence-based** | Split on sentence boundaries | Clean prose | Chunks too small for dense docs |
| **Semantic chunking** | Embed sentences, split where distance spikes | High quality | Slow, expensive |
| **Parent-child** | Small chunks retrieved, large parent returned | Best precision + context | More complex store |
| **AST-based** | Parse code structure | Codebases only | Only for code |

**Chosen: Recursive character + Parent-Child pattern**
- Child chunks: 256 tokens — small, precise, used for retrieval
- Parent chunks: 1024 tokens — large, contextual, sent to the LLM
- Best quality-to-complexity ratio without the cost of semantic chunking

### Theory: Embedding Models

| Model | Dimensions | Speed | Quality | Cost |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Very fast | Good baseline | Free, local |
| `BAAI/bge-base-en-v1.5` | 768 | Fast | Better | Free, local |
| `BAAI/bge-large-en-v1.5` | 1024 | Slower | Best open-source | Free, local |
| `text-embedding-3-small` | 1536 | API call | Excellent | Paid |
| `gemini-embedding` | 768 | API call | Excellent | Paid |

**Chosen: `BAAI/bge-base-en-v1.5`** — best balance of quality and local speed, no API calls during retrieval, free.

### Theory: Two-Stage Retrieval

```
Query
  ↓
Bi-encoder (embed query) → ANN search in ChromaDB → Top-K candidates (K=20)
  ↓
Cross-encoder (re-score each candidate vs query) → Re-ranked Top-N (N=5)
  ↓
Return parent chunks for Top-N → LLM context
```

**Why two stages:**
- Bi-encoder: fast (pre-computed embeddings, ANN search) but approximate
- Cross-encoder: slow (compares query+doc together at inference) but accurate
- Use bi-encoder to narrow the field, cross-encoder to pick the best

**Cross-encoder model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, accurate, runs locally.

### Theory: Vector Store Options

| | ChromaDB | Qdrant | Pinecone | pgvector |
|---|---|---|---|---|
| Local dev | ✅ Easy | ✅ Docker | ❌ Cloud only | ✅ Postgres |
| AWS deployment | ❌ Not managed | ✅ ECS/managed | ✅ Managed | ✅ RDS |
| Performance | Good | Excellent | Excellent | Good |
| Filtering | Basic | Advanced | Advanced | SQL |
| Learning curve | Low | Medium | Low | Low |

**Phase 1:** ChromaDB locally (familiar, zero setup overhead)
**Phase 7 (AWS):** Migrate to Qdrant — better filtering, proper production performance

### Deliverables

- [ ] `rag/ingestion.py` — upload PDF/TXT → recursive chunk → parent-child pairs → embed → store
- [ ] `rag/retrieval.py` — bi-encoder retrieval (top 20) → cross-encoder rerank (top 5) → return parent chunks
- [ ] `POST /ingest` — accepts file upload, runs ingestion pipeline
- [ ] `POST /query` — RAG-only endpoint (no agent yet)
- [ ] Manual test with 2-3 PDFs, compare chunk quality vs old fixed-size system

### Packages Introduced

```
langchain==0.3.15
langchain-google-genai==2.1.4
langchain-community==0.3.15
langchain-huggingface==0.1.2
langchain-chroma==0.1.4
langchain-text-splitters==0.3.8
chromadb==0.5.3
sentence-transformers
pypdf
python-multipart
numpy<2
```

---

## Phase 2 — Multi-Step RAG (Agentic Retrieval)

**Goal:** Make retrieval smarter. Single-shot RAG breaks on complex queries. Fix with query decomposition, HyDE, and iterative retrieval.

### Theory: Why Single-Shot RAG Fails

**Problem 1 — Compound queries:**

> "Compare the refund policy in the uploaded contract with standard industry terms"

Single-shot embeds the whole sentence. The embedding averages semantics. Neither "refund policy" nor "industry terms" gets retrieved cleanly.

**Fix: Query Decomposition**

```
Original query
      ↓
LLM breaks into sub-questions:
  1. "What is the refund policy in the contract?"
  2. "What are standard industry refund terms?"
      ↓
Retrieve independently for each sub-question
      ↓
Synthesize answers together into one response
```

**Problem 2 — Vocabulary mismatch:**

> You ask: "What's the cancellation process?"
> Doc says: "Termination procedure is outlined in section 4..."

Embeddings don't align because the words differ even though semantics are similar.

**Fix: HyDE (Hypothetical Document Embedding)**

```
Query: "What's the cancellation process?"
      ↓
LLM generates a hypothetical ideal answer:
"The cancellation process requires 30 days written notice..."
      ↓
Embed the hypothetical answer (not the original query)
      ↓
Retrieve — now your query embedding looks like a document embedding
```

Works because hypothetical answers use document-like language. Better alignment in embedding space.

**Problem 3 — Insufficient context after one retrieval pass:**

**Fix: Iterative Retrieval**

```
Retrieve → LLM reads chunks → decides "I need more info about X"
      ↓
Refine query → Retrieve again (max 3 hops)
      ↓
Synthesize all gathered context
```

This is the "agentic" part — the LLM decides whether retrieval is sufficient, not a fixed pipeline.

### Theory: RAG Evaluation with RAGAS

Before adding complexity, measure whether it's actually helping. This is how production teams validate RAG changes.

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved chunks? (no hallucination) |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Context Precision** | Are the retrieved chunks actually useful? |
| **Context Recall** | Did retrieval miss relevant chunks? |

**Workflow:** Run RAGAS after Phase 1 to get a baseline. Run again after Phase 2. Prove the improvement with numbers.

### Deliverables

- [ ] `rag/query_processor.py` — query decomposition using Gemini structured output
- [ ] HyDE retrieval as a configurable option alongside standard retrieval
- [ ] Iterative retrieval loop (max 3 hops, configurable)
- [ ] `eval/ragas_eval.py` — RAGAS evaluation script with a test question set
- [ ] A/B comparison: standard vs multi-step RAG across 10 test queries
- [ ] Document metric results in `eval/results.md`

### Packages Introduced

```
ragas
datasets
```

---

## Phase 3 — LangGraph Orchestrator + Agent Nodes

**Goal:** Build the multi-agent system. An orchestrator routes queries to the right specialist agent. Each agent is a LangGraph node.

### Theory: LangGraph Core Concepts

LangGraph models your system as a **stateful directed graph.**

```
Nodes  = agents or functions (do work, transform state)
Edges  = transitions between nodes (conditional or fixed)
State  = shared typed dict passed through the entire graph
```

**Supervisor pattern (chosen architecture):**

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
          │  Synthesizer │  (merges multi-agent results if needed)
          └─────────────┘
```

**Why Supervisor over Peer-to-Peer:**
Peer-to-peer agents calling each other is hard to debug and can loop infinitely. Supervisor is hub-and-spoke — one node always knows current state, making it inspectable and controllable.

### Theory: Agent State Design

The state is the most important design decision in LangGraph. Everything flows through it.

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

Good state design = easy debugging + easy to add new agents later without refactoring.

### Theory: Routing Strategies

| Strategy | How | Pros | Cons |
|---|---|---|---|
| **LLM-based routing** | Ask LLM "which agent should handle this?" | Flexible, handles edge cases | Costs a token call |
| **Classifier-based** | Fine-tuned intent classifier | Fast, cheap | Needs labeled training data |
| **Keyword/regex** | Pattern match on query | Very fast | Brittle, breaks on paraphrasing |
| **Embedding similarity** | Match query to agent description embeddings | Good balance | Needs good agent descriptions |

**Chosen: LLM-based routing with structured output**
Gemini returns `{"agent": "rag_agent"}` via structured output. Most flexible for learning, easy to inspect routing decisions, easy to add new agents.

### Deliverables

- [ ] `agents/orchestrator.py` — LangGraph supervisor with structured routing
- [ ] `agents/rag_agent.py` — wraps Phase 1+2 RAG pipeline as a LangGraph node
- [ ] `agents/web_search_agent.py` — Tavily web search as a LangGraph node
- [ ] Full `AgentState` TypedDict definition
- [ ] Routing correctly dispatches across all 3 agents
- [ ] LangSmith tracing enabled — visualize graph execution in the UI
- [ ] `POST /chat` endpoint — replaces `/query`, goes through orchestrator
- [ ] Test: "What does my document say about X?" → RAG agent
- [ ] Test: "What's the latest news on Y?" → Web agent

### Packages Introduced

```
langgraph
langsmith
tavily-python
```

---

## Phase 4 — Memory (Short-Term + Long-Term)

**Goal:** Agents remember conversation context within a session (short-term) and persist useful knowledge across sessions (long-term).

### Theory: Types of Agent Memory

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

**We implement:** Short-term (LangGraph state with summarization buffer) + Long-term semantic memory (separate ChromaDB collection).

### Theory: Short-Term Memory Management Strategies

| Strategy | How | Tradeoff |
|---|---|---|
| **Full history** | All messages in context | Simple, breaks on long chats |
| **Sliding window** | Keep last N messages | Simple, loses early context |
| **Summarization buffer** | Summarize old messages, keep recent verbatim | Best balance |
| **Token-aware trimming** | Trim by token count not message count | Most precise |

**Chosen: Summarization buffer** — after K messages, Gemini summarizes older messages into one "history summary" message. Recent messages kept verbatim. LangChain has this built in.

### Theory: Long-Term Memory Approaches

**Retrieval-based:** Store facts/interactions as embeddings in vector DB. At session start, embed the new query, retrieve relevant past memories. Works exactly like RAG but over your own history.

**Structured extraction:** After each session, run an LLM pass to extract key facts ("user prefers formal tone", "working on document X") and store as structured data. Query at session start.

**Chosen: Both** — retrieval-based for past conversation context (in ChromaDB), structured extraction for persistent user facts (in SQLite). Session ID ties everything together.

### Deliverables

- [ ] `memory/short_term.py` — summarization buffer integrated into LangGraph state
- [ ] `memory/long_term.py` — ChromaDB memory collection + retrieval + write
- [ ] SQLite store for structured user/session facts
- [ ] Session ID tracking in FastAPI (UUID per conversation)
- [ ] Memory injected into agent context at session start
- [ ] Memory written and indexed at session end
- [ ] Integration test: reference something from session 1 in session 2

### Packages Introduced

```
sqlalchemy
aiosqlite
```

---

## Phase 5 — MCP Server + Email/Calendar Agent

**Goal:** Expose the entire system via MCP protocol. Add Gmail and GCal tools. The MCP server is the standard interface for any client — UI, Claude Desktop, another agent.

### Theory: What MCP Actually Is

MCP (Model Context Protocol) is an open standard for exposing tools and resources to LLMs. Think of it as a USB-C standard — any MCP-compatible client can plug in and use your tools without knowing the internals.

```
MCP Client (UI / Claude Desktop / another agent)
        ↓  calls tool via MCP protocol (JSON-RPC over SSE)
MCP Server (your FastMCP server)
        ↓  routes internally to
FastAPI Backend → LangGraph Orchestrator → Agents
```

**MCP vs REST API:**

| | MCP | REST API |
|---|---|---|
| Tool discovery | Client auto-discovers tools + schemas | Manual docs required |
| Type safety | Strongly typed tool definitions | OpenAPI optional |
| LLM-native | Designed for LLM tool use | General purpose |
| Streaming | First-class SSE support | Needs custom SSE/WS |

MCP is the right interface here: any LLM that understands MCP can use your system as a tool without custom integration per client.

### Theory: Gmail/GCal Integration Options

| Option | Effort | Control | Approach |
|---|---|---|---|
| Direct Gmail API (OAuth2) | High | Full | Write integration yourself |
| Existing Google MCP servers | Low | Standard | Connect your MCP to Google's MCP |
| LangChain Gmail toolkit | Medium | Good | Wrapper over Gmail API |

**Chosen: Existing Google MCP servers** — Google publishes official MCP servers for Gmail and GCal. Your Email agent calls your MCP server, which connects to Google's MCP. Clean separation, no raw OAuth2 code to maintain.

### MCP Tools to Expose

```
chat(message, session_id)             → full orchestrator response (streaming)
query_documents(query, doc_filter?)   → RAG results
ingest_document(file_path)            → ingestion status
web_search(query)                     → search results
read_emails(max_results, query?)      → email list + summaries
read_calendar(start_date, end_date)   → events list
list_sessions()                       → active session IDs
```

### Deliverables

- [ ] `mcp/server.py` — FastMCP server exposing all tools above
- [ ] `agents/email_calendar_agent.py` — LangGraph node using Gmail + GCal MCP tools
- [ ] Orchestrator updated to route email and calendar intent
- [ ] Test with Claude Desktop as MCP client
- [ ] End-to-end: "What meetings do I have tomorrow and summarize related emails?" → routes correctly through the graph
- [ ] Streaming responses working through MCP

### Packages Introduced

```
fastmcp
mcp
```

---

## Phase 6 — Simple UI

**Goal:** Minimal, functional chat UI that calls your backend. Not a design project — enough to demo and use the system day-to-day.

### Theory: UI Options for LLM Apps

| Option | Effort | Quality | Customizable | Best for |
|---|---|---|---|---|
| **Chainlit** | Very low | Good, chat-native | Medium | LLM chat apps exactly like this |
| **Gradio** | Very low | Functional | Low | ML demos |
| **Streamlit** | Low | Functional | Medium | Data apps |
| **Raw HTML + JS** | Medium | Whatever you build | Full | Full control |
| **Next.js** | High | Production | Full | Deployed product |

**Chosen: Chainlit** — purpose-built for LLM chat apps. File upload, streaming, session management, source citation display — all out of the box. You write almost no frontend code.

**For AWS deployment later:** Chainlit can be containerized and deployed to ECS, or replaced with a Next.js frontend hitting FastAPI directly. The MCP and FastAPI layers stay unchanged.

### UI Features

- Chat interface with streaming token-by-token responses
- File upload → triggers `/ingest` endpoint
- Agent attribution per response ("Answered by: RAG Agent")
- Source chunk display — expandable citations showing which chunks were used
- Simple toggle panel: web search on/off, memory on/off
- Session display in sidebar

### Deliverables

- [ ] `ui/app.py` — Chainlit app wired to FastAPI backend
- [ ] Streaming responses rendering correctly
- [ ] File upload working end-to-end (upload → ingest → queryable)
- [ ] Agent attribution label shown per message
- [ ] Source chunks shown as expandable citations below answers
- [ ] Settings toggle panel working

### Packages Introduced

```
chainlit
```

---

## Phase 7 — AWS Serverless Deployment

**Goal:** Move everything to AWS. Serverless where possible. Production-grade configuration.

### Theory: AWS Architecture for This System

```
                    CloudFront (CDN)
                          ↓
                    API Gateway (HTTP)
                    ↙              ↘
     Lambda (FastAPI via Mangum)   Lambda (MCP Server)
               ↓                           ↓
       ECS Fargate Tasks           Google MCP Servers
       (long-running LangGraph      (Gmail, GCal)
        workflows, async)
               ↓
      ┌─────────────────┐
      │  Qdrant on ECS  │   S3 (document storage)
      │  (vector store) │   ElastiCache Redis (sessions)
      └─────────────────┘   Secrets Manager (env vars)
                            CloudWatch (logs + traces)
```

**Why not Lambda for everything:**
Lambda has a 15-minute timeout. LangGraph workflows with multiple agent hops + retrieval can exceed this. Short API calls (health, ingest triggers) go to Lambda. Long agentic workflows go to ECS Fargate tasks triggered asynchronously.

### Theory: ChromaDB → Qdrant Migration

ChromaDB is not designed as a managed distributed service. Qdrant runs well on ECS, has a managed cloud offering, better filtering capabilities, and better performance at scale.

| | ChromaDB | Qdrant |
|---|---|---|
| Managed AWS option | None | ECS or Qdrant Cloud |
| Filtering | Basic metadata | Advanced payload filtering |
| Multi-tenancy | Manual | Built-in collection isolation |
| Performance | Good | Excellent |
| Migration effort | One-time script | Worth it |

### AWS Services Mapping

| AWS Service | Replaces | Purpose |
|---|---|---|
| API Gateway + Lambda | `uvicorn` local server | Serverless HTTP |
| ECS Fargate | Long-running local processes | LangGraph async workflows |
| S3 | Local file storage | Document uploads + storage |
| Qdrant on ECS | ChromaDB | Production vector store |
| ElastiCache Redis | In-memory session state | Distributed session memory |
| Secrets Manager | `.env` file | Secure config management |
| CloudWatch | Print statements | Logging + tracing |
| ECR | Local Docker images | Container registry |

### Deliverables

- [ ] `Dockerfile` for FastAPI app
- [ ] Mangum adapter wiring FastAPI to Lambda
- [ ] `scripts/migrate_chroma_to_qdrant.py` — one-time migration script
- [ ] CDK or Terraform for all infrastructure (prefer CDK for Python familiarity)
- [ ] GitHub Actions CI/CD pipeline: push → ECR → ECS deploy
- [ ] Environment-aware config: `config.py` switches between local and AWS automatically
- [ ] Redis session store replacing in-memory
- [ ] Full system running end-to-end on AWS
- [ ] Load test: 10 concurrent users, measure p95 latency

### Packages Introduced

```
mangum
boto3
qdrant-client
redis
```

---

## Full Dependency Timeline

```
Phase 0:  fastapi  uvicorn[standard]  pydantic-settings  python-dotenv  httpx

Phase 1:  + langchain==0.3.15  langchain-google-genai==2.1.4
            langchain-community==0.3.15  langchain-huggingface==0.1.2
            langchain-chroma==0.1.4  langchain-text-splitters==0.3.8
            chromadb==0.5.3  sentence-transformers  pypdf
            python-multipart  numpy<2

Phase 2:  + ragas  datasets

Phase 3:  + langgraph  langsmith  tavily-python

Phase 4:  + sqlalchemy  aiosqlite

Phase 5:  + fastmcp  mcp

Phase 6:  + chainlit

Phase 7:  + mangum  boto3  qdrant-client  redis
```

---

## Learning Resources Per Phase

| Phase | Read Before Starting |
|---|---|
| **Phase 1** | LangChain RAG docs · Parent Document Retriever guide · BEIR benchmark (why BGE beats MiniLM) |
| **Phase 2** | RAGAS paper (arxiv 2309.15217) · HyDE paper — Gao et al. 2022 · LangChain query decomposition cookbook |
| **Phase 3** | LangGraph conceptual docs · ReAct paper (Yao et al. 2022) · LangGraph supervisor example in official repo |
| **Phase 4** | LangGraph memory guide · MemGPT paper (for long-term memory intuition) |
| **Phase 5** | MCP specification at modelcontextprotocol.io · FastMCP docs · Google MCP server repos |
| **Phase 6** | Chainlit docs · Chainlit + LangGraph integration example |
| **Phase 7** | AWS Lambda + Mangum guide · Qdrant production docs · AWS CDK Python getting started |

---

## Key Concepts Glossary

| Term | What it means in this project |
|---|---|
| **Bi-encoder** | Separate encoders for query and document. Fast ANN lookup. Used in first retrieval stage. |
| **Cross-encoder** | Joint encoding of query+document pair. Slow but accurate. Used for reranking. |
| **ANN** | Approximate Nearest Neighbor search. How ChromaDB/Qdrant find similar vectors fast. |
| **Parent-child chunking** | Small chunks retrieved, large parent chunk sent to LLM for richer context. |
| **HyDE** | Generate a hypothetical ideal answer, embed it, use that embedding for retrieval. |
| **LangGraph node** | A function or agent that reads from state, does work, writes back to state. |
| **LangGraph edge** | A transition between nodes. Can be conditional (routing) or fixed. |
| **Supervisor pattern** | One orchestrator node that routes to specialist nodes. Hub-and-spoke. |
| **MCP** | Model Context Protocol. Standard for exposing tools to LLMs across clients. |
| **RAGAS** | Retrieval Augmented Generation Assessment. Framework for measuring RAG quality. |
| **Mangum** | ASGI adapter that lets FastAPI run inside AWS Lambda. |
| **Session ID** | UUID per conversation. Ties together short-term memory, history, and long-term facts. |
```