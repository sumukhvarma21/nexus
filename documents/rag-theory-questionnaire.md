# RAG Theory Questionnaire

Answer all questions in writing. Where MCQ is provided, justify your answer in 1–2 sentences.
Hints are collapsed at the bottom of each section — try without them first.

---

## Section 1 — Chunking

**Q1.** What is the core problem with fixed-size chunking? Give a concrete example of how it breaks retrieval.

**Q2.** Explain the parent-child chunking pattern. What is stored at each level, which chunk is used for retrieval, and which is sent to the LLM — and why are these different?

**Q3.** We chose child chunk size = 256 tokens and parent = 1024. Why not just use 1024 for both retrieval and context?

**Q4.** In production at scale (millions of documents), what would you change about the chunking pipeline?
> Think about: async processing, queue systems, re-ingestion on document update, deduplication.

**Q5.** Semantic chunking splits on meaning boundaries rather than token counts. Why did we **not** choose it despite higher quality?

---

## Section 2 — Embedding & Vector Store

**Q6.** We use `BAAI/bge-base-en-v1.5` (768 dimensions) over `all-MiniLM-L6-v2` (384 dimensions). What does a higher dimensional embedding space give you, and what does it cost?

**Q7.** Embeddings are pre-computed at ingest time. What would break if you changed the embedding model after documents were already stored?

**Q8.** We use ChromaDB locally and plan to migrate to Qdrant in Phase 7. Name two concrete reasons Qdrant is better suited for production.

**Q9.** What is ANN (Approximate Nearest Neighbour) search? What does "approximate" mean here and why is it acceptable?

**Q10.** In production, a user uploads the same PDF twice. What happens in our current ChromaDB setup and how would you fix it?
> This was a bug we explicitly identified in the project.

---

## Section 3 — Two-Stage Retrieval (Bi-encoder + Cross-encoder)

**Q11.** Explain in your own words why bi-encoders are fast but approximate, and why cross-encoders are slow but accurate. What is the architectural difference that causes this?

**Q12.** We retrieve top-20 with the bi-encoder and rerank to top-5 with the cross-encoder. Why not just run the cross-encoder over all chunks in the DB?

**Q13.** The cross-encoder reranks the same 20 chunks that were retrieved by the bi-encoder. A student argues: "if the cross-encoder is smarter, why did we bother with the bi-encoder — could the cross-encoder have found better chunks that the bi-encoder missed?" How would you respond?

**Q14.** After cross-encoder reranking, we return the **parent** chunks (1024 tokens) to the LLM, not the child chunks (256 tokens) that were actually retrieved. Why?

**Q15.** In production with 10M document chunks, what changes would you make to the two-stage retrieval pipeline?
> Think about: GPU inference for cross-encoder, async batching, caching frequent queries, index sharding.

---

## Section 4 — Query Transformation

**Q16.** Single-shot RAG fails on compound queries. Give an example compound query and walk through exactly how multi-step decomposition handles it.

**Q17.** HyDE generates a hypothetical answer before retrieval. What specific gap does it close? Write the exact query and hypothetical answer for this example:
> User asks: "What are the termination conditions in the contract?"

**Q18.** What is the risk of HyDE? In what scenario would it actively hurt retrieval quality?

**Q19.** Iterative retrieval makes up to 3 hops. What determines whether a second hop is triggered? What component makes this decision?

**Q20.** We identified that combining web_search_agent + HyDE is a meaningless combination. Explain why.

**Q21.** Our current `rag_agent` always uses `retrieve_multi_step` regardless of query complexity. Why is this suboptimal, and what is the proposed architectural fix?

---

## Section 5 — RAGAS Evaluation

**Q22.** Name the four RAGAS metrics and what each measures in one sentence each.

**Q23.** You run RAGAS and get high Faithfulness but low Answer Relevancy. What does this tell you about your system?

**Q24.** You get high Context Recall but low Context Precision. What does this mean and what would you tune?

**Q25.** Why do you need ground truth answers to run RAGAS? What happens if your ground truths are wrong?

**Q26.** In production, you can't manually write ground truths for thousands of queries. How would you scale evaluation?
> Think about: LLM-as-judge, human feedback loops, implicit signals (thumbs up/down, query reformulation).

---

## Section 6 — Memory

**Q27.** A user sends 30 messages in one session. Without any memory management, what breaks?

**Q28.** Explain the summarization buffer strategy. What gets summarized, what stays verbatim, and what data structure holds the summary?

**Q29.** We have two long-term stores: ChromaDB (episodic) and SQLite (semantic facts). For each, answer:
- What is stored?
- When is it written?
- When is it read?
- What would break if you removed it?

**Q30.** Session ID is a UUID generated per conversation. The server is stateless. Walk through exactly what happens across two separate API calls from the same user, and what the session ID enables.

**Q31.** The current implementation has a critical gap: `AgentState["messages"]` is not persisted between HTTP requests. What is the consequence of this, and what would the production fix be?

**Q32.** `get_user_facts()` injects all SQLite facts into every query. Name two failure modes this causes over time and propose a fix for each.

**Q33.** In what order are SQLite facts and ChromaDB memories injected into `memory_context`, and why does the order matter for LLM attention?

---

## Section 7 — Architecture & Production

**Q34.** A new engineer joins and asks: "why do we have both a FastAPI REST layer and an MCP server — aren't they redundant?" How do you explain the design decision?

**Q35.** The supervisor uses Gemini structured output with `Literal["rag_agent", "web_search_agent", "email_calendar_agent"]`. What would happen if you used a plain `str` instead, and why is the Literal important?

**Q36.** LangGraph validates conditional edge targets at compile time. Describe what "compile time" means in this context (it's not a traditional compiled language).

**Q37.** You need to add a new agent to the system (e.g. a `sql_agent` that queries a database). List every file you would need to change and what change each requires.

**Q38.** In production, you have 10,000 concurrent users all hitting `/chat`. Walk through every bottleneck in the current architecture and how you would address each one in Phase 7.
> Think about: stateless workers, session store, vector DB connections, LLM rate limits, async LangGraph execution.

**Q39.** A document contains contradictory information in two different sections (e.g. "payment is due in 30 days" in section 2, "payment is due in 45 days" in section 7). Both chunks are retrieved. What does the LLM do, and how would you handle this systematically?

**Q40.** You are asked to add per-user document isolation — User A's uploads should not be visible to User B's queries. What is the minimum change required to the current system?
> Think about: metadata filtering in ChromaDB, user_id in ingestion and retrieval.

---

## Hints

<details>
<summary>Q2 hint</summary>
Think about the tradeoff between precision (finding the right passage) and context (giving the LLM enough surrounding text to answer well).
</details>

<details>
<summary>Q7 hint</summary>
Embeddings are vectors in a space defined by the model's weights. Changing the model changes the geometry of that space.
</details>

<details>
<summary>Q11 hint</summary>
Bi-encoder: query and document are embedded separately, then compared by dot product. Cross-encoder: query and document are concatenated and fed through the model together. Joint attention = richer signal.
</details>

<details>
<summary>Q13 hint</summary>
The cross-encoder only sees what the bi-encoder retrieved. It cannot retrieve — it can only reorder. Think of it as a judge, not a scout.
</details>

<details>
<summary>Q18 hint</summary>
What if the LLM's hypothetical answer uses vocabulary or framing that doesn't match the actual documents?
</details>

<details>
<summary>Q21 hint</summary>
Two sequential routing nodes: first picks agent, second picks retrieval mode. This prevents invalid combinations and avoids over-decomposition of simple queries.
</details>

<details>
<summary>Q23 hint</summary>
Faithful = answer only says things the chunks say. Relevant = answer actually addresses the question. High faithfulness + low relevancy means the retrieved chunks were technically accurate but off-topic.
</details>

<details>
<summary>Q31 hint</summary>
Each HTTP request creates a new AgentState with messages=[]. What external store would let you load and save message history across requests?
</details>

<details>
<summary>Q38 hint</summary>
Start from the entry point (API Gateway) and trace a request all the way to the LLM and back. Where does shared state live? Where are the I/O waits? Which components can be horizontally scaled?
</details>
