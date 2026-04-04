# Phase 2 — Multi-Step RAG (Agentic Retrieval)

**Goal:** Fix single-shot RAG failures with query decomposition, HyDE, and iterative retrieval. Measure improvement with RAGAS.

**Depends on:** [Phase 1](phase-1.md)

## Why Single-Shot RAG Fails

**Problem 1 — Compound queries:**
> "Compare the refund policy in the uploaded contract with standard industry terms"

Embedding averages semantics — neither sub-topic gets retrieved cleanly.

**Fix: Query Decomposition**
```
Original query
      ↓
LLM breaks into sub-questions:
  1. "What is the refund policy in the contract?"
  2. "What are standard industry refund terms?"
      ↓
Retrieve independently for each sub-question → Synthesize
```

**Problem 2 — Vocabulary mismatch:**
> You ask: "What's the cancellation process?"
> Doc says: "Termination procedure is outlined in section 4..."

**Fix: HyDE (Hypothetical Document Embedding)**
```
Query → LLM generates hypothetical ideal answer
      → Embed the hypothetical answer (not the query)
      → Retrieve — query embedding now looks like a document embedding
```

**Problem 3 — Insufficient context after one pass:**

**Fix: Iterative Retrieval**
```
Retrieve → LLM decides "I need more info about X" → Refine query → Retrieve again (max 3 hops)
```

## RAGAS Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved chunks? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are retrieved chunks actually useful? |
| **Context Recall** | Did retrieval miss relevant chunks? |

**Workflow:** Run RAGAS after Phase 1 for a baseline. Run again after Phase 2. Prove improvement with numbers.

## Deliverables

- [ ] `rag/query_processor.py` — query decomposition using Gemini structured output
- [ ] HyDE retrieval as a configurable option alongside standard retrieval
- [ ] Iterative retrieval loop (max 3 hops, configurable)
- [ ] `eval/ragas_eval.py` — RAGAS evaluation script with test question set
- [ ] A/B comparison: standard vs multi-step RAG across 10 test queries
- [ ] Document metric results in `eval/results.md`

## Packages

```
ragas
datasets
```

## Next

→ [Phase 3: LangGraph Orchestrator](phase-3.md)
