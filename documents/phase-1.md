# Phase 1 — RAG Pipeline (Built from Scratch)

**Goal:** Ingest documents, chunk intelligently, embed, store, retrieve with bi-encoder + cross-encoder reranking.

**Depends on:** [Phase 0](phase-0.md)

## Chunking Strategy

| Strategy | Best for | Weakness |
|---|---|---|
| Fixed-size | Simple baseline | Cuts mid-sentence |
| Recursive character | General text | Still size-based |
| Semantic chunking | High quality | Slow, expensive |
| **Parent-child** ✓ | Best precision + context | More complex store |

**Chosen: Recursive character + Parent-Child pattern**
- Child chunks: 256 tokens — small, precise, used for retrieval
- Parent chunks: 1024 tokens — large, contextual, sent to the LLM

## Embedding Model

| Model | Dimensions | Quality | Cost |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Good baseline | Free, local |
| **`BAAI/bge-base-en-v1.5`** ✓ | 768 | Better | Free, local |
| `BAAI/bge-large-en-v1.5` | 1024 | Best open-source | Free, local |

**Chosen:** Best balance of quality and local speed, no API calls during retrieval.

## Two-Stage Retrieval

```
Query
  ↓
Bi-encoder (embed query) → ANN search in ChromaDB → Top-K candidates (K=20)
  ↓
Cross-encoder (ms-marco-MiniLM-L-6-v2) → Re-ranked Top-N (N=5)
  ↓
Return parent chunks for Top-N → LLM context
```

Bi-encoder is fast (pre-computed, ANN search) but approximate. Cross-encoder is slow (compares query+doc at inference) but accurate.

## Vector Store Decision

| | ChromaDB | Qdrant |
|---|---|---|
| Local dev | ✅ Easy | ✅ Docker |
| AWS deployment | ❌ Not managed | ✅ ECS/managed |
| Performance | Good | Excellent |

**Phase 1:** ChromaDB. **Phase 7:** Migrate to Qdrant.

## Deliverables

- [ ] `rag/ingestion.py` — PDF/TXT → recursive chunk → parent-child pairs → embed → store in ChromaDB
- [ ] `rag/retrieval.py` — bi-encoder (top 20) → cross-encoder rerank (top 5) → return parent chunks
- [ ] `POST /ingest` — file upload endpoint
- [ ] `POST /query` — RAG-only endpoint (no agent yet)
- [ ] Manual test with 2-3 PDFs, compare chunk quality vs fixed-size

## Packages

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

## Next

→ [Phase 2: Multi-Step RAG](phase-2.md)
