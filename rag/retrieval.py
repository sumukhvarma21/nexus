"""
Two-stage retrieval (standard) + multi-step retrieval strategies:

Standard:
  1. Bi-encoder (BGE) → ANN search in ChromaDB → top-K child chunks
  2. Cross-encoder (ms-marco) → rerank → top-N
  3. Fetch parent chunks for top-N → return to LLM

Multi-step strategies (Phase 2):
  - HyDE: embed a hypothetical answer instead of the raw query
  - Multi-step: decompose query → retrieve per sub-question → merge
  - Iterative: retrieve → check sufficiency → refine → repeat (max 3 hops)
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from config import settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def retrieve(query: str) -> list[dict]:
    """
    Standard two-stage retrieval: bi-encoder → cross-encoder rerank → parent chunks.

    Returns a list of dicts with keys: content, source_file, score
    """
    embeddings = _get_embeddings()

    child_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    parent_store = Chroma(
        collection_name="parent_chunks",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    # Stage 1: bi-encoder ANN search over child chunks
    child_results = child_store.similarity_search(query, k=settings.retrieval_top_k)

    if not child_results:
        return []

    # Stage 2: cross-encoder reranking
    reranker = _get_reranker()
    pairs = [[query, doc.page_content] for doc in child_results]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, child_results), key=lambda x: x[0], reverse=True)
    top_children = ranked[: settings.rerank_top_n]

    # Stage 3: fetch parent chunks for top-N children
    seen_parent_ids: set[str] = set()
    parent_chunks: list[dict] = []

    for score, child in top_children:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        parents = parent_store.get(where={"doc_id": parent_id})
        if parents and parents["documents"]:
            parent_chunks.append(
                {
                    "content": parents["documents"][0],
                    "source_file": child.metadata.get("source_file", "unknown"),
                    "score": float(score),
                }
            )

    return parent_chunks


def _merge_chunks(chunks_list: list[list[dict]]) -> list[dict]:
    """Merge multiple retrieval results, deduplicating by content."""
    seen_content: set[str] = set()
    merged: list[dict] = []
    for chunks in chunks_list:
        for chunk in chunks:
            # Use first 200 chars as dedup key (full content may be long)
            key = chunk["content"][:200]
            if key not in seen_content:
                seen_content.add(key)
                merged.append(chunk)
    # Sort by score descending after merging
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Phase 2 Strategy 1: HyDE Retrieval
# ---------------------------------------------------------------------------

def retrieve_with_hyde(query: str) -> list[dict]:
    """
    HyDE retrieval: embed a hypothetical answer instead of the raw query.

    The hypothetical answer sits in 'answer vector space' which more closely
    matches document embeddings than a question does.
    """
    from rag.query_processor import generate_hypothetical_document

    hypothetical_doc = generate_hypothetical_document(query)
    # Retrieve using hypothetical doc as the search query (its embedding is what matters)
    return retrieve(hypothetical_doc)


# ---------------------------------------------------------------------------
# Phase 2 Strategy 2: Multi-Step / Decomposition Retrieval
# ---------------------------------------------------------------------------

def retrieve_multi_step(query: str) -> list[dict]:
    """
    Decompose query into sub-questions, retrieve for each, merge results.

    Handles compound queries where a single embedding would average out semantics
    and miss relevant chunks for either sub-topic.
    """
    from rag.query_processor import decompose_query

    sub_questions = decompose_query(query)

    if len(sub_questions) == 1:
        # No decomposition needed — fall back to standard retrieval
        return retrieve(query)

    all_chunks = [retrieve(sq) for sq in sub_questions]
    return _merge_chunks(all_chunks)


# ---------------------------------------------------------------------------
# Phase 2 Strategy 3: Iterative Retrieval
# ---------------------------------------------------------------------------

def retrieve_iterative(query: str, max_hops: int = 3) -> tuple[list[dict], int]:
    """
    Iterative retrieval: retrieve → check sufficiency → refine → repeat.

    Returns (chunks, hops_used).
    The LLM decides after each retrieval whether more context is needed and
    provides a refined query targeting the missing information.
    """
    from rag.query_processor import check_context_sufficiency

    all_chunks: list[dict] = []
    current_query = query

    for hop in range(1, max_hops + 1):
        new_chunks = retrieve(current_query)
        all_chunks = _merge_chunks([all_chunks, new_chunks])

        if not all_chunks:
            return all_chunks, hop

        context = "\n\n---\n\n".join(c["content"] for c in all_chunks[:5])
        check = check_context_sufficiency(query, context)

        if check.sufficient or not check.refined_query:
            return all_chunks, hop

        current_query = check.refined_query

    return all_chunks, max_hops
