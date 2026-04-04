"""
Two-stage retrieval:
  1. Bi-encoder (BGE) → ANN search in ChromaDB → top-K child chunks
  2. Cross-encoder (ms-marco) → rerank → top-N
  3. Fetch parent chunks for top-N → return to LLM
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
    Retrieve relevant parent chunks for a query.

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
