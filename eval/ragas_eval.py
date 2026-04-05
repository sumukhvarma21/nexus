"""
RAGAS evaluation script — compares standard vs multi-step RAG.

Usage:
    python eval/ragas_eval.py

Requires test questions in eval/test_questions.py and documents already ingested.

What it measures:
  - Faithfulness:      Is the answer grounded in retrieved chunks?
  - Answer Relevancy:  Does the answer address the question?
  - Context Precision: Are retrieved chunks actually useful (not noisy)?
  - Context Recall:    Did retrieval miss relevant chunks? (needs ground_truth)
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import settings
from rag.retrieval import retrieve, retrieve_with_hyde, retrieve_multi_step, retrieve_iterative
from eval.test_questions import TEST_QUESTIONS


def _build_answer(query: str, chunks: list[dict]) -> str:
    """Generate an answer from retrieved chunks using Gemini."""
    from langchain.schema import HumanMessage, SystemMessage

    if not chunks:
        return "No relevant documents found."

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )
    context = "\n\n---\n\n".join(c["content"] for c in chunks)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. Be concise and accurate. "
                "If the context doesn't contain the answer, say so."
            )
        ),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    return response.content


def _run_retrieval(query: str, mode: str) -> list[dict]:
    """Run retrieval with the given mode."""
    if mode == "standard":
        return retrieve(query)
    elif mode == "hyde":
        return retrieve_with_hyde(query)
    elif mode == "multi_step":
        return retrieve_multi_step(query)
    elif mode == "iterative":
        chunks, _ = retrieve_iterative(query)
        return chunks
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _build_ragas_dataset(mode: str) -> Dataset:
    """Build a RAGAS dataset for one retrieval mode."""
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in TEST_QUESTIONS:
        query = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"  [{mode}] Retrieving: {query[:60]}...")
        chunks = _run_retrieval(query, mode)
        answer = _build_answer(query, chunks)

        rows["question"].append(query)
        rows["answer"].append(answer)
        rows["contexts"].append([c["content"] for c in chunks] if chunks else [""])
        rows["ground_truth"].append(ground_truth)

    return Dataset.from_dict(rows)


def run_evaluation(modes: list[str] | None = None) -> dict:
    """
    Run RAGAS evaluation for each mode and return results.

    Args:
        modes: List of modes to evaluate. Defaults to ["standard", "hyde", "multi_step"].

    Returns:
        Dict mapping mode → metric scores.
    """
    if modes is None:
        modes = ["standard", "hyde", "multi_step"]

    # RAGAS needs an LLM and embeddings for its own evaluation
    ragas_llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )
    ragas_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.google_api_key,
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    all_results = {}

    for mode in modes:
        print(f"\nEvaluating mode: {mode}")
        dataset = _build_ragas_dataset(mode)

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        scores = {
            "faithfulness": round(result["faithfulness"], 4),
            "answer_relevancy": round(result["answer_relevancy"], 4),
            "context_precision": round(result["context_precision"], 4),
            "context_recall": round(result["context_recall"], 4),
        }
        all_results[mode] = scores
        print(f"  Scores: {scores}")

    return all_results


def save_results(results: dict, output_path: str = "eval/results.md") -> None:
    """Write A/B comparison results to a markdown file."""
    lines = [
        "# RAG Evaluation Results\n",
        "Metrics produced by RAGAS (LLM-as-judge). Higher is better (0–1 scale).\n",
        "| Metric | " + " | ".join(results.keys()) + " |",
        "|--------|" + "--------|" * len(results),
    ]

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for metric in metrics:
        row = f"| {metric} |"
        for mode_scores in results.values():
            row += f" {mode_scores.get(metric, 'N/A')} |"
        lines.append(row)

    lines.append("\n## Raw Results\n")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    results = run_evaluation()
    save_results(results)
