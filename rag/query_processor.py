"""
Query transformation utilities for multi-step RAG.

Three strategies:
  1. Decomposition  — break compound queries into independent sub-questions
  2. HyDE           — generate a hypothetical answer to embed instead of the query
  3. Sufficiency    — ask the LLM if retrieved context is enough (iterative retrieval)
"""

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )


# ---------------------------------------------------------------------------
# Strategy 1: Query Decomposition
# ---------------------------------------------------------------------------

class DecomposedQuery(BaseModel):
    sub_questions: list[str] = Field(
        description=(
            "List of independent sub-questions that together cover the original query. "
            "Each sub-question should be self-contained and retrievable on its own. "
            "Return 1 sub-question if the query is already simple."
        )
    )


def decompose_query(query: str) -> list[str]:
    """
    Break a compound query into independent sub-questions using Gemini structured output.

    Example:
      "Compare the refund policy in the contract with standard industry terms"
      → ["What is the refund policy in the contract?",
         "What are standard industry refund terms?"]
    """
    llm = _llm().with_structured_output(DecomposedQuery)

    prompt = (
        "You are a query decomposition expert for a RAG (Retrieval-Augmented Generation) system.\n\n"
        "Break the following query into independent sub-questions that can each be answered "
        "by retrieving a separate document chunk. If the query is already simple and focused, "
        "return it as a single sub-question.\n\n"
        f"Query: {query}"
    )

    result: DecomposedQuery = llm.invoke(prompt)
    return result.sub_questions


# ---------------------------------------------------------------------------
# Strategy 2: HyDE (Hypothetical Document Embedding)
# ---------------------------------------------------------------------------

def generate_hypothetical_document(query: str) -> str:
    """
    Generate a hypothetical answer to the query for HyDE retrieval.

    The hypothetical answer is embedded instead of the raw query — its embedding
    sits in 'answer space' rather than 'question space', so similarity search
    finds real document chunks that use the same vocabulary.

    The content may be partially hallucinated; that's fine — we only use it for
    embedding, not as an answer.
    """
    llm = _llm()

    prompt = (
        "Write a concise, factual paragraph (3-5 sentences) that would be a good answer "
        "to the following question. This will be used to find similar real documents, so "
        "use plausible terminology and vocabulary a document on this topic would use. "
        "Do NOT hedge or say you don't know — write a confident hypothetical answer.\n\n"
        f"Question: {query}"
    )

    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# Strategy 3: Context Sufficiency Check (Iterative Retrieval)
# ---------------------------------------------------------------------------

class SufficiencyCheck(BaseModel):
    sufficient: bool = Field(
        description="True if the provided context is sufficient to answer the question."
    )
    refined_query: str = Field(
        description=(
            "If not sufficient, a more specific query to retrieve additional context. "
            "If sufficient, return an empty string."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining what information is missing or why context is sufficient."
    )


def check_context_sufficiency(query: str, context: str) -> SufficiencyCheck:
    """
    Ask the LLM whether retrieved context is sufficient to answer the query.

    Returns a SufficiencyCheck with:
      - sufficient: bool
      - refined_query: a more targeted query if more context is needed
      - reasoning: brief explanation
    """
    llm = _llm().with_structured_output(SufficiencyCheck)

    prompt = (
        "You are evaluating whether retrieved context is sufficient to answer a question.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved Context:\n{context}\n\n"
        "Determine if this context is sufficient to give a complete, accurate answer. "
        "If not sufficient, provide a specific refined query to retrieve the missing information."
    )

    return llm.invoke(prompt)
