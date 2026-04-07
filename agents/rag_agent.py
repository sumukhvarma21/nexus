"""
RAG Agent — LangGraph node that runs the full Phase 1+2 retrieval pipeline.

Routing hint: called when the query is about ingested documents.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from config import settings
from rag.retrieval import retrieve_multi_step
from agents.state import AgentState


def rag_agent(state: AgentState) -> AgentState:
    """Retrieve relevant chunks and synthesize an answer from ingested documents."""
    query = state["query"]

    # Use multi_step retrieval — handles both simple and compound queries well
    chunks = retrieve_multi_step(query)

    if not chunks:
        return {
            **state,
            "retrieved_context": [],
            "agent_called": "rag_agent",
            "final_answer": "I couldn't find relevant information in the uploaded documents.",
        }

    context_strings = [c["content"] for c in chunks]
    context = "\n\n---\n\n".join(context_strings)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided document context. If the context doesn't contain the "
                "answer, say so clearly."
            )
        ),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "retrieved_context": context_strings,
        "agent_called": "rag_agent",
        "final_answer": response.content,
    }
