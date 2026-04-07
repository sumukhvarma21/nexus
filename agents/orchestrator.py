"""
Supervisor Orchestrator — LangGraph graph that routes queries to specialist agents.

Graph structure:
    supervisor → rag_agent            → END
              → web_agent             → END
              → email_calendar_agent  → END
"""

import uuid
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from config import settings
from agents.state import AgentState
from agents.rag_agent import rag_agent
from agents.web_search_agent import web_search_agent
from agents.email_calendar_agent import email_calendar_agent
from memory.short_term import maybe_summarize
from memory.long_term import save_interaction, retrieve_relevant_memories, get_user_facts


# ---------------------------------------------------------------------------
# Routing schema — Gemini returns this via structured output
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    agent: Literal["rag_agent", "web_search_agent", "email_calendar_agent"]
    reasoning: str


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def supervisor(state: AgentState) -> AgentState:
    """
    Reads the user query and picks which agent should handle it.

    - rag_agent       → query is about uploaded/ingested documents
    - web_search_agent → query needs current or external information
    """
    query = state["query"]

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )
    structured_llm = llm.with_structured_output(RoutingDecision)

    messages = [
        SystemMessage(
            content=(
                "You are a routing agent. Given a user query, decide which specialist "
                "agent should handle it.\n\n"
                "Choose 'rag_agent' if the query is about documents the user has uploaded "
                "(e.g., contracts, handbooks, reports, FAQs).\n"
                "Choose 'web_search_agent' if the query needs current or external information "
                "not likely to be in uploaded documents (e.g., news, recent events, live data).\n"
                "Choose 'email_calendar_agent' if the query is about emails, calendar events, "
                "meetings, or scheduling.\n\n"
                "When in doubt, prefer 'rag_agent'."
            )
        ),
        HumanMessage(content=f"User query: {query}"),
    ]

    decision: RoutingDecision = structured_llm.invoke(messages)

    return {
        **state,
        "next": decision.agent,
    }


# ---------------------------------------------------------------------------
# Conditional edge — reads state["next"] to decide which node to go to
# ---------------------------------------------------------------------------

def route(state: AgentState) -> str:
    return state["next"]


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("web_search_agent", web_search_agent)
    graph.add_node("email_calendar_agent", email_calendar_agent)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "rag_agent": "rag_agent",
            "web_search_agent": "web_search_agent",
            "email_calendar_agent": "email_calendar_agent",
        },
    )

    graph.add_edge("rag_agent", END)
    graph.add_edge("web_search_agent", END)
    graph.add_edge("email_calendar_agent", END)

    return graph.compile()


# Singleton — compiled once at import time
orchestrator = build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_chat(query: str, session_id: str = "") -> dict:
    """
    Run the full orchestrator for a single query.

    Args:
        query:      The user's question.
        session_id: UUID for the session. If empty string, a new UUID is generated.

    Returns:
        {
            "answer": str,
            "agent_called": str,
            "retrieved_context": list[str],  # non-empty for rag_agent
            "web_results": list[str],         # non-empty for web_search_agent
            "session_id": str,
        }
    """
    # Generate a session ID if none provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Retrieve long-term memories relevant to this query
    relevant_memories = retrieve_relevant_memories(query)
    user_facts = get_user_facts()
    memory_context = user_facts + relevant_memories

    initial_state: AgentState = {
        "messages": [],
        "query": query,
        "sub_queries": [],
        "retrieved_context": [],
        "web_results": [],
        "agent_called": "",
        "final_answer": "",
        "next": "",
        "session_id": session_id,
        "memory_context": memory_context,
    }

    final_state = orchestrator.invoke(initial_state)

    answer = final_state["final_answer"]
    messages = final_state.get("messages", [])

    # Append this turn's Q&A to the messages list for summarization
    from langchain.schema import HumanMessage as HM, AIMessage as AM
    messages = messages + [HM(content=query), AM(content=answer)]

    # Save interaction to long-term ChromaDB memory
    save_interaction(session_id=session_id, query=query, answer=answer)

    # Summarize messages if over threshold (side-effect: trimmed history for next caller)
    maybe_summarize(messages)

    return {
        "answer": answer,
        "agent_called": final_state["agent_called"],
        "retrieved_context": final_state["retrieved_context"],
        "web_results": final_state["web_results"],
        "session_id": session_id,
    }
