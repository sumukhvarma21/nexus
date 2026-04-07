"""
Tests for the Phase 3 LangGraph orchestrator.

Unit tests mock the LLM and retrieval.
Integration tests run against the real Gemini API.
"""

import pytest
from unittest.mock import patch, MagicMock

from agents.state import AgentState
from agents.orchestrator import supervisor, route


# ---------------------------------------------------------------------------
# Unit tests — no LLM or ChromaDB
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "messages": [],
        "query": "test query",
        "sub_queries": [],
        "retrieved_context": [],
        "web_results": [],
        "agent_called": "",
        "final_answer": "",
        "next": "",
    }
    state.update(overrides)
    return state


def test_route_returns_next_field():
    state = _base_state(next="rag_agent")
    assert route(state) == "rag_agent"


def test_route_web_search():
    state = _base_state(next="web_search_agent")
    assert route(state) == "web_search_agent"


def test_supervisor_sets_next_field():
    """Supervisor must write a valid agent name into state['next']."""
    from pydantic import BaseModel
    from typing import Literal

    class FakeDecision(BaseModel):
        agent: Literal["rag_agent", "web_search_agent"]
        reasoning: str

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = FakeDecision(
        agent="rag_agent", reasoning="Query is about documents."
    )

    with patch("agents.orchestrator.ChatGoogleGenerativeAI", return_value=mock_llm):
        state = _base_state(query="What does the contract say about payment?")
        result = supervisor(state)

    assert result["next"] == "rag_agent"


# ---------------------------------------------------------------------------
# Integration tests — hit real Gemini API
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_supervisor_routes_doc_query_to_rag():
    """A question about uploaded documents should route to rag_agent."""
    from agents.orchestrator import supervisor
    state = _base_state(query="What does the employee handbook say about parental leave?")
    result = supervisor(state)
    assert result["next"] == "rag_agent"


@pytest.mark.integration
def test_supervisor_routes_news_query_to_web():
    """A question about current events should route to web_search_agent."""
    from agents.orchestrator import supervisor
    state = _base_state(query="What is the latest news about AI regulation in Europe?")
    result = supervisor(state)
    assert result["next"] == "web_search_agent"


@pytest.mark.integration
def test_run_chat_rag_path():
    """End-to-end: query about documents flows through rag_agent."""
    from agents.orchestrator import run_chat
    result = run_chat("What is the payment schedule in the service contract?")
    assert result["agent_called"] == "rag_agent"
    assert isinstance(result["answer"], str) and len(result["answer"]) > 10
