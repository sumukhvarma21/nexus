from typing import TypedDict

from langchain.schema import BaseMessage


class AgentState(TypedDict):
    messages: list[BaseMessage]   # full conversation history
    query: str                    # current user query
    sub_queries: list[str]        # decomposed sub-questions (populated by rag_agent multi_step)
    retrieved_context: list[str]  # RAG chunks as strings
    web_results: list[str]        # Tavily search result snippets
    agent_called: str             # which agent handled this turn
    final_answer: str             # synthesized response
    next: str                     # routing decision from supervisor
    session_id: str               # UUID for this conversation
    memory_context: list[str]     # injected long-term memories
