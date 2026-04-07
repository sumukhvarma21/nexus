"""
Web Search Agent — LangGraph node that uses Tavily to search the web.

Routing hint: called when the query needs current or external information
not present in uploaded documents (news, prices, recent events, etc.).
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

from config import settings
from agents.state import AgentState


def web_search_agent(state: AgentState) -> AgentState:
    """Search the web with Tavily and synthesize an answer."""
    query = state["query"]

    # Import here so missing tavily-python gives a clear error only when used
    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            **state,
            "web_results": [],
            "agent_called": "web_search_agent",
            "final_answer": "Web search is unavailable: tavily-python is not installed.",
        }

    if not settings.tavily_api_key:
        return {
            **state,
            "web_results": [],
            "agent_called": "web_search_agent",
            "final_answer": "Web search is unavailable: TAVILY_API_KEY is not configured.",
        }

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query, max_results=5)

    snippets = [
        f"{r['title']}: {r['content']}"
        for r in response.get("results", [])
    ]

    if not snippets:
        return {
            **state,
            "web_results": [],
            "agent_called": "web_search_agent",
            "final_answer": "Web search returned no results.",
        }

    context = "\n\n".join(snippets)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using the "
                "web search results provided. Cite sources where relevant."
            )
        ),
        HumanMessage(content=f"Search results:\n{context}\n\nQuestion: {query}"),
    ]

    llm_response = llm.invoke(messages)

    return {
        **state,
        "web_results": snippets,
        "agent_called": "web_search_agent",
        "final_answer": llm_response.content,
    }
