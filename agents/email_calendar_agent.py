"""
Email/Calendar Agent — LangGraph node for handling email and calendar queries.

Currently a stub that returns configuration instructions.

TODO: Real implementation
    The production implementation should connect to Google's MCP servers via the
    MCP protocol. Steps:
    1. Spin up (or connect to) a Google Workspace MCP server that exposes tools
       like `gmail_read`, `gmail_send`, `calendar_list_events`, etc.
    2. Create an MCP client session (e.g., via `mcp.ClientSession`) pointing at
       that server's stdio or SSE transport.
    3. Retrieve available tools with `session.list_tools()` and wrap them as
       LangChain tools using `langchain_mcp_adapters` or a manual wrapper.
    4. Build a LangGraph ReAct agent (or ToolNode) around those tools.
    5. Replace the stub return below with `agent_executor.invoke(state)`.
"""

from agents.state import AgentState


def email_calendar_agent(state: AgentState) -> AgentState:
    """
    Handles queries about emails and calendar events.
    Currently a stub — returns instructions on how to configure Google MCP.
    """
    stub_message = (
        "The Email/Calendar agent is not yet configured. "
        "To enable this feature, connect a Google Workspace MCP server "
        "that exposes Gmail and Google Calendar tools, then update "
        "agents/email_calendar_agent.py with the MCP client integration. "
        "See the TODO comment in that file for detailed steps."
    )

    return {
        **state,
        "agent_called": "email_calendar_agent",
        "final_answer": stub_message,
    }
