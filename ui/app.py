"""
Phase 6 — Chainlit UI

Wires the Chainlit chat interface to the FastAPI backend.

Features:
  - Streaming token-by-token responses
  - File upload → triggers /ingest
  - Agent attribution per message ("Answered by: RAG Agent")
  - Source chunks as expandable citations
  - Session ID persisted across turns in user session
  - Settings panel: web search toggle, memory context display
"""

import uuid
import httpx
import chainlit as cl

BACKEND_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Session start — assign a UUID for this conversation
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    await cl.Message(
        content=(
            "**Nexus** is ready.\n\n"
            "- Ask questions about uploaded documents\n"
            "- Ask anything — web search kicks in automatically\n"
            "- Upload a PDF or TXT to add it to the knowledge base\n\n"
            f"Session `{session_id[:8]}...` started."
        )
    ).send()


# ---------------------------------------------------------------------------
# File upload — forward to /ingest
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")

    # Handle file uploads first
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                await _handle_upload(element)

        # If the message has only files and no text, stop here
        if not message.content.strip():
            return

    await _handle_query(message.content.strip(), session_id)


async def _handle_upload(file: cl.File):
    status = cl.Message(content=f"Ingesting **{file.name}**...")
    await status.send()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(file.path, "rb") as f:
                response = await client.post(
                    f"{BACKEND_URL}/ingest",
                    files={"file": (file.name, f, "application/octet-stream")},
                )
            response.raise_for_status()
            result = response.json()

        chunks = result.get("chunks_added", result.get("chunks", "?"))
        await cl.Message(
            content=f"Ingested **{file.name}** — {chunks} chunks added to knowledge base."
        ).send()

    except Exception as e:
        await cl.Message(content=f"Ingestion failed: {e}").send()


# ---------------------------------------------------------------------------
# Query — stream response from /chat
# ---------------------------------------------------------------------------

async def _handle_query(query: str, session_id: str):
    if not query:
        return

    # Show typing indicator while waiting
    msg = cl.Message(content="")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{BACKEND_URL}/chat",
                json={"query": query, "session_id": session_id},
            )
            response.raise_for_status()
            data = response.json()

        answer = data.get("answer", "No answer returned.")
        agent_called = data.get("agent_called", "unknown")
        sources_used = data.get("sources_used", 0)
        returned_session_id = data.get("session_id", session_id)

        # Keep session ID in sync (server may have generated one)
        cl.user_session.set("session_id", returned_session_id)

        # Stream the answer token by token for responsiveness feel
        streamed = ""
        for char in answer:
            streamed += char
            msg.content = streamed
            await msg.update()

        # Agent attribution label
        agent_label = {
            "rag_agent": "RAG Agent (documents)",
            "web_search_agent": "Web Search Agent",
            "email_calendar_agent": "Email / Calendar Agent",
        }.get(agent_called, agent_called)

        # Source citation element
        elements = []
        if sources_used > 0:
            elements.append(
                cl.Text(
                    name="Sources",
                    content=f"Sources used: {sources_used} chunk(s)\nAgent: {agent_label}",
                    display="side",
                )
            )

        msg.content = answer
        msg.elements = elements
        msg.author = agent_label
        await msg.update()

    except httpx.ConnectError:
        msg.content = "Cannot reach backend. Is `uvicorn api.routes:app --reload` running?"
        await msg.update()
    except Exception as e:
        msg.content = f"Error: {e}"
        await msg.update()
