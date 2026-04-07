"""
Short-term memory: summarization buffer for within-session conversation history.

After SUMMARY_THRESHOLD messages, older messages are summarized into a single
SystemMessage using Gemini. The most recent RECENT_KEEP messages are kept verbatim.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from config import settings

SUMMARY_THRESHOLD = 10  # summarize when history exceeds this
RECENT_KEEP = 4         # always keep this many recent messages verbatim


def maybe_summarize(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    If messages exceed SUMMARY_THRESHOLD, summarize the older ones.
    Returns a new list: [SystemMessage(summary), ...recent messages]

    If messages are at or below the threshold, returns them unchanged.
    """
    if len(messages) <= SUMMARY_THRESHOLD:
        return messages

    # Split: older messages to summarize, recent to keep verbatim
    older = messages[:-RECENT_KEEP]
    recent = messages[-RECENT_KEEP:]

    # Build a transcript of the older messages for summarization
    transcript_lines = []
    for msg in older:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, SystemMessage):
            role = "System"
        else:
            role = type(msg).__name__
        transcript_lines.append(f"{role}: {msg.content}")

    transcript = "\n".join(transcript_lines)

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )

    summary_prompt = [
        SystemMessage(
            content=(
                "You are a conversation summarizer. Given the following conversation "
                "transcript, produce a concise summary that captures the key topics "
                "discussed, decisions made, and important context. Write in third person "
                "and be factual. The summary will be used as context for continuing the conversation."
            )
        ),
        HumanMessage(
            content=f"Conversation transcript to summarize:\n\n{transcript}"
        ),
    ]

    response = llm.invoke(summary_prompt)
    summary_text = f"[Conversation summary] {response.content}"

    return [SystemMessage(content=summary_text)] + recent
