"""
Long-term memory: two stores.

1. ChromaDB memory store — stores past Q&A interactions as embeddings.
   Uses a separate collection called "memory".
   - save_interaction(): write a Q&A pair to ChromaDB
   - retrieve_relevant_memories(): retrieve top-k relevant past interactions

2. SQLite store — structured user facts extracted by Gemini after each session.
   - extract_and_save_facts(): use Gemini to extract facts, store in SQLite
   - get_user_facts(): load all stored facts for injection at session start
"""

import json
import uuid
from datetime import datetime
from typing import Optional

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Text, DateTime

from config import settings

# ---------------------------------------------------------------------------
# ChromaDB memory store
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.ClientAPI] = None
_memory_collection: Optional[chromadb.Collection] = None


def _get_memory_collection() -> chromadb.Collection:
    global _chroma_client, _memory_collection
    if _memory_collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _memory_collection = _chroma_client.get_or_create_collection(
            name="memory",
            metadata={"hnsw:space": "cosine"},
        )
    return _memory_collection


def save_interaction(session_id: str, query: str, answer: str) -> None:
    """
    Save a Q&A interaction to the ChromaDB memory collection.
    The document stored is the combined query + answer for semantic retrieval.
    """
    collection = _get_memory_collection()
    doc_id = str(uuid.uuid4())
    document = f"Q: {query}\nA: {answer}"
    collection.add(
        documents=[document],
        metadatas=[{
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat(),
        }],
        ids=[doc_id],
    )


def retrieve_relevant_memories(query: str, top_k: int = 3) -> list[str]:
    """
    Retrieve the top-k most relevant past interactions from ChromaDB memory.
    Returns a list of formatted strings: "Past interaction: Q: ... A: ..."
    """
    collection = _get_memory_collection()

    # If collection is empty, return early
    count = collection.count()
    if count == 0:
        return []

    actual_k = min(top_k, count)
    results = collection.query(
        query_texts=[query],
        n_results=actual_k,
    )

    memories = []
    if results and results["documents"]:
        for doc in results["documents"][0]:
            memories.append(f"Past interaction: {doc}")

    return memories


# ---------------------------------------------------------------------------
# SQLite store for structured user facts
# ---------------------------------------------------------------------------

_DB_URL = "sqlite:///./memory.db"
_engine = None
_metadata = MetaData()

_user_facts_table = Table(
    "user_facts",
    _metadata,
    Column("id", String(36), primary_key=True),
    Column("session_id", String(36), nullable=False),
    Column("fact", Text, nullable=False),
    Column("created_at", String(50), nullable=False),
)


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
        _metadata.create_all(_engine)
    return _engine


def extract_and_save_facts(session_id: str, messages: list[BaseMessage]) -> None:
    """
    Use Gemini to extract structured facts from the session's messages,
    then store them in SQLite.

    Facts are things like:
      - "User is building a fintech app"
      - "User prefers concise answers"
      - "User is using Python 3.12"
    """
    if not messages:
        return

    # Build transcript
    transcript_lines = []
    for msg in messages:
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

    extraction_prompt = [
        SystemMessage(
            content=(
                "You are a fact extractor. Given a conversation transcript, extract "
                "concise, reusable facts about the user — things that would be useful "
                "to remember across future sessions. Focus on:\n"
                "- What the user is building or working on\n"
                "- User preferences (e.g., concise vs. detailed answers)\n"
                "- Technical stack, tools, or constraints they mentioned\n"
                "- Goals or context about their project\n\n"
                "Return ONLY a JSON array of short fact strings. Example:\n"
                '["User is building a fintech app", "User prefers concise answers"]\n\n'
                "If there are no noteworthy facts, return an empty array: []"
            )
        ),
        HumanMessage(
            content=f"Conversation transcript:\n\n{transcript}"
        ),
    ]

    response = llm.invoke(extraction_prompt)
    content = response.content.strip()

    # Parse JSON — strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    try:
        facts = json.loads(content)
        if not isinstance(facts, list):
            facts = []
    except (json.JSONDecodeError, ValueError):
        facts = []

    if not facts:
        return

    engine = _get_engine()
    with engine.begin() as conn:
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                conn.execute(
                    _user_facts_table.insert().values(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        fact=fact.strip(),
                        created_at=datetime.utcnow().isoformat(),
                    )
                )


def get_user_facts() -> list[str]:
    """
    Load all stored user facts from SQLite.
    Returns a list of fact strings.
    """
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            _user_facts_table.select().order_by(_user_facts_table.c.created_at)
        ).fetchall()

    return [row.fact for row in rows]
