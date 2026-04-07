"""
Tests for Phase 4 Memory implementation.

Unit tests:
  - maybe_summarize returns fewer messages when over threshold
  - saving and retrieving from ChromaDB memory store

Integration tests (marked @pytest.mark.integration):
  - Full two-session test: save a fact in session 1, retrieve it in session 2
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch

from langchain.schema import HumanMessage, AIMessage, SystemMessage


# ---------------------------------------------------------------------------
# Unit tests: short_term.maybe_summarize
# ---------------------------------------------------------------------------

class TestMaybeSummarize:
    """Tests for memory.short_term.maybe_summarize"""

    def _make_messages(self, n: int) -> list:
        """Create n alternating Human/AI messages."""
        messages = []
        for i in range(n):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User message {i}"))
            else:
                messages.append(AIMessage(content=f"AI response {i}"))
        return messages

    def test_no_summarize_below_threshold(self):
        """Messages at or below SUMMARY_THRESHOLD are returned unchanged."""
        from memory.short_term import maybe_summarize, SUMMARY_THRESHOLD

        messages = self._make_messages(SUMMARY_THRESHOLD)
        result = maybe_summarize(messages)
        assert result == messages
        assert len(result) == SUMMARY_THRESHOLD

    def test_no_summarize_at_threshold(self):
        """Messages exactly at SUMMARY_THRESHOLD are returned unchanged."""
        from memory.short_term import maybe_summarize, SUMMARY_THRESHOLD

        messages = self._make_messages(SUMMARY_THRESHOLD)
        result = maybe_summarize(messages)
        assert len(result) == len(messages)

    def test_summarize_above_threshold(self):
        """Messages exceeding SUMMARY_THRESHOLD are summarized."""
        from memory.short_term import maybe_summarize, SUMMARY_THRESHOLD, RECENT_KEEP

        messages = self._make_messages(SUMMARY_THRESHOLD + 5)

        # Mock the LLM call
        mock_response = MagicMock()
        mock_response.content = "Summary of older conversation."

        with patch("memory.short_term.ChatGoogleGenerativeAI") as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm_instance

            result = maybe_summarize(messages)

        # Result should be fewer messages than original
        assert len(result) < len(messages)

        # First message should be a SystemMessage (summary)
        assert isinstance(result[0], SystemMessage)
        assert "Summary of older conversation." in result[0].content

        # Last RECENT_KEEP messages should be the originals
        assert result[1:] == messages[-RECENT_KEEP:]

    def test_summarize_returns_correct_count(self):
        """Summarized result has exactly RECENT_KEEP + 1 messages."""
        from memory.short_term import maybe_summarize, SUMMARY_THRESHOLD, RECENT_KEEP

        # Create well above threshold
        messages = self._make_messages(SUMMARY_THRESHOLD + 10)

        mock_response = MagicMock()
        mock_response.content = "Condensed summary."

        with patch("memory.short_term.ChatGoogleGenerativeAI") as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm_instance

            result = maybe_summarize(messages)

        # 1 summary SystemMessage + RECENT_KEEP verbatim messages
        assert len(result) == RECENT_KEEP + 1

    def test_recent_messages_preserved_verbatim(self):
        """The last RECENT_KEEP messages are preserved exactly."""
        from memory.short_term import maybe_summarize, SUMMARY_THRESHOLD, RECENT_KEEP

        messages = self._make_messages(SUMMARY_THRESHOLD + 3)

        mock_response = MagicMock()
        mock_response.content = "Some summary."

        with patch("memory.short_term.ChatGoogleGenerativeAI") as MockLLM:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_response
            MockLLM.return_value = mock_llm_instance

            result = maybe_summarize(messages)

        expected_recent = messages[-RECENT_KEEP:]
        actual_recent = result[1:]  # skip the summary SystemMessage
        assert actual_recent == expected_recent


# ---------------------------------------------------------------------------
# Unit tests: long_term ChromaDB memory store
# ---------------------------------------------------------------------------

class TestChromaMemoryStore:
    """Tests for memory.long_term ChromaDB functions."""

    def test_save_and_retrieve_interaction(self, tmp_path, monkeypatch):
        """Saving an interaction allows it to be retrieved."""
        import memory.long_term as lt

        # Use a temp ChromaDB path
        monkeypatch.setattr("memory.long_term._chroma_client", None)
        monkeypatch.setattr("memory.long_term._memory_collection", None)
        monkeypatch.setattr(
            "memory.long_term.settings",
            type("S", (), {"chroma_persist_dir": str(tmp_path)})(),
        )

        session_id = str(uuid.uuid4())
        query = "What is LangChain?"
        answer = "LangChain is a framework for building LLM applications."

        lt.save_interaction(session_id=session_id, query=query, answer=answer)

        memories = lt.retrieve_relevant_memories("LangChain framework")

        assert len(memories) > 0
        # The retrieved memory should contain the original Q&A content
        assert any("LangChain" in m for m in memories)

    def test_retrieve_empty_collection(self, tmp_path, monkeypatch):
        """Retrieving from empty collection returns empty list."""
        import memory.long_term as lt

        monkeypatch.setattr("memory.long_term._chroma_client", None)
        monkeypatch.setattr("memory.long_term._memory_collection", None)
        monkeypatch.setattr(
            "memory.long_term.settings",
            type("S", (), {"chroma_persist_dir": str(tmp_path / "empty")})(),
        )

        memories = lt.retrieve_relevant_memories("anything")
        assert memories == []

    def test_save_multiple_retrieve_top_k(self, tmp_path, monkeypatch):
        """top_k limits the number of results returned."""
        import memory.long_term as lt

        monkeypatch.setattr("memory.long_term._chroma_client", None)
        monkeypatch.setattr("memory.long_term._memory_collection", None)
        monkeypatch.setattr(
            "memory.long_term.settings",
            type("S", (), {"chroma_persist_dir": str(tmp_path / "multi")})(),
        )

        session_id = str(uuid.uuid4())
        for i in range(5):
            lt.save_interaction(
                session_id=session_id,
                query=f"Question {i} about Python",
                answer=f"Answer {i} about Python programming.",
            )

        memories = lt.retrieve_relevant_memories("Python", top_k=2)
        assert len(memories) <= 2


# ---------------------------------------------------------------------------
# Integration tests: two-session fact persistence
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTwoSessionIntegration:
    """
    Full integration test: facts saved in session 1 are retrieved in session 2.
    Requires a real Gemini API key.
    """

    def test_facts_persist_across_sessions(self, tmp_path, monkeypatch):
        """
        Session 1: extract and save facts from messages.
        Session 2: get_user_facts() returns the facts from session 1.
        """
        import memory.long_term as lt

        # Use a temporary SQLite DB for isolation
        temp_db = f"sqlite:///{tmp_path}/test_memory.db"
        monkeypatch.setattr("memory.long_term._DB_URL", temp_db)
        monkeypatch.setattr("memory.long_term._engine", None)

        # Re-bind the metadata to fresh tables for this test
        from sqlalchemy import MetaData, Table, Column, String, Text
        test_meta = MetaData()
        test_table = Table(
            "user_facts",
            test_meta,
            Column("id", String(36), primary_key=True),
            Column("session_id", String(36), nullable=False),
            Column("fact", Text, nullable=False),
            Column("created_at", String(50), nullable=False),
        )
        monkeypatch.setattr("memory.long_term._metadata", test_meta)
        monkeypatch.setattr("memory.long_term._user_facts_table", test_table)

        # Session 1: simulate a conversation about a fintech app
        session_1_id = str(uuid.uuid4())
        session_1_messages = [
            HumanMessage(content="I'm building a fintech app using Python and FastAPI."),
            AIMessage(content="That sounds great! What kind of financial data are you working with?"),
            HumanMessage(content="I'm processing payment transactions. I prefer concise answers."),
            AIMessage(content="Understood. I'll keep my responses brief and to the point."),
        ]

        # Extract and save facts from session 1
        lt.extract_and_save_facts(
            session_id=session_1_id,
            messages=session_1_messages,
        )

        # Session 2: retrieve facts — should include what was learned in session 1
        session_2_facts = lt.get_user_facts()

        assert len(session_2_facts) > 0, "Expected facts to be saved from session 1"

        # Check that relevant facts were captured
        all_facts_text = " ".join(session_2_facts).lower()
        # At least one of these topics should appear
        relevant_keywords = ["fintech", "python", "fastapi", "payment", "concise"]
        assert any(kw in all_facts_text for kw in relevant_keywords), (
            f"Expected at least one of {relevant_keywords} in facts, got: {session_2_facts}"
        )
