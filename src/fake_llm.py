"""Offline stand-in for the real LLM. Used when USE_FAKE=true."""

from src.models import Answer, Question


def fake_ask_llm(question: Question) -> Answer:
    """Return a canned Answer. Does not call the network."""
    return Answer(content=f"[FAKE] {question.question}")
