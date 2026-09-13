"""Offline stand-in for the real LLM. Used when USE_FAKE=true."""

import random

from src.models import Answer, Question


class FakeLLMError(Exception):
    """Simulated transient API failure for retry practice."""


def fake_ask_llm(question: Question, fail_rate: float = 0.0) -> Answer:
    """Return a canned Answer. Does not call the network.

    Raises FakeLLMError with probability fail_rate so ask_llm can retry.
    """
    if random.random() < fail_rate:
        raise FakeLLMError(f"simulated transient failure for: {question.question[:40]}")
    return Answer(content=f"[FAKE] {question.question}")
