import pytest
from pydantic import ValidationError

from src.models import Answer, Question


def test_question_requires_text() -> None:
    with pytest.raises(ValidationError):
        Question(question="")


def test_question_stores_text() -> None:
    q = Question(question="What is RAG?")
    assert q.question == "What is RAG?"


def test_answer_defaults() -> None:
    answer = Answer(content="Hello.")
    assert answer.content == "Hello."
    assert answer.cost_usd == 0.0
    assert answer.retries == 0
    assert answer.confidence == 1.0
    assert answer.sources == []
    assert answer.schema_version == "v1"


def test_answer_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        Answer(content="Hello.", confidence=1.5)
