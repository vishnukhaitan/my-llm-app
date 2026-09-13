from src.fake_llm import fake_ask_llm
from src.main import ask_llm
from src.models import Answer, Question
from src.settings import Settings


def test_fake_ask_llm_does_not_need_api() -> None:
    question = Question(question="What is RAG?")
    answer = fake_ask_llm(question)
    assert isinstance(answer, Answer)
    assert answer.content.startswith("[FAKE]")
    assert "What is RAG?" in answer.content


def test_ask_llm_uses_fake_when_flag_set() -> None:
    settings = Settings(use_fake=True, openai_api_key="", openai_model="gpt-4o-mini")
    answer = ask_llm(Question(question="Explain RAG"), settings=settings)
    assert answer.content == "[FAKE] Explain RAG"
