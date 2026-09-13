from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.cost import compute_cost_usd
from src.fake_llm import FakeLLMError, fake_ask_llm
from src.main import ask_llm
from src.models import Answer, Question
from src.settings import Settings


def _settings(**kwargs: object) -> Settings:
    defaults: dict[str, object] = {
        "use_fake": True,
        "openai_api_key": "test-key",
        "openai_model": "gpt-4o-mini",
        "max_retries": 2,
        "retry_delay_s": 0.0,
        "fail_rate": 0.0,
    }
    defaults.update(kwargs)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_fake_ask_llm_does_not_need_api() -> None:
    question = Question(question="What is RAG?")
    answer = fake_ask_llm(question)
    assert isinstance(answer, Answer)
    assert answer.content.startswith("[FAKE]")
    assert "What is RAG?" in answer.content


def test_ask_llm_uses_fake_when_flag_set() -> None:
    answer = ask_llm(Question(question="Explain RAG"), settings=_settings())
    assert answer.content == "[FAKE] Explain RAG"
    assert answer.cost_usd == 0.0
    assert answer.retries == 0


def test_fake_fail_rate_always_raises() -> None:
    with pytest.raises(FakeLLMError):
        fake_ask_llm(Question(question="x"), fail_rate=1.0)


def test_ask_llm_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(question: Question, fail_rate: float = 0.0) -> Answer:
        calls["n"] += 1
        if calls["n"] == 1:
            raise FakeLLMError("boom")
        return Answer(content="[FAKE] ok")

    monkeypatch.setattr("src.main.fake_ask_llm", flaky)
    answer = ask_llm(Question(question="x"), settings=_settings())
    assert answer.content == "[FAKE] ok"
    assert answer.retries == 1
    assert calls["n"] == 2


def test_ask_llm_exhausts_fake_retries() -> None:
    with pytest.raises(FakeLLMError):
        ask_llm(
            Question(question="x"),
            settings=_settings(fail_rate=1.0, max_retries=1),
        )


def test_real_path_sets_cost_from_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=usage,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    monkeypatch.setattr("src.main._make_client", lambda settings: mock_client)

    answer = ask_llm(
        Question(question="q"),
        settings=_settings(use_fake=False),
    )
    assert answer.content == "hello"
    assert answer.retries == 0
    assert answer.cost_usd == pytest.approx(
        compute_cost_usd("gpt-4o-mini", 100, 50)
    )
