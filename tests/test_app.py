import pytest
from pydantic import ValidationError

from src.settings import Settings


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

    settings = Settings()

    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_output_tokens == 2000
    assert settings.openai_base_url == "https://openai.vocareum.com/v1"


def test_settings_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    with pytest.raises(ValidationError):
        Settings()


def test_settings_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "")

    with pytest.raises(ValidationError):
        Settings()


def test_empty_base_url_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    settings = Settings()

    assert settings.openai_base_url is None
