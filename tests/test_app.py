import importlib

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "2000")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")


def test_config_loads_required_values():
    from src import config

    importlib.reload(config)
    assert config.OPENAI_API_KEY == "test-key"
    assert config.OPENAI_MODEL == "gpt-4o-mini"
    assert config.APP_ENV == "development"
    assert config.LOG_LEVEL == "INFO"
    assert config.LLM_TEMPERATURE == 0.2
    assert config.LLM_MAX_OUTPUT_TOKENS == 2000
    assert config.OPENAI_BASE_URL == "https://openai.vocareum.com/v1"


def test_config_requires_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    from src import config

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        importlib.reload(config)


def test_config_requires_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "")
    from src import config

    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        importlib.reload(config)
