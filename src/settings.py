"""Typed runtime configuration loaded from environment variables or `.env`."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Field names map to env vars (e.g. openai_api_key → OPENAI_API_KEY)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    use_fake: bool = False
    fail_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_delay_s: float = Field(default=0.5, ge=0.0, le=10.0)
    app_env: str = "development"
    app_name: str = "my-llm-app"
    log_level: str = "INFO"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=2000, gt=0)
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, le=65535)

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def empty_base_url_is_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def require_credentials_unless_fake(self) -> Self:
        if self.use_fake:
            return self
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when USE_FAKE is false")
        if not self.openai_model:
            raise ValueError("OPENAI_MODEL is required when USE_FAKE is false")
        return self


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process. Tests should construct Settings() directly."""
    return Settings()
