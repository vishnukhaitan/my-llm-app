import os

from dotenv import load_dotenv

load_dotenv()

_openai_api_key = os.getenv("OPENAI_API_KEY")
_openai_model = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
APP_ENV = os.getenv("APP_ENV", "development")
APP_NAME = os.getenv("APP_NAME", "my-llm-app")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2000"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

if not _openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not configured."
    )

if not _openai_model:
    raise RuntimeError(
        "OPENAI_MODEL environment variable is not configured."
    )

OPENAI_API_KEY: str = _openai_api_key
OPENAI_MODEL: str = _openai_model
