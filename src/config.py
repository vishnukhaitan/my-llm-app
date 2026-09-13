"""Back-compat exports. New code should import Settings from src.settings."""

from src.settings import Settings

_settings = Settings()

OPENAI_API_KEY = _settings.openai_api_key
OPENAI_MODEL = _settings.openai_model
OPENAI_BASE_URL = _settings.openai_base_url
APP_ENV = _settings.app_env
APP_NAME = _settings.app_name
LOG_LEVEL = _settings.log_level
LLM_TEMPERATURE = _settings.llm_temperature
LLM_MAX_OUTPUT_TOKENS = _settings.llm_max_output_tokens
HOST = _settings.host
PORT = _settings.port
