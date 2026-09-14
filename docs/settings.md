# Settings (`src/settings.py`)

Typed runtime configuration. Secrets and environment-specific values live in `.env`, not in Python source.

**Related files**

- Code: `src/settings.py`
- Used by: `src/main.py`, `src/config.py`
- Template (safe to commit): `.env.example`
- Secrets (never commit): `.env`
- Tests: `tests/test_app.py`

## Purpose

`Settings` is a Pydantic model. When you construct it, **pydantic-settings** fills each field from:

1. Arguments passed in code (if any)
2. Process environment variables (`export OPENAI_API_KEY=...`)
3. The `.env` file
4. Defaults declared on the class

`get_settings()` wraps `Settings()` and caches the result so the process reads config once.

```python
from src.settings import get_settings

settings = get_settings()
# settings.openai_api_key, settings.openai_model, ...
```

Tests should call `Settings()` directly so each test can set different env vars. Cached `get_settings()` would reuse the first load.

## How `.env` is loaded

```python
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)
```

| Option | Meaning |
|---|---|
| `env_file=".env"` | Also read `.env` from the current working directory (project root when you run `python -m src.main`) |
| `env_file_encoding="utf-8"` | File encoding |
| `case_sensitive=False` | `OPENAI_API_KEY` matches the field even if casing differs |
| `extra="ignore"` | Unknown keys in `.env` (for example `OPENAI_WEBHOOK_SECRET`) are ignored |

Run commands from the **project root** so `.env` is found.

## Name mapping

Pydantic-settings maps a **snake_case** field to an **UPPER_SNAKE** environment variable:

`openai_api_key` → `OPENAI_API_KEY`

Values are coerced to the field type: `"0.2"` becomes `float`, `"2000"` becomes `int`.

| Field | Env var | Required | Default | Notes |
|---|---|---|---|---|
| `openai_api_key` | `OPENAI_API_KEY` | Yes, unless `USE_FAKE=true` | `""` | Never hard-code in source |
| `openai_model` | `OPENAI_MODEL` | Yes, unless `USE_FAKE=true` | `gpt-4o-mini` | |
| `openai_base_url` | `OPENAI_BASE_URL` | No | `None` | Vocareum: `https://openai.vocareum.com/v1`. Empty string becomes `None` |
| `use_fake` | `USE_FAKE` | No | `false` | Skip the real API; see [fake_llm.md](fake_llm.md) |
| `fail_rate` | `FAIL_RATE` | No | `0.0` | Probability the fake raises `FakeLLMError` |
| `max_retries` | `MAX_RETRIES` | No | `2` | Extra attempts after the first |
| `retry_delay_s` | `RETRY_DELAY_S` | No | `0.5` | Base delay; doubles each attempt |
| `app_env` | `APP_ENV` | No | `development` | |
| `app_name` | `APP_NAME` | No | `my-llm-app` | |
| `log_level` | `LOG_LEVEL` | No | `INFO` | |
| `llm_temperature` | `LLM_TEMPERATURE` | No | `0.2` | Must be 0.0–2.0 |
| `llm_max_output_tokens` | `LLM_MAX_OUTPUT_TOKENS` | No | `2000` | Must be > 0 |
| `host` | `HOST` | No | `0.0.0.0` | For a future API server |
| `port` | `PORT` | No | `8000` | 1–65535 |
| `results_db` | `RESULTS_DB` | No | `data/answers.db` | SQLite file; see [store.md](store.md) |

Copy `.env.example` to `.env` and fill `OPENAI_API_KEY` and `OPENAI_MODEL`. Vocareum keys (`voc-...`) need `OPENAI_BASE_URL` set to the Vocareum gateway.

To see `_answer_from_response` step logs, set `LOG_LEVEL=DEBUG` then `python -m src.main`.

## Empty base URL

If `OPENAI_BASE_URL=` is blank, a validator converts it to `None`. Then `OpenAI()` uses the SDK default host (`https://api.openai.com/v1`).

If the value is a URL, it is passed through as `OpenAI(base_url=...)`.

## Failure behavior

Missing or empty `OPENAI_API_KEY` / `OPENAI_MODEL` raises `pydantic.ValidationError` when `USE_FAKE` is false. That is intentional: fail before an API call. Fake mode may omit the key.

## Inspect without printing the key

From the project root, with the venv active:

```bash
python -c "from src.settings import Settings; s=Settings(); print(s.openai_model, s.openai_base_url, bool(s.openai_api_key))"
```

You should see the model name, the gateway URL (if set), and `True` if a key is present.
