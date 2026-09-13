# Fake LLM (`src/fake_llm.py`)

Offline stand-in for Vocareum / OpenAI. Tests and local runs can use it without a network call or a real API key.

**Related files**

- Code: `src/fake_llm.py`
- Wired in: `src/main.py` (`ask_llm`)
- Flag: `USE_FAKE` on `Settings` (`src/settings.py`)
- Tests: `tests/test_fake_llm.py`

## Why it exists

`ask_llm` with the real client needs `OPENAI_API_KEY` and HTTP. Pytest should not depend on that. `USE_FAKE=true` returns a canned `Answer` immediately.

## How to turn it on

`.env` or the shell:

```bash
USE_FAKE=true
```

That maps to `Settings.use_fake` (boolean). `true` / `false` / `1` / `0` all work via pydantic-settings.

When `use_fake` is true, `OPENAI_API_KEY` may be empty. When it is false, the key and model are still required.

## What the fake returns

```python
Answer(content=f"[FAKE] {question.question}")
```

Same `Answer` type as the real path. Fake `cost_usd` stays `0.0`. With probability `fail_rate` it raises `FakeLLMError` so `ask_llm` can retry (see [ask_llm.md](ask_llm.md)).

Pass `settings=` in tests so you do not rely on cached `get_settings()` or `.env`.

## Commands

```bash
# no API call
USE_FAKE=true python -m src.main
pytest

# real Vocareum / OpenAI (needs key in .env)
USE_FAKE=false python -m src.main
```

Pytest does not need `USE_FAKE` in the shell: tests construct `Settings(use_fake=True, ...)`.
