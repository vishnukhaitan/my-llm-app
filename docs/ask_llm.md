# ask_llm retries and cost (`src/main.py`)

Step 5 connects `src/cost.py` to a real (or mocked) chat completion and retries transient failures.

**Related files**

- Code: `src/main.py` (`ask_llm`, `_call_real_llm`)
- Fake failures: `src/fake_llm.py` (`FakeLLMError`, `fail_rate`)
- Settings: `MAX_RETRIES`, `RETRY_DELAY_S`, `FAIL_RATE`
- Tests: `tests/test_fake_llm.py`

## Real path: cost from `usage`

After `chat.completions.create`, the SDK may include `response.usage`:

- `prompt_tokens` — input
- `completion_tokens` — output

Those go into `compute_cost_usd(model, prompt_tokens, completion_tokens)` and onto `Answer.cost_usd`. Missing usage is treated as 0 tokens → `$0`.

Fake answers still use `cost_usd=0.0` (no real tokens).

## Retries

`ask_llm` tries up to `max_retries + 1` times (default 2 retries → 3 attempts).

Retryable:

- `FakeLLMError` (offline simulation)
- `APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError`

Not retried (examples): bad request, invalid API key (`AuthenticationError`).

Delay between attempts: `retry_delay_s * 2**attempt` (exponential backoff). Tests set `retry_delay_s=0`.

`Answer.retries` is the **successful attempt index** (`0` = first try worked).

## Fake failures

`FAIL_RATE` / `fail_rate` is a probability in `[0, 1]`. `1.0` always raises `FakeLLMError`; `0.0` never does. Use this to practice retries without Vocareum.

```bash
USE_FAKE=true FAIL_RATE=0 python -m src.main
```

## Settings

| Field | Env | Default |
|---|---|---|
| `max_retries` | `MAX_RETRIES` | `2` |
| `retry_delay_s` | `RETRY_DELAY_S` | `0.5` |
| `fail_rate` | `FAIL_RATE` | `0.0` |

## Not in this step

Tool-calling (`confidence` / `sources` from the model) is Step 6. SQLite and FastAPI come after that.
