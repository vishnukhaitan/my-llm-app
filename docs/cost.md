# Cost (`src/cost.py`)

Turns **token counts** into **USD**. This step is a pure function: no API, no `.env`, no `ask_llm` wiring yet.

**Related files**

- Code: `src/cost.py`
- Tests: `tests/test_cost.py`

Step 5 will pass `response.usage.prompt_tokens` and `completion_tokens` into `compute_cost_usd` and set `Answer.cost_usd`.

## Formula

Rates are **USD per 1 million tokens**, stored as `(input_rate, output_rate)`:

```text
cost = (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000
```

Example for `gpt-4o-mini` (`0.15` in, `0.60` out), 100 prompt + 50 completion tokens:

```text
(100 * 0.15 + 50 * 0.60) / 1_000_000 = 0.000045
```

## `RATES`

| Model | Input / 1M | Output / 1M |
|---|---|---|
| `gpt-4o-mini` | 0.15 | 0.60 |
| `gpt-4o` | 2.50 | 10.00 |
| `llama3.2:3b` | 0.0 | 0.0 |

These are order-of-magnitude figures from the demo, not a live price feed. Check the provider’s pricing before quoting in production.

Unknown model names return **`0.0`** instead of raising, so a batch job is not killed by one missing rate.

## Usage

```python
from src.cost import compute_cost_usd

cost = compute_cost_usd("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
```

## Not in this step

Tool-calling is still later. Cost is applied in `ask_llm` on the real path; see [ask_llm.md](ask_llm.md).
