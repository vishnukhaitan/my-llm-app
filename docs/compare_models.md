# Compare models (`scripts/compare_models.py`)

Runs the **same questions** through **two (or more) models**, saves every `Answer` to SQLite, and prints cost/time totals. This is the last Week 4 lab piece.

**Related files**

- Script: `scripts/compare_models.py`
- Questions: `data/questions.csv`
- Reuses: `ask_llm`, `persist_answer`, `Settings`, `src/store.py`
- Tests: `tests/test_compare_models.py`

## Why

`gpt-4o` is usually better and **more expensive** than `gpt-4o-mini`. The script gives numbers you can defend: same prompts, same settings, different `OPENAI_MODEL`.

## Usage

```bash
# wiring only — no Vocareum spend
USE_FAKE=true python scripts/compare_models.py

python scripts/compare_models.py \
  --models gpt-4o-mini,gpt-4o \
  --questions data/questions.csv \
  --db data/answers.db
```

Default CSV has **3** questions (6 real API calls if you compare two models). Add rows to spend more. Fake mode: `cost_usd` stays `0.0` for every model, so the $ ratio is not meaningful until `USE_FAKE=false`.

Inspect:

```bash
sqlite3 data/answers.db \
  "SELECT model, COUNT(*), ROUND(SUM(cost_usd), 6) FROM answers GROUP BY model;"
```

## Tiktoken (optional)

The demo listed `tiktoken` for **pre-call** token counts. This script uses **`response.usage`** after the call (already in `ask_llm`). You do not need tiktoken installed for Step 10.

Add it later if you want to estimate tokens before sending a prompt.

## Not in this step

Docker, RAG retrieval, Streamlit UI. Those were “later” in the original demo, not this baseline.
