# Question and Answer (`src/models.py`)

Pydantic models for what goes **into** the LLM and what comes **out**. Settings (`src/settings.py`) stay environment/config. These models are the data contract for later pipeline, storage, and FastAPI steps.

**Related files**

- Code: `src/models.py`
- Used by: `src/main.py`
- Tests: `tests/test_models.py`

## `Question`

| Field | Type | Rules |
|---|---|---|
| `question` | `str` | Required, `min_length=1` |

Empty string is rejected at construction time.

```python
from src.models import Question

q = Question(question="What is RAG?")
```

## `Answer`

| Field | Type | Default | Later |
|---|---|---|---|
| `content` | `str` | required | LLM text |
| `cost_usd` | `float` | `0.0` | Real path: `compute_cost_usd` from `response.usage` |
| `retries` | `int` | `0` | Set by `ask_llm` after retries |
| `confidence` | `float` | `1.0` (0.0–1.0) | Real path: tool `answer_question` |
| `sources` | `list[str]` | `[]` | Real path: tool `answer_question` |
| `schema_version` | `str` | `"v1"` | Bump only on breaking changes |

You can build an answer with only `content`; the rest get defaults so older callers keep working when new fields are added.

```python
from src.models import Answer

answer = Answer(content="RAG retrieves documents, then the model answers.")
print(answer.content)
print(answer.model_dump())
```

## How `ask_llm` uses them

`src/main.py` no longer takes a raw `str` or returns a raw `str`:

```python
def ask_llm(question: Question) -> Answer:
    ...
    return Answer(content=content)
```

The CLI still prints `answer.content` so the smoke test looks the same:

```bash
python -m src.main
```

## Not in this step

Fake LLM, token cost, SQLite, and FastAPI. Those will fill `cost_usd`, `retries`, `confidence`, and `sources` instead of leaving defaults.
