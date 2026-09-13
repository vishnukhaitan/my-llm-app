# Tool calling (`src/tools.py`)

Forces a **structured** `Answer` (`content`, `confidence`, `sources`). This is **not** an external weather/search API. The model fills JSON that matches `ANSWER_TOOL`; Python maps it onto `Answer`.

**Related files**

- Schema: `src/tools.py`
- Used in: `src/main.py` (`_call_real_llm`, `_answer_from_response`)
- Tests: `tests/test_fake_llm.py`

## What we send

`tools=[ANSWER_TOOL]` plus `tool_choice` so the model should call `answer_question` instead of only writing `message.content`.

Required JSON fields: `content`, `confidence` (0–1), `sources` (list of strings; `[]` is fine).

## What we read

```text
response.choices[0].message.tool_calls[0].function.arguments
```

That string is `json.loads`’d, then:

```python
Answer(
    content=args["content"],
    confidence=args["confidence"],
    sources=args.get("sources", []),
    cost_usd=cost,  # still from response.usage
)
```

`finish_reason` is often `tool_calls`. `message.content` is often `None`.

## Fallback

If the gateway ignores tools and only sets `message.content` (some Vocareum setups), we still build an `Answer` with `confidence=1.0` and `sources=[]`.

## Fake path

`USE_FAKE=true` does not send tools. Fake answers use `confidence=1.0` and `sources=[]`.

## Not in this step

No second round-trip (we do not execute a real tool and send results back). SQLite and FastAPI come next.
