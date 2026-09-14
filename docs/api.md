# HTTP API (`src/api/main.py`)

Week 3-style FastAPI wrapper around `ask_llm` and SQLite. No new LLM logic.

**Related files**

- Code: `src/api/main.py`
- Tests: `tests/test_api.py`
- Settings: `HOST`, `PORT`, `RESULTS_DB`

## Endpoints

| Method | Path | Body | Result |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/ask_batched` | `{"question": "..."}` | `Answer` JSON, row saved in SQLite |
| `POST` | `/ask` | `{"question": "..."}` | `text/plain` stream; **not** saved to SQLite |

`/ask_batched` is one complete JSON answer. `/ask` yields tokens as they arrive (`StreamingResponse`). Fake mode splits the canned text on spaces. Real mode uses `stream=True` (no tools on this path, so you get `delta.content` instead of `tool_calls`).

Empty `question` → HTTP **422** (Pydantic `min_length=1`).

## Run

```bash
pip install -r requirements-dev.txt
USE_FAKE=true uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Another terminal:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ask_batched \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is RAG?"}'

# streaming ( -N = no buffer )
curl -N -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is RAG?"}'
```

Interactive docs while the server runs: http://localhost:8000/docs

`USE_FAKE=false` uses Vocareum/OpenAI like the CLI.

Tests use FastAPI `TestClient` and `dependency_overrides` so they do not need a live server or a real key.
