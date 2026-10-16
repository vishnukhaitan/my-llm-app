# HTTP API (`src/api/main.py`)

`POST /ask_batched` is naive RAG (A6): load `data/embeddings.json`, call `ask_rag`, save the row. `POST /ask` is still a plain stream and does not search the corpus.

**Related files**

- Code: `src/api/main.py`
- RAG: `src/rag/ask_rag.py` (see [ask_rag.md](ask_rag.md))
- Tests: `tests/test_api.py`
- Settings: `HOST`, `PORT`, `RESULTS_DB`

## Endpoints

| Method | Path | Body | Result |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/ask_batched` | `{"question": "..."}` | `Answer` JSON, row saved in SQLite |
| `POST` | `/ask` | `{"question": "..."}` | `text/plain` stream; **not** saved to SQLite |

`/ask_batched` is one complete JSON answer. `sources` are retrieved chunk ids such as `leave_policy#0`. Missing `data/embeddings.json` → HTTP **503**. The question embedding must match the index dimension (1536 for a real index, 8 only if the file was built with `USE_FAKE=true`).

`/ask` yields tokens as they arrive (`StreamingResponse`). Fake mode splits the canned text on spaces. Real mode uses `stream=True` (no tools on this path, so you get `delta.content` instead of `tool_calls`). It does not load the index.

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

`USE_FAKE=false` embeds the question and calls the chat model through Vocareum, same as `scripts/ask_rag.py`. Build the index first:

```bash
python scripts/build_index.py
uvicorn src.api.main:app --reload --port 8000
```

Tests use FastAPI `TestClient` and `dependency_overrides` so they do not need a live server or a real key.
