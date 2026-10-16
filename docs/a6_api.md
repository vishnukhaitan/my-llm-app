# A6 — HTTP RAG (`POST /ask_batched`)

A5 answers from the terminal (`scripts/ask_rag.py`). A6 is the same function behind FastAPI. The client sends JSON; the server loads `data/embeddings.json`, retrieves chunks, asks the chat model, saves a SQLite row, and returns JSON.

`POST /ask` is still a plain text stream. It does not search the corpus.

**Related files**

- Route: `src/api/main.py`
- RAG: `src/rag/ask_rag.py`, `src/rag/retrieve.py`
- Index: `data/embeddings.json` (built by `scripts/build_index.py`)
- Tests: `tests/test_api.py`

## What the live call did

Server already running on port 8000. Index: 3 chunks from `leave_policy`, dim 1536.

```bash
curl -s -X POST http://127.0.0.1:8000/ask_batched \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many annual leave days does Northstar grant?"}'
```

Response:

```json
{
  "content": "Northstar Labs grants annual leave based on years of continuous service: employees with 0 to 2 years of service receive 18 working days, those with 2 to 5 years receive 22 working days, and employees with 5 or more years receive 25 working days per calendar year. Part-time staff receive a pro-rata entitlement based on their contracted hours.",
  "cost_usd": 0.00012495,
  "retries": 0,
  "confidence": 1.0,
  "sources": ["leave_policy#0", "leave_policy#2", "leave_policy#1"],
  "schema_version": "v1"
}
```

## Steps inside that one HTTP request

1. **FastAPI reads the body.** `question` must be a non-empty string. Empty → HTTP 422.
2. **Load the index.** `get_rag_index()` reads `data/embeddings.json`. Missing file → HTTP 503. This run had 3 rows.
3. **Embed the question.** Same model as A4: `text-embedding-3-small` via `https://openai.vocareum.com/v1/embeddings`. One 1536-d vector.
4. **NumPy cosine.** Rank the 3 stored vectors. Same order as the A5 CLI: `leave_policy#0` (entitlement table), then `#2`, then `#1`.
5. **Chat.** `gpt-4o-mini` at `/chat/completions`. The prompt is those three chunk texts plus the question. The model must answer only from that context.
6. **`sources` on the JSON** are the retrieved chunk ids, in score order, not whatever the model typed inside the tool call.
7. **SQLite.** `persist_answer` writes one row to `data/answers.db` (question, content, cost, confidence, sources).
8. **HTTP 200** returns the `Answer` fields above. This run cost about $0.000125.

The answer text matches the policy: 18 days (0–2 years), 22 (2–5), 25 (5+), plus pro-rata for part-time. That sentence is in `leave_policy#0`, which is why it ranked first.

## Start the server

```bash
source .venv/bin/activate
python scripts/build_index.py
uvicorn src.api.main:app --reload --port 8000
```

`USE_FAKE=true` only works if the JSON file was also built with fake 8-d vectors. The file from the real embed call is 1536-d, so leave `USE_FAKE=false`.

Interactive docs while the server runs: http://127.0.0.1:8000/docs

## Tests (no Vocareum)

```bash
pytest tests/test_api.py -q
```

These inject a tiny index and a fake answer. They check that `/ask_batched` returns chunk ids in `sources` and that a missing index is 503.
