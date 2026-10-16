# Naive RAG retrieve and answer (A5)

A5 reads `data/embeddings.json` (built in A4), ranks chunks with NumPy cosine, then asks the chat model to answer from those chunks only. It does not start FastAPI. The HTTP route still calls `ask_llm` until A6.

**Related files**

- Retrieve: `src/rag/retrieve.py` (`cosine`, `retrieve`)
- Generate: `src/rag/ask_rag.py` (`ask_rag`)
- CLI: `scripts/ask_rag.py`
- Index: `src/rag/embeddings.py` (`load_index`, `embed_batch`)
- Tests: `tests/test_retrieve.py`, `tests/test_ask_rag.py`

## Run

Activate the venv first (`python` is not on the path otherwise):

```bash
source .venv/bin/activate
LOG_LEVEL=DEBUG python scripts/ask_rag.py "How many annual leave days does Northstar grant?"
```

The index and the question must use the same embedding model. A file built with the real API is 1536-d (`text-embedding-3-small`). `USE_FAKE=true` uses 8-d vectors and will raise a dimension mismatch against a 1536-d file.

`--k` sets how many chunks go into the prompt (default 3). You cannot get more hits than rows in the JSON file.

## What the live run did

Question: *How many annual leave days does Northstar grant?*

Index at the time: 3 chunks, all from `leave_policy`, dim 1536.

### 1. Load the index

```text
STEP load_index (data/embeddings.json)
```

Reads the JSON A4 wrote. No API call. Each row has `chunk_id`, `source_id`, `text`, and `vector`.

### 2. Embed the question

```text
STEP ask_rag START k=3
STEP retrieve START (k=3 index=3)
STEP embed_batch START (OpenAI model=text-embedding-3-small n=1)
POST https://openai.vocareum.com/v1/embeddings  →  200 OK
STEP embed_batch DONE (1 vectors dim=1536)
```

One embedding request. The body is the question string and `text-embedding-3-small`. `200 OK` means the key worked and the vector length matches the index (1536).

The long `httpcore` / `openai._base_client` lines are the OpenAI SDK’s own debug log (TCP, TLS, headers). They are not extra RAG steps. `LOG_LEVEL=INFO` hides most of that.

### 3. Rank chunks (NumPy, local)

```text
hit leave_policy#0  score=0.7415  chars=500
hit leave_policy#2  score=0.5522  chars=350
hit leave_policy#1  score=0.4676  chars=500
STEP retrieve DONE top=leave_policy#0,leave_policy#2,leave_policy#1
```

Cosine compares the question vector to every stored vector. Higher is closer. The top hit (`leave_policy#0`, 0.74) is the chunk that contains the 18 / 22 / 25 day table. The other two are booking and carry-over text, so they score lower but still fit in `k=3`.

### 4. Chat completion

```text
ask_rag retrieved=['leave_policy#0', 'leave_policy#2', 'leave_policy#1']
rag prompt chars=1479
POST https://openai.vocareum.com/v1/chat/completions  →  200 OK
```

The user message is those three chunk texts plus the question. The system prompt says to use only that context. The model is `gpt-4o-mini` (`OPENAI_MODEL`), not the embedding model.

`finish_reason=stop` and `n_tool_calls=1` means the model returned structured fields (`content`, `confidence`, `sources`) instead of plain text. Usage on this run: 469 prompt tokens, 77 completion tokens, `cost_usd≈0.000117`.

### 5. Printed answer

```text
Northstar Labs grants annual leave based on years of continuous service:
0–2 years → 18 working days, 2–5 → 22, 5+ → 25.
sources=['leave_policy#0', 'leave_policy#2', 'leave_policy#1']
confidence=1.0
```

`sources` on the final line are the **retrieved** chunk ids (what search returned), in score order. The model itself cited only `leave_policy#0` inside the tool call. `ask_rag` replaces that with the retrieved ids so the caller sees what was actually passed as context.

## Tests (no API)

```bash
pytest tests/test_retrieve.py tests/test_ask_rag.py -q
```

These inject vectors and a fake LLM function. They do not read `data/embeddings.json` and do not call Vocareum.
