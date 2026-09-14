# SQLite store (`src/store.py`)

Saves each `Question` + `Answer` to a local SQLite file so runs survive process exit. FastAPI (step 8) will use the same helpers.

**Related files**

- Code: `src/store.py`
- CLI persist: `persist_answer` in `src/main.py`
- Migration: `scripts/migrate_store.py`
- Path: `RESULTS_DB` / `Settings.results_db` (default `data/answers.db`)
- Tests: `tests/test_store.py`

## Schema (`answers`)

| Column | Meaning |
|---|---|
| `id` | Auto increment |
| `question` | User text |
| `content` | Model text |
| `retries` | Successful attempt index |
| `cost_usd` | Estimated USD |
| `created_at` | SQLite timestamp |
| `model` | e.g. `gpt-4o-mini` |
| `confidence` | 0–1 |
| `sources_json` | JSON list of strings |
| `schema_version` | `v1` |

`ensure_schema` creates the table, then `ALTER TABLE` for any missing extra columns. Safe to run many times.

## Helpers

- `connect(path)` — context manager; schema guaranteed; commits on success
- `save_answer(...)` — insert one row, return `id`
- `query_results(conn, model=None)` — list of dicts

SQL values use `?` placeholders (no string-built user SQL).

## Migrate / smoke

```bash
python scripts/migrate_store.py
python scripts/migrate_store.py data/answers.db
USE_FAKE=true python -m src.main
```

The CLI prints `saved row id=...`. Inspect:

```bash
sqlite3 data/answers.db "SELECT id, model, cost_usd, substr(content,1,60) FROM answers;"
```

`data/*.db` is gitignored.

## Not in this step

FastAPI. Step 8 will call `persist_answer` (or `save_answer`) from `POST /ask_batched`.
