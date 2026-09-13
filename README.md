# my-llm-app

Local Python baseline for an OpenAI-based LLM application.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env: OPENAI_API_KEY, OPENAI_MODEL, and OPENAI_BASE_URL
# Vocareum keys (voc-...) need:
# OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

Configuration is loaded by `src/settings.py` (`pydantic-settings`) from `.env`. See [docs/settings.md](docs/settings.md).

LLM request/response shapes are `Question` and `Answer` in `src/models.py`. See [docs/models.md](docs/models.md).

Offline answers: `USE_FAKE=true` (see [docs/fake_llm.md](docs/fake_llm.md)).

Token pricing helper: [docs/cost.md](docs/cost.md) (`src/cost.py`).

## Run smoke test

```bash
python -m src.main
# or without an API call:
USE_FAKE=true python -m src.main
```

## Run tests

```bash
pytest
```
