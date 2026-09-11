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

## Run smoke test

```bash
python -m src.main
```

## Run tests

```bash
pytest
```
