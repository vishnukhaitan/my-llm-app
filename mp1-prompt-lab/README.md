# MP1 · Prompt Lab

Compare four prompting strategies (zero-shot, few-shot, structured, chain-of-thought) on job-snippet field extraction. This folder is self-contained: it does not use `src/` from the parent app.

## Setup

```bash
cd mp1-prompt-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. If you use Vocareum, also set `OPENAI_BASE_URL`.

## Run

From this folder, with the venv active:

```bash
set -a && source .env && set +a
jupyter notebook mp1_prompt_lab.ipynb
```

Or open the notebook in Cursor / VS Code with interpreter `mp1-prompt-lab/.venv/bin/python`.

Load data from `data/` (10 snippets + 10 golden rows). Fill the TODOs in the notebook, then write `mp1_writeup.md`.

## Layout

```
mp1-prompt-lab/
  mp1_prompt_lab.ipynb
  requirements.txt
  .env.example
  data/job_snippets.jsonl
  data/golden_set.jsonl
```
