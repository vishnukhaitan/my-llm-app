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
python mp1_prompt_lab.py --checks-only   # parser + accuracy, no API
python mp1_prompt_lab.py                 # 40 extractions + judge (uses pickle cache)
python mp1_prompt_lab.py --refresh       # ignore cache, call the API again
```

Or open `mp1_prompt_lab.ipynb` (same helpers) with interpreter `mp1-prompt-lab/.venv/bin/python`.

The first successful extraction and judge passes write `results_cache.pkl` and `scored_cache.pkl`. Reflection is in `mp1_writeup.md`; the table is also in `mp1_comparison.md`.

## Layout

```
mp1-prompt-lab/
  mp1_prompt_lab.py
  mp1_prompt_lab.ipynb
  mp1_writeup.md
  mp1_comparison.md
  requirements.txt
  .env.example
  data/job_snippets.jsonl
  data/golden_set.jsonl
```
