# Internal LLM assistant usage

Northstar runs an internal question-answering assistant for staff. It is meant to search company policies and engineering notes, not to replace legal or medical advice.

## What it may be used for

- Looking up leave, expense, VPN, and onboarding rules
- Summarising an internal markdown doc you already have permission to read
- Drafting a first version of a status update from notes you paste

## What it must not be used for

- Customer personal data, payroll files, or unpublished financial results
- Generating production secrets, API keys, or VPN profiles
- Completely unsupervised emails to customers

## How answers work

The assistant retrieves short passages from the policy corpus, then writes an answer. Prefer answers that cite a source filename. If the retrieved text does not contain the answer, the assistant should say it does not know rather than guess.

Staff may use the `/ask` HTTP API from the office network or VPN. Rate limit: 60 questions per user per hour.

## Models

Default chat model is `gpt-4o-mini`. Embedding for retrieval is `text-embedding-3-small`. Do not switch production retrieval to a different embedding model without IT, or search quality collapses.

Fake-LLM mode (`USE_FAKE=true`) is for local tests only and must never be enabled on the staging or production API.
