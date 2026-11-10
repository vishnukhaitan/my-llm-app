# MP2 Mini-RAG

Question answering over five Sherlock Holmes story summaries. The script loads the stories, chunks them, embeds the chunks with OpenAI, stores them in Qdrant, and answers from the retrieved excerpts.

## Setup

From the repository root, with the project virtual environment active:

```bash
source .venv/bin/activate
pip install -r mp2-mini-rag/requirements.txt
cp mp2-mini-rag/.env.example mp2-mini-rag/.env
```

Edit `mp2-mini-rag/.env`:

- `OPENAI_API_KEY` is required.
- `OPENAI_BASE_URL=https://openai.vocareum.com/v1` is required for a Vocareum key. A Vocareum key sent to the public OpenAI API returns 401.
- `QDRANT_URL` and `QDRANT_API_KEY` send vectors to Qdrant Cloud. If `QDRANT_URL` is unset, vectors are stored in `mp2-mini-rag/qdrant_storage/`, which is gitignored.

Load the file in the same shell before any command. `source` does not persist to a new terminal:

```bash
set -a && source mp2-mini-rag/.env && set +a
```

## Run

```bash
python mp2-mini-rag/mp2_rag.py ingest
python mp2-mini-rag/mp2_rag.py validate
python mp2-mini-rag/mp2_rag.py ask
```

`ingest` rebuilds the `mp2_sherlock` collection. The current chunker produces 89 chunks: 15, 18, 15, 20, and 21 across the five stories. `ask` reads a question from the terminal. An empty line exits. `validate` runs the two predefined questions and then `data/learner_questions.jsonl`.

## How to read validation

A check mark means one of the three retrieved chunks came from `expected_source`. That is the score that matters. The last run was 2/2 on the predefined questions and 3/3 on the learner questions. The full log is `mp2_validation.txt`.

`Facts matched` counts how many `expected_facts` strings appear in the answer text. It is a diagnostic. A correct answer can score low when it paraphrases, or when the right story was retrieved but the top 3 chunks did not contain the phrase.

Latency is the time to embed the question, search Qdrant, and generate the answer.

## Pipeline

| Step | Function | Choice |
|---|---|---|
| Load | `load_corpus` | One document per `.txt` file. Title is the first non-empty line. |
| Chunk | `chunk_document` | Paragraphs packed to 500 characters, 80-character overlap. Short unpunctuated lines become the section name. |
| Embed | `embed_texts` | `text-embedding-3-small`, 1536 dimensions. |
| Store | `setup_collection`, `ingest_chunks` | Qdrant collection `mp2_sherlock`, cosine distance. |
| Retrieve | `retrieve` | Top 3 chunks. This client has `query_points` rather than `search`. |
| Answer | `answer` | `gpt-4o-mini` at temperature 0. The prompt allows only the retrieved excerpts and asks for the title and section. |
