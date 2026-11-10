# MP2 reflection

## What worked

The pipeline runs from the five story files to a cited answer without an orchestration framework. `load_corpus` produced five documents. Paragraph chunking with section-header detection produced 89 chunks, inside the expected range, and each chunk kept `source`, `title`, and `section`. Those fields are what validation checks and what `ask` prints.

Embedding with `text-embedding-3-small` and a cosine Qdrant collection was enough for story-level retrieval. On the final run, all five questions cited the expected file: source-match 2/2 on the predefined questions and 3/3 on the learner questions. The learner questions cover the three stories the predefined set does not: the blue carbuncle, the engineer's thumb, and the scandal in Bohemia. Question 5 is the multi-section case. The smoke-rocket plan and the photograph's hiding place are different sections, and the top 3 still came from `05_scandal_in_bohemia.txt`.

The grounded prompt did its job when context was incomplete. For the Red-Headed League question, the answer named Vincent Spaulding and said the excerpts did not give his real identity, instead of inventing John Clay from the model's own knowledge of the original story.

## Challenges

The first Qdrant ingest never appeared in the cloud console. `QDRANT_URL` was not in the environment, so the client wrote to the local `qdrant_storage/` folder. Putting the cloud URL and key in `.env.example` also did nothing, because that file is a template and the script only reads the process environment. The cloud collection showed 89 points only after the values were moved to `.env` and `ingest` was run again.

A Vocareum key failed with 401 until `OPENAI_BASE_URL` was set to `https://openai.vocareum.com/v1`. Without that setting the client called the public OpenAI API. The installed `qdrant-client` also no longer has `search`, so retrieval uses `query_points`. `recreate_collection` still runs, but it prints a deprecation warning.

Chunk boundaries are the retrieval weakness. Paragraphs longer than 500 characters are cut mid-word. In *The Red-Headed League*, one chunk ends at "told t" and the next begins at "hfully". The assistant question shows the practical cost. The top 3 chunks are from the right story, but they are the investigation and the copying job, not "The Identity of the Assistant," where John Clay, the murder charge, and the white patches on his knees are stated. Facts matched stayed at 1/5 because those strings were not in the context. The same pattern appeared while drafting the engineer's-thumb question. A broad question retrieved the fuller's-earth cover story ahead of the coining paragraph, and the model reported the cover story as the real use of the press. Asking specifically about Beechmoor pulled "The Coining Operation" into first place, and the answer then included half-crowns, nickel and tin, and fuller's earth.

## Lessons

Source-match and fact-match measure different things. Source-match asks whether the right document was retrieved. Fact-match asks whether the answer copied particular phrases. A faithful answer can miss facts when the right passage is chunk four or five. Raising `k` would add those passages, and it would also add more distractors, which is what happened when the cover story and the true use of the press were both in the top 3.

Section labels made the failures easy to see. The citation "The Red-Headed League — Holmes Investigates" showed which part of the story had been retrieved, so the missing John Clay passage was obvious without reading the whole file.

## Improvements

Split long paragraphs on sentence boundaries so a chunk does not start or end mid-word. Keep the 500-character target, but do not slice a token in half.

Retrieve a few extra chunks and keep the ones that add a new section, so a story-level hit still surfaces the resolution paragraph. For this corpus, that would bring in "The Identity of the Assistant" beside "Holmes Investigates."

A later hybrid index would help the rare names that semantic search under-ranks: John Clay, swamp adder, Beechmoor. Those strings are exact, and the failures above were exact-phrase misses inside an otherwise correct document.
