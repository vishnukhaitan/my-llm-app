# MP1 writeup — Prompt Lab

## 1. Which strategy won, and on what dimension?

On this run, **few-shot** and **structured** tied for highest exact-match accuracy (**2.9 / 3**). All four strategies parsed at **100%**. **Zero-shot** was cheapest (**$0.000322**) and **CoT** was slowest (median **~3.4s**) and most expensive (**$0.000785**) with slightly lower accuracy (**2.8**). Structured matched few-shot on accuracy while beating CoT on cost and latency, and it had the highest judge score (**4.0**) alongside CoT.

## 2. What surprised you?

Parse rate still did not separate the strategies. Without stripping markdown fences, zero-shot often wraps JSON in ` ```json ` blocks and a strict parse of the whole reply would look like a parse failure even when the fields are right.

This run, **CoT was not the accuracy winner**. Both CoT misses were company punctuation (`Northwind Ltd.` / `Wonka Confectionery Ltd.` vs golden without the trailing period). The judge still gave **4** on those rows, so golden-set accuracy is stricter than the judge.

Zero-shot used **`null` for j05** (fresh grads) instead of **0**, and added the same `Ltd.` period. Few-shot missed only **j09**, averaging a 3–5 range to **4** instead of the minimum **3**. Structured’s only miss was the Wonka trailing period; years were all correct this time.

## 3. For my capstone domain, which strategy would I reach for first?

For `my-llm-app` I would start with **structured JSON** (schema, null/0 rules, no markdown). It tied for best accuracy here and was cheaper and faster than CoT. I would add **one or two few-shot examples** for ranges and “fresh grads → 0”. I would keep CoT for debugging, not as the default path.

## 4. If I had another day, what would I try next?

Normalise company suffixes (strip a trailing `.`) before scoring, then re-check whether CoT and zero-shot move up. I would also dump by-field errors automatically and try the winning prompt on 5–10 real capstone questions. I would not mix models into this four-way prompt comparison; that would need a separate table.
