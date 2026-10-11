"""MP1 Prompt Lab — compare four prompting strategies on job-field extraction.

Run from this folder:

    python mp1_prompt_lab.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
RESULTS_CACHE = LAB_DIR / "results_cache.pkl"
SCORED_CACHE = LAB_DIR / "scored_cache.pkl"

MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"
TEMPERATURE = 0.0
JSON_FIELDS = "company, role, years_experience_required"

RATES = {
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o": {"in": 2.50 / 1_000_000, "out": 10.00 / 1_000_000},
}


class JudgeResult(BaseModel):
    """Structured schema for the gpt-4o judge (score 1–4 plus a short reason)."""

    score: int = Field(ge=1, le=4)
    reasoning: str


def load_env() -> None:
    """Load OPENAI_API_KEY / OPENAI_BASE_URL from this folder's .env, else cwd .env.

    Does not override variables already set in the shell.
    """
    for candidate in (LAB_DIR / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break


def make_client() -> AsyncOpenAI:
    """Build an async OpenAI client (Vocareum-compatible if OPENAI_BASE_URL is set).

    Returns:
        Client used for extraction (mini) and judging (gpt-4o).

    Raises:
        SystemExit: if OPENAI_API_KEY is missing.
    """
    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in mp1-prompt-lab/.env")
    return AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)


def load_snippets() -> list[dict]:
    """Load the 10 job posting inputs from data/job_snippets.jsonl.

    Returns:
        List of dicts with keys id, snippet.
    """
    path = DATA_DIR / "job_snippets.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_golden() -> dict[str, dict]:
    """Load the human answer key, keyed by snippet id.

    Returns:
        Mapping id → {company, role, years_experience_required, notes, ...}.
    """
    path = DATA_DIR / "golden_set.jsonl"
    return {
        row["id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def prompt_zero_shot(snippet_text: str) -> list[dict]:
    """Strategy 1 — one user message, no examples, no persona.

    Args:
        snippet_text: Raw job posting text.

    Returns:
        OpenAI chat `messages` list.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Extract {JSON_FIELDS} from this job snippet as JSON. "
                "years_experience_required must be an integer or null if not stated.\n\n"
                f"{snippet_text}"
            ),
        }
    ]


def prompt_few_shot(snippet_text: str) -> list[dict]:
    """Strategy 2 — same ask plus 3 worked examples (not from j01–j10).

    Examples cover a stated year, a missing year (null), and fresh grads (0).

    Args:
        snippet_text: Raw job posting text.

    Returns:
        OpenAI chat `messages` list.
    """
    examples = (
        "Snippet: Maple Street Bakery needs a Junior Baker. One year of shop experience required.\n"
        'JSON: {"company": "Maple Street Bakery", "role": "Junior Baker", "years_experience_required": 1}\n\n'
        "Snippet: Orion Labs is hiring a Staff Hardware Engineer. We hire on demonstrated impact and do not list a years requirement.\n"
        'JSON: {"company": "Orion Labs", "role": "Staff Hardware Engineer", "years_experience_required": null}\n\n'
        "Snippet: Redwood Civic Trust seeks a Grant Writer. Fresh graduates welcome; no prior experience required.\n"
        'JSON: {"company": "Redwood Civic Trust", "role": "Grant Writer", "years_experience_required": 0}\n\n'
    )
    return [
        {
            "role": "user",
            "content": (
                f"Extract {JSON_FIELDS} as JSON. Use null when years are not stated. "
                "Use 0 when the posting says no experience is required. Follow the examples.\n\n"
                f"{examples}"
                f"Snippet: {snippet_text}\nJSON:"
            ),
        }
    ]


def prompt_structured(snippet_text: str) -> list[dict]:
    """Strategy 3 — recruiter system prompt, JSON schema, explicit null/0 rules.

    Args:
        snippet_text: Raw job posting text.

    Returns:
        OpenAI chat `messages` list (system + user).
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an expert recruiter extracting structured fields from job postings. "
                "Return only a JSON object with exactly these keys: "
                "company (string), role (string), years_experience_required (integer or null). "
                "Rules: take the stated minimum (around 6 → 6, 7+ → 7, 3-5 → 3). "
                "Fresh grads / no experience required → 0. "
                "If years are not stated, return null — do not infer or guess. "
                "Do not wrap the JSON in markdown."
            ),
        },
        {"role": "user", "content": snippet_text},
    ]


def prompt_cot(snippet_text: str) -> list[dict]:
    """Strategy 4 — ask for step-by-step reasoning, then a JSON object.

    Parser must find JSON inside the prose (see parse_response).

    Args:
        snippet_text: Raw job posting text.

    Returns:
        OpenAI chat `messages` list.
    """
    return [
        {
            "role": "user",
            "content": (
                "Think step by step about company name, job title, and minimum years of experience. "
                "Then output a JSON object with keys company, role, years_experience_required "
                "(integer or null if not stated).\n\n"
                f"{snippet_text}"
            ),
        }
    ]


# Only the prompt function changes across the four strategies (same model + temperature).
STRATEGIES = {
    "zero_shot": prompt_zero_shot,
    "few_shot": prompt_few_shot,
    "structured": prompt_structured,
    "cot": prompt_cot,
}


def parse_response(text: str | None) -> dict | None:
    """Turn model text into a dict, or None if there is no JSON object.

    Strips ```json fences, then takes the first `{` through the last `}` so CoT
    preambles do not fail the parse.

    Args:
        text: Raw model content.

    Returns:
        Parsed object, or None on empty / invalid JSON / non-dict.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _usage_cost(model: str, usage) -> float:
    """USD cost from prompt/completion token counts using the RATES table.

    Args:
        model: gpt-4o-mini or gpt-4o.
        usage: OpenAI usage object, or None.

    Returns:
        Cost in dollars (0.0 if usage is missing).
    """
    if usage is None:
        return 0.0
    rates = RATES[model]
    return usage.prompt_tokens * rates["in"] + usage.completion_tokens * rates["out"]


def _norm_str(value) -> str | None:
    """Normalise company/role for scoring: trim, collapse spaces, casefold.

    Treats empty / 'null' / 'none' as missing (None).

    Args:
        value: Extracted or golden string (any type).

    Returns:
        Comparable string, or None.
    """
    if value is None:
        return None
    text = " ".join(str(value).split()).casefold()
    if text in {"", "null", "none"}:
        return None
    return text


def _norm_years(value):
    """Normalise years so 5, '5+', and '5 years' all compare as 5.

    Missing / null / n/a → None (correct for j10). First integer in the string
    is used so '7+' and 'between 3 and 5' become 7 and 3.

    Args:
        value: Extracted or golden years field.

    Returns:
        int, None, or the original text if no digit is found.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().casefold()
    if text in {"", "null", "none", "n/a", "na"}:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return text
    return int(match.group(0))


def score_accuracy(extracted: dict | None, gold: dict) -> int:
    """Count how many of the 3 fields match the golden row (0, 1, 2, or 3).

    Args:
        extracted: Parsed model JSON, or None if parse failed.
        gold: Golden-set row for the same snippet id.

    Returns:
        Integer field-match count. None extract → 0.
    """
    if not isinstance(extracted, dict):
        return 0
    score = 0
    if _norm_str(extracted.get("company")) == _norm_str(gold.get("company")):
        score += 1
    if _norm_str(extracted.get("role")) == _norm_str(gold.get("role")):
        score += 1
    if _norm_years(extracted.get("years_experience_required")) == _norm_years(
        gold.get("years_experience_required")
    ):
        score += 1
    return score


def field_hit(extracted: dict | None, gold: dict, field: str) -> bool:
    """Whether one field matches gold (used for the by-field table).

    Args:
        extracted: Parsed JSON or None.
        gold: Golden-set row.
        field: company, role, or years_experience_required.

    Returns:
        True if that field matches after normalisation.
    """
    if not isinstance(extracted, dict):
        return False
    if field == "years_experience_required":
        return _norm_years(extracted.get(field)) == _norm_years(gold.get(field))
    return _norm_str(extracted.get(field)) == _norm_str(gold.get(field))


async def run_one(client: AsyncOpenAI, strategy_name: str, snippet: dict) -> dict:
    """Call gpt-4o-mini once: one strategy × one snippet.

    Records raw text, parsed JSON, parse flag, USD cost, and latency.

    Args:
        client: Shared async OpenAI client.
        strategy_name: Key in STRATEGIES.
        snippet: Job snippet dict (id + snippet).

    Returns:
        Result row for later scoring / caching.
    """
    messages = STRATEGIES[strategy_name](snippet["snippet"])
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    latency_s = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    extracted = parse_response(raw)
    return {
        "strategy": strategy_name,
        "snippet_id": snippet["id"],
        "raw_response": raw,
        "extracted": extracted,
        "parse_success": extracted is not None,
        "cost_usd": _usage_cost(MODEL, resp.usage),
        "latency_s": latency_s,
    }


async def run_all(client: AsyncOpenAI, snippets: list[dict]) -> list[dict]:
    """Run all 4 strategies × 10 snippets = 40 extraction calls in parallel.

    Args:
        client: Shared async OpenAI client.
        snippets: Loaded job snippets.

    Returns:
        List of 40 result dicts from run_one.
    """
    tasks = [
        run_one(client, name, snippet)
        for name in STRATEGIES
        for snippet in snippets
    ]
    return list(await asyncio.gather(*tasks))


async def score_llm_judge(
    client: AsyncOpenAI,
    snippet_text: str,
    extracted: dict | None,
    gold: dict,
) -> int:
    """Ask gpt-4o for a 1–4 quality score (not the same as exact-match accuracy).

    Prefers chat.completions.parse(JudgeResult). If the gateway rejects structured
    outputs, falls back to JSON in the text, then to the first digit 1–4.

    Args:
        client: Shared async OpenAI client.
        snippet_text: Original posting (judge can see fabrication vs gold).
        extracted: Parsed extraction or None.
        gold: Golden-set row.

    Returns:
        Integer 1–4. Unparsable extract → 1.
    """
    if extracted is None:
        return 1
    payload = {
        "snippet": snippet_text,
        "extracted": extracted,
        "gold": {
            "company": gold.get("company"),
            "role": gold.get("role"),
            "years_experience_required": gold.get("years_experience_required"),
        },
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You score job-field extraction against gold. "
                "Fields: company, role, years_experience_required. "
                "4 = all three correct; 3 = two correct and nothing fabricated; "
                "2 = one correct or a fabricated field; 1 = none correct or unparsable. "
                "A trailing period on Ltd is still a company match."
            ),
        },
        {"role": "user", "content": json.dumps(payload, default=str)},
    ]
    try:
        resp = await client.chat.completions.parse(
            model=JUDGE_MODEL,
            temperature=0.0,
            messages=messages,
            response_format=JudgeResult,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is not None:
            return int(parsed.score)
    except Exception:
        # Vocareum / older APIs may not support .parse(); try free-form JSON next.
        pass
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=messages
        + [
            {
                "role": "user",
                "content": "Reply with JSON {\"score\": <1-4>, \"reasoning\": \"...\"} only.",
            }
        ],
    )
    obj = parse_response(resp.choices[0].message.content or "")
    if obj and "score" in obj:
        try:
            return max(1, min(4, int(obj["score"])))
        except (TypeError, ValueError):
            pass
    match = re.search(r"[1-4]", resp.choices[0].message.content or "")
    return int(match.group(0)) if match else 1


async def score_row(
    client: AsyncOpenAI,
    row: dict,
    golden: dict[str, dict],
    snippet_by_id: dict[str, str],
) -> dict:
    """Attach parse_success, accuracy (0–3), and llm_judge_score to one result row.

    Args:
        client: Shared async OpenAI client.
        row: Output of run_one.
        golden: Full golden-set map.
        snippet_by_id: id → snippet text for the judge.

    Returns:
        Copy of row plus scoring fields.
    """
    gold = golden[row["snippet_id"]]
    out = dict(row)
    out["parse_success"] = row.get("extracted") is not None
    out["accuracy"] = score_accuracy(row.get("extracted"), gold)
    out["llm_judge_score"] = await score_llm_judge(
        client,
        snippet_by_id[row["snippet_id"]],
        row.get("extracted"),
        gold,
    )
    return out


def comparison_tables(scored: list[dict], golden: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate 40 scored rows into the MP1 summary table and a by-field hit-rate table.

    Cost is summed (6 decimals); latency is median seconds; accuracy is mean of 0–3.

    Args:
        scored: Rows after score_row.
        golden: Golden-set map (for by-field hits).

    Returns:
        (summary_by_strategy, by_field_hit_rates).
    """
    df = pd.DataFrame(scored)
    summary = df.groupby("strategy").agg(
        {
            "accuracy": "mean",
            "parse_success": "mean",
            "llm_judge_score": "mean",
            "cost_usd": "sum",
            "latency_s": "median",
        }
    )
    summary = summary.round(
        {
            "accuracy": 3,
            "parse_success": 3,
            "llm_judge_score": 3,
            "cost_usd": 6,
            "latency_s": 3,
        }
    )
    summary.columns = [
        "Accuracy (mean of 3)",
        "Parse rate",
        "Judge score",
        "Total cost ($)",
        "Latency p50 (s)",
    ]
    by_field = (
        pd.DataFrame(
            {
                "company": [field_hit(r.get("extracted"), golden[r["snippet_id"]], "company") for r in scored],
                "role": [field_hit(r.get("extracted"), golden[r["snippet_id"]], "role") for r in scored],
                "years": [
                    field_hit(r.get("extracted"), golden[r["snippet_id"]], "years_experience_required")
                    for r in scored
                ],
                "strategy": [r["strategy"] for r in scored],
            }
        )
        .groupby("strategy")
        .mean()
        .round(3)
    )
    return summary, by_field


def run_offline_checks() -> None:
    """Assert parser + accuracy helpers without calling the API.

    Covers fences, CoT-wrapped JSON, 5+ vs 5, j10 null vs fabricated years.
    """
    assert parse_response('{"company": "Acme Corp", "role": "x", "years_experience_required": 5}')["company"] == "Acme Corp"
    assert parse_response("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert parse_response('Reasoning here... {"a": 1}') == {"a": 1}
    assert parse_response("not json") is None
    gold_j01 = {
        "company": "Acme Corp",
        "role": "Senior Software Engineer",
        "years_experience_required": 5,
    }
    assert (
        score_accuracy(
            {
                "company": "acme corp",
                "role": " Senior Software Engineer ",
                "years_experience_required": "5+",
            },
            gold_j01,
        )
        == 3
    )
    gold_j10 = {
        "company": "Cyberdyne Systems",
        "role": "AI/ML Research Scientist",
        "years_experience_required": None,
    }
    assert score_accuracy({**gold_j10, "years_experience_required": 5}, gold_j10) == 2
    assert score_accuracy(gold_j10, gold_j10) == 3
    assert score_accuracy(None, gold_j01) == 0
    print("Offline parse/score checks passed.")


async def run_pipeline(*, refresh: bool = False) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Full lab: load data, extract (or cache), judge (or cache), print tables.

    Args:
        refresh: If True, ignore pickle caches and pay for new API calls.

    Returns:
        scored rows, summary table, by-field table.
    """
    snippets = load_snippets()
    golden = load_golden()
    snippet_by_id = {row["id"]: row["snippet"] for row in snippets}
    client = make_client()

    if RESULTS_CACHE.exists() and not refresh:
        results = pickle.loads(RESULTS_CACHE.read_bytes())
        print(f"Loaded {len(results)} cached extraction results.")
    else:
        results = await run_all(client, snippets)
        RESULTS_CACHE.write_bytes(pickle.dumps(results))
        print(f"Got {len(results)} results and wrote {RESULTS_CACHE.name}.")

    if SCORED_CACHE.exists() and not refresh:
        scored = pickle.loads(SCORED_CACHE.read_bytes())
        print(f"Loaded {len(scored)} cached scored rows.")
    else:
        scored = list(
            await asyncio.gather(
                *[score_row(client, row, golden, snippet_by_id) for row in results]
            )
        )
        SCORED_CACHE.write_bytes(pickle.dumps(scored))
        print(f"Scored {len(scored)} results and wrote {SCORED_CACHE.name}.")

    summary, by_field = comparison_tables(scored, golden)
    print(summary.to_markdown())
    print()
    print("By-field hit rate")
    print(by_field.to_markdown())
    print(f"\nTotal extraction cost: ${sum(r['cost_usd'] for r in results):.6f}")
    return scored, summary, by_field


def main() -> None:
    """CLI entry: optional --checks-only or --refresh, then run_pipeline."""
    parser = argparse.ArgumentParser(description="MP1 prompt strategy comparison")
    parser.add_argument("--refresh", action="store_true", help="Ignore pickle caches and call the API")
    parser.add_argument("--checks-only", action="store_true", help="Run parse/score asserts, no API")
    args = parser.parse_args()
    run_offline_checks()
    if args.checks_only:
        return
    asyncio.run(run_pipeline(refresh=args.refresh))


if __name__ == "__main__":
    main()
