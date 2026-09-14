import json
import logging
import time
from collections.abc import Iterator

import openai
from openai import OpenAI

from src.cost import compute_cost_usd
from src.fake_llm import FakeLLMError, fake_ask_llm
from src.models import Answer, Question
from src.settings import Settings, get_settings
from src.store import connect, save_answer
from src.tools import ANSWER_TOOL

logger = logging.getLogger(__name__)

_RETRYABLE = (
    FakeLLMError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _make_client(settings: Settings) -> OpenAI:
    if settings.openai_base_url:
        return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return OpenAI(api_key=settings.openai_api_key)


def _cost_from_usage(settings: Settings, usage: object | None) -> float:
    return compute_cost_usd(
        settings.openai_model,
        getattr(usage, "prompt_tokens", 0) if usage else 0,
        getattr(usage, "completion_tokens", 0) if usage else 0,
    )


def _answer_from_response(response: object, settings: Settings) -> Answer:
    """Map a ChatCompletion (or a test double) onto Answer."""
    logger.debug("step 1: map SDK response model=%s", settings.openai_model)
    choices = getattr(response, "choices", None) or []
    logger.debug("step 2: n_choices=%s", len(choices))
    if not choices:
        raise RuntimeError("LLM response has no choices")

    message = choices[0].message
    finish_reason = getattr(choices[0], "finish_reason", None)
    content_preview = (getattr(message, "content", None) or "")[:120]
    logger.debug(
        "step 3: finish_reason=%s content_preview=%r",
        finish_reason,
        content_preview,
    )

    tool_calls = getattr(message, "tool_calls", None) or []
    usage = getattr(response, "usage", None)
    logger.debug(
        "step 4: n_tool_calls=%s usage=%s",
        len(tool_calls),
        usage,
    )
    cost = _cost_from_usage(settings, usage)
    logger.debug("step 5: cost_usd=%s", cost)

    if tool_calls:
        raw_args = tool_calls[0].function.arguments
        logger.debug("step 6: tool raw arguments=%r", raw_args)
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        logger.debug("step 7: parsed tool args keys=%s", list(args.keys()))
        answer = Answer(
            content=args["content"],
            confidence=args["confidence"],
            sources=args.get("sources", []),
            cost_usd=cost,
        )
        logger.debug("step 8: Answer from tool_calls %s", answer.model_dump())
        return answer

    logger.debug("step 6: no tool_calls; fallback to message.content")
    answer = Answer(content=getattr(message, "content", None) or "", cost_usd=cost)
    logger.debug("step 7: Answer from content fallback %s", answer.model_dump())
    return answer


def _call_real_llm(question: Question, settings: Settings) -> Answer:
    # Vocareum and similar OpenAI-compatible gateways support Chat Completions,
    # not the newer Responses API used by api.openai.com.
    response = _make_client(settings).chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": question.question}],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_output_tokens,
        tools=[ANSWER_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "answer_question"},
        },
    )
    if hasattr(response, "model_dump"):
        print(response.model_dump())
    return _answer_from_response(response, settings)


def ask_llm(question: Question, settings: Settings | None = None) -> Answer:
    cfg = settings if settings is not None else get_settings()
    last_err: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            if cfg.use_fake:
                answer = fake_ask_llm(question, fail_rate=cfg.fail_rate)
            else:
                answer = _call_real_llm(question, cfg)
            return answer.model_copy(update={"retries": attempt})
        except _RETRYABLE as exc:
            last_err = exc
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_delay_s * (2**attempt))
                continue
            raise

    raise RuntimeError(f"ask_llm exhausted retries: {last_err}")


def stream_answer(
    question: Question, settings: Settings | None = None
) -> Iterator[str]:
    """Yield text chunks. Fake path splits words; real path uses stream=True.

    Does not persist to SQLite (same as the demo streaming route).
    """
    cfg = settings if settings is not None else get_settings()
    if cfg.use_fake:
        text = fake_ask_llm(question, fail_rate=cfg.fail_rate).content
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
        return

    stream = _make_client(cfg).chat.completions.create(
        model=cfg.openai_model,
        messages=[{"role": "user", "content": question.question}],
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_output_tokens,
        stream=True,
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


def persist_answer(question: Question, answer: Answer, settings: Settings) -> int:
    """Write one Answer row to SQLite and return the row id."""
    with connect(settings.results_db) as conn:
        return save_answer(
            conn,
            question=question.question,
            content=answer.content,
            retries=answer.retries,
            cost_usd=answer.cost_usd,
            model=settings.openai_model,
            confidence=answer.confidence,
            sources=answer.sources,
            schema_version=answer.schema_version,
        )


if __name__ == "__main__":
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    question = Question(
        question="Explain retrieval-augmented generation in three sentences."
    )
    answer = ask_llm(question, settings=settings)
    row_id = persist_answer(question, answer, settings)
    print(answer.content)
    print(
        f"cost_usd={answer.cost_usd} retries={answer.retries} "
        f"confidence={answer.confidence} sources={answer.sources}"
    )
    print(f"saved row id={row_id} db={settings.results_db}")
