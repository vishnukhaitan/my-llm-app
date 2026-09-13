import json
import logging
import time

import openai
from openai import OpenAI

from src.cost import compute_cost_usd
from src.fake_llm import FakeLLMError, fake_ask_llm
from src.models import Answer, Question
from src.settings import Settings, get_settings
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


if __name__ == "__main__":
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    answer = ask_llm(
        Question(question="Explain retrieval-augmented generation in three sentences."),
        settings=settings,
    )
    print(answer.content)
    print(
        f"cost_usd={answer.cost_usd} retries={answer.retries} "
        f"confidence={answer.confidence} sources={answer.sources}"
    )
