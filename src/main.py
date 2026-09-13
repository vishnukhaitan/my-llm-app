import time

import openai
from openai import OpenAI

from src.cost import compute_cost_usd
from src.fake_llm import FakeLLMError, fake_ask_llm
from src.models import Answer, Question
from src.settings import Settings, get_settings

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


def _call_real_llm(question: Question, settings: Settings) -> Answer:
    # Vocareum and similar OpenAI-compatible gateways support Chat Completions,
    # not the newer Responses API used by api.openai.com.
    response = _make_client(settings).chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": question.question}],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_output_tokens,
    )
    if hasattr(response, "model_dump"):
        print(response.model_dump())
    content = response.choices[0].message.content or ""
    usage = response.usage
    cost = compute_cost_usd(
        settings.openai_model,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )
    return Answer(content=content, cost_usd=cost)


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
    answer = ask_llm(
        Question(question="Explain retrieval-augmented generation in three sentences.")
    )
    print(answer.content)
    print(f"cost_usd={answer.cost_usd} retries={answer.retries}")
