from openai import OpenAI

from src.fake_llm import fake_ask_llm
from src.models import Answer, Question
from src.settings import Settings, get_settings


def _make_client(settings: Settings) -> OpenAI:
    if settings.openai_base_url:
        return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return OpenAI(api_key=settings.openai_api_key)


def ask_llm(question: Question, settings: Settings | None = None) -> Answer:
    cfg = settings if settings is not None else get_settings()
    if cfg.use_fake:
        return fake_ask_llm(question)

    # Vocareum and similar OpenAI-compatible gateways support Chat Completions,
    # not the newer Responses API used by api.openai.com.
    response = _make_client(cfg).chat.completions.create(
        model=cfg.openai_model,
        messages=[{"role": "user", "content": question.question}],
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_output_tokens,
    )
    content = response.choices[0].message.content or ""
    return Answer(content=content)


if __name__ == "__main__":
    answer = ask_llm(
        Question(question="Explain retrieval-augmented generation in three sentences.")
    )
    print(answer.content)
