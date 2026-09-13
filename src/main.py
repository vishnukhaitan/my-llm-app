from openai import OpenAI

from src.settings import get_settings

settings = get_settings()
client = (
    OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    if settings.openai_base_url
    else OpenAI(api_key=settings.openai_api_key)
)


def ask_llm(prompt: str) -> str:
    # Vocareum and similar OpenAI-compatible gateways support Chat Completions,
    # not the newer Responses API used by api.openai.com.
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_output_tokens,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    answer = ask_llm(
        "Explain retrieval-augmented generation in three sentences."
    )
    print(answer)
