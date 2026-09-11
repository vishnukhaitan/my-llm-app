from openai import OpenAI

from src.config import (
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)

client = OpenAI(base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI()


def ask_llm(prompt: str) -> str:
    # Vocareum and similar OpenAI-compatible gateways support Chat Completions,
    # not the newer Responses API used by api.openai.com.
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    answer = ask_llm(
        "Explain retrieval-augmented generation in three sentences."
    )
    print(answer)
