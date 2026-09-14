"""HTTP API: health check and one-shot (non-streaming) ask."""

from fastapi import Depends, FastAPI

from src.main import ask_llm, persist_answer
from src.models import Answer, Question
from src.settings import Settings, get_settings

app = FastAPI(title="my-llm-app")


def get_api_settings() -> Settings:
    return get_settings()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask_batched", response_model=Answer)
def ask_batched(
    q: Question,
    settings: Settings = Depends(get_api_settings),  # noqa: B008
) -> Answer:
    answer = ask_llm(q, settings)
    persist_answer(q, answer, settings)
    return answer
