"""Request and response shapes for LLM calls."""

from pydantic import BaseModel, Field


class Question(BaseModel):
    """User input to the model."""

    question: str = Field(..., min_length=1)


class Answer(BaseModel):
    """Structured LLM result. Extra fields have defaults so later steps can fill them in."""

    content: str
    cost_usd: float = 0.0
    retries: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)
    schema_version: str = "v1"
