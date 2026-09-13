"""OpenAI tool schema used to force a structured Answer (not an external API)."""

ANSWER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "answer_question",
        "description": (
            "Return a structured answer with content, confidence, and sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The answer in 2-4 sentences.",
                },
                "confidence": {
                    "type": "number",
                    "description": "How confident you are in the answer, 0.0 to 1.0.",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Source identifiers or URLs you used. Empty list is fine "
                        "if you used general knowledge."
                    ),
                },
            },
            "required": ["content", "confidence", "sources"],
        },
    },
}
