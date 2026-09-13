"""Convert token usage into USD using a per-model rate table."""

# USD per 1M tokens: (input_rate, output_rate).
# Order-of-magnitude rates; confirm against current provider pricing
# before quoting in production.
RATES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama3.2:3b": (0.0, 0.0),
}


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD for a prompt/completion token pair.

    Unknown models return 0.0 so a missing rate does not fail a batch.
    """
    rates = RATES.get(model)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0
