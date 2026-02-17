"""
Custom model pricing for models not available in LiteLLM's pricing data.

Prices are per token (not per 1M tokens).
"""

from typing import Any

# Custom model pricing data
# Format matches LiteLLM's model_prices_and_context_window.json structure
CUSTOM_MODEL_PRICING: dict[str, dict[str, Any]] = {
    "glm-4.7": {
        "max_input_tokens": 202752,
        "max_output_tokens": 16384,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "zai-org/GLM-5-FP8": {
        "max_input_tokens": 202752,
        "max_output_tokens": 16384,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "gpt-4.1": {
        "max_input_tokens": 111424,
        "max_output_tokens": 16384,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "gpt-5-mini": {
        "max_input_tokens": 127805,
        "max_output_tokens": 64000,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "gpt-4o": {
        "max_input_tokens": 63805,
        "max_output_tokens": 4096,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "grok-code-fast-1": {
        "max_input_tokens": 108609,
        "max_output_tokens": 64000,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "oswe-vscode-prime": {
        "max_input_tokens": 199804,
        "max_output_tokens": 64000,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
    "moonshotai/Kimi-K2.5": {
        "max_input_tokens": 262144,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "Kimi-K2.5": {
        "max_input_tokens": 262144,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "deepseek-chat": {
        "max_input_tokens": 128000,
        "max_output_tokens": 8000,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "deepseek-reasoner": {
        "max_input_tokens": 128000,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "MiniMax-M2.1": {
        "max_input_tokens": 204800,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "MiniMax-M2.5": {
        "max_input_tokens": 204800,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "MiniMaxAI/MiniMax-M2.5": {
        "max_input_tokens": 204800,
        "max_output_tokens": 16384,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
    "kimi-for-coding":{
        "max_input_tokens": 262144,
        "max_output_tokens": 32768,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    }
}


def resolve_max_input_tokens_from_custom_pricing(model_name: str) -> int | None:
    """Resolve max_input_tokens from custom pricing by model name.

    Lookup strategy:
    1. Exact key match
    2. Case-insensitive exact match
    3. Strip provider prefix and retry (e.g. "vendor/model" -> "model")
    4. Add common provider prefixes and retry (e.g. "model" -> "anthropic/model")
    """
    model = (model_name or "").strip()
    if not model:
        return None

    def _get_max_input_tokens(key: str) -> int | None:
        pricing = CUSTOM_MODEL_PRICING.get(key)
        if pricing is None:
            return None
        value = pricing.get("max_input_tokens")
        if value is None:
            return None
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            return None
        return tokens if tokens > 0 else None

    direct = _get_max_input_tokens(model)
    if direct is not None:
        return direct

    lower_lookup = {k.lower(): k for k in CUSTOM_MODEL_PRICING}
    lower_key = lower_lookup.get(model.lower())
    if lower_key is not None:
        return _get_max_input_tokens(lower_key)

    candidates: list[str] = []
    if "/" in model:
        _, bare = model.split("/", 1)
        if bare:
            candidates.append(bare)
    else:
        candidates.extend(
            [
                f"anthropic/{model}",
                f"minimax/{model}",
                f"MiniMaxAI/{model}",
            ]
        )

    for candidate in candidates:
        resolved = _get_max_input_tokens(candidate)
        if resolved is not None:
            return resolved

        lower_candidate_key = lower_lookup.get(candidate.lower())
        if lower_candidate_key is not None:
            resolved = _get_max_input_tokens(lower_candidate_key)
            if resolved is not None:
                return resolved

    return None
