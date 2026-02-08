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
    "glm-4.7-thinking": {
        "max_input_tokens": 202752,
        "max_output_tokens": 16384,
        "input_cost_per_token": 4e-07,
        "output_cost_per_token": 2e-06,
    },
}
