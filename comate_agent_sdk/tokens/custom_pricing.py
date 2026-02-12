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
    "MiniMaxAI/MiniMax-M2.1": {
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
