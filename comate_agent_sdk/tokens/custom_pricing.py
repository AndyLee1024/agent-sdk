"""
Custom model pricing for models not available in LiteLLM's pricing data.

Prices are per token (not per 1M tokens).
"""

from typing import Any

# Custom model pricing data
# Format matches LiteLLM's model_prices_and_context_window.json structure
CUSTOM_MODEL_PRICING: dict[str, dict[str, Any]] = {}
