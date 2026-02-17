"""Unified interface for LLM thinking/reasoning capabilities."""

from comate_agent_sdk.llm.thinking_presets import get_thinking_preset


def supports_thinking(llm: object) -> bool:
    """Check if LLM supports thinking capability based on its model preset.

    Args:
        llm: LLM instance to check (must have a .model attribute).

    Returns:
        True if the model's ThinkingPreset has supports_thinking=True.
    """
    model = getattr(llm, "model", None)
    if model is None:
        return False
    return get_thinking_preset(str(model)).supports_thinking
