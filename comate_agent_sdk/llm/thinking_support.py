"""Unified interface for LLM thinking/reasoning capabilities."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ThinkingCapable(Protocol):
    """Protocol for LLMs that support extended thinking/reasoning.

    Providers implement this by adding a set_thinking_budget() method
    that adapts the budget parameter to their internal representation:
    - Anthropic: thinking_budget_tokens (int | None)
    - Google: thinking_budget (int | None)
    - OpenAI: reasoning_effort ("low" | "medium" | "high")
    """

    def set_thinking_budget(self, budget: int | None) -> None:
        """Set thinking token budget (None to disable).

        Args:
            budget: Token budget for thinking, or None to disable.
                   Each provider interprets this differently:
                   - Anthropic/Google: Direct token budget
                   - OpenAI: Converted to reasoning effort level
        """
        ...


def supports_thinking(llm: object) -> bool:
    """Check if LLM supports thinking capability.

    Args:
        llm: LLM instance to check.

    Returns:
        True if the LLM has a set_thinking_budget() method.
    """
    return hasattr(llm, "set_thinking_budget")
