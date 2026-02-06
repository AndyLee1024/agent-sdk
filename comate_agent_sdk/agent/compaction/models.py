"""
Models for the compaction subservice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comate_agent_sdk.llm.views import ChatInvokeUsage

# Default ratio of context window to use before triggering compaction
DEFAULT_THRESHOLD_RATIO = 0.80

DEFAULT_SUMMARY_PROMPT = """You have been working on the task described above but have not yet completed it. Write a continuation summary that will allow you (or another instance of yourself) to resume work efficiently in a future context window where the conversation history will be replaced with this summary. Your summary should be structured, concise, and actionable. Include:

1. Task Overview
The user's core request and success criteria
Any clarifications or constraints they specified

2. Current State
What has been completed so far
Files created, modified, or analyzed (with paths if relevant)
Key outputs or artifacts produced

3. Important Discoveries
Technical constraints or requirements uncovered
Decisions made and their rationale
Errors encountered and how they were resolved
What approaches were tried that didn't work (and why)

4. Next Steps
Specific actions needed to complete the task
Any blockers or open questions to resolve
Priority order if multiple steps remain

5. Context to Preserve
User preferences or style requirements
Domain-specific details that aren't obvious
Any promises made to the user

Be concise but complete - err on the side of including information that would prevent duplicate work or repeated mistakes. Write in a way that enables immediate resumption of the task.

Wrap your summary in <summary></summary> tags."""


@dataclass
class CompactionConfig:
    """Configuration for the compaction service.

    The compaction service monitors token usage and automatically summarizes
    conversation history when approaching the model's context window limit.

    Attributes:
            enabled: Whether compaction is enabled. Defaults to True.
            threshold_ratio: Ratio of context window at which compaction triggers (0.0-1.0).
                    E.g., 0.80 means compact when context reaches 80% of model's limit.
            model: Optional model to use for generating summaries. If None, uses the agent's model.
            summary_prompt: Custom prompt for summary generation.
    """

    enabled: bool = True
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    model: str | None = None
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT


@dataclass
class CompactionResult:
    """Result of a compaction operation.

    Attributes:
            compacted: Whether compaction was performed.
            original_tokens: Token count before compaction.
            new_tokens: Token count after compaction (estimated from summary output tokens).
            summary: The generated summary text (if compaction was performed).
    """

    compacted: bool
    original_tokens: int = 0
    new_tokens: int = 0
    summary: str | None = None


@dataclass
class TokenUsage:
    """Token usage tracking for compaction decisions.

    Attributes:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cache_creation_tokens: Number of tokens used to create cache (Anthropic).
            cache_read_tokens: Number of cached tokens read.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens for compaction threshold check.

        This matches the Anthropic SDK's calculation:
        input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens
        """
        return (
            self.input_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
            + self.output_tokens
        )

    @classmethod
    def from_usage(cls, usage: ChatInvokeUsage | None) -> TokenUsage:
        """Create TokenUsage from ChatInvokeUsage."""
        if usage is None:
            return cls()

        return cls(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_creation_tokens=usage.prompt_cache_creation_tokens or 0,
            cache_read_tokens=usage.prompt_cached_tokens or 0,
        )
