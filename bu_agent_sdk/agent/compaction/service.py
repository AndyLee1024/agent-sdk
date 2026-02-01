"""
Compaction service for managing conversation context.

This service monitors token usage and automatically compresses conversation
history when it approaches the model's context window limit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bu_agent_sdk.agent.compaction.models import (
    CompactionConfig,
    CompactionResult,
    TokenUsage,
)
from bu_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from bu_agent_sdk.llm.base import BaseChatModel
    from bu_agent_sdk.llm.views import ChatInvokeUsage
    from bu_agent_sdk.tokens import TokenCost

log = logging.getLogger(__name__)

# Default context window if model info not available
DEFAULT_CONTEXT_WINDOW = 128_000


@dataclass
class CompactionService:
    """Service for managing conversation context through compaction.

    The service monitors token usage after each LLM response and triggers
    compaction when the threshold is exceeded. During compaction:
    1. The conversation history is sent to an LLM with a summary prompt
    2. The LLM generates a structured summary
    3. The entire message history is replaced with the summary

    The threshold is calculated dynamically based on the model's context window:
    threshold = context_window * threshold_ratio

    Attributes:
            config: Configuration for compaction behavior.
            llm: The language model to use for generating summaries.
                 If None, must be set before calling check_and_compact.
            token_cost: TokenCost service for fetching model context limits.
    """

    config: CompactionConfig = field(default_factory=CompactionConfig)
    llm: BaseChatModel | None = None
    token_cost: TokenCost | None = None

    # Internal state
    _last_usage: TokenUsage = field(default_factory=TokenUsage, repr=False)
    _context_limit_cache: dict[str, int] = field(default_factory=dict, repr=False)
    _threshold_cache: dict[str, int] = field(default_factory=dict, repr=False)

    def update_usage(self, usage: ChatInvokeUsage | None) -> None:
        """Update the tracked token usage from a response.

        Args:
                usage: The usage information from the last LLM response.
        """
        self._last_usage = TokenUsage.from_usage(usage)

    async def get_model_context_limit(self, model: str) -> int:
        """Get the context window limit for a model."""
        # Check cache first
        if model in self._context_limit_cache:
            return self._context_limit_cache[model]

        context_limit = DEFAULT_CONTEXT_WINDOW

        if self.token_cost is not None:
            try:
                pricing = await self.token_cost.get_model_pricing(model)
                if pricing:
                    # Use max_input_tokens if available, otherwise max_tokens
                    if pricing.max_input_tokens:
                        context_limit = pricing.max_input_tokens
                    elif pricing.max_tokens:
                        context_limit = pricing.max_tokens
            except Exception as e:
                log.debug(f"Failed to fetch model pricing for {model}: {e}")

        # Cache the result
        self._context_limit_cache[model] = context_limit
        log.debug(f"Model {model} context limit: {context_limit}")
        return context_limit

    async def get_threshold_for_model(self, model: str) -> int:
        """Get the compaction threshold for a specific model."""
        # Check cache first
        if model in self._threshold_cache:
            return self._threshold_cache[model]

        context_limit = await self.get_model_context_limit(model)
        threshold = int(context_limit * self.config.threshold_ratio)

        # Cache the result
        self._threshold_cache[model] = threshold
        log.debug(
            f"Model {model} compaction threshold: {threshold} ({self.config.threshold_ratio * 100:.0f}% of {context_limit})"
        )
        return threshold

    async def should_compact(self, model: str) -> bool:
        """Check if compaction should be triggered based on current token usage.

        Returns:
                True if token usage exceeds the threshold and compaction is enabled.
        """
        if not self.config.enabled:
            return False

        threshold = await self.get_threshold_for_model(model)
        should = self._last_usage.total_tokens >= threshold

        if should:
            log.info(
                f"Compaction triggered: {self._last_usage.total_tokens} tokens >= {threshold} threshold "
                f"(model: {model}, ratio: {self.config.threshold_ratio})"
            )

        return should

    async def compact(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel | None = None,
        *,
        level: str | None = None,
    ) -> CompactionResult:
        """Perform compaction on the message history.

        This method:
        1. Prepares the messages for summarization (removing pending tool calls)
        2. Appends the summary prompt as a user message
        3. Calls the LLM to generate a summary
        4. Extracts the summary and returns it

        Args:
                messages: The current message history to compact.
                llm: Optional LLM to use for summarization. Falls back to self.llm.

        Returns:
                CompactionResult containing the summary and token information.

        Raises:
                ValueError: If no LLM is available for summarization.
        """
        model = llm or self.llm
        if model is None:
            raise ValueError(
                "No LLM available for compaction. Provide an LLM or set self.llm."
            )

        original_tokens = self._last_usage.total_tokens
        threshold = await self.get_threshold_for_model(model.model)

        log.info(
            f"Token usage {original_tokens} has exceeded the threshold of "
            f"{threshold}. Performing compaction."
        )

        # Prepare messages for summarization
        prepared_messages = self._prepare_messages_for_summary(messages)

        # Add the summary prompt
        prepared_messages.append(UserMessage(content=self.config.summary_prompt))

        # Generate the summary
        response = await model.ainvoke(messages=prepared_messages)

        if response.usage and self.token_cost is not None:
            self.token_cost.add_usage(
                model.model,
                response.usage,
                level=level,
                source="compaction",
            )

        summary_text = response.content or ""

        # Extract summary from tags if present
        extracted_summary = self._extract_summary(summary_text)

        new_tokens = response.usage.completion_tokens if response.usage else 0

        log.info(f"Compaction complete. New token usage: {new_tokens}")

        return CompactionResult(
            compacted=True,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
            summary=extracted_summary,
        )

    async def check_and_compact(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel | None = None,
    ) -> tuple[list[BaseMessage], CompactionResult]:
        """Check token usage and compact if threshold exceeded.

        This is the main entry point for the compaction service. It checks
        if compaction is needed and performs it if so.

        Args:
                messages: The current message history.
                llm: Optional LLM to use for summarization.

        Returns:
                A tuple of (new_messages, result) where new_messages is either
                the original messages (if no compaction) or a single summary
                message (if compacted).
        """
        model = llm or self.llm
        if model is None:
            return messages, CompactionResult(compacted=False)

        if not await self.should_compact(model.model):
            return messages, CompactionResult(compacted=False)

        result = await self.compact(messages, llm)

        # Replace entire history with summary as a user message
        # This matches the Anthropic SDK behavior
        new_messages: list[BaseMessage] = [
            UserMessage(content=result.summary or ""),
        ]

        return new_messages, result

    def create_compacted_messages(self, summary: str) -> list[BaseMessage]:
        """Create a new message list from a summary.

        Args:
                summary: The summary text to use as the new conversation start.

        Returns:
                A list containing a single user message with the summary.
        """
        return [UserMessage(content=summary)]

    def _prepare_messages_for_summary(
        self,
        messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        """Prepare messages for summarization.

        This removes tool_calls from the last assistant message to avoid
        API errors (tool_use requires tool_result which we won't have).

        Args:
                messages: The original message history.

        Returns:
                A cleaned copy of the messages suitable for summarization.
        """
        if not messages:
            return []

        # Make a copy to avoid modifying the original
        prepared: list[BaseMessage] = []

        for i, msg in enumerate(messages):
            is_last = i == len(messages) - 1

            if is_last and isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Remove tool_calls from the last assistant message
                # Keep the content if there is any text
                if msg.content:
                    prepared.append(
                        AssistantMessage(
                            content=msg.content,
                            tool_calls=None,
                        )
                    )
                # If no content, skip this message entirely
            else:
                prepared.append(msg)

        return prepared

    def _extract_summary(self, text: str) -> str:
        """Extract summary content from <summary></summary> tags.

        If tags are not found, returns the original text.

        Args:
                text: The response text that may contain summary tags.

        Returns:
                The extracted summary or the original text.
        """
        # Try to extract content between <summary> tags
        pattern = r"<summary>(.*?)</summary>"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # No tags found, return original text
        return text.strip()

    def reset(self) -> None:
        """Reset the service state.

        Clears tracked token usage and cached thresholds.
        """
        self._last_usage = TokenUsage()
        self._context_limit_cache.clear()
        self._threshold_cache.clear()
