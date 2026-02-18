"""
Compaction service for managing conversation context.

This service monitors token usage and automatically compresses conversation
history when it approaches the model's context window limit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from html import escape as xml_escape
from typing import TYPE_CHECKING

from comate_agent_sdk.agent.compaction.models import (
    CompactionConfig,
    CompactionResult,
)
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.tokens.custom_pricing import (
    resolve_max_input_tokens_from_custom_pricing,
)

if TYPE_CHECKING:
    from comate_agent_sdk.llm.base import BaseChatModel
    from comate_agent_sdk.tokens import TokenCost

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
    usage_source: str = "compaction"

    # Internal state
    _context_limit_cache: dict[str, int] = field(default_factory=dict, repr=False)
    _threshold_cache: dict[str, int] = field(default_factory=dict, repr=False)

    def _resolve_context_limit_from_provider(
        self,
        *,
        model: str,
        llm: BaseChatModel | None,
    ) -> int | None:
        if llm is None:
            return None

        getter = getattr(llm, "get_context_window", None)
        if not callable(getter):
            return None

        try:
            resolved = getter()
        except Exception as e:
            log.debug(f"Failed to get provider context window for {model}: {e}")
            return None

        if resolved is None:
            return None

        try:
            context_limit = int(resolved)
        except (TypeError, ValueError):
            return None

        return context_limit if context_limit > 0 else None

    async def get_model_context_limit(
        self,
        model: str,
        *,
        llm: BaseChatModel | None = None,
    ) -> int:
        """Get the context window limit for a model."""
        # Check cache first
        if model in self._context_limit_cache:
            return self._context_limit_cache[model]

        context_limit = DEFAULT_CONTEXT_WINDOW
        provider_limit = self._resolve_context_limit_from_provider(model=model, llm=llm)
        if provider_limit is None:
            provider_limit = self._resolve_context_limit_from_provider(
                model=model,
                llm=self.llm,
            )
        if provider_limit is not None:
            context_limit = provider_limit

        custom_pricing_limit = None
        if provider_limit is None:
            custom_pricing_limit = resolve_max_input_tokens_from_custom_pricing(model)
            if custom_pricing_limit is not None:
                context_limit = custom_pricing_limit

        if provider_limit is None and custom_pricing_limit is None and self.token_cost is not None:
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

    async def get_threshold_for_model(
        self,
        model: str,
        *,
        llm: BaseChatModel | None = None,
    ) -> int:
        """Get the compaction threshold for a specific model."""
        # Check cache first
        if model in self._threshold_cache:
            return self._threshold_cache[model]

        context_limit = await self.get_model_context_limit(model, llm=llm)
        threshold = int(context_limit * self.config.threshold_ratio)

        # Cache the result
        self._threshold_cache[model] = threshold
        log.debug(
            f"Model {model} compaction threshold: {threshold} ({self.config.threshold_ratio * 100:.0f}% of {context_limit})"
        )
        return threshold

    async def should_compact(
        self,
        model: str,
        *,
        context_usage: int,
        llm: BaseChatModel | None = None,
    ) -> bool:
        """Check if compaction should be triggered based on current token usage.

        Args:
                model: The model name to look up threshold for.
                context_usage: Current context token count (from ContextUsageTracker).
                llm: Optional LLM instance for provider context window resolution.

        Returns:
                True if context_usage exceeds the threshold and compaction is enabled.
        """
        if not self.config.enabled:
            return False

        threshold = await self.get_threshold_for_model(model, llm=llm)
        should = context_usage >= threshold

        if should:
            log.info(
                f"Compaction triggered: {context_usage} tokens >= {threshold} threshold "
                f"(model: {model}, ratio: {self.config.threshold_ratio})"
            )

        return should

    async def compact(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel | None = None,
        *,
        level: str | None = None,
        original_tokens: int = 0,
    ) -> CompactionResult:
        """Perform compaction on the message history.

        This method:
        1. Serializes conversation messages into XML-like plain text
        2. Builds an isolated summarization call (system + user message)
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

        threshold = await self.get_threshold_for_model(model.model, llm=model)

        log.info(
            f"Token usage {original_tokens} has exceeded the threshold of "
            f"{threshold}. Performing compaction."
        )

        serialized = self._serialize_messages_to_text(messages)
        summary_messages: list[BaseMessage] = [
            SystemMessage(content=self.config.summary_system_prompt),
            UserMessage(content=serialized),
        ]

        # Generate the summary
        response = await model.ainvoke(messages=summary_messages)

        if response.usage and self.token_cost is not None:
            self.token_cost.add_usage(
                model.model,
                response.usage,
                level=level,
                source=self.usage_source,
            )

        summary_text = response.content or ""

        # Extract summary from tags if present
        extracted_summary = self._extract_summary(summary_text)
        normalized_summary = extracted_summary.strip()
        stop_reason = getattr(response, "stop_reason", None)
        failure_reason: str | None = None
        failure_detail: str | None = None
        compacted = True
        raw_content_length = len(response.content) if response.content is not None else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        has_summary_tags = bool(re.search(r"<summary>.*?</summary>", summary_text, re.DOTALL))
        empty_tag_body = bool(re.search(r"<summary>\s*</summary>", summary_text, re.DOTALL))
        blank_content = bool(response.content is not None and not summary_text.strip())

        if not normalized_summary:
            compacted = False
            if response.content is None:
                failure_reason = "content_none"
            elif stop_reason == "max_tokens":
                failure_reason = "max_tokens_no_content"
            elif empty_tag_body:
                failure_reason = "empty_tag_body"
            elif getattr(response, "thinking", None) and blank_content:
                failure_reason = "thinking_only_no_content"
            elif blank_content:
                failure_reason = "blank_content"
            else:
                failure_reason = "empty_summary"
            failure_detail = (
                "diag("
                f"raw_content_len={raw_content_length},"
                f"completion_tokens={completion_tokens},"
                f"has_summary_tags={has_summary_tags},"
                f"empty_tag_body={empty_tag_body}"
                ")"
            )

        new_tokens = response.usage.completion_tokens if response.usage else 0

        if compacted:
            log.info(f"Compaction complete. New token usage: {new_tokens}")
        else:
            log.warning(
                "Compaction summary is empty; summary replacement skipped. "
                f"reason={failure_reason}, stop_reason={stop_reason}, {failure_detail}"
            )

        return CompactionResult(
            compacted=compacted,
            original_tokens=original_tokens,
            new_tokens=new_tokens,
            summary=normalized_summary or None,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
            stop_reason=stop_reason,
            raw_content_length=raw_content_length,
        )

    async def check_and_compact(
        self,
        messages: list[BaseMessage],
        llm: BaseChatModel | None = None,
        *,
        context_usage: int = 0,
    ) -> tuple[list[BaseMessage], CompactionResult]:
        """Check token usage and compact if threshold exceeded.

        This is the main entry point for the compaction service. It checks
        if compaction is needed and performs it if so.

        Args:
                messages: The current message history.
                llm: Optional LLM to use for summarization.
                context_usage: Current context token count (from ContextUsageTracker).

        Returns:
                A tuple of (new_messages, result) where new_messages is either
                the original messages (if no compaction) or a single summary
                message (if compacted).
        """
        model = llm or self.llm
        if model is None:
            return messages, CompactionResult(compacted=False)

        if not await self.should_compact(model.model, context_usage=context_usage, llm=model):
            return messages, CompactionResult(compacted=False)

        result = await self.compact(messages, llm, original_tokens=context_usage)
        if not result.compacted or not result.summary:
            return messages, result

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

    def _serialize_messages_to_text(self, messages: list[BaseMessage]) -> str:
        """Serialize conversation messages into XML-like plain text."""
        lines = ["<conversation>"]

        for message in messages:
            if isinstance(message, UserMessage):
                lines.append('  <message role="user">')
                self._append_escaped_text_lines(lines, message.text, indent=4)
                lines.append("  </message>")
                continue

            if isinstance(message, AssistantMessage):
                lines.append('  <message role="assistant">')
                self._append_escaped_text_lines(lines, message.text, indent=4)
                if message.tool_calls:
                    lines.append("    <tool_calls>")
                    for call in message.tool_calls:
                        call_line = self._render_tool_call_line(call.function.name, call.function.arguments)
                        lines.append(f"      {xml_escape(call_line, quote=True)}")
                    lines.append("    </tool_calls>")
                lines.append("  </message>")
                continue

            if isinstance(message, ToolMessage):
                tool_name = xml_escape(message.tool_name, quote=True)
                tool_call_id = xml_escape(message.tool_call_id, quote=True)
                lines.append(
                    f'  <message role="tool" name="{tool_name}" tool_call_id="{tool_call_id}">'
                )
                self._append_escaped_text_lines(lines, message.text, indent=4)
                lines.append("  </message>")
                continue

            lines.append('  <message role="unknown">')
            unknown_text = getattr(message, "text", "")
            if not isinstance(unknown_text, str) or not unknown_text:
                raw_content = getattr(message, "content", "")
                unknown_text = "" if raw_content is None else str(raw_content)
            self._append_escaped_text_lines(lines, unknown_text, indent=4)
            lines.append("  </message>")

        lines.append("</conversation>")
        return "\n".join(lines)

    def _append_escaped_text_lines(self, lines: list[str], text: str, *, indent: int) -> None:
        if not text:
            return
        prefix = " " * indent
        for row in text.splitlines():
            lines.append(f"{prefix}{xml_escape(row, quote=True)}")

    def _render_tool_call_line(self, name: str, arguments: str) -> str:
        return f"- {name}({arguments})"

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

        Clears cached context limits and thresholds.
        """
        self._context_limit_cache.clear()
        self._threshold_cache.clear()
