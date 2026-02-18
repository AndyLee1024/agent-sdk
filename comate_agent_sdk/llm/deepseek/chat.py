"""ChatDeepSeek: DeepSeek LLM integration with thinking mode support.

DeepSeek API specifics:
- Uses OpenAI-compatible protocol
- Supports thinking mode via `reasoning_content` field in responses
- Requires `reasoning_content` to be preserved in tool call loops (same turn)
- Recommends cleaning historical `reasoning_content` on new turns for bandwidth optimization
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion

from comate_agent_sdk.llm.base import ToolChoice, ToolDefinition
from comate_agent_sdk.llm.deepseek.serializer import DeepSeekMessageSerializer
from comate_agent_sdk.llm.messages import (
    BaseMessage,
    ContentPartTextParam,
    ContentPartThinkingParam,
)
from comate_agent_sdk.llm.openai.like import ChatOpenAILike
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.tokens.custom_pricing import resolve_max_input_tokens_from_custom_pricing

logger = logging.getLogger("comate_agent_sdk.llm.deepseek")


@dataclass
class ChatDeepSeek(ChatOpenAILike):
    """
    DeepSeek LLM client with thinking mode (reasoning_content) support.

    DeepSeek provides extended thinking capabilities through the `reasoning_content`
    field in API responses. This class maps it to the unified thinking infrastructure
    (ContentPartThinkingParam, raw_content_blocks) for consistent handling across
    providers.

    Key features:
    - Extracts `reasoning_content` from responses and maps to `thinking`
    - Preserves structured content blocks for tool call loops
    - Auto-cleans historical `reasoning_content` on new turns (bandwidth optimization)
    - Uses custom serializer to convert thinking blocks back to `reasoning_content`

    Example:
        ```python
        from comate_agent_sdk.llm.deepseek import ChatDeepSeek
        from comate_agent_sdk.llm.messages import UserMessage

        llm = ChatDeepSeek(
            model='deepseek-reasoner',
            api_key='...',
            base_url='https://api.deepseek.com'
        )

        response = await llm.ainvoke(
            messages=[UserMessage(content="Explain quantum computing")],
            tools=None
        )

        # Access thinking content
        if response.thinking:
            print(f"Model reasoning: {response.thinking}")
        print(f"Final answer: {response.content}")
        ```

    Args:
        model: DeepSeek model name (e.g., 'deepseek-chat', 'deepseek-reasoner')
        api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
        base_url: API endpoint (defaults to https://api.deepseek.com)
        auto_clean_reasoning: Auto-clean historical reasoning_content on new turns (default: True)
    """

    model: str
    auto_clean_reasoning: bool = True
    """Automatically remove reasoning_content from historical messages on new turns.

    DeepSeek recommends this to save bandwidth, as historical thinking is not needed
    for new requests. Only the current turn's reasoning_content is preserved.
    """

    # Override default reasoning models list (DeepSeek uses different naming)
    reasoning_models: list[str] | None = field(
        default_factory=lambda: ["deepseek-reasoner"]
    )

    @property
    def provider(self) -> str:
        return "deepseek"

    def get_context_window(self) -> int | None:
        """Resolve context window from custom pricing data when available."""
        return resolve_max_input_tokens_from_custom_pricing(str(self.model))

    def get_usage(self, response: "Any") -> "ChatInvokeUsage | None":
        """Return usage statistics from a DeepSeek API response."""
        return self._get_usage(response)

    def _extract_reasoning_content(self, response: ChatCompletion) -> str | None:
        """Extract reasoning_content from DeepSeek API response.

        DeepSeek returns reasoning_content as an extra field in the message.
        We use getattr for safe access since it's not in the official OpenAI schema.

        Args:
            response: OpenAI ChatCompletion response

        Returns:
            Reasoning content string, or None if not present
        """
        message = response.choices[0].message
        return getattr(message, "reasoning_content", None)

    def _get_usage(self, response: Any) -> "ChatInvokeUsage | None":
        """Extract usage statistics from DeepSeek response.

        DeepSeek uses different cache field names than OpenAI:
        - OpenAI: prompt_tokens_details.cached_tokens
        - DeepSeek: prompt_cache_hit_tokens, prompt_cache_miss_tokens

        Args:
            response: ChatCompletion response from DeepSeek API

        Returns:
            Usage statistics with DeepSeek-specific cache fields
        """
        if response.usage is None:
            return None

        from comate_agent_sdk.llm.views import ChatInvokeUsage

        # DeepSeek 的缓存字段（从 extra 中提取）
        prompt_cache_hit = getattr(response.usage, "prompt_cache_hit_tokens", None)
        prompt_cache_miss = getattr(response.usage, "prompt_cache_miss_tokens", None)

        # 计算总的 prompt tokens（命中 + 未命中）
        # 注意：DeepSeek 的 prompt_tokens 可能只包含未命中的部分
        # 需要根据实际 API 返回调整
        total_prompt_tokens = response.usage.prompt_tokens
        if prompt_cache_hit and prompt_cache_miss:
            # 如果两个字段都存在，使用它们的和
            total_prompt_tokens = prompt_cache_hit + prompt_cache_miss

        return ChatInvokeUsage(
            prompt_tokens=total_prompt_tokens,
            prompt_cached_tokens=prompt_cache_hit,  # DeepSeek 的缓存命中
            prompt_cache_creation_tokens=None,  # DeepSeek 不单独统计缓存创建
            prompt_image_tokens=None,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _build_structured_content(
        self, reasoning_content: str | None, text_content: str | None
    ) -> list[ContentPartThinkingParam | ContentPartTextParam] | None:
        """Build structured content blocks (thinking + text).

        This creates the raw_content_blocks list that will be preserved in
        AssistantMessage for tool call loops.

        Args:
            reasoning_content: The thinking/reasoning content
            text_content: The final answer text

        Returns:
            List of content blocks, or None if both are empty
        """
        blocks: list[ContentPartThinkingParam | ContentPartTextParam] = []

        # Add thinking block first (matches Anthropic order)
        if reasoning_content:
            blocks.append(
                ContentPartThinkingParam(
                    thinking=reasoning_content,
                    signature=None,  # DeepSeek doesn't use signatures
                )
            )

        # Add text content
        if text_content:
            blocks.append(ContentPartTextParam(text=text_content))

        return blocks if blocks else None

    def _should_clean_historical_reasoning(self, messages: list[BaseMessage]) -> bool:
        """Check if we should clean historical reasoning_content.

        Cleaning is triggered when:
        1. auto_clean_reasoning is enabled
        2. We're starting a new turn (last message is from user)
        3. There's message history to clean

        Args:
            messages: Message list

        Returns:
            True if should clean historical reasoning
        """
        if not self.auto_clean_reasoning:
            return False

        if len(messages) < 2:
            # No history to clean
            return False

        # Check if last message is user (new turn)
        from comate_agent_sdk.llm.messages import UserMessage

        return isinstance(messages[-1], UserMessage)

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        """
        Invoke DeepSeek with messages and optional tools.

        This method:
        1. Cleans historical reasoning_content if starting a new turn
        2. Calls the OpenAI-compatible API
        3. Extracts reasoning_content from response
        4. Maps to unified thinking infrastructure

        Args:
            messages: List of chat messages
            tools: Optional list of tools the model can call
            tool_choice: Control how the model uses tools
            **kwargs: Additional arguments passed to parent

        Returns:
            ChatInvokeCompletion with content, thinking, and/or tool_calls
        """
        # Clean historical reasoning_content on new turns
        if self._should_clean_historical_reasoning(messages):
            logger.debug(
                "New turn detected, cleaning historical reasoning_content from messages"
            )
            messages = DeepSeekMessageSerializer.strip_thinking_blocks(messages)

        # Serialize messages using DeepSeek serializer (handles thinking -> reasoning_content)
        openai_messages = DeepSeekMessageSerializer.serialize_messages(messages)

        try:
            # Call parent's implementation to get raw OpenAI response
            # We can't use super().ainvoke() because it returns ChatInvokeCompletion
            # Instead, we need to duplicate the call logic but extract reasoning_content
            model_params: dict[str, Any] = {}

            if self.temperature is not None:
                model_params["temperature"] = self.temperature

            if self.frequency_penalty is not None:
                model_params["frequency_penalty"] = self.frequency_penalty

            if self.max_completion_tokens is not None:
                model_params["max_completion_tokens"] = self.max_completion_tokens

            if self.top_p is not None:
                model_params["top_p"] = self.top_p

            if self.seed is not None:
                model_params["seed"] = self.seed

            if self.service_tier is not None:
                model_params["service_tier"] = self.service_tier

            extra_body: dict[str, Any] = {}
            if self.prompt_cache_key is not None:
                extra_body["prompt_cache_key"] = self.prompt_cache_key
            cache_retention = self._resolve_prompt_cache_retention()
            if cache_retention is not None:
                extra_body["prompt_cache_retention"] = cache_retention
            if extra_body:
                model_params["extra_body"] = extra_body

            # Handle reasoning models (DeepSeek reasoner)
            if self.reasoning_models and any(
                str(m).lower() in str(self.model).lower() for m in self.reasoning_models
            ):
                model_params["reasoning_effort"] = self.reasoning_effort
                model_params.pop("temperature", None)
                model_params.pop("frequency_penalty", None)

            # Add tools if provided
            if tools:
                model_params["tools"] = self._serialize_tools(tools)
                model_params["parallel_tool_calls"] = self.parallel_tool_calls

                openai_tool_choice = self._get_tool_choice(tool_choice, tools)
                if openai_tool_choice is not None:
                    model_params["tool_choice"] = openai_tool_choice

            # Make the API call
            response = await self.get_client().chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **model_params,
            )

            # Extract usage (use DeepSeek-specific implementation)
            usage = self._get_usage(response)

            # Log token usage if comate_agent_sdk_LLM_DEBUG is set
            import os

            if usage and os.getenv("comate_agent_sdk_LLM_DEBUG"):
                cached = usage.prompt_cached_tokens or 0
                input_tokens = usage.prompt_tokens - cached
                logger.info(
                    f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached (hit) + {usage.completion_tokens:,} out"
                )
                # DeepSeek specific: show cache efficiency if available
                if cached > 0:
                    cache_rate = (cached / usage.prompt_tokens * 100) if usage.prompt_tokens > 0 else 0
                    logger.info(f"   💾 Cache hit rate: {cache_rate:.1f}%")

            # Extract DeepSeek-specific reasoning_content
            reasoning_content = self._extract_reasoning_content(response)

            # Extract standard content and tool calls
            content = response.choices[0].message.content
            tool_calls = self._extract_tool_calls(response)

            # Build structured content blocks (for tool call loop preservation)
            raw_content_blocks = self._build_structured_content(
                reasoning_content, content
            )

            # Log reasoning extraction
            if reasoning_content:
                logger.debug(
                    f"Extracted reasoning_content: {len(reasoning_content)} chars"
                )

            return ChatInvokeCompletion(
                content=content,
                thinking=reasoning_content,  # Unified thinking field
                raw_content_blocks=raw_content_blocks,  # Structured blocks for history
                tool_calls=tool_calls,
                usage=usage,
                stop_reason=response.choices[0].finish_reason
                if response.choices
                else None,
            )

        except Exception as e:
            # Let parent's exception handling deal with it
            # This will convert OpenAI errors to ModelProviderError
            from comate_agent_sdk.llm.exceptions import ModelProviderError

            if isinstance(e, ModelProviderError):
                raise
            raise ModelProviderError(message=str(e), model=self.name) from e
