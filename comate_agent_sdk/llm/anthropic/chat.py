import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("comate_agent_sdk.llm.anthropic")
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    NotGiven,
    RateLimitError,
    omit,
)
from anthropic.types import CacheControlEphemeralParam, Message, ToolParam
from anthropic.types.model_param import ModelParam
from anthropic.types.redacted_thinking_block import RedactedThinkingBlock
from anthropic.types.text_block import TextBlock
from anthropic.types.thinking_block import ThinkingBlock
from anthropic.types.tool_choice_any_param import ToolChoiceAnyParam
from anthropic.types.tool_choice_auto_param import ToolChoiceAutoParam
from anthropic.types.tool_choice_none_param import ToolChoiceNoneParam
from anthropic.types.tool_choice_tool_param import ToolChoiceToolParam
from anthropic.types.tool_use_block import ToolUseBlock
from httpx import Timeout

from comate_agent_sdk.llm.anthropic.serializer import AnthropicMessageSerializer
from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from comate_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from comate_agent_sdk.llm.thinking_presets import get_thinking_preset, resolve_api_model_name
from comate_agent_sdk.llm.messages import (
    BaseMessage,
    ContentPartRedactedThinkingParam,
    ContentPartTextParam,
    ContentPartThinkingParam,
    Function,
    ToolCall,
)
from comate_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from comate_agent_sdk.tokens.custom_pricing import (
    resolve_max_input_tokens_from_custom_pricing,
)


@dataclass
class ChatAnthropic(BaseChatModel):
    """
    A wrapper around Anthropic's chat model.
    """

    # Model configuration
    model: str | ModelParam
    max_tokens: int = 16384
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    # Client initialization parameters
    api_key: str | None = None
    auth_token: str | None = None
    base_url: str | httpx.URL | None = None
    timeout: float | Timeout | None | NotGiven = NotGiven()
    max_retries: int = 10
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None
    http_client: httpx.AsyncClient | None = None
    prompt_cache_beta: str | None = "prompt-caching-2024-07-31"
    max_cached_tool_definitions: int = 1

    # Internal client cache
    _client: AsyncAnthropic | None = field(default=None, repr=False, compare=False)

    # Static
    @property
    def provider(self) -> str:
        return "anthropic"

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
        headers: dict[str, str] = {}
        if self.default_headers is not None:
            headers = dict(self.default_headers)
        if self.prompt_cache_beta:
            existing = headers.get("anthropic-beta")
            if existing:
                if self.prompt_cache_beta not in existing:
                    headers["anthropic-beta"] = f"{existing}, {self.prompt_cache_beta}"
            else:
                headers["anthropic-beta"] = self.prompt_cache_beta

        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "auth_token": self.auth_token,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": headers or None,
            "default_query": self.default_query,
            "http_client": self.http_client,
        }

        # Create client_params dict with non-None values and non-NotGiven values
        client_params = {}
        for k, v in base_params.items():
            if v is not None and v is not NotGiven():
                client_params[k] = v

        return client_params

    def _get_client_params_for_invoke(self):
        """Prepare client parameters dictionary for invoke."""

        client_params = {}

        if self.temperature is not None:
            client_params["temperature"] = self.temperature

        if self.max_tokens is not None:
            client_params["max_tokens"] = self.max_tokens

        if self.top_p is not None:
            client_params["top_p"] = self.top_p

        if self.seed is not None:
            client_params["seed"] = self.seed

        # Extended thinking: enable and drop temperature (not allowed with thinking)
        preset = get_thinking_preset(str(self.model))
        if preset.supports_thinking and preset.budget_tokens is not None:
            client_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": preset.budget_tokens,
            }
            client_params.pop("temperature", None)

        return client_params

    _langfuse_instrumented: bool = field(default=False, repr=False, compare=False)

    def _is_langfuse_configured(self) -> bool:
        """Check if all required Langfuse environment variables are set."""
        required_vars = [
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_BASE_URL",
        ]
        return all(os.getenv(var) for var in required_vars)

    def _ensure_langfuse_instrumentation(self) -> None:
        """Initialize Langfuse OpenTelemetry instrumentation for Anthropic if configured."""
        if self._langfuse_instrumented:
            return

        if self._is_langfuse_configured():
            try:
                from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

                # Only instrument if not already instrumented
                if not AnthropicInstrumentor().is_instrumented_by_opentelemetry:
                    AnthropicInstrumentor().instrument()
                    logger.debug("Langfuse OpenTelemetry instrumentation enabled for Anthropic")
                self._langfuse_instrumented = True
            except ImportError:
                logger.warning(
                    "Langfuse environment variables detected but opentelemetry-instrumentation-anthropic "
                    "is not installed. Install it with: pip install opentelemetry-instrumentation-anthropic"
                )
        else:
            logger.debug("Langfuse not configured, using native Anthropic client without instrumentation")

    def get_client(self) -> AsyncAnthropic:
        """
        Returns an AsyncAnthropic client (cached).

        If Langfuse environment variables (LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY,
        LANGFUSE_BASE_URL) are all set, enables OpenTelemetry instrumentation for
        observability.

        Returns:
                AsyncAnthropic: An instance of the AsyncAnthropic client.
        """
        if self._client is not None:
            return self._client

        async_http_client = httpx.AsyncClient(verify=False)
        # Enable Langfuse instrumentation if configured
        self._ensure_langfuse_instrumentation()

        client_params = self._get_client_params()
        self._client = AsyncAnthropic(**client_params, http_client=async_http_client)
        return self._client

    @property
    def name(self) -> str:
        return str(self.model)

    def get_context_window(self) -> int | None:
        """Resolve context window from custom pricing data when available."""
        return resolve_max_input_tokens_from_custom_pricing(str(self.model))

    def get_usage(self, response: Message) -> ChatInvokeUsage | None:
        """Map Anthropic usage into SDK usage fields.

        Anthropic prompt usage is composed of:
        - input_tokens (non-cached suffix)
        - cache_read_input_tokens (cached prefix read)
        - cache_creation_input_tokens (cached prefix write)
        """
        cache_read = response.usage.cache_read_input_tokens or 0
        cache_creation = response.usage.cache_creation_input_tokens or 0
        prompt_tokens = response.usage.input_tokens + cache_read + cache_creation
        total_tokens = prompt_tokens + response.usage.output_tokens
        usage = ChatInvokeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=total_tokens,
            prompt_cached_tokens=response.usage.cache_read_input_tokens,
            prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
            prompt_image_tokens=None,
        )
        return usage

    def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
        """Backward-compatible wrapper for existing call sites."""
        return self.get_usage(response)

    def _serialize_tools(self, tools: list[ToolDefinition]) -> list[ToolParam]:
        """Convert ToolDefinitions to Anthropic's tool format.

        Only the last N tools get cache_control to stay within Anthropic's limit of 4 cache blocks.
        """
        anthropic_tools: list[ToolParam] = []
        cache_count = max(self.max_cached_tool_definitions, 0)
        cache_start = max(len(tools) - cache_count, 0)

        for i, tool in enumerate(tools):
            # Remove title from schema if present (Anthropic doesn't like it in parameters)
            schema = tool.parameters.copy()
            if "title" in schema:
                del schema["title"]

            # Cache only the last N tools (Anthropic allows max 4 cache blocks)
            if i >= cache_start:
                anthropic_tools.append(
                    ToolParam(
                        name=tool.name,
                        description=tool.description,
                        input_schema=schema,
                        cache_control=CacheControlEphemeralParam(type="ephemeral"),
                    )
                )
            else:
                anthropic_tools.append(
                    ToolParam(
                        name=tool.name,
                        description=tool.description,
                        input_schema=schema,
                    )
                )
        return anthropic_tools

    def _get_tool_choice(
        self, tool_choice: ToolChoice | None, tools: list[ToolDefinition] | None
    ) -> (
        ToolChoiceAutoParam
        | ToolChoiceAnyParam
        | ToolChoiceToolParam
        | ToolChoiceNoneParam
        | None
    ):
        """Convert our tool_choice to Anthropic's format."""
        if tool_choice is None or tools is None:
            return None

        if tool_choice == "auto":
            return ToolChoiceAutoParam(type="auto")
        elif tool_choice == "required":
            # Anthropic uses "any" to mean "must use a tool"
            return ToolChoiceAnyParam(type="any")
        elif tool_choice == "none":
            return ToolChoiceNoneParam(type="none")
        else:
            # Specific tool name - force that tool
            return ToolChoiceToolParam(type="tool", name=tool_choice)

    def _extract_tool_calls(self, response: Message) -> list[ToolCall]:
        """Extract tool calls from Anthropic response."""
        import json

        tool_calls: list[ToolCall] = []

        for content_block in response.content:
            if isinstance(content_block, ToolUseBlock):
                # Convert input dict to JSON string for consistency with OpenAI
                arguments = (
                    json.dumps(content_block.input)
                    if isinstance(content_block.input, dict)
                    else str(content_block.input)
                )

                tool_calls.append(
                    ToolCall(
                        id=content_block.id,
                        function=Function(
                            name=content_block.name,
                            arguments=arguments,
                        ),
                        type="function",
                    )
                )

        return tool_calls

    def _extract_text_content(self, response: Message) -> str | None:
        """Extract text content from Anthropic response."""
        text_parts: list[str] = []

        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                text_parts.append(content_block.text)

        return "\n".join(text_parts) if text_parts else None

    def _extract_thinking(self, response: Message) -> tuple[str | None, str | None]:
        """Extract thinking and redacted thinking from Anthropic response.

        Returns:
            Tuple of (thinking_content, redacted_thinking_data)
        """
        thinking_parts: list[str] = []
        redacted_parts: list[str] = []

        for content_block in response.content:
            if isinstance(content_block, ThinkingBlock):
                thinking_parts.append(content_block.thinking)
            elif isinstance(content_block, RedactedThinkingBlock):
                redacted_parts.append(content_block.data)

        thinking = "\n".join(thinking_parts) if thinking_parts else None
        redacted = "\n".join(redacted_parts) if redacted_parts else None

        return thinking, redacted

    @staticmethod
    def _build_structured_content(
        response: Message,
    ) -> list[ContentPartTextParam | ContentPartThinkingParam | ContentPartRedactedThinkingParam] | None:
        """Build structured content blocks from Anthropic Message, preserving thinking signatures."""
        if not response.content:
            return None

        blocks: list[ContentPartTextParam | ContentPartThinkingParam | ContentPartRedactedThinkingParam] = []
        for content_block in response.content:
            if isinstance(content_block, TextBlock):
                blocks.append(ContentPartTextParam(text=content_block.text))
            elif isinstance(content_block, ThinkingBlock):
                blocks.append(
                    ContentPartThinkingParam(
                        thinking=content_block.thinking,
                        signature=content_block.signature,
                    )
                )
            elif isinstance(content_block, RedactedThinkingBlock):
                blocks.append(
                    ContentPartRedactedThinkingParam(data=content_block.data)
                )
            # ToolUseBlock excluded — already extracted to tool_calls
        return blocks if blocks else None

    @staticmethod
    def _strip_thinking_blocks(messages: list[dict]) -> list[dict]:
        """Remove thinking/redacted_thinking blocks from serialized assistant messages.

        Called when thinking is disabled to avoid sending stale thinking blocks
        from previous turns (which the API would reject).
        """
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            filtered = [
                b for b in content
                if not isinstance(b, dict) or b.get("type") not in ("thinking", "redacted_thinking")
            ]
            if not filtered:
                filtered = [{"type": "text", "text": ""}]
            msg["content"] = filtered
        return messages

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        """
        Invoke the model with the given messages and optional tools.

        Args:
            messages: List of chat messages
            tools: Optional list of tools the model can call
            tool_choice: Control how the model uses tools

        Returns:
            ChatInvokeCompletion with content and/or tool_calls
        """
        anthropic_messages, system_prompt = (
            AnthropicMessageSerializer.serialize_messages(messages)
        )

        # Strip or preserve historical thinking blocks based on model preset
        preset = get_thinking_preset(str(self.model))
        should_strip = (
            preset.history_policy == "strip"
            or (preset.history_policy == "auto" and not preset.supports_thinking)
        )
        if should_strip:
            self._strip_thinking_blocks(anthropic_messages)

        try:
            invoke_params = self._get_client_params_for_invoke()

            # Add tools if provided
            if tools:
                invoke_params["tools"] = self._serialize_tools(tools)

                anthropic_tool_choice = self._get_tool_choice(tool_choice, tools)
                if anthropic_tool_choice is not None:
                    invoke_params["tool_choice"] = anthropic_tool_choice

            response = await self.get_client().messages.create(
                model=resolve_api_model_name(str(self.model)),
                messages=anthropic_messages,
                system=system_prompt or omit,
                **invoke_params,
            )

            # Ensure we have a valid Message object before accessing attributes
            if not isinstance(response, Message):
                raise ModelProviderError(
                    message=f"Unexpected response type from Anthropic API: {type(response).__name__}. Response: {str(response)[:200]}",
                    status_code=502,
                    model=self.name,
                )

            usage = self._get_usage(response)

            # Log token usage if comate_agent_sdk_LLM_DEBUG is set
            if usage and os.getenv("comate_agent_sdk_LLM_DEBUG"):
                cached = usage.prompt_cached_tokens or 0
                input_tokens = usage.prompt_tokens - cached
                logger.info(
                    f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached + {usage.completion_tokens:,} out"
                )

            # Extract content, tool calls, and thinking
            content = self._extract_text_content(response)
            tool_calls = self._extract_tool_calls(response)
            thinking, redacted_thinking = self._extract_thinking(response)

            return ChatInvokeCompletion(
                content=content,
                tool_calls=tool_calls,
                thinking=thinking,
                redacted_thinking=redacted_thinking,
                raw_content_blocks=self._build_structured_content(response),
                usage=usage,
                stop_reason=response.stop_reason,
            )

        except APIConnectionError as e:
            raise ModelProviderError(message=e.message, model=self.name) from e
        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e
        except APIStatusError as e:
            raise ModelProviderError(
                message=e.message, status_code=e.status_code, model=self.name
            ) from e
        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
