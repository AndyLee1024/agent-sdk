import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

logger = logging.getLogger("comate_agent_sdk.llm.openai")
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared.chat_model import ChatModel
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.shared_params.reasoning_effort import ReasoningEffort

from comate_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from comate_agent_sdk.llm.exceptions import ModelProviderError, ModelRateLimitError
from comate_agent_sdk.llm.messages import BaseMessage, Function, ToolCall
from comate_agent_sdk.llm.openai.serializer import OpenAIMessageSerializer
from comate_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage



@dataclass
class ChatOpenAI(BaseChatModel):
    """
    A wrapper around AsyncOpenAI that implements the BaseChatModel protocol.

    This class provides tool calling support for OpenAI models.

    Example:
        ```python
        from comate_agent_sdk.llm import ChatOpenAI
        from comate_agent_sdk.llm.base import ToolDefinition
        from comate_agent_sdk.llm.messages import UserMessage

        llm = ChatOpenAI(model='gpt-4o', api_key='...')

        # Define tools
        tools = [ToolDefinition(name='get_weather', description='Get weather for a location', parameters={'type': 'object', 'properties': {...}})]

        # Invoke with tools
        response = await llm.ainvoke(messages=[UserMessage(content="What's the weather?")], tools=tools)

        if response.has_tool_calls:
            for tc in response.tool_calls:
                logger.info(f'Call {tc.function.name} with {tc.function.arguments}')
        ```
    """

    # Model configuration
    model: ChatModel | str

    # Model params
    temperature: float | None = 0.2
    frequency_penalty: float | None = (
        0.3  # this avoids infinite generation of \t for models like 4.1-mini
    )
    reasoning_effort: ReasoningEffort = "low"
    seed: int | None = None
    service_tier: Literal["auto", "default", "flex", "priority", "scale"] | None = None
    top_p: float | None = None
    parallel_tool_calls: bool = True  # Allow multiple tool calls in a single response
    prompt_cache_key: str | None = "comate_agent_sdk-agent"
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    extended_cache_models: tuple[str, ...] = field(
        default_factory=lambda: (
            "gpt-5.2",
            "gpt-5.1-codex-max",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-codex-mini",
            "gpt-5.1-chat-latest",
            "gpt-5",
            "gpt-5-codex",
            "gpt-4.1",
        )
    )

    # Client initialization parameters
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | httpx.URL | None = None
    websocket_base_url: str | httpx.URL | None = None
    timeout: float | httpx.Timeout | None = None
    max_retries: int = 5  # Increase default retries for automation reliability
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None
    http_client: httpx.AsyncClient | None = None
    _strict_response_validation: bool = False
    max_completion_tokens: int | None = 16384
    reasoning_models: list[ChatModel | str] | None = field(
        default_factory=lambda: [
            "o4-mini",
            "o3",
            "o3-mini",
            "o1",
            "o1-pro",
            "o3-pro",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ]
    )

    # Internal client cache
    _client: AsyncOpenAI | None = field(default=None, repr=False, compare=False)

    # Static
    @property
    def provider(self) -> str:
        return "openai"

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "project": self.project,
            "base_url": self.base_url,
            "websocket_base_url": self.websocket_base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
            "_strict_response_validation": self._strict_response_validation,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        # Add http_client if provided
        if self.http_client is not None:
            client_params["http_client"] = self.http_client

        return client_params

    def _is_langfuse_configured(self) -> bool:
        """Check if all required Langfuse environment variables are set."""
        required_vars = [
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_BASE_URL",
        ]
        return all(os.getenv(var) for var in required_vars)

    def get_client(self) -> AsyncOpenAI:
        """
        Returns an AsyncOpenAI client (cached).

        If Langfuse environment variables (LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY,
        LANGFUSE_BASE_URL) are all set, returns a Langfuse-wrapped client for
        observability. Otherwise, returns the native OpenAI client.

        Returns:
                AsyncOpenAI: An instance of the AsyncOpenAI client.
        """

        if self._client is not None:
            return self._client
        client_params = self._get_client_params()

        if self._is_langfuse_configured():
            # Lazy import to avoid Langfuse initialization when not configured
            from langfuse.openai import openai as oai

            logger.debug("Langfuse environment variables detected, using Langfuse-wrapped OpenAI client")
            self._client = oai.AsyncOpenAI(**client_params)
        else:
            logger.debug("Langfuse not configured, using native OpenAI client")
            self._client = AsyncOpenAI(**client_params)

        return self._client

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
        if response.usage is not None:
            usage = ChatInvokeUsage(
                prompt_tokens=response.usage.prompt_tokens,
                prompt_cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                if response.usage.prompt_tokens_details is not None
                else None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                # Completion
                # OpenAI 的 completion_tokens 已包含推理相关 token，避免重复累计。
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            usage = None

        return usage

    def _serialize_tools(
        self, tools: list[ToolDefinition]
    ) -> list[ChatCompletionToolParam]:
        """Convert ToolDefinitions to OpenAI's tool format."""
        result = []
        for tool in tools:
            params = tool.parameters
            # For strict mode, OpenAI requires ALL properties in 'required'
            # Transform optional params to required + nullable
            if tool.strict and params.get("properties"):
                params = self._make_strict_schema(params)

            result.append(
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=params,
                        strict=tool.strict,
                    ),
                )
            )
        return result

    def _make_strict_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Transform schema for OpenAI strict mode: all properties must be required."""
        schema = schema.copy()
        props = schema.get("properties", {})
        required = set(schema.get("required", []))

        new_props = {}
        for name, prop in props.items():
            prop = self._make_strict_property(prop, name in required)
            new_props[name] = prop

        schema["properties"] = new_props
        schema["required"] = list(props.keys())  # All properties required
        schema["additionalProperties"] = False
        return schema

    def _make_strict_property(
        self, prop: dict[str, Any], is_required: bool
    ) -> dict[str, Any]:
        """Transform a single property for strict mode, recursively handling nested objects."""
        prop = prop.copy()

        # Handle nested objects
        if prop.get("type") == "object" and "properties" in prop:
            prop = self._make_strict_schema(prop)

        # Handle arrays with object items
        if prop.get("type") == "array" and "items" in prop:
            items = prop["items"]
            if (
                isinstance(items, dict)
                and items.get("type") == "object"
                and "properties" in items
            ):
                prop["items"] = self._make_strict_schema(items)

        # Make optional params nullable
        if not is_required:
            if "type" in prop:
                prop["type"] = [prop["type"], "null"]
            elif "anyOf" not in prop:
                prop["anyOf"] = [prop, {"type": "null"}]

        return prop

    def _resolve_prompt_cache_retention(self) -> str | None:
        """Select prompt cache retention based on model support."""
        if self.prompt_cache_retention is not None:
            return self.prompt_cache_retention

        model_name = str(self.model).lower()
        if any(key in model_name for key in self.extended_cache_models):
            return "24h"
        return None

    def _get_tool_choice(
        self, tool_choice: ToolChoice | None, tools: list[ToolDefinition] | None
    ) -> Any:
        """Convert our tool_choice to OpenAI's format."""
        if tool_choice is None or tools is None:
            return None

        if tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"
        elif tool_choice == "none":
            return "none"
        else:
            # Specific tool name - force that tool
            return {"type": "function", "function": {"name": tool_choice}}

    def _extract_tool_calls(self, response: ChatCompletion) -> list[ToolCall]:
        """Extract tool calls from OpenAI response."""
        tool_calls: list[ToolCall] = []
        message = response.choices[0].message

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        function=Function(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                        type="function",
                    )
                )

        return tool_calls

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
        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

        try:
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

            # Handle reasoning models (o1, o3, etc.)
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

            # Extract usage
            usage = self._get_usage(response)

            # Log token usage if comate_agent_sdk_LLM_DEBUG is set
            if usage and os.getenv("comate_agent_sdk_LLM_DEBUG"):
                cached = usage.prompt_cached_tokens or 0
                input_tokens = usage.prompt_tokens - cached
                logger.info(
                    f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached + {usage.completion_tokens:,} out"
                )

            # Extract content
            content = response.choices[0].message.content

            # Extract tool calls
            tool_calls = self._extract_tool_calls(response)

            return ChatInvokeCompletion(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                stop_reason=response.choices[0].finish_reason
                if response.choices
                else None,
            )

        except RateLimitError as e:
            raise ModelRateLimitError(message=e.message, model=self.name) from e

        except APIConnectionError as e:
            raise ModelProviderError(message=str(e), model=self.name) from e

        except APIStatusError as e:
            raise ModelProviderError(
                message=e.message, status_code=e.status_code, model=self.name
            ) from e

        except Exception as e:
            raise ModelProviderError(message=str(e), model=self.name) from e
