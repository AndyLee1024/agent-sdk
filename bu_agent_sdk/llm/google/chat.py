import asyncio
import hashlib
import inspect
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from google import genai
from google.auth.credentials import Credentials
from google.genai import types
from google.genai.types import MediaModality

from bu_agent_sdk.llm.base import BaseChatModel, ToolChoice, ToolDefinition
from bu_agent_sdk.llm.exceptions import ModelProviderError
from bu_agent_sdk.llm.google.serializer import GoogleMessageSerializer
from bu_agent_sdk.llm.messages import BaseMessage, Function, ToolCall
from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage

VerifiedGeminiModels = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite-preview-02-05",
    "Gemini-2.0-exp",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemma-3-27b-it",
    "gemma-3-4b",
    "gemma-3-12b",
    "gemma-3n-e2b",
    "gemma-3n-e4b",
]


@dataclass
class ChatGoogle(BaseChatModel):
    """
    A wrapper around Google's Gemini chat model using the genai client.

    This class provides tool calling support for Google Gemini models.

    Example:
        ```python
        from bu_agent_sdk.llm import ChatGoogle
        from bu_agent_sdk.llm.base import ToolDefinition
        from bu_agent_sdk.llm.messages import UserMessage

        llm = ChatGoogle(model='gemini-2.0-flash', api_key='...')

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
    model: VerifiedGeminiModels | str
    temperature: float | None = 0.5
    top_p: float | None = None
    seed: int | None = None
    thinking_budget: int | None = (
        None  # for gemini-2.5 flash and flash-lite models, default will be set to 0
    )
    max_output_tokens: int | None = 8096
    config: types.GenerateContentConfigDict | None = None
    include_system_in_user: bool = False
    max_retries: int = 5  # Number of retries for retryable errors
    retryable_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )  # Status codes to retry on
    retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    retry_max_delay: float = 60.0  # Maximum delay in seconds between retries
    explicit_context_caching: bool = True
    explicit_cache_ttl_seconds: int | None = 3600

    # Client initialization parameters
    api_key: str | None = None
    vertexai: bool | None = None
    credentials: Credentials | None = None
    project: str | None = None
    location: str | None = None
    http_options: types.HttpOptions | types.HttpOptionsDict | None = None

    # Internal client cache to prevent connection issues
    _client: genai.Client | None = None
    _cached_content_name: str | None = None
    _cached_content_key: str | None = None

    # Static
    @property
    def provider(self) -> str:
        return "google"

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this chat instance"""
        return logging.getLogger(f"bu_agent_sdk.llm.google.{self.model}")

    def _get_client_params(self) -> dict[str, Any]:
        """Prepare client parameters dictionary."""
        # Define base client params
        base_params = {
            "api_key": self.api_key,
            "vertexai": self.vertexai,
            "credentials": self.credentials,
            "project": self.project,
            "location": self.location,
            "http_options": self.http_options,
        }

        # Create client_params dict with non-None values
        client_params = {k: v for k, v in base_params.items() if v is not None}

        return client_params

    def get_client(self) -> genai.Client:
        """
        Returns a genai.Client instance.

        Returns:
                genai.Client: An instance of the Google genai client.
        """
        if self._client is not None:
            return self._client

        client_params = self._get_client_params()
        self._client = genai.Client(**client_params)
        return self._client

    @property
    def name(self) -> str:
        return str(self.model)

    def _get_stop_reason(self, response: types.GenerateContentResponse) -> str | None:
        """Extract stop_reason from Google response."""
        if hasattr(response, "candidates") and response.candidates:
            return (
                str(response.candidates[0].finish_reason)
                if hasattr(response.candidates[0], "finish_reason")
                else None
            )
        return None

    def _get_usage(
        self, response: types.GenerateContentResponse
    ) -> ChatInvokeUsage | None:
        usage: ChatInvokeUsage | None = None

        if response.usage_metadata is not None:
            image_tokens = 0
            if response.usage_metadata.prompt_tokens_details is not None:
                image_tokens = sum(
                    detail.token_count or 0
                    for detail in response.usage_metadata.prompt_tokens_details
                    if detail.modality == MediaModality.IMAGE
                )

            usage = ChatInvokeUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=(response.usage_metadata.candidates_token_count or 0)
                + (response.usage_metadata.thoughts_token_count or 0),
                total_tokens=response.usage_metadata.total_token_count or 0,
                prompt_cached_tokens=response.usage_metadata.cached_content_token_count,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=image_tokens,
            )

        return usage

    def _serialize_tools(self, tools: list[ToolDefinition]) -> list[types.Tool]:
        """Convert ToolDefinitions to Google's tool format."""
        function_declarations = []
        for tool in tools:
            # Fix schema for Gemini compatibility
            fixed_schema = self._fix_gemini_schema(tool.parameters.copy())

            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=fixed_schema,
                )
            )

        return [types.Tool(function_declarations=function_declarations)]

    def _get_tool_choice(
        self, tool_choice: ToolChoice | None, tools: list[ToolDefinition] | None
    ) -> types.ToolConfigDict | None:
        """Convert our tool_choice to Google's format."""
        if tool_choice is None or tools is None:
            return None

        if tool_choice == "auto":
            return types.ToolConfigDict(
                function_calling_config=types.FunctionCallingConfigDict(mode="AUTO")
            )
        elif tool_choice == "required":
            return types.ToolConfigDict(
                function_calling_config=types.FunctionCallingConfigDict(mode="ANY")
            )
        elif tool_choice == "none":
            return types.ToolConfigDict(
                function_calling_config=types.FunctionCallingConfigDict(mode="NONE")
            )
        else:
            # Specific tool name - force that tool
            return types.ToolConfigDict(
                function_calling_config=types.FunctionCallingConfigDict(
                    mode="ANY", allowed_function_names=[tool_choice]
                )
            )

    def _extract_tool_calls(
        self, response: types.GenerateContentResponse
    ) -> list[ToolCall]:
        """Extract tool calls from Google response."""
        tool_calls: list[ToolCall] = []

        if not response.candidates:
            return tool_calls

        content = response.candidates[0].content
        if content is None or not content.parts:
            return tool_calls

        for part in content.parts:
            if hasattr(part, "function_call") and part.function_call is not None:
                fc = part.function_call
                # Convert args dict to JSON string for consistency with OpenAI
                arguments = json.dumps(fc.args) if fc.args else "{}"

                # Use Gemini's function call ID if provided, otherwise generate one
                if fc.id:
                    tool_call_id = fc.id
                else:
                    import uuid

                    tool_call_id = f"call_{uuid.uuid4().hex[:24]}"

                # Capture thought_signature from the Part (required by Gemini for history)
                thought_signature = getattr(part, "thought_signature", None)

                tool_calls.append(
                    ToolCall(
                        id=tool_call_id,
                        function=Function(
                            name=fc.name,
                            arguments=arguments,
                        ),
                        type="function",
                        thought_signature=thought_signature,
                    )
                )

        return tool_calls

    def _build_cache_key(
        self, system_instruction: str | None, tools: list[ToolDefinition] | None
    ) -> str:
        """Build a stable cache key for system/tool definitions."""
        tool_fingerprint = []
        if tools:
            for tool in tools:
                tool_fingerprint.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                )

        payload = {
            "model": str(self.model),
            "system_instruction": system_instruction,
            "tools": tool_fingerprint,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def _create_cached_content(
        self,
        system_instruction: str | None,
        tools: list[ToolDefinition] | None,
    ) -> str | None:
        """Create a cached content resource for stable system/tools."""
        if not self.explicit_context_caching:
            return None
        if system_instruction is None and not tools:
            return None
        if self.include_system_in_user:
            # Cannot cache when system is injected into user content.
            return None

        cache_key = self._build_cache_key(system_instruction, tools)
        if self._cached_content_name and self._cached_content_key == cache_key:
            return self._cached_content_name

        client = self.get_client()
        cache_root = getattr(client, "aio", None) or client
        cache_api = (
            getattr(cache_root, "caches", None)
            or getattr(cache_root, "cached_contents", None)
            or getattr(cache_root, "cache", None)
        )
        if cache_api is None or not hasattr(cache_api, "create"):
            self.logger.debug(
                "Gemini client does not expose explicit caching APIs; skipping."
            )
            return None

        create_kwargs: dict[str, Any] = {"model": self.model}
        if system_instruction:
            create_kwargs["system_instruction"] = system_instruction
        if tools:
            create_kwargs["tools"] = self._serialize_tools(tools)
        if self.explicit_cache_ttl_seconds is not None:
            create_kwargs["ttl"] = f"{self.explicit_cache_ttl_seconds}s"

        try:
            create_call = cache_api.create(**create_kwargs)
            cached = (
                await create_call if inspect.isawaitable(create_call) else create_call
            )
            cached_name = getattr(cached, "name", None) or getattr(cached, "id", None)
            if cached_name:
                self._cached_content_name = cached_name
                self._cached_content_key = cache_key
            return cached_name
        except Exception as e:
            self.logger.warning(f"Gemini explicit cache create failed: {e}")
            return None

    def _extract_text_content(
        self, response: types.GenerateContentResponse
    ) -> str | None:
        """Extract text content from Google response."""
        if not response.candidates:
            return None

        content = response.candidates[0].content
        if content is None or not content.parts:
            return None

        text_parts: list[str] = []
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        return "\n".join(text_parts) if text_parts else None

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
        # Serialize messages to Google format with the include_system_in_user flag
        contents, system_instruction = GoogleMessageSerializer.serialize_messages(
            messages, include_system_in_user=self.include_system_in_user
        )

        # Build config dictionary starting with user-provided config
        config: types.GenerateContentConfigDict = {}
        if self.config:
            config = self.config.copy()

        # Apply model-specific configuration (these can override config)
        if self.temperature is not None:
            config["temperature"] = self.temperature

        cached_content = await self._create_cached_content(system_instruction, tools)
        if cached_content:
            config["cached_content"] = cached_content
        elif system_instruction:
            config["system_instruction"] = system_instruction

        if self.top_p is not None:
            config["top_p"] = self.top_p

        if self.seed is not None:
            config["seed"] = self.seed

        # set default for flash, flash-lite, gemini-flash-lite-latest, and gemini-flash-latest models
        if self.thinking_budget is None and (
            "gemini-2.5-flash" in self.model or "gemini-flash" in self.model
        ):
            self.thinking_budget = 0

        if self.thinking_budget is not None:
            thinking_config_dict: types.ThinkingConfigDict = {
                "thinking_budget": self.thinking_budget
            }
            config["thinking_config"] = thinking_config_dict

        if self.max_output_tokens is not None:
            config["max_output_tokens"] = self.max_output_tokens

        # Add tools if provided
        if tools and not cached_content:
            config["tools"] = self._serialize_tools(tools)

        google_tool_choice = self._get_tool_choice(tool_choice, tools)
        if google_tool_choice is not None:
            config["tool_config"] = google_tool_choice

        async def _make_api_call():
            start_time = time.time()
            self.logger.debug(f"🚀 Starting API call to {self.model}")

            try:
                response = await self.get_client().aio.models.generate_content(
                    model=self.model,
                    contents=contents,  # type: ignore
                    config=config,
                )

                elapsed = time.time() - start_time
                self.logger.debug(f"✅ Got response in {elapsed:.2f}s")

                # Check for empty/blocked response and provide diagnostic info
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content is None:
                        finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                        self.logger.warning(
                            f"Gemini returned empty content. finish_reason={finish_reason}. "
                            f"This may indicate a safety block or model limitation."
                        )

                usage = self._get_usage(response)

                # Log token usage if bu_agent_sdk_LLM_DEBUG is set
                if usage and os.getenv("bu_agent_sdk_LLM_DEBUG"):
                    cached = usage.prompt_cached_tokens or 0
                    input_tokens = usage.prompt_tokens - cached
                    self.logger.info(
                        f"📊 {self.model}: {input_tokens:,} in + {cached:,} cached + {usage.completion_tokens:,} out"
                    )

                # Extract content and tool calls
                content = self._extract_text_content(response)
                tool_calls = self._extract_tool_calls(response)

                return ChatInvokeCompletion(
                    content=content,
                    tool_calls=tool_calls,
                    usage=usage,
                    stop_reason=self._get_stop_reason(response),
                )

            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"💥 API call failed after {elapsed:.2f}s: {type(e).__name__}: {e}"
                )
                # Re-raise the exception
                raise

        # Retry logic for certain errors with exponential backoff
        assert self.max_retries >= 1, "max_retries must be at least 1"

        for attempt in range(self.max_retries):
            try:
                return await _make_api_call()
            except ModelProviderError as e:
                # Retry if status code is in retryable list and we have attempts left
                if (
                    e.status_code in self.retryable_status_codes
                    and attempt < self.max_retries - 1
                ):
                    # Exponential backoff with jitter: base_delay * 2^attempt + random jitter
                    delay = min(
                        self.retry_base_delay * (2**attempt), self.retry_max_delay
                    )
                    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
                    total_delay = delay + jitter
                    self.logger.warning(
                        f"⚠️ Got {e.status_code} error, retrying in {total_delay:.1f}s... (attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(total_delay)
                    continue
                # Otherwise raise
                raise
            except Exception as e:
                # For non-ModelProviderError, wrap and raise
                error_message = str(e)
                status_code: int | None = None

                # Try to extract status code if available
                if hasattr(e, "response"):
                    response_obj = getattr(e, "response", None)
                    if response_obj and hasattr(response_obj, "status_code"):
                        status_code = getattr(response_obj, "status_code", None)

                # Enhanced timeout error handling
                if (
                    "timeout" in error_message.lower()
                    or "cancelled" in error_message.lower()
                ):
                    if isinstance(e, asyncio.CancelledError) or "CancelledError" in str(
                        type(e)
                    ):
                        error_message = "Gemini API request was cancelled (likely timeout). Consider: 1) Reducing input size, 2) Using a different model, 3) Checking network connectivity."
                        status_code = 504
                    else:
                        status_code = 408
                elif any(
                    indicator in error_message.lower()
                    for indicator in ["forbidden", "403"]
                ):
                    status_code = 403
                elif any(
                    indicator in error_message.lower()
                    for indicator in [
                        "rate limit",
                        "resource exhausted",
                        "quota exceeded",
                        "too many requests",
                        "429",
                    ]
                ):
                    status_code = 429
                elif any(
                    indicator in error_message.lower()
                    for indicator in [
                        "service unavailable",
                        "internal server error",
                        "bad gateway",
                        "503",
                        "502",
                        "500",
                    ]
                ):
                    status_code = 503

                raise ModelProviderError(
                    message=error_message,
                    status_code=status_code or 502,
                    model=self.name,
                ) from e

        raise RuntimeError("Retry loop completed without return or exception")

    def _fix_gemini_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a Pydantic model to a Gemini-compatible schema.

        This function removes unsupported properties like 'additionalProperties' and resolves
        $ref references that Gemini doesn't support.
        """

        # Handle $defs and $ref resolution
        if "$defs" in schema:
            defs = schema.pop("$defs")

            def resolve_refs(obj: Any) -> Any:
                if isinstance(obj, dict):
                    if "$ref" in obj:
                        ref = obj.pop("$ref")
                        ref_name = ref.split("/")[-1]
                        if ref_name in defs:
                            # Replace the reference with the actual definition
                            resolved = defs[ref_name].copy()
                            # Merge any additional properties from the reference
                            for key, value in obj.items():
                                if key != "$ref":
                                    resolved[key] = value
                            return resolve_refs(resolved)
                        return obj
                    else:
                        # Recursively process all dictionary values
                        return {k: resolve_refs(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [resolve_refs(item) for item in obj]
                return obj

            schema = resolve_refs(schema)

        # Remove unsupported properties
        def clean_schema(obj: Any, parent_key: str | None = None) -> Any:
            if isinstance(obj, dict):
                # Remove unsupported properties
                cleaned = {}
                for key, value in obj.items():
                    # Only strip 'title' when it's a JSON Schema metadata field (not inside 'properties')
                    # 'title' as a metadata field appears at schema level, not as a property name
                    is_metadata_title = key == "title" and parent_key != "properties"
                    if (
                        key not in ["additionalProperties", "default"]
                        and not is_metadata_title
                    ):
                        cleaned_value = clean_schema(value, parent_key=key)
                        # Handle empty object properties - Gemini doesn't allow empty OBJECT types
                        if (
                            key == "properties"
                            and isinstance(cleaned_value, dict)
                            and len(cleaned_value) == 0
                            and isinstance(obj.get("type", ""), str)
                            and obj.get("type", "").upper() == "OBJECT"
                        ):
                            # Convert empty object to have at least one property
                            cleaned["properties"] = {"_placeholder": {"type": "string"}}
                        else:
                            cleaned[key] = cleaned_value

                # If this is an object type with empty properties, add a placeholder
                if (
                    isinstance(cleaned.get("type", ""), str)
                    and cleaned.get("type", "").upper() == "OBJECT"
                    and "properties" in cleaned
                    and isinstance(cleaned["properties"], dict)
                    and len(cleaned["properties"]) == 0
                ):
                    cleaned["properties"] = {"_placeholder": {"type": "string"}}

                return cleaned
            elif isinstance(obj, list):
                return [clean_schema(item, parent_key=parent_key) for item in obj]
            return obj

        return clean_schema(schema)
