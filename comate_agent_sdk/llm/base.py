"""
LLM abstraction layer with type-safe tool calling support.

This module provides a unified interface for chat models across different providers
(OpenAI, Anthropic, Google) with first-class support for tool calling.
"""

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from comate_agent_sdk.llm.messages import BaseMessage
from comate_agent_sdk.llm.views import ChatInvokeCompletion


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model.

    This is a provider-agnostic representation that gets serialized
    appropriately for each provider's API.
    """

    name: str
    """The name of the tool. Must be unique within the tools list."""

    description: str
    """A description of what the tool does. The model uses this to decide when to call it."""

    parameters: dict[str, Any]
    """JSON Schema describing the tool's parameters.

	Use SchemaOptimizer.create_optimized_json_schema(YourModel) to generate this
	from a Pydantic model.
	"""

    strict: bool = True
    """Whether to enforce strict schema validation (OpenAI specific).

	When True, the model will be forced to follow the schema exactly.
	"""


# Type alias for tool_choice parameter
ToolChoice = Literal["auto", "required", "none"] | str
"""
Tool choice options:
- 'auto': Model decides whether to call tools (default)
- 'required': Model must call at least one tool
- 'none': Model cannot call any tools
- str: Force a specific tool by name (e.g., 'get_weather')
"""


@runtime_checkable
class BaseChatModel(Protocol):
    """Protocol defining the interface for chat models.

    All LLM implementations (OpenAI, Anthropic, Google, etc.) must implement this protocol.
    """

    _verified_api_keys: bool = False

    model: str

    @property
    def provider(self) -> str:
        """The provider name (e.g., 'openai', 'anthropic', 'google')."""
        ...

    @property
    def name(self) -> str:
        """The model name/identifier."""
        ...

    @property
    def model_name(self) -> str:
        """Legacy alias for model. Use `name` instead."""
        return self.model

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion:
        """Invoke the model with the given messages and optional tools.

        Args:
                messages: List of chat messages forming the conversation.
                tools: Optional list of tools the model can call.
                tool_choice: Control how the model uses tools:
                        - None/'auto': Model decides (default)
                        - 'required': Must call at least one tool
                        - 'none': Cannot call tools
                        - str: Force specific tool by name
                **kwargs: Additional provider-specific parameters.

        Returns:
                ChatInvokeCompletion containing the model's response.
                Check `response.has_tool_calls` to see if the model wants to call tools.

        Example:
                ```python
                from comate_agent_sdk.llm import ChatOpenAI
                from comate_agent_sdk.llm.base import ToolDefinition
                from comate_agent_sdk.llm.messages import UserMessage, ToolMessage

                llm = ChatOpenAI(model='gpt-4o')

                # Define a tool
                weather_tool = ToolDefinition(
                    name='get_weather',
                    description='Get the current weather for a location',
                    parameters={'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'City name'}}, 'required': ['location']},
                )

                # Send message with tools
                response = await llm.ainvoke(messages=[UserMessage(content="What's the weather in Tokyo?")], tools=[weather_tool])

                # Handle tool calls
                if response.has_tool_calls:
                    for tool_call in response.tool_calls:
                        # Execute the tool...
                        result = execute_tool(tool_call)

                        # Send result back
                        tool_result = ToolMessage(tool_call_id=tool_call.id, tool_name=tool_call.function.name, content=result)
                        messages.append(tool_result)

                    # Continue conversation
                    response = await llm.ainvoke(messages=messages, tools=[weather_tool])
                ```
        """
        ...

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type,
        handler: Any,
    ) -> Any:
        """Allow this Protocol to be used in Pydantic models.

        This is useful for type-safe agent settings.
        Returns a schema that accepts any object (since this is a Protocol).
        """
        from pydantic_core import core_schema

        return core_schema.any_schema()
