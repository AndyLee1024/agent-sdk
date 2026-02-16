"""DeepSeek message serializer with thinking blocks support.

Extends OpenAI serializer to handle DeepSeek's `reasoning_content` field.
"""

from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from comate_agent_sdk.llm.messages import AssistantMessage, BaseMessage
from comate_agent_sdk.llm.openai.serializer import OpenAIMessageSerializer


class DeepSeekMessageSerializer(OpenAIMessageSerializer):
    """Serializer for DeepSeek messages with reasoning_content support.

    DeepSeek uses OpenAI-compatible API but adds `reasoning_content` field
    for thinking mode. This serializer:
    1. Converts thinking blocks to `reasoning_content` when serializing
    2. Preserves reasoning_content in tool call loops (same turn)
    3. Provides utility to strip thinking blocks on new turns
    """

    @staticmethod
    def serialize(message: BaseMessage) -> ChatCompletionMessageParam:
        """Serialize a message to OpenAI format with DeepSeek extensions.

        For AssistantMessage with thinking blocks, this adds `reasoning_content`
        field to the serialized message.

        Args:
            message: Message to serialize

        Returns:
            OpenAI-compatible message param with optional reasoning_content
        """
        # Use parent serializer for base conversion
        serialized = OpenAIMessageSerializer.serialize(message)

        # Handle AssistantMessage with thinking blocks
        if isinstance(message, AssistantMessage) and isinstance(message.content, list):
            # Extract thinking content from content blocks
            reasoning_parts: list[str] = []

            for part in message.content:
                if hasattr(part, "type") and part.type == "thinking":
                    reasoning_parts.append(part.thinking)

            # Add reasoning_content field if thinking blocks present
            if reasoning_parts:
                reasoning_content = "\n".join(reasoning_parts)
                # Cast to dict to add custom field (type checkers won't like this)
                serialized_dict = dict(serialized)  # type: ignore
                serialized_dict["reasoning_content"] = reasoning_content
                return serialized_dict  # type: ignore

        return serialized

    @staticmethod
    def serialize_messages(
        messages: list[BaseMessage],
    ) -> list[ChatCompletionMessageParam]:
        """Serialize a list of messages with DeepSeek extensions.

        Args:
            messages: List of messages to serialize

        Returns:
            List of OpenAI-compatible message params with reasoning_content
        """
        # Validate tool call sequence (inherited from OpenAI)
        from comate_agent_sdk.llm.protocol_invariants import validate_tool_call_sequence

        validate_tool_call_sequence(messages, provider="deepseek")

        return [DeepSeekMessageSerializer.serialize(m) for m in messages]

    @staticmethod
    def strip_thinking_blocks(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Remove thinking blocks from assistant messages (for new turns).

        DeepSeek recommends cleaning historical reasoning_content to save bandwidth.
        This creates a new message list with thinking blocks removed from
        AssistantMessage content.

        Args:
            messages: Original message list

        Returns:
            New message list with thinking blocks removed
        """
        cleaned_messages: list[BaseMessage] = []

        for msg in messages:
            if not isinstance(msg, AssistantMessage):
                cleaned_messages.append(msg)
                continue

            # Check if content has thinking blocks
            if not isinstance(msg.content, list):
                cleaned_messages.append(msg)
                continue

            # Filter out thinking blocks
            filtered_content = [
                part for part in msg.content if not (hasattr(part, "type") and part.type == "thinking")
            ]

            # If all content was thinking blocks, keep at least empty text
            if not filtered_content:
                from comate_agent_sdk.llm.messages import ContentPartTextParam

                filtered_content = [ContentPartTextParam(text="")]

            # Create new message with filtered content
            cleaned_msg = AssistantMessage(
                content=filtered_content,
                tool_calls=msg.tool_calls,
                name=msg.name,
                refusal=msg.refusal,
            )
            cleaned_messages.append(cleaned_msg)

        return cleaned_messages
