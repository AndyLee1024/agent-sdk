from typing import overload

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import Function

from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartDocumentParam,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    DeveloperMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.llm.protocol_invariants import validate_tool_call_sequence


class OpenAIMessageSerializer:
    """Serializer for converting between custom message types and OpenAI message param types."""

    @staticmethod
    def _serialize_content_part_text(
        part: ContentPartTextParam,
    ) -> ChatCompletionContentPartTextParam:
        return ChatCompletionContentPartTextParam(text=part.text, type="text")

    @staticmethod
    def _serialize_content_part_image(
        part: ContentPartImageParam,
    ) -> ChatCompletionContentPartImageParam:
        return ChatCompletionContentPartImageParam(
            image_url=ImageURL(url=part.image_url.url, detail=part.image_url.detail),
            type="image_url",
        )

    @staticmethod
    def _serialize_content_part_refusal(
        part: ContentPartRefusalParam,
    ) -> ChatCompletionContentPartRefusalParam:
        return ChatCompletionContentPartRefusalParam(
            refusal=part.refusal, type="refusal"
        )

    @staticmethod
    def _serialize_user_content(
        content: str
        | list[ContentPartTextParam | ContentPartImageParam | ContentPartDocumentParam],
    ) -> (
        str
        | list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam]
    ):
        """Serialize content for user messages (text and images allowed).

        No native PDF/document support, so docs converted to a text placeholder.
        """
        if isinstance(content, str):
            return content

        serialized_parts: list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam
        ] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(
                    OpenAIMessageSerializer._serialize_content_part_text(part)
                )
            elif part.type == "image_url":
                serialized_parts.append(
                    OpenAIMessageSerializer._serialize_content_part_image(part)
                )
            elif part.type == "document":
                serialized_parts.append(
                    ChatCompletionContentPartTextParam(
                        text="[PDF document attached]", type="text"
                    )
                )
        return serialized_parts

    @staticmethod
    def _serialize_system_content(
        content: str | list[ContentPartTextParam],
    ) -> str | list[ChatCompletionContentPartTextParam]:
        """Serialize content for system messages (text only)."""
        if isinstance(content, str):
            return content

        serialized_parts: list[ChatCompletionContentPartTextParam] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(
                    OpenAIMessageSerializer._serialize_content_part_text(part)
                )
        return serialized_parts

    @staticmethod
    def _serialize_assistant_content(
        content: str | list[ContentPartTextParam | ContentPartRefusalParam] | None,
    ) -> (
        str
        | list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam
        ]
        | None
    ):
        """Serialize content for assistant messages (text and refusal allowed)."""
        if content is None:
            return None
        if isinstance(content, str):
            return content

        serialized_parts: list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartRefusalParam
        ] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(
                    OpenAIMessageSerializer._serialize_content_part_text(part)
                )
            elif part.type == "refusal":
                serialized_parts.append(
                    OpenAIMessageSerializer._serialize_content_part_refusal(part)
                )
        return serialized_parts

    @staticmethod
    def _serialize_tool_call(
        tool_call: ToolCall,
    ) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=tool_call.id,
            function=Function(
                name=tool_call.function.name, arguments=tool_call.function.arguments
            ),
            type="function",
        )

    @staticmethod
    def _serialize_tool_message_content(
        content: str | list[ContentPartTextParam | ContentPartImageParam],
    ) -> str | list[ChatCompletionContentPartTextParam]:
        """Serialize content for tool messages (text only, images converted to text description)."""
        if isinstance(content, str):
            return content

        # OpenAI tool messages only support text content
        serialized_parts: list[ChatCompletionContentPartTextParam] = []
        for part in content:
            if part.type == "text":
                serialized_parts.append(
                    OpenAIMessageSerializer._serialize_content_part_text(part)
                )
            elif part.type == "image_url":
                # Images in tool results need special handling - convert to description
                serialized_parts.append(
                    ChatCompletionContentPartTextParam(
                        text="[Image attached]", type="text"
                    )
                )
        return serialized_parts if serialized_parts else ""

    # endregion

    # region - Serialize overloads
    @overload
    @staticmethod
    def serialize(message: UserMessage) -> ChatCompletionUserMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> ChatCompletionSystemMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> ChatCompletionAssistantMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: ToolMessage) -> ChatCompletionToolMessageParam: ...

    @overload
    @staticmethod
    def serialize(message: DeveloperMessage) -> ChatCompletionDeveloperMessageParam: ...

    @staticmethod
    def serialize(message: BaseMessage) -> ChatCompletionMessageParam:
        """Serialize a custom message to an OpenAI message param."""

        if isinstance(message, UserMessage):
            user_result: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": OpenAIMessageSerializer._serialize_user_content(
                    message.content
                ),
            }
            if message.name is not None:
                user_result["name"] = message.name
            return user_result

        elif isinstance(message, SystemMessage):
            system_result: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": OpenAIMessageSerializer._serialize_system_content(
                    message.content
                ),
            }
            if message.name is not None:
                system_result["name"] = message.name
            return system_result

        elif isinstance(message, DeveloperMessage):
            # Developer messages are for o1+ models, replacing system messages
            developer_result: ChatCompletionDeveloperMessageParam = {
                "role": "developer",
                "content": OpenAIMessageSerializer._serialize_system_content(
                    message.content
                ),
            }
            if message.name is not None:
                developer_result["name"] = message.name
            return developer_result

        elif isinstance(message, AssistantMessage):
            # Handle content serialization
            content = None
            if message.content is not None:
                content = OpenAIMessageSerializer._serialize_assistant_content(
                    message.content
                )

            assistant_result: ChatCompletionAssistantMessageParam = {
                "role": "assistant"
            }

            # Only add content if it's not None
            if content is not None:
                assistant_result["content"] = content

            if message.name is not None:
                assistant_result["name"] = message.name
            if message.refusal is not None:
                assistant_result["refusal"] = message.refusal
            if message.tool_calls:
                assistant_result["tool_calls"] = [
                    OpenAIMessageSerializer._serialize_tool_call(tc)
                    for tc in message.tool_calls
                ]

            return assistant_result

        elif isinstance(message, ToolMessage):
            # Tool messages contain the result of a tool call
            # Use placeholder if message was destroyed to save context
            if message.destroyed:
                # 如果有 offload_path，显示完整路径；否则使用简单占位符
                if message.offloaded and message.offload_path:
                    content = (
                        f"[Content offloaded]\n"
                        f"Path: {message.offload_path}\n"
                        f"Use Read tool to view details."
                    )
                else:
                    content = "<removed to save context>"
            else:
                content = OpenAIMessageSerializer._serialize_tool_message_content(
                    message.content
                )
                # If content is a list, join it into a string (OpenAI expects string for tool messages)
                if isinstance(content, list):
                    content = "\n".join(part["text"] for part in content)

            tool_result: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": content,
            }
            return tool_result

        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def serialize_messages(
        messages: list[BaseMessage],
    ) -> list[ChatCompletionMessageParam]:
        validate_tool_call_sequence(messages, provider="openai")
        return [OpenAIMessageSerializer.serialize(m) for m in messages]
