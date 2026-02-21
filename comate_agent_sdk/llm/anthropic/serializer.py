import json
from typing import overload

from anthropic.types import (
    Base64ImageSourceParam,
    Base64PDFSourceParam,
    CacheControlEphemeralParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    RedactedThinkingBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSourceParam,
)

from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartDocumentParam,
    ContentPartImageParam,
    ContentPartTextParam,
    DeveloperMessage,
    SupportedImageMediaType,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.llm.protocol_invariants import validate_tool_call_sequence

NonSystemMessage = UserMessage | AssistantMessage | ToolMessage


class AnthropicMessageSerializer:
    """Serializer for converting between custom message types and Anthropic message param types."""

    @staticmethod
    def _is_base64_image(url: str) -> bool:
        """Check if the URL is a base64 encoded image."""
        return url.startswith("data:image/")

    @staticmethod
    def _parse_base64_url(url: str) -> tuple[SupportedImageMediaType, str]:
        """Parse a base64 data URL to extract media type and data."""
        # Format: data:image/jpeg;base64,<data>
        if not url.startswith("data:"):
            raise ValueError(f"Invalid base64 URL: {url}")

        header, data = url.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "")

        # Ensure it's a supported media type
        supported_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if media_type not in supported_types:
            # Default to jpeg if not recognized
            media_type = "image/jpeg"

        return media_type, data  # type: ignore

    @staticmethod
    def _serialize_cache_control(use_cache: bool) -> CacheControlEphemeralParam | None:
        """Serialize cache control."""
        if use_cache:
            return CacheControlEphemeralParam(type="ephemeral")
        return None

    @staticmethod
    def _serialize_content_part_text(
        part: ContentPartTextParam, use_cache: bool
    ) -> TextBlockParam:
        """Convert a text content part to Anthropic's TextBlockParam."""
        block = TextBlockParam(text=part.text, type="text")
        if use_cache:
            block["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
        return block

    @staticmethod
    def _serialize_content_part_image(part: ContentPartImageParam) -> ImageBlockParam:
        """Convert an image content part to Anthropic's ImageBlockParam."""
        url = part.image_url.url

        if AnthropicMessageSerializer._is_base64_image(url):
            # Handle base64 encoded images
            media_type, data = AnthropicMessageSerializer._parse_base64_url(url)
            return ImageBlockParam(
                source=Base64ImageSourceParam(
                    data=data,
                    media_type=media_type,
                    type="base64",
                ),
                type="image",
            )
        else:
            # Handle URL images
            return ImageBlockParam(
                source=URLImageSourceParam(url=url, type="url"), type="image"
            )

    @staticmethod
    def _serialize_content_part_document(
        part: ContentPartDocumentParam,
    ) -> DocumentBlockParam:
        """Convert a document content part to Anthropic's DocumentBlockParam."""
        return DocumentBlockParam(
            source=Base64PDFSourceParam(
                data=part.source.data,
                media_type="application/pdf",
                type="base64",
            ),
            type="document",
        )

    @staticmethod
    def _is_session_state_user_message(message: UserMessage) -> bool:
        """识别 ContextIR lowering 生成的 session_state 消息。

        约定：session_state 通过 UserMessage(is_meta=True) 注入，内容包含以下标签之一。
        """
        if not bool(getattr(message, "is_meta", False)):
            return False

        text = message.text or ""
        markers = ("<system_env>", "<git_env>", "<mcp_tools>", "<output_style>")
        return any(marker in text for marker in markers)

    @staticmethod
    def _serialize_content_to_str(
        content: str | list[ContentPartTextParam], use_cache: bool = False
    ) -> list[TextBlockParam] | str:
        """Serialize content to a string."""
        cache_control = AnthropicMessageSerializer._serialize_cache_control(use_cache)

        if isinstance(content, str):
            if cache_control:
                return [
                    TextBlockParam(
                        text=content, type="text", cache_control=cache_control
                    )
                ]
            else:
                return content

        serialized_blocks: list[TextBlockParam] = []
        for i, part in enumerate(content):
            is_last = i == len(content) - 1
            if part.type == "text":
                serialized_blocks.append(
                    AnthropicMessageSerializer._serialize_content_part_text(
                        part, use_cache=use_cache and is_last
                    )
                )

        return serialized_blocks

    @staticmethod
    def _serialize_content(
        content: str
        | list[ContentPartTextParam | ContentPartImageParam | ContentPartDocumentParam],
        use_cache: bool = False,
    ) -> str | list[TextBlockParam | ImageBlockParam | DocumentBlockParam]:
        """Serialize content to Anthropic format."""
        if isinstance(content, str):
            if use_cache:
                return [
                    TextBlockParam(
                        text=content,
                        type="text",
                        cache_control=CacheControlEphemeralParam(type="ephemeral"),
                    )
                ]
            else:
                return content

        serialized_blocks: list[
            TextBlockParam | ImageBlockParam | DocumentBlockParam
        ] = []
        for i, part in enumerate(content):
            is_last = i == len(content) - 1
            if part.type == "text":
                serialized_blocks.append(
                    AnthropicMessageSerializer._serialize_content_part_text(
                        part, use_cache=use_cache and is_last
                    )
                )
            elif part.type == "image_url":
                serialized_blocks.append(
                    AnthropicMessageSerializer._serialize_content_part_image(part)
                )
            elif part.type == "document":
                serialized_blocks.append(
                    AnthropicMessageSerializer._serialize_content_part_document(part)
                )

        return serialized_blocks

    @staticmethod
    def _serialize_tool_calls_to_content(
        tool_calls, use_cache: bool = False
    ) -> list[ToolUseBlockParam]:
        """Convert tool calls to Anthropic's ToolUseBlockParam format."""
        blocks: list[ToolUseBlockParam] = []
        for i, tool_call in enumerate(tool_calls):
            # Parse the arguments JSON string to object

            try:
                input_obj = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                # If arguments aren't valid JSON, use as string
                input_obj = {"arguments": tool_call.function.arguments}

            is_last = i == len(tool_calls) - 1
            block = ToolUseBlockParam(
                id=tool_call.id,
                input=input_obj,
                name=tool_call.function.name,
                type="tool_use",
            )
            if use_cache and is_last:
                block["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
            blocks.append(block)
        return blocks

    @staticmethod
    def _serialize_tool_result_content(
        content: str | list,
    ) -> str | list[TextBlockParam | ImageBlockParam]:
        """Serialize tool result content for Anthropic."""
        if isinstance(content, str):
            return content

        blocks: list[TextBlockParam | ImageBlockParam] = []
        for part in content:
            if part.type == "text":
                blocks.append(TextBlockParam(text=part.text, type="text"))
            elif part.type == "image_url":
                blocks.append(
                    AnthropicMessageSerializer._serialize_content_part_image(part)
                )
        return blocks if blocks else ""

    @staticmethod
    def _serialize_tool_message(
        message: ToolMessage, use_cache: bool = False
    ) -> ToolResultBlockParam:
        """Convert a ToolMessage to Anthropic's ToolResultBlockParam."""
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
            content = AnthropicMessageSerializer._serialize_tool_result_content(
                message.content
            )

        block = ToolResultBlockParam(
            tool_use_id=message.tool_call_id,
            type="tool_result",
            content=content,
            is_error=message.is_error,
        )
        if use_cache:
            block["cache_control"] = CacheControlEphemeralParam(type="ephemeral")
        return block

    # region - Serialize overloads
    @overload
    @staticmethod
    def serialize(message: UserMessage) -> MessageParam: ...

    @overload
    @staticmethod
    def serialize(message: SystemMessage) -> SystemMessage: ...

    @overload
    @staticmethod
    def serialize(message: DeveloperMessage) -> SystemMessage: ...

    @overload
    @staticmethod
    def serialize(message: AssistantMessage) -> MessageParam: ...

    @overload
    @staticmethod
    def serialize(message: ToolMessage) -> MessageParam: ...

    @staticmethod
    def serialize(
        message: BaseMessage,
    ) -> MessageParam | SystemMessage | DeveloperMessage:
        """Serialize a custom message to an Anthropic MessageParam.

        Note: Anthropic doesn't have a 'system' or 'developer' role. These messages
        should be handled separately as the system parameter in the API call.
        """
        if isinstance(message, UserMessage):
            use_cache = bool(message.cache) or AnthropicMessageSerializer._is_session_state_user_message(message)
            content = AnthropicMessageSerializer._serialize_content(
                message.content, use_cache=use_cache
            )
            return MessageParam(role="user", content=content)

        elif isinstance(message, SystemMessage):
            # Anthropic doesn't have system messages in the messages array
            # System prompts are passed separately.
            return message

        elif isinstance(message, DeveloperMessage):
            # Developer messages are treated like system messages for Anthropic
            # They get extracted and passed as the system parameter
            return message

        elif isinstance(message, ToolMessage):
            # Tool results in Anthropic go in a user message with ToolResultBlockParam
            tool_result = AnthropicMessageSerializer._serialize_tool_message(
                message, use_cache=message.cache
            )
            return MessageParam(role="user", content=[tool_result])

        elif isinstance(message, AssistantMessage):
            # Handle content and tool calls
            blocks: list[
                TextBlockParam
                | ToolUseBlockParam
                | ThinkingBlockParam
                | RedactedThinkingBlockParam
            ] = []

            # Add content blocks if present
            if message.content is not None:
                if isinstance(message.content, str):
                    # String content: only cache if it's the only/last block (no tool calls)
                    _block = TextBlockParam(text=message.content, type="text")
                    if message.cache and not message.tool_calls:
                        _block["cache_control"] = CacheControlEphemeralParam(
                            type="ephemeral"
                        )
                    blocks.append(_block)
                else:
                    # Process content parts (text, refusal, thinking, redacted_thinking)
                    for i, part in enumerate(message.content):
                        # Only last content block gets cache if there are no tool calls
                        is_last_content = (
                            i == len(message.content) - 1
                        ) and not message.tool_calls
                        if part.type == "text":
                            blocks.append(
                                AnthropicMessageSerializer._serialize_content_part_text(
                                    part, use_cache=message.cache and is_last_content
                                )
                            )
                        elif part.type == "thinking":
                            # Thinking blocks - must include signature for Anthropic
                            blocks.append(
                                ThinkingBlockParam(
                                    type="thinking",
                                    thinking=part.thinking,
                                    signature=part.signature or "",
                                )
                            )
                        elif part.type == "redacted_thinking":
                            # Redacted thinking blocks - preserve the encrypted data
                            blocks.append(
                                RedactedThinkingBlockParam(
                                    type="redacted_thinking",
                                    data=part.data,
                                )
                            )
                            # # Note: Anthropic doesn't have a specific refusal block type,
                            # # so we convert refusals to text blocks
                            # elif part.type == 'refusal':
                            # 	blocks.append(TextBlockParam(text=f'[Refusal] {part.refusal}', type='text'))

            # Add tool use blocks if present
            if message.tool_calls:
                tool_blocks = (
                    AnthropicMessageSerializer._serialize_tool_calls_to_content(
                        message.tool_calls, use_cache=message.cache
                    )
                )
                blocks.extend(tool_blocks)

            # If no content or tool calls, add empty text block
            # (Anthropic requires at least one content block)
            if not blocks:
                _fallback = TextBlockParam(text="", type="text")
                if message.cache:
                    _fallback["cache_control"] = CacheControlEphemeralParam(
                        type="ephemeral"
                    )
                blocks.append(_fallback)

            # If caching is enabled or we have multiple blocks, return blocks as-is
            # Otherwise, simplify single text blocks to plain string
            if message.cache or len(blocks) > 1:
                content = blocks
            else:
                # Only simplify when no caching and single block
                single_block = blocks[0]
                if single_block["type"] == "text" and not single_block.get(
                    "cache_control"
                ):
                    content = single_block["text"]
                else:
                    content = blocks

            return MessageParam(
                role="assistant",
                content=content,
            )

        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def _clean_cache_messages(
        messages: list[NonSystemMessage],
    ) -> list[NonSystemMessage]:
        """保留最后 2 个 cache=True 的消息，其余全部置 False。

        Anthropic 允许最多 4 个 cache breakpoint。
        当前策略中 system 不再使用 cache_control，因此 messages 层默认保留 2 个
        显式 cache=True 断点（session_state 断点独立按规则注入）。

        Note: This method mutates messages in place. Caller must pass pre-copied data
        (serialize_messages handles this via model_copy).

        Args:
                messages: List of non-system messages to clean (will be mutated)

        Returns:
                The same list with cleaned cache settings
        """
        if not messages:
            return messages

        keep_indices: set[int] = set()
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].cache:
                keep_indices.add(i)
                if len(keep_indices) == 2:
                    break

        for i, msg in enumerate(messages):
            if msg.cache and i not in keep_indices:
                msg.cache = False

        return messages

    @staticmethod
    def serialize_messages(
        messages: list[BaseMessage],
    ) -> tuple[list[MessageParam], list[TextBlockParam] | str | None]:
        """Serialize a list of messages, extracting any system/developer message.

        Returns:
            A tuple of (messages, system_message) where system_message is extracted
            from any SystemMessage or DeveloperMessage in the list.
        """
        validate_tool_call_sequence(messages, provider="anthropic")
        messages = [m.model_copy(deep=True) for m in messages]

        # Separate system/developer messages from normal messages
        normal_messages: list[NonSystemMessage] = []
        system_message: SystemMessage | DeveloperMessage | None = None

        for message in messages:
            if isinstance(message, (SystemMessage, DeveloperMessage)):
                # Last system/developer message wins
                system_message = message
            else:
                normal_messages.append(message)

        # 自动在对话末尾注入 cache breakpoint（文档关键建议）
        if normal_messages:
            normal_messages[-1].cache = True

        # Clean cache messages so only the last 2 cache=True messages remain cached
        normal_messages = AnthropicMessageSerializer._clean_cache_messages(
            normal_messages
        )

        # Serialize normal messages
        serialized_messages: list[MessageParam] = []
        for message in normal_messages:
            serialized_messages.append(AnthropicMessageSerializer.serialize(message))

        # Serialize system message
        serialized_system_message: list[TextBlockParam] | str | None = None
        if system_message:
            # system prompt 不使用 cache_control（由 session_state 承担 ephemeral cache 断点）
            serialized_system_message = (
                AnthropicMessageSerializer._serialize_content_to_str(
                    system_message.content, use_cache=False
                )
            )

        return serialized_messages, serialized_system_message
