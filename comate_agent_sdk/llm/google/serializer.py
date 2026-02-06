import base64
import json

from google.genai.types import Content, ContentListUnion, FunctionCall, Part

from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    BaseMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


class GoogleMessageSerializer:
    """Serializer for converting messages to Google Gemini format."""

    @staticmethod
    def _serialize_tool_message(message: ToolMessage) -> Part:
        """Convert a ToolMessage to Google's FunctionResponse Part."""
        # Use placeholder if message was destroyed to save context
        if message.destroyed:
            # 如果有 offload_path，显示完整路径；否则使用简单占位符
            if message.offloaded and message.offload_path:
                response_data = {
                    "result": (
                        f"[Content offloaded]\n"
                        f"Path: {message.offload_path}\n"
                        f"Use Read tool to view details."
                    )
                }
            else:
                response_data = {"result": "<removed to save context>"}
        elif message.is_error:
            response_data = {"error": message.text}
        else:
            # Try to parse content as JSON, otherwise use as string
            try:
                if isinstance(message.content, str):
                    response_data = json.loads(message.content)
                else:
                    response_data = {"result": message.text}
            except json.JSONDecodeError:
                response_data = {"result": message.text}

        return Part.from_function_response(
            name=message.tool_name,
            response=response_data,
        )

    @staticmethod
    def serialize_messages(
        messages: list[BaseMessage], include_system_in_user: bool = False
    ) -> tuple[ContentListUnion, str | None]:
        """
        Convert a list of BaseMessages to Google format, extracting system message.

        Google handles system instructions separately from the conversation, so we need to:
        1. Extract any system/developer messages and return them separately as a string
        2. Convert the remaining messages to Content objects
        3. Group consecutive ToolMessages into a single Content with multiple function_response parts

        Args:
            messages: List of messages to convert
            include_system_in_user: If True, system/developer messages are prepended to the first user message

        Returns:
            A tuple of (formatted_messages, system_message) where:
            - formatted_messages: List of Content objects for the conversation
            - system_message: System instruction string or None
        """

        messages = [m.model_copy(deep=True) for m in messages]

        formatted_messages: ContentListUnion = []
        system_message: str | None = None
        system_parts: list[str] = []

        # Collect pending tool response parts to merge consecutive ToolMessages
        pending_tool_response_parts: list[Part] = []

        def flush_tool_responses():
            """Flush any pending tool response parts as a single Content message."""
            nonlocal pending_tool_response_parts
            if pending_tool_response_parts:
                tool_content = Content(role="user", parts=pending_tool_response_parts)
                formatted_messages.append(tool_content)  # type: ignore
                pending_tool_response_parts = []

        for i, message in enumerate(messages):
            role = message.role if hasattr(message, "role") else None

            # Handle system/developer messages
            if isinstance(message, (SystemMessage, DeveloperMessage)):
                flush_tool_responses()  # Flush before non-tool message
                # Extract system message content as string
                if isinstance(message.content, str):
                    if include_system_in_user:
                        system_parts.append(message.content)
                    else:
                        system_message = message.content
                elif message.content is not None:
                    # Handle Iterable of content parts
                    parts = []
                    for part in message.content:
                        if part.type == "text":
                            parts.append(part.text)
                    combined_text = "\n".join(parts)
                    if include_system_in_user:
                        system_parts.append(combined_text)
                    else:
                        system_message = combined_text
                continue

            # Handle tool messages (function responses)
            # Collect consecutive ToolMessages to merge into a single Content
            if isinstance(message, ToolMessage):
                tool_response_part = GoogleMessageSerializer._serialize_tool_message(
                    message
                )
                pending_tool_response_parts.append(tool_response_part)
                continue

            # For any non-tool message, flush pending tool responses first
            flush_tool_responses()

            # Determine the role for non-system messages
            if isinstance(message, UserMessage):
                role = "user"
            elif isinstance(message, AssistantMessage):
                role = "model"
            else:
                # Default to user for any unknown message types
                role = "user"

            # Initialize message parts
            message_parts: list[Part] = []

            # If this is the first user message and we have system parts, prepend them
            if (
                include_system_in_user
                and system_parts
                and role == "user"
                and not formatted_messages
            ):
                system_text = "\n\n".join(system_parts)
                if isinstance(message.content, str) and message.content:
                    message_parts.append(
                        Part.from_text(text=f"{system_text}\n\n{message.content}")
                    )
                else:
                    # Add system text as the first part
                    message_parts.append(Part.from_text(text=system_text))
                system_parts = []  # Clear after using
            else:
                # Extract content and create parts normally
                # Only add text part if content is non-empty string
                if isinstance(message.content, str) and message.content:
                    message_parts = [Part.from_text(text=message.content)]
                elif message.content is not None and not isinstance(
                    message.content, str
                ):
                    # Handle Iterable of content parts
                    for part in message.content:
                        if part.type == "text" and part.text:
                            message_parts.append(Part.from_text(text=part.text))
                        elif part.type == "refusal":
                            message_parts.append(
                                Part.from_text(text=f"[Refusal] {part.refusal}")
                            )
                        elif part.type == "image_url":
                            # Handle images
                            url = part.image_url.url

                            # Format: data:image/jpeg;base64,<data>
                            header, data = url.split(",", 1)
                            # Decode base64 to bytes
                            image_bytes = base64.b64decode(data)

                            # Add image part
                            image_part = Part.from_bytes(
                                data=image_bytes, mime_type="image/jpeg"
                            )

                            message_parts.append(image_part)
                        elif part.type == "document":
                            # Handle PDF documents
                            # Decode base64 to bytes
                            pdf_bytes = base64.b64decode(part.source.data)
                            pdf_part = Part.from_bytes(
                                data=pdf_bytes, mime_type="application/pdf"
                            )
                            message_parts.append(pdf_part)

            # Handle tool_calls for AssistantMessage
            if isinstance(message, AssistantMessage) and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Parse the arguments JSON string to a dict
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {"raw_arguments": tool_call.function.arguments}

                    # Create Part with function_call and thought_signature (if present)
                    # Using Part() directly instead of Part.from_function_call() to include thought_signature
                    function_call_part = Part(
                        function_call=FunctionCall(
                            name=tool_call.function.name,
                            args=args,
                            id=tool_call.id,
                        ),
                        thought_signature=tool_call.thought_signature,
                    )
                    message_parts.append(function_call_part)

            # Create the Content object
            if message_parts:
                final_message = Content(role=role, parts=message_parts)
                # for some reason, the type checker is not able to infer the type of formatted_messages
                formatted_messages.append(final_message)  # type: ignore

        # Flush any remaining tool responses at the end
        flush_tool_responses()

        return formatted_messages, system_message
