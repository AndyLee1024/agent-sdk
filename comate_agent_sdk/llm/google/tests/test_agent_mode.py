"""
Tests for Google Gemini agent mode (multi-turn tool calling).

These tests verify that:
1. Tool calls are properly serialized with thought_signature
2. Tool responses are properly linked to tool calls
3. Multi-turn conversations with tools work correctly

Run unit tests:
    python comate_agent_sdk/llm/google/tests/test_agent_mode.py

Run with pytest (if installed):
    pytest comate_agent_sdk/llm/google/tests/test_agent_mode.py -v

For integration tests (require GOOGLE_API_KEY):
    python comate_agent_sdk/llm/google/tests/test_agent_mode.py --integration
    # or with pytest:
    pytest comate_agent_sdk/llm/google/tests/test_agent_mode.py -v -m integration
"""

import json
import os

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    # Create dummy decorators when pytest not available
    class pytest:  # type: ignore
        @staticmethod
        def fixture(func):
            return func

        class mark:
            @staticmethod
            def integration(cls):
                return cls

            @staticmethod
            def asyncio(func):
                return func


from comate_agent_sdk.llm.google.serializer import GoogleMessageSerializer
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    Function,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class TestSerializerToolCalls:
    """Test serializer handles tool calls correctly."""

    def test_assistant_message_with_tool_calls(self):
        """Test that assistant messages with tool_calls are serialized correctly."""
        assistant_msg = AssistantMessage(
            content="I will search for flights.",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function=Function(
                        name="browser_navigate",
                        arguments='{"url": "https://google.com"}',
                    ),
                )
            ],
        )

        messages = [UserMessage(content="Find flights"), assistant_msg]
        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        assert len(formatted) == 2

        # Check model message has both text and function_call parts
        model_msg = formatted[1]
        assert model_msg.role == "model"
        assert len(model_msg.parts) == 2

        # First part should be text
        assert model_msg.parts[0].text == "I will search for flights."

        # Second part should be function_call
        fc = model_msg.parts[1].function_call
        assert fc is not None
        assert fc.name == "browser_navigate"
        assert fc.args == {"url": "https://google.com"}
        assert fc.id == "call_123"

    def test_assistant_message_with_thought_signature(self):
        """Test that thought_signature is preserved in serialization."""
        thought_sig = b"test_thought_signature_bytes"

        assistant_msg = AssistantMessage(
            content="Calling tool",
            tool_calls=[
                ToolCall(
                    id="call_456",
                    function=Function(name="test_tool", arguments="{}"),
                    thought_signature=thought_sig,
                )
            ],
        )

        messages = [UserMessage(content="Test"), assistant_msg]
        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        # Check thought_signature is preserved
        model_msg = formatted[1]
        fc_part = model_msg.parts[1]
        assert fc_part.thought_signature == thought_sig

    def test_tool_message_serialization(self):
        """Test that tool messages (function responses) are serialized correctly."""
        tool_msg = ToolMessage(
            tool_call_id="call_123",
            tool_name="browser_navigate",
            content='{"success": true, "message": "Navigated successfully"}',
        )

        messages = [UserMessage(content="Test"), tool_msg]
        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        # Tool response should be in a user role message
        tool_response_msg = formatted[1]
        assert tool_response_msg.role == "user"

        # Check function_response
        fr = tool_response_msg.parts[0].function_response
        assert fr is not None
        assert fr.name == "browser_navigate"
        assert fr.response == {"success": True, "message": "Navigated successfully"}

    def test_full_agent_turn(self):
        """Test a full agent turn: user -> assistant (with tool) -> tool result."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Navigate to google.com"),
            AssistantMessage(
                content="I will navigate to Google.",
                tool_calls=[
                    ToolCall(
                        id="call_nav_1",
                        function=Function(
                            name="browser_navigate",
                            arguments='{"url": "https://google.com"}',
                        ),
                        thought_signature=b"sig1",
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="call_nav_1",
                tool_name="browser_navigate",
                content='{"success": true}',
            ),
        ]

        formatted, system = GoogleMessageSerializer.serialize_messages(messages)

        # System message should be extracted
        assert system == "You are a helpful assistant."

        # Should have 3 content messages (user, model, user/tool-response)
        assert len(formatted) == 3

        # Message 0: user
        assert formatted[0].role == "user"
        assert formatted[0].parts[0].text == "Navigate to google.com"

        # Message 1: model with text + function_call
        assert formatted[1].role == "model"
        assert len(formatted[1].parts) == 2
        assert formatted[1].parts[0].text == "I will navigate to Google."
        assert formatted[1].parts[1].function_call.name == "browser_navigate"
        assert formatted[1].parts[1].thought_signature == b"sig1"

        # Message 2: function response
        assert formatted[2].role == "user"
        assert formatted[2].parts[0].function_response.name == "browser_navigate"

    def test_multiple_tool_calls_single_message(self):
        """Test assistant message with multiple tool calls."""
        messages = [
            UserMessage(content="Do multiple things"),
            AssistantMessage(
                content="I will do two things.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=Function(name="tool_a", arguments='{"x": 1}'),
                        thought_signature=b"sig_a",
                    ),
                    ToolCall(
                        id="call_2",
                        function=Function(name="tool_b", arguments='{"y": 2}'),
                        thought_signature=b"sig_b",
                    ),
                ],
            ),
            ToolMessage(
                tool_call_id="call_1", tool_name="tool_a", content='{"result": "a"}'
            ),
            ToolMessage(
                tool_call_id="call_2", tool_name="tool_b", content='{"result": "b"}'
            ),
        ]

        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        # Should have 3 messages: user, model (with tool calls), user (with merged tool responses)
        assert len(formatted) == 3

        # Message 1 (model) should have 3 parts: text + 2 function calls
        model_msg = formatted[1]
        assert len(model_msg.parts) == 3
        assert model_msg.parts[0].text == "I will do two things."
        assert model_msg.parts[1].function_call.name == "tool_a"
        assert model_msg.parts[2].function_call.name == "tool_b"

        # Message 2 should have both tool responses merged into one Content
        tool_response_msg = formatted[2]
        assert tool_response_msg.role == "user"
        assert len(tool_response_msg.parts) == 2
        assert tool_response_msg.parts[0].function_response.name == "tool_a"
        assert tool_response_msg.parts[1].function_response.name == "tool_b"

    def test_empty_assistant_content_with_tool_calls(self):
        """Test that empty assistant content doesn't create empty text parts."""
        messages = [
            UserMessage(content="Do something"),
            AssistantMessage(
                content="",  # Empty content
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=Function(name="some_tool", arguments="{}"),
                    ),
                ],
            ),
        ]

        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        # Model message should only have function_call part, no empty text
        model_msg = formatted[1]
        assert len(model_msg.parts) == 1
        assert model_msg.parts[0].function_call is not None
        assert model_msg.parts[0].text is None

    def test_none_assistant_content_with_tool_calls(self):
        """Test that None assistant content doesn't cause issues."""
        messages = [
            UserMessage(content="Do something"),
            AssistantMessage(
                content=None,  # None content
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=Function(name="some_tool", arguments="{}"),
                    ),
                ],
            ),
        ]

        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        # Model message should only have function_call part
        model_msg = formatted[1]
        assert len(model_msg.parts) == 1
        assert model_msg.parts[0].function_call is not None

    def test_tool_error_response(self):
        """Test that tool error responses are serialized correctly."""
        tool_msg = ToolMessage(
            tool_call_id="call_err",
            tool_name="failing_tool",
            content="Something went wrong",
            is_error=True,
        )

        messages = [UserMessage(content="Test"), tool_msg]
        formatted, _ = GoogleMessageSerializer.serialize_messages(messages)

        fr = formatted[1].parts[0].function_response
        assert fr.response == {"error": "Something went wrong"}


# Integration tests - require GOOGLE_API_KEY
@pytest.mark.integration
class TestGeminiAgentModeIntegration:
    """Integration tests that actually call Gemini API."""

    @pytest.fixture
    def api_key(self):
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            pytest.skip("GOOGLE_API_KEY not set")
        return key

    @pytest.fixture
    def chat_model(self, api_key):
        from comate_agent_sdk.llm.google.chat import ChatGoogle

        return ChatGoogle(model="gemini-2.0-flash", api_key=api_key)

    @pytest.fixture
    def sample_tools(self):
        from comate_agent_sdk.llm.base import ToolDefinition

        return [
            ToolDefinition(
                name="get_weather",
                description="Get weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                    },
                    "required": ["location"],
                },
            ),
            ToolDefinition(
                name="search_web",
                description="Search the web for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_single_tool_call_round_trip(self, chat_model, sample_tools):
        """Test a complete tool call round trip with Gemini."""
        # First turn: user asks, model should call a tool
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Use tools when appropriate."
            ),
            UserMessage(content="What's the weather in Paris?"),
        ]

        response1 = await chat_model.ainvoke(
            messages, tools=sample_tools, tool_choice="required"
        )

        assert response1.has_tool_calls, "Expected model to make a tool call"
        assert len(response1.tool_calls) >= 1

        tool_call = response1.tool_calls[0]
        print(f"\nTool call: {tool_call.function.name}({tool_call.function.arguments})")
        print(f"thought_signature present: {tool_call.thought_signature is not None}")

        # Add assistant response to messages
        messages.append(
            AssistantMessage(
                content=response1.content,
                tool_calls=response1.tool_calls,
            )
        )

        # Add tool result
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                content=json.dumps(
                    {"temperature": 22, "condition": "sunny", "humidity": 45}
                ),
            )
        )

        # Second turn: model should respond with the weather info
        response2 = await chat_model.ainvoke(messages, tools=sample_tools)

        print(f"\nFinal response: {response2.content}")
        assert response2.content is not None, (
            "Expected model to provide a text response"
        )

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_tools(self, chat_model, sample_tools):
        """Test multiple turns of conversation with tool usage."""
        messages = [
            SystemMessage(content="You are a helpful assistant with tool access."),
            UserMessage(content="What's the weather in Tokyo?"),
        ]

        # Turn 1: Get weather
        response1 = await chat_model.ainvoke(
            messages, tools=sample_tools, tool_choice="required"
        )

        if response1.has_tool_calls:
            tc1 = response1.tool_calls[0]
            messages.append(
                AssistantMessage(
                    content=response1.content, tool_calls=response1.tool_calls
                )
            )
            messages.append(
                ToolMessage(
                    tool_call_id=tc1.id,
                    tool_name=tc1.function.name,
                    content=json.dumps({"temperature": 28, "condition": "cloudy"}),
                )
            )

            # Get model's interpretation
            response1b = await chat_model.ainvoke(messages, tools=sample_tools)
            messages.append(
                AssistantMessage(
                    content=response1b.content, tool_calls=response1b.tool_calls
                )
            )

        # Turn 2: Follow-up question
        messages.append(UserMessage(content="And what about New York?"))

        response2 = await chat_model.ainvoke(
            messages, tools=sample_tools, tool_choice="required"
        )

        if response2.has_tool_calls:
            tc2 = response2.tool_calls[0]
            print(f"\nSecond tool call: {tc2.function.name}({tc2.function.arguments})")
            messages.append(
                AssistantMessage(
                    content=response2.content, tool_calls=response2.tool_calls
                )
            )
            messages.append(
                ToolMessage(
                    tool_call_id=tc2.id,
                    tool_name=tc2.function.name,
                    content=json.dumps({"temperature": 15, "condition": "rainy"}),
                )
            )

            # Final response
            response2b = await chat_model.ainvoke(messages, tools=sample_tools)
            print(f"\nFinal response: {response2b.content}")
            assert response2b.content is not None

    @pytest.mark.asyncio
    async def test_tool_call_without_content(self, chat_model, sample_tools):
        """Test that tool calls work even when assistant has no text content."""
        messages = [
            UserMessage(content="Search for Python tutorials"),
        ]

        response = await chat_model.ainvoke(
            messages, tools=sample_tools, tool_choice="required"
        )

        # Model may or may not include text with the tool call
        assert response.has_tool_calls, "Expected tool call"

        # Simulate tool response
        tc = response.tool_calls[0]
        messages.append(
            AssistantMessage(
                content=response.content,  # Could be None
                tool_calls=response.tool_calls,
            )
        )
        messages.append(
            ToolMessage(
                tool_call_id=tc.id,
                tool_name=tc.function.name,
                content=json.dumps({"results": ["Tutorial 1", "Tutorial 2"]}),
            )
        )

        # Model should be able to continue
        response2 = await chat_model.ainvoke(messages, tools=sample_tools)
        print(f"\nResponse after tool: {response2.content}")


def run_unit_tests():
    """Run unit tests without pytest."""
    print("=" * 60)
    print("Running Google Gemini Serializer Unit Tests")
    print("=" * 60)

    tests = TestSerializerToolCalls()
    test_methods = [m for m in dir(tests) if m.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            print(f"\n{method_name}... ", end="")
            getattr(tests, method_name)()
            print("PASSED")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


async def run_integration_tests():
    """Run integration tests (requires GOOGLE_API_KEY)."""
    import asyncio

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY=your_key_here")
        return False

    from comate_agent_sdk.llm.base import ToolDefinition
    from comate_agent_sdk.llm.google.chat import ChatGoogle

    print("=" * 60)
    print("Running Google Gemini Integration Tests")
    print("=" * 60)

    chat_model = ChatGoogle(model="gemini-2.0-flash", api_key=api_key)

    sample_tools = [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                },
                "required": ["location"],
            },
        ),
    ]

    try:
        print("\nTest: Single tool call round trip")

        messages = [
            SystemMessage(
                content="You are a helpful assistant. Use tools when appropriate."
            ),
            UserMessage(content="What's the weather in Paris?"),
        ]

        response1 = await chat_model.ainvoke(
            messages, tools=sample_tools, tool_choice="required"
        )

        if not response1.has_tool_calls:
            print("  FAILED: Expected model to make a tool call")
            return False

        tool_call = response1.tool_calls[0]
        print(f"  Tool call: {tool_call.function.name}({tool_call.function.arguments})")
        print(f"  thought_signature present: {tool_call.thought_signature is not None}")

        # Add assistant response to messages
        messages.append(
            AssistantMessage(
                content=response1.content,
                tool_calls=response1.tool_calls,
            )
        )

        # Add tool result
        messages.append(
            ToolMessage(
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                content=json.dumps(
                    {"temperature": 22, "condition": "sunny", "humidity": 45}
                ),
            )
        )

        # Second turn: model should respond with the weather info
        response2 = await chat_model.ainvoke(messages, tools=sample_tools)

        if response2.content is None:
            print("  FAILED: Expected model to provide a text response")
            return False

        print(f"  Final response: {response2.content[:100]}...")
        print("  PASSED")

        print("\n" + "=" * 60)
        print("Integration tests PASSED!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    if "--integration" in sys.argv:
        # Run integration tests
        import asyncio

        success = asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)
    elif HAS_PYTEST and "--pytest" in sys.argv:
        # Run with pytest
        pytest.main([__file__, "-v", "-m", "not integration"])
    else:
        # Run unit tests without pytest
        success = run_unit_tests()
        sys.exit(0 if success else 1)
