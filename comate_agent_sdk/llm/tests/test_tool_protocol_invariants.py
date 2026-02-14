import unittest

from comate_agent_sdk.llm.anthropic.serializer import AnthropicMessageSerializer
from comate_agent_sdk.llm.messages import AssistantMessage, Function, ToolCall, ToolMessage, UserMessage
from comate_agent_sdk.llm.openai.serializer import OpenAIMessageSerializer


def _build_tool_call(tool_call_id: str, *, name: str = "Read") -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        function=Function(name=name, arguments="{}"),
    )


def _build_tool_result(tool_call_id: str, *, name: str = "Read") -> ToolMessage:
    return ToolMessage(
        tool_call_id=tool_call_id,
        tool_name=name,
        content=f"result:{tool_call_id}",
    )


class TestToolProtocolInvariants(unittest.TestCase):
    def test_openai_and_anthropic_raise_when_non_tool_message_breaks_tool_block(self) -> None:
        messages = [
            UserMessage(content="hello"),
            AssistantMessage(content=None, tool_calls=[_build_tool_call("tc1")]),
            UserMessage(content="hook", is_meta=True),
            _build_tool_result("tc1"),
        ]

        with self.assertRaisesRegex(ValueError, "Invalid tool-call message sequence"):
            OpenAIMessageSerializer.serialize_messages(messages)

        with self.assertRaisesRegex(ValueError, "Invalid tool-call message sequence"):
            AnthropicMessageSerializer.serialize_messages(messages)

    def test_openai_and_anthropic_raise_when_tool_results_missing(self) -> None:
        messages = [
            AssistantMessage(
                content=None,
                tool_calls=[_build_tool_call("tc1"), _build_tool_call("tc2")],
            ),
            _build_tool_result("tc1"),
        ]

        with self.assertRaisesRegex(ValueError, "Invalid tool-call message sequence"):
            OpenAIMessageSerializer.serialize_messages(messages)

        with self.assertRaisesRegex(ValueError, "Invalid tool-call message sequence"):
            AnthropicMessageSerializer.serialize_messages(messages)

    def test_openai_and_anthropic_accept_valid_contiguous_tool_block(self) -> None:
        messages = [
            UserMessage(content="hello"),
            AssistantMessage(
                content=None,
                tool_calls=[_build_tool_call("tc1"), _build_tool_call("tc2")],
            ),
            _build_tool_result("tc2"),
            _build_tool_result("tc1"),
            UserMessage(content="done"),
        ]

        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)
        anthropic_messages, _ = AnthropicMessageSerializer.serialize_messages(messages)
        self.assertEqual(len(openai_messages), 5)
        self.assertEqual(len(anthropic_messages), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
