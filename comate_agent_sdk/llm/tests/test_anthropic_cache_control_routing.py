import unittest

from comate_agent_sdk.llm.anthropic.serializer import AnthropicMessageSerializer
from comate_agent_sdk.llm.messages import SystemMessage, UserMessage


class TestAnthropicCacheControlRouting(unittest.TestCase):
    def test_system_prompt_never_uses_cache_control(self) -> None:
        messages = [
            SystemMessage(content="SYSTEM_PROMPT", cache=True),
            UserMessage(content="hello"),
        ]

        _, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)
        self.assertIsInstance(system_prompt, str)
        self.assertEqual(system_prompt, "SYSTEM_PROMPT")

    def test_session_state_user_message_uses_ephemeral_cache_control(self) -> None:
        session_state = UserMessage(
            content="<output_style>\ncompact\n</output_style>",
            is_meta=True,
            cache=False,
        )
        messages = [
            SystemMessage(content="SYSTEM_PROMPT", cache=True),
            session_state,
        ]

        serialized_messages, _ = AnthropicMessageSerializer.serialize_messages(messages)
        self.assertEqual(len(serialized_messages), 1)
        user_msg = serialized_messages[0]
        self.assertEqual(user_msg["role"], "user")
        self.assertIsInstance(user_msg["content"], list)
        block = user_msg["content"][0]
        self.assertEqual(block["type"], "text")
        self.assertEqual(block["cache_control"]["type"], "ephemeral")


if __name__ == "__main__":
    unittest.main(verbosity=2)
