import asyncio
import unittest

from comate_agent_sdk.agent.compaction.service import CompactionService
from comate_agent_sdk.llm.messages import (
    AssistantMessage,
    DeveloperMessage,
    Function,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from comate_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage


class _FakeLLM:
    def __init__(self, response: ChatInvokeCompletion) -> None:
        self.response = response
        self.model = "fake-model"
        self.last_messages = None

    async def ainvoke(self, *, messages, tools=None, tool_choice=None):
        self.last_messages = list(messages)
        return self.response


class TestCompactionServiceSummaryDiagnostics(unittest.TestCase):
    def test_blank_content_is_classified_with_detail(self) -> None:
        llm = _FakeLLM(
            ChatInvokeCompletion(
                content="   \n\t",
                stop_reason="stop",
                usage=None,
            )
        )
        service = CompactionService(llm=llm)

        result = asyncio.run(service.compact([UserMessage(content="hello")], llm))

        self.assertFalse(result.compacted)
        self.assertIsNone(result.summary)
        self.assertIsNotNone(llm.last_messages)
        self.assertEqual(len(llm.last_messages or []), 2)
        self.assertIsInstance(llm.last_messages[0], SystemMessage)
        self.assertIsInstance(llm.last_messages[1], UserMessage)
        self.assertEqual(result.failure_reason, "blank_content")
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual(result.raw_content_length, 5)
        self.assertIsNotNone(result.failure_detail)
        self.assertIn("raw_content_len=5", str(result.failure_detail))
        self.assertIn("has_summary_tags=False", str(result.failure_detail))

    def test_empty_summary_tag_body_is_classified(self) -> None:
        llm = _FakeLLM(
            ChatInvokeCompletion(
                content="<summary>   </summary>",
                stop_reason="stop",
                usage=ChatInvokeUsage(
                    prompt_tokens=10,
                    prompt_cached_tokens=0,
                    prompt_cache_creation_tokens=0,
                    prompt_image_tokens=None,
                    completion_tokens=12,
                    total_tokens=22,
                ),
            )
        )
        service = CompactionService(llm=llm)

        result = asyncio.run(service.compact([UserMessage(content="hello")], llm))

        self.assertFalse(result.compacted)
        self.assertIsNone(result.summary)
        self.assertIsNotNone(llm.last_messages)
        self.assertEqual(len(llm.last_messages or []), 2)
        self.assertIsInstance(llm.last_messages[0], SystemMessage)
        self.assertIsInstance(llm.last_messages[1], UserMessage)
        self.assertEqual(result.failure_reason, "empty_tag_body")
        self.assertIsNotNone(result.failure_detail)
        self.assertIn("completion_tokens=12", str(result.failure_detail))
        self.assertIn("has_summary_tags=True", str(result.failure_detail))
        self.assertIn("empty_tag_body=True", str(result.failure_detail))

    def test_serialize_messages_to_text(self) -> None:
        service = CompactionService()
        messages = [
            UserMessage(content="user <input> & value"),
            AssistantMessage(
                content="assistant response",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=Function(name="Read", arguments='{"path":"a<b>.py"}'),
                    )
                ],
            ),
            ToolMessage(
                tool_call_id="call_1",
                tool_name="Read",
                content="tool <result> & done",
            ),
            DeveloperMessage(content="dev-only"),
        ]

        serialized = service._serialize_messages_to_text(messages)

        self.assertIn("<conversation>", serialized)
        self.assertIn("</conversation>", serialized)
        self.assertIn('<message role="user">', serialized)
        self.assertIn("user &lt;input&gt; &amp; value", serialized)
        self.assertIn('<message role="assistant">', serialized)
        self.assertIn("<tool_calls>", serialized)
        self.assertIn("- Read({&quot;path&quot;:&quot;a&lt;b&gt;.py&quot;})", serialized)
        self.assertIn('<message role="tool" name="Read" tool_call_id="call_1">', serialized)
        self.assertIn("tool &lt;result&gt; &amp; done", serialized)
        self.assertIn('<message role="unknown">', serialized)
        self.assertIn("dev-only", serialized)

    def test_compact_uses_system_and_user_messages(self) -> None:
        llm = _FakeLLM(
            ChatInvokeCompletion(
                content="<summary>done</summary>",
                stop_reason="stop",
                usage=None,
            )
        )
        service = CompactionService(llm=llm)
        result = asyncio.run(
            service.compact(
                [
                    UserMessage(content="hello"),
                    AssistantMessage(content="world"),
                ],
                llm,
            )
        )

        self.assertTrue(result.compacted)
        self.assertEqual(result.summary, "done")
        self.assertIsNotNone(llm.last_messages)
        self.assertEqual(len(llm.last_messages or []), 2)
        self.assertIsInstance(llm.last_messages[0], SystemMessage)
        self.assertIsInstance(llm.last_messages[1], UserMessage)
        user_payload = llm.last_messages[1].text
        self.assertIn(service.config.summary_prompt, user_payload)
        self.assertIn("<conversation>", user_payload)
        self.assertIn('<message role="assistant">', user_payload)


if __name__ == "__main__":
    unittest.main()
