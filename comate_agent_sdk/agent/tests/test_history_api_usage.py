import unittest

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.context.observer import EventType
from comate_agent_sdk.llm.messages import AssistantMessage, UserMessage


class _FakeChatModel:
    def __init__(self) -> None:
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("This test should not call the LLM")


class TestHistoryApiUsage(unittest.TestCase):
    def test_load_history_replaces_conversation_with_event(self) -> None:
        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
                setting_sources=None,
            ),
        )
        runtime = template.create_runtime()
        runtime._context.add_message(UserMessage(content="old"))

        replaced_before = sum(
            1
            for event in runtime._context.event_bus.event_log
            if event.event_type == EventType.CONVERSATION_REPLACED
        )
        runtime.load_history([
            UserMessage(content="new-user"),
            AssistantMessage(content="new-assistant"),
        ])
        replaced_after = sum(
            1
            for event in runtime._context.event_bus.event_log
            if event.event_type == EventType.CONVERSATION_REPLACED
        )

        self.assertEqual(replaced_after, replaced_before + 1)
        self.assertEqual(
            [message.role for message in runtime._context.conversation_messages],
            ["user", "assistant"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
