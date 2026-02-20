import json
import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import StopEvent, UserQuestionEvent
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.tools import tool
from comate_agent_sdk.system_tools.tools import AskUserQuestion


class _FakeChatModel:
    def __init__(self, completions: list[ChatInvokeCompletion]):
        self._completions = completions
        self._idx = 0
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


class TestRunnerStreamAskUserQuestion(unittest.IsolatedAsyncioTestCase):
    async def test_query_stream_uses_raw_envelope_for_user_question_event(self) -> None:
        tool_call = ToolCall(
            id="tc_question_1",
            function=Function(
                name="AskUserQuestion",
                arguments=json.dumps(
                    {
                        "questions": [
                            {
                                "question": "Which implementation do you prefer?",
                                "header": "Impl",
                                "options": [
                                    {"label": "Plan A", "description": "Stable and explicit"},
                                    {"label": "Plan B", "description": "Smaller change set"},
                                ],
                                "multiSelect": False,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            ),
        )

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(tool_calls=[tool_call]),
            ]
        )

        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(AskUserQuestion,),
                agents=(),
                offload_enabled=False,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("need input"):
            events.append(event)

        question_events = [e for e in events if isinstance(e, UserQuestionEvent)]
        self.assertEqual(len(question_events), 1)
        self.assertEqual(len(question_events[0].questions), 1)
        self.assertEqual(question_events[0].questions[0]["header"], "Impl")

        stop_events = [e for e in events if isinstance(e, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "waiting_for_input")

        tool_messages = [m for m in agent.messages if getattr(m, "role", None) == "tool"]
        self.assertEqual(len(tool_messages), 1)
        tool_message = tool_messages[0]
        self.assertIsNotNone(getattr(tool_message, "raw_envelope", None))
        self.assertIn("# AskUserQuestion", tool_message.text)

    async def test_query_stream_does_not_fallback_to_args_when_envelope_missing(self) -> None:
        @tool("Broken AskUserQuestion that returns plain text", name="AskUserQuestion")
        async def BrokenAskUserQuestion(questions: list[dict]) -> str:
            del questions
            return "plain text without envelope"

        tool_call = ToolCall(
            id="tc_question_2",
            function=Function(
                name="AskUserQuestion",
                arguments=json.dumps(
                    {
                        "questions": [
                            {
                                "question": "Should not be emitted?",
                                "header": "LegacyFallback",
                                "options": [
                                    {"label": "Yes", "description": "legacy"},
                                    {"label": "No", "description": "strict"},
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            ),
        )

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(tool_calls=[tool_call]),
                ChatInvokeCompletion(content="done"),
            ]
        )
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(BrokenAskUserQuestion,),
                agents=(),
                offload_enabled=False,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("need input"):
            events.append(event)

        question_events = [e for e in events if isinstance(e, UserQuestionEvent)]
        self.assertEqual(question_events, [])

        stop_events = [e for e in events if isinstance(e, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
