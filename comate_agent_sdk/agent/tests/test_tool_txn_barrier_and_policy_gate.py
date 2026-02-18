import asyncio
import json
import unittest
from types import SimpleNamespace

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import StopEvent, ToolResultEvent, UserQuestionEvent
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.protocol_invariants import validate_tool_call_sequence
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.system_tools.tools import AskUserQuestion
from comate_agent_sdk.tools import tool


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
        del messages, tools, tool_choice, kwargs
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


def _tool_call(*, tool_call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        function=Function(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


class TestToolTxnPolicyGate(unittest.IsolatedAsyncioTestCase):
    async def test_policy_gate_mixed_askuser_commits_tool_errors_then_repairs(self) -> None:
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        _tool_call(
                            tool_call_id="tc_ask_1",
                            name="AskUserQuestion",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Pick one",
                                        "header": "Pick",
                                        "options": [
                                            {"label": "A", "description": "a"},
                                            {"label": "B", "description": "b"},
                                        ],
                                        "multiSelect": False,
                                    }
                                ]
                            },
                        ),
                        _tool_call(
                            tool_call_id="tc_read_1",
                            name="Read",
                            arguments={"file_path": "README.md"},
                        ),
                    ]
                ),
                ChatInvokeCompletion(
                    tool_calls=[
                        _tool_call(
                            tool_call_id="tc_ask_2",
                            name="AskUserQuestion",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Pick one",
                                        "header": "Pick",
                                        "options": [
                                            {"label": "A", "description": "a"},
                                            {"label": "B", "description": "b"},
                                        ],
                                        "multiSelect": False,
                                    }
                                ]
                            },
                        )
                    ]
                ),
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

        events: list[object] = []
        async for event in agent.query_stream("need input"):
            events.append(event)

        user_questions = [e for e in events if isinstance(e, UserQuestionEvent)]
        self.assertEqual(len(user_questions), 1)
        self.assertEqual(user_questions[0].tool_call_id, "tc_ask_2")

        stop_events = [e for e in events if isinstance(e, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "waiting_for_input")

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        first_attempt_results = [e for e in tool_results if e.tool_call_id in {"tc_ask_1", "tc_read_1"}]
        self.assertEqual(len(first_attempt_results), 2)
        self.assertTrue(all(e.is_error for e in first_attempt_results))

        validate_tool_call_sequence(agent.messages, provider="anthropic")

    async def test_policy_gate_mixed_retry_then_truncates_to_askuser(self) -> None:
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        _tool_call(
                            tool_call_id="tc_ask_1",
                            name="AskUserQuestion",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Pick one",
                                        "header": "Pick",
                                        "options": [
                                            {"label": "A", "description": "a"},
                                            {"label": "B", "description": "b"},
                                        ],
                                        "multiSelect": False,
                                    }
                                ]
                            },
                        ),
                        _tool_call(
                            tool_call_id="tc_read_1",
                            name="Read",
                            arguments={"file_path": "README.md"},
                        ),
                    ]
                ),
                ChatInvokeCompletion(
                    tool_calls=[
                        _tool_call(
                            tool_call_id="tc_ask_2",
                            name="AskUserQuestion",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Pick one",
                                        "header": "Pick",
                                        "options": [
                                            {"label": "A", "description": "a"},
                                            {"label": "B", "description": "b"},
                                        ],
                                        "multiSelect": False,
                                    }
                                ]
                            },
                        ),
                        _tool_call(
                            tool_call_id="tc_read_2",
                            name="Read",
                            arguments={"file_path": "README.md"},
                        ),
                    ]
                ),
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

        async for _event in agent.query_stream("need input"):
            pass

        assistant_messages = [m for m in agent.messages if getattr(m, "role", None) == "assistant"]
        self.assertGreaterEqual(len(assistant_messages), 2)
        last_with_tool_calls = next(
            (m for m in reversed(assistant_messages) if getattr(m, "tool_calls", None)),
            None,
        )
        self.assertIsNotNone(last_with_tool_calls)
        tool_calls = getattr(last_with_tool_calls, "tool_calls", None) or []
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].id, "tc_ask_2")
        self.assertEqual(tool_calls[0].function.name, "AskUserQuestion")

        tool_ids = [m.tool_call_id for m in agent.messages if getattr(m, "role", None) == "tool"]
        self.assertNotIn("tc_read_2", tool_ids)


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.count_calls = 0

    async def count_tokens(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        self.count_calls += 1
        return SimpleNamespace(input_tokens=123)


class _FakeAnthropicClient:
    def __init__(self, messages: _FakeAnthropicMessages) -> None:
        self.messages = messages


class _FakeAnthropicChatModel:
    def __init__(self, completions: list[ChatInvokeCompletion], *, messages: _FakeAnthropicMessages):
        self._completions = completions
        self._idx = 0
        self.model = "fake:anthropic"
        self._messages = messages

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def name(self) -> str:
        return self.model

    def get_client(self):  # type: ignore[no-untyped-def]
        return _FakeAnthropicClient(self._messages)

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        del messages, tools, tool_choice, kwargs
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


class TestToolTxnBarrierTokenCount(unittest.IsolatedAsyncioTestCase):
    async def test_get_context_info_does_not_block_while_tool_is_running(self) -> None:
        """get_context_info() must return promptly even when a tool is executing."""
        started = asyncio.Event()

        @tool("Slow tool")
        async def SlowTool(path: str) -> str:
            del path
            started.set()
            await asyncio.sleep(0.3)
            return "ok"

        messages = _FakeAnthropicMessages()
        llm = _FakeAnthropicChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        _tool_call(
                            tool_call_id="tc_slow",
                            name="SlowTool",
                            arguments={"path": "x"},
                        )
                    ]
                ),
                ChatInvokeCompletion(content="done"),
            ],
            messages=messages,
        )

        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(SlowTool,),
                agents=(),
                offload_enabled=False,
            ),
        )
        agent = template.create_runtime()

        async def _drive() -> None:
            async for _event in agent.query_stream("run"):
                pass

        run_task = asyncio.create_task(_drive())
        await asyncio.wait_for(started.wait(), timeout=1.0)

        # get_context_info uses incremental IR delta now, no API calls needed —
        # verify it completes promptly without blocking on the running tool.
        ctx_info = await asyncio.wait_for(agent.get_context_info(), timeout=0.5)
        self.assertIsNotNone(ctx_info)

        await asyncio.wait_for(run_task, timeout=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

