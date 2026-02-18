import json
import unittest
from unittest.mock import AsyncMock, patch

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import PreCompactEvent, StopEvent, TextEvent
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.tools import tool


class _FakeChatModel:
    def __init__(self, completions: list[ChatInvokeCompletion], *, call_order: list[str] | None = None):
        self._completions = completions
        self._idx = 0
        self.model = "fake:model"
        self._call_order = call_order

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        del messages, tools, tool_choice, kwargs
        if self._call_order is not None:
            self._call_order.append("invoke")
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


class TestRunnerPrecheckIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_query_runs_precheck_before_each_invoke(self) -> None:
        @tool("Echo value")
        async def Echo(value: str) -> str:
            return value

        call_order: list[str] = []
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        ToolCall(
                            id="tc_echo",
                            function=Function(
                                name="Echo",
                                arguments=json.dumps({"value": "ok"}, ensure_ascii=False),
                            ),
                        )
                    ]
                ),
                ChatInvokeCompletion(content="done"),
            ],
            call_order=call_order,
        )
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Echo,),
                agents=(),
                offload_enabled=False,
            ),
        )
        agent = template.create_runtime()

        async def _fake_precheck(runtime) -> tuple[bool, PreCompactEvent | None, list]:
            del runtime
            call_order.append("precheck")
            return False, None, []

        with patch(
            "comate_agent_sdk.agent.runner.precheck_and_compact",
            new=AsyncMock(side_effect=_fake_precheck),
        ):
            result = await agent.query("run precheck")

        self.assertEqual(result, "done")
        self.assertEqual(call_order, ["precheck", "invoke", "precheck", "invoke"])

    async def test_query_stream_runs_precheck_before_each_invoke(self) -> None:
        @tool("Echo value")
        async def Echo(value: str) -> str:
            return value

        call_order: list[str] = []
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        ToolCall(
                            id="tc_echo",
                            function=Function(
                                name="Echo",
                                arguments=json.dumps({"value": "ok"}, ensure_ascii=False),
                            ),
                        )
                    ]
                ),
                ChatInvokeCompletion(content="done"),
            ],
            call_order=call_order,
        )
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Echo,),
                agents=(),
                offload_enabled=False,
            ),
        )
        agent = template.create_runtime()

        async def _fake_precheck(runtime) -> tuple[bool, PreCompactEvent | None, list]:
            del runtime
            call_order.append("precheck")
            return False, None, []

        with patch(
            "comate_agent_sdk.agent.runner_stream.precheck_and_compact",
            new=AsyncMock(side_effect=_fake_precheck),
        ):
            async for _ in agent.query_stream("run precheck stream"):
                pass

        self.assertEqual(call_order, ["precheck", "invoke", "precheck", "invoke"])

    async def test_query_stream_emits_precheck_event_before_text(self) -> None:
        llm = _FakeChatModel([ChatInvokeCompletion(content="done")])
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
            ),
        )
        agent = template.create_runtime()
        precheck_event = PreCompactEvent(
            current_tokens=120,
            threshold=100,
            trigger="precheck",
        )

        async def _fake_precheck(runtime) -> tuple[bool, PreCompactEvent | None, list]:
            del runtime
            return False, precheck_event, []

        events: list[object] = []
        with patch(
            "comate_agent_sdk.agent.runner_stream.precheck_and_compact",
            new=AsyncMock(side_effect=_fake_precheck),
        ):
            async for event in agent.query_stream("hello"):
                events.append(event)

        event_types = [type(event) for event in events]
        precheck_index = event_types.index(PreCompactEvent)
        text_index = event_types.index(TextEvent)
        self.assertLess(precheck_index, text_index)

        stop_events = [event for event in events if isinstance(event, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
