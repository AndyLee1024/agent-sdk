import asyncio
import json
import time
import unittest

from bu_agent_sdk.agent import Agent
from bu_agent_sdk.agent.events import SubagentStartEvent, SubagentStopEvent, ToolResultEvent
from bu_agent_sdk.llm.messages import Function, ToolCall
from bu_agent_sdk.llm.views import ChatInvokeCompletion
from bu_agent_sdk.tools import tool


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


class TestParallelTaskEvents(unittest.IsolatedAsyncioTestCase):
    async def test_query_stream_emits_subagent_start_stop_and_completion_order(self) -> None:
        delays = {"a": 0.25, "b": 0.05, "c": 0.15}

        @tool("Task tool (test)")
        async def Task(subagent_type: str, prompt: str, description: str = "") -> str:
            await asyncio.sleep(delays[subagent_type])
            return f"ok:{subagent_type}"

        tool_calls = [
            ToolCall(
                id="tc_a",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {"subagent_type": "a", "prompt": "p", "description": "alpha"},
                        ensure_ascii=False,
                    ),
                ),
            ),
            ToolCall(
                id="tc_b",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {"subagent_type": "b", "prompt": "p", "description": "beta"},
                        ensure_ascii=False,
                    ),
                ),
            ),
            ToolCall(
                id="tc_c",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {"subagent_type": "c", "prompt": "p", "description": "gamma"},
                        ensure_ascii=False,
                    ),
                ),
            ),
        ]

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(tool_calls=tool_calls),
                ChatInvokeCompletion(content="done"),
            ]
        )

        agent = Agent(
            llm=llm,  # type: ignore[arg-type]
            tools=[Task],
            offload_enabled=False,
            task_parallel_enabled=True,
            task_parallel_max_concurrency=4,
        )

        events = []
        async for event in agent.query_stream("hi"):
            events.append(event)

        start_names = [e.subagent_name for e in events if isinstance(e, SubagentStartEvent)]
        stop_names = [e.subagent_name for e in events if isinstance(e, SubagentStopEvent)]

        self.assertEqual(start_names, ["a", "b", "c"])
        self.assertEqual(stop_names, ["b", "c", "a"])

        tool_results = [
            e.tool_call_id
            for e in events
            if isinstance(e, ToolResultEvent) and e.tool == "Task"
        ]
        self.assertEqual(tool_results, ["tc_b", "tc_c", "tc_a"])

        # ToolMessage(s) are written to context in call order for reproducibility
        tool_msgs = [m for m in agent.messages if getattr(m, "role", None) == "tool"]
        tool_msg_ids = [m.tool_call_id for m in tool_msgs if getattr(m, "tool_name", "") == "Task"]
        self.assertEqual(tool_msg_ids, ["tc_a", "tc_b", "tc_c"])

    async def test_query_parallel_task_writes_context_in_call_order_and_tasks_overlap(self) -> None:
        starts: list[float] = []
        ends: list[float] = []

        @tool("Task tool (test)")
        async def Task(subagent_type: str, prompt: str, description: str = "") -> str:
            starts.append(time.perf_counter())
            await asyncio.sleep(0.2)
            ends.append(time.perf_counter())
            return f"ok:{subagent_type}"

        tool_calls = [
            ToolCall(
                id="tc1",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {"subagent_type": "a", "prompt": "p", "description": "d1"},
                        ensure_ascii=False,
                    ),
                ),
            ),
            ToolCall(
                id="tc2",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {"subagent_type": "b", "prompt": "p", "description": "d2"},
                        ensure_ascii=False,
                    ),
                ),
            ),
            ToolCall(
                id="tc3",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {"subagent_type": "c", "prompt": "p", "description": "d3"},
                        ensure_ascii=False,
                    ),
                ),
            ),
        ]

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(tool_calls=tool_calls),
                ChatInvokeCompletion(content="done"),
            ]
        )

        agent = Agent(
            llm=llm,  # type: ignore[arg-type]
            tools=[Task],
            offload_enabled=False,
            task_parallel_enabled=True,
            task_parallel_max_concurrency=4,
        )

        _ = await agent.query("hi")

        self.assertEqual(len(starts), 3)
        self.assertEqual(len(ends), 3)
        # If tasks are parallelized, all starts should happen before the first end.
        self.assertLess(max(starts), min(ends))

        tool_msgs = [m for m in agent.messages if getattr(m, "role", None) == "tool"]
        tool_msg_ids = [m.tool_call_id for m in tool_msgs if getattr(m, "tool_name", "") == "Task"]
        self.assertEqual(tool_msg_ids, ["tc1", "tc2", "tc3"])

