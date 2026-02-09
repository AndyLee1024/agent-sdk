import asyncio
import json
import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import (
    SubagentProgressEvent,
    SubagentStartEvent,
    SubagentStopEvent,
    UsageDeltaEvent,
)
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from comate_agent_sdk.tools import tool
from comate_agent_sdk.tools.system_context import get_system_tool_context


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


def _build_task_call(*, tool_call_id: str, subagent_type: str, description: str) -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        function=Function(
            name="Task",
            arguments=json.dumps(
                {
                    "subagent_type": subagent_type,
                    "prompt": "p",
                    "description": description,
                },
                ensure_ascii=False,
            ),
        ),
    )


class TestSubagentProgressEvents(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_task_emits_usage_and_progress_events(self) -> None:
        @tool("Task tool (test)")
        async def Task(subagent_type: str, prompt: str, description: str = "") -> str:
            ctx = get_system_tool_context()
            assert ctx.token_cost is not None
            assert ctx.tool_call_id is not None
            source_prefix = f"subagent:{subagent_type}:{ctx.tool_call_id}"
            ctx.token_cost.add_usage(
                "subagent-mid",
                ChatInvokeUsage(
                    prompt_tokens=10,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=8,
                    total_tokens=18,
                ),
                level="MID",
                source=source_prefix,
            )
            await asyncio.sleep(0.12)
            ctx.token_cost.add_usage(
                "subagent-low",
                ChatInvokeUsage(
                    prompt_tokens=7,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=5,
                    total_tokens=12,
                ),
                level="LOW",
                source=f"{source_prefix}:webfetch",
            )
            return f"ok:{subagent_type}"

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        _build_task_call(
                            tool_call_id="tc_explorer",
                            subagent_type="Explorer",
                            description="探索 ContextIR 功能",
                        )
                    ]
                ),
                ChatInvokeCompletion(content="done"),
            ]
        )

        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Task,),
                agents=(),
                offload_enabled=False,
                task_parallel_enabled=True,
                task_parallel_max_concurrency=4,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("hi"):
            events.append(event)

        start_events = [e for e in events if isinstance(e, SubagentStartEvent)]
        self.assertEqual(len(start_events), 1)
        self.assertEqual(start_events[0].subagent_name, "Explorer")

        usage_events = [e for e in events if isinstance(e, UsageDeltaEvent)]
        prefixed_usage = [
            e for e in usage_events if e.source.startswith("subagent:Explorer:tc_explorer")
        ]
        self.assertGreaterEqual(len(prefixed_usage), 2)

        progress_events = [e for e in events if isinstance(e, SubagentProgressEvent)]
        running_progress = [
            e
            for e in progress_events
            if e.tool_call_id == "tc_explorer" and e.status == "running" and e.tokens > 0
        ]
        self.assertTrue(running_progress)

        completed_progress = [
            e
            for e in progress_events
            if e.tool_call_id == "tc_explorer" and e.status == "completed"
        ]
        self.assertTrue(completed_progress)
        self.assertEqual(completed_progress[-1].tokens, 30)

        stop_events = [e for e in events if isinstance(e, SubagentStopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].status, "completed")

    async def test_serial_task_emits_usage_and_progress_events(self) -> None:
        @tool("Task tool (test)")
        async def Task(subagent_type: str, prompt: str, description: str = "") -> str:
            ctx = get_system_tool_context()
            assert ctx.token_cost is not None
            assert ctx.tool_call_id is not None
            ctx.token_cost.add_usage(
                "subagent-mid",
                ChatInvokeUsage(
                    prompt_tokens=5,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=4,
                    total_tokens=9,
                ),
                level="MID",
                source=f"subagent:{subagent_type}:{ctx.tool_call_id}",
            )
            await asyncio.sleep(0.08)
            return f"ok:{subagent_type}"

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        _build_task_call(
                            tool_call_id="tc_serial",
                            subagent_type="Explorer",
                            description="串行任务",
                        )
                    ]
                ),
                ChatInvokeCompletion(content="done"),
            ]
        )

        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Task,),
                agents=(),
                offload_enabled=False,
                task_parallel_enabled=False,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("hi"):
            events.append(event)

        self.assertTrue(any(isinstance(e, SubagentStartEvent) for e in events))
        self.assertTrue(any(isinstance(e, SubagentStopEvent) for e in events))
        self.assertTrue(
            any(
                isinstance(e, UsageDeltaEvent)
                and e.source.startswith("subagent:Explorer:tc_serial")
                for e in events
            )
        )
        self.assertTrue(
            any(
                isinstance(e, SubagentProgressEvent)
                and e.tool_call_id == "tc_serial"
                and e.status == "completed"
                and e.tokens == 9
                for e in events
            )
        )
