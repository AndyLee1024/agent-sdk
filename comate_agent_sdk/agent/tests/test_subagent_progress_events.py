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
from comate_agent_sdk.subagent.models import AgentDefinition
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


class _StreamingTaskFakeChatModel:
    def __init__(self, task_calls: list[ToolCall]):
        self._task_calls = task_calls
        self._parent_invoke_count = 0
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    @staticmethod
    def _tool_names(tools) -> set[str]:  # type: ignore[no-untyped-def]
        names: set[str] = set()
        if not tools:
            return names
        for tool_def in tools:
            name = getattr(tool_def, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    @staticmethod
    def _resolve_subagent_tag(messages) -> str:  # type: ignore[no-untyped-def]
        for message in reversed(messages):
            if getattr(message, "role", "") != "user":
                continue
            text = getattr(message, "text", None)
            if not isinstance(text, str) or not text:
                content = getattr(message, "content", "")
                text = content if isinstance(content, str) else ""
            lowered = text.lower()
            if "beta" in lowered:
                return "beta"
            if "alpha" in lowered:
                return "alpha"
        return "alpha"

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        del tool_choice, kwargs
        tool_names = self._tool_names(tools)

        if "Task" in tool_names:
            if self._parent_invoke_count == 0:
                self._parent_invoke_count += 1
                return ChatInvokeCompletion(tool_calls=self._task_calls)
            if self._parent_invoke_count == 1:
                self._parent_invoke_count += 1
                return ChatInvokeCompletion(content="done")
            raise AssertionError(f"Unexpected parent ainvoke call #{self._parent_invoke_count + 1}")

        if "EmitUsage" in tool_names:
            has_tool_message = any(getattr(message, "role", "") == "tool" for message in messages)
            tag = self._resolve_subagent_tag(messages)
            if not has_tool_message:
                return ChatInvokeCompletion(
                    tool_calls=[
                        ToolCall(
                            id=f"emit_{tag}",
                            function=Function(
                                name="EmitUsage",
                                arguments=json.dumps({"tag": tag}, ensure_ascii=False),
                            ),
                        )
                    ]
                )
            return ChatInvokeCompletion(content=f"subagent done:{tag}")

        raise AssertionError(f"Unexpected tool set in ainvoke: {sorted(tool_names)}")


def _build_task_call(
    *,
    tool_call_id: str,
    subagent_type: str,
    description: str,
    prompt: str = "p",
) -> ToolCall:
    return ToolCall(
        id=tool_call_id,
        function=Function(
            name="Task",
            arguments=json.dumps(
                {
                    "subagent_type": subagent_type,
                    "prompt": prompt,
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

    async def test_parallel_streaming_task_progress_tokens_ignore_foreign_usage(self) -> None:
        @tool("Emit deterministic token usage for streaming Task tests")
        async def EmitUsage(tag: str) -> str:
            ctx = get_system_tool_context()
            assert ctx.token_cost is not None
            assert ctx.subagent_source_prefix is not None

            own_total = 11 if tag == "alpha" else 13
            ctx.token_cost.add_usage(
                "subagent-own",
                ChatInvokeUsage(
                    prompt_tokens=own_total,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=0,
                    total_tokens=own_total,
                ),
                level="MID",
                source=ctx.subagent_source_prefix,
            )
            # Noise that must not be counted into current subagent progress.
            ctx.token_cost.add_usage(
                "subagent-noise",
                ChatInvokeUsage(
                    prompt_tokens=97,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=0,
                    total_tokens=97,
                ),
                level="LOW",
                source="subagent:noise:tc_noise",
            )
            await asyncio.sleep(0.05 if tag == "alpha" else 0.02)
            return f"ok:{tag}"

        tool_calls = [
            _build_task_call(
                tool_call_id="tc_alpha",
                subagent_type="alpha",
                description="alpha task",
                prompt="run alpha",
            ),
            _build_task_call(
                tool_call_id="tc_beta",
                subagent_type="beta",
                description="beta task",
                prompt="run beta",
            ),
        ]
        llm = _StreamingTaskFakeChatModel(task_calls=tool_calls)
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(EmitUsage,),
                agents=(
                    AgentDefinition(
                        name="alpha",
                        description="alpha",
                        prompt="You are alpha",
                        tools=["EmitUsage"],
                    ),
                    AgentDefinition(
                        name="beta",
                        description="beta",
                        prompt="You are beta",
                        tools=["EmitUsage"],
                    ),
                ),
                offload_enabled=False,
                task_parallel_enabled=True,
                task_parallel_max_concurrency=4,
                use_streaming_task=True,
                max_iterations=8,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("hi"):
            events.append(event)

        completed_progress = {
            event.tool_call_id: event.tokens
            for event in events
            if isinstance(event, SubagentProgressEvent) and event.status == "completed"
        }
        self.assertEqual(completed_progress.get("tc_alpha"), 11)
        self.assertEqual(completed_progress.get("tc_beta"), 13)
        self.assertTrue(
            any(
                isinstance(event, UsageDeltaEvent) and event.source == "subagent:noise:tc_noise"
                for event in events
            )
        )

    async def test_serial_streaming_task_progress_tokens_ignore_foreign_usage(self) -> None:
        @tool("Emit deterministic token usage for streaming Task serial tests")
        async def EmitUsage(tag: str) -> str:
            ctx = get_system_tool_context()
            assert ctx.token_cost is not None
            assert ctx.subagent_source_prefix is not None
            assert tag == "alpha"

            ctx.token_cost.add_usage(
                "subagent-own",
                ChatInvokeUsage(
                    prompt_tokens=17,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=0,
                    total_tokens=17,
                ),
                level="MID",
                source=ctx.subagent_source_prefix,
            )
            ctx.token_cost.add_usage(
                "subagent-noise",
                ChatInvokeUsage(
                    prompt_tokens=91,
                    prompt_cached_tokens=None,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=0,
                    total_tokens=91,
                ),
                level="LOW",
                source="subagent:noise:tc_noise",
            )
            await asyncio.sleep(0.02)
            return "ok:alpha"

        llm = _StreamingTaskFakeChatModel(
            task_calls=[
                _build_task_call(
                    tool_call_id="tc_alpha",
                    subagent_type="alpha",
                    description="alpha serial task",
                    prompt="run alpha",
                )
            ]
        )
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(EmitUsage,),
                agents=(
                    AgentDefinition(
                        name="alpha",
                        description="alpha",
                        prompt="You are alpha",
                        tools=["EmitUsage"],
                    ),
                ),
                offload_enabled=False,
                task_parallel_enabled=False,
                use_streaming_task=True,
                max_iterations=8,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("hi"):
            events.append(event)

        completed_progress = [
            event
            for event in events
            if isinstance(event, SubagentProgressEvent)
            and event.tool_call_id == "tc_alpha"
            and event.status == "completed"
        ]
        self.assertTrue(completed_progress)
        self.assertEqual(completed_progress[-1].tokens, 17)

    async def test_streaming_task_progress_excludes_cached_prompt_tokens(self) -> None:
        @tool("Emit usage with high prompt cache hit")
        async def EmitUsage(tag: str) -> str:
            ctx = get_system_tool_context()
            assert ctx.token_cost is not None
            assert ctx.subagent_source_prefix is not None
            assert tag == "alpha"

            ctx.token_cost.add_usage(
                "subagent-cached",
                ChatInvokeUsage(
                    prompt_tokens=100,
                    prompt_cached_tokens=90,
                    prompt_cache_creation_tokens=None,
                    prompt_image_tokens=None,
                    completion_tokens=5,
                    total_tokens=105,
                ),
                level="MID",
                source=ctx.subagent_source_prefix,
            )
            await asyncio.sleep(0.01)
            return "ok:alpha"

        llm = _StreamingTaskFakeChatModel(
            task_calls=[
                _build_task_call(
                    tool_call_id="tc_alpha_cached",
                    subagent_type="alpha",
                    description="alpha cached task",
                    prompt="run alpha",
                )
            ]
        )
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(EmitUsage,),
                agents=(
                    AgentDefinition(
                        name="alpha",
                        description="alpha",
                        prompt="You are alpha",
                        tools=["EmitUsage"],
                    ),
                ),
                offload_enabled=False,
                task_parallel_enabled=False,
                use_streaming_task=True,
                max_iterations=8,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("hi"):
            events.append(event)

        completed_progress = [
            event
            for event in events
            if isinstance(event, SubagentProgressEvent)
            and event.tool_call_id == "tc_alpha_cached"
            and event.status == "completed"
        ]
        self.assertTrue(completed_progress)
        # expected effective tokens: prompt_new(100-90) + completion(5) = 15
        self.assertEqual(completed_progress[-1].tokens, 15)
