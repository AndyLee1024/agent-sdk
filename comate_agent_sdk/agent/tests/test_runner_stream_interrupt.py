import asyncio
import json
import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import StepCompleteEvent, StopEvent, SubagentStopEvent
from comate_agent_sdk.agent.interrupt import SessionRunController
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.tools import tool


class _FakeChatModel:
    def __init__(self, completions: list[ChatInvokeCompletion], *, delay_seconds: float = 0.0):
        self._completions = completions
        self._idx = 0
        self._delay_seconds = delay_seconds
        self.calls = 0
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        del messages, tools, tool_choice, kwargs
        self.calls += 1
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


class TestRunnerStreamInterrupt(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_before_llm_stops_with_interrupted_reason(self) -> None:
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

        controller = SessionRunController()
        controller.interrupt()
        agent._run_controller = controller

        events = []
        async for event in agent.query_stream("hello"):
            events.append(event)

        stop_events = [e for e in events if isinstance(e, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "interrupted")
        self.assertEqual(llm.calls, 0)

    async def test_interrupt_during_serial_tool_wait_cancels_step(self) -> None:
        started = asyncio.Event()

        @tool("Slow tool for interruption test")
        async def SlowTool(path: str) -> str:
            del path
            started.set()
            await asyncio.sleep(10)
            return "ok"

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        ToolCall(
                            id="tc_slow",
                            function=Function(
                                name="SlowTool",
                                arguments=json.dumps({"path": "/tmp/a.txt"}, ensure_ascii=False),
                            ),
                        )
                    ]
                )
            ]
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
        controller = SessionRunController()
        agent._run_controller = controller

        events: list[object] = []

        async def _collect() -> None:
            async for event in agent.query_stream("run slow tool"):
                events.append(event)

        run_task = asyncio.create_task(_collect())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        controller.interrupt()
        await asyncio.wait_for(run_task, timeout=2.0)

        stop_events = [e for e in events if isinstance(e, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "interrupted")

        cancelled_steps = [
            e for e in events if isinstance(e, StepCompleteEvent) and e.step_id == "tc_slow"
        ]
        self.assertEqual(len(cancelled_steps), 1)
        self.assertEqual(cancelled_steps[0].status, "cancelled")

    async def test_interrupt_during_parallel_task_wait_marks_all_pending_tasks_cancelled(self) -> None:
        started_count = 0
        started_lock = asyncio.Lock()
        all_started = asyncio.Event()

        @tool("Task tool for interruption test")
        async def Task(subagent_type: str, prompt: str, description: str = "") -> str:
            del prompt, description
            nonlocal started_count
            async with started_lock:
                started_count += 1
                if started_count >= 2:
                    all_started.set()
            await asyncio.sleep(10)
            return f"ok:{subagent_type}"

        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(
                    tool_calls=[
                        ToolCall(
                            id="tc_a",
                            function=Function(
                                name="Task",
                                arguments=json.dumps(
                                    {"subagent_type": "A", "prompt": "p", "description": "alpha"},
                                    ensure_ascii=False,
                                ),
                            ),
                        ),
                        ToolCall(
                            id="tc_b",
                            function=Function(
                                name="Task",
                                arguments=json.dumps(
                                    {"subagent_type": "B", "prompt": "p", "description": "beta"},
                                    ensure_ascii=False,
                                ),
                            ),
                        ),
                    ]
                )
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
        controller = SessionRunController()
        agent._run_controller = controller

        events: list[object] = []

        async def _collect() -> None:
            async for event in agent.query_stream("run parallel tasks"):
                events.append(event)

        run_task = asyncio.create_task(_collect())
        await asyncio.wait_for(all_started.wait(), timeout=1.0)
        controller.interrupt()
        await asyncio.wait_for(run_task, timeout=2.0)

        stop_events = [e for e in events if isinstance(e, StopEvent)]
        self.assertEqual(len(stop_events), 1)
        self.assertEqual(stop_events[0].reason, "interrupted")

        cancelled_task_ids = {
            e.tool_call_id
            for e in events
            if isinstance(e, SubagentStopEvent) and e.status == "cancelled"
        }
        self.assertSetEqual(cancelled_task_ids, {"tc_a", "tc_b"})

        cancelled_steps = {
            e.step_id
            for e in events
            if isinstance(e, StepCompleteEvent) and e.status == "cancelled"
        }
        self.assertSetEqual(cancelled_steps, {"tc_a", "tc_b"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
