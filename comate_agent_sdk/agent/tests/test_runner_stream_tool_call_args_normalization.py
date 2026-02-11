import json
import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import ToolCallEvent
from comate_agent_sdk.llm.messages import Function, ToolCall
from comate_agent_sdk.llm.views import ChatInvokeCompletion
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
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


class TestRunnerStreamToolCallArgsNormalization(unittest.IsolatedAsyncioTestCase):
    async def test_serial_tool_call_event_args_are_schema_normalized(self) -> None:
        @tool("Echo todos")
        async def EchoTodos(todos: list[dict[str, str]]) -> str:
            return f"items={len(todos)}"

        tool_call = ToolCall(
            id="tc_echo_1",
            function=Function(
                name="EchoTodos",
                arguments=json.dumps(
                    {
                        "todos": '[{"id":"1","content":"Read docs","status":"pending","priority":"high"}]'
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
                tools=(EchoTodos,),
                agents=(),
                offload_enabled=False,
                task_parallel_enabled=False,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("normalize args"):
            events.append(event)

        tool_events = [
            event for event in events if isinstance(event, ToolCallEvent) and event.tool == "EchoTodos"
        ]
        self.assertEqual(len(tool_events), 1)
        args = tool_events[0].args
        self.assertIn("todos", args)
        self.assertIsInstance(args["todos"], list)
        self.assertEqual(args["todos"][0]["id"], "1")
        self.assertEqual(args["todos"][0]["status"], "pending")

    async def test_parallel_task_tool_call_event_args_are_schema_normalized(self) -> None:
        @tool("Task tool (test)")
        async def Task(
            subagent_type: str,
            prompt: str,
            description: str = "",
            tags: list[str] | None = None,
        ) -> str:
            _ = tags
            return f"ok:{subagent_type}"

        tool_calls = [
            ToolCall(
                id="tc_task_1",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {
                            "subagent_type": "alpha",
                            "prompt": "p1",
                            "description": "A",
                            "tags": '["one","two"]',
                        },
                        ensure_ascii=False,
                    ),
                ),
            ),
            ToolCall(
                id="tc_task_2",
                function=Function(
                    name="Task",
                    arguments=json.dumps(
                        {
                            "subagent_type": "beta",
                            "prompt": "p2",
                            "description": "B",
                            "tags": '["three"]',
                        },
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

        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(Task,),
                agents=(),
                offload_enabled=False,
                task_parallel_enabled=True,
                task_parallel_max_concurrency=2,
            ),
        )
        agent = template.create_runtime()

        events = []
        async for event in agent.query_stream("parallel normalize args"):
            events.append(event)

        task_events = [event for event in events if isinstance(event, ToolCallEvent) and event.tool == "Task"]
        self.assertEqual(len(task_events), 2)

        args_by_call_id = {event.tool_call_id: event.args for event in task_events}
        self.assertIsInstance(args_by_call_id["tc_task_1"]["tags"], list)
        self.assertIsInstance(args_by_call_id["tc_task_2"]["tags"], list)
        self.assertEqual(args_by_call_id["tc_task_1"]["tags"], ["one", "two"])
        self.assertEqual(args_by_call_id["tc_task_2"]["tags"], ["three"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
