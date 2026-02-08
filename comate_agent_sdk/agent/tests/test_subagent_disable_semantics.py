import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.subagent.models import AgentDefinition
from comate_agent_sdk.subagent.task_tool import create_task_tool
from comate_agent_sdk.tools.registry import ToolRegistry


class _FakeChatModel:
    def __init__(self, content: str = "ok") -> None:
        self.model = "fake:model"
        self._content = content

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        return ChatInvokeCompletion(content=self._content)


class TestSubagentDisableSemantics(unittest.TestCase):
    def test_default_keeps_builtin_subagents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    tools=(),
                    offload_enabled=False,
                    setting_sources=None,
                    project_root=Path(tmp),
                ),
            )

            names = {a.name for a in (template.resolved_agents or ())}

        self.assertIn("Explorer", names)
        self.assertIn("Plan", names)

    def test_agents_empty_tuple_disables_all_subagents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    tools=(),
                    agents=(),
                    offload_enabled=False,
                    setting_sources=None,
                    project_root=Path(tmp),
                ),
            )
            runtime = template.create_runtime()

        self.assertEqual(template.resolved_agents, ())
        self.assertEqual(runtime.agents, [])


class TestTaskToolNoNestedSubagentError(unittest.IsolatedAsyncioTestCase):
    async def test_task_created_subagent_no_longer_hits_nested_agents_guard(self) -> None:
        task_tool = create_task_tool(
            agents=[
                AgentDefinition(
                    name="worker",
                    description="test worker",
                    prompt="You are a worker.",
                    tools=[],
                )
            ],
            parent_tools=[],
            tool_registry=ToolRegistry(),
            parent_llm=_FakeChatModel(content="subagent ok"),  # type: ignore[arg-type]
        )

        result = await task_tool.execute(
            subagent_type="worker",
            prompt="run task",
            description="run worker",
        )

        self.assertEqual(result, "subagent ok")
        self.assertNotIn("不能再定义 agents", result)

