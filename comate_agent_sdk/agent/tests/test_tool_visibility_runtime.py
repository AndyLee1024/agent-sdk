import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.context.items import ItemType
from comate_agent_sdk.llm.views import ChatInvokeCompletion
from comate_agent_sdk.system_tools.tools import AskUserQuestion
from comate_agent_sdk.tools.decorator import tool


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


class TestToolVisibilityRuntime(unittest.TestCase):
    def test_main_agent_exposes_task_and_ask_user_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    offload_enabled=False,
                    setting_sources=None,
                    project_root=Path(tmp),
                ),
            )
            runtime = template.create_runtime()

        strategy_item = runtime._context.header.find_one_by_type(ItemType.TOOL_STRATEGY)
        self.assertIsNotNone(strategy_item)
        content = strategy_item.content_text if strategy_item is not None else ""
        self.assertIn("- **Task**:", content)
        self.assertIn("- **AskUserQuestion**:", content)

        tool_names = {d.name for d in runtime.tool_definitions}
        self.assertIn("Task", tool_names)
        self.assertIn("AskUserQuestion", tool_names)

    def test_subagent_hides_task_and_ask_user_question(self) -> None:
        @tool("Custom task", name="Task", usage_rules="custom task policy")
        async def custom_task_tool(prompt: str) -> str:
            return prompt

        with tempfile.TemporaryDirectory() as tmp:
            template = Agent(
                llm=_FakeChatModel(),  # type: ignore[arg-type]
                config=AgentConfig(
                    tools=(custom_task_tool, AskUserQuestion),
                    agents=(),
                    offload_enabled=False,
                    setting_sources=None,
                    project_root=Path(tmp),
                ),
            )
            runtime = template.create_runtime(is_subagent=True, name="worker")

        runtime_tool_names = {t.name for t in runtime.options.tools}
        self.assertNotIn("Task", runtime_tool_names)
        self.assertNotIn("AskUserQuestion", runtime_tool_names)

        schema_tool_names = {d.name for d in runtime.tool_definitions}
        self.assertNotIn("Task", schema_tool_names)
        self.assertNotIn("AskUserQuestion", schema_tool_names)

        strategy_item = runtime._context.header.find_one_by_type(ItemType.TOOL_STRATEGY)
        self.assertIsNotNone(strategy_item)
        content = strategy_item.content_text if strategy_item is not None else ""
        self.assertNotIn("- **Task**:", content)
        self.assertNotIn("- **AskUserQuestion**:", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
