import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.core import AgentRuntime, AgentTemplate
from comate_agent_sdk.llm.views import ChatInvokeCompletion


class _FakeChatModel:
    def __init__(self) -> None:
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        return ChatInvokeCompletion(content="ok")


class TestRuntimeOptionsHardCut(unittest.TestCase):
    def test_runtime_uses_options_as_single_source(self) -> None:
        template = Agent(
            llm=_FakeChatModel(),  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
                max_iterations=7,
            ),
        )
        runtime = template.create_runtime()

        self.assertEqual(runtime.options.max_iterations, 7)
        with self.assertRaises(AttributeError):
            _ = runtime.max_iterations

    def test_core_import_reexport_still_available(self) -> None:
        self.assertTrue(issubclass(AgentRuntime, object))
        self.assertTrue(issubclass(AgentTemplate, object))


if __name__ == "__main__":
    unittest.main(verbosity=2)
