import asyncio
import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk import Agent
from comate_agent_sdk.agent import AgentConfig
from comate_agent_sdk.agent.events import StopEvent
from comate_agent_sdk.llm.views import ChatInvokeCompletion


class _FakeChatModel:
    def __init__(self, completions: list[ChatInvokeCompletion], *, delay_seconds: float = 0.0):
        self._completions = completions
        self._idx = 0
        self._delay_seconds = delay_seconds
        self.model = "fake:model"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def name(self) -> str:
        return self.model

    async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
        del messages, tools, tool_choice, kwargs
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        if self._idx >= len(self._completions):
            raise AssertionError(f"Unexpected ainvoke call #{self._idx + 1}")
        completion = self._completions[self._idx]
        self._idx += 1
        return completion


class TestChatSessionInterrupt(unittest.IsolatedAsyncioTestCase):
    async def test_chat_session_run_controller_interrupts_and_resets_binding(self) -> None:
        llm = _FakeChatModel([ChatInvokeCompletion(content="done")], delay_seconds=0.5)
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
            ),
        )

        with tempfile.TemporaryDirectory() as td:
            session = template.chat(storage_root=Path(td) / "s1")
            events: list[object] = []

            async def _collect() -> None:
                async for event in session.query_stream("hello"):
                    events.append(event)

            run_task = asyncio.create_task(_collect())
            await asyncio.sleep(0.05)
            session.run_controller.interrupt()
            await asyncio.wait_for(run_task, timeout=2.0)

            stop_events = [e for e in events if isinstance(e, StopEvent)]
            self.assertEqual(len(stop_events), 1)
            self.assertEqual(stop_events[0].reason, "interrupted")

            self.assertFalse(session.run_controller.is_interrupted)
            self.assertIsNone(session._agent._run_controller)


if __name__ == "__main__":
    unittest.main(verbosity=2)
