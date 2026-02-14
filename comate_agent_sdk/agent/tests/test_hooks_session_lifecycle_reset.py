import tempfile
import unittest
from pathlib import Path

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.chat_session import ChatSession
from comate_agent_sdk.llm.views import ChatInvokeCompletion


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
        result = self._completions[self._idx]
        self._idx += 1
        return result


class TestHooksSessionLifecycleReset(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_reuse_triggers_session_hooks_again_after_close(self) -> None:
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(content="s1"),
                ChatInvokeCompletion(content="s2"),
            ]
        )
        template = Agent(
            llm=llm,  # type: ignore[arg-type]
            config=AgentConfig(
                tools=(),
                agents=(),
                offload_enabled=False,
            ),
        )
        runtime = template.create_runtime(session_id="session-reuse")
        counters = {"start": 0, "end": 0}
        runtime.register_python_hook(
            event_name="SessionStart",
            callback=lambda _: counters.__setitem__("start", counters["start"] + 1),
        )
        runtime.register_python_hook(
            event_name="SessionEnd",
            callback=lambda _: counters.__setitem__("end", counters["end"] + 1),
        )

        with tempfile.TemporaryDirectory() as tmp:
            s1_root = Path(tmp) / "s1"
            s2_root = Path(tmp) / "s2"
            session1 = ChatSession(
                template,
                runtime=runtime,
                session_id=runtime.session_id,
                storage_root=s1_root,
            )
            async for _ in session1.query_stream("first"):
                pass
            await session1.close()

            session2 = ChatSession(
                template,
                runtime=runtime,
                session_id=runtime.session_id,
                storage_root=s2_root,
            )
            async for _ in session2.query_stream("second"):
                pass
            await session2.close()

        self.assertEqual(counters["start"], 2)
        self.assertEqual(counters["end"], 2)
