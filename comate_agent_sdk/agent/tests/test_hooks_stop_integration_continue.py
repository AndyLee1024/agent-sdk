import unittest

from comate_agent_sdk.agent import Agent, AgentConfig
from comate_agent_sdk.agent.events import HiddenUserMessageEvent, StopEvent, TextEvent
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


class TestHooksStopIntegrationContinue(unittest.IsolatedAsyncioTestCase):
    async def test_stop_block_continues_next_iteration_with_hidden_message(self) -> None:
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(content="first"),
                ChatInvokeCompletion(content="second"),
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
        runtime = template.create_runtime()

        block_state = {"count": 0}

        def _stop_hook(_):
            if block_state["count"] == 0:
                block_state["count"] += 1
                return {"decision": "block", "reason": "continue-working"}
            return None

        runtime.register_python_hook(event_name="Stop", callback=_stop_hook)

        events = []
        async for event in runtime.query_stream("hello"):
            events.append(event)

        self.assertEqual(llm._idx, 2)
        self.assertEqual(len([e for e in events if isinstance(e, StopEvent)]), 1)
        self.assertEqual(len([e for e in events if isinstance(e, TextEvent)]), 2)
        hidden_events = [e for e in events if isinstance(e, HiddenUserMessageEvent)]
        self.assertTrue(any("continue-working" in e.content for e in hidden_events))

    async def test_query_non_stream_does_not_accumulate_hidden_queue(self) -> None:
        llm = _FakeChatModel(
            [
                ChatInvokeCompletion(content="first"),
                ChatInvokeCompletion(content="second"),
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
        runtime = template.create_runtime()
        state = {"blocked": False}

        def _stop_hook(_):
            if not state["blocked"]:
                state["blocked"] = True
                return {"decision": "block", "reason": "keep-going"}
            return None

        runtime.register_python_hook(event_name="Stop", callback=_stop_hook)
        text = await runtime.query("hello")
        self.assertEqual(text, "second")
        self.assertEqual(runtime.drain_hidden_user_messages(), [])
