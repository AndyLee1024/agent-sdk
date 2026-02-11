import asyncio
import unittest
from types import SimpleNamespace

from comate_agent_sdk.context.accounting import ContextTokenAccounting
from comate_agent_sdk.context.ir import ContextIR
from comate_agent_sdk.llm.messages import UserMessage


class _FakeTokenCounter:
    def __init__(self, *, message_tokens: int, tool_tokens: int) -> None:
        self.message_tokens = message_tokens
        self.tool_tokens = tool_tokens
        self.count_messages_calls = 0
        self.count_calls = 0

    async def count_messages_for_model(self, messages, *, llm, timeout_ms: int = 300) -> int:
        self.count_messages_calls += 1
        return self.message_tokens

    def count(self, text: str) -> int:
        self.count_calls += 1
        return self.tool_tokens


class TestContextTokenAccounting(unittest.TestCase):
    def test_estimate_next_step_with_margin(self) -> None:
        context = ContextIR()
        context.add_message(UserMessage(content="hello"))
        fake_counter = _FakeTokenCounter(message_tokens=80, tool_tokens=20)
        context.token_counter = fake_counter  # type: ignore[assignment]

        accounting = ContextTokenAccounting(safety_margin_ratio=0.1)
        llm = SimpleNamespace(provider="openai", model="gpt-4o")
        tool_definition = SimpleNamespace(model_dump=lambda: {"name": "add", "type": "function"})

        estimate = asyncio.run(
            accounting.estimate_next_step(
                context=context,
                llm=llm,
                tool_definitions=[tool_definition],
                timeout_ms=300,
            )
        )

        self.assertEqual(estimate.raw_total_tokens, 100)
        self.assertEqual(estimate.calibrated_tokens, 100)
        self.assertEqual(estimate.buffered_tokens, 110)
        self.assertEqual(fake_counter.count_messages_calls, 1)
        self.assertEqual(fake_counter.count_calls, 1)

    def test_observe_reported_usage_updates_ema_ratio(self) -> None:
        context = ContextIR()
        context.add_message(UserMessage(content="hello"))
        fake_counter = _FakeTokenCounter(message_tokens=100, tool_tokens=0)
        context.token_counter = fake_counter  # type: ignore[assignment]

        accounting = ContextTokenAccounting(safety_margin_ratio=0.12, ema_alpha=0.2)
        llm = SimpleNamespace(provider="openai", model="gpt-4o")

        accounting.observe_reported_usage(
            llm=llm,
            reported_total_tokens=150,
            estimated_raw_tokens=100,
        )

        model_key = accounting.build_model_key(llm)
        self.assertAlmostEqual(accounting.get_ratio(model_key), 1.1, places=6)

        estimate = asyncio.run(
            accounting.estimate_next_step(
                context=context,
                llm=llm,
                tool_definitions=None,
                timeout_ms=300,
            )
        )

        self.assertEqual(estimate.raw_total_tokens, 100)
        self.assertEqual(estimate.calibrated_tokens, 110)
        self.assertEqual(estimate.buffered_tokens, 123)


if __name__ == "__main__":
    unittest.main()
