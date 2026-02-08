import asyncio
import unittest
from types import SimpleNamespace

from comate_agent_sdk.agent.runner import precheck_and_compact
from comate_agent_sdk.llm.messages import UserMessage


class _FakeCompactionService:
    def __init__(self, threshold: int, enabled: bool = True) -> None:
        self.config = SimpleNamespace(enabled=enabled)
        self._threshold = threshold
        self.llm = SimpleNamespace(model="compaction-mid")

    async def get_threshold_for_model(self, model: str) -> int:
        return self._threshold


class _FakeTokenCounter:
    def __init__(self, estimated_tokens: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.calls: list[int] = []

    async def count_messages_for_model(self, messages, *, llm, timeout_ms: int = 300) -> int:
        self.calls.append(timeout_ms)
        return self.estimated_tokens


class _FakeContext:
    def __init__(self, token_counter: _FakeTokenCounter) -> None:
        self.token_counter = token_counter
        self.total_tokens = 999
        self.auto_compact_calls: list[int] = []

    def lower(self):
        return [UserMessage(content="hi")]

    async def auto_compact(self, policy, current_total_tokens: int | None = None) -> bool:
        self.auto_compact_calls.append(int(current_total_tokens or 0))
        return True


class TestRunnerPrecheckBuffer(unittest.TestCase):
    def test_precheck_triggers_with_buffered_tokens(self) -> None:
        token_counter = _FakeTokenCounter(estimated_tokens=90)
        context = _FakeContext(token_counter)
        compaction_service = _FakeCompactionService(threshold=100, enabled=True)

        agent = SimpleNamespace(
            _compaction_service=compaction_service,
            _context=context,
            token_count_timeout_ms=300,
            precheck_buffer_ratio=0.12,
            llm=SimpleNamespace(model="gpt-4o", provider="openai"),
            offload_enabled=False,
            _context_fs=None,
            offload_policy=None,
            offload_token_threshold=2000,
            _token_cost=None,
            _effective_level=None,
        )

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        self.assertTrue(compacted)
        self.assertIsNotNone(event)
        self.assertEqual(meta_events, [])
        self.assertEqual(event.current_tokens, 100)
        self.assertEqual(event.threshold, 100)
        self.assertEqual(event.trigger, "precheck")
        self.assertEqual(context.auto_compact_calls, [100])
        self.assertEqual(token_counter.calls, [300])

    def test_precheck_skips_when_buffered_tokens_below_threshold(self) -> None:
        token_counter = _FakeTokenCounter(estimated_tokens=80)
        context = _FakeContext(token_counter)
        compaction_service = _FakeCompactionService(threshold=100, enabled=True)

        agent = SimpleNamespace(
            _compaction_service=compaction_service,
            _context=context,
            token_count_timeout_ms=300,
            precheck_buffer_ratio=0.12,
            llm=SimpleNamespace(model="gpt-4o", provider="openai"),
            offload_enabled=False,
            _context_fs=None,
            offload_policy=None,
            offload_token_threshold=2000,
            _token_cost=None,
            _effective_level=None,
        )

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        self.assertFalse(compacted)
        self.assertIsNone(event)
        self.assertEqual(meta_events, [])
        self.assertEqual(context.auto_compact_calls, [])


if __name__ == "__main__":
    unittest.main()
