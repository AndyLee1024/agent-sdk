import asyncio
import time
import unittest
from types import SimpleNamespace

from comate_agent_sdk.agent.runner import precheck_and_compact
from comate_agent_sdk.context.compaction import CompactionMetaRecord
from comate_agent_sdk.llm.messages import UserMessage


class _FakeCompactionService:
    def __init__(self, threshold: int, enabled: bool = True) -> None:
        self.config = SimpleNamespace(enabled=enabled)
        self._threshold = threshold
        self.llm = SimpleNamespace(model="compaction-mid")

    async def get_threshold_for_model(self, model: str) -> int:
        return self._threshold


class _FakeTokenCounter:
    async def count_messages_for_model(self, messages, *, llm, timeout_ms: int = 300) -> int:
        return 120


class _FailingContext:
    def __init__(self) -> None:
        self.token_counter = _FakeTokenCounter()
        self.total_tokens = 999
        self.auto_compact_calls = 0

    def lower(self):
        return [UserMessage(content="hello")]

    async def auto_compact(self, policy, current_total_tokens: int | None = None) -> bool:
        self.auto_compact_calls += 1
        policy.meta_records = [
            CompactionMetaRecord(
                phase="rollback",
                tokens_before=120,
                tokens_after=120,
                tool_blocks_kept=5,
                tool_blocks_dropped=0,
                tool_calls_truncated=0,
                tool_results_truncated=0,
                reason="summary_failed_or_empty:empty_summary",
            )
        ]
        return False


class TestCompactionSummaryCooldown(unittest.TestCase):
    def _build_agent(self):
        return SimpleNamespace(
            _compaction_service=_FakeCompactionService(threshold=100, enabled=True),
            _context=_FailingContext(),
            token_count_timeout_ms=300,
            precheck_buffer_ratio=0.12,
            llm=SimpleNamespace(model="gpt-4o", provider="openai"),
            offload_enabled=False,
            _context_fs=None,
            offload_policy=None,
            offload_token_threshold=2000,
            _token_cost=None,
            _effective_level=None,
            emit_compaction_meta_events=False,
        )

    def test_enters_cooldown_after_repeated_summary_failures(self) -> None:
        agent = self._build_agent()

        compacted_1, event_1, _ = asyncio.run(precheck_and_compact(agent))
        compacted_2, event_2, _ = asyncio.run(precheck_and_compact(agent))
        compacted_3, event_3, _ = asyncio.run(precheck_and_compact(agent))

        self.assertFalse(compacted_1)
        self.assertFalse(compacted_2)
        self.assertFalse(compacted_3)
        self.assertIsNotNone(event_1)
        self.assertIsNotNone(event_2)
        self.assertIsNotNone(event_3)
        self.assertEqual(agent._context.auto_compact_calls, 2)
        self.assertGreater(float(getattr(agent, "_summary_compaction_cooldown_until", 0.0)), time.monotonic())

    def test_cooldown_expired_allows_compaction_attempt_again(self) -> None:
        agent = self._build_agent()

        asyncio.run(precheck_and_compact(agent))
        asyncio.run(precheck_and_compact(agent))
        self.assertEqual(agent._context.auto_compact_calls, 2)

        setattr(agent, "_summary_compaction_cooldown_until", time.monotonic() - 0.1)
        asyncio.run(precheck_and_compact(agent))
        self.assertEqual(agent._context.auto_compact_calls, 3)


if __name__ == "__main__":
    unittest.main()
