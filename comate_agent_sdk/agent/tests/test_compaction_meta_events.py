import asyncio
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


class _FakeContext:
    def __init__(self) -> None:
        self.token_counter = _FakeTokenCounter()
        self.total_tokens = 1000

    def lower(self):
        return [UserMessage(content="hello")]

    async def auto_compact(self, policy, current_total_tokens: int | None = None) -> bool:
        policy.meta_records = [
            CompactionMetaRecord(
                phase="selective_start",
                tokens_before=1000,
                tokens_after=1000,
                tool_blocks_kept=5,
                tool_blocks_dropped=3,
                tool_calls_truncated=1,
                tool_results_truncated=2,
                reason="started",
            ),
            CompactionMetaRecord(
                phase="selective_done",
                tokens_before=1000,
                tokens_after=700,
                tool_blocks_kept=5,
                tool_blocks_dropped=3,
                tool_calls_truncated=1,
                tool_results_truncated=2,
                reason="done",
            ),
        ]
        return True


class TestCompactionMetaEvents(unittest.TestCase):
    def _build_agent(self, *, emit_compaction_meta_events: bool):
        return SimpleNamespace(
            _compaction_service=_FakeCompactionService(threshold=100, enabled=True),
            _context=_FakeContext(),
            token_count_timeout_ms=300,
            precheck_buffer_ratio=0.12,
            llm=SimpleNamespace(model="gpt-4o", provider="openai"),
            offload_enabled=False,
            _context_fs=None,
            offload_policy=None,
            offload_token_threshold=2000,
            _token_cost=None,
            _effective_level=None,
            emit_compaction_meta_events=emit_compaction_meta_events,
        )

    def test_compaction_meta_events_disabled_by_default(self) -> None:
        agent = self._build_agent(emit_compaction_meta_events=False)
        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        self.assertTrue(compacted)
        self.assertIsNotNone(event)
        self.assertEqual(meta_events, [])

    def test_compaction_meta_events_emitted_when_enabled(self) -> None:
        agent = self._build_agent(emit_compaction_meta_events=True)
        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        self.assertTrue(compacted)
        self.assertIsNotNone(event)
        self.assertEqual(len(meta_events), 2)
        self.assertEqual(meta_events[0].phase, "selective_start")
        self.assertEqual(meta_events[0].tokens_before, 1000)
        self.assertEqual(meta_events[0].tool_blocks_dropped, 3)
        self.assertEqual(meta_events[1].phase, "selective_done")
        self.assertEqual(meta_events[1].tokens_after, 700)


if __name__ == "__main__":
    unittest.main()
