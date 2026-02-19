import asyncio
import unittest
from types import SimpleNamespace

from comate_agent_sdk.agent.runner import precheck_and_compact
from comate_agent_sdk.context.compaction import CompactionMetaRecord
from comate_agent_sdk.context.usage_tracker import ContextUsageTracker
from comate_agent_sdk.llm.messages import UserMessage


class _FakeCompactionService:
    def __init__(self, threshold: int, enabled: bool = True) -> None:
        self.config = SimpleNamespace(enabled=enabled)
        self._threshold = threshold
        self.llm = SimpleNamespace(model="compaction-mid")

    async def get_threshold_for_model(self, model: str) -> int:
        return self._threshold


class _FakeContext:
    def __init__(self) -> None:
        self.total_tokens = 1000
        self.purge_calls = 0

    def lower(self):
        return [UserMessage(content="hello")]

    def purge_system_reminders(self, *, include_persistent: bool = True) -> int:
        assert include_persistent is True
        self.purge_calls += 1
        return 0

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
        # context_window=1000, threshold_ratio=0.85 → threshold=850
        # estimate_precheck(1000) = 1000+500=1500 >= 850 → triggers compaction
        tracker = ContextUsageTracker(context_window=1000, threshold_ratio=0.85)
        return SimpleNamespace(
            _compaction_service=_FakeCompactionService(threshold=100, enabled=True),
            _context=_FakeContext(),
            _context_usage_tracker=tracker,
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
        self.assertEqual(agent._context.purge_calls, 1)

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
        self.assertEqual(agent._context.purge_calls, 1)


if __name__ == "__main__":
    unittest.main()
