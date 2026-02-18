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


class _FakeContext:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens
        self.auto_compact_calls: list[int] = []

    def lower(self):
        return [UserMessage(content="hi")]

    async def auto_compact(self, policy, current_total_tokens: int | None = None) -> bool:
        self.auto_compact_calls.append(int(current_total_tokens or 0))
        return True


class _FakeTokenAccounting:
    """Minimal fake that exposes the baseline properties used by incremental estimation."""

    def __init__(self, last_reported: int = 0, ir_total_at_last_report: int = 0) -> None:
        self.last_reported_total_tokens = last_reported
        self.ir_total_at_last_report = ir_total_at_last_report
        self.reset_calls = 0

    def reset_baseline(self) -> None:
        self.reset_calls += 1


def _make_agent(
    context: _FakeContext,
    compaction_service: _FakeCompactionService,
    token_accounting: _FakeTokenAccounting | None = None,
    precheck_buffer_ratio: float = 0.12,
) -> SimpleNamespace:
    agent = SimpleNamespace(
        _compaction_service=compaction_service,
        _context=context,
        _token_accounting=token_accounting,
        precheck_buffer_ratio=precheck_buffer_ratio,
        llm=SimpleNamespace(model="gpt-4o", provider="openai"),
        offload_enabled=False,
        _context_fs=None,
        offload_policy=None,
        offload_token_threshold=2000,
        _token_cost=None,
        _effective_level=None,
    )
    return agent


class TestRunnerPrecheckBuffer(unittest.TestCase):
    # ── fallback path (no token_accounting / last_reported == 0) ─────────────

    def test_precheck_triggers_with_buffered_tokens_ir_fallback(self) -> None:
        """Without token_accounting, precheck uses context.total_tokens as base."""
        context = _FakeContext(total_tokens=90)
        compaction_service = _FakeCompactionService(threshold=100, enabled=True)
        agent = _make_agent(context, compaction_service)

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        # 90 * 1.12 = 100.8 → int = 100 ≥ 100 → trigger
        self.assertTrue(compacted)
        self.assertIsNotNone(event)
        self.assertEqual(meta_events, [])
        self.assertEqual(event.current_tokens, 100)
        self.assertEqual(event.threshold, 100)
        self.assertEqual(event.trigger, "precheck")
        self.assertEqual(context.auto_compact_calls, [100])

    def test_precheck_skips_when_buffered_tokens_below_threshold(self) -> None:
        """Without token_accounting, precheck skips when total_tokens * buffer < threshold."""
        context = _FakeContext(total_tokens=80)
        compaction_service = _FakeCompactionService(threshold=100, enabled=True)
        agent = _make_agent(context, compaction_service)

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        # 80 * 1.12 = 89.6 → int = 89 < 100 → skip
        self.assertFalse(compacted)
        self.assertIsNone(event)
        self.assertEqual(meta_events, [])
        self.assertEqual(context.auto_compact_calls, [])

    # ── incremental path (token_accounting with last_reported > 0) ───────────

    def test_precheck_uses_incremental_estimation_when_reported_available(self) -> None:
        """With a known last_reported baseline, precheck uses last_reported + IR delta."""
        # last_reported=900, ir_snapshot=850, current_ir=920 → delta=70 → est=970
        # 970 * 1.12 = 1086.4 → 1086 ≥ threshold=1000 → trigger
        context = _FakeContext(total_tokens=920)
        compaction_service = _FakeCompactionService(threshold=1000, enabled=True)
        token_accounting = _FakeTokenAccounting(last_reported=900, ir_total_at_last_report=850)
        agent = _make_agent(context, compaction_service, token_accounting)

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        expected_estimated = 900 + (920 - 850)  # 970
        expected_buffered = int(970 * 1.12)  # 1086
        self.assertTrue(compacted)
        self.assertIsNotNone(event)
        self.assertEqual(event.current_tokens, expected_buffered)
        self.assertEqual(context.auto_compact_calls, [expected_buffered])

    def test_precheck_skips_when_incremental_estimate_below_threshold(self) -> None:
        """Incremental estimate below threshold: no compaction."""
        # last_reported=500, ir_snapshot=480, current_ir=490 → delta=10 → est=510
        # 510 * 1.12 = 571.2 → 571 < 1000 → skip
        context = _FakeContext(total_tokens=490)
        compaction_service = _FakeCompactionService(threshold=1000, enabled=True)
        token_accounting = _FakeTokenAccounting(last_reported=500, ir_total_at_last_report=480)
        agent = _make_agent(context, compaction_service, token_accounting)

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        self.assertFalse(compacted)
        self.assertIsNone(event)
        self.assertEqual(context.auto_compact_calls, [])

    def test_precheck_falls_back_to_ir_when_last_reported_is_zero(self) -> None:
        """When token_accounting exists but last_reported=0, use context.total_tokens."""
        # total_tokens=920, last_reported=0 → fallback → 920 * 1.12 = 1030 ≥ 1000
        context = _FakeContext(total_tokens=920)
        compaction_service = _FakeCompactionService(threshold=1000, enabled=True)
        token_accounting = _FakeTokenAccounting(last_reported=0, ir_total_at_last_report=0)
        agent = _make_agent(context, compaction_service, token_accounting)

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        expected_buffered = int(920 * 1.12)  # 1030
        self.assertTrue(compacted)
        self.assertEqual(event.current_tokens, expected_buffered)

    def test_precheck_clamps_negative_ir_delta(self) -> None:
        """When IR shrinks (e.g. messages removed), delta is clamped to 0."""
        # current_ir=800, ir_snapshot=900 → delta=max(0,-100)=0 → est=last_reported=1000
        # 1000 * 1.12 = 1120 ≥ 1000 → trigger
        context = _FakeContext(total_tokens=800)
        compaction_service = _FakeCompactionService(threshold=1000, enabled=True)
        token_accounting = _FakeTokenAccounting(last_reported=1000, ir_total_at_last_report=900)
        agent = _make_agent(context, compaction_service, token_accounting)

        compacted, event, meta_events = asyncio.run(precheck_and_compact(agent))

        expected_buffered = int(1000 * 1.12)  # 1120
        self.assertTrue(compacted)
        self.assertEqual(event.current_tokens, expected_buffered)

    def test_reset_baseline_called_after_compaction(self) -> None:
        """reset_baseline() is called on token_accounting after compaction fires."""
        context = _FakeContext(total_tokens=90)
        compaction_service = _FakeCompactionService(threshold=100, enabled=True)
        token_accounting = _FakeTokenAccounting(last_reported=0, ir_total_at_last_report=0)
        agent = _make_agent(context, compaction_service, token_accounting)

        compacted, _, _ = asyncio.run(precheck_and_compact(agent))

        self.assertTrue(compacted)
        self.assertEqual(token_accounting.reset_calls, 1)

    def test_no_reset_when_no_compaction(self) -> None:
        """reset_baseline() is NOT called when compaction is skipped."""
        context = _FakeContext(total_tokens=50)
        compaction_service = _FakeCompactionService(threshold=1000, enabled=True)
        token_accounting = _FakeTokenAccounting(last_reported=0, ir_total_at_last_report=0)
        agent = _make_agent(context, compaction_service, token_accounting)

        compacted, _, _ = asyncio.run(precheck_and_compact(agent))

        self.assertFalse(compacted)
        self.assertEqual(token_accounting.reset_calls, 0)


if __name__ == "__main__":
    unittest.main()
