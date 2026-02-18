"""单元测试: ContextUsageTracker"""

import unittest

from comate_agent_sdk.context.usage_tracker import ContextUsageTracker


class TestContextUsageTrackerBasic(unittest.TestCase):
    def test_initial_state(self) -> None:
        tracker = ContextUsageTracker(context_window=100_000, threshold_ratio=0.85)
        self.assertEqual(tracker.context_usage, 0)
        self.assertEqual(tracker.threshold, 85_000)
        self.assertEqual(tracker.remaining_tokens, 100_000)
        self.assertAlmostEqual(tracker.utilization_ratio, 0.0)

    def test_threshold_calculation(self) -> None:
        tracker = ContextUsageTracker(context_window=128_000, threshold_ratio=0.85)
        self.assertEqual(tracker.threshold, int(128_000 * 0.85))

    def test_observe_response_updates_context_usage(self) -> None:
        tracker = ContextUsageTracker(context_window=128_000, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=50_000, ir_total=48_000)

        self.assertEqual(tracker.context_usage, 50_000)
        self.assertEqual(tracker._ir_total_at_last_report, 48_000)
        self.assertAlmostEqual(tracker.utilization_ratio, 50_000 / 128_000)
        self.assertEqual(tracker.remaining_tokens, 78_000)

    def test_observe_response_clamps_negative(self) -> None:
        tracker = ContextUsageTracker()
        tracker.observe_response(total_tokens=-100, ir_total=-50)
        self.assertEqual(tracker.context_usage, 0)
        self.assertEqual(tracker._ir_total_at_last_report, 0)

    def test_reset_after_compaction(self) -> None:
        tracker = ContextUsageTracker(context_window=128_000, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=100_000, ir_total=95_000)
        tracker.reset_after_compaction()

        self.assertEqual(tracker.context_usage, 0)
        self.assertEqual(tracker._ir_total_at_last_report, 0)


class TestShouldCompactPostResponse(unittest.TestCase):
    def test_no_compact_when_context_usage_is_zero(self) -> None:
        tracker = ContextUsageTracker(context_window=100, threshold_ratio=0.85)
        # threshold=85, but context_usage=0 → no compact
        self.assertFalse(tracker.should_compact_post_response())

    def test_compact_when_at_threshold(self) -> None:
        tracker = ContextUsageTracker(context_window=100, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=85, ir_total=80)
        self.assertTrue(tracker.should_compact_post_response())

    def test_compact_when_above_threshold(self) -> None:
        tracker = ContextUsageTracker(context_window=100, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=90, ir_total=85)
        self.assertTrue(tracker.should_compact_post_response())

    def test_no_compact_when_below_threshold(self) -> None:
        tracker = ContextUsageTracker(context_window=100, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=84, ir_total=80)
        self.assertFalse(tracker.should_compact_post_response())


class TestShouldCompactPrecheck(unittest.TestCase):
    def test_fallback_to_ir_total_when_no_report(self) -> None:
        """首轮（context_usage=0）: 回退到 ir_total + PRECHECK_BUFFER。"""
        tracker = ContextUsageTracker(context_window=1000, threshold_ratio=0.85)
        # threshold = 850
        # estimate = 0 + 400 + 500 = 900 >= 850 → True
        self.assertTrue(tracker.should_compact_precheck(ir_total=400))

    def test_fallback_below_threshold(self) -> None:
        tracker = ContextUsageTracker(context_window=1000, threshold_ratio=0.85)
        # threshold = 850
        # estimate = 0 + 300 + 500 = 800 < 850 → False
        self.assertFalse(tracker.should_compact_precheck(ir_total=300))

    def test_incremental_with_ir_delta(self) -> None:
        """有 API 报告基线时：context_usage + ir_delta + buffer。"""
        tracker = ContextUsageTracker(context_window=1000, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=700, ir_total=650)
        # ir_delta = max(0, 750 - 650) = 100
        # estimate = 700 + 100 + 500 = 1300 >= 850 → True
        self.assertTrue(tracker.should_compact_precheck(ir_total=750))

    def test_incremental_below_threshold(self) -> None:
        tracker = ContextUsageTracker(context_window=1500, threshold_ratio=0.85)
        # threshold = 1275
        tracker.observe_response(total_tokens=500, ir_total=490)
        # ir_delta = max(0, 495 - 490) = 5
        # estimate = 500 + 5 + 500 = 1005 < 1275 → False
        self.assertFalse(tracker.should_compact_precheck(ir_total=495))

    def test_ir_delta_clamped_to_zero_when_ir_shrinks(self) -> None:
        """IR 缩小（如 compaction 后）时，delta 取 max(0, ...) 防止负值。"""
        tracker = ContextUsageTracker(context_window=1000, threshold_ratio=0.85)
        tracker.observe_response(total_tokens=800, ir_total=900)
        # ir_total=800 < ir_snapshot=900 → delta=0
        # estimate = 800 + 0 + 500 = 1300 >= 850 → True
        self.assertTrue(tracker.should_compact_precheck(ir_total=800))


class TestEstimatePrecheck(unittest.TestCase):
    def test_fallback_path_when_no_report(self) -> None:
        tracker = ContextUsageTracker()
        estimate = tracker.estimate_precheck(ir_total=1000)
        self.assertEqual(estimate, 1000 + 500)  # ir_total + PRECHECK_BUFFER

    def test_incremental_path(self) -> None:
        tracker = ContextUsageTracker()
        tracker.observe_response(total_tokens=900, ir_total=850)
        # ir_delta = max(0, 950 - 850) = 100
        # estimate = 900 + 100 + 500 = 1500
        estimate = tracker.estimate_precheck(ir_total=950)
        self.assertEqual(estimate, 1500)

    def test_incremental_path_no_new_tokens(self) -> None:
        tracker = ContextUsageTracker()
        tracker.observe_response(total_tokens=900, ir_total=850)
        # ir_delta = max(0, 850 - 850) = 0
        # estimate = 900 + 0 + 500 = 1400
        estimate = tracker.estimate_precheck(ir_total=850)
        self.assertEqual(estimate, 1400)

    def test_precheck_buffer_is_500(self) -> None:
        tracker = ContextUsageTracker()
        self.assertEqual(tracker.PRECHECK_BUFFER, 500)


if __name__ == "__main__":
    unittest.main()
