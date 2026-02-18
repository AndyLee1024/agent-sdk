import unittest

from comate_agent_sdk.context.accounting import ContextTokenAccounting


class TestContextTokenAccounting(unittest.TestCase):
    def test_initial_state(self) -> None:
        acc = ContextTokenAccounting()
        self.assertEqual(acc.last_reported_total_tokens, 0)
        self.assertEqual(acc.ir_total_at_last_report, 0)

    def test_observe_reported_usage_stores_baseline(self) -> None:
        acc = ContextTokenAccounting()
        acc.observe_reported_usage(reported_total_tokens=1500, ir_total=1200)
        self.assertEqual(acc.last_reported_total_tokens, 1500)
        self.assertEqual(acc.ir_total_at_last_report, 1200)

    def test_observe_reported_usage_clamps_negatives(self) -> None:
        acc = ContextTokenAccounting()
        acc.observe_reported_usage(reported_total_tokens=-5, ir_total=-10)
        self.assertEqual(acc.last_reported_total_tokens, 0)
        self.assertEqual(acc.ir_total_at_last_report, 0)

    def test_observe_reported_usage_updates_baseline(self) -> None:
        acc = ContextTokenAccounting()
        acc.observe_reported_usage(reported_total_tokens=1000, ir_total=900)
        acc.observe_reported_usage(reported_total_tokens=2000, ir_total=1800)
        self.assertEqual(acc.last_reported_total_tokens, 2000)
        self.assertEqual(acc.ir_total_at_last_report, 1800)

    def test_reset_baseline_clears_both_fields(self) -> None:
        acc = ContextTokenAccounting()
        acc.observe_reported_usage(reported_total_tokens=1500, ir_total=1200)
        acc.reset_baseline()
        self.assertEqual(acc.last_reported_total_tokens, 0)
        self.assertEqual(acc.ir_total_at_last_report, 0)

    def test_safety_margin_ratio_stored(self) -> None:
        acc = ContextTokenAccounting(safety_margin_ratio=0.15)
        self.assertAlmostEqual(acc.safety_margin_ratio, 0.15)

    def test_safety_margin_ratio_negative_clamped_to_zero(self) -> None:
        acc = ContextTokenAccounting(safety_margin_ratio=-0.5)
        self.assertEqual(acc.safety_margin_ratio, 0.0)

    def test_incremental_estimate_pattern(self) -> None:
        """Verify the consumer-side incremental logic using the baseline fields."""
        acc = ContextTokenAccounting()

        # After first invoke: reported=1500, ir snapshot=1200
        acc.observe_reported_usage(reported_total_tokens=1500, ir_total=1200)

        # IR grows by 300 new tokens (tool result added)
        current_ir = 1500
        ir_delta = max(0, current_ir - acc.ir_total_at_last_report)  # 1500 - 1200 = 300
        estimated = acc.last_reported_total_tokens + ir_delta  # 1500 + 300 = 1800

        self.assertEqual(ir_delta, 300)
        self.assertEqual(estimated, 1800)

    def test_incremental_estimate_no_delta(self) -> None:
        """When IR hasn't grown, estimated equals last_reported."""
        acc = ContextTokenAccounting()
        acc.observe_reported_usage(reported_total_tokens=1000, ir_total=950)

        ir_delta = max(0, 950 - acc.ir_total_at_last_report)  # 0
        estimated = acc.last_reported_total_tokens + ir_delta  # 1000

        self.assertEqual(estimated, 1000)

    def test_incremental_estimate_after_reset_falls_back_to_zero(self) -> None:
        """After reset, last_reported is 0 so caller should use ir_total as fallback."""
        acc = ContextTokenAccounting()
        acc.observe_reported_usage(reported_total_tokens=1500, ir_total=1200)
        acc.reset_baseline()

        self.assertEqual(acc.last_reported_total_tokens, 0)
        # Caller checks: if last_reported == 0 → use context.total_tokens as fallback


if __name__ == "__main__":
    unittest.main(verbosity=2)
