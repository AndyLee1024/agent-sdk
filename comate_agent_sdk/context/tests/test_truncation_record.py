import unittest

from comate_agent_sdk.context.truncation import TruncationRecord


class TestTruncationRecord(unittest.TestCase):
    def test_default_not_truncated(self) -> None:
        record = TruncationRecord()
        self.assertFalse(record.is_truncated)
        self.assertFalse(record.is_formatter_truncated)

    def test_formatter_truncated(self) -> None:
        record = TruncationRecord(
            formatter_truncated=True,
            formatter_reason="line_limit",
            formatter_total_estimate=100,
        )
        self.assertTrue(record.is_truncated)
        self.assertTrue(record.is_formatter_truncated)
        self.assertEqual(record.formatter_reason, "line_limit")
        self.assertEqual(record.formatter_total_estimate, 100)

    def test_compaction_details(self) -> None:
        record = TruncationRecord()
        record.compaction_details.append(
            {
                "field": "tool_result.content",
                "original_tokens": 800,
                "kept_tokens": 200,
            }
        )
        self.assertTrue(record.is_truncated)
        self.assertFalse(record.is_formatter_truncated)
        self.assertEqual(len(record.compaction_details), 1)

    def test_to_dict(self) -> None:
        record = TruncationRecord(
            formatter_truncated=True,
            formatter_reason="line_limit",
            formatter_total_estimate=999,
            formatter_shown_range={"start_line": 1, "end_line": 20},
            compaction_details=[{"field": "tool_result.content"}],
        )
        payload = record.to_dict()
        self.assertEqual(payload["formatter_truncated"], True)
        self.assertEqual(payload["formatter_reason"], "line_limit")
        self.assertEqual(payload["formatter_total_estimate"], 999)
        self.assertEqual(payload["formatter_shown_range"]["start_line"], 1)
        self.assertEqual(len(payload["compaction_details"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
