from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.env_utils import read_env_int
from terminal_agent.input_geometry import (
    compute_visual_line_ranges,
    index_for_visual_col,
    visual_col_for_index,
)


class TestInputBehaviorHelpers(unittest.TestCase):
    def test_compute_visual_line_ranges_wraps_ascii(self) -> None:
        text = "abcdefghij"
        ranges = compute_visual_line_ranges(text, max_cols=4)
        self.assertEqual(ranges, [(0, 4), (4, 8), (8, 10)])

    def test_compute_visual_line_ranges_respects_newlines(self) -> None:
        text = "abc\ndef"
        ranges = compute_visual_line_ranges(text, max_cols=4)
        self.assertEqual(ranges, [(0, 3), (4, 7)])

    def test_compute_visual_line_ranges_handles_wide_chars(self) -> None:
        text = "你a你"
        ranges = compute_visual_line_ranges(text, max_cols=3)
        self.assertEqual(ranges, [(0, 2), (2, 3)])

    def test_visual_col_and_index_roundtrip(self) -> None:
        text = "abcdef"
        start, end = 0, len(text)
        max_cols = 10
        for idx in range(start, end + 1):
            col = visual_col_for_index(text, start, end, max_cols, idx)
            roundtrip = index_for_visual_col(text, start, end, max_cols, col)
            self.assertLessEqual(roundtrip, idx)

    def test_read_env_int_defaults_and_validation(self) -> None:
        name = "AGENT_SDK_TEST_ENV_INT"
        os.environ.pop(name, None)
        self.assertEqual(read_env_int(name, 123), 123)

        os.environ[name] = "456"
        self.assertEqual(read_env_int(name, 123), 456)

        os.environ[name] = "not-a-number"
        self.assertEqual(read_env_int(name, 123), 123)

        os.environ[name] = "0"
        self.assertEqual(read_env_int(name, 123), 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
