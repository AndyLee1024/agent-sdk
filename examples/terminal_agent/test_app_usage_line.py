from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.app import _format_exit_usage_line


class TestAppUsageLine(unittest.TestCase):
    def test_format_exit_usage_line_uses_uncached_input_for_total(self) -> None:
        usage = SimpleNamespace(
            total_prompt_tokens=258_784,
            total_prompt_cached_tokens=13_762,
            total_completion_tokens=37_492,
            total_reasoning_tokens=15_587,
        )

        line = _format_exit_usage_line(usage)

        self.assertEqual(
            line,
            (
                "Token usage: total=282,514 "
                "input=245,022 (+ 13,762 cached) "
                "output=37,492 (reasoning 15,587)"
            ),
        )

    def test_format_exit_usage_line_clamps_negative_input(self) -> None:
        usage = SimpleNamespace(
            total_prompt_tokens=100,
            total_prompt_cached_tokens=120,
            total_completion_tokens=5,
            total_reasoning_tokens=0,
        )

        line = _format_exit_usage_line(usage)

        self.assertEqual(
            line,
            "Token usage: total=5 input=0 (+ 120 cached) output=5 (reasoning 0)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
