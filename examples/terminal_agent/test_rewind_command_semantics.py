from __future__ import annotations

import unittest

from terminal_agent.tui_parts.commands import CommandsMixin


class TestRewindCommandSemantics(unittest.TestCase):
    def test_rewind_turn_before_checkpoint(self) -> None:
        self.assertEqual(CommandsMixin._rewind_turn_before_checkpoint(1), 0)
        self.assertEqual(CommandsMixin._rewind_turn_before_checkpoint(2), 1)
        self.assertEqual(CommandsMixin._rewind_turn_before_checkpoint(10), 9)


if __name__ == "__main__":
    unittest.main()
