from __future__ import annotations

import sys
import unittest
from pathlib import Path

from rich.console import Console
from rich.text import Text

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.layout_coordinator import TerminalLayoutCoordinator, _renderable_to_lines


def _multiline(prefix: str, count: int) -> Text:
    return Text("\n".join(f"{prefix}-{idx}" for idx in range(count)))


def _to_plain_lines(console: Console, renderable) -> list[str]:
    lines = _renderable_to_lines(console, renderable)
    plain_lines: list[str] = []
    for line in lines:
        plain_lines.append("".join(segment.text for segment in line).rstrip("\n"))
    return plain_lines


class TestLayoutCoordinator(unittest.TestCase):
    def test_compose_order_is_message_then_todo_then_loading(self) -> None:
        console = Console(width=80, height=20, record=True)
        layout = TerminalLayoutCoordinator(console)
        layout.update_layers(
            message=_multiline("msg", 2),
            todo=_multiline("todo", 2),
            loading=_multiline("load", 1),
        )

        plain_lines = _to_plain_lines(console, layout._compose())
        msg_indices = [idx for idx, line in enumerate(plain_lines) if "msg-" in line]
        todo_indices = [idx for idx, line in enumerate(plain_lines) if "todo-" in line]
        load_indices = [idx for idx, line in enumerate(plain_lines) if "load-" in line]

        self.assertTrue(msg_indices)
        self.assertTrue(todo_indices)
        self.assertTrue(load_indices)
        self.assertLess(max(msg_indices), min(todo_indices))
        self.assertLess(max(todo_indices), min(load_indices))

    def test_crop_prefers_message_then_todo_min_then_loading(self) -> None:
        console = Console(width=80, height=12, record=True)
        layout = TerminalLayoutCoordinator(console)
        layout.update_layers(
            message=_multiline("msg", 20),
            todo=_multiline("todo", 8),
            loading=_multiline("load", 3),
        )

        plain_lines = _to_plain_lines(console, layout._compose())
        todo_kept = [line for line in plain_lines if "todo-" in line]
        loading_kept = [line for line in plain_lines if "load-" in line]

        self.assertGreaterEqual(len(todo_kept), 6)
        self.assertGreaterEqual(len(loading_kept), 1)

    def test_extreme_height_still_keeps_one_loading_line(self) -> None:
        console = Console(width=80, height=3, record=True)
        layout = TerminalLayoutCoordinator(console)
        layout.update_layers(
            message=None,
            todo=_multiline("todo", 8),
            loading=_multiline("load", 3),
        )

        plain_lines = _to_plain_lines(console, layout._compose())
        loading_kept = [line for line in plain_lines if "load-" in line]

        self.assertEqual(len(loading_kept), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
