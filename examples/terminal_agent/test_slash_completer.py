from __future__ import annotations

import sys
import unittest
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.app import SLASH_COMMANDS, _SlashCommandCompleter


def _collect_suggestions(completer: _SlashCommandCompleter, text: str) -> list[str]:
    doc = Document(text=text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=False)
    return [item.text for item in completer.get_completions(doc, event)]


class TestSlashCommandCompleter(unittest.TestCase):
    def setUp(self) -> None:
        self.completer = _SlashCommandCompleter(SLASH_COMMANDS)

    def test_starts_with_slash_show_all_commands(self) -> None:
        suggestions = _collect_suggestions(self.completer, "/")
        self.assertEqual(suggestions, list(SLASH_COMMANDS))

    def test_slash_prefix_filters_commands(self) -> None:
        suggestions = _collect_suggestions(self.completer, "/co")
        self.assertEqual(suggestions, ["/context"])

    def test_leading_space_before_slash_does_not_trigger(self) -> None:
        suggestions = _collect_suggestions(self.completer, " /")
        self.assertEqual(suggestions, [])

    def test_non_slash_prefix_does_not_trigger(self) -> None:
        suggestions = _collect_suggestions(self.completer, "abc/")
        self.assertEqual(suggestions, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
