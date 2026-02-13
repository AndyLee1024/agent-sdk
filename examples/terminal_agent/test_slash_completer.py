from __future__ import annotations

import sys
import unittest
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.slash_commands import (
    SLASH_COMMANDS,
    SLASH_COMMAND_SPECS,
    SlashCommandCompleter,
)


def _collect_suggestions(completer: SlashCommandCompleter, text: str) -> list[str]:
    doc = Document(text=text, cursor_position=len(text))
    event = CompleteEvent(completion_requested=True)
    return [item.text for item in completer.get_completions(doc, event)]


class TestSlashCommandCompleter(unittest.TestCase):
    def setUp(self) -> None:
        self.completer = SlashCommandCompleter(SLASH_COMMAND_SPECS)

    def test_starts_with_slash_show_all_commands(self) -> None:
        suggestions = _collect_suggestions(self.completer, "/")
        self.assertEqual(set(suggestions), set(SLASH_COMMANDS))

    def test_slash_prefix_filters_commands(self) -> None:
        suggestions = _collect_suggestions(self.completer, "/co")
        self.assertEqual(suggestions, ["/context"])

    def test_exact_command_match_hides_completions(self) -> None:
        suggestions = _collect_suggestions(self.completer, "/help")
        self.assertEqual(suggestions, [])

    def test_exact_alias_match_hides_completions(self) -> None:
        suggestions = _collect_suggestions(self.completer, "/h")
        self.assertEqual(suggestions, [])

    def test_leading_space_before_slash_does_not_trigger(self) -> None:
        suggestions = _collect_suggestions(self.completer, " /")
        self.assertEqual(suggestions, [])

    def test_description_meta_is_exposed(self) -> None:
        doc = Document(text="/co", cursor_position=3)
        event = CompleteEvent(completion_requested=True)
        items = list(self.completer.get_completions(doc, event))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "/context")
        self.assertEqual(items[0].display_meta_text.strip(), "Show context usage summary")

    def test_non_slash_prefix_does_not_trigger(self) -> None:
        suggestions = _collect_suggestions(self.completer, "abc/")
        self.assertEqual(suggestions, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
