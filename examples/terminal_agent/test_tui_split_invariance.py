from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.tui import TerminalAgentTUI, UIMode


class _FakeMentionCompleter:
    def extract_context(self, text: str):
        if "@" in text:
            return {"match": text}
        return None


class _FakeMessage:
    def __init__(self, *, text=None, content=None) -> None:
        self.text = text
        self.content = content


class _FakeItem:
    def __init__(self, message) -> None:
        self.message = message


class TestTUISplitInvariance(unittest.TestCase):
    def test_public_api_stays_stable_after_split(self) -> None:
        self.assertEqual(UIMode.NORMAL.value, "normal")
        self.assertEqual(UIMode.QUESTION.value, "question")
        self.assertEqual(UIMode.SELECTION.value, "selection")
        self.assertTrue(callable(getattr(TerminalAgentTUI, "_build_key_bindings")))
        self.assertTrue(callable(getattr(TerminalAgentTUI, "_execute_command")))
        self.assertTrue(callable(getattr(TerminalAgentTUI, "_loading_text")))

    def test_completion_context_logic_remains_unchanged(self) -> None:
        tui = TerminalAgentTUI.__new__(TerminalAgentTUI)
        tui._mention_completer = _FakeMentionCompleter()

        self.assertTrue(tui._completion_context_active("/", ""))
        self.assertTrue(tui._completion_context_active("@src/main.py", ""))
        self.assertFalse(tui._completion_context_active("/model", " trailing"))
        self.assertFalse(tui._completion_context_active("hello world", ""))

    def test_extract_assistant_text_behavior_is_preserved(self) -> None:
        text_item = _FakeItem(_FakeMessage(text="plain text", content="ignored"))
        self.assertEqual(
            TerminalAgentTUI._extract_assistant_text(text_item),
            "plain text",
        )

        content_item = _FakeItem(
            _FakeMessage(
                text=None,
                content=[
                    {"type": "text", "text": "hello "},
                    {"type": "image", "url": "noop"},
                    {"type": "text", "text": "world"},
                ],
            )
        )
        self.assertEqual(
            TerminalAgentTUI._extract_assistant_text(content_item),
            "hello world",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
