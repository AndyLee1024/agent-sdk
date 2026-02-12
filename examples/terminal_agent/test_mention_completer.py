from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.mention_completer import LocalFileMentionCompleter


class TestLocalFileMentionCompleter(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        (self.root / "main.py").write_text("print('ok')", encoding="utf-8")
        (self.root / "README.md").write_text("# readme", encoding="utf-8")
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "main_helper.py").write_text("x = 1", encoding="utf-8")
        self.completer = LocalFileMentionCompleter(self.root, refresh_interval=0.0, limit=200)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_guard_ignores_email_trigger(self) -> None:
        context = self.completer.extract_context("contact me: foo@example.com")
        self.assertIsNone(context)

    def test_suggest_prefers_basename_prefix_match(self) -> None:
        result = self.completer.suggest("@main", max_items=8)
        self.assertIsNotNone(result)
        assert result is not None
        _, candidates = result
        self.assertTrue(candidates)
        self.assertEqual(candidates[0], "main.py")
        self.assertIn("src/main_helper.py", candidates)

    def test_suggest_returns_none_for_completed_file(self) -> None:
        result = self.completer.suggest("open @main.py")
        self.assertIsNone(result)

    def test_apply_completion_replaces_fragment_and_appends_space(self) -> None:
        original = "open @ma"
        suggest_result = self.completer.suggest(original, max_items=8)
        self.assertIsNotNone(suggest_result)
        assert suggest_result is not None
        context, _ = suggest_result
        new_text, new_cursor = self.completer.apply_completion(
            full_text=original,
            cursor_position=len(original),
            context=context,
            completion="main.py",
            append_space=True,
        )
        self.assertEqual(new_text, "open @main.py ")
        self.assertEqual(new_cursor, len("open @main.py "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
