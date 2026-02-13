from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.tui import TerminalAgentTUI, UIMode


class _FakeBuffer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.cursor_position = len(text)

    def cancel_completion(self) -> None:
        return None

    def set_document(self, document, bypass_readonly: bool = False) -> None:
        del bypass_readonly
        self.text = document.text
        self.cursor_position = document.cursor_position


class _FakeInputArea:
    def __init__(self, text: str) -> None:
        self.text = text
        self.buffer = _FakeBuffer(text)
        self.buffer.complete_state = None


def _build_tui(threshold: int = 5) -> TerminalAgentTUI:
    tui = TerminalAgentTUI.__new__(TerminalAgentTUI)
    tui._suppress_input_change_hook = False
    tui._busy = False
    tui._last_input_len = 0
    tui._last_input_text = ""
    tui._paste_threshold_chars = threshold
    tui._paste_placeholder_text = None
    tui._active_paste_token = None
    tui._paste_payload_by_token = {}
    tui._paste_token_seq = 0
    tui._invalidate = lambda: None
    return tui


class TestTUIPastePlaceholder(unittest.TestCase):
    def test_handle_large_paste_replaces_with_placeholder_and_stores_payload(self) -> None:
        tui = _build_tui(threshold=5)
        payload = "abcdefgh"
        buffer = _FakeBuffer(payload)

        handled = tui._handle_large_paste(buffer)

        self.assertTrue(handled)
        self.assertEqual(tui._paste_placeholder_text, "[Pasted Content 8 chars]")
        self.assertEqual(buffer.text, "[Pasted Content 8 chars]")
        self.assertIsNotNone(tui._active_paste_token)
        self.assertEqual(len(tui._paste_payload_by_token), 1)
        token = tui._active_paste_token
        self.assertIsNotNone(token)
        self.assertEqual(tui._paste_payload_by_token[token], payload)

    def test_resolve_submit_uses_payload_for_placeholder_with_whitespace(self) -> None:
        tui = _build_tui(threshold=5)
        payload = "abcdefgh"
        buffer = _FakeBuffer(payload)
        tui._handle_large_paste(buffer)
        placeholder = tui._paste_placeholder_text
        self.assertIsNotNone(placeholder)

        display_text, submit_text = tui._resolve_submit_texts(f"  {placeholder}\n")

        self.assertEqual(display_text, payload)
        self.assertEqual(submit_text, payload)

    def test_handle_large_paste_preserves_existing_prefix_text(self) -> None:
        tui = _build_tui(threshold=5)
        prefix = "abcde"
        payload = "fghijklm"
        buffer = _FakeBuffer(f"{prefix}{payload}")
        tui._last_input_text = prefix
        tui._last_input_len = len(prefix)

        handled = tui._handle_large_paste(buffer)

        self.assertTrue(handled)
        self.assertEqual(buffer.text, f"{prefix}[Pasted Content 8 chars]")
        display_text, submit_text = tui._resolve_submit_texts(buffer.text)
        self.assertEqual(display_text, f"{prefix}{payload}")
        self.assertEqual(submit_text, f"{prefix}{payload}")

    def test_whitespace_change_does_not_clear_active_placeholder_mapping(self) -> None:
        tui = _build_tui(threshold=5)
        payload = "abcdefgh"
        buffer = _FakeBuffer(payload)
        tui._handle_large_paste(buffer)
        token_before = tui._active_paste_token
        placeholder = tui._paste_placeholder_text
        self.assertIsNotNone(token_before)
        self.assertIsNotNone(placeholder)

        buffer.text = f"{placeholder}\n"
        handled = tui._handle_large_paste(buffer)

        self.assertFalse(handled)
        self.assertEqual(tui._active_paste_token, token_before)
        self.assertEqual(tui._paste_placeholder_text, placeholder)
        self.assertEqual(tui._paste_payload_by_token[token_before], payload)

    def test_substantive_placeholder_edit_clears_mapping(self) -> None:
        tui = _build_tui(threshold=5)
        payload = "abcdefgh"
        buffer = _FakeBuffer(payload)
        tui._handle_large_paste(buffer)
        placeholder = tui._paste_placeholder_text
        self.assertIsNotNone(placeholder)

        buffer.text = placeholder.replace("chars", "bytes")
        handled = tui._handle_large_paste(buffer)

        self.assertFalse(handled)
        self.assertIsNone(tui._active_paste_token)
        self.assertIsNone(tui._paste_placeholder_text)
        self.assertEqual(tui._paste_payload_by_token, {})

    def test_resolve_submit_falls_back_to_text_after_mapping_cleared(self) -> None:
        tui = _build_tui(threshold=5)
        payload = "abcdefgh"
        buffer = _FakeBuffer(payload)
        tui._handle_large_paste(buffer)
        placeholder = tui._paste_placeholder_text
        self.assertIsNotNone(placeholder)

        buffer.text = placeholder.replace("chars", "bytes")
        tui._handle_large_paste(buffer)

        display_text, submit_text = tui._resolve_submit_texts(buffer.text)
        self.assertEqual(display_text, buffer.text.strip())
        self.assertEqual(submit_text, buffer.text.strip())

    def test_submit_from_input_sends_original_payload_not_placeholder(self) -> None:
        tui = _build_tui(threshold=5)
        payload = "abcdefgh"
        paste_buffer = _FakeBuffer(payload)
        tui._handle_large_paste(paste_buffer)
        placeholder = tui._paste_placeholder_text
        self.assertIsNotNone(placeholder)

        captured: dict[str, str | None] = {
            "text": None,
            "display_text": None,
        }

        async def _fake_submit_user_message(
            text: str,
            *,
            display_text: str | None = None,
        ) -> None:
            captured["text"] = text
            captured["display_text"] = display_text

        tui._busy = False
        tui._ui_mode = UIMode.NORMAL
        tui._input_area = _FakeInputArea(f"{placeholder}\n")
        tui._execute_command = lambda _: None
        tui._clear_input_area = lambda: None
        tui._submit_user_message = _fake_submit_user_message

        def _run_immediately(coro) -> None:
            asyncio.run(coro)

        tui._schedule_background = _run_immediately

        tui._submit_from_input()

        self.assertEqual(captured["text"], payload)
        self.assertEqual(captured["display_text"], payload)

    def test_submit_from_input_with_prefix_expands_placeholder_segment(self) -> None:
        tui = _build_tui(threshold=5)
        prefix = "abcde"
        payload = "fghijklm"
        tui._last_input_text = prefix
        tui._last_input_len = len(prefix)
        paste_buffer = _FakeBuffer(f"{prefix}{payload}")
        tui._handle_large_paste(paste_buffer)
        expected_display = f"{prefix}[Pasted Content 8 chars]"

        captured: dict[str, str | None] = {
            "text": None,
            "display_text": None,
        }

        async def _fake_submit_user_message(
            text: str,
            *,
            display_text: str | None = None,
        ) -> None:
            captured["text"] = text
            captured["display_text"] = display_text

        tui._busy = False
        tui._ui_mode = UIMode.NORMAL
        tui._input_area = _FakeInputArea(f"{expected_display}\n")
        tui._execute_command = lambda _: None
        tui._clear_input_area = lambda: None
        tui._submit_user_message = _fake_submit_user_message

        def _run_immediately(coro) -> None:
            asyncio.run(coro)

        tui._schedule_background = _run_immediately

        tui._submit_from_input()

        self.assertEqual(captured["text"], f"{prefix}{payload}")
        self.assertEqual(captured["display_text"], f"{prefix}{payload}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
