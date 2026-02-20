from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.rewind_store import RewindCheckpoint, RewindRestorePlan
from terminal_agent.tui_parts.commands import CommandsMixin


class _DummyRenderer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def append_system_message(self, content: str, is_error: bool = False) -> None:
        self.messages.append((content, is_error))


class _DummyStatusBar:
    def __init__(self) -> None:
        self.refresh_calls = 0

    async def refresh(self) -> None:
        self.refresh_calls += 1


class _DummyRewindStore:
    def __init__(self, plan: RewindRestorePlan) -> None:
        self._plan = plan
        self.restore_calls: list[int] = []
        self.prune_calls: list[int] = []

    def restore_code_before_checkpoint(self, *, checkpoint_id: int) -> RewindRestorePlan:
        self.restore_calls.append(checkpoint_id)
        return self._plan

    def prune_after_checkpoint(self, *, checkpoint_id: int) -> int:
        self.prune_calls.append(checkpoint_id)
        return 2


class _DummyCommands(CommandsMixin):
    def __init__(self, checkpoint: RewindCheckpoint, plan: RewindRestorePlan) -> None:
        self._busy = False
        self._renderer = _DummyRenderer()
        self._status_bar = _DummyStatusBar()
        self._rewind_store = _DummyRewindStore(plan)
        self._prefilled_text = ""
        self._refresh_count = 0
        self._replay_calls = 0
        self._checkpoint = checkpoint

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy

    def _refresh_layers(self) -> None:
        self._refresh_count += 1

    def _resolve_checkpoint_user_text(
        self,
        *,
        checkpoint: RewindCheckpoint,
        fallback: str,
    ) -> str:
        if checkpoint == self._checkpoint:
            return "restored user input"
        return fallback

    async def _replay_scrollback_after_rewind(self) -> None:
        self._replay_calls += 1

    def _prefill_user_input(self, text: str) -> None:
        self._prefilled_text = text


def _build_checkpoint() -> RewindCheckpoint:
    return RewindCheckpoint(
        checkpoint_id=3,
        turn_number=5,
        user_preview="hello",
        user_message="hello world",
        created_at="2026-01-01T00:00:00Z",
        touched_files=("a.py",),
    )


def _build_plan(checkpoint: RewindCheckpoint) -> RewindRestorePlan:
    return RewindRestorePlan(
        checkpoint=checkpoint,
        files=(),
        total_added_lines=1,
        total_removed_lines=1,
        writable_files_count=1,
        skipped_binary_count=0,
        skipped_unknown_count=0,
    )


class TestRewindCommandSemantics(unittest.TestCase):
    def test_rewind_turn_before_checkpoint(self) -> None:
        self.assertEqual(CommandsMixin._rewind_turn_before_checkpoint(1), 0)
        self.assertEqual(CommandsMixin._rewind_turn_before_checkpoint(2), 1)
        self.assertEqual(CommandsMixin._rewind_turn_before_checkpoint(10), 9)

    def test_restore_code_triggers_replay_and_prefill_without_done_message(self) -> None:
        checkpoint = _build_checkpoint()
        plan = _build_plan(checkpoint)
        commands = _DummyCommands(checkpoint, plan)

        asyncio.run(
            commands._execute_rewind_mode(
                checkpoint=checkpoint,
                mode="restore_code",
            )
        )

        self.assertEqual(commands._replay_calls, 1)
        self.assertEqual(commands._prefilled_text, "restored user input")
        self.assertEqual(commands._status_bar.refresh_calls, 1)
        self.assertEqual(commands._rewind_store.restore_calls, [3])
        self.assertEqual(commands._rewind_store.prune_calls, [3])
        self.assertFalse(commands._busy)
        self.assertGreaterEqual(commands._refresh_count, 1)
        self.assertFalse(
            any("Rewind done" in content for content, _ in commands._renderer.messages)
        )


if __name__ == "__main__":
    unittest.main()
