from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from comate_agent_sdk.context.items import ContextItem, ItemType
from comate_agent_sdk.llm.messages import ToolMessage
from terminal_agent.rewind_store import RewindStore


def _tool_item(*, turn: int, relpath: str, created: bool, operation: str) -> ContextItem:
    return ContextItem(
        item_type=ItemType.TOOL_RESULT,
        message=ToolMessage(
            tool_call_id=f"tc_{turn}_{relpath}",
            tool_name="Write",
            content="ok",
            is_error=False,
        ),
        content_text="ok",
        token_count=1,
        tool_name="Write",
        created_turn=turn,
        is_tool_error=False,
        metadata={
            "tool_raw_envelope": {
                "meta": {
                    "file_path": relpath,
                    "operation": operation,
                },
                "data": {
                    "relpath": relpath,
                    "created": created,
                },
            }
        },
    )


class TestRewindStore(unittest.TestCase):
    def test_capture_and_restore_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage_root = root / "session"
            project_root = root / "repo"
            project_root.mkdir(parents=True, exist_ok=True)

            target_file = project_root / "a.txt"
            target_file.write_text("v1\n", encoding="utf-8")

            context = SimpleNamespace(
                _turn_number=1,
                conversation=SimpleNamespace(items=[]),
            )
            agent = SimpleNamespace(_context=context)
            session = SimpleNamespace(
                session_id="s1",
                _storage_root=storage_root,
                _agent=agent,
            )
            store = RewindStore(session=session, project_root=project_root)

            # turn1: v1 -> v2
            target_file.write_text("v2\n", encoding="utf-8")
            item1 = _tool_item(turn=1, relpath="a.txt", created=False, operation="overwrite")
            context.conversation.items = [item1]
            cp1 = store.capture_checkpoint_for_latest_turn(user_preview="first")
            self.assertIsNotNone(cp1)

            # turn2: v2 -> v3
            context._turn_number = 2
            target_file.write_text("v3\n", encoding="utf-8")
            item2 = _tool_item(turn=2, relpath="a.txt", created=False, operation="overwrite")
            context.conversation.items = [item1, item2]
            cp2 = store.capture_checkpoint_for_latest_turn(user_preview="second")
            self.assertIsNotNone(cp2)

            plan = store.build_restore_plan(checkpoint_id=1)
            self.assertEqual(plan.checkpoint.checkpoint_id, 1)
            self.assertEqual(plan.writable_files_count, 1)
            self.assertEqual(plan.skipped_binary_count, 0)

            applied = store.restore_code_to_checkpoint(checkpoint_id=1)
            self.assertEqual(applied.writable_files_count, 1)
            self.assertEqual(target_file.read_text(encoding="utf-8"), "v2\n")

    def test_rewind_plan_skips_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage_root = root / "session"
            project_root = root / "repo"
            project_root.mkdir(parents=True, exist_ok=True)

            binary_path = project_root / "image.bin"
            binary_path.write_bytes(b"\x00\x01\x02")

            context = SimpleNamespace(
                _turn_number=1,
                conversation=SimpleNamespace(items=[]),
            )
            agent = SimpleNamespace(_context=context)
            session = SimpleNamespace(
                session_id="s1",
                _storage_root=storage_root,
                _agent=agent,
            )
            store = RewindStore(session=session, project_root=project_root)

            item1 = _tool_item(turn=1, relpath="image.bin", created=True, operation="create")
            context.conversation.items = [item1]
            cp1 = store.capture_checkpoint_for_latest_turn(user_preview="binary-1")
            self.assertIsNotNone(cp1)

            context._turn_number = 2
            binary_path.write_bytes(b"\x00\x01\x02\x03")
            item2 = _tool_item(turn=2, relpath="image.bin", created=False, operation="overwrite")
            context.conversation.items = [item1, item2]
            cp2 = store.capture_checkpoint_for_latest_turn(user_preview="binary-2")
            self.assertIsNotNone(cp2)

            plan = store.build_restore_plan(checkpoint_id=1)
            self.assertEqual(plan.writable_files_count, 0)
            self.assertEqual(plan.skipped_binary_count, 1)

    def test_list_order_and_prune_after_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage_root = root / "session"
            project_root = root / "repo"
            project_root.mkdir(parents=True, exist_ok=True)

            target_file = project_root / "a.txt"
            target_file.write_text("v1\n", encoding="utf-8")

            context = SimpleNamespace(
                _turn_number=1,
                conversation=SimpleNamespace(items=[]),
            )
            agent = SimpleNamespace(_context=context)
            session = SimpleNamespace(
                session_id="s1",
                _storage_root=storage_root,
                _agent=agent,
            )
            store = RewindStore(session=session, project_root=project_root)

            item1 = _tool_item(turn=1, relpath="a.txt", created=False, operation="overwrite")
            context.conversation.items = [item1]
            store.capture_checkpoint_for_latest_turn(user_preview="first")

            context._turn_number = 2
            target_file.write_text("v2\n", encoding="utf-8")
            item2 = _tool_item(turn=2, relpath="a.txt", created=False, operation="overwrite")
            context.conversation.items = [item1, item2]
            store.capture_checkpoint_for_latest_turn(user_preview="second")

            checkpoints = store.list_checkpoints()
            self.assertEqual([cp.checkpoint_id for cp in checkpoints], [1, 2])

            dropped = store.prune_after_checkpoint(checkpoint_id=1)
            self.assertEqual(dropped, 1)

            checkpoints_after = store.list_checkpoints()
            self.assertEqual([cp.checkpoint_id for cp in checkpoints_after], [1])

    def test_checkpoint_preserves_full_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            storage_root = root / "session"
            project_root = root / "repo"
            project_root.mkdir(parents=True, exist_ok=True)

            context = SimpleNamespace(
                _turn_number=1,
                conversation=SimpleNamespace(items=[]),
            )
            agent = SimpleNamespace(_context=context)
            session = SimpleNamespace(
                session_id="s1",
                _storage_root=storage_root,
                _agent=agent,
            )
            store = RewindStore(session=session, project_root=project_root)

            message = "我来自福冈，今天想聊一下我来自哪里以及如何表达。"
            cp = store.capture_checkpoint_for_latest_turn(user_preview=message)
            self.assertIsNotNone(cp)
            assert cp is not None
            self.assertEqual(cp.user_message, message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
