from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from comate_agent_sdk.agent.events import (
    SubagentProgressEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)

from terminal_agent.event_renderer import EventRenderer


def _todo_payload(size: int) -> list[dict[str, str]]:
    todos: list[dict[str, str]] = []
    for idx in range(size):
        todos.append(
            {
                "content": f"todo-{idx}",
                "status": "pending",
                "priority": "medium",
            }
        )
    return todos


class TestEventRenderer(unittest.TestCase):
    def test_history_distinguishes_user_tool_assistant(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.seed_user_message("hello")
        renderer.handle_event(
            ToolCallEvent(
                tool="Read",
                args={"path": "/tmp/a.py"},
                tool_call_id="tc_1",
            )
        )
        renderer.handle_event(
            ToolResultEvent(
                tool="Read",
                result="ok",
                tool_call_id="tc_1",
                is_error=False,
            )
        )
        renderer.handle_event(TextEvent(content="assistant answer"))
        renderer.finalize_turn()

        entry_types = [entry.entry_type for entry in renderer.history_entries()]
        self.assertIn("user", entry_types)
        self.assertIn("tool_call", entry_types)
        self.assertIn("tool_result", entry_types)
        self.assertIn("assistant", entry_types)

    def test_loading_line_tracks_running_task(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="Task",
                args={"subagent_type": "Planner", "description": "拆解任务"},
                tool_call_id="tc_task_1",
            )
        )
        self.assertIn("⏳", renderer.loading_line())
        self.assertIn("拆解任务", renderer.loading_line())

        renderer.handle_event(
            SubagentProgressEvent(
                tool_call_id="tc_task_1",
                subagent_name="Planner",
                description="",
                status="running",
                elapsed_ms=250,
                tokens=180,
            )
        )
        self.assertIn("tok", renderer.loading_line())

        renderer.handle_event(
            ToolResultEvent(
                tool="Task",
                result="done",
                tool_call_id="tc_task_1",
                is_error=False,
            )
        )
        self.assertEqual(renderer.loading_line(), "")

    def test_todo_lines_are_folded_to_six_lines(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="TodoWrite",
                args={"todos": _todo_payload(10)},
                tool_call_id="tc_todo_1",
            )
        )

        todo_lines = renderer.todo_lines()
        self.assertTrue(todo_lines)
        self.assertLessEqual(len(todo_lines), 6)
        self.assertIn("折叠", todo_lines[-1])

    def test_task_tool_history_hides_prompt_and_prefix_icons(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="Task",
                args={
                    "subagent_type": "Explorer",
                    "description": "研究 kimi-cli-main 实现",
                    "prompt": "这是很长的提示词，不应出现在历史里",
                },
                tool_call_id="tc_task_2",
            )
        )
        renderer.handle_event(
            ToolResultEvent(
                tool="Task",
                result="done",
                tool_call_id="tc_task_2",
                is_error=False,
            )
        )

        tool_call_entry = next(
            entry for entry in renderer.history_entries() if entry.entry_type == "tool_call"
        )
        tool_result_entry = next(
            entry for entry in renderer.history_entries() if entry.entry_type == "tool_result"
        )

        self.assertEqual(tool_call_entry.text, "研究 kimi-cli-main 实现")
        self.assertNotIn("prompt", tool_call_entry.text.lower())
        self.assertFalse(tool_call_entry.text.startswith("→"))
        self.assertFalse(tool_result_entry.text.startswith("✓"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
