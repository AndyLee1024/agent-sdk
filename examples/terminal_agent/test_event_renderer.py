from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from comate_agent_sdk.agent.events import (
    StopEvent,
    SubagentProgressEvent,
    TextEvent,
    TodoUpdatedEvent,
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
        self.assertIn("tool_result", entry_types)
        self.assertIn("assistant", entry_types)

    def test_tool_panel_tracks_running_task_and_subagent(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="Task",
                args={"subagent_type": "Planner", "description": "拆解任务"},
                tool_call_id="tc_task_1",
            )
        )
        entries = renderer.tool_panel_entries(max_lines=4)
        self.assertTrue(entries)
        self.assertTrue(any("拆解任务" in line for _, line in entries))
        self.assertTrue(any("|_ init" in line for _, line in entries))

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
        entries = renderer.tool_panel_entries(max_lines=4)
        self.assertTrue(any("tok" in line for _, line in entries))
        self.assertFalse(any("|_ init" in line for _, line in entries))

        renderer.handle_event(
            ToolResultEvent(
                tool="Task",
                result="done",
                tool_call_id="tc_task_1",
                is_error=False,
            )
        )
        self.assertFalse(renderer.has_running_tools())

    def test_todo_lines_are_folded_to_six_lines(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            TodoUpdatedEvent(todos=_todo_payload(10))
        )

        todo_lines = renderer.todo_lines()
        self.assertTrue(todo_lines)
        self.assertLessEqual(len(todo_lines), 6)
        self.assertIn("…", todo_lines[-1])

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

        tool_result_entry = next(
            entry for entry in renderer.history_entries() if entry.entry_type == "tool_result"
        )

        self.assertIn("研究 kimi-cli-main 实现", tool_result_entry.text)
        self.assertNotIn("prompt", tool_result_entry.text.lower())
        self.assertFalse(tool_result_entry.text.startswith("✓"))

    def test_tool_result_includes_error_summary(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="Read",
                args={"path": "/tmp/a.py"},
                tool_call_id="tc_err_1",
            )
        )
        renderer.handle_event(
            ToolResultEvent(
                tool="Read",
                result={"error": "boom"},
                tool_call_id="tc_err_1",
                is_error=True,
            )
        )
        tool_result_entry = next(
            entry for entry in renderer.history_entries() if entry.entry_type == "tool_result"
        )
        self.assertIn("Read(", tool_result_entry.text)
        self.assertIn("boom", tool_result_entry.text)

    def test_todo_completed_appends_summary_and_hides_panel(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            TodoUpdatedEvent(
                todos=[
                    {"content": "a", "status": "completed", "priority": "medium"},
                    {"content": "b", "status": "completed", "priority": "medium"},
                ]
            )
        )
        tool_results = [e for e in renderer.history_entries() if e.entry_type == "tool_result"]
        self.assertTrue(tool_results)
        self.assertIn("todo 2/2 completed", tool_results[-1].text)
        self.assertFalse(renderer.has_active_todos())

    def test_stop_event_interrupted_appends_system_message(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.seed_user_message("hello")
        renderer.handle_event(StopEvent(reason="interrupted"))

        system_entries = [e for e in renderer.history_entries() if e.entry_type == "system"]
        self.assertTrue(system_entries)
        self.assertIn("中断", system_entries[-1].text)

    def test_reset_history_view_clears_runtime_state(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.seed_user_message("hello")
        renderer.handle_event(
            ToolCallEvent(
                tool="Read",
                args={"file_path": "a.py"},
                tool_call_id="tc_reset_1",
            )
        )
        renderer.handle_event(
            TodoUpdatedEvent(
                todos=[{"content": "a", "status": "pending", "priority": "medium"}]
            )
        )
        self.assertTrue(renderer.history_entries())
        self.assertTrue(renderer.has_running_tools())
        self.assertTrue(renderer.has_active_todos())

        renderer.reset_history_view()

        self.assertEqual(renderer.history_entries(), [])
        self.assertFalse(renderer.has_running_tools())
        self.assertFalse(renderer.has_active_todos())

    def test_edit_tool_result_with_diff_renders_rich_text(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="Edit",
                args={"file_path": "/tmp/test.py", "old_string": "old", "new_string": "new"},
                tool_call_id="tc_edit_1",
            )
        )
        diff_metadata = {
            "diff": [
                "--- /tmp/test.py",
                "+++ /tmp/test.py",
                "@@ -1,3 +1,3 @@",
                " line1",
                "-old",
                "+new",
                " line3",
            ]
        }
        renderer.handle_event(
            ToolResultEvent(
                tool="Edit",
                result="ok",
                tool_call_id="tc_edit_1",
                is_error=False,
                metadata=diff_metadata,
            )
        )

        tool_results = [e for e in renderer.history_entries() if e.entry_type == "tool_result"]
        self.assertTrue(tool_results)
        from rich.text import Text
        last_result = tool_results[-1]
        self.assertIsInstance(last_result.text, Text)
        plain_text = last_result.text.plain
        self.assertIn("+new", plain_text)
        self.assertIn("-old", plain_text)
        self.assertIn("@@", plain_text)

    def test_edit_tool_result_error_no_diff(self) -> None:
        renderer = EventRenderer()
        renderer.start_turn()
        renderer.handle_event(
            ToolCallEvent(
                tool="Edit",
                args={"file_path": "/tmp/test.py", "old_string": "missing", "new_string": "new"},
                tool_call_id="tc_edit_err",
            )
        )
        diff_metadata = {"diff": ["-old", "+new"]}
        renderer.handle_event(
            ToolResultEvent(
                tool="Edit",
                result="old_string not found",
                tool_call_id="tc_edit_err",
                is_error=True,
                metadata=diff_metadata,
            )
        )

        tool_results = [e for e in renderer.history_entries() if e.entry_type == "tool_result"]
        self.assertTrue(tool_results)
        last_result = tool_results[-1]
        self.assertTrue(last_result.is_error)
        # Should be a plain string, not Rich Text, since error results don't show diff
        self.assertIsInstance(last_result.text, str)

    def test_render_diff_text_truncation(self) -> None:
        diff_lines = [f"+added_line_{i}" for i in range(100)]
        result = EventRenderer._render_diff_text(diff_lines, max_lines=10)
        plain = result.plain
        self.assertIn("+added_line_0", plain)
        self.assertIn("+added_line_9", plain)
        self.assertNotIn("+added_line_10", plain)
        self.assertIn("90 more lines", plain)

    def test_render_diff_text_colors(self) -> None:
        diff_lines = [
            "--- a/file.py",
            "+++ b/file.py",
            "@@ -1,3 +1,3 @@",
            " context",
            "-removed",
            "+added",
        ]
        result = EventRenderer._render_diff_text(diff_lines, max_lines=50)
        # Check that each span has the correct style
        spans = result._spans
        plain = result.plain
        # Verify green background style for added line
        added_start = plain.index("+added")
        green_spans = [
            s for s in spans if str(s.style) == "on #154018" and s.start <= added_start < s.end
        ]
        self.assertTrue(green_spans, "'+added' should be styled with green background")
        # Verify red background style for removed line
        removed_start = plain.index("-removed")
        red_spans = [
            s for s in spans if str(s.style) == "on #3f1715" and s.start <= removed_start < s.end
        ]
        self.assertTrue(red_spans, "'-removed' should be styled with red background")


if __name__ == "__main__":
    unittest.main(verbosity=2)
