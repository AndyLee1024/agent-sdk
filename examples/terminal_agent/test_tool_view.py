from __future__ import annotations

import unittest

from terminal_agent.tool_view import ToolEventView


class TestToolEventView(unittest.TestCase):
    def test_render_result_mutates_same_text_reference(self) -> None:
        view = ToolEventView(fancy_progress_effect=False)
        text_ref = view.render_call(
            "Read",
            {"path": "/tmp/demo.py", "offset_line": 1, "limit_lines": 50},
            "tc_read_1",
        )

        self.assertIn("→ Read", text_ref.plain)
        self.assertIn("path=/tmp/demo.py", text_ref.plain)

        view.render_result(
            tool_name="Read",
            tool_call_id="tc_read_1",
            result="ok",
            is_error=False,
        )
        self.assertIn("✓ Read", text_ref.plain)
        self.assertIn("path=/tmp/demo.py", text_ref.plain)

    def test_interrupt_running_marks_text_ref_as_interrupted(self) -> None:
        view = ToolEventView(fancy_progress_effect=False)
        text_ref = view.render_call(
            "Bash",
            {"command": "npm test"},
            "tc_bash_1",
        )

        view.interrupt_running()
        self.assertIn("已中断", text_ref.plain)
        self.assertIn("⏹ Bash", text_ref.plain)

    def test_has_running_tasks_only_counts_task_tool(self) -> None:
        view = ToolEventView(fancy_progress_effect=False)
        view.render_call("Read", {"path": "/tmp/a.py"}, "tc_read_2")
        self.assertFalse(view.has_running_tasks())

        task_ref = view.render_call(
            "Task",
            {"subagent_type": "Planner", "description": "拆解任务"},
            "tc_task_1",
        )
        self.assertTrue(view.has_running_tasks())

        view.render_result(
            tool_name="Task",
            tool_call_id="tc_task_1",
            result="done",
            is_error=False,
        )
        self.assertFalse(view.has_running_tasks())
        self.assertIn("✓ Planner(拆解任务) · 完成", task_ref.plain)


if __name__ == "__main__":
    unittest.main(verbosity=2)
