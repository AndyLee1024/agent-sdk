import unittest

from comate_agent_sdk.context.truncation import TruncationRecord
from comate_agent_sdk.system_tools.output_formatter import OutputFormatter, _default_truncation
from comate_agent_sdk.system_tools.tool_result import err, ok


class TestOutputFormatter(unittest.TestCase):
    def test_format_read_with_truncation_footer_and_hints(self) -> None:
        payload = ok(
            data={
                "content": "     1\tline1\n     2\tline2",
                "total_lines": 1200,
                "lines_returned": 2,
                "has_more": True,
                "next_offset_line": 2,
                "truncated": True,
                "artifact": {"relpath": ".agent_workspace/.artifacts/read/abc.txt"},
            },
            meta={
                "file_path": "src/main.py",
                "offset_line": 0,
                "limit_lines": 2,
            },
        )

        formatted = OutputFormatter.format(
            tool_name="Read",
            tool_call_id="tc_read_1",
            result_dict=payload,
        )

        self.assertIn("# Read: src/main.py", formatted.text)
        self.assertIn("Lines 1-2 of 1200", formatted.text)
        self.assertIn("Recommended next step (token-efficient):", formatted.text)
        self.assertEqual(formatted.meta.status, "ok")
        self.assertIsNotNone(formatted.meta.truncation)
        self.assertIsInstance(formatted.meta.truncation, TruncationRecord)
        self.assertTrue(formatted.meta.truncation.formatter_truncated)
        self.assertEqual(formatted.meta.truncation.formatter_reason, "line_limit,output_spilled")
        self.assertIsNotNone(formatted.meta.retrieval_hints)
        self.assertGreaterEqual(len(formatted.meta.retrieval_hints), 2)

    def test_format_write_includes_file_ops(self) -> None:
        payload = ok(
            data={
                "bytes_written": 12,
                "file_bytes": 40,
                "created": False,
                "sha256": "after_hash",
                "relpath": "src/config.json",
            },
            meta={
                "file_path": "src/config.json",
                "operation": "overwrite",
                "sha256_before": "before_hash",
                "sha256_after": "after_hash",
            },
        )

        formatted = OutputFormatter.format(
            tool_name="Write",
            tool_call_id="tc_write_1",
            result_dict=payload,
        )

        self.assertIn("# Write: src/config.json", formatted.text)
        self.assertIn("Operation: overwrite", formatted.text)
        self.assertEqual(formatted.meta.status, "ok")
        self.assertIsNotNone(formatted.meta.file_ops)
        self.assertEqual(formatted.meta.file_ops["operation"], "overwrite")
        self.assertEqual(formatted.meta.file_ops["sha256_after"], "after_hash")

    def test_format_error_includes_retry_footer(self) -> None:
        payload = err(
            "INVALID_ARGUMENT",
            "invalid path",
            field_errors=[{"field": "path", "message": "must be absolute"}],
        )
        formatted = OutputFormatter.format(
            tool_name="Grep",
            tool_call_id="tc_grep_1",
            result_dict=payload,
        )

        self.assertIn("# Grep Error", formatted.text)
        self.assertIn("Code: INVALID_ARGUMENT", formatted.text)
        self.assertIn("Recommended next step (token-efficient):", formatted.text)
        self.assertEqual(formatted.meta.status, "error")
        self.assertEqual(formatted.meta.error_code, "INVALID_ARGUMENT")
        self.assertEqual(formatted.meta.error_field, "path")

    def test_format_ask_user_question(self) -> None:
        payload = ok(
            data={
                "status": "waiting_for_input",
                "questions": [
                    {
                        "question": "Which option?",
                        "header": "Choice",
                        "options": [
                            {"label": "A", "description": "Option A"},
                            {"label": "B", "description": "Option B"},
                        ],
                        "multiSelect": False,
                    }
                ],
            },
            message="Prepared 1 question(s) for user",
        )
        formatted = OutputFormatter.format(
            tool_name="AskUserQuestion",
            tool_call_id="tc_question_1",
            result_dict=payload,
        )

        self.assertIn("# AskUserQuestion", formatted.text)
        self.assertIn("Question count: 1", formatted.text)
        self.assertEqual(formatted.meta.status, "ok")

    def test_default_truncation_returns_truncation_record(self) -> None:
        record = _default_truncation(
            {
                "truncated": True,
                "raw_output_truncated": True,
                "total_matches": 123,
            }
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertIsInstance(record, TruncationRecord)
        self.assertTrue(record.formatter_truncated)
        self.assertIn("output_capture_limit", record.formatter_reason)
        self.assertEqual(record.formatter_total_estimate, 123)

    def test_format_read_no_truncation_returns_none(self) -> None:
        payload = ok(
            data={
                "content": "     1\tline1\n     2\tline2",
                "total_lines": 2,
                "lines_returned": 2,
                "has_more": False,
                "truncated": False,
            },
            meta={
                "file_path": "src/main.py",
                "offset_line": 0,
                "limit_lines": 2,
            },
        )
        formatted = OutputFormatter.format(
            tool_name="Read",
            tool_call_id="tc_read_no_truncation",
            result_dict=payload,
        )
        self.assertIsNone(formatted.meta.truncation)

    def test_format_todo_result_with_active_todos(self) -> None:
        """测试 TodoWrite 格式化器显示活跃 todos 列表"""
        payload = ok(
            data={
                "count": 4,
                "active_count": 2,
                "persisted": True,
                "todo_path": "todos.json",
                "todos": [
                    {"id": "1", "content": "Fix authentication bug", "status": "pending", "priority": "high"},
                    {"id": "2", "content": "Add unit tests", "status": "in_progress", "priority": "medium"},
                    {"id": "3", "content": "Update docs", "status": "completed", "priority": "low"},
                    {"id": "4", "content": "Refactor code", "status": "completed", "priority": "medium"},
                ],
            },
            message="Remember to keep using the TODO list...",
        )

        formatted = OutputFormatter.format(
            tool_name="TodoWrite",
            tool_call_id="tc_todo_1",
            result_dict=payload,
        )

        # 验证基本统计信息
        self.assertIn("# TodoWrite", formatted.text)
        self.assertIn("Total todos: 4", formatted.text)
        self.assertIn("Active todos: 2", formatted.text)
        self.assertIn("Persisted: True", formatted.text)
        self.assertIn("Todo path: todos.json", formatted.text)

        # 验证活跃 todos 列表
        self.assertIn("Active Tasks:", formatted.text)
        self.assertIn("- [pending] #1 (high) Fix authentication bug", formatted.text)
        self.assertIn("- [in_progress] #2 (medium) Add unit tests", formatted.text)

        # 确保不显示已完成的 todos
        self.assertNotIn("Update docs", formatted.text)
        self.assertNotIn("Refactor code", formatted.text)
        self.assertNotIn("completed", formatted.text)

        # 验证消息
        self.assertIn("Remember to keep using the TODO list...", formatted.text)
        self.assertEqual(formatted.meta.status, "ok")

    def test_format_todo_result_no_active_todos(self) -> None:
        """测试所有 todos 都已完成时不显示 Active Tasks 部分"""
        payload = ok(
            data={
                "count": 2,
                "active_count": 0,
                "persisted": False,
                "todo_path": None,
                "todos": [
                    {"id": "1", "content": "Task 1", "status": "completed", "priority": "high"},
                    {"id": "2", "content": "Task 2", "status": "completed", "priority": "medium"},
                ],
            },
            message="All tasks completed!",
        )

        formatted = OutputFormatter.format(
            tool_name="TodoWrite",
            tool_call_id="tc_todo_2",
            result_dict=payload,
        )

        # 验证基本信息
        self.assertIn("Total todos: 2", formatted.text)
        self.assertIn("Active todos: 0", formatted.text)

        # 不应该显示 Active Tasks 部分
        self.assertNotIn("Active Tasks:", formatted.text)

    def test_format_todo_result_backward_compatible(self) -> None:
        """测试向后兼容：没有 todos 字段时仍能正常工作"""
        payload = ok(
            data={
                "count": 3,
                "active_count": 1,
                "persisted": True,
                "todo_path": "todos.json",
                # 注意：没有 todos 字段
            },
            message="TodoWrite executed",
        )

        formatted = OutputFormatter.format(
            tool_name="TodoWrite",
            tool_call_id="tc_todo_3",
            result_dict=payload,
        )

        # 应该能正常显示统计信息
        self.assertIn("Total todos: 3", formatted.text)
        self.assertIn("Active todos: 1", formatted.text)

        # 没有 todos 数据，不应该显示 Active Tasks 部分
        self.assertNotIn("Active Tasks:", formatted.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
