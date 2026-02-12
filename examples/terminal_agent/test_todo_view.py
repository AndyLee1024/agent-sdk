import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.models import TodoItemState
from terminal_agent.todo_view import extract_todos
from terminal_agent.todo_view import TodoStateStore


class TestTodoViewExtraction(unittest.TestCase):
    def test_extract_todos_supports_stringified_todos(self) -> None:
        args = {
            "todos": '[{"id":"1","content":"Read docs","status":"pending","priority":"high"}]',
        }

        todos = extract_todos(args)

        self.assertIsNotNone(todos)
        if todos is None:
            return
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0].content, "Read docs")
        self.assertEqual(todos[0].status, "pending")
        self.assertEqual(todos[0].priority, "high")

    def test_extract_todos_supports_stringified_params_todos(self) -> None:
        args = {
            "params": {
                "todos": '[{"id":"2","content":"Write tests","status":"in_progress","priority":"medium"}]',
            }
        }

        todos = extract_todos(args)

        self.assertIsNotNone(todos)
        if todos is None:
            return
        self.assertEqual(len(todos), 1)
        self.assertEqual(todos[0].content, "Write tests")
        self.assertEqual(todos[0].status, "in_progress")
        self.assertEqual(todos[0].priority, "medium")

    def test_extract_todos_returns_none_on_invalid_string(self) -> None:
        args = {"todos": "not-a-json-array"}
        todos = extract_todos(args)
        self.assertIsNone(todos)


class TestTodoViewRendering(unittest.TestCase):
    def test_completed_items_are_displayed_when_all_done(self) -> None:
        store = TodoStateStore()
        store.update(
            [
                TodoItemState(content="task-1", status="completed", priority="medium"),
                TodoItemState(content="task-2", status="completed", priority="low"),
            ]
        )

        lines = store.visible_lines(max_visible_items=6)

        self.assertTrue(lines)
        self.assertIn("completed=2", lines[0])
        self.assertTrue(any("✔ ~~task-1~~" in line for line in lines))
        self.assertTrue(any("✔ ~~task-2~~" in line for line in lines))

    def test_completed_items_are_moved_below_open_items(self) -> None:
        store = TodoStateStore()
        store.update(
            [
                TodoItemState(content="done", status="completed", priority="medium"),
                TodoItemState(content="doing", status="in_progress", priority="high"),
                TodoItemState(content="todo", status="pending", priority="low"),
            ]
        )

        lines = store.visible_lines(max_visible_items=6)
        open_line_indices = [idx for idx, line in enumerate(lines) if line.startswith("  ◉") or line.startswith("  ○")]
        done_line_indices = [idx for idx, line in enumerate(lines) if line.startswith("  ✔")]

        self.assertTrue(open_line_indices)
        self.assertTrue(done_line_indices)
        self.assertLess(max(open_line_indices), min(done_line_indices))
        self.assertTrue(any(line.startswith("  ─") for line in lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
