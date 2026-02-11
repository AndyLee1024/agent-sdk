import sys
import unittest
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from terminal_agent.todo_view import extract_todos


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
