import tempfile
import unittest
import json
from pathlib import Path

from bu_agent_sdk.system_tools.tools import Bash, Edit, Glob, Grep, Read, Write
from bu_agent_sdk.tools.system_context import bind_system_tool_context


class TestSystemTools(unittest.TestCase):
    def test_read_offset_is_zero_based_and_default_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "a.txt"
            p.write_text("l1\nl2\nl3\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out0 = self._run(Read, file_path=str(p))
                self.assertIn("\t" + "l1", out0["content"])
                self.assertEqual(out0["total_lines"], 3)
                self.assertEqual(out0["lines_returned"], 3)

                out1 = self._run(Read, file_path=str(p), offset=1, limit=1)
                self.assertIn("\t" + "l2", out1["content"])
                self.assertNotIn("\t" + "l1", out1["content"])
                self.assertEqual(out1["lines_returned"], 1)

    def test_write_creates_parent_dirs_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "sub" / "b.txt"

            with bind_system_tool_context(project_root=root):
                out = self._run(Write, file_path=str(p), content="hello")
                self.assertEqual(out["bytes_written"], 5)
                self.assertTrue(p.exists())
                self.assertEqual(p.read_text(encoding="utf-8"), "hello")

                self._run(Write, file_path=str(p), content="hi")
                self.assertEqual(p.read_text(encoding="utf-8"), "hi")

    def test_edit_requires_unique_old_string_unless_replace_all(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "c.txt"
            p.write_text("x\nx\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Edit, file_path=str(p), old_string="x", new_string="y")
                self.assertEqual(out["replacements"], 0)
                self.assertTrue(out["message"].startswith("Error:"))
                self.assertEqual(p.read_text(encoding="utf-8"), "x\nx\n")

                out2 = self._run(
                    Edit,
                    file_path=str(p),
                    old_string="x",
                    new_string="y",
                    replace_all=True,
                )
                self.assertEqual(out2["replacements"], 2)
                self.assertEqual(p.read_text(encoding="utf-8"), "y\ny\n")

    def test_glob_returns_paths_relative_to_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "d").mkdir()
            (root / "d" / "f1.txt").write_text("a", encoding="utf-8")
            (root / "d" / "f2.txt").write_text("b", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Glob, pattern="**/*.txt")
                self.assertEqual(out["count"], 2)
                self.assertTrue(all(not Path(m).is_absolute() for m in out["matches"]))

    def test_grep_alias_flags_are_parsed_and_context_attaches_without_n(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "e.txt"
            p.write_text("a\nfoo\nb\n", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(
                    Grep,
                    **{
                        "pattern": "foo",
                        "path": str(root),
                        "output_mode": "content",
                        "-A": 1,
                        "-n": False,
                    },
                )
                self.assertEqual(out["total_matches"], 1)
                m = out["matches"][0]
                self.assertIsNone(m["line_number"])
                self.assertEqual(m["after_context"], ["b"])

    def test_bash_runs_in_project_root_and_returns_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x.txt").write_text("ok", encoding="utf-8")

            with bind_system_tool_context(project_root=root):
                out = self._run(Bash, command="cat x.txt")
                self.assertEqual(out["exitCode"], 0)
                self.assertIn("ok", out["output"])

    @staticmethod
    def _run(tool_obj, /, **kwargs):
        # Tool.execute 是 async，这里用 event loop 运行最小封装
        import asyncio

        raw = asyncio.run(tool_obj.execute(**kwargs))
        return json.loads(raw) if isinstance(raw, str) else raw


if __name__ == "__main__":
    unittest.main()
