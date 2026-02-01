import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bu_agent_sdk.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from bu_agent_sdk.system_tools.tools import (
    Bash,
    Edit,
    Glob,
    Grep,
    LS,
    MultiEdit,
    Read,
    TodoWrite,
    WebFetch,
    Write,
)
from bu_agent_sdk.tools.system_context import bind_system_tool_context
from bu_agent_sdk.tokens import TokenCost


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

    def test_multiedit_creates_new_file_with_first_empty_old_string(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = root / "new.txt"

            with bind_system_tool_context(project_root=root):
                out = self._run(
                    MultiEdit,
                    file_path=str(p),
                    edits=[
                        {"old_string": "", "new_string": "hello\n", "replace_all": False},
                        {"old_string": "hello", "new_string": "hi", "replace_all": False},
                    ],
                )
                self.assertEqual(out["replacements"], 1)
                self.assertTrue(p.exists())
                self.assertEqual(p.read_text(encoding="utf-8"), "hi\n")

    def test_ls_returns_entries_and_supports_ignore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "d").mkdir()

            with bind_system_tool_context(project_root=root):
                out = self._run(LS, path=str(root), ignore=["b.txt"])
                names = [e["name"] for e in out["entries"]]
                self.assertIn("a.txt", names)
                self.assertIn("d", names)
                self.assertNotIn("b.txt", names)

    def test_todowrite_persists_to_session_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            session_root = root / "sessions" / "s1"

            with bind_system_tool_context(project_root=root, session_id="s1", session_root=session_root):
                out = self._run(
                    TodoWrite,
                    todos=[
                        {"content": "t1", "status": "pending", "id": "1"},
                        {"content": "t2", "status": "completed", "id": "2"},
                    ],
                )
                self.assertIn("todos", out)
                self.assertTrue((session_root / "todos.json").exists())
                data = json.loads((session_root / "todos.json").read_text(encoding="utf-8"))
                self.assertEqual(len(data["todos"]), 2)

    def test_webfetch_uses_low_llm_and_records_usage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class DummyLLM:
                model = "dummy-low"

                @property
                def provider(self) -> str:
                    return "dummy"

                @property
                def name(self) -> str:
                    return self.model

                async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
                    return ChatInvokeCompletion(
                        content="answer",
                        usage=ChatInvokeUsage(
                            prompt_tokens=10,
                            prompt_cached_tokens=None,
                            prompt_cache_creation_tokens=None,
                            prompt_image_tokens=None,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )

            token_cost = TokenCost(include_cost=False)
            llm_levels = {"LOW": DummyLLM()}

            def fake_get(url, timeout, impersonate):
                return SimpleNamespace(status_code=200, url=url, text="<h1>Hello</h1>")

            with bind_system_tool_context(
                project_root=root,
                session_id="s1",
                session_root=root / "sessions" / "s1",
                token_cost=token_cost,
                llm_levels=llm_levels,
            ):
                with mock.patch.dict("sys.modules", {"curl_cffi": SimpleNamespace(requests=SimpleNamespace(get=fake_get))}):
                    with mock.patch.dict("sys.modules", {"markdownify": SimpleNamespace(markdownify=lambda html: "# Hello")}):
                        out = self._run_raw(WebFetch, url="https://example.com", prompt="Summarize")
                        self.assertEqual(out, "answer")
                        self.assertEqual(len(token_cost.usage_history), 1)
                        self.assertEqual(token_cost.usage_history[0].model, "dummy-low")
                        self.assertEqual(token_cost.usage_history[0].level, "LOW")
                        self.assertEqual(token_cost.usage_history[0].source, "webfetch")

    def test_webfetch_non_200_does_not_call_llm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            class ExplodingLLM:
                model = "dummy-low"

                @property
                def provider(self) -> str:
                    return "dummy"

                @property
                def name(self) -> str:
                    return self.model

                async def ainvoke(self, messages, tools=None, tool_choice=None, **kwargs):
                    raise AssertionError("should not be called")

            token_cost = TokenCost(include_cost=False)
            llm_levels = {"LOW": ExplodingLLM()}

            def fake_get(url, timeout, impersonate):
                return SimpleNamespace(status_code=404, url=url, text="not found")

            with bind_system_tool_context(
                project_root=root,
                session_id="s1",
                session_root=root / "sessions" / "s1",
                token_cost=token_cost,
                llm_levels=llm_levels,
            ):
                with mock.patch.dict("sys.modules", {"curl_cffi": SimpleNamespace(requests=SimpleNamespace(get=fake_get))}):
                    out = self._run_raw(WebFetch, url="https://example.com", prompt="Summarize")
                    self.assertTrue(out.startswith("Error: HTTP 404"))
                    self.assertEqual(len(token_cost.usage_history), 0)

    @staticmethod
    def _run(tool_obj, /, **kwargs):
        # Tool.execute 是 async，这里用 event loop 运行最小封装
        import asyncio

        raw = asyncio.run(tool_obj.execute(**kwargs))
        if isinstance(raw, str) and raw.strip().startswith(("{", "[")):
            return json.loads(raw)
        return raw

    @staticmethod
    def _run_raw(tool_obj, /, **kwargs):
        import asyncio

        raw = asyncio.run(tool_obj.execute(**kwargs))
        return raw


if __name__ == "__main__":
    unittest.main()
